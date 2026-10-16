"""Three matched experimental branches; no changes to the formal suite."""
from pathlib import Path
from datetime import datetime
import contextlib, hashlib, json, os, sys, traceback

PROJECT = Path(r"E:\wheel_leg_rl-main\wheelleg_dreamwaq")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "scripts"))
from run_training_suite import (
    _suite_source_fingerprint, _assert_suite_source_fingerprint,
    _stream_process, _monitor_report, _run_checked, _stable_hash, _write_json)
from run_dreamwaq_training_suite import _training_fingerprint
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import load_dreamwaq_checkpoint_artifact
from wheelleg_dreamwaq.schemas.randomization import canonical_tensor_sha256
from wheelleg_dreamwaq.training.weights_initialization import require_initialization_compatibility
from wheelleg_dreamwaq.deployment.dreamwaq_inference import DreamWaQInferenceActorV1
from export_dreamwaq_actor import _build_policy
import torch

SOURCE = PROJECT / "logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-11_04-20-34_rtx5070_fudan-v1_dreamwaq-v1-suite-20261011-024927-run03-seed41882943/model_999.pt"
SOURCE_SHA = "3D621280CF7F11088C972B0DD84CF2E698C29A4C8CDA8BC370BB80A2669DF885"
ARMS = [("baseline", "stop_reverse_v1"), ("budget12", "stop_reverse_reversal12_v1"),
        ("budget8", "stop_reverse_reversal8_v1")]
RESET_CACHE = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/isaac_evaluation/evaluation-reset-cache.pt"
MJPY = PROJECT / "sim2sim/mujoco/.venv/Scripts/python.exe"
DIAGNOSTIC = OUT / "evaluate_mujoco_trial.py"
ARCH = PROJECT.parent / "docs/2026-10-03-wheelleg-dreamwaq-architecture.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protected():
    baseline = read(OUT / "baseline.json")
    changed = [p for p, h in baseline["protected"].items() if sha(PROJECT.parent / p) != h]
    changed += [p for p, h in baseline["preexisting_edit_hashes"].items() if sha(PROJECT.parent / p) != h]
    assert not changed, changed


def save(record):
    _write_json(OUT / "experiment-manifest.json", record)


def freeze(record):
    _assert_suite_source_fingerprint(record["source_fingerprint_hash"])
    assert sha(ARCH) == record["architecture_sha256"]
    assert sha(SOURCE) == SOURCE_SHA
    assert sha(DIAGNOSTIC) == record["diagnostic_sha256"]
    protected()


def main():
    assert sha(SOURCE) == SOURCE_SHA
    source_payload, source_manifest, source_metadata = load_dreamwaq_checkpoint_artifact(SOURCE)
    assert source_metadata["completed_iterations"] == 1000
    assert RESET_CACHE.is_file()
    source_fingerprint, source_hash = _suite_source_fingerprint()
    _write_json(OUT / "suite-source-fingerprint.json", source_fingerprint)
    record = {
        "schema_version": "LegReversalTrialV1", "status": "ready",
        "source_checkpoint": str(SOURCE), "source_checkpoint_sha256": SOURCE_SHA,
        "source_model_tensor_sha256": canonical_tensor_sha256(source_payload["model_state_dict"]),
        "source_fingerprint_hash": source_hash, "architecture_sha256": sha(ARCH),
        "diagnostic_sha256": sha(DIAGNOSTIC), "seed": 41882943,
        "iterations_per_arm": 200, "weights_only": True,
        "monitor_interval_seconds": 1800, "pid": os.getpid(), "arms": [],
        "acceptance": {"unit_passed": 201, "real_isaac_integration_passed": 2},
    }
    freeze(record)
    save(record)
    if "--validate-only" in sys.argv:
        print("READY: source/physics/protected files and three-arm configuration verified", flush=True)
        return
    record["status"] = "training"
    record["started_at"] = datetime.now().astimezone().isoformat()
    save(record)
    cache_hashes = []
    for index, (name, profile) in enumerate(ARMS, 1):
        freeze(record)
        arm = {"name": name, "profile": profile, "index": index, "status": "training"}
        record["arms"].append(arm)
        command = [sys.executable, "scripts/train_dreamwaq.py", "--profile", "rtx5070",
                   "--task-profile", profile, "--seed", "41882943", "--max-iterations", "200",
                   "--initialize-from", str(SOURCE), "--run-name", f"leg-reversal-trial-{name}-200", "--headless"]
        arm["training_command"] = command
        save(record)
        print(f"START training {name}: 200 iterations", flush=True)
        with (OUT / f"{name}-orchestrator.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            code, run_dir = _stream_process(command, OUT / f"{name}-training.log",
                monitor_dir=OUT / "monitor" / name, monitor_interval_s=1800)
        assert code == 0 and run_dir is not None, (name, code, run_dir)
        freeze(record)
        summary = read(run_dir / "training_summary.json")
        manifest = read(run_dir / "run_manifest.json")
        require_initialization_compatibility(
            source_manifest["base_task_contract"], manifest["base_task_contract"],
            source_manifest["dreamwaq_algorithm_contract"], manifest["dreamwaq_algorithm_contract"],
            source_manifest["dreamwaq_export_contract"], manifest["dreamwaq_export_contract"])
        assert summary["starting_completed_iterations"] == 0 and summary["iterations_executed"] == 200
        assert summary["completed_iterations"] == 200 and summary["resumed_from"] is None
        assert summary["initial_model_tensor_sha256"] == record["source_model_tensor_sha256"]
        assert manifest["seed"] == 41882943 and manifest["num_envs"] == 256 and manifest["num_mini_batches"] == 8
        assert manifest["task_profile"] == profile
        assert manifest["randomization_audit"]["realized_plan_hash"] == source_manifest["randomization_audit"]["realized_plan_hash"]
        cache_hashes.append(manifest["closed_chain_reset_cache"]["file_sha256"])
        assert len(set(cache_hashes)) == 1
        monitoring = _monitor_report(OUT / "monitor" / name)
        assert not monitoring["hard_anomalies"], monitoring
        fingerprint, fingerprint_hash = _training_fingerprint(manifest)
        _write_json(OUT / f"{name}-training-fingerprint.json", fingerprint)
        arm.update(status="trained", run_dir=str(run_dir), checkpoint=summary["final_checkpoint"],
                   checkpoint_sha256=sha(summary["final_checkpoint"]), duration_seconds=summary["duration_seconds"],
                   context_acceptance=summary["context_acceptance"], monitor=monitoring,
                   gain_state=summary.get("leg_reversal_gain_state"), training_fingerprint_hash=fingerprint_hash)
        save(record)
        print(f"DONE training {name}: {summary['duration_seconds']/60:.2f} minutes", flush=True)

    record["status"] = "evaluating"
    save(record)
    reference_interface = None
    for arm in record["arms"]:
        name = arm["name"]
        arm_dir = OUT / name
        arm_dir.mkdir(exist_ok=True)
        export = arm_dir / "export"
        print(f"START export/evaluations {name}", flush=True)
        _run_checked([sys.executable, "scripts/export_dreamwaq_actor.py", "--checkpoint",
                      arm["checkpoint"], "--output", str(export)], arm_dir / "export.log")
        pm = read(export / "policy_manifest.json")
        interface_fields = ["network", "observation", "normalization", "q_nominal", "action", "control",
                            "frames", "timing", "initial_state", "history_adapter_version", "asset_bundle_hash",
                            "actuators", "mujoco_model"]
        interface = {key: pm[key] for key in interface_fields}
        if reference_interface is None:
            reference_interface = interface
        assert interface == reference_interface
        payload, _, _ = load_dreamwaq_checkpoint_artifact(Path(arm["checkpoint"]))
        python_actor = DreamWaQInferenceActorV1.from_policy(_build_policy(payload["model_state_dict"])).eval()
        actor = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
        golden = torch.load(export / "golden_vectors.pt", map_location="cpu", weights_only=False)
        with torch.inference_mode():
            golden_errors = {str(batch): float((actor(golden["history"][:batch]) -
                python_actor(golden["history"][:batch])).abs().max()) for batch in (1, 3, 32)}
            full_error = float((actor(golden["history"]) - golden["expected_action_mean"]).abs().max())
        assert max(golden_errors.values()) <= 1e-7 and full_error <= 1e-7
        _write_json(arm_dir / "golden-verification.json", {
            "same_batch_python_vs_torchscript": golden_errors, "standard_golden_max_error": full_error})
        context = {"schema_version": "LegReversalTrialEvaluationContextV1",
                   "suite_mode": "single_seed_leg_reversal_trial", "suite_id": OUT.name,
                   "run_count": 3, "iterations_per_run": 200,
                   "training_fingerprint_hash": arm["training_fingerprint_hash"],
                   "suite_source_fingerprint_hash": record["source_fingerprint_hash"]}
        context["context_hash"] = _stable_hash(context)
        context_path = arm_dir / "evaluation-context.json"
        _write_json(context_path, context)
        stages = [
            ("isaac", [sys.executable, "scripts/evaluate_isaac.py", "--checkpoint", arm["checkpoint"],
                       "--candidate-only", "--reset-cache", str(RESET_CACHE), "--output",
                       str(arm_dir / "isaac"), "--headless"]),
            ("mujoco", [str(MJPY), str(DIAGNOSTIC), "--policy", str(export / "actor.ts"),
                        "--manifest", str(export / "policy_manifest.json"), "--output", str(arm_dir / "mujoco"),
                        "--suite-context", str(context_path), "--run-index", str(arm["index"]),
                        "--completed-iterations", "200"]),
            ("command_practice", [str(MJPY), "scripts/evaluate_command_practice_mujoco.py",
                                  "--policy", str(export / "actor.ts"), "--manifest", str(export / "policy_manifest.json"),
                                  "--output", str(arm_dir / "command_practice")]),
        ]
        for stage, command in stages:
            freeze(record)
            arm["status"] = f"evaluating_{stage}"
            save(record)
            _run_checked(command, arm_dir / f"{stage}.log")
            summary_path = arm_dir / stage / "summary.json"
            data = read(summary_path)
            if "report_hash" in data:
                assert data["report_hash"] == _stable_hash({k: v for k, v in data.items() if k != "report_hash"})
            arm[stage] = {"summary": str(summary_path), "summary_sha256": sha(summary_path)}
            save(record)
        arm["status"] = "evaluated"
        arm["export"] = str(export)
        arm["actor_sha256"] = sha(export / "actor.ts")
        save(record)
        print(f"DONE evaluations {name}", flush=True)
    freeze(record)
    record.update(status="evaluated", finished_at=datetime.now().astimezone().isoformat(),
                  all_protected_hashes_unchanged=True, matched_network_start=True, matched_reset_cache=True)
    save(record)
    print("All three arms trained and evaluated; analysis pending.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        report = read(OUT / "experiment-manifest.json") if (OUT / "experiment-manifest.json").exists() else {}
        report.update(status="failed", error=traceback.format_exc())
        save(report)
        (OUT / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
