from __future__ import annotations

import argparse
import json
import secrets
import sys
from datetime import datetime
from pathlib import Path

from run_training_suite import (
    _assert_suite_source_fingerprint,
    _monitor_report,
    _run_checked,
    _stable_hash,
    _stream_process,
    _suite_source_fingerprint,
    _write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM2SIM_PROJECT = PROJECT_ROOT / "sim2sim" / "mujoco"
FORMAL_RUN_COUNT = 4
FORMAL_ITERATIONS = 1000
TRAINING_READY_STATUS = "trained"
EVALUATION_COMPLETE_STATUSES = {"completed", "completed_acceptance_failed"}
BASELINE_CHECKPOINT = Path(
    "E:\\wheel_leg_rl-main\\wheelleg_dreamwaq\\logs\\rsl_rl\\wheelleg_flat_ppo\\"
    "2026-10-07_07-32-32_rtx5070_fudan-v1_phase1r-v3-suite-20261007-062147-run03-seed1884612625\\"
    "model_999.pt"
)


def _training_fingerprint(manifest: dict) -> tuple[dict, str]:
    payload = {
        "schema_version": "DreamWaQTrainingFingerprintV1",
        "runner_class": manifest["runner_class"],
        "profile_name": manifest["profile_name"],
        "profile": manifest["profile"],
        "num_envs": manifest["num_envs"],
        "num_steps_per_env": manifest["num_steps_per_env"],
        "num_mini_batches": manifest["num_mini_batches"],
        "max_iterations": manifest["max_iterations"],
        "device": manifest["device"],
        "torch": manifest["torch"],
        "torch_cuda_runtime": manifest["torch_cuda_runtime"],
        "tensordict": manifest["tensordict"],
        "isaac_sim": manifest["isaac_sim"],
        "rsl_rl": manifest["rsl_rl"],
        "isaac_lab_commit": manifest["isaac_lab_commit"],
        "asset_bundle_version": manifest["asset_bundle_version"],
        "asset_bundle_hash": manifest["asset_bundle_hash"],
        "architecture_sha256": manifest["architecture_sha256"],
        "dependency_manifest_sha256": manifest["dependency_manifest_sha256"],
        "pyproject_sha256": manifest["pyproject_sha256"],
        "uv_lock_sha256": manifest["uv_lock_sha256"],
        "project_source_sha256": manifest["project_source_sha256"],
        "base_task_contract": manifest["base_task_contract"],
        "dreamwaq_algorithm_contract": manifest["dreamwaq_algorithm_contract"],
        "dreamwaq_export_contract": manifest["dreamwaq_export_contract"],
        "randomization_streams": manifest["base_task_contract"]["randomization"]["profile_contract"]["streams"],
        "closed_chain_reset_cache_contract": manifest["base_task_contract"]["randomization"]["closed_chain_reset_cache"],
    }
    return payload, _stable_hash(payload)


def _unique_seeds(count: int) -> list[int]:
    source = secrets.SystemRandom()
    seeds: list[int] = []
    while len(seeds) < count:
        seed = source.randrange(1, 2**31)
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def _training_gate_failures(records: list[dict], expected_count: int) -> list[str]:
    failures = []
    if len(records) != expected_count:
        failures.append(f"run_count:{len(records)}!={expected_count}")
    for record in records:
        index = record.get("index", "unknown")
        status = record.get("status")
        if status != TRAINING_READY_STATUS:
            failures.append(f"run-{index}:{status}")
        elif not record.get("checkpoint"):
            failures.append(f"run-{index}:missing_checkpoint")
    return failures


def _evaluation_suite_complete(records: list[dict], expected_count: int) -> bool:
    return len(records) == expected_count and all(
        record.get("status") in EVALUATION_COMPLETE_STATUSES for record in records
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the WheelLeg DreamWaQ training/evaluation suite.")
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--profile", choices=("portable", "rtx4060", "rtx5070"), default="rtx5070")
    parser.add_argument("--iterations", type=int, default=FORMAL_ITERATIONS)
    parser.add_argument("--runs", type=int, default=FORMAL_RUN_COUNT)
    parser.add_argument("--monitor-interval-seconds", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.mode == "formal" and (
        args.profile != "rtx5070" or args.runs != FORMAL_RUN_COUNT or args.iterations != FORMAL_ITERATIONS
    ):
        raise ValueError("Formal DreamWaQ suite requires rtx5070, four runs, and 1000 iterations")
    if args.runs <= 0 or args.iterations <= 0 or args.monitor_interval_seconds <= 0.0:
        raise ValueError("Suite runs, iterations, and monitor interval must be positive")
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(BASELINE_CHECKPOINT)

    suite_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_root = PROJECT_ROOT / "artifacts" / "phase2_dreamwaq"
    suite_dir = (args.output or suite_root / f"training-suite-{suite_id}").resolve()
    suite_dir.mkdir(parents=True, exist_ok=False)
    source_payload, source_hash = _suite_source_fingerprint()
    _write_json(suite_dir / "suite-source-fingerprint.json", source_payload)
    seeds = _unique_seeds(args.runs)
    suite_manifest_path = suite_dir / "training-suite-manifest.json"
    suite_manifest = {
        "schema_version": "DreamWaQTrainingSuiteV1",
        "created_at": datetime.now().astimezone().isoformat(),
        "suite_id": suite_id,
        "suite_mode": args.mode,
        "profile": args.profile,
        "iterations_per_run": args.iterations,
        "run_count": args.runs,
        "fresh_start": True,
        "seeds": seeds,
        "suite_source_fingerprint_hash": source_hash,
        "runs": [],
    }
    _write_json(suite_manifest_path, suite_manifest)

    reference_fingerprint = None
    suite_warnings: set[str] = set()
    suite_hard_anomalies: set[str] = set()
    for index, seed in enumerate(seeds, start=1):
        _assert_suite_source_fingerprint(source_hash)
        run_name = f"dreamwaq-v1-suite-{suite_id}-run{index:02d}-seed{seed}"
        record = {"index": index, "seed": seed, "run_name": run_name, "status": "training"}
        suite_manifest["runs"].append(record)
        _write_json(suite_manifest_path, suite_manifest)
        command = [
            sys.executable,
            "scripts/train_dreamwaq.py",
            "--profile",
            args.profile,
            "--max-iterations",
            str(args.iterations),
            "--seed",
            str(seed),
            "--run-name",
            run_name,
            "--headless",
        ]
        return_code, run_dir = _stream_process(
            command,
            suite_dir / f"run-{index:02d}-training.log",
            monitor_dir=suite_dir / "monitor" / f"run-{index:02d}",
            monitor_interval_s=args.monitor_interval_seconds,
        )
        if return_code != 0 or run_dir is None:
            record.update({"status": "training_failed", "return_code": return_code})
            _write_json(suite_manifest_path, suite_manifest)
            continue
        _assert_suite_source_fingerprint(source_hash)
        summary_path = run_dir / "training_summary.json"
        manifest_path = run_dir / "run_manifest.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            summary["completed_iterations"] != args.iterations
            or summary["resumed_from"] is not None
            or manifest["resume_provenance"] != {"schema_version": "ResumeMetadataV2", "mode": "fresh"}
            or manifest["seed"] != seed
        ):
            record.update({"status": "training_summary_rejected", "run_dir": str(run_dir)})
            _write_json(suite_manifest_path, suite_manifest)
            continue
        fingerprint_payload, fingerprint_hash = _training_fingerprint(manifest)
        if reference_fingerprint is None:
            reference_fingerprint = (fingerprint_payload, fingerprint_hash)
            suite_manifest["training_fingerprint_hash"] = fingerprint_hash
            _write_json(suite_dir / "training-fingerprint.json", fingerprint_payload)
        elif reference_fingerprint != (fingerprint_payload, fingerprint_hash):
            record.update({"status": "training_fingerprint_mismatch", "run_dir": str(run_dir)})
            _write_json(suite_manifest_path, suite_manifest)
            continue
        monitor = _monitor_report(suite_dir / "monitor" / f"run-{index:02d}")
        suite_warnings.update(monitor["warnings"])
        suite_hard_anomalies.update(monitor["hard_anomalies"])
        record.update(
            {
                "status": "training_hard_invalid" if monitor["hard_anomalies"] else TRAINING_READY_STATUS,
                "run_dir": str(run_dir),
                "training_summary": str(summary_path),
                "run_manifest": str(manifest_path),
                "checkpoint": summary["final_checkpoint"],
                "training_fingerprint_hash": fingerprint_hash,
                "monitor": monitor,
                "context_acceptance": bool(summary["context_acceptance"]),
            }
        )
        _write_json(suite_manifest_path, suite_manifest)

    suite_manifest["training_monitor_gate"] = {
        "status": "hard_anomaly" if suite_hard_anomalies else "warning" if suite_warnings else "ok",
        "warnings": sorted(suite_warnings),
        "hard_anomalies": sorted(suite_hard_anomalies),
    }
    training_failures = _training_gate_failures(suite_manifest["runs"], args.runs)
    if training_failures or suite_hard_anomalies:
        suite_manifest["training_gate_failures"] = training_failures
        suite_manifest["status"] = "training_gate_failed"
        _write_json(suite_manifest_path, suite_manifest)
        raise RuntimeError(
            "DreamWaQ training hard gate failed before export/evaluation: "
            + ", ".join(training_failures or sorted(suite_hard_anomalies))
        )
    if reference_fingerprint is None:
        raise RuntimeError("DreamWaQ suite produced no valid training run")
    evaluation_context = {
        "schema_version": "DreamWaQTrainingEvaluationContextV1",
        "suite_mode": args.mode,
        "suite_id": suite_id,
        "run_count": args.runs,
        "iterations_per_run": args.iterations,
        "training_fingerprint_hash": reference_fingerprint[1],
        "suite_source_fingerprint_hash": source_hash,
    }
    evaluation_context["context_hash"] = _stable_hash(evaluation_context)
    evaluation_context_path = suite_dir / "evaluation-suite-context.json"
    _write_json(evaluation_context_path, evaluation_context)

    isaac_root = suite_dir / "isaac_evaluation"
    reset_cache = isaac_root / "evaluation-reset-cache.pt"
    baseline_dir = isaac_root / "phase1r-baseline"
    _run_checked(
        [
            sys.executable,
            "scripts/evaluate_isaac.py",
            "--checkpoint",
            str(BASELINE_CHECKPOINT),
            "--reset-cache",
            str(reset_cache),
            "--output",
            str(baseline_dir),
            "--headless",
        ],
        suite_dir / "isaac-baseline.log",
    )
    baseline_report = baseline_dir / "summary.json"

    mujoco_summaries: list[Path] = []
    for record in suite_manifest["runs"]:
        if "checkpoint" not in record:
            continue
        index = int(record["index"])
        checkpoint = Path(record["checkpoint"])
        export_dir = suite_dir / "exports" / f"run-{index:02d}"
        try:
            _assert_suite_source_fingerprint(source_hash)
            _run_checked(
                [
                    sys.executable,
                    "scripts/export_dreamwaq_actor.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(export_dir),
                ],
                suite_dir / f"run-{index:02d}-export.log",
            )
            isaac_dir = isaac_root / f"run-{index:02d}"
            _run_checked(
                [
                    sys.executable,
                    "scripts/evaluate_isaac.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--reset-cache",
                    str(reset_cache),
                    "--baseline-report",
                    str(baseline_report),
                    "--output",
                    str(isaac_dir),
                    "--headless",
                ],
                suite_dir / f"run-{index:02d}-isaac.log",
            )
            isaac_report = json.loads((isaac_dir / "summary.json").read_text(encoding="utf-8"))
            record["export_dir"] = str(export_dir)
            record["isaac_evaluation"] = str(isaac_dir / "summary.json")
            record["velocity_acceptance"] = bool(isaac_report["estimator"]["velocity_acceptance"])
            record["baseline_acceptance"] = bool(
                isaac_report["baseline_comparison"]["candidate_not_worse"]
            )
        except Exception as error:
            record.update({"status": "artifact_or_isaac_failed", "error": f"{type(error).__name__}: {error}"})
            _write_json(suite_manifest_path, suite_manifest)
            continue

        mujoco_dir = suite_dir / "mujoco_evaluation" / f"run-{index:02d}"
        command = [
            "uv",
            "run",
            "--project",
            str(SIM2SIM_PROJECT),
            "python",
            "scripts/evaluate_mujoco.py",
            "--policy",
            str(export_dir / "actor.ts"),
            "--manifest",
            str(export_dir / "policy_manifest.json"),
            "--output",
            str(mujoco_dir),
        ]
        if args.mode == "formal":
            command.extend(
                (
                    "--suite-context",
                    str(evaluation_context_path),
                    "--run-index",
                    str(index),
                    "--completed-iterations",
                    str(args.iterations),
                )
            )
        else:
            command.append("--smoke")
        try:
            _run_checked(command, suite_dir / f"run-{index:02d}-mujoco.log")
            summary_path = mujoco_dir / "summary.json"
            mujoco_summaries.append(summary_path)
            record["mujoco_evaluation"] = str(summary_path)
            accepted = record["context_acceptance"] and record["velocity_acceptance"] and record["baseline_acceptance"]
            record["phase2_acceptance"] = accepted
            record["status"] = "completed" if accepted else "completed_acceptance_failed"
        except Exception as error:
            record.update({"status": "mujoco_failed", "error": f"{type(error).__name__}: {error}"})
        _write_json(suite_manifest_path, suite_manifest)

    ranking_path = suite_dir / "mujoco-ranking-summary.json"
    if args.mode == "formal" and len(mujoco_summaries) == args.runs:
        ranking_command = [
            "uv",
            "run",
            "--project",
            str(SIM2SIM_PROJECT),
            "python",
            "scripts/rank_mujoco_runs.py",
        ]
        for path in mujoco_summaries:
            ranking_command.extend(("--evaluation", str(path)))
        ranking_command.extend(("--output", str(ranking_path)))
        _run_checked(ranking_command, suite_dir / "mujoco-ranking.log")
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    else:
        ranking = {
            "schema_version": "DreamWaQMujocoEvidenceIncompleteV1",
            "evaluation_summaries": [str(path) for path in mujoco_summaries],
            "expected_count": args.runs,
        }
        _write_json(ranking_path, ranking)

    suite_manifest["baseline_isaac_report"] = str(baseline_report)
    suite_manifest["shared_isaac_reset_cache"] = str(reset_cache)
    suite_manifest["mujoco_ranking_summary"] = str(ranking_path)
    suite_complete = _evaluation_suite_complete(suite_manifest["runs"], args.runs)
    suite_manifest["status"] = "completed" if suite_complete else "incomplete"
    _write_json(suite_manifest_path, suite_manifest)
    if not suite_complete:
        raise RuntimeError("DreamWaQ suite did not produce complete export, Isaac, and MuJoCo evidence for every run")
    pointer = suite_root / ("latest_training_suite.txt" if args.mode == "formal" else "latest_training_smoke.txt")
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(suite_dir), encoding="utf-8")
    print(json.dumps(suite_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
