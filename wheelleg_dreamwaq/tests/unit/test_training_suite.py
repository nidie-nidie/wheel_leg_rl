from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_training_suite",
    PROJECT_ROOT / "scripts" / "run_training_suite.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
sys.modules["run_training_suite"] = MODULE

DREAMWAQ_SPEC = importlib.util.spec_from_file_location(
    "run_dreamwaq_training_suite",
    PROJECT_ROOT / "scripts" / "run_dreamwaq_training_suite.py",
)
assert DREAMWAQ_SPEC is not None and DREAMWAQ_SPEC.loader is not None
DREAMWAQ_MODULE = importlib.util.module_from_spec(DREAMWAQ_SPEC)
DREAMWAQ_SPEC.loader.exec_module(DREAMWAQ_MODULE)


def _manifest(seed: int) -> dict:
    return {
        "created_at": "now",
        "command": ["train", "--seed", str(seed)],
        "seed": seed,
        "resume": None,
        "contract": {"hash": "same"},
        "randomization_audit": {
            "realized_plan_hash": f"plan-{seed}",
            "actuator_plan_hash": f"actuator-{seed}",
        },
        "closed_chain_reset_cache": {
            "file_sha256": f"file-{seed}",
            "tensor_sha256": f"tensor-{seed}",
        },
        "project_source_sha256": "source",
        "profile_name": "rtx5070",
        "profile": {"num_envs": 256, "num_mini_batches": 8},
        "resolved_config_sha256": {
            "environment": f"raw-env-with-seed-and-log-dir-{seed}",
            "agent": f"raw-agent-with-seed-and-run-name-{seed}",
            "hardware": "same-hardware",
        },
        "randomness": {
            "master_seed": seed,
            "effective_process_seed": seed,
            "environment_streams": {"command_rng": {"seed": seed}},
            "cuda_matmul_allow_tf32": True,
        },
    }


def test_training_fingerprint_ignores_only_per_run_fields() -> None:
    first_payload, first_hash = MODULE._training_fingerprint(_manifest(1))
    second_payload, second_hash = MODULE._training_fingerprint(_manifest(2))
    assert first_payload == second_payload
    assert first_hash == second_hash

    changed = _manifest(2)
    changed["contract"]["hash"] = "different"
    _, changed_hash = MODULE._training_fingerprint(changed)
    assert changed_hash != first_hash

    changed_hardware = _manifest(2)
    changed_hardware["resolved_config_sha256"]["hardware"] = "different-hardware"
    _, changed_hardware_hash = MODULE._training_fingerprint(changed_hardware)
    assert changed_hardware_hash != first_hash


def test_monitor_classifier_separates_warnings_from_hard_anomalies() -> None:
    values = {
        "Loss/value_function": [1.0, 2.0],
        "Metric/loop_closure_error_max_m": [1.0e-4, 6.0e-3],
        "Termination/invalid": [0.0, 0.0],
        "Action/saturation_fraction": [0.99] * 20,
    }
    warnings, hard = MODULE._classify_monitor_values(
        values,
        missing_tags=[],
        event_file_missing=False,
        reason="post_run",
    )
    assert "action_saturation_sustained_above_95_percent" in warnings
    assert "loop_closure_above_5mm" in hard


def test_post_run_missing_tag_is_hard_but_initial_missing_tag_is_warning() -> None:
    initial_warning, initial_hard = MODULE._classify_monitor_values(
        {},
        missing_tags=["Train/mean_reward"],
        event_file_missing=False,
        reason="initial",
    )
    final_warning, final_hard = MODULE._classify_monitor_values(
        {},
        missing_tags=["Train/mean_reward"],
        event_file_missing=False,
        reason="post_run",
    )
    assert initial_warning and not initial_hard
    assert not final_warning and final_hard


def test_monitor_report_ignores_recovered_startup_availability(tmp_path: Path) -> None:
    (tmp_path / "initial.json").write_text(
        json.dumps(
            {
                "reason": "30_minute_interval",
                "warnings": ["event_file_missing", "missing_tag:Train/mean_reward"],
                "hard_anomalies": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "final.json").write_text(
        json.dumps({"reason": "post_run", "warnings": [], "hard_anomalies": []}),
        encoding="utf-8",
    )

    report = MODULE._monitor_report(tmp_path)

    assert report["status"] == "ok"
    assert report["warnings"] == []
    assert report["hard_anomalies"] == []


def test_formal_suite_requires_exactly_four_runs_of_1000_iterations() -> None:
    MODULE._validate_suite_mode("formal", runs=4, iterations=1000)
    MODULE._validate_suite_mode("smoke", runs=2, iterations=1)

    with pytest.raises(ValueError, match="4 runs x 1000 iterations"):
        MODULE._validate_suite_mode("formal", runs=2, iterations=1000)
    with pytest.raises(ValueError, match="4 runs x 1000 iterations"):
        MODULE._validate_suite_mode("formal", runs=4, iterations=1)


def test_suite_source_fingerprint_covers_mujoco_runtime_and_lock() -> None:
    payload, fingerprint = MODULE._suite_source_fingerprint()

    assert len(fingerprint) == 64
    assert "sim2sim/mujoco/uv.lock" in payload["files"]
    assert "sim2sim/mujoco/pyproject.toml" in payload["files"]
    assert "sim2sim/mujoco/wheelleg_mujoco/evaluation.py" in payload["files"]
    assert "scripts/evaluate_mujoco.py" in payload["files"]
    assert "scripts/rank_mujoco_runs.py" in payload["files"]


def test_dreamwaq_training_gate_rejects_hard_invalid_or_missing_runs() -> None:
    valid = [
        {"index": index, "status": "trained", "checkpoint": f"model-{index}.pt"}
        for index in range(1, 5)
    ]
    assert DREAMWAQ_MODULE._training_gate_failures(valid, 4) == []

    hard_invalid = [dict(record) for record in valid]
    hard_invalid[2]["status"] = "training_hard_invalid"
    assert DREAMWAQ_MODULE._training_gate_failures(hard_invalid, 4) == [
        "run-3:training_hard_invalid"
    ]

    missing = valid[:-1]
    assert DREAMWAQ_MODULE._training_gate_failures(missing, 4) == ["run_count:3!=4"]


def test_dreamwaq_suite_completion_allows_acceptance_failure_but_not_missing_evidence() -> None:
    complete = [
        {"index": 1, "status": "completed"},
        {"index": 2, "status": "completed_acceptance_failed"},
    ]
    assert DREAMWAQ_MODULE._evaluation_suite_complete(complete, 2)

    incomplete = [dict(record) for record in complete]
    incomplete[1]["status"] = "mujoco_failed"
    assert not DREAMWAQ_MODULE._evaluation_suite_complete(incomplete, 2)
    assert not DREAMWAQ_MODULE._evaluation_suite_complete(complete[:1], 2)

def test_new_profile_rejects_historical_comparison_before_training():
    DREAMWAQ_MODULE._validate_task_mode("stop_reverse_v1", True)
    DREAMWAQ_MODULE._validate_task_mode("legacy_v1", False)
    with pytest.raises(ValueError, match="candidate-only"):
        DREAMWAQ_MODULE._validate_task_mode("stop_reverse_v1", False)


def test_failed_isaac_artifact_still_runs_both_mujoco_evaluations(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    suite = root / "suite"
    run = root / "run"
    run.mkdir()
    (run / "training_summary.json").write_text(json.dumps({
        "completed_iterations": 1, "resumed_from": None, "final_checkpoint": str(run / "model_0.pt"),
        "context_acceptance": False,
    }), encoding="utf-8")
    monkeypatch.setattr(DREAMWAQ_MODULE, "PROJECT_ROOT", root)
    monkeypatch.setattr(DREAMWAQ_MODULE, "_suite_source_fingerprint", lambda: ({"files": {}}, "A" * 64))
    monkeypatch.setattr(DREAMWAQ_MODULE, "_assert_suite_source_fingerprint", lambda _: None)
    monkeypatch.setattr(DREAMWAQ_MODULE, "_unique_seeds", lambda _: [123])
    monkeypatch.setattr(DREAMWAQ_MODULE, "_training_fingerprint", lambda _: ({}, "B" * 64))
    monkeypatch.setattr(DREAMWAQ_MODULE, "_monitor_report",
                        lambda _: {"warnings": [], "hard_anomalies": [], "status": "ok"})
    calls = []
    def train(command, *args, **kwargs):
        assert command[command.index("--task-profile") + 1] == "stop_reverse_v1"
        (run / "run_manifest.json").write_text(json.dumps({
            "seed": 123, "resume_provenance": {"schema_version": "ResumeMetadataV2", "mode": "fresh"}}))
        return 0, run
    def evaluate(command, log):
        script = next(item for item in command if item.startswith("scripts/"))
        calls.append(script)
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if script == "scripts/evaluate_isaac.py":
            assert "--candidate-only" in command and "--baseline-report" not in command
            raise RuntimeError("missing Isaac artifact")
        if script == "scripts/export_dreamwaq_actor.py":
            for name in ("actor.ts", "policy_manifest.json", "golden_vectors.pt"):
                (output / name).write_text("test fixture")
        else:
            (output / "summary.json").write_text(json.dumps({
                "aggregate": {"scenarios": []}, "performance_accepted": False}))
    monkeypatch.setattr(DREAMWAQ_MODULE, "_stream_process", train)
    monkeypatch.setattr(DREAMWAQ_MODULE, "_run_checked", evaluate)
    monkeypatch.setattr(sys, "argv", ["suite", "--mode", "smoke", "--runs", "1", "--iterations", "1",
                                    "--task-profile", "stop_reverse_v1", "--candidate-only", "--output", str(suite)])
    with pytest.raises(RuntimeError, match="complete export"):
        DREAMWAQ_MODULE.main()
    assert calls == ["scripts/export_dreamwaq_actor.py", "scripts/evaluate_isaac.py",
                     "scripts/evaluate_mujoco.py", "scripts/evaluate_command_practice_mujoco.py"]
    report = json.loads((suite / "training-suite-manifest.json").read_text())
    assert report["runs"][0]["status"] == "evaluation_incomplete"
    assert "isaac" in report["runs"][0]["evaluation_errors"]
    assert report["baseline_isaac_report"] is None and report["baseline_status"] == "not_comparable"
