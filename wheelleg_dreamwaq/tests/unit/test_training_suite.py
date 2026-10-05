from __future__ import annotations

import importlib.util
import json
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


def _manifest(seed: int) -> dict:
    return {
        "created_at": "now",
        "command": ["train", "--seed", str(seed)],
        "seed": seed,
        "resume": None,
        "contract": {"hash": "same"},
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
