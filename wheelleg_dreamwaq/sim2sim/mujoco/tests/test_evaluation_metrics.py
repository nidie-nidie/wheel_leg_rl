from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from wheelleg_mujoco.evaluation import (
    EVALUATION_CONTRACT_VERSION,
    EVALUATION_SCHEMA_VERSION,
    FORMAL_SCENARIOS,
    aggregate_run,
    build_evaluation_source_fingerprint,
    build_ranking_summary,
    rank_runs,
    summarize_scenario,
    unwrap_yaw,
    validate_evaluation_reports_for_ranking,
)
from wheelleg_mujoco.model_semantics import stable_hash
from wheelleg_mujoco.contract import sha256_file


def _rows(count: int, *, vx_error: float, yaw_rate_error: float = 0.0) -> list[dict[str, float]]:
    return [
        {
            "base_height_m": 0.20,
            "roll_rad": 0.01,
            "pitch_rad": -0.02,
            "yaw_rad": 0.1 * index,
            "vx_mps": 1.0 - vx_error,
            "yaw_rate_rad_s": 0.5 - yaw_rate_error,
            "phi0_delta_abs_rad": 0.03,
            "action_saturation_fraction": 0.1,
            "effort_saturation_fraction": 0.2,
            "max_loop_closure_error_m": 0.001,
            "tilt_rad": 0.03,
        }
        for index in range(count)
    ]


def test_yaw_unwrap_removes_pi_boundary_jump() -> None:
    result = unwrap_yaw(np.array((3.0, -3.0, -2.8)))
    assert np.all(np.diff(result) > 0.0)
    assert np.max(np.abs(np.diff(result))) < 0.5


def test_aggregate_weights_scenarios_equally_not_by_tick_count() -> None:
    long = summarize_scenario("long", (1.0, 0.0, 0.20), _rows(100, vx_error=0.3), expected_ticks=100)
    short = summarize_scenario("short", (1.0, 0.0, 0.20), _rows(10, vx_error=0.9), expected_ticks=10)
    aggregate = aggregate_run("run", [long, short])
    assert math.isclose(aggregate["metrics"]["vx_mae"], 0.6)


def test_nonzero_yaw_uses_yaw_rate_and_zero_yaw_uses_drift() -> None:
    zero = summarize_scenario("stand", (1.0, 0.0, 0.20), _rows(5, vx_error=0.0), expected_ticks=5)
    turn = summarize_scenario(
        "turn",
        (1.0, 0.5, 0.20),
        _rows(5, vx_error=0.0, yaw_rate_error=0.2),
        expected_ticks=5,
    )
    assert zero["yaw_drift_rms"] is not None and zero["yaw_rate_mae"] is None
    assert turn["yaw_drift_rms"] is None and math.isclose(turn["yaw_rate_mae"], 0.2)


def test_yaw_drift_uses_reset_yaw_not_first_post_step_sample() -> None:
    rows = _rows(2, vx_error=0.0)
    rows[0]["yaw_rad"] = 0.2
    rows[1]["yaw_rad"] = 0.3
    summary = summarize_scenario(
        "stand",
        (0.0, 0.0, 0.20),
        rows,
        expected_ticks=2,
        initial_yaw_rad=0.1,
    )
    assert math.isclose(summary["yaw_drift_rms"], math.sqrt((0.1**2 + 0.2**2) / 2.0))


def test_empty_prefix_has_infinite_score() -> None:
    empty = summarize_scenario("empty", (0.0, 0.0, 0.20), [], expected_ticks=500)
    aggregate = aggregate_run("failed", [empty])
    assert math.isinf(empty["prefix_score"])
    assert math.isinf(aggregate["score"])
    assert not aggregate["full_survival"]


def test_ranking_is_survival_first_then_uses_tie_breakers() -> None:
    complete_high_score = {
        "run_id": "complete",
        "full_survival": True,
        "completed_scenarios": 8,
        "survival_fraction": 1.0,
        "score": 10.0,
        "metrics": {"effort_saturation_fraction": 0.2, "action_saturation_fraction": 0.1, "max_tilt_rad": 0.2},
    }
    failed_low_score = {
        "run_id": "failed",
        "full_survival": False,
        "completed_scenarios": 7,
        "survival_fraction": 0.99,
        "score": 0.1,
        "metrics": {"effort_saturation_fraction": 0.0, "action_saturation_fraction": 0.0, "max_tilt_rad": 0.0},
    }
    tie_lower_effort = {
        "run_id": "tie",
        "full_survival": True,
        "completed_scenarios": 8,
        "survival_fraction": 1.0,
        "score": 10.0 + 1.0e-7,
        "metrics": {"effort_saturation_fraction": 0.1, "action_saturation_fraction": 0.5, "max_tilt_rad": 0.5},
    }
    ranked = rank_runs([failed_low_score, complete_high_score, tie_lower_effort])
    assert [item["run_id"] for item in ranked] == ["tie", "complete", "failed"]


def test_ranking_accepts_immediate_failure_with_missing_metrics() -> None:
    immediate_failure = {
        "run_id": "immediate",
        "full_survival": False,
        "completed_scenarios": 0,
        "survival_fraction": 0.0,
        "score": math.inf,
        "metrics": {
            "effort_saturation_fraction": None,
            "action_saturation_fraction": None,
            "max_tilt_rad": None,
        },
    }
    partial_failure = {
        "run_id": "partial",
        "full_survival": False,
        "completed_scenarios": 1,
        "survival_fraction": 0.2,
        "score": 20.0,
        "metrics": {
            "effort_saturation_fraction": 0.5,
            "action_saturation_fraction": 0.5,
            "max_tilt_rad": 0.5,
        },
    }
    ranked = rank_runs([immediate_failure, partial_failure])
    assert [item["run_id"] for item in ranked] == ["partial", "immediate"]


def _write_scenario_csv(path: Path, command: tuple[float, float, float]) -> None:
    rows = []
    for tick, metrics in enumerate(_rows(500, vx_error=0.0), start=1):
        row = {
            "tick": tick,
            "target_vx_mps": command[0],
            "target_yaw_rate_rad_s": command[1],
            "target_base_height_m": command[2],
            "survived": 1,
            "failure_reason": "",
        }
        row.update(metrics)
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _formal_report(root: Path, run_index: int) -> dict:
    root.mkdir(parents=True)
    checkpoint_path = root / "model.pt"
    actor_path = root / "actor.ts"
    checkpoint_path.write_bytes(f"checkpoint-{run_index}".encode("ascii"))
    actor_path.write_bytes(f"actor-{run_index}".encode("ascii"))
    checkpoint_hash = sha256_file(checkpoint_path)
    actor_hash = sha256_file(actor_path)

    policy_manifest = {
        "actor_sha256": actor_hash,
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": checkpoint_hash,
        "source_completed_iterations": 1000,
        "phase1_contract_hash": "contract",
    }
    policy_manifest["manifest_hash"] = stable_hash(policy_manifest)
    policy_manifest_path = root / "policy_manifest.json"
    policy_manifest_path.write_text(json.dumps(policy_manifest, sort_keys=True), encoding="utf-8")

    training_suite = {
        "schema_version": "TrainingEvaluationContextV1",
        "suite_mode": "formal",
        "suite_id": "suite",
        "run_count": 4,
        "iterations_per_run": 1000,
        "training_fingerprint_hash": "A" * 64,
        "suite_source_fingerprint_hash": "B" * 64,
    }
    training_suite["context_hash"] = stable_hash(training_suite)
    scenarios = []
    scenario_files = {}
    for name, command in FORMAL_SCENARIOS:
        csv_path = root / f"{name}.csv"
        _write_scenario_csv(csv_path, command)
        scenario_files[name] = {"path": str(csv_path.resolve()), "sha256": sha256_file(csv_path)}
        summary = summarize_scenario(name, command, _rows(500, vx_error=0.0), expected_ticks=500)
        scenarios.append(summary)
    contract = {
        "schema_version": EVALUATION_CONTRACT_VERSION,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "ranking_version": "MujocoRankingV1",
        "smoke": False,
        "training_suite": training_suite,
        "evaluation_implementation": build_evaluation_source_fingerprint(),
        "phase1_contract_hash": "contract",
        "control_dt_s": 0.02,
        "physics_dt_s": 0.001,
        "physics_steps_per_action": 20,
        "expected_ticks_per_scenario": 500,
        "scenario_duration_s": 10.0,
        "scenarios": [{"name": name, "command": list(command)} for name, command in FORMAL_SCENARIOS],
        "model_version": "MujocoModelV1",
        "model_xml_sha256": "xml",
        "model_manifest_sha256": "manifest",
        "dynamics_semantics_hash": "dynamics",
        "reset_initial_yaw_rad": 0.0,
        "adapter_versions": {"observation": "obs", "action": "action"},
        "failure_limits": {},
    }
    report = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_contract": contract,
        "evaluation_contract_hash": stable_hash(contract),
        "smoke": False,
        "policy": str(actor_path.resolve()),
        "policy_sha256": actor_hash,
        "policy_manifest": str(policy_manifest_path.resolve()),
        "policy_manifest_sha256": sha256_file(policy_manifest_path),
        "policy_manifest_hash": policy_manifest["manifest_hash"],
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": checkpoint_hash,
        "scenario_files": scenario_files,
        "training_run": {
            "schema_version": "TrainingEvaluationRunV1",
            "suite_id": "suite",
            "run_count": 4,
            "run_index": run_index,
            "completed_iterations": 1000,
            "training_fingerprint_hash": "A" * 64,
            "suite_source_fingerprint_hash": "B" * 64,
        },
        "aggregate": aggregate_run(checkpoint_hash[:12], scenarios),
    }
    report["report_hash"] = stable_hash(report)
    return report


def test_formal_ranking_rejects_smoke_mismatch_and_duplicate_checkpoints(tmp_path: Path) -> None:
    reports = [_formal_report(tmp_path / f"run-{index}", index) for index in range(1, 5)]
    assert validate_evaluation_reports_for_ranking(reports) == reports[0]["evaluation_contract"]

    smoke = _formal_report(tmp_path / "smoke", 1)
    smoke["smoke"] = True
    smoke["report_hash"] = stable_hash({key: value for key, value in smoke.items() if key != "report_hash"})
    try:
        validate_evaluation_reports_for_ranking([smoke, *reports[1:]])
    except ValueError as error:
        assert "Smoke" in str(error)
    else:
        raise AssertionError("Smoke evaluation entered formal ranking")

    mismatched = _formal_report(tmp_path / "mismatched", 2)
    mismatched["evaluation_contract"]["model_xml_sha256"] = "different"
    mismatched["evaluation_contract_hash"] = stable_hash(mismatched["evaluation_contract"])
    mismatched["report_hash"] = stable_hash(
        {key: value for key, value in mismatched.items() if key != "report_hash"}
    )
    try:
        validate_evaluation_reports_for_ranking([reports[0], mismatched, *reports[2:]])
    except ValueError as error:
        assert "different contracts" in str(error)
    else:
        raise AssertionError("Different evaluation contracts entered one ranking")

    duplicate = dict(reports[0])
    duplicate["training_run"] = dict(duplicate["training_run"], run_index=2)
    duplicate["report_hash"] = stable_hash(
        {key: value for key, value in duplicate.items() if key != "report_hash"}
    )
    try:
        validate_evaluation_reports_for_ranking([reports[0], duplicate, *reports[2:]])
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("Duplicate checkpoint entered formal ranking")


def test_evaluation_source_fingerprint_covers_runtime_scripts_and_lock() -> None:
    fingerprint = build_evaluation_source_fingerprint()
    files = fingerprint["files"]

    assert len(fingerprint["hash"]) == 64
    assert "sim2sim/mujoco/uv.lock" in files
    assert "sim2sim/mujoco/wheelleg_mujoco/evaluation.py" in files
    assert "scripts/evaluate_mujoco.py" in files
    assert "scripts/rank_mujoco_runs.py" in files


def test_formal_ranking_rejects_tampered_aggregate(tmp_path: Path) -> None:
    reports = [_formal_report(tmp_path / f"run-{index}", index) for index in range(1, 5)]
    reports[0]["aggregate"]["full_survival"] = False
    reports[0]["aggregate"]["score"] = 0.0
    reports[0]["report_hash"] = stable_hash(
        {key: value for key, value in reports[0].items() if key != "report_hash"}
    )

    with pytest.raises(ValueError, match="aggregate"):
        validate_evaluation_reports_for_ranking(reports)


def test_ranking_summary_never_auto_qualifies_phase1(tmp_path: Path) -> None:
    reports = [_formal_report(tmp_path / f"run-{index}", index) for index in range(1, 5)]
    summary = build_ranking_summary(reports, [f"run-{index}.json" for index in range(1, 5)])

    assert summary["has_full_survival_candidate"] is True
    assert summary["best_candidate"] is not None
    assert summary["selected"] is None
    assert summary["phase1_qualified"] is False
    assert summary["qualification_status"] == "pending_g08"
