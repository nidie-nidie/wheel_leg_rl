"""Audit aligned results, immutable policies and the exact production diff."""
from __future__ import annotations

import difflib
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
OLD = ROOT.parent / "mechanics-isolation-20261010-v1"
SUITE = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(PROJECT / "sim2sim/mujoco"))
sys.path.insert(0, str(PROJECT / "source/wheelleg_dreamwaq"))
from analyze import JOINTS, SIGNS, TIMES, sample, velocity, rms
from wheelleg_mujoco.angular_limit import angular_limit_contract
from wheelleg_mujoco.evaluation import build_evaluation_source_fingerprint, validate_evaluation_reports_for_ranking


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    execution = read(ROOT / "aligned-execution.json")
    records = execution["records"]
    assert len(records) == 8
    assert all(r["exit_code"] == 0 and r["evidence_complete"] for r in records)
    assert "40 passed" in Path(records[0]["console_log"]).read_text(encoding="utf-8")
    for path, expected in execution["immutable_inputs_sha256"].items():
        assert digest(path) == expected, path
    provenance = dict(execution["immutable_inputs_sha256"])
    for record in records:
        for path, expected in record["input_hashes"].items():
            assert digest(path) == expected
            provenance[path] = expected
        provenance[record["console_log"]] = digest(record["console_log"])
    protected = read(ROOT / "protected-before.json")
    allowed = {str((PROJECT / p).resolve()) for p in (
        "sim2sim/mujoco/wheelleg_mujoco/runner.py", "sim2sim/mujoco/wheelleg_mujoco/contract.py",
        "sim2sim/mujoco/wheelleg_mujoco/evaluation.py", "../docs/2026-10-03-wheelleg-dreamwaq-architecture.md")}
    assert {p for p, h in protected.items() if digest(p) != h} == allowed
    for path, expected in protected.items():
        if path not in allowed:
            assert digest(path) == expected
    torque_input = read(ROOT.parent / "same-torque-20261010-v1/inputs.json")
    torques = np.asarray(torque_input["shared_first_torque_control_nm"])
    response_rows = []
    candidate_errors = {}
    initial_errors = {}
    for case in ("baseline", "closure_off"):
        current_path = ROOT / f"mujoco-{case}-aligned/evidence.json"
        candidate_path = ROOT / f"mujoco-{case}-limit-0.005/evidence.json"
        isaac_path = OLD / f"isaac-{case}/evidence.json"
        baseline_path = OLD / f"mujoco-{case}/evidence.json"
        current, candidate, isaac, baseline = map(read, (current_path, candidate_path, isaac_path, baseline_path))
        for path in (current_path, candidate_path, isaac_path, baseline_path):
            provenance[str(path)] = digest(path)
        assert current["angular_limit"] == angular_limit_contract()
        assert current["physics_dt_s"] == .001 and current["control_substeps"] == 20
        assert current["reset_physics_time_advanced_s"] == 0.
        assert current["mechanics_before"] == baseline["mechanics_before"]
        assert current["mechanics_after"] == baseline["mechanics_after"]
        assert current["sources_unchanged"] and current["observer_replay_max_errors"] == [0.] * 8
        fields = ("root_qpos", "root_qvel", "joint_position", "joint_velocity")
        candidate_errors[case] = max(float(np.abs(np.asarray(sample(current, "mujoco", e, ms)[f]) -
                                                        np.asarray(sample(candidate, "mujoco", e, ms)[f])).max())
                                     for e in range(8) for ms in (0, *TIMES) for f in fields)
        assert candidate_errors[case] == 0.
        initial_errors[case] = max(float(np.abs(np.asarray(sample(current, "mujoco", e, 0)[f]) -
                                                      np.asarray(sample(baseline, "mujoco", e, 0)[f])).max())
                                   for e in range(8) for f in fields)
        assert initial_errors[case] == 0.
        for row in current["samples"]:
            if row["physics_step"] == 0:
                continue
            expected = torques[row["environment_index"]] * SIGNS
            for field in ("external_ctrl_native_nm", "actuator_force_native_nm", "generalized_actuator_force_native_nm"):
                assert np.array_equal(np.asarray(row[field]), expected)
            assert row["current_geometry_contact_count"] == 0 and not row["contacts_last_solve"]
        for ms in TIMES:
            before_angular, after_angular, before_joint, after_joint = [], [], [], []
            for e in range(8):
                ia, ij, _ = velocity(isaac, "isaac", e, ms)
                ba, bj, _ = velocity(baseline, "mujoco", e, ms)
                aa, aj, _ = velocity(current, "mujoco", e, ms)
                before_angular.append(float(np.linalg.norm(ia - ba)))
                after_angular.append(float(np.linalg.norm(ia - aa)))
                before_joint.append(rms(ij - bj))
                after_joint.append(rms(ij - aj))
            response_rows.append({"case": case, "time_ms": ms,
                "mean_angular_error_before_rad_s": float(np.mean(before_angular)),
                "mean_angular_error_after_rad_s": float(np.mean(after_angular)),
                "mean_joint_rms_before_rad_s": float(np.mean(before_joint)),
                "mean_joint_rms_after_rad_s": float(np.mean(after_joint))})
    reports = [read(ROOT / f"mujoco-evaluation/run-{i:02d}/summary.json") for i in range(1, 5)]
    validated = validate_evaluation_reports_for_ranking(reports)
    assert validated["rigid_body_angular_limit"] == angular_limit_contract()
    ranking = read(ROOT / "mujoco-ranking-summary.json")
    assert ranking["has_full_survival_candidate"]
    policy_rows = []
    for i, report in enumerate(reports, 1):
        export = SUITE / f"exports/run-{i:02d}"
        policy = read(export / "policy_manifest.json")
        manifest = read(policy["source_run_manifest"])
        task = manifest["base_task_contract"]
        assert task["contract_hash"] == policy["base_task_contract_hash"] == report["base_task_contract_hash"]
        isaac_cap = task["runtime"]["robot_except_asset_absolute_path"]["spawn"]["rigid_props"]["max_angular_velocity"]
        assert isaac_cap == 100.
        checkpoint = torch.load(policy["source_checkpoint"], map_location="cpu", weights_only=False)
        assert checkpoint["infos"]["base_task_contract_hash"] == task["contract_hash"]
        assert checkpoint["infos"]["completed_iterations"] == 1000
        golden = torch.load(export / "golden_vectors.pt", map_location="cpu", weights_only=False)
        assert golden["seed"] == 20261007 and list(golden["history"].shape) == [32, 125]
        actor = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
        with torch.inference_mode():
            error = float((actor(golden["history"]) - golden["expected_action_mean"]).abs().max())
        assert error <= 1.e-7
        before = read(SUITE / f"mujoco_evaluation/run-{i:02d}/summary.json")["aggregate"]
        after = report["aggregate"]
        assert after["completed_scenarios"] == 8 and after["full_survival"] and after["survival_fraction"] == 1.
        assert all(s["ticks"] == 500 and s["failure_reason"] is None for s in after["scenarios"])
        isaac = read(SUITE / f"isaac_evaluation/run-{i:02d}/summary.json")
        tracking = {}
        for scenario in after["scenarios"]:
            name = scenario["name"]
            csv_path = Path(report["scenario_files"][name]["path"])
            assert digest(csv_path) == report["scenario_files"][name]["sha256"]
            provenance[str(csv_path)] = digest(csv_path)
            with csv_path.open(encoding="utf-8", newline="") as stream:
                samples = list(csv.DictReader(stream))
            assert len(samples) == 500 and all(s["survived"] == "1" for s in samples)
            values = np.asarray([float(s["vx_mps"]) for s in samples])
            times = np.asarray([float(s["time_s"]) for s in samples])
            target = float(samples[0]["target_vx_mps"])
            steady = values[times >= 2.]
            assert len(steady) >= 399
            assert abs(float(np.mean(np.abs(values - target))) - scenario["vx_mae"]) < 1.e-12
            tracking[name] = {"target_vx_mps": target, "full_mean_vx_mps": float(values.mean()),
                              "full_vx_mae_mps": float(np.abs(values - target).mean()),
                              "steady_window_start_s": 2., "steady_mean_vx_mps": float(steady.mean()),
                              "steady_vx_mae_mps": float(np.abs(steady - target).mean())}
        row = {"run_index": i, "source_seed": policy["source_seed"], "isaac_cap_deg_s": isaac_cap,
               "golden_max_abs_error": error, "completed_before": before["completed_scenarios"],
               "completed_after": after["completed_scenarios"], "survival_before": before["survival_fraction"],
               "survival_after": after["survival_fraction"], "score_after": after["score"], "metrics_after": after["metrics"],
               "isaac_velocity_gate": isaac["estimator"]["velocity_acceptance"],
               "isaac_not_worse_than_baseline": isaac["baseline_comparison"]["candidate_not_worse"],
               "velocity_tracking": tracking}
        policy_rows.append(row)
        provenance[str(ROOT / f"mujoco-evaluation/run-{i:02d}/summary.json")] = digest(ROOT / f"mujoco-evaluation/run-{i:02d}/summary.json")
        provenance[str(SUITE / f"isaac_evaluation/run-{i:02d}/summary.json")] = digest(SUITE / f"isaac_evaluation/run-{i:02d}/summary.json")
    mappings = {
        "../docs/2026-10-03-wheelleg-dreamwaq-architecture.md": ROOT / "source-before/2026-10-03-wheelleg-dreamwaq-architecture.md",
        "sim2sim/mujoco/wheelleg_mujoco/runner.py": ROOT / "source-before/runner.py",
        "sim2sim/mujoco/wheelleg_mujoco/contract.py": ROOT / "source-before/sim2sim/mujoco/wheelleg_mujoco/contract.py",
        "sim2sim/mujoco/wheelleg_mujoco/evaluation.py": ROOT / "source-before/sim2sim/mujoco/wheelleg_mujoco/evaluation.py",
        "sim2sim/mujoco/tests/test_evaluation_metrics.py": ROOT / "source-before/sim2sim/mujoco/tests/test_evaluation_metrics.py",
        "sim2sim/mujoco/wheelleg_mujoco/angular_limit.py": None,
        "sim2sim/mujoco/tests/test_angular_limit.py": None,
        "docs/superpowers/plans/2026-10-10-mujoco-angular-limit-alignment.md": None,
    }
    changes = []
    diff_parts = []
    for relative, original in mappings.items():
        path = (PROJECT / relative).resolve()
        old = [] if original is None else original.read_text(encoding="utf-8").splitlines(keepends=True)
        new = path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines = list(difflib.unified_diff(old, new, fromfile="before/" + relative, tofile="after/" + relative))
        diff_parts.extend(lines)
        changes.append({"path": relative, "added": sum(line.startswith('+') and not line.startswith('+++') for line in lines),
                        "removed": sum(line.startswith('-') and not line.startswith('---') for line in lines)})
        provenance[str(path)] = digest(path)
    (ROOT / "production-changes.diff").write_text(''.join(diff_parts), encoding="utf-8")
    result = {"schema_version": "AngularAlignmentResultsV1", "passed": True, "tests_passed": 40,
              "training_performed": False, "protected_original_files": len(protected), "allowed_original_changes": sorted(allowed),
              "immutable_input_count": len(execution["immutable_inputs_sha256"]), "response": response_rows,
              "initial_state_max_errors": initial_errors, "candidate_reproduction_max_errors": candidate_errors,
              "policies": policy_rows, "best_checkpoint": ranking["best_candidate"]["source_checkpoint"],
              "best_run_id": ranking["best_candidate"]["run_id"], "production_changes": changes,
              "architecture_sha256": digest(PROJECT.parent / "docs/2026-10-03-wheelleg-dreamwaq-architecture.md"),
              "evaluation_implementation": build_evaluation_source_fingerprint(), "provenance_sha256": provenance,
              "phase2_performance_qualified": False,
              "remaining_limits": ["first_5ms_closed_constraint_response", "standing_vx_drift", "isaac_baseline_performance_gate"]}
    (ROOT / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("ALIGNED_RESULTS_VERIFIED", len(records), "processes", len(policy_rows) * 8, "scenarios", "golden", [r["golden_max_abs_error"] for r in policy_rows])
    print("PRODUCTION_CHANGES", json.dumps(changes))


if __name__ == "__main__":
    main()
