from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovery.contracts import file_sha256, write_json
from recovery.independent_evaluate import IndependentMujocoRuntime
from recovery.independent_poses import IndependentPoseBank


def verify(args):
    bank = IndependentPoseBank.load(args.pose_bank)
    geometry = json.loads(Path(args.geometry_config).read_text(encoding="utf-8"))
    reference = json.loads(Path(args.reference_manifest).read_text(encoding="utf-8"))
    runtime = IndependentMujocoRuntime(geometry["contact_manifest"], geometry_config=geometry,
                                      q_nominal=reference["target_contract"]["q_nominal"])
    results = []
    for source in ("front", "back", "left", "right", "supine", "near_upright"):
        record = next(r for r in bank.records if r["source"] == source and r["split"] == "test")
        for joint in range(6):
            for bound in (-1., 1.):
                runtime.restore_pose(record)
                action = runtime.previous_action.copy()
                action[joint] = bound
                physics = runtime.step(action)
                inferred = np.r_[(runtime.last_pd_command[:4] + 4. * runtime.last_feedback_velocity[:4]) / 120.
                                    + runtime.last_feedback_position[:4],
                                 runtime.last_pd_command[4:] / .6 + runtime.last_feedback_velocity[4:]]
                expected = np.r_[action[:4], 30. * action[4:]]
                mapping_error = float(np.max(np.abs(inferred - expected)))
                actual_torque_finite = bool(np.isfinite(runtime.last_torque).all()
                    and np.all(np.abs(runtime.last_torque) <= runtime.effort + 1e-7))
                results.append({"source": source, "initial_state_id": record["record_id"],
                    "joint": runtime.model_map.joint_names[joint], "bound": bound,
                    "expected_targets": expected.tolist(), "targets_inferred_from_real_pd": inferred.tolist(),
                    "maximum_target_error": mapping_error, "actual_torque": runtime.last_torque.tolist(),
                    "actual_torque_within_limits": actual_torque_finite,
                    "loop_error_m": physics["loop_error_m"], "penetration_m": physics["penetration_m"],
                    "physical_failure": physics["physical_failure"], "physical_reasons": physics["reasons"]})
    report = {"schema_version": "IndependentStandingActionPhysicalCheckV1", "cases": results,
        "mapped_targets_match": all(r["maximum_target_error"] < 1e-10 for r in results),
        "actual_torque_within_limits": all(r["actual_torque_within_limits"] for r in results),
        "physical_failure_cases": sum(r["physical_failure"] for r in results),
        "six_categories": sorted({r["source"] for r in results}),
        "contact_manifest_sha256": file_sha256(runtime.contact_manifest_path), "pose_bank_sha256": bank.sha256,
        "action_range_scope": "simulation_control_design_ranges; no_hardware_mechanical_range_claim"}
    report["passed"] = report["mapped_targets_match"] and report["actual_torque_within_limits"] and report["physical_failure_cases"] == 0
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-bank", required=True)
    parser.add_argument("--geometry-config", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--output", required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
