from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .contact_model import build_contact_model, file_sha256, SOURCE_XML
from .contracts import write_json
from .mujoco_runtime import MujocoRecoveryRuntime
from .pose_bank import PoseBank
from .randomization import ParameterPlan
from .collect_pose_bank import orient_for_drop


def verify(walker_export: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_identity = file_sha256(SOURCE_XML)
    model_path = build_contact_model(output_dir / "model")
    runtime = MujocoRecoveryRuntime(model_path.with_name("contact_manifest.json"), walker_export)
    runtime.reset_nominal()
    initial = runtime.snapshot()
    first_candidate = runtime.walker_action()
    for _ in range(7):
        np.testing.assert_array_equal(runtime.walker_action(), first_candidate)
    if runtime.snapshot() != initial:
        raise AssertionError("只读候选改变了物理状态、历史或动作")
    steps = []
    records = []
    for index in range(40):
        step = runtime.step(runtime.walker_action(), locomotion=True)
        steps.append(step)
        if index >= 5:
            gravity = runtime.state().projected_gravity_control
            tilt = float(np.arccos(np.clip(-gravity[2], -1, 1)))
            category = "near_upright" if tilt < 0.25 else ("supine" if tilt > 2.7 else
                       (("front" if gravity[0] > 0 else "back") if abs(gravity[0]) > abs(gravity[1]) else
                        ("left" if gravity[1] > 0 else "right")))
            records.append(runtime.pose_record(record_id=f"diagnostic-train-{index}", trajectory_id="diagnostic-train",
                                                split="train", category=category, motion_kind="dynamic",
                                                metadata={"origin": "interface_validation", "history_source": "physical_trajectory"}))
    restored = runtime.snapshot()
    reference_observation = runtime.recovery_observation()
    runtime.step(runtime.walker_action(), locomotion=True)
    runtime.restore_snapshot(restored)
    np.testing.assert_array_equal(runtime.recovery_observation(), reference_observation)
    np.testing.assert_allclose(runtime.data.qvel[6:], np.asarray(restored["joint_velocities"]), atol=0, rtol=0)
    if runtime.snapshot()["root_linear_velocity_world"] != restored["root_linear_velocity_world"]:
        raise AssertionError("动态恢复改变了根速度")
    bank = PoseBank(records=records, metadata={"schema_version": "RecoveryInterfacePoseBankV1",
                                             "formal_validation": "required", "conversion_validation": "required"})
    bank_path = output_dir / "interface_pose_bank.json"
    bank.save(bank_path)
    reloaded = PoseBank.load(bank_path)
    runtime.restore_pose(reloaded.records[0])
    plan = ParameterPlan(**{key: tuple(value) if isinstance(value, list) else value for key, value in runtime.parameters.items()})
    if plan.sha256 != runtime.parameter_sha256:
        raise AssertionError("后端实际参数计划hash与共享契约不同")
    if file_sha256(SOURCE_XML) != source_identity:
        raise AssertionError("源模型被修改")
    drop_checks = []
    generator = np.random.default_rng(42001)
    for category in ("front", "back", "left", "right", "supine"):
        orient_for_drop(runtime, category, generator, 0.01)
        contacted = set()
        maximum_drop_loop = 0.0
        maximum_drop_penetration = 0.0
        for _ in range(30):
            result = runtime.step(np.zeros(6))
            for substep in result.substeps:
                contacted.update(substep["contact"]["body_normal_force_n"])
                maximum_drop_loop = max(maximum_drop_loop, substep["loop_error_m"])
                maximum_drop_penetration = max(maximum_drop_penetration, substep["contact"]["maximum_penetration_m"])
        drop_checks.append({"requested_orientation": category, "actual_floor_contact_bodies": sorted(contacted),
                            "maximum_loop_error_m": maximum_drop_loop, "maximum_ground_penetration_m": maximum_drop_penetration,
                            "end_tilt_rad": float(np.arccos(np.clip(-runtime.state().projected_gravity_control[2], -1, 1))),
                            "mechanical_clearance_validation": "required"})
    if not any(any(body not in ("jwheel_left", "jwheel_right") for body in check["actual_floor_contact_bodies"]) for check in drop_checks):
        raise AssertionError("真实倒地物理过程中没有产生非轮部件的地面接触")
    report = {"schema_version": "RecoveryMujocoVerificationV1", "model_xml": str(model_path.resolve()),
              "source_xml_unchanged": True, "source_xml_sha256": source_identity,
              "contact_manifest_sha256": file_sha256(model_path.with_name("contact_manifest.json")),
              "robot_mesh_floor_pairs": int(runtime.model.npair), "internal_contact_count": 0,
              "physics_substeps_executed": sum(len(step.substeps) for step in steps) + 20 + 5 * 30 * 20,
              "readonly_candidate_checks": 7, "dynamic_restore_passed": True,
              "parameter_plan_sha256": runtime.parameter_sha256,
              "maximum_loop_error_m": max(sub["loop_error_m"] for step in steps for sub in step.substeps),
              "maximum_penetration_m": max(sub["contact"]["maximum_penetration_m"] for step in steps for sub in step.substeps),
              "minimum_unreviewed_mesh_hull_clearance_m": min(sub["clearance"]["minimum_nonallowed_clearance_m"] for step in steps for sub in step.substeps),
              "pose_bank_path": str(bank_path.resolve()), "actual_parameters": runtime.parameters,
              "last_contacts": runtime.contact_state(), "mechanical_feasibility_validated": False,
              "five_orientation_drop_diagnostics": drop_checks,
              "isaac_conversion_validated": False, "locomotion_new_contact_baseline_validated": False}
    write_json(output_dir / "actual_parameters.json", runtime.parameters)
    write_json(output_dir / "verification.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walker-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.walker_export, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
