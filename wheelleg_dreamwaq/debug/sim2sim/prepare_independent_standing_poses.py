from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from recovery.contracts import CANONICAL_JOINT_ORDER, file_sha256, stable_hash, write_json
from recovery.independent_evaluate import IndependentMujocoRuntime, R_CONTROL_FROM_USD
from recovery.independent_poses import IndependentPoseBank, import_fallen_records, solve_passive_joints
from recovery.independent_state import StandingTask
from recovery.standing_reference import StandingReference


def snapshot(runtime):
    model, data, mapping = runtime.model, runtime.data, runtime.model_map
    free = mapping.free_joint_id
    qa, va = int(model.jnt_qposadr[free]), int(model.jnt_dofadr[free])
    joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    rotation = data.xmat[mapping.base_body_id].reshape(3, 3)
    return {"joint_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints],
        "joint_positions": [float(data.qpos[model.jnt_qposadr[j]]) for j in joints],
        "joint_velocities": [float(data.qvel[model.jnt_dofadr[j]]) for j in joints],
        "root_position": data.qpos[qa:qa + 3].tolist(), "root_quaternion_wxyz": data.qpos[qa + 3:qa + 7].tolist(),
        "root_linear_velocity_world": data.qvel[va:va + 3].tolist(),
        "root_angular_velocity_world": (rotation @ data.qvel[va + 3:va + 6]).tolist(),
        "joint_coordinate_convention": "USD_native_by_name", "root_frame": "USD_world_wxyz"}


def make_record(runtime, *, source, split, serial, frame, asset_sha256, parameter_sha256, evidence):
    state = runtime.state()
    physical = runtime.inspect_physics()
    if physical["physical_failure"]:
        raise ValueError(f"实际生成初态物理检查失败: {physical['reasons']}")
    native = snapshot(runtime)
    base = runtime.model_map.base_body_id
    root = np.r_[runtime.data.xpos[base], runtime.data.xquat[base],
                 native["root_linear_velocity_world"], native["root_angular_velocity_world"]]
    q = state.joint_position_control[:4]
    if np.any(np.abs(q) > 1.):
        raise ValueError("生成初态超出有界动作")
    return {"record_id": f"independent-{source}-{split}-{serial:04d}",
        "trajectory_id": f"401-nominal-{split}-{source}", "split": split, "source": source,
        "joint_names": native["joint_names"], "joint_position": native["joint_positions"],
        "joint_velocity": native["joint_velocities"], "root_state_usd": root.tolist(),
        "bounded_action": np.r_[q, np.zeros(2)].tolist(), "asset_bundle_sha256": asset_sha256,
        "parameter_plan_sha256": parameter_sha256, "contact_model_sha256": file_sha256(runtime.contact_manifest_path),
        "metadata": {"origin": "401_measured_standing" if source == "standing" else "401_nearstanding_perturbation",
            "mujoco_snapshot": native, "physical_validation_passed": True,
            "standing_gate_passed": source == "standing", "standing_gate_evidence": evidence,
            "q_control_rad": q.tolist(), "projected_gravity": state.projected_gravity_control.tolist(),
            "com_height_m": state.base_height_m, "loop_error_m": physical["loop_error_m"],
            "maximum_penetration_m": physical["penetration_m"], "source_frame": frame,
            "initial_contact_window": "cleared_on_training_reset",
            "source_coordinates": "MuJoCo_full_native_to_USD_by_joint_name_root_link_and_world_twist"}}


def lowest_wheel_height(runtime):
    values = []
    for geom in range(runtime.model.ngeom):
        if runtime.model.geom_bodyid[geom] not in runtime.wheel_bodies or runtime.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = runtime.model.geom_dataid[geom]
        start, count = runtime.model.mesh_vertadr[mesh], runtime.model.mesh_vertnum[mesh]
        vertices = runtime.model.mesh_vert[start:start + count]
        world = vertices @ runtime.data.geom_xmat[geom].reshape(3, 3).T + runtime.data.geom_xpos[geom]
        values.append(world[:, 2].min())
    if len(values) != 2:
        raise ValueError("需要两个真实轮子接触网格")
    return float(min(values))


def perturb(runtime, generator):
    model, data, mapping = runtime.model, runtime.data, runtime.model_map
    free = mapping.free_joint_id
    qa, va = int(model.jnt_qposadr[free]), int(model.jnt_dofadr[free])
    roll, pitch = generator.uniform(-.08, .08, 2)
    for axis, angle in (([1., 0., 0.], roll), ([0., 1., 0.], pitch)):
        rotation, result = np.empty(4), np.empty(4)
        mujoco.mju_axisAngle2Quat(rotation, R_CONTROL_FROM_USD.T @ np.asarray(axis), angle)
        mujoco.mju_mulQuat(result, data.qpos[qa + 3:qa + 7].copy(), rotation)
        data.qpos[qa + 3:qa + 7] = result
    leg_delta = generator.uniform(-.05, .05, 4)
    data.qpos[mapping.qpos_addresses[:4]] += leg_delta
    closure = solve_passive_joints(model, data, CANONICAL_JOINT_ORDER)
    data.qpos[qa + 2] += .001 - lowest_wheel_height(runtime)
    mujoco.mj_forward(model, data)
    rotation = data.xmat[mapping.base_body_id].reshape(3, 3)
    velocity_control = np.r_[generator.uniform(-.05, .05, 2), 0.]
    omega_control = generator.uniform(-.10, .10, 3)
    data.qvel[va:va + 3] = rotation @ R_CONTROL_FROM_USD.T @ velocity_control
    data.qvel[va + 3:va + 6] = R_CONTROL_FROM_USD.T @ omega_control
    mujoco.mj_forward(model, data)
    passive = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
               and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) not in CANONICAL_JOINT_ORDER]
    addresses = np.array([model.jnt_dofadr[j] for j in passive], dtype=int)
    equality = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    jacobian = data.efc_J.reshape(data.nefc, model.nv)[equality]
    data.qvel[addresses] = 0.
    data.qvel[addresses] = np.linalg.lstsq(jacobian[:, addresses], -(jacobian @ data.qvel), rcond=None)[0]
    mujoco.mj_forward(model, data)
    return {"roll_rad": float(roll), "pitch_rad": float(pitch), "leg_delta_rad": leg_delta.tolist(),
            "linear_velocity_control": velocity_control.tolist(), "angular_velocity_control": omega_control.tolist(),
            "closure_solution": closure}


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.reference_dir)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    geometry = json.loads(Path(args.geometry_config).read_text(encoding="utf-8"))
    reference_path = source / Path(summary["models"]["recovery_contact"]["paired_reference"]).name
    reference = StandingReference.load(source / "manifest.json", reference_path, file_sha256(reference_path))
    records, metadata = import_fallen_records(args.fallen_bank)
    plans = {r["parameter_plan_sha256"] for r in records}
    if len(plans) != 1:
        raise ValueError("独立站立初态需要单一已核验名义物理计划")
    parameter_sha256 = next(iter(plans))
    asset_hashes = {r["asset_bundle_sha256"] for r in records}
    if len(asset_hashes) != 1:
        raise ValueError("倒地初态资产身份不唯一")
    runtime = IndependentMujocoRuntime(manifest["recovery_contact_manifest"], geometry_config=geometry,
        q_nominal=manifest["target_contract"]["q_nominal"])
    for record in records:
        runtime.restore_pose(record)
        record["metadata"]["physical_validation_passed"] = True
        record["metadata"]["independent_conversion_rechecked"] = True
    generator = np.random.default_rng(args.seed)
    generated, rejections = [], []
    for split, episode in (("train", 0), ("test", 1)):
        archive_path = source / f"recovery_contact_{episode:02d}.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        task = StandingTask(1, "cpu", reference, runtime.weight_n, mode="eval")
        accepted_frames = []
        for frame in range(len(arrays["time_s"])):
            if frame % 800 == 0:
                task.reset(torch.tensor([0]))
            t = lambda key: torch.tensor(arrays[key][frame:frame + 1], dtype=torch.float32)
            result = task.step(q=t("q_control_rad")[:, :4], dq=t("dq_control_rad_s"),
                omega=t("angular_velocity_control_rad_s"), gravity=t("projected_gravity"),
                linear=t("linear_velocity_control_m_s"), height=t("height_m"),
                wheel_forces=t("wheel_normal_force_n"), nonwheel_force=t("nonwheel_normal_force_n"),
                world_xy=t("base_com_world_m")[:, :2], physical_failure=torch.tensor([False]))
            if frame >= 100 and bool(result.qualified[0]) and int(task.hold_count[0]) >= 100:
                accepted_frames.append(frame)
        if len(accepted_frames) < args.per_group:
            raise ValueError("真实401轨迹未提供足量连续合格站立状态")
        selection = np.linspace(0, len(accepted_frames) - 1, args.per_group, dtype=int)
        for serial, index in enumerate(selection):
            frame = accepted_frames[int(index)]
            for origin in ("standing", "perturbed"):
                accepted = False
                for attempt in range(100):
                    mujoco.mj_resetData(runtime.model, runtime.data)
                    runtime.data.qpos[:] = arrays["qpos_mujoco"][frame]
                    runtime.data.qvel[:] = arrays["qvel_mujoco"][frame]
                    mujoco.mj_forward(runtime.model, runtime.data)
                    perturbation = None if origin == "standing" else perturb(runtime, generator)
                    physical = runtime.inspect_physics()
                    if physical["physical_failure"]:
                        rejections.append({"split": split, "source": origin, "serial": serial,
                                           "attempt": attempt, "reasons": physical["reasons"]})
                        if origin == "standing":
                            raise ValueError("401站立实测状态未通过同源几何核验")
                        continue
                    evidence = {"archive": str(archive_path), "archive_sha256": file_sha256(archive_path),
                        "frame": frame, "consecutive_actual_hold_steps_at_least": 100,
                        "support_evidence": "actual_401_trace_10_control_step_contact_window",
                        "perturbation": perturbation, "seed": args.seed}
                    record = make_record(runtime, source=origin, split=split, serial=serial, frame=frame,
                        asset_sha256=next(iter(asset_hashes)), parameter_sha256=parameter_sha256, evidence=evidence)
                    generated.append(record)
                    accepted = True
                    break
                if not accepted:
                    raise RuntimeError("同来源初态生成达到明确尝试上限，保留拒绝记录")
    bank = IndependentPoseBank(records + generated, {**metadata, "reference_manifest_sha256": file_sha256(source / "manifest.json"),
        "seed": args.seed, "generated_records": len(generated), "rejected_records": len(rejections),
        "rejections": rejections, "generator_sha256": file_sha256(__file__)})
    write_json(output / "pose_bank.json", {"schema_version": "IndependentStandingPoseBankV1",
                                           "metadata": bank.metadata, "records": bank.records})
    initialization = {"fallen_fraction": .6, "standing_fraction": .2, "perturbed_fraction": .2,
                      "train_ids": bank.split_ids()["train"], "test_ids": bank.split_ids()["test"]}
    report = bank.validate_split_ids(initialization)
    write_json(output / "initialization.json", initialization)
    write_json(output / "verification.json", {**report, "generated_records": len(generated),
        "source_fallen_records": len(records), "rejected_records": rejections,
        "physics": "actual_MuJoCo_named_joint_root_COM_conversion_and_closed_chain_solution"})
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallen-bank", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--geometry-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42002)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
