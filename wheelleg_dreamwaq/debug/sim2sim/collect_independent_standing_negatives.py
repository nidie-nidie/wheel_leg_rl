from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from recovery.contracts import file_sha256, write_json
from recovery.independent_evaluate import IndependentMujocoRuntime, R_CONTROL_FROM_USD
from recovery.independent_poses import IndependentPoseBank
from recovery.independent_state import StandingTask
from recovery.standing_reference import StandingReference
from debug.sim2sim.prepare_independent_standing_poses import lowest_wheel_height


def rotate_single_wheel(runtime, angle):
    data, model = runtime.data, runtime.model
    qa = model.jnt_qposadr[runtime.model_map.free_joint_id]
    delta, result = np.empty(4), np.empty(4)
    mujoco.mju_axisAngle2Quat(delta, R_CONTROL_FROM_USD.T @ np.array([1., 0., 0.]), angle)
    mujoco.mju_mulQuat(result, data.qpos[qa + 3:qa + 7].copy(), delta)
    data.qpos[qa + 3:qa + 7] = result
    data.qvel[:] = 0.
    mujoco.mj_forward(model, data)
    data.qpos[qa + 2] += .0001 - lowest_wheel_height(runtime)
    mujoco.mj_forward(model, data)
    runtime.initialize_history()


def collect(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    bank = IndependentPoseBank.load(args.pose_bank)
    geometry = json.loads(Path(args.geometry_config).read_text(encoding="utf-8"))
    manifest = json.loads((Path(args.reference_dir) / "manifest.json").read_text(encoding="utf-8"))
    refpath = Path(args.reference_dir) / "recovery_contact_paired_reference.npz"
    reference = StandingReference.load(Path(args.reference_dir) / "manifest.json", refpath, file_sha256(refpath))
    runtime = IndependentMujocoRuntime(geometry["contact_manifest"], geometry_config=geometry,
                                      q_nominal=manifest["target_contract"]["q_nominal"])
    standing = next(r for r in bank.records if r["source"] == "standing" and r["split"] == "test")
    cases = [("single_wheel", standing), ("fast_sliding", standing)]
    cases.extend((name, next(r for r in bank.records if r["source"] == source and r["split"] == "test"))
                 for name, source in (("body_supported", "front"), ("inverted", "supine"), ("low_body", "back")))
    reports = []
    for name, record in cases:
        runtime.restore_pose(record)
        if name == "single_wheel":
            rotate_single_wheel(runtime, .25)
        if name == "fast_sliding":
            va = runtime.model.jnt_dofadr[runtime.model_map.free_joint_id]
            rotation = runtime.data.xmat[runtime.model_map.base_body_id].reshape(3, 3)
            runtime.data.qvel[va:va + 3] = rotation @ R_CONTROL_FROM_USD.T @ np.array([1., 0., 0.])
            mujoco.mj_forward(runtime.model, runtime.data)
            runtime.initialize_history()
        action = runtime.previous_action.copy()
        task = StandingTask(1, "cpu", reference, runtime.weight_n, mode="eval")
        rows = {key: [] for key in ("q_control_rad", "dq_control_rad_s", "angular_velocity_control_rad_s",
            "projected_gravity", "linear_velocity_control_m_s", "height_m", "wheel_normal_force_n",
            "nonwheel_normal_force_n", "base_com_world_m", "physical_failure", "gate_qualified",
            "qpos_mujoco", "qvel_mujoco", "body_normal_force_n")}
        failures = []
        for _ in range(60):
            measurement = runtime.step(action)
            state = runtime.state()
            t = lambda value: torch.tensor(np.asarray(value)[None], dtype=torch.float32)
            result = task.step(q=t(state.joint_position_control[:4]), dq=t(state.joint_velocity_control),
                omega=t(state.angular_velocity_control), gravity=t(state.projected_gravity_control),
                linear=t(state.com_linear_velocity_control), height=t(state.base_height_m).reshape(1),
                wheel_forces=t(measurement["wheel_forces"]), nonwheel_force=t(measurement["nonwheel_force"]).reshape(1),
                world_xy=t(runtime.data.xipos[runtime.model_map.base_body_id, :2]),
                physical_failure=torch.tensor([measurement["physical_failure"]]))
            body_forces = np.zeros(runtime.model.nbody)
            floor = runtime.model.geom("floor").id
            for contact_index in range(runtime.data.ncon):
                contact = runtime.data.contact[contact_index]
                force = np.zeros(6)
                mujoco.mj_contactForce(runtime.model, runtime.data, contact_index, force)
                geom = contact.geom2 if contact.geom1 == floor else contact.geom1
                body_forces[runtime.model.geom_bodyid[geom]] += max(0., force[0])
            values = (state.joint_position_control, state.joint_velocity_control, state.angular_velocity_control,
                state.projected_gravity_control, state.com_linear_velocity_control, state.base_height_m,
                measurement["wheel_forces"], measurement["nonwheel_force"],
                runtime.data.xipos[runtime.model_map.base_body_id].copy(), measurement["physical_failure"],
                bool(result.qualified[0]), runtime.data.qpos.copy(), runtime.data.qvel.copy(), body_forces)
            for key, value in zip(rows, values, strict=True):
                rows[key].append(value)
            failures.extend(measurement["reasons"])
            if result.success[0]:
                raise ValueError(f"真实异常支撑被完整门控错误接受: {name}")
            if measurement["physical_failure"]:
                break
        arrays = {key: np.asarray(value) for key, value in rows.items()}
        wheel = arrays["wheel_normal_force_n"]
        observed = {"single_wheel": bool(np.any((wheel.max(axis=1) > .1 * runtime.weight_n)
                        & (wheel.min(axis=1) < .01 * runtime.weight_n))),
                    "fast_sliding": bool(np.max(np.linalg.norm(arrays["linear_velocity_control_m_s"][:, :2], axis=1)) > .5),
                    "body_supported": bool(np.max(arrays["nonwheel_normal_force_n"]) > .1 * runtime.weight_n),
                    "inverted": bool(np.max(arrays["projected_gravity"][:, 2]) > .8),
                    "low_body": bool(np.min(arrays["height_m"]) < .185)}
        if not observed[name]:
            raise ValueError(f"真实物理未产生要求的异常状态: {name}")
        np.savez_compressed(output / f"{name}.npz", **arrays)
        reports.append({"name": name, "file": f"{name}.npz", "sha256": file_sha256(output / f"{name}.npz"),
            "source_record_id": record["record_id"], "steps": len(arrays["height_m"]),
            "required_condition_observed": observed[name], "success": False,
            "qualified_steps": int(arrays["gate_qualified"].sum()), "physical_failures": sorted(set(failures))})
    report = {"schema_version": "IndependentStandingPhysicalNegativesV1", "passed": True,
        "contact_manifest": str(runtime.contact_manifest_path), "contact_manifest_sha256": file_sha256(runtime.contact_manifest_path),
        "contact_model_sha256": runtime.contact_manifest["model_xml_sha256"], "pose_bank_sha256": bank.sha256,
        "body_names": [mujoco.mj_id2name(runtime.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(runtime.model.nbody)],
        "wheel_body_names": [mujoco.mj_id2name(runtime.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in runtime.wheel_bodies],
        "weight_n": runtime.weight_n, "episodes": reports, "force_source": "mj_contactForce_actual_floor_normal_force",
        "limits": "finite_actual_rollout_may_end_at_physical_failure; no fabricated_contact_or_dynamics"}
    write_json(output / "manifest.json", report)
    print(json.dumps({"passed": True, "episodes": reports}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-bank", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--geometry-config", required=True)
    parser.add_argument("--output", required=True)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
