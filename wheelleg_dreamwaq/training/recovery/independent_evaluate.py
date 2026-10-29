from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from wheelleg_mujoco.metrics import _max_loop_error
from wheelleg_mujoco.model_map import CONTROLLED_JOINT_SIGNS, build_model_map
from wheelleg_mujoco.observation import collect_kinematic_state

from .contact_model import GeometryAudit, load_contact_model
from .contracts import file_sha256, stable_hash, write_json
from .independent_poses import IndependentPoseBank, solve_passive_joints, solve_passive_velocity
from .independent_state import StandingGateConfig, StandingTask, REASON_LABELS
from .observations import build_actor_frame
from .standing_reference import StandingReference


R_CONTROL_FROM_USD = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))


class IndependentMujocoRuntime:
    def __init__(self, contact_manifest, *, geometry_config, q_nominal):
        self.contact_manifest_path = Path(contact_manifest)
        self.model, self.contact_manifest = load_contact_model(self.contact_manifest_path)
        if self.model.opt.timestep != .001:
            raise ValueError("独立站立MuJoCo要求1ms物理周期")
        self.data = mujoco.MjData(self.model)
        self.model_map = build_model_map(self.model)
        self.kinematic_contract = SimpleNamespace(r_control_from_mujoco=R_CONTROL_FROM_USD)
        self.q_nominal = np.asarray(q_nominal)
        self.geometry_config = geometry_config
        self.geometry_audit = GeometryAudit(self.model, geometry_config["allowed_body_pairs"],
                                             geometry_config["distance_limit_m"])
        self.effort = np.array([18.] * 4 + [9.] * 2)
        self.model.actuator_ctrlrange[self.model_map.actuator_ids, 0] = -self.effort
        self.model.actuator_ctrlrange[self.model_map.actuator_ids, 1] = self.effort
        self.previous_action = np.zeros(6)
        self.history = np.zeros((5, 22), dtype=np.float32)
        self.frame_id = 0
        self.last_torque = np.zeros(6)
        self.last_pd_command = np.zeros(6)
        self.last_feedback_position = np.zeros(6)
        self.last_feedback_velocity = np.zeros(6)
        self.weight_n = float(self.model.body_mass.sum() * 9.81)
        self.wheel_bodies = self.model.jnt_bodyid[self.model_map.joint_ids[4:]]

    def state(self):
        return collect_kinematic_state(self.model, self.data, self.model_map, self.kinematic_contract)

    def _frame(self):
        s = self.state()
        return build_actor_frame(*(torch.tensor(value, dtype=torch.float32) for value in (
            s.angular_velocity_control, s.projected_gravity_control, s.joint_position_control,
            s.joint_velocity_control, self.previous_action, self.q_nominal))).numpy()

    def initialize_history(self):
        q = self.state().joint_position_control[:4]
        if np.any(np.abs(q) > 1.):
            raise ValueError("初态腿位置超出有界动作，不能伪造对应目标")
        self.previous_action = np.r_[q, np.zeros(2)]
        self.history[:] = self._frame()
        self.frame_id = 0
        self.last_torque.fill(0.)
        self.last_pd_command.fill(0.)

    def restore_pose(self, record):
        snapshot = record["metadata"]["mujoco_snapshot"]
        if record["contact_model_sha256"] != file_sha256(self.contact_manifest_path):
            raise ValueError("评估初态接触模型身份不同")
        if snapshot["joint_coordinate_convention"] != "USD_native_by_name" or snapshot["root_frame"] != "USD_world_wxyz":
            raise ValueError("初态坐标约定不支持")
        mujoco.mj_resetData(self.model, self.data)
        free = self.model_map.free_joint_id
        qa, va = self.model.jnt_qposadr[free], self.model.jnt_dofadr[free]
        self.data.qpos[qa:qa + 3] = snapshot["root_position"]
        self.data.qpos[qa + 3:qa + 7] = snapshot["root_quaternion_wxyz"]
        actual = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                  for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
        if actual != set(snapshot["joint_names"]) or len(actual) != len(snapshot["joint_names"]):
            raise ValueError("评估初态没有完整具名主动与从动关节")
        for name, position, velocity in zip(snapshot["joint_names"], snapshot["joint_positions"], snapshot["joint_velocities"], strict=True):
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[j]] = position
            self.data.qvel[self.model.jnt_dofadr[j]] = velocity
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.xmat[self.model_map.base_body_id].reshape(3, 3)
        self.data.qvel[va:va + 3] = snapshot["root_linear_velocity_world"]
        self.data.qvel[va + 3:va + 6] = rotation.T @ snapshot["root_angular_velocity_world"]
        mujoco.mj_forward(self.model, self.data)
        self.initialize_history()
        s = self.state()
        if not np.allclose(s.joint_position_control[:4], record["metadata"]["q_control_rad"], atol=1e-7, rtol=0.):
            raise ValueError("真实恢复后的腿位置不同")
        if abs(s.base_height_m - record["metadata"]["com_height_m"]) > 1e-7:
            raise ValueError("真实恢复后的COM高度不同")
        if self.inspect_physics()["physical_failure"]:
            raise ValueError("评估初态物理检查失败")

    def contacts(self):
        floor = self.model.geom("floor").id
        forces = np.zeros(self.model.nbody)
        penetration = 0.
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if floor not in (contact.geom1, contact.geom2):
                raise RuntimeError("独立站立完整接触模型出现内部实际接触")
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            geom = contact.geom2 if contact.geom1 == floor else contact.geom1
            forces[self.model.geom_bodyid[geom]] += max(0., force[0])
            penetration = max(penetration, -float(contact.dist))
        wheel = forces[self.wheel_bodies]
        return wheel, float(forces.sum() - wheel.sum()), penetration

    def inspect_physics(self):
        wheel, nonwheel, penetration = self.contacts()
        clearance = self.geometry_audit.inspect(self.data)
        loop = float(_max_loop_error(self.model, self.data))
        limited = self.model.jnt_limited.astype(bool)
        violation = False
        for j in np.flatnonzero(limited):
            if self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            q = self.data.qpos[self.model.jnt_qposadr[j]]
            violation |= not self.model.jnt_range[j, 0] <= q <= self.model.jnt_range[j, 1]
        failures = {"nonfinite": not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()),
            "loop_closure": loop >= .005, "joint_limit": bool(violation),
            "excessive_ground_penetration": penetration > self.geometry_config["maximum_ground_penetration_m"],
            "nonallowed_body_clearance": self.geometry_config["internal_clearance_mode"] == "strict"
                and clearance["minimum_nonallowed_clearance_m"] < self.geometry_config["minimum_clearance_m"]}
        reasons = [name for name, value in failures.items() if value]
        return {"physical_failure": bool(reasons), "reasons": reasons,
            "loop_error_m": loop, "penetration_m": penetration,
            "minimum_nonallowed_clearance_m": clearance["minimum_nonallowed_clearance_m"],
            "wheel_forces": wheel, "nonwheel_force": nonwheel}

    def step(self, action, *, external_force_control=None, external_torque_control=None):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.):
            raise ValueError("MuJoCo执行动作必须为六维有界输出")
        failures, max_loop, max_penetration = [], 0., 0.
        warnings_before = sum(int(w.number) for w in self.data.warning)
        impulse = np.zeros(6)
        for _ in range(20):
            s = self.state()
            self.data.xfrc_applied[:] = 0.
            rotation = self.data.xmat[self.model_map.base_body_id].reshape(3, 3)
            for start, value in ((0, external_force_control), (3, external_torque_control)):
                if value is not None:
                    value = np.asarray(value, dtype=np.float64)
                    if value.shape != (3,) or not np.isfinite(value).all():
                        raise ValueError("测评外力与转矩必须为真实有限三维向量")
                    self.data.xfrc_applied[self.model_map.base_body_id, start:start + 3] = rotation @ R_CONTROL_FROM_USD.T @ value
            impulse += self.data.xfrc_applied[self.model_map.base_body_id] * .001
            self.last_feedback_position = s.joint_position_control.copy()
            self.last_feedback_velocity = s.joint_velocity_control.copy()
            pd = np.r_[120. * (action[:4] - s.joint_position_control[:4]) - 4. * s.joint_velocity_control[:4],
                       .6 * (30. * action[4:] - s.joint_velocity_control[4:])]
            torque = np.clip(pd, -self.effort, self.effort)
            self.data.ctrl[:] = 0.
            self.data.ctrl[self.model_map.actuator_ids] = torque * CONTROLLED_JOINT_SIGNS
            mujoco.mj_step(self.model, self.data)
            self.last_torque = self.data.actuator_force[self.model_map.actuator_ids].copy() * CONTROLLED_JOINT_SIGNS
            self.last_pd_command = pd
            mujoco.mj_forward(self.model, self.data)
            measurement = self.inspect_physics()
            failures.extend(measurement["reasons"])
            max_loop = max(max_loop, measurement["loop_error_m"])
            max_penetration = max(max_penetration, measurement["penetration_m"])
            if measurement["physical_failure"]:
                break
        if sum(int(w.number) for w in self.data.warning) != warnings_before:
            failures.append("mujoco_warning")
        self.previous_action = action.copy()
        self.history[:-1] = self.history[1:]
        self.history[-1] = self._frame()
        self.frame_id += 1
        measurement = self.inspect_physics()
        measurement.update({"physical_failure": bool(failures), "reasons": sorted(set(failures)),
                            "loop_error_m": max_loop, "penetration_m": max_penetration,
                            "actual_external_impulse_world": impulse})
        return measurement


def wilson(successes, count):
    if count == 0:
        return None
    z = 1.959963984540054
    fraction, denominator = successes / count, 1. + z * z / count
    center = (fraction + z * z / (2. * count)) / denominator
    radius = z * np.sqrt(fraction * (1. - fraction) / count + z * z / (4. * count * count)) / denominator
    return [0. if successes == 0 else float(max(0., center - radius)),
            1. if successes == count else float(min(1., center + radius))]


def final_suite_cases(bank, *, seed=42003, per_category=100, perturbation_episodes=100):
    if per_category < 1 or perturbation_episodes < 1:
        raise ValueError("最终测评必须包含六类及真实扰动集合")
    cases = []
    serial = 0
    for source in ("front", "back", "left", "right", "supine", "near_upright"):
        candidates = [r for r in bank.records if r["source"] == source and r["split"] == "test"]
        if not candidates:
            raise ValueError(f"最终测评缺少留出姿态: {source}")
        for index in range(per_category):
            cases.append({"case_id": f"nominal-{source}-{index:03d}", "base_record_id": candidates[index % len(candidates)]["record_id"],
                          "seed": seed + serial, "collection": "six_category", "variant": True, "external_perturbation": None})
            serial += 1
    standing = [r for r in bank.records if r["source"] == "standing" and r["split"] == "test"]
    if not standing:
        raise ValueError("扰动测评缺少真实门控接受的留出站立初态")
    stimuli = [{"force_control_n": [sign * magnitude, 0., 0.], "torque_control_nm": [0., 0., 0.],
                "start_s": 5., "duration_s": .10}
               for magnitude in (.5, 1.) for sign in (-1, 1)]
    stimuli += [{"force_control_n": [0., sign * .25, 0.], "torque_control_nm": [0., 0., 0.],
                 "start_s": 5., "duration_s": .10} for sign in (-1, 1)]
    stimuli += [{"force_control_n": [0., 0., 0.], "torque_control_nm": [0., sign * .02, 0.],
                 "start_s": 5., "duration_s": .10} for sign in (-1, 1)]
    for index in range(perturbation_episodes):
        cases.append({"case_id": f"standing-perturbation-{index:03d}",
                      "base_record_id": standing[index % len(standing)]["record_id"], "seed": seed + serial,
                      "collection": "standing_perturbation", "variant": True,
                      "external_perturbation": stimuli[index % len(stimuli)]})
        serial += 1
    return cases


def physical_initial_variant(runtime, record, seed):
    generator = np.random.default_rng(seed)
    rejection_reasons = []
    for attempt in range(100):
        runtime.restore_pose(record)
        model, data, mapping = runtime.model, runtime.data, runtime.model_map
        qa, va = model.jnt_qposadr[mapping.free_joint_id], model.jnt_dofadr[mapping.free_joint_id]
        yaw, roll, pitch = generator.uniform(-np.pi, np.pi), generator.uniform(-.005, .005), generator.uniform(-.005, .005)
        yaw_quat, result = np.empty(4), np.empty(4)
        mujoco.mju_axisAngle2Quat(yaw_quat, np.array([0., 0., 1.]), yaw)
        mujoco.mju_mulQuat(result, yaw_quat, data.qpos[qa + 3:qa + 7].copy())
        data.qpos[qa + 3:qa + 7] = result
        world_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        data.qvel[va:va + 3] = world_yaw @ data.qvel[va:va + 3]
        for axis, angle in (([1., 0., 0.], roll), ([0., 1., 0.], pitch)):
            delta = np.empty(4)
            mujoco.mju_axisAngle2Quat(delta, R_CONTROL_FROM_USD.T @ np.asarray(axis), angle)
            mujoco.mju_mulQuat(result, data.qpos[qa + 3:qa + 7].copy(), delta)
            data.qpos[qa + 3:qa + 7] = result
        leg_delta = generator.uniform(-.005, .005, 4)
        data.qpos[mapping.qpos_addresses[:4]] += leg_delta
        closure = solve_passive_joints(model, data, mapping.joint_names)
        minimum = np.inf
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] == 0 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            world = model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            minimum = min(minimum, float(world[:, 2].min()))
        if not np.isfinite(minimum):
            raise ValueError("最终测评需要真实机器人网格")
        data.qpos[qa + 2] += .0001 - minimum
        mujoco.mj_forward(model, data)
        velocity_residual = solve_passive_velocity(model, data, mapping.joint_names)
        physical = runtime.inspect_physics()
        if physical["physical_failure"]:
            rejection_reasons.append(physical["reasons"])
            continue
        runtime.initialize_history()
        state = runtime.state()
        return {"seed": seed, "attempt": attempt + 1, "yaw_rad": float(yaw), "roll_delta_rad": float(roll),
                "pitch_delta_rad": float(pitch), "leg_delta_rad": leg_delta.tolist(),
                "closure_solution": closure, "rejected_attempts": rejection_reasons,
                "closed_chain_velocity_residual": velocity_residual,
                "physical_validation_passed": True, "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
                "actual_q_control_rad": state.joint_position_control.tolist(),
                "actual_projected_gravity": state.projected_gravity_control.tolist(),
                "actual_com_height_m": state.base_height_m,
                "initial_state_sha256": stable_hash({"qpos": data.qpos.tolist(), "qvel": data.qvel.tolist()})}
    raise RuntimeError("最终测评初态生成达到明确尝试上限")


def evaluate_actor(config, actor, *, record_ids=None, case_specs=None, output_dir=None, trace_limit_bytes=512 * 1024 * 1024):
    torch.set_num_threads(1)
    bank = IndependentPoseBank.load(config["environment"]["pose_bank_path"])
    bank.validate_split_ids(config["initialization"])
    reference = config["reference"]
    reference = StandingReference.load(reference["manifest_path"], reference["bank_path"], reference["bank_sha256"],
                                      expected_manifest_sha256=reference["manifest_sha256"])
    runtime = IndependentMujocoRuntime(config["geometry_guard"]["contact_manifest"],
                                     geometry_config=config["geometry_guard"], q_nominal=config["contract"]["q_nominal"])
    selected = (record_ids if record_ids is not None else
                [value for values in config["initialization"]["test_ids"].values() for value in values])
    if len(selected) != len(set(selected)) or any(value not in bank.by_id or bank.by_id[value]["split"] != "test" for value in selected):
        raise ValueError("独立测评必须使用不重复的留出初态ID")
    cases = ([{"case_id": value, "base_record_id": value, "seed": None, "collection": "selected_test_ids",
               "variant": False, "external_perturbation": None} for value in selected]
             if case_specs is None else case_specs)
    if len({case["case_id"] for case in cases}) != len(cases) or any(
            case["base_record_id"] not in bank.by_id or bank.by_id[case["base_record_id"]]["split"] != "test" for case in cases):
        raise ValueError("批量测评需要独立case ID与真实留出来源")
    output = None if output_dir is None else Path(output_dir)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    episodes, written = [], 0
    for number, case in enumerate(cases):
        record_id = case["base_record_id"]
        record = bank.by_id[record_id]
        runtime.restore_pose(record)
        initialization = physical_initial_variant(runtime, record, case["seed"]) if case["variant"] else None
        task = StandingTask(1, "cpu", reference, runtime.weight_n, mode="eval", config=StandingGateConfig(**config["standing"]))
        rows = []
        max_tilt = max_speed = max_displacement = max_loop = max_penetration = 0.
        saturation = 0
        applied_perturbation_steps = 0
        actual_external_impulse = np.zeros(6)
        physical_reasons = []
        for step in range(900):
            with torch.inference_mode():
                action = actor(torch.from_numpy(runtime.history.reshape(1, 110))).detach().cpu().numpy()[0]
            perturbation = case["external_perturbation"]
            active = perturbation is not None and perturbation["start_s"] <= step * .02 < perturbation["start_s"] + perturbation["duration_s"]
            measurement = runtime.step(action, external_force_control=perturbation["force_control_n"] if active else None,
                external_torque_control=perturbation["torque_control_nm"] if active else None)
            applied_perturbation_steps += int(active)
            actual_external_impulse += measurement["actual_external_impulse_world"]
            s = runtime.state()
            t = lambda v: torch.tensor(np.asarray(v)[None], dtype=torch.float32)
            result = task.step(q=t(s.joint_position_control[:4]), dq=t(s.joint_velocity_control),
                omega=t(s.angular_velocity_control), gravity=t(s.projected_gravity_control),
                linear=t(s.com_linear_velocity_control), height=t(s.base_height_m).reshape(1),
                wheel_forces=t(measurement["wheel_forces"]), nonwheel_force=t(measurement["nonwheel_force"]).reshape(1),
                world_xy=t(runtime.data.xipos[runtime.model_map.base_body_id, :2]),
                physical_failure=torch.tensor([measurement["physical_failure"]]))
            max_tilt = max(max_tilt, float(np.arccos(np.clip(-s.projected_gravity_control[2], -1., 1.))))
            max_speed = max(max_speed, float(np.linalg.norm(s.com_linear_velocity_control[:2])))
            max_displacement = max(max_displacement, float(task.position_delta.norm()))
            max_loop, max_penetration = max(max_loop, measurement["loop_error_m"]), max(max_penetration, measurement["penetration_m"])
            saturation += int(np.any(np.abs(runtime.last_torque) >= runtime.effort * .99))
            physical_reasons.extend(measurement["reasons"])
            rows.append(np.r_[step + 1, s.joint_position_control, s.joint_velocity_control,
                              s.angular_velocity_control, s.projected_gravity_control, s.base_height_m,
                              s.com_linear_velocity_control, action, measurement["wheel_forces"],
                              measurement["nonwheel_force"], int(task.phase[0]), int(task.hold_count[0])])
            if result.success[0] or result.failure[0]:
                break
        episode = {"record_id": record_id, "trajectory_id": record["trajectory_id"], "source": record["source"],
            "case_id": case["case_id"], "collection": case["collection"], "seed": case["seed"],
            "initialization": initialization, "external_perturbation": case["external_perturbation"],
            "applied_perturbation_control_steps": applied_perturbation_steps,
            "actual_external_impulse_world": actual_external_impulse.tolist(),
            "success": bool(result.success[0]), "reason": REASON_LABELS[int(result.reason_codes[0])],
            "physical_reasons": sorted(set(physical_reasons)), "control_steps": step + 1,
            "first_qualified_step": int(task.first_qualified_step[0]), "hold_steps": int(task.hold_count[0]),
            "maximum_tilt_rad": max_tilt, "maximum_horizontal_speed_m_s": max_speed,
            "maximum_hold_displacement_m": max_displacement, "maximum_loop_error_m": max_loop,
            "maximum_penetration_m": max_penetration, "torque_saturation_step_fraction": saturation / (step + 1)}
        if output is not None and number < 12:
            values = np.asarray(rows)
            if written + values.nbytes > trace_limit_bytes:
                raise RuntimeError("独立测评完整轨迹超出明确配额")
            np.savez_compressed(output / f"trace_{number:03d}.npz", values=values)
            written += values.nbytes
        episodes.append(episode)
    groups = {source: {"successes": sum(e["success"] for e in episodes if e["source"] == source),
                       "episodes": sum(e["source"] == source for e in episodes),
                       "wilson_95": wilson(sum(e["success"] for e in episodes if e["source"] == source),
                                           sum(e["source"] == source for e in episodes))}
              for source in sorted({e["source"] for e in episodes})}
    collections = {}
    for name in sorted({e["collection"] for e in episodes}):
        subset = [e for e in episodes if e["collection"] == name]
        successes = sum(e["success"] for e in subset)
        getup = sum(e["first_qualified_step"] >= 0 for e in subset)
        collections[name] = {"successes": successes, "episodes": len(subset), "wilson_95": wilson(successes, len(subset)),
                             "completed_getup": getup, "hold_failures_after_getup": getup - successes,
                             "hold_failure_fraction_after_getup": None if getup == 0 else (getup - successes) / getup}
    report = {"schema_version": "IndependentStandingEvaluationV1", "config_sha256": stable_hash(config),
        "pose_bank_sha256": bank.sha256, "reference_identity": reference.identity,
        "successes": sum(e["success"] for e in episodes), "episodes": episodes, "source_groups": groups,
        "collections": collections, "base_record_counts": dict(Counter(e["record_id"] for e in episodes)),
        "sampling_scope": "finite_base_pose_bank_with_independently_generated_physical_yaw_and_small_perturbations",
        "statistical_limit": "reused_base_poses_are_correlated; Wilson_intervals_do_not_establish_pose_diversity",
        "physical_failure_counts": dict(Counter(reason for e in episodes for reason in e["physical_reasons"])),
        "trace_array_bytes": written, "torque_semantics": "MuJoCo_actual_motor_generalized_force",
        "gate": "8s_GETUP_then_10s_continuous_HOLD"}
    report["final_suite_complete"] = len(episodes) == 700 and collections.get("six_category", {}).get("episodes") == 600 and collections.get("standing_perturbation", {}).get("episodes") == 100
    report["nominal_behavior_targets_met"] = None
    if report["final_suite_complete"]:
        nominal = collections["six_category"]
        report["nominal_behavior_targets_met"] = (nominal["successes"] >= 540
            and all(groups[source]["successes"] >= 80 for source in ("front", "back", "left", "right", "supine", "near_upright"))
            and nominal["hold_failure_fraction_after_getup"] is not None
            and nominal["hold_failure_fraction_after_getup"] <= .05
            and not report["physical_failure_counts"])
    if output is not None:
        write_json(output / "evaluation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--record-ids", nargs="+")
    parser.add_argument("--suite", choices=("selected", "final"), default="selected")
    parser.add_argument("--seed", type=int, default=42003)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    actor = torch.jit.load(args.actor, map_location="cpu").eval()
    if args.suite == "final" and args.record_ids:
        raise ValueError("完整最终测评自行构建600+100集合")
    bank = IndependentPoseBank.load(config["environment"]["pose_bank_path"])
    cases = final_suite_cases(bank, seed=args.seed) if args.suite == "final" else None
    report = evaluate_actor(config, actor, record_ids=args.record_ids, case_specs=cases, output_dir=args.output)
    report["actor_sha256"] = file_sha256(args.actor)
    write_json(Path(args.output) / "evaluation.json", report)
    print(json.dumps({"successes": report["successes"], "episodes": len(report["episodes"])}))


if __name__ == "__main__":
    main()
