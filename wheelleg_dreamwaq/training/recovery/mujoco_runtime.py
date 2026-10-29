from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from wheelleg_mujoco.contract import AdapterContract, load_policy_contract
from wheelleg_mujoco.metrics import _max_loop_error
from wheelleg_mujoco.model_map import CONTROLLED_JOINT_SIGNS, build_model_map
from wheelleg_mujoco.observation import build_actor_observation, collect_kinematic_state, _normalize_command

from .contact_model import GeometryAudit, PROJECT_ROOT, file_sha256, load_contact_model, payload_sha256
from .contracts import RecoveryContractV2
from .control import decode_action, decode_walker_action, estimate_pd_torque
from .observations import build_actor_frame


def _tensor(values) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32)


@dataclass(frozen=True)
class RuntimeStep:
    frame_id: int
    action: np.ndarray
    leg_target: np.ndarray
    wheel_target: np.ndarray
    torque: np.ndarray
    substeps: tuple[dict, ...]
    contacts: dict
    loop_error_m: float
    clearance: dict


class MujocoRecoveryRuntime:
    def __init__(self, contact_manifest: str | Path, walker_export: str | Path, *,
                 allowed_body_pairs: list[list[str]] | None = None, actual_parameters: dict | None = None):
        self.contact_manifest_path = Path(contact_manifest).resolve()
        self.model, self.contact_manifest = load_contact_model(self.contact_manifest_path)
        export = Path(walker_export).resolve()
        source_model = PROJECT_ROOT / "sim2sim/mujoco/model_manifest.json"
        self.walker_contract, self.walker_manifest, _ = load_policy_contract(
            export / "policy_manifest.json", export / "actor.ts", source_model)
        self.walker = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
        for parameter in self.walker.parameters():
            parameter.requires_grad_(False)
        if not np.isinf(self.walker_contract.action_clip):
            raise ValueError("起身组合需要已确认的原始动作行走导出")
        self.contract = RecoveryContractV2()
        if not np.allclose(self.walker_contract.q_nominal, self.contract.q_nominal, atol=1e-7, rtol=0):
            raise ValueError("起身与行走名义站姿不一致")
        if abs(float(self.model.opt.timestep) - self.contract.mujoco_physics_dt_s) > 1e-15:
            raise ValueError("MuJoCo 控制时序不一致")
        self.data = mujoco.MjData(self.model)
        self.model_map = build_model_map(self.model)
        self.geometry_audit = GeometryAudit(self.model, allowed_body_pairs or [])
        self.allowed_body_pairs = allowed_body_pairs
        self.previous_action = np.zeros(6)
        self.prior_actions: list[np.ndarray] = []
        self.frame_id = 0
        self.history = np.empty((5, 22), dtype=np.float32)
        self.clean_history = np.empty((5, 22), dtype=np.float32)
        self.walker_history = np.empty((5, 25), dtype=np.float32)
        self.leg_target = np.asarray(self.contract.q_nominal).copy()
        self.wheel_target = np.zeros(2)
        self.last_torque = np.zeros(6)
        self._ready = False
        self.walker_identity = {"manifest_sha256": file_sha256(export / "policy_manifest.json"),
                                "actor_sha256": file_sha256(export / "actor.ts"), "directory": str(export)}
        self.configure_parameters(actual_parameters)

    def configure_parameters(self, values: dict | None) -> None:
        if self._ready:
            raise RuntimeError("实际物理参数只能在首次 reset 前设置")
        nominal = {"kp": [120.0] * 4 + [0.0] * 2, "kd": [4.0] * 4 + [0.6] * 2,
                   "effort_limits": [18.0] * 4 + [9.0] * 2, "q_reference": list(self.contract.q_nominal),
                   "friction": float(self.contact_manifest["friction"])}
        self.parameters = nominal if values is None else dict(values)
        if set(self.parameters) != set(nominal):
            raise ValueError("实际参数必须完整指定 kp/kd/effort_limits/q_reference/friction")
        self.kp = np.asarray(self.parameters["kp"], dtype=np.float64)
        self.kd = np.asarray(self.parameters["kd"], dtype=np.float64)
        self.effort_limits = np.asarray(self.parameters["effort_limits"], dtype=np.float64)
        self.q_reference = np.asarray(self.parameters["q_reference"], dtype=np.float64)
        if any(value.shape != (6,) or not np.isfinite(value).all() for value in (self.kp, self.kd, self.effort_limits)):
            raise ValueError("实际执行器参数必须为六维有限向量")
        if self.q_reference.shape != (4,) or not np.isfinite(self.q_reference).all():
            raise ValueError("执行器参考必须为四维有限向量")
        if np.any(self.effort_limits <= 0) or np.any(self.effort_limits > np.array(self.contract.effort_limits)):
            raise ValueError("实际力矩能力必须处于名义能力以内")
        if np.any(self.kp < 0) or np.any(self.kd < 0) or np.any(self.kp[4:] != 0):
            raise ValueError("实际执行器增益无效")
        friction = float(self.parameters["friction"])
        if not np.isfinite(friction) or friction <= 0:
            raise ValueError("实际摩擦系数无效")
        self.model.pair_friction[:, :2] = friction
        ids = np.flatnonzero(self.model.geom_contype == 2)
        self.model.geom_friction[ids, 0] = friction
        actuator_ids = self.model_map.actuator_ids
        self.model.actuator_ctrlrange[actuator_ids, 0] = -self.effort_limits
        self.model.actuator_ctrlrange[actuator_ids, 1] = self.effort_limits
        if not np.all(self.model.pair_friction[:, :2] == friction):
            raise RuntimeError("实际接触系数读回失败")
        self.parameter_sha256 = payload_sha256(self.parameters)

    def state(self):
        return collect_kinematic_state(self.model, self.data, self.model_map, self.walker_contract)

    def _frame(self) -> np.ndarray:
        state = self.state()
        result = build_actor_frame(_tensor(state.angular_velocity_control), _tensor(state.projected_gravity_control),
                                   _tensor(state.joint_position_control), _tensor(state.joint_velocity_control),
                                   _tensor(self.previous_action), _tensor(self.q_reference))
        return result.numpy().copy()

    def _physical_frame(self) -> np.ndarray:
        state = self.state()
        return np.concatenate((state.angular_velocity_control, state.projected_gravity_control,
                               state.joint_position_control[:4] - self.q_reference,
                               state.joint_velocity_control, self.previous_action)).astype(np.float32)

    def _walker_frame(self, command: np.ndarray) -> np.ndarray:
        frame = build_actor_observation(self.state(), command, self.previous_action, self.walker_contract)
        frame[9:13] = np.clip(self.state().joint_position_control[:4] - self.q_reference, -1.5, 1.5)
        frame[19:25] = self.previous_action
        return frame

    def reset_nominal(self, command: np.ndarray | None = None) -> None:
        command = np.array([0.0, 0.0, 0.20]) if command is None else np.asarray(command, dtype=np.float64)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model_map.reset_keyframe_id)
        self.data.ctrl[:] = 0
        self.previous_action.fill(0)
        self.prior_actions.clear()
        self.leg_target = self.q_reference.copy()
        self.wheel_target.fill(0)
        self.last_torque.fill(0)
        self.frame_id = 0
        mujoco.mj_forward(self.model, self.data)
        self.history[:] = self._frame()
        self.clean_history[:] = self._physical_frame()
        self.walker_history[:] = self._walker_frame(command)
        self._ready = True

    def recovery_observation(self) -> np.ndarray:
        if not self._ready:
            raise RuntimeError("必须先恢复真实初态")
        return self.history.reshape(-1).copy()

    def walker_candidate_observation(self, command: np.ndarray | None = None) -> np.ndarray:
        if not self._ready:
            raise RuntimeError("必须先恢复真实初态")
        command = np.array([0.0, 0.0, 0.20]) if command is None else np.asarray(command, dtype=np.float64)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("行走命令必须为三维有限向量")
        history = self.walker_history.copy()
        history[-1, 6:9] = _normalize_command(command, self.walker_contract)
        return history.reshape(-1)

    @torch.inference_mode()
    def walker_action(self, command: np.ndarray | None = None) -> np.ndarray:
        observation = self.walker_candidate_observation(command)
        result = self.walker(torch.from_numpy(observation).unsqueeze(0)).squeeze(0).numpy().copy()
        if result.shape != (6,) or not np.isfinite(result).all():
            raise FloatingPointError("冻结行走候选输出无效")
        return result

    def decoded(self, action: np.ndarray, *, locomotion: bool = False) -> tuple[np.ndarray, np.ndarray]:
        decoder = decode_walker_action if locomotion else decode_action
        leg, wheel = decoder(_tensor(action), _tensor(self.q_reference), self.contract)
        return leg.numpy(), wheel.numpy()

    def estimated_torque(self, action: np.ndarray, *, locomotion: bool = False) -> np.ndarray:
        leg, wheel = self.decoded(action, locomotion=locomotion)
        state = self.state()
        torque = estimate_pd_torque(_tensor(leg), _tensor(wheel), _tensor(state.joint_position_control),
                                    _tensor(state.joint_velocity_control), _tensor(self.kp), _tensor(self.kd),
                                    _tensor(self.effort_limits))
        return torque.numpy()

    def contact_state(self) -> dict:
        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        body_forces = {}
        contacts = []
        maximum_penetration = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if floor not in (contact.geom1, contact.geom2):
                raise RuntimeError("完整接触模型出现机器人内部实际接触")
            geom = int(contact.geom2 if contact.geom1 == floor else contact.geom1)
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            normal = max(0.0, float(force[0]))
            body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[geom]))
            body_forces[body] = body_forces.get(body, 0.0) + normal
            maximum_penetration = max(maximum_penetration, -float(contact.dist))
            contacts.append({"body": body, "normal_force_n": normal, "distance_m": float(contact.dist),
                             "geom": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)})
        return {"body_normal_force_n": body_forces, "contacts": contacts, "maximum_penetration_m": maximum_penetration}

    def step(self, action: np.ndarray, command: np.ndarray | None = None, *, locomotion: bool = False) -> RuntimeStep:
        if not self._ready:
            raise RuntimeError("必须先恢复真实初态")
        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise FloatingPointError("实际动作必须为六维有限向量")
        command = np.array([0.0, 0.0, 0.20]) if command is None else np.asarray(command, dtype=np.float64)
        self.leg_target, self.wheel_target = self.decoded(raw, locomotion=locomotion)
        traces = []
        for _ in range(self.contract.mujoco_decimation):
            torque = self.estimated_torque(raw, locomotion=locomotion)
            self.data.ctrl[:] = 0
            self.data.ctrl[self.model_map.actuator_ids] = torque * CONTROLLED_JOINT_SIGNS
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise FloatingPointError("MuJoCo 子步物理状态非有限")
            # mj_step 的派生量对应积分前状态，重新计算用于同一结束状态审计。
            mujoco.mj_forward(self.model, self.data)
            clearance = self.geometry_audit.inspect(self.data)
            traces.append({"time_s": float(self.data.time), "torque": torque.tolist(),
                           "joint_position": self.model_map.joint_position_control(self.data).tolist(),
                           "joint_velocity": self.model_map.joint_velocity_control(self.data).tolist(),
                           "loop_error_m": _max_loop_error(self.model, self.data),
                           "contact": self.contact_state(), "clearance": clearance})
        self.last_torque = torque.copy()
        self.prior_actions.append(raw.copy())
        self.prior_actions = self.prior_actions[-2:]
        self.previous_action = raw.copy()
        self.frame_id += 1
        self.history[:-1] = self.history[1:]
        self.history[-1] = self._frame()
        self.clean_history[:-1] = self.clean_history[1:]
        self.clean_history[-1] = self._physical_frame()
        self.walker_history[:-1] = self.walker_history[1:]
        self.walker_history[-1] = self._walker_frame(command)
        return RuntimeStep(self.frame_id, raw.copy(), self.leg_target.copy(), self.wheel_target.copy(), torque.copy(),
                           tuple(traces), self.contact_state(), _max_loop_error(self.model, self.data), clearance)

    def snapshot(self) -> dict:
        if not self._ready:
            raise RuntimeError("必须先恢复真实初态")
        root_joint = self.model_map.free_joint_id
        qadr = int(self.model.jnt_qposadr[root_joint])
        vadr = int(self.model.jnt_dofadr[root_joint])
        rotation = self.data.xmat[self.model_map.base_body_id].reshape(3, 3)
        names = []
        positions = []
        velocities = []
        for joint in range(self.model.njnt):
            if self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint))
            positions.append(float(self.data.qpos[self.model.jnt_qposadr[joint]]))
            velocities.append(float(self.data.qvel[self.model.jnt_dofadr[joint]]))
        return {"joint_names": names, "joint_positions": positions, "joint_velocities": velocities,
                "root_position": self.data.qpos[qadr:qadr + 3].tolist(),
                "root_quaternion_wxyz": self.data.qpos[qadr + 3:qadr + 7].tolist(),
                "root_linear_velocity_world": self.data.qvel[vadr:vadr + 3].tolist(),
                "root_angular_velocity_world": (rotation @ self.data.qvel[vadr + 3:vadr + 6]).tolist(),
                "previous_action": self.previous_action.tolist(), "prior_actions": [a.tolist() for a in self.prior_actions],
                "actor_history": self.history.tolist(), "walker_history": self.walker_history.tolist(),
                "clean_history": self.clean_history.tolist(),
                "leg_target": self.leg_target.tolist(), "wheel_target": self.wheel_target.tolist(),
                "contacts": self.contact_state(), "loop_error_m": _max_loop_error(self.model, self.data),
                "clearance": self.geometry_audit.inspect(self.data), "parameter_sha256": self.parameter_sha256,
                "contact_model_sha256": file_sha256(self.contact_manifest_path), "time_s": float(self.data.time),
                "joint_coordinate_convention": "USD_native_by_name", "root_frame": "USD_world_wxyz"}

    def restore_snapshot(self, record: dict) -> None:
        if record["parameter_sha256"] != self.parameter_sha256 or record["contact_model_sha256"] != file_sha256(self.contact_manifest_path):
            raise ValueError("姿态记录物理参数或接触模型身份不匹配")
        if record["joint_coordinate_convention"] != "USD_native_by_name" or record["root_frame"] != "USD_world_wxyz":
            raise ValueError("姿态坐标约定不支持")
        mujoco.mj_resetData(self.model, self.data)
        root = self.model_map.free_joint_id
        qadr, vadr = int(self.model.jnt_qposadr[root]), int(self.model.jnt_dofadr[root])
        self.data.qpos[qadr:qadr + 3] = record["root_position"]
        quaternion = np.asarray(record["root_quaternion_wxyz"], dtype=np.float64)
        if quaternion.shape != (4,) or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-6):
            raise ValueError("根四元数无效")
        self.data.qpos[qadr + 3:qadr + 7] = quaternion
        expected = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    for joint in range(self.model.njnt) if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE}
        if len(record["joint_names"]) != len(expected) or set(record["joint_names"]) != expected:
            raise ValueError("姿态记录必须包含全部主动和被动关节")
        for name, position, velocity in zip(record["joint_names"], record["joint_positions"], record["joint_velocities"], strict=True):
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[joint]] = position
            self.data.qvel[self.model.jnt_dofadr[joint]] = velocity
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.xmat[self.model_map.base_body_id].reshape(3, 3)
        self.data.qvel[vadr:vadr + 3] = record["root_linear_velocity_world"]
        self.data.qvel[vadr + 3:vadr + 6] = rotation.T @ record["root_angular_velocity_world"]
        self.previous_action = np.asarray(record["previous_action"], dtype=np.float64)
        self.prior_actions = [np.asarray(a, dtype=np.float64) for a in record["prior_actions"]]
        self.history[:] = record["actor_history"]
        self.clean_history[:] = record["clean_history"]
        self.walker_history[:] = record["walker_history"]
        self.leg_target = np.asarray(record["leg_target"], dtype=np.float64)
        self.wheel_target = np.asarray(record["wheel_target"], dtype=np.float64)
        self.frame_id = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._ready = True
        if not np.allclose(self.history[-1], self._frame(), atol=2e-6, rtol=0):
            raise ValueError("姿态记录的当前观测与真实恢复状态不一致")

    def pose_record(self, *, record_id: str, trajectory_id: str, split: str, category: str,
                    motion_kind: str, metadata: dict):
        from .pose_bank import PoseRecord

        snapshot = self.snapshot()
        if len(self.prior_actions) < 2:
            raise ValueError("姿态采集前需要至少两个真实控制步")
        body = self.model_map.base_body_id
        root_state = np.concatenate((self.data.xpos[body], self.data.xquat[body],
                                     snapshot["root_linear_velocity_world"], snapshot["root_angular_velocity_world"]))
        details = dict(metadata)
        details.update({"mujoco_snapshot": snapshot, "conversion_validation": "required",
                        "history_source": "physical_trajectory",
                        "tilt_rad": float(np.arccos(np.clip(-self.state().projected_gravity_control[2], -1, 1))),
                        "com_height_m": self.state().base_height_m, "source_walker": self.walker_identity,
                        "actual_parameters": self.parameters, "allowed_body_pairs": self.allowed_body_pairs,
                        "loop_error_m": snapshot["loop_error_m"], "clearance": snapshot["clearance"],
                        "contacts": snapshot["contacts"]})
        return PoseRecord(record_id=record_id, trajectory_id=trajectory_id, split=split,
                          initial_category=category, motion_kind=motion_kind, source_simulator="mujoco",
                          joint_names=tuple(snapshot["joint_names"]), joint_position=tuple(snapshot["joint_positions"]),
                          joint_velocity=tuple(snapshot["joint_velocities"]), root_state_usd=tuple(root_state),
                          clean_history=tuple(tuple(float(v) for v in row) for row in self.clean_history),
                          prior_actions=tuple(tuple(float(v) for v in row) for row in self.prior_actions),
                          motor_targets=tuple(float(v) for v in np.concatenate((self.leg_target, self.wheel_target))),
                          parameter_plan_sha256=self.parameter_sha256,
                          asset_bundle_sha256=self.walker_manifest["asset_bundle_hash"],
                          contact_model_sha256=file_sha256(self.contact_manifest_path), metadata=details)

    def restore_pose(self, record) -> None:
        if record.source_simulator != "mujoco" or "mujoco_snapshot" not in record.metadata:
            raise ValueError("MuJoCo需要已验收的具名物理转换及原生根状态")
        if record.parameter_plan_sha256 != self.parameter_sha256:
            raise ValueError("姿态库实际参数计划不匹配")
        self.restore_snapshot(record.metadata["mujoco_snapshot"])
        if not np.allclose(self.clean_history, record.clean_history, atol=1e-7, rtol=0):
            raise ValueError("姿态库真实历史与原生恢复记录不一致")


def supervisor_measurements(runtime: MujocoRecoveryRuntime, result: RuntimeStep, *, support_force_threshold_n: float,
                            maximum_contact_penetration_m: float, minimum_clearance_m: float,
                            action_compatible: bool, compatibility_blockers: tuple[str, ...] = ()):
    from .supervisor import SupervisorMeasurements

    if support_force_threshold_n <= 0 or maximum_contact_penetration_m < 0:
        raise ValueError("支撑和接触门限尚未正确配置")
    state = runtime.state()
    forces = result.contacts["body_normal_force_n"]
    failure = None
    for substep in result.substeps:
        if substep["loop_error_m"] >= runtime.contract.max_loop_error_m:
            failure = "closed_chain_error"
            break
        if substep["contact"]["maximum_penetration_m"] > maximum_contact_penetration_m:
            failure = "excessive_ground_penetration"
            break
        if substep["clearance"]["minimum_nonallowed_clearance_m"] < minimum_clearance_m:
            failure = "nonallowed_body_clearance"
            break
    return SupervisorMeasurements(frame_id=result.frame_id,
        tilt_rad=float(np.arccos(np.clip(-state.projected_gravity_control[2], -1, 1))),
        angular_velocity=tuple(float(value) for value in state.angular_velocity_control),
        leg_position_error=tuple(float(value) for value in state.joint_position_control[:4] - runtime.contract.q_nominal),
        leg_velocity=tuple(float(value) for value in state.joint_velocity_control[:4]),
        wheel_velocity=tuple(float(value) for value in state.joint_velocity_control[4:]),
        com_height_m=state.base_height_m,
        left_wheel_support=forces.get("jwheel_left", 0) >= support_force_threshold_n,
        right_wheel_support=forces.get("jwheel_right", 0) >= support_force_threshold_n,
        body_or_leg_support=any(value >= support_force_threshold_n for name, value in forces.items()
                                if name not in ("jwheel_left", "jwheel_right")),
        action_compatible=action_compatible, compatibility_blockers=compatibility_blockers,
        physical_failure_reason=failure, finite=True)
