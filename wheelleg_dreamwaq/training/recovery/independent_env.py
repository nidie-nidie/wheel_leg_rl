from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from isaaclab.envs import DirectRLEnv

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.kinematics.virtual_leg import load_usd_anchor_audit
from wheelleg_dreamwaq.schemas.action import CANONICAL_JOINT_ORDER, wheel_targets_control_to_usd
from wheelleg_dreamwaq.schemas.frames import quat_rotate_inverse_wxyz, transform_usd_vector_to_control

from .contact_model import assert_isaac_full_contact
from .env import RecoveryEnv
from .geometry_gpu import GeometryGpuGuard
from .independent_poses import IndependentPoseBank, IndependentPoseTensorPool
from .independent_rewards import compute_independent_reward
from .independent_state import StandingGateConfig, StandingTask, REASON_LABELS
from .observations import HistoryBuffer, build_actor_frame
from .performance import RecoveryPerformance
from .randomization import RecoveryRandomization, RecoveryRandomizationV1
from .standing_reference import StandingReference


class IndependentStandingEnv(RecoveryEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.resolved_config = cfg.resolved_config
        if self.resolved_config["schema_version"] != "RecoveryIndependentStandV1":
            raise ValueError("独立站立环境需要新配置契约")
        self.contract = SimpleNamespace(**self.resolved_config["contract"])
        self.asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
        self.pose_bank = IndependentPoseBank.load(self.resolved_config["environment"]["pose_bank_path"])
        self.pose_bank.validate_split_ids(self.resolved_config["initialization"])
        self._geometry_guard = None
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)
        self._state_cache_key = self._state_cache = None
        self._write_generation = 0
        self._performance = RecoveryPerformance(self.device)
        environment = self.resolved_config["environment"]
        margin = float(environment.get("ground_activity_margin_m", .5))
        half = torch.tensor(environment["ground_size_xy_m"], device=self.device) * .5 - margin
        if (self.scene.env_origins[:, :2].abs() >= half).any():
            raise ValueError("真实环境 origin 超出完整地面范围")
        self._controlled_ids, names = self.robot.find_joints(list(CANONICAL_JOINT_ORDER), preserve_order=True)
        if tuple(names) != CANONICAL_JOINT_ORDER:
            raise ValueError("执行器具名顺序不同")
        self._leg_ids, self._wheel_ids = self._controlled_ids[:4], self._controlled_ids[4:]
        profile = RecoveryRandomizationV1(**self.resolved_config["randomization"])
        if profile.enabled:
            raise ValueError("独立站立实验使用名义物理参数")
        self._randomization = RecoveryRandomization(self.num_envs, int(cfg.seed), self.device,
                                                    profile, self.contract.q_nominal)
        self._q_reference = self._randomization.q_reference
        self._apply_parameters()
        self._pose_pool = IndependentPoseTensorPool(self.pose_bank, self.robot.joint_names,
                                                    self.asset_report.bundle_hash, self.device)
        reference = self.resolved_config["reference"]
        self.reference = StandingReference.load(reference["manifest_path"], reference["bank_path"],
            reference["bank_sha256"], device=self.device,
            expected_manifest_sha256=reference["manifest_sha256"])
        self.contact_manifest = assert_isaac_full_contact(self.sim.stage, "/World/envs/env_0/Robot")
        masses = self.robot.root_physx_view.get_masses().to(self.device)
        self.weight_n = masses.sum(dim=1) * 9.81
        if not torch.allclose(self.weight_n, self.weight_n.new_full(self.weight_n.shape,
                              self.reference.identity["weight_n"]), atol=.005, rtol=0.):
            raise ValueError("真实机器人重量与401参考不一致")
        self.task = StandingTask(self.num_envs, self.device, self.reference, self.weight_n, mode="train",
                                config=StandingGateConfig(**self.resolved_config["standing"]))
        self._history = HistoryBuffer(self.num_envs, device=self.device)
        self._action = torch.zeros(self.num_envs, 6, device=self.device)
        self._previous_action = self._action.clone()
        self._previous_previous_action = self._action.clone()
        self._leg_targets = torch.zeros(self.num_envs, 4, device=self.device)
        self._wheel_targets = torch.zeros(self.num_envs, 2, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._substep_loop_failure = self._success.clone()
        self._substep_joint_limit_failure = self._success.clone()
        self._substep_geometry_reasons = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_generations = torch.zeros_like(self._substep_geometry_reasons)
        self._frame_ids = torch.zeros_like(self._substep_geometry_reasons)
        self._history_step = 0
        self._episode_return = torch.zeros(self.num_envs, device=self.device)
        self._episode_sources = [""] * self.num_envs
        self._reset_records = [None] * self.num_envs
        self._physical_counts = Counter()
        self._reward_terms, self._last_result = {}, None
        self._observation_cache = None
        self._initialize_physical_audit()
        geometry = self.resolved_config["geometry_guard"]
        if geometry is None:
            raise ValueError("独立站立需要逐物理子步真实几何检查")
        self._geometry_guard = GeometryGpuGuard(geometry, environment["run_dir"],
                                               self.device, self.robot.body_names)

    def _initialize_physical_audit(self):
        audit = load_usd_anchor_audit()
        self._loop_body_ids, anchors = [], []
        for loop in audit["loop_joints"]:
            for side in loop["sides"]:
                indices, names = self.robot.find_bodies([side["body_name"]], preserve_order=True)
                if names != [side["body_name"]]:
                    raise ValueError("闭链锚点缺少真实刚体")
                self._loop_body_ids.append(indices[0])
                anchors.append(side["local_anchor_m"])
        self._loop_anchors = torch.tensor(anchors, dtype=torch.float32, device=self.device)
        wheels = [audit["legs"][side]["wheel"]["body_name"] for side in ("left", "right")]
        self._wheel_contact_ids, names = self.contacts.find_bodies(wheels, preserve_order=True)
        if names != wheels:
            raise ValueError("双轮真实接触刚体缺失")
        self._nonwheel_contact_ids = [i for i in range(len(self.contacts.body_names)) if i not in self._wheel_contact_ids]

    def check_workers(self):
        if self._geometry_guard is not None:
            self._geometry_guard.check_health()

    def rollout_barrier(self):
        self.check_workers()

    def _pre_physics_step(self, actions):
        self.check_workers()
        if actions.shape != self._action.shape or not torch.isfinite(actions).all() or (actions.abs() > 1.).any():
            raise ValueError("独立站立执行动作必须为有限六维tanh输出")
        self._previous_previous_action.copy_(self._previous_action)
        self._previous_action.copy_(self._action)
        self._action.copy_(actions)
        self._leg_targets.copy_(actions[:, :4])
        self._wheel_targets.copy_(30. * actions[:, 4:])
        self._substep_loop_failure.zero_()
        self._substep_joint_limit_failure.zero_()
        self._substep_geometry_reasons.zero_()
        self._observation_cache = None
        self.extras["episodes"] = []

    def _apply_action(self):
        self._performance.end_physics_interval()
        self._substep_loop_failure |= self._loop_failures()
        self._substep_joint_limit_failure |= self._joint_limit_failures()
        self._inspect_geometry(before_physics=True)
        self.robot.set_joint_position_target(self._leg_targets, joint_ids=self._leg_ids)
        self.robot.set_joint_velocity_target(wheel_targets_control_to_usd(self._wheel_targets), joint_ids=self._wheel_ids)
        self._performance.begin_physics_interval()

    def _frame(self, state):
        return build_actor_frame(state["angular_velocity"], state["projected_gravity"], state["joint_position"],
                                 state["joint_velocity"], self._action, self._q_reference)

    def _critic(self, state):
        task = self.task
        displacement_world = torch.cat((task.position_delta, torch.zeros(self.num_envs, 1, device=self.device)), dim=1)
        local = transform_usd_vector_to_control(quat_rotate_inverse_wxyz(self.robot.data.root_link_quat_w,
                                                                        displacement_world))[:, :2]
        effort = state["torque"].new_tensor(self.contract.effort_limits)
        return torch.cat((self._frame(state), (state["linear_velocity"] * 2.).clamp(-6., 6.),
                          ((state["height"] - .20) / .04).clamp(-5., 5.)[:, None],
                          (state["torque"] / effort).clamp(-5., 5.), task.qualification_progress[:, None],
                          task.remaining[:, None], task.hold_progress[:, None], task.phase.float()[:, None],
                          task.contact_window.wheel_mean / self.weight_n[:, None],
                          (task.contact_window.nonwheel_mean / self.weight_n)[:, None],
                          (local / .20).clamp(-5., 5.)), dim=1)

    def _get_dones(self):
        self._performance.end_physics_interval()
        self._inspect_geometry()
        state = self._read_state()
        self._history_step += 1
        self._history.append(self._frame(state), self._history_step)
        self._frame_ids += 1
        loop = self._substep_loop_failure | self._loop_failures()
        joints = self._substep_joint_limit_failure | self._joint_limit_failures()
        reason = self._substep_geometry_reasons.clone()
        reason = torch.where(joints, 3, reason)
        reason = torch.where(loop, 2, reason)
        physical = reason != 0
        forces = self.contacts.data.net_forces_w[..., 2].clamp_min(0.)
        wheel, nonwheel = forces[:, self._wheel_contact_ids], forces[:, self._nonwheel_contact_ids].sum(dim=1)
        result = self.task.step(q=state["joint_position"][:, :4], dq=state["joint_velocity"],
            omega=state["angular_velocity"], gravity=state["projected_gravity"], linear=state["linear_velocity"],
            height=state["height"], wheel_forces=wheel, nonwheel_force=nonwheel,
            world_xy=self.robot.data.root_com_pos_w[:, :2] - self.scene.env_origins[:, :2],
            physical_failure=physical, physical_reason=reason)
        self._last_result = result
        self._reward, self._reward_terms = compute_independent_reward(q=state["joint_position"][:, :4],
            dq=state["joint_velocity"], omega=state["angular_velocity"], gravity=state["projected_gravity"],
            linear=state["linear_velocity"], height=state["height"], torque=state["torque"],
            actions=self._action, previous_actions=self._previous_action,
            previous_previous_actions=self._previous_previous_action,
            wheel_forces_mean=self.task.contact_window.wheel_mean,
            nonwheel_force_mean=self.task.contact_window.nonwheel_mean, reference=self.reference,
            weight_n=self.weight_n, success=result.success, physical_failure=physical, timeout=result.timeout)
        self._episode_return += self._reward
        done = result.success | result.failure
        episodes = []
        for env_id in done.nonzero(as_tuple=False).flatten().cpu().tolist():
            physical_name = {2: "loop_closure", 3: "joint_limit", 4: "nonallowed_body_clearance",
                             5: "excessive_ground_penetration"}.get(int(reason[env_id]), "none")
            if physical[env_id]:
                self._physical_counts[physical_name] += 1
            episodes.append({"env_id": env_id, "reward": float(self._episode_return[env_id]),
                "length": int(self.task.elapsed_steps[env_id]), "source": self._episode_sources[env_id],
                "initial_state_id": self._reset_records[env_id]["record_id"],
                "success": bool(result.success[env_id]), "reason": REASON_LABELS[int(result.reason_codes[env_id])],
                "physical_reason": physical_name,
                "first_qualified_step": int(self.task.first_qualified_step[env_id]),
                "qualification_time_s": None if self.task.first_qualified_step[env_id] < 0
                    else float(self.task.first_qualified_step[env_id]) * .02,
                "hold_failed": bool(self.task.hold_interruptions[env_id] > 0),
                "hold_steps": int(self.task.hold_count[env_id]),
                "hold_interruptions": int(self.task.hold_interruptions[env_id])})
        self.extras["episodes"] = episodes
        self.extras["reward_components"] = self._reward_terms
        effort = state["torque"].new_tensor(self.contract.effort_limits)
        self.extras["effort_saturation_fraction"] = (state["torque"].abs() >= .99 * effort).float().mean(dim=1)
        self.extras["terminal_observations"] = {"env_ids": done.nonzero(as_tuple=False).flatten(),
            "policy": self._history.flattened()[done], "critic": self._critic(state)[done]}
        return done, torch.zeros_like(done)

    def _get_rewards(self):
        return self._reward

    def _get_observations(self):
        state = self._read_state()
        result = {"policy": self._history.flattened(), "critic": self._critic(state)}
        if result["policy"].shape != (self.num_envs, 110) or result["critic"].shape != (self.num_envs, 41):
            raise ValueError("独立站立真实观测维度无效")
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("独立站立真实观测包含非有限数值")
        return result

    def get_observations(self):
        return self._get_observations()

    def _invalidate_written_state(self):
        self.robot.data._root_com_pose_w.timestamp = -1.0
        self.robot.data._root_link_vel_w.timestamp = -1.0
        self._invalidate_state()

    def _reset_idx(self, env_ids):
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        DirectRLEnv._reset_idx(self, ids)
        self.robot.reset(ids)
        self.contacts.reset(ids)
        records = self.pose_bank.sample(ids.cpu().tolist(), self._randomization, self.resolved_config["initialization"])
        indices = self._pose_pool.indices(records)
        root = self._pose_pool.root_state[indices].clone()
        root[:, :3] += self.scene.env_origins[ids]
        self.robot.write_root_link_state_to_sim(root, env_ids=ids)
        self.robot.write_joint_state_to_sim(self._pose_pool.joint_position[indices],
                                             self._pose_pool.joint_velocity[indices], env_ids=ids)
        action = self._pose_pool.actions[indices]
        self._action[ids] = self._previous_action[ids] = self._previous_previous_action[ids] = action
        self._leg_targets[ids], self._wheel_targets[ids] = action[:, :4], 0.
        self.robot.set_joint_position_target(self._leg_targets[ids], joint_ids=self._leg_ids, env_ids=ids)
        self.robot.set_joint_velocity_target(self._wheel_targets[ids], joint_ids=self._wheel_ids, env_ids=ids)
        self.task.reset(ids)
        self._episode_return[ids] = 0.
        self._substep_loop_failure[ids] = self._substep_joint_limit_failure[ids] = False
        self._substep_geometry_reasons[ids] = 0
        self._reset_generations[ids] += 1
        self._frame_ids[ids] = 0
        self._invalidate_written_state()
        state = self._read_state()
        frames = self._frame(state)[ids, None].expand(-1, 5, -1).clone()
        self._history.reset(ids, frames)
        for env_id, record in zip(ids.cpu().tolist(), records, strict=True):
            self._reset_records[env_id], self._episode_sources[env_id] = record, record["source"]
            expected = record["metadata"]
            actual_q = state["joint_position"][env_id, :4]
            if not torch.allclose(actual_q, actual_q.new_tensor(expected["q_control_rad"]), atol=1e-4, rtol=0.):
                raise ValueError("跨模拟器初态实际控制关节角度不一致")
            if abs(float(state["height"][env_id]) - expected["com_height_m"]) > 1e-4:
                raise ValueError("跨模拟器初态真实COM高度不一致")
            if not torch.allclose(state["projected_gravity"][env_id], actual_q.new_tensor(expected["projected_gravity"]), atol=1e-4, rtol=0.):
                raise ValueError("跨模拟器初态重力方向不一致")
            snapshot = expected["mujoco_snapshot"]
            expected_twist = actual_q.new_tensor(snapshot["root_linear_velocity_world"] +
                                                snapshot["root_angular_velocity_world"])
            if not torch.allclose(self.robot.data.root_link_vel_w[env_id], expected_twist, atol=1e-4, rtol=0.):
                raise ValueError("跨模拟器初态真实root link世界线速度或角速度不一致")
        if self._loop_failures()[ids].any() or self._joint_limit_failures()[ids].any():
            raise ValueError("独立站立初态闭链或真实关节范围检查失败")
        self._inspect_geometry(ids, restoring=True)

    def physical_failure_counts(self):
        return dict(self._physical_counts)

    def environment_state(self):
        tensor_names = ("_action", "_previous_action", "_previous_previous_action", "_leg_targets", "_wheel_targets",
                        "_frame_ids", "_reset_generations", "_episode_return")
        task_tensors = {name: value.detach().clone() for name, value in vars(self.task).items() if isinstance(value, torch.Tensor)}
        window = self.task.contact_window
        return {"schema_version": "IndependentStandingEnvironmentStateV1", "pose_bank_sha256": self.pose_bank.sha256,
            "reference_identity": self.reference.identity, "num_envs": self.num_envs,
            "physical_restart_semantics": "root_and_all_joint_states_without_solver_impulse_cache",
            "last_measured_state": {name: value.detach().clone() for name, value in self._read_state().items()},
            "root_state": self.robot.data.root_state_w.detach().clone() - torch.cat((self.scene.env_origins,
                 torch.zeros(self.num_envs, 10, device=self.device)), dim=1),
            "joint_position": self.robot.data.joint_pos.detach().clone(),
            "joint_velocity": self.robot.data.joint_vel.detach().clone(),
            "tensors": {name: getattr(self, name).detach().clone() for name in tensor_names},
            "task_tensors": task_tensors,
            "contact_window": {name: getattr(window, name).detach().clone() for name in ("buffer", "sum", "count", "cursor")},
            "history": self._history.frames(), "history_step": self._history_step,
            "reset_record_ids": [r["record_id"] for r in self._reset_records],
            "episode_sources": list(self._episode_sources), "randomization": self._randomization.state_dict(),
            "physical_failure_counts": self.physical_failure_counts(), "episode_length_buf": self.episode_length_buf.clone()}

    def environment_state_dict(self):
        return self.environment_state()

    def restore_environment_state(self, state):
        if (state["schema_version"] != "IndependentStandingEnvironmentStateV1" or state["num_envs"] != self.num_envs
                or state["pose_bank_sha256"] != self.pose_bank.sha256 or state["reference_identity"] != self.reference.identity):
            raise ValueError("独立站立物理恢复身份不一致")
        ids = self.robot._ALL_INDICES
        root = state["root_state"].to(self.device).clone()
        root[:, :3] += self.scene.env_origins
        self.robot.write_root_state_to_sim(root)
        self.robot.write_joint_state_to_sim(state["joint_position"].to(self.device), state["joint_velocity"].to(self.device))
        for name, value in state["tensors"].items():
            getattr(self, name).copy_(value.to(self.device))
        for name, value in state["task_tensors"].items():
            getattr(self.task, name).copy_(value.to(self.device))
        for name, value in state["contact_window"].items():
            getattr(self.task.contact_window, name).copy_(value.to(self.device))
        self._history.reset(ids, state["history"].to(self.device))
        self._history_step = int(state["history_step"])
        self._history._last_frame_id = self._history_step
        self._reset_records = [self.pose_bank.by_id[name] for name in state["reset_record_ids"]]
        self._episode_sources = list(state["episode_sources"])
        self._randomization.load_state_dict(state["randomization"])
        self._physical_counts = Counter(state["physical_failure_counts"])
        self.episode_length_buf.copy_(state["episode_length_buf"].to(self.device))
        self.contacts.reset()
        self.robot.set_joint_position_target(self._leg_targets, joint_ids=self._leg_ids)
        self.robot.set_joint_velocity_target(wheel_targets_control_to_usd(self._wheel_targets), joint_ids=self._wheel_ids)
        self._invalidate_written_state()
        self._inspect_geometry(restoring=True)
        measured = {name: value.to(self.device).clone() for name, value in state["last_measured_state"].items()}
        current = self._read_state()
        for name in measured:
            if measured[name].shape != current[name].shape or not torch.isfinite(measured[name]).all():
                raise ValueError("恢复时保存的最后真实测量无效")
            if name != "torque" and not torch.allclose(measured[name], current[name], atol=1e-5, rtol=0.):
                raise ValueError(f"恢复真实物理状态与保存测量不一致: {name}")
        # 隐式PD最后子步估计来自保存时真实测量；下一个物理步骤重新读取执行器数据。
        self._state_cache = measured
        self._state_cache_key = (self._sim_step_counter, self._write_generation)

    def close(self):
        if hasattr(self, "_performance"):
            self._performance.end_physics_interval()
        if self._geometry_guard is not None:
            self._geometry_guard.close()
            self._geometry_guard = None
        DirectRLEnv.close(self)
