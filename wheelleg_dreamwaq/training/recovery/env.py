from __future__ import annotations

from collections import deque
from contextlib import ExitStack
import json
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.kinematics.virtual_leg import load_usd_anchor_audit
from wheelleg_dreamwaq.schemas.action import CANONICAL_JOINT_ORDER, controlled_joint_feedback_usd_to_control, wheel_targets_control_to_usd
from wheelleg_dreamwaq.schemas.frames import projected_gravity_from_quaternion, quat_rotate_wxyz, transform_usd_vector_to_control

from .contact_model import assert_isaac_full_contact, configure_isaac_full_contact
from .contracts import RecoveryContractV2
from .contracts import CATEGORIES, file_sha256, verify_walker_export
from .control import check_action_compatibility, decode_action
from .env_cfg import RecoveryEnvCfg
from .geometry_gpu import GeometryGpuGuard
from .batched_supervisor import BatchedRecoverySupervisor, BLOCKER_BITS, REASON_CODES
from .observations import HistoryBuffer, build_actor_frame, build_critic_frame, normalize_physical_frame
from .metrics import AttemptContext
from .metrics_stream import RecoveryMetricsStream
from .performance import RecoveryPerformance, profile_phase
from .pose_bank import PoseBank
from .pose_pool import PoseTensorPool
from .randomization import RecoveryRandomization, RecoveryRandomizationV1
from .rewards import compute_reward
from .supervisor import SupervisorConfig
from .validation import all_finite


class RecoveryEnv(DirectRLEnv):
    cfg: RecoveryEnvCfg

    def __init__(self, cfg: RecoveryEnvCfg, render_mode=None, **kwargs):
        self.resolved_config = cfg.resolved_config
        self.contract = RecoveryContractV2(**self.resolved_config["contract"])
        self.asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
        self.pose_bank = PoseBank.load(self.resolved_config["environment"]["pose_bank_path"])
        self.pose_bank.validate_coverage(formal=self.resolved_config["environment"]["task_type"] == "formal_training")
        geometry_config = self.resolved_config["readiness"]["geometry_guard"]
        if self.resolved_config["environment"]["task_type"] != "interface_validation" and geometry_config is None:
            raise ValueError("研究初训和正式训练必须配置逐物理子步的同源几何检查")
        self._geometry_guard = None
        super().__init__(cfg, render_mode, **kwargs)
        self._state_cache_key = None
        self._state_cache = None
        self._write_generation = 0
        self._performance = RecoveryPerformance(self.device)
        environment = self.resolved_config["environment"]
        usable_half_size = torch.tensor(environment["ground_size_xy_m"], device=self.device) * 0.5 - environment["ground_activity_margin_m"]
        if (self.scene.env_origins[:, :2].abs() >= usable_half_size).any():
            raise ValueError("真实环境 origin 超出地面及规定活动余量")
        self.ground_coverage = {"size_xy_m": list(environment["ground_size_xy_m"]),
                               "activity_margin_m": environment["ground_activity_margin_m"],
                               "maximum_absolute_origin_xy_m": self.scene.env_origins[:, :2].abs().amax(dim=0).tolist(),
                               "all_environment_origins_covered": True}
        self._controlled_ids, names = self.robot.find_joints(list(CANONICAL_JOINT_ORDER), preserve_order=True)
        if tuple(names) != CANONICAL_JOINT_ORDER:
            raise RuntimeError("Isaac 电机关节顺序与 canonical 契约不同")
        self._leg_ids, self._wheel_ids = self._controlled_ids[:4], self._controlled_ids[4:]
        self._randomization = RecoveryRandomization(
            self.num_envs, int(cfg.seed), self.device,
            RecoveryRandomizationV1(**self.resolved_config["randomization"]), self.contract.q_nominal,
        )
        self._q_reference = self._randomization.q_reference
        self._pose_plan_hashes = [plan.sha256 for plan in self._randomization.plans]
        self._pose_pool = PoseTensorPool(self.pose_bank, self.robot.joint_names,
                                         self.asset_report.bundle_hash, self.device)
        self._apply_parameters()
        self._guidance = None
        self.guidance_execution = None
        if self.resolved_config["schema_version"] == "RecoveryResolvedConfigV3":
            from .guidance import RecoveryForceGuidance

            body_ids, body_names = self.robot.find_bodies(["base_link"], preserve_order=True)
            if body_names != ["base_link"] or len(body_ids) != 1:
                raise ValueError("辅助力要求唯一 base_link 刚体")
            self._guidance_body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.int32)
            masses = self.robot.root_physx_view.get_masses().to(self.device)
            torch.testing.assert_close(masses, self.robot.data.default_mass.to(self.device), rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(self.robot.data.root_com_pos_w,
                self.robot.data.body_com_pos_w[:, body_ids[0]], rtol=0, atol=1e-5)
            self._guidance = RecoveryForceGuidance(self.resolved_config["guidance"], masses.sum(dim=1))
            self.guidance_execution = self._guidance.identity(body_name="base_link", body_index=body_ids[0])
            self._guidance_initial_critic = torch.zeros(self.num_envs, 35, device=self.device)
        self.contact_manifest = assert_isaac_full_contact(self.sim.stage, "/World/envs/env_0/Robot")
        self._history = HistoryBuffer(self.num_envs, device=self.device)
        self._walker_history = HistoryBuffer(self.num_envs, frame_dim=25, device=self.device)
        walker_path = Path(self.resolved_config["environment"]["walker_export_dir"]) / "actor.ts"
        self.walker_manifest = verify_walker_export(walker_path.parent, self.contract)
        self._walker = torch.jit.load(str(walker_path), map_location=self.device).eval()
        for parameter in self._walker.parameters():
            parameter.requires_grad_(False)
        self._action = torch.zeros(self.num_envs, 6, device=self.device)
        self._previous_action = self._action.clone()
        self._previous_previous_action = self._action.clone()
        self._valid_prior_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._leg_targets = self._q_reference.clone()
        self._wheel_targets = torch.zeros(self.num_envs, 2, device=self.device)
        self._previous_potential = torch.zeros(self.num_envs, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._failure = self._success.clone()
        self._frame_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._joint_confirm_steps = torch.zeros(self.num_envs, device=self.device)
        self._observation_cache = None
        self._reset_records = [None] * self.num_envs
        self.reset_validation_reports = []
        self.reset_validation_statistics = {"validated_states": 0, "physical_frame_max_abs_error": 0.0,
                                            "maximum_com_height_error_m": 0.0}
        self._supervisor_config = SupervisorConfig(**self.resolved_config["supervisor"])
        self._context_registry = {
            category_id * 2 + motion_id: AttemptContext(initial_category=category, motion_kind=motion,
                randomization_profile=self._randomization.profile.name, stage="independent")
            for category_id, category in enumerate(CATEGORIES) for motion_id, motion in enumerate(("static", "dynamic"))
        }
        self._context_ids = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._metrics_archive_generation = 0
        self._supervisor = BatchedRecoverySupervisor(self.num_envs, self.device, self._supervisor_config,
                                                      standalone=True, defer_initial_begin=True)
        audit = load_usd_anchor_audit()
        self._loop_body_ids = []
        local_anchors = []
        for loop in audit["loop_joints"]:
            for side in loop["sides"]:
                body_ids, body_names = self.robot.find_bodies([side["body_name"]], preserve_order=True)
                if body_names != [side["body_name"]]:
                    raise RuntimeError("闭链锚点刚体名称无法唯一解析")
                self._loop_body_ids.append(body_ids[0])
                local_anchors.append(side["local_anchor_m"])
        self._loop_anchors = torch.tensor(local_anchors, dtype=torch.float32, device=self.device)
        wheel_names = [audit["legs"][side]["wheel"]["body_name"] for side in ("left", "right")]
        self._wheel_contact_ids, contact_names = self.contacts.find_bodies(wheel_names, preserve_order=True)
        if contact_names != wheel_names:
            raise RuntimeError("接触传感器没有解析到左右轮刚体")
        self._nonwheel_contact_ids = [index for index in range(len(self.contacts.body_names)) if index not in self._wheel_contact_ids]
        self._substep_loop_failure = self._success.clone()
        self._substep_joint_limit_failure = self._success.clone()
        self._reward_terms = {}
        self._curriculum_level = 0
        self._curriculum_outcomes = deque()
        curriculum_path = self.resolved_config["pose_sampling"]["curriculum_path"]
        self._curriculum = None if curriculum_path is None else json.loads(Path(curriculum_path).read_text(encoding="utf-8"))
        self.curriculum_sha256 = None if curriculum_path is None else file_sha256(curriculum_path)
        if self._curriculum is not None:
            self._validate_curriculum()
            self._curriculum_level = self._curriculum["initial_level"]
        self._substep_geometry_reasons = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_generations = torch.zeros_like(self._frame_ids)
        self.geometry_statistics = {"inspected_states": 0, "maximum_ground_penetration_m": 0.0,
                                    "minimum_nonallowed_clearance_m": None, "failure_states": 0,
                                    "internal_clearance_violation_states": 0, "internal_clearance_inspected_states": 0}
        with ExitStack() as pending_resources:
            if geometry_config is not None:
                self._geometry_guard = GeometryGpuGuard(geometry_config, self.resolved_config["environment"]["run_dir"],
                                                        self.device, self.robot.body_names)
                pending_resources.callback(self._geometry_guard.close)
            self._metrics_stream = self._make_metrics_stream()
            pending_resources.callback(self._metrics_stream.abort)
            pending_resources.pop_all()

    def _make_metrics_stream(self, restore_state=None):
        suffix = "metrics_archive" if self._metrics_archive_generation == 0 else f"metrics_resume_{self._metrics_archive_generation}"
        directory = Path(self.resolved_config["environment"]["run_dir"]) / suffix
        while directory.exists():
            self._metrics_archive_generation += 1
            directory = directory.parent / f"metrics_resume_{self._metrics_archive_generation}"
        return RecoveryMetricsStream(self.num_envs, self.device, directory, self._context_registry,
                                      self._supervisor_config, restore_state=restore_state)

    def check_workers(self):
        self._metrics_stream.poll()
        if self._geometry_guard is not None:
            self._geometry_guard.check_health()

    def rollout_barrier(self):
        self.check_workers()
        self._metrics_stream.submit_rollout()

    def _invalidate_state(self):
        self._write_generation += 1
        self._state_cache_key = None
        self._state_cache = None

    def _validate_curriculum(self):
        curriculum = self._curriculum
        if set(curriculum) != {"schema_version", "initial_level", "levels"} or curriculum["schema_version"] != "RecoveryCurriculumV1":
            raise ValueError("起身课程格式无效")
        if not curriculum["levels"] or not isinstance(curriculum["initial_level"], int) or not 0 <= curriculum["initial_level"] < len(curriculum["levels"]):
            raise ValueError("起身课程初始阶段无效")
        required = {"name", "category_probabilities", "minimum_completed_episodes", "promotion_success_rate", "demotion_success_rate"}
        for level in curriculum["levels"]:
            if set(level) != required or not level["name"]:
                raise ValueError("起身课程阶段字段不完整")
            probabilities = torch.tensor(level["category_probabilities"], dtype=torch.float64)
            if probabilities.shape != (6,) or not torch.isfinite(probabilities).all() or (probabilities < 0).any() or abs(float(probabilities.sum()) - 1) > 1e-9:
                raise ValueError("起身课程类别比例必须为合计一的六维概率")
            if not isinstance(level["minimum_completed_episodes"], int) or level["minimum_completed_episodes"] <= 0:
                raise ValueError("起身课程晋退窗口必须为正整数")
            if not 0 <= level["demotion_success_rate"] < level["promotion_success_rate"] <= 1:
                raise ValueError("起身课程晋退成功率范围无效")
        if any(abs(float(actual) - expected) > 1e-9 for actual, expected in zip(curriculum["levels"][-1]["category_probabilities"], [0.18] * 5 + [0.10])):
            raise ValueError("课程最终阶段必须覆盖规定的全部倒地类别")

    def _curriculum_observe(self, success: bool):
        if self._curriculum is None:
            return
        level = self._curriculum["levels"][self._curriculum_level]
        self._curriculum_outcomes.append(bool(success))
        window = level["minimum_completed_episodes"]
        while len(self._curriculum_outcomes) > window:
            self._curriculum_outcomes.popleft()
        if len(self._curriculum_outcomes) < window:
            return
        rate = sum(self._curriculum_outcomes) / window
        updated = self._curriculum_level
        if rate >= level["promotion_success_rate"] and updated < len(self._curriculum["levels"]) - 1:
            updated += 1
        elif rate < level["demotion_success_rate"] and updated > 0:
            updated -= 1
        if updated != self._curriculum_level:
            self._curriculum_level = updated
            self._curriculum_outcomes.clear()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        configure_isaac_full_contact(self.sim.get_initial_stage(), "/World/envs/env_0/Robot")
        self.cfg.ground.spawn.func(self.cfg.ground.prim_path, self.cfg.ground.spawn, translation=self.cfg.ground.translation)
        self.contacts = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", update_period=self.cfg.sim.dt,
            history_length=1, track_air_time=False,
        ))
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.ground.prim_path])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contacts"] = self.contacts
        light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _apply_parameters(self):
        parameters = self._randomization
        self.robot.write_joint_stiffness_to_sim(parameters.realized_stiffness, joint_ids=self._controlled_ids)
        self.robot.write_joint_damping_to_sim(parameters.realized_damping, joint_ids=self._controlled_ids)
        self.robot.write_joint_effort_limit_to_sim(parameters.realized_effort_limit, joint_ids=self._controlled_ids)
        canonical_columns = {joint_id: index for index, joint_id in enumerate(self._controlled_ids)}
        for name in ("legs", "wheels"):
            actuator = self.robot.actuators[name]
            ids = list(range(self.robot.num_joints))[actuator.joint_indices] if isinstance(actuator.joint_indices, slice) else list(actuator.joint_indices)
            columns = [canonical_columns[int(value)] for value in ids]
            actuator.stiffness.copy_(parameters.realized_stiffness[:, columns])
            actuator.damping.copy_(parameters.realized_damping[:, columns])
            actuator.effort_limit.copy_(parameters.realized_effort_limit[:, columns])
            actuator.effort_limit_sim.copy_(parameters.realized_effort_limit[:, columns])
        materials = self.robot.root_physx_view.get_material_properties()
        friction = parameters.friction.detach().to(device=materials.device, dtype=materials.dtype)
        materials[..., 0] = friction[:, None]
        materials[..., 1] = friction[:, None]
        materials[..., 2] = 0.0
        ids = torch.arange(self.num_envs, device="cpu", dtype=torch.int32)
        self.robot.root_physx_view.set_material_properties(materials, ids)
        actual_materials = self.robot.root_physx_view.get_material_properties()
        torch.testing.assert_close(actual_materials, materials, rtol=0, atol=1e-6)
        self.material_readback = actual_materials.detach().cpu().clone()

    @profile_phase("state_read")
    def _read_state(self):
        key = (self._sim_step_counter, self._write_generation)
        if key == self._state_cache_key:
            return self._state_cache
        state = {
            "joint_position": controlled_joint_feedback_usd_to_control(self.robot.data.joint_pos[:, self._controlled_ids]),
            "joint_velocity": controlled_joint_feedback_usd_to_control(self.robot.data.joint_vel[:, self._controlled_ids]),
            "angular_velocity": transform_usd_vector_to_control(self.robot.data.root_link_ang_vel_b),
            "projected_gravity": transform_usd_vector_to_control(projected_gravity_from_quaternion(self.robot.data.root_link_quat_w)),
            "linear_velocity": transform_usd_vector_to_control(self.robot.data.root_com_lin_vel_b),
            "height": self.robot.data.root_com_pos_w[:, 2] - self.scene.env_origins[:, 2],
            "torque": controlled_joint_feedback_usd_to_control(self.robot.data.applied_torque[:, self._controlled_ids]),
        }
        if not all_finite(*state.values()):
            raise FloatingPointError("Isaac 起身状态包含非有限数值")
        self._state_cache_key, self._state_cache = key, state
        return state

    def _loop_failures(self):
        positions = self.robot.data.body_link_pos_w[:, self._loop_body_ids]
        orientations = self.robot.data.body_link_quat_w[:, self._loop_body_ids]
        anchors = positions + quat_rotate_wxyz(orientations, self._loop_anchors[None].expand(self.num_envs, -1, -1))
        error = (anchors.reshape(self.num_envs, -1, 2, 3)[:, :, 0] - anchors.reshape(self.num_envs, -1, 2, 3)[:, :, 1]).norm(dim=-1)
        if not torch.isfinite(error).all():
            raise FloatingPointError("闭链锚点误差包含非有限数值")
        return (error >= self.contract.max_loop_error_m).any(dim=1)

    def _joint_limit_failures(self):
        position = self.robot.data.joint_pos
        limits = self.robot.data.joint_pos_limits
        if not torch.isfinite(position).all():
            raise FloatingPointError("真实关节角度包含非有限数值")
        return ((position < limits[..., 0]) | (position > limits[..., 1])).any(dim=1)

    def _inspect_geometry(self, env_ids=None, *, restoring=False, before_physics=False):
        if self._geometry_guard is None:
            return
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        positions = self.robot.data.body_link_pos_w[ids] - self.scene.env_origins[ids, None, :]
        orientations = self.robot.data.body_link_quat_w[ids]
        # DirectRLEnv先增加计数再调用_apply_action，此时刚体仍属于上一物理帧。
        physical_frame = self._sim_step_counter - int(before_physics)
        with self._performance.measure("geometry"):
            result = self._geometry_guard.inspect(self.robot.body_names, positions, orientations,
                env_ids=ids, generations=self._reset_generations[ids], frame_id=physical_frame,
                substep_id=physical_frame % self.cfg.decimation, restoring=restoring,
                state_key=None if env_ids is not None else (physical_frame, self._write_generation))
        reasons = torch.where(result.ground_failure, REASON_CODES["excessive_ground_penetration"],
            torch.where(result.internal_failure, REASON_CODES["nonallowed_body_clearance"], 0))
        if restoring and reasons.any():
            raise ValueError("恢复姿态未通过同源几何检查")
        old = self._substep_geometry_reasons[ids]
        self._substep_geometry_reasons[ids] = torch.where(old == 0, reasons, old)

    def _pre_physics_step(self, actions):
        self.check_workers()
        if actions.shape != self._action.shape or not torch.isfinite(actions).all():
            raise FloatingPointError("起身动作形状或数值无效")
        self._previous_potential.copy_((1.0 - self._read_state()["projected_gravity"][:, 2]) * 0.5)
        self._previous_previous_action.copy_(self._previous_action)
        self._previous_action.copy_(self._action)
        self._action.copy_(actions)
        leg_targets, wheel_targets = decode_action(actions, self._q_reference, self.contract)
        self._leg_targets.copy_(leg_targets)
        self._wheel_targets.copy_(wheel_targets)
        self._observation_cache = None
        self._substep_loop_failure.zero_()
        self._substep_joint_limit_failure.zero_()
        self._substep_geometry_reasons.zero_()
        if self._guidance is not None:
            self._guidance.begin_transition()

    def _apply_action(self):
        self._performance.end_physics_interval()
        # 每个物理子步读取上个子步的闭链状态，最后一个子步由 _get_dones 再次检查。
        self._substep_loop_failure |= self._loop_failures()
        self._substep_joint_limit_failure |= self._joint_limit_failures()
        self._inspect_geometry(before_physics=True)
        self.robot.set_joint_position_target(self._leg_targets, joint_ids=self._leg_ids)
        self.robot.set_joint_velocity_target(wheel_targets_control_to_usd(self._wheel_targets), joint_ids=self._wheel_ids)
        if self._guidance is not None:
            force = self._guidance.substep(self._read_state()["height"])
            self.robot.instantaneous_wrench_composer.set_forces_and_torques(
                forces=force, torques=self._guidance.zero_torque_world, positions=None,
                body_ids=self._guidance_body_ids, is_global=True)
        self._performance.begin_physics_interval()

    @staticmethod
    def _walker_frame(recovery_frame):
        command = recovery_frame.new_zeros(recovery_frame.shape[0], 3)
        return torch.cat((recovery_frame[:, :6], command, recovery_frame[:, 6:]), dim=-1)

    @profile_phase("observation")
    def _prepare_observation(self):
        if self._observation_cache is not None:
            return
        state = self._read_state()
        self._clean_frame = build_actor_frame(
            state["angular_velocity"], state["projected_gravity"], state["joint_position"],
            state["joint_velocity"], self._action, self._q_reference,
        )
        noisy_frame = build_actor_frame(
            state["angular_velocity"], state["projected_gravity"], state["joint_position"],
            state["joint_velocity"], self._action, self._q_reference,
            actor_noise=self._randomization.sample_actor_noise(self.num_envs),
        )
        self._history.append(noisy_frame, self.common_step_counter)
        self._walker_history.append(self._walker_frame(self._clean_frame), self.common_step_counter)
        self._observation_cache = {"policy": self._policy_observation(), "critic": self._critic(state)}

    def _policy_observation(self):
        history = self._history.flattened()
        return history if self._guidance is None else torch.cat((history, self._guidance.alpha[:, None]), dim=-1)

    def _critic(self, state):
        frame = build_critic_frame(
            self._clean_frame, state["linear_velocity"], state["height"], state["torque"],
            self._joint_confirm_steps / 50.0,
            (self.contract.max_recovery_steps - self.episode_length_buf).clamp_min(0).float() / self.contract.max_recovery_steps,
        )
        return frame if self._guidance is None else torch.cat((frame, self._guidance.alpha[:, None]), dim=-1)

    def _get_dones(self):
        self._performance.end_physics_interval()
        if self._guidance is not None:
            self.extras["guidance"] = self._guidance.finish_transition()
        self._prepare_observation()
        state = self._read_state()
        self._substep_loop_failure |= self._loop_failures()
        self._substep_joint_limit_failure |= self._joint_limit_failures()
        self._inspect_geometry()
        with torch.inference_mode():
            candidate = self._walker(self._walker_history.flattened().clone())
        if candidate.shape != self._action.shape or not torch.isfinite(candidate).all():
            raise FloatingPointError("冻结行走候选动作无效")
        readiness = self.resolved_config["readiness"]
        calibrated = all(readiness[key] is not None for key in ("support_force_n", "target_difference_limits", "torque_difference_limits"))
        if calibrated:
            compatibility = check_action_compatibility(
                self._action, candidate, state["joint_position"], state["joint_velocity"], self._q_reference,
                self._randomization.realized_stiffness, self._randomization.realized_damping,
                self._randomization.realized_effort_limit,
                self._action.new_tensor(readiness["target_difference_limits"]),
                self._action.new_tensor(readiness["torque_difference_limits"]), self.contract,
            )
            target_bad, torque_bad = compatibility["target_blocked"], compatibility["torque_blocked"]
        else:
            target_bad = torch.ones_like(self._action, dtype=torch.bool)
            torque_bad = torch.ones_like(self._action, dtype=torch.bool)
        force = self.contacts.data.net_forces_w
        if not torch.isfinite(force).all():
            raise FloatingPointError("接触传感器包含非有限数值")
        if calibrated:
            wheel_support = force[:, self._wheel_contact_ids, 2] > readiness["support_force_n"]
            nonwheel_support = (force[:, self._nonwheel_contact_ids, 2] > readiness["support_force_n"]).any(dim=1)
        else:
            wheel_support = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.bool)
            nonwheel_support = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        tilt = torch.acos((-state["projected_gravity"][:, 2]).clamp(-1.0, 1.0))
        self._frame_ids += 1
        values = torch.cat((tilt[:, None], state["angular_velocity"],
                            state["joint_position"][:, :4] - state["joint_position"].new_tensor(self.contract.q_nominal),
                            state["joint_velocity"], state["height"][:, None]), dim=1)
        if calibrated:
            weights = self._frame_ids.new_tensor([BLOCKER_BITS[f"target_{j}"] for j in range(6)])
            torque_weights = self._frame_ids.new_tensor([BLOCKER_BITS[f"torque_{j}"] for j in range(6)])
            blockers = (target_bad.to(torch.int64) * weights).sum(1) | (torque_bad.to(torch.int64) * torque_weights).sum(1)
        else:
            blockers = torch.full_like(self._frame_ids, BLOCKER_BITS["calibration_unavailable"])
        reasons = torch.where(self._substep_loop_failure, REASON_CODES["loop_closure"],
            torch.where(self._substep_joint_limit_failure, REASON_CODES["joint_position_limit"], self._substep_geometry_reasons))
        with self._performance.measure("supervisor"):
            result = self._supervisor.post_step(values, self._frame_ids, wheel_support=wheel_support,
                body_support=nonwheel_support, action_compatible=blockers == 0,
                compatibility_mask=blockers, physical_failure=reasons)
        self._metrics_stream.submit(result)
        self._success.copy_(result.success)
        self._failure.copy_(result.failure)
        self._joint_confirm_steps.copy_(result.joint_confirm_steps)
        terminal_ids = (self._success | self._failure).nonzero(as_tuple=False).flatten()
        if self._curriculum is not None and terminal_ids.numel():
            for success in self._success[terminal_ids].cpu().tolist():
                self._curriculum_observe(success)
        self._valid_prior_count += 1
        self._observation_cache["critic"] = self._critic(state)
        return self._success | self._failure, torch.zeros_like(self._success)

    @profile_phase("reward")
    def _get_rewards(self):
        state = self._read_state()
        reward, self._reward_terms = compute_reward(
            state["projected_gravity"], state["joint_position"], state["joint_velocity"], state["angular_velocity"],
            state["height"], state["torque"], self._randomization.realized_effort_limit, self._action,
            self._previous_action, self._previous_previous_action, self._previous_potential,
            self._success, self._failure, self._valid_prior_count >= 3,
            q_nominal=self._action.new_tensor(self.contract.q_nominal), contract=self.contract,
        )
        self.extras["log"] = {"Recovery/success": self._success.float().mean(), "Recovery/failure": self._failure.float().mean()}
        self.extras["log"]["Recovery/curriculum_level"] = float(self._curriculum_level)
        self.extras["log"].update({f"RecoveryReward/{name}": value.mean() for name, value in self._reward_terms.items()})
        return reward

    def _get_observations(self):
        if self._observation_cache is None:
            raise RuntimeError("起身观测尚未由 reset 或物理动作准备")
        return {key: value.clone() for key, value in self._observation_cache.items()}

    @torch.inference_mode()
    @profile_phase("reset_and_curriculum_sampling")
    def _reset_idx(self, env_ids):
        ids = torch.as_tensor(self.robot._ALL_INDICES if env_ids is None else env_ids, dtype=torch.long, device=self.device)
        if not len(ids):
            return
        if not hasattr(self, "_supervisor"):
            raise RuntimeError("起身环境初始化必须在首次 reset 前完成")
        active = ids[self._supervisor.event_active[ids] | self._supervisor.attempt_active[ids]]
        if active.numel():
            self._metrics_stream.submit(self._supervisor.interrupt("manual_reset", env_ids=active), kind="interrupt")
        super()._reset_idx(ids)
        self.robot.reset(ids)
        self.contacts.reset(ids)
        if self._guidance is not None:
            self._guidance.reset(ids, completed=(self._success | self._failure)[ids])
        selected_records, context_ids = [], []
        for index in ids.tolist():
            category = None
            generator = self._randomization.pose_generator(index)
            if self._curriculum is not None:
                probabilities = torch.tensor(self._curriculum["levels"][self._curriculum_level]["category_probabilities"])
                category = CATEGORIES[int(torch.multinomial(probabilities, 1, generator=generator))]
            record = self.pose_bank.sample(index, self._pose_plan_hashes[index], generator, split="train", category=category)
            selected_records.append(record)
            self._reset_records[index] = record.record_id
            context_ids.append(CATEGORIES.index(record.initial_category) * 2 + int(record.motion_kind == "dynamic"))
        self._context_ids[ids] = torch.tensor(context_ids, dtype=torch.int64, device=self.device)
        self._metrics_stream.submit(self._supervisor.reset(ids, context_ids=self._context_ids, frame_ids=self._frame_ids), kind="reset")
        self._reset_generations[ids] += 1
        pool, selected = self._pose_pool, self._pose_pool.indices(selected_records)
        self._action[ids] = pool.action[selected]
        self._previous_action[ids] = pool.previous_action[selected]
        self._previous_previous_action[ids] = self._previous_action[ids]
        self._valid_prior_count[ids] = pool.prior_count[selected]
        targets = pool.motor_targets[selected]
        self._leg_targets[ids], self._wheel_targets[ids] = targets[:, :4], targets[:, 4:]
        self._success[ids], self._failure[ids], self._joint_confirm_steps[ids] = False, False, 0
        root = pool.root_state[selected].clone()
        root[:, :3] += self.scene.env_origins[ids]
        q, qd = pool.joint_position[selected], pool.joint_velocity[selected]
        self.robot.write_root_pose_to_sim(root[:, :7], ids)
        self.robot.write_root_link_velocity_to_sim(root[:, 7:], ids)
        self.robot.write_joint_state_to_sim(q, qd, None, ids)
        self.robot.set_joint_position_target(self._leg_targets[ids], joint_ids=self._leg_ids, env_ids=ids)
        self.robot.set_joint_velocity_target(wheel_targets_control_to_usd(self._wheel_targets[ids]), joint_ids=self._wheel_ids, env_ids=ids)
        self.robot.data._root_link_vel_w.timestamp = -1.0
        self.robot.data._root_com_pose_w.timestamp = -1.0
        self._invalidate_state()
        physical = pool.history[selected]
        normalized = []
        for column in range(5):
            frame = physical[:, column]
            q_frame = torch.cat((frame[:, 6:10] + self._q_reference[ids], frame.new_zeros(len(ids), 2)), dim=-1)
            normalized.append(build_actor_frame(frame[:, :3], frame[:, 3:6], q_frame, frame[:, 10:16], frame[:, 16:22], self._q_reference[ids], actor_noise=self._randomization.sample_actor_noise(len(ids))))
        normalized_history = torch.stack(normalized, dim=1)
        self._history.reset(ids, normalized_history)
        clean_normalized_history = normalize_physical_frame(physical)
        walker_frames = torch.stack([self._walker_frame(clean_normalized_history[:, column]) for column in range(5)], dim=1)
        self._walker_history.reset(ids, walker_frames)
        state = self._read_state()
        actual_physical = torch.cat((state["angular_velocity"], state["projected_gravity"],
                                     state["joint_position"][:, :4] - self._q_reference,
                                     state["joint_velocity"], self._action), dim=-1)
        physical_errors = (actual_physical[ids] - physical[:, -1]).abs().amax(dim=1).cpu().tolist()
        actual_heights = state["height"][ids].cpu().tolist()
        for row, index in enumerate(ids.tolist()):
            error = physical_errors[row]
            record = selected_records[row]
            expected_height = record.metadata.get("com_height_m")
            height_error = None if expected_height is None else abs(actual_heights[row] - expected_height)
            report = {"env_id": index, "record_id": record.record_id,
                      "physical_frame_max_abs_error": error, "com_height_error_m": height_error,
                      "actual_com_height_m": actual_heights[row]}
            if len(self.reset_validation_reports) < 64:
                self.reset_validation_reports.append(report)
            stats = self.reset_validation_statistics
            stats["validated_states"] += 1
            stats["physical_frame_max_abs_error"] = max(stats["physical_frame_max_abs_error"], error)
            stats["maximum_com_height_error_m"] = max(stats["maximum_com_height_error_m"], height_error or 0.0)
            if error > 1e-4 or (height_error is not None and height_error > 1e-4):
                raise ValueError(f"姿态恢复读回与源物理状态不同: {report}")
        self._clean_frame = build_actor_frame(state["angular_velocity"], state["projected_gravity"], state["joint_position"], state["joint_velocity"], self._action, self._q_reference)
        self._observation_cache = {"policy": self._policy_observation(), "critic": self._critic(state)}
        if self._guidance is not None:
            self._guidance_initial_critic[ids] = self._observation_cache["critic"][ids]
        if self._loop_failures()[ids].any() or self._joint_limit_failures()[ids].any():
            raise ValueError("恢复姿态真实读回未通过闭链或机械关节范围检查")
        self._inspect_geometry(ids, restoring=True)
        self.check_workers()

    def close(self):
        try:
            if hasattr(self, "_metrics_stream") and not self._metrics_stream._closed:
                try:
                    self._metrics_stream.close(self._frame_ids)
                finally:
                    self._metrics_stream.abort()
        finally:
            try:
                if self._geometry_guard is not None:
                    self._geometry_guard.close()
                    self._geometry_guard = None
            finally:
                super().close()

    def finalize_metrics(self):
        geometry = None if self._geometry_guard is None else self._geometry_guard.flush()
        if geometry is not None:
            self.geometry_statistics = geometry["statistics"]
        self._final_metrics = self._metrics_stream.close(self._frame_ids, reason="training_run_end")
        return self._final_metrics

    def environment_state_dict(self):
        geometry = None if self._geometry_guard is None else self._geometry_guard.flush()
        if geometry is not None:
            self.geometry_statistics = geometry["statistics"]
        snapshot = self._metrics_stream.checkpoint(self._frame_ids, reason="checkpoint_resume")
        return {"randomization": self._randomization.state_dict(), "frame_ids": self._frame_ids.detach().cpu().clone(),
                **snapshot, "geometry_barrier": None if geometry is None else geometry["barrier"],
                "curriculum_sha256": self.curriculum_sha256,
                "curriculum_level": self._curriculum_level, "curriculum_outcomes": list(self._curriculum_outcomes)}

    def restore_environment_state(self, state):
        self._metrics_stream.close(self._frame_ids, reason="checkpoint_resume")
        self._metrics_archive_generation += 1
        self._metrics_stream = self._make_metrics_stream(restore_state=state)
        self._randomization.load_state_dict(state["randomization"])
        if state["curriculum_sha256"] != self.curriculum_sha256:
            raise ValueError("恢复时起身课程身份不一致")
        self._curriculum_level = state["curriculum_level"]
        self._curriculum_outcomes = deque(state["curriculum_outcomes"])
        self._frame_ids.copy_(state["frame_ids"].to(self.device))
        next_ids = self._metrics_stream.next_ids()
        self._supervisor = BatchedRecoverySupervisor(self.num_envs, self.device, self._supervisor_config,
            standalone=True, defer_initial_begin=True,
            next_event_ids=torch.tensor(next_ids["next_event_id"], dtype=torch.int64, device=self.device),
            next_attempt_ids=torch.tensor(next_ids["next_attempt_id"], dtype=torch.int64, device=self.device))
        self._supervisor.generation.copy_(torch.tensor(next_ids["generations"], dtype=torch.int64, device=self.device).clamp_min(0))
        self._reset_generations.copy_(self._supervisor.generation)
        self.reset()
