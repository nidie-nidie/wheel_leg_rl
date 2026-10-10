from __future__ import annotations

from typing import Any

import torch
from isaacsim.core.simulation_manager import SimulationManager

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import math as math_utils

from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG
from wheelleg_dreamwaq.kinematics.virtual_leg import virtual_leg_from_body_vector
from wheelleg_dreamwaq.schemas.action import controlled_joint_feedback_usd_to_control
from wheelleg_dreamwaq.schemas.frames import (
    R_CONTROL_FROM_USD,
    projected_gravity_from_quaternion,
    quat_rotate_inverse_wxyz,
    quat_rotate_wxyz,
    transform_usd_vector_to_control,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import CommandRanges
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg

from .pitch_torque_pulse import BasePitchTorquePulse


def make_debug_env_cfg(*, device: str, enable_contact_sensors: bool) -> WheelLegFlatEnvCfg:
    debug_robot_cfg = WHEELLEG_CFG.replace(
        spawn=WHEELLEG_CFG.spawn.replace(
            activate_contact_sensors=enable_contact_sensors,
        ),
    )
    cfg = WheelLegFlatEnvCfg()
    cfg.robot_cfg = debug_robot_cfg
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    cfg.seed = 0
    cfg.commands = CommandRanges(
        vx=(0.0, 0.0),
        yaw_rate=(0.0, 0.0),
        base_height=(0.20, 0.20),
        mode_probabilities=(1.0, 0.0, 0.0, 0.0),
        hold_for_episode=True,
    )
    if WHEELLEG_CFG.spawn.activate_contact_sensors:
        raise RuntimeError("Debug configuration mutated the formal WHEELLEG_CFG")
    if cfg.robot_cfg.spawn is WHEELLEG_CFG.spawn:
        raise RuntimeError("Debug robot spawn must be an independent nested config")
    return cfg


class WheelLegSim2SimDebugEnv(WheelLegFlatEnv):
    def __init__(
        self,
        cfg: WheelLegFlatEnvCfg,
        render_mode: str | None = None,
        *,
        enable_contact_sensors: bool = False,
        **kwargs,
    ) -> None:
        self._debug_enable_contact_sensors = enable_contact_sensors
        self._debug_all_hinge_ids: list[int] | None = None
        self._debug_all_hinge_names: tuple[str, ...] | None = None
        self._debug_initial_com_world: torch.Tensor | None = None
        self._debug_reset_snapshot: dict[str, Any] | None = None
        self._debug_substeps: list[dict[str, Any]] = []
        self._debug_terminal_state: dict[str, Any] | None = None
        self._debug_action_clipped: torch.Tensor | None = None
        self._debug_next_obs: torch.Tensor | None = None
        self._debug_next_obs_is_reset: torch.Tensor | None = None
        self._debug_pitch_torque_pulse = BasePitchTorquePulse.disabled()
        super().__init__(cfg, render_mode, **kwargs)
        body_ids, body_names = self.robot.find_bodies(["base_link"], preserve_order=True)
        if body_names != ["base_link"] or len(body_ids) != 1:
            raise RuntimeError(f"base_link did not resolve uniquely: {body_names}")
        self._debug_base_body_ids = torch.tensor(body_ids, dtype=torch.int32, device=self.device)
        self._debug_external_forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._debug_external_torques = torch.zeros_like(self._debug_external_forces)

    def _setup_scene(self) -> None:
        super()._setup_scene()
        if not self._debug_enable_contact_sensors:
            self._debug_left_contact = None
            self._debug_right_contact = None
            return
        sensor_kwargs = {
            "track_contact_points": False,
            "track_friction_forces": False,
            "max_contact_data_count_per_prim": 8,
        }
        self._debug_left_contact = ContactSensor(
            ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/jwheel_left", **sensor_kwargs)
        )
        self._debug_right_contact = ContactSensor(
            ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/jwheel_right", **sensor_kwargs)
        )
        self.scene.sensors["debug_left_wheel_contact"] = self._debug_left_contact
        self.scene.sensors["debug_right_wheel_contact"] = self._debug_right_contact

    def configure_debug_joint_order(self, joint_names: list[str]) -> None:
        if len(joint_names) != 26 or len(set(joint_names)) != 26:
            raise ValueError("Debug all-hinge order must contain 26 unique names")
        joint_ids, resolved = self.robot.find_joints(joint_names, preserve_order=True)
        if resolved != joint_names or len(set(joint_ids)) != 26:
            raise RuntimeError(f"All-hinge order did not resolve uniquely: {resolved}")
        self._debug_all_hinge_ids = joint_ids
        self._debug_all_hinge_names = tuple(resolved)

    def configure_debug_pitch_torque(self, pulse: BasePitchTorquePulse) -> None:
        self._debug_pitch_torque_pulse = pulse

    def _debug_write_pitch_torque(self) -> tuple[torch.Tensor, torch.Tensor]:
        value = self._debug_pitch_torque_pulse.value_at(int(self.common_step_counter))
        self._debug_external_torques.zero_()
        self._debug_external_torques[..., 0] = value
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._debug_external_forces,
            torques=self._debug_external_torques,
            body_ids=self._debug_base_body_ids,
            is_global=False,
        )
        control = self._debug_external_torques.new_zeros((self.num_envs, 3))
        control[:, 1] = value
        world = quat_rotate_wxyz(
            self.robot.data.root_link_quat_w,
            self._debug_external_torques[:, 0],
        )
        return control, world

    def _debug_contact_state(self) -> dict[str, torch.Tensor]:
        if not self._debug_enable_contact_sensors:
            return {
                "wheel_contact_active": torch.full((self.num_envs, 2), -1, dtype=torch.int8, device=self.device),
                "wheel_normal_force_n": torch.full((self.num_envs, 2), torch.nan, device=self.device),
                "wheel_normal_impulse_ns": torch.full((self.num_envs, 2), torch.nan, device=self.device),
                "wheel_normal_force_world": torch.full(
                    (self.num_envs, 2, 3), torch.nan, device=self.device
                ),
                "wheel_friction_force_world": torch.full(
                    (self.num_envs, 2, 3), torch.nan, device=self.device
                ),
                "unexpected_contact": torch.full((self.num_envs,), -1, dtype=torch.int8, device=self.device),
            }
        normal_vectors = []
        for sensor in (self._debug_left_contact, self._debug_right_contact):
            data = sensor.data
            normal_vectors.append(
                torch.nan_to_num(data.net_forces_w).reshape(self.num_envs, -1, 3).sum(dim=1)
            )
        # Isaac Lab defines net_forces_w as normal force only, excluding friction.
        normal_world = torch.stack(normal_vectors, dim=1)
        normal_force = torch.linalg.vector_norm(normal_world, dim=-1)
        friction_world = torch.full_like(normal_world, torch.nan)
        return {
            "wheel_contact_active": (normal_force > 1.0).to(torch.int8),
            "wheel_normal_force_n": normal_force,
            "wheel_normal_impulse_ns": normal_force * self.physics_dt,
            "wheel_normal_force_world": normal_world,
            "wheel_friction_force_world": friction_world,
            "unexpected_contact": torch.full((self.num_envs,), -1, dtype=torch.int8, device=self.device),
        }

    def _debug_capture_direct_state(self) -> dict[str, torch.Tensor]:
        if self._debug_all_hinge_ids is None:
            raise RuntimeError("configure_debug_joint_order() must run before trace collection")
        view = self.robot.root_physx_view
        self.robot.data._physics_sim_view.update_articulations_kinematic()
        joint_position_native_all = view.get_dof_positions().clone()
        joint_velocity_native_all = view.get_dof_velocities().clone()
        root_pose = view.get_root_transforms().clone()
        root_pose[:, 3:7] = math_utils.convert_quat(root_pose[:, 3:7], to="wxyz")
        root_quaternion = root_pose[:, 3:7]
        root_velocity_world = view.get_root_velocities().clone()
        link_pose = view.get_link_transforms().clone()
        link_pose[..., 3:7] = math_utils.convert_quat(link_pose[..., 3:7], to="wxyz")

        root_com_world = root_pose[:, :3] + math_utils.quat_apply(
            root_quaternion, self.robot.data.body_com_pos_b[:, 0]
        )
        root_linear_body = quat_rotate_inverse_wxyz(root_quaternion, root_velocity_world[:, :3])
        root_angular_body = quat_rotate_inverse_wxyz(root_quaternion, root_velocity_world[:, 3:])
        root_linear_control = transform_usd_vector_to_control(root_linear_body)
        root_angular_control = transform_usd_vector_to_control(root_angular_body)
        projected_gravity = transform_usd_vector_to_control(
            projected_gravity_from_quaternion(root_quaternion)
        )

        controlled_position_native = joint_position_native_all[:, self._controlled_joint_ids]
        controlled_velocity_native = joint_velocity_native_all[:, self._controlled_joint_ids]
        controlled_position = controlled_joint_feedback_usd_to_control(controlled_position_native)
        controlled_velocity = controlled_joint_feedback_usd_to_control(controlled_velocity_native)

        body_position = link_pose[..., :3]
        body_quaternion = link_pose[..., 3:7]
        hip_local = self._virtual_leg_hip_local_anchors.unsqueeze(0).expand(self.num_envs, -1, -1)
        wheel_local = self._virtual_leg_wheel_local_anchors.unsqueeze(0).expand(self.num_envs, -1, -1)
        hip_world = body_position[:, self._virtual_leg_hip_body_ids] + quat_rotate_wxyz(
            body_quaternion[:, self._virtual_leg_hip_body_ids], hip_local
        )
        wheel_world = body_position[:, self._virtual_leg_wheel_body_ids] + quat_rotate_wxyz(
            body_quaternion[:, self._virtual_leg_wheel_body_ids], wheel_local
        )
        leg_vector_world = wheel_world - hip_world
        root_for_legs = root_quaternion.unsqueeze(1).expand(-1, 2, -1)
        leg_vector_body = quat_rotate_inverse_wxyz(root_for_legs, leg_vector_world)
        virtual_leg_length, virtual_leg_phi0 = virtual_leg_from_body_vector(leg_vector_body)

        loop_local = self._loop_local_anchors.unsqueeze(0).expand(self.num_envs, -1, -1)
        loop_world = body_position[:, self._loop_body_ids] + quat_rotate_wxyz(
            body_quaternion[:, self._loop_body_ids], loop_local
        )
        loop_world = loop_world.reshape(self.num_envs, -1, 2, 3)
        loop_each = torch.linalg.vector_norm(loop_world[:, :, 0] - loop_world[:, :, 1], dim=-1)
        if loop_each.shape[1] != 4:
            raise RuntimeError(f"Expected four loop-closure constraints, got {loop_each.shape[1]}")
        loop_error = torch.stack((loop_each[:, :2].amax(dim=-1), loop_each[:, 2:].amax(dim=-1)), dim=-1)

        rotation_world_from_body = math_utils.matrix_from_quat(root_quaternion)
        control_rotation = R_CONTROL_FROM_USD.to(device=self.device, dtype=root_quaternion.dtype)
        rotation_control_world_from_body = (
            control_rotation.unsqueeze(0)
            @ rotation_world_from_body
            @ control_rotation.T.unsqueeze(0)
        )
        orientation_control = math_utils.quat_from_matrix(rotation_control_world_from_body)
        orientation_control = torch.where(
            orientation_control[:, :1] < 0.0, -orientation_control, orientation_control
        )

        if self._debug_initial_com_world is None:
            position_diag = torch.zeros_like(root_com_world)
        else:
            position_diag = transform_usd_vector_to_control(root_com_world - self._debug_initial_com_world)
        base_height = root_com_world[:, 2] - self.scene.env_origins[:, 2]
        position_diag[:, 2] = base_height
        contact = self._debug_contact_state()
        return {
            "active_joint_position_engine_native": controlled_position_native,
            "active_joint_velocity_engine_native": controlled_velocity_native,
            "active_joint_position_canonical": controlled_position,
            "active_joint_velocity_canonical": controlled_velocity,
            "all_hinge_position_named": joint_position_native_all[:, self._debug_all_hinge_ids],
            "all_hinge_velocity_named": joint_velocity_native_all[:, self._debug_all_hinge_ids],
            "base_com_position_engine_world": root_com_world,
            "base_com_position_diag": position_diag,
            "base_orientation_control_wxyz": orientation_control,
            "base_linear_velocity_control": root_linear_control,
            "base_angular_velocity_control": root_angular_control,
            "projected_gravity": projected_gravity,
            "base_height": base_height,
            "virtual_leg_length": virtual_leg_length,
            "virtual_leg_phi0": virtual_leg_phi0,
            "loop_closure_error": loop_error,
            **contact,
        }

    @staticmethod
    def _debug_clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in state.items()}

    def _debug_target_and_reference(
        self, pre_state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_native = torch.cat((self._leg_position_targets, self._wheel_velocity_targets), dim=-1)
        target_canonical = controlled_joint_feedback_usd_to_control(target_native)
        q = pre_state["active_joint_position_canonical"]
        qd = pre_state["active_joint_velocity_canonical"]
        unclipped = torch.empty_like(q)
        unclipped[:, :4] = 120.0 * (target_canonical[:, :4] - q[:, :4]) - 4.0 * qd[:, :4]
        unclipped[:, 4:6] = 0.6 * (target_canonical[:, 4:6] - qd[:, 4:6])
        clipped = torch.clamp(
            unclipped,
            -self._controlled_effort_limits.unsqueeze(0),
            self._controlled_effort_limits.unsqueeze(0),
        )
        return target_canonical, target_native, unclipped, clipped

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, torch.Tensor], dict]:
        del options
        if seed is not None:
            self.seed(seed)
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)
        pre_forward_state = self._debug_capture_direct_state()
        pre_forward_cached_obs = self._get_observations()["policy"].detach().clone()
        self.scene.write_data_to_sim()
        self.sim.forward()
        post_forward_state = self._debug_capture_direct_state()
        self._debug_initial_com_world = post_forward_state["base_com_position_engine_world"].detach().clone()
        post_forward_state = self._debug_capture_direct_state()

        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()
        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        observations = self._get_observations()
        returned_policy = observations["policy"].detach().clone()
        self.obs_buf = observations
        self._debug_reset_snapshot = {
            "reset_written_pre_forward": self._debug_clone_state(pre_forward_state),
            "reset_written_cached_actor_obs_policy": pre_forward_cached_obs,
            "reset_forwarded_post_forward": self._debug_clone_state(post_forward_state),
            "reset_returned_actor_obs_policy": returned_policy,
        }
        return observations, self.extras

    def step(self, action: torch.Tensor):
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)
        self._debug_substeps = []
        self._pre_physics_step(action)
        self._debug_action_clipped = self._canonical_action.detach().clone()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        post_state: dict[str, torch.Tensor] | None = None
        for substep_index in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            pre_state = self._debug_capture_direct_state()
            target_canonical, target_native, unclipped, clipped = self._debug_target_and_reference(pre_state)
            velocity_limits = self.robot.data.joint_vel_limits[0, self._controlled_joint_ids]
            velocity_event = (
                torch.abs(pre_state["active_joint_velocity_engine_native"]) >= velocity_limits.unsqueeze(0)
            ).to(torch.int8)
            effort_event = (torch.abs(unclipped) > self._controlled_effort_limits.unsqueeze(0)).to(torch.int8)
            external_torque_control, external_torque_world = self._debug_write_pitch_torque()
            self.scene.write_data_to_sim()
            host_torque_native = self.robot.data.applied_torque[:, self._controlled_joint_ids].detach().clone()
            host_torque_canonical = controlled_joint_feedback_usd_to_control(host_torque_native)
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            post_state = self._debug_capture_direct_state()
            row: dict[str, Any] = {
                "control_tick": int(self.common_step_counter),
                "substep_index": substep_index,
                "physics_time_s": float(self._sim_step_counter * self.physics_dt),
                "target_command_canonical": target_canonical.detach().clone(),
                "target_command_engine_native": target_native.detach().clone(),
                "pd_torque_unclipped_canonical": unclipped.detach().clone(),
                "pd_torque_unclipped_engine_native": controlled_joint_feedback_usd_to_control(unclipped).detach().clone(),
                "pd_torque_effort_clipped_canonical": clipped.detach().clone(),
                "pd_torque_effort_clipped_engine_native": controlled_joint_feedback_usd_to_control(clipped).detach().clone(),
                "isaac_host_pd_torque_estimate_canonical": host_torque_canonical,
                "isaac_host_pd_torque_estimate_engine_native": host_torque_native,
                "mujoco_commanded_torque_canonical": torch.full_like(unclipped, torch.nan),
                "mujoco_commanded_torque_engine_native": torch.full_like(unclipped, torch.nan),
                "effort_limit_event": effort_event,
                "joint_velocity_limit_exceeded": velocity_event,
                "mujoco_velocity_guard_active": torch.full_like(effort_event, -1),
                "external_base_torque_control": external_torque_control.detach().clone(),
                "external_base_torque_engine_world": external_torque_world.detach().clone(),
            }
            for suffix, state in (("pre_step", pre_state), ("post_step", post_state)):
                for name in (
                    "active_joint_position_canonical",
                    "active_joint_velocity_canonical",
                    "active_joint_position_engine_native",
                    "active_joint_velocity_engine_native",
                    "all_hinge_position_named",
                    "all_hinge_velocity_named",
                    "base_com_position_engine_world",
                    "base_com_position_diag",
                    "base_orientation_control_wxyz",
                    "base_linear_velocity_control",
                    "base_angular_velocity_control",
                ):
                    row[f"{name}_{suffix}"] = state[name].detach().clone()
            for name in (
                "projected_gravity",
                "virtual_leg_length",
                "virtual_leg_phi0",
                "loop_closure_error",
            ):
                row[f"{name}_post_step"] = post_state[name].detach().clone()
            for name in (
                "wheel_contact_active",
                "wheel_normal_force_n",
                "wheel_normal_impulse_ns",
                "wheel_normal_force_world",
                "wheel_friction_force_world",
            ):
                row[name] = post_state[name].detach().clone()
            self._debug_substeps.append(row)

        if post_state is None:
            raise RuntimeError("Debug step did not execute a physics substep")
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self._debug_terminal_state = self._debug_clone_state(post_state)
        self._debug_termination_diagnostics = {
            name: value.detach().clone()
            for name, value in self.extras.get("termination_diagnostics", {}).items()
        }
        self.reward_buf = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])
        self._debug_next_obs = self.obs_buf["policy"].detach().clone()
        self._debug_next_obs_is_reset = self.reset_buf.to(torch.int8).detach().clone()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
