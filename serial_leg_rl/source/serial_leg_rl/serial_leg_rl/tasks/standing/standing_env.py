from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from .standing_env_cfg import StandingEnvCfg


class StandingEnv(DirectRLEnv):
    cfg: StandingEnvCfg

    def __init__(self, cfg: StandingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._leg_joint_ids, self._leg_joint_names = self.robot.find_joints(self.cfg.leg_joint_names, preserve_order=True)
        self._wheel_joint_ids, self._wheel_joint_names = self.robot.find_joints(
            self.cfg.wheel_joint_names, preserve_order=True
        )
        self._passive_joint_ids, self._passive_joint_names = self.robot.find_joints(
            self.cfg.passive_joint_names, preserve_order=True
        )
        print(f"[StandingEnv] leg joints: {self._leg_joint_names} -> {self._leg_joint_ids}")
        print(f"[StandingEnv] wheel joints: {self._wheel_joint_names} -> {self._wheel_joint_ids}")
        self.left_wheel_contact = self.scene.sensors["left_wheel_contact"]
        self.right_wheel_contact = self.scene.sensors["right_wheel_contact"]

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self._leg_position_targets = self.robot.data.default_joint_pos[:, self._leg_joint_ids].clone()
        self._wheel_velocity_targets = torch.zeros((self.num_envs, len(self._wheel_joint_ids)), device=self.device)
        self._passive_position_targets = self.robot.data.default_joint_pos[:, self._passive_joint_ids].clone()

        self.commands = torch.full((self.num_envs, 1), self.cfg.target_base_height, device=self.device)
        self._single_obs = torch.zeros((self.num_envs, self.cfg.single_observation_dim), device=self.device)
        self._obs_history = torch.zeros(
            (self.num_envs, self.cfg.history_length, self.cfg.single_observation_dim), device=self.device
        )
        self._episode_reward_sums = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.robot = self.scene.articulations["robot"]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions[:] = self.actions
        target_actions = torch.clamp(actions, -1.0, 1.0)
        delta = torch.clamp(
            target_actions - self.actions,
            min=-self.cfg.action_rate_limit,
            max=self.cfg.action_rate_limit,
        )
        self.actions[:] = torch.clamp(self.actions + delta, -1.0, 1.0)

        default_leg_pos = self.robot.data.default_joint_pos[:, self._leg_joint_ids]
        self._leg_position_targets[:] = default_leg_pos + self.cfg.action_scale_leg_pos * self.actions[:, :4]
        self._wheel_velocity_targets[:] = self.cfg.action_scale_wheel_vel * self.actions[:, 4:6]

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._leg_position_targets, joint_ids=self._leg_joint_ids)
        self.robot.set_joint_velocity_target(self._wheel_velocity_targets, joint_ids=self._wheel_joint_ids)
        self.robot.set_joint_position_target(self._passive_position_targets, joint_ids=self._passive_joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        leg_pos_rel = self.robot.data.joint_pos[:, self._leg_joint_ids] - self.robot.data.default_joint_pos[
            :, self._leg_joint_ids
        ]
        leg_vel = self.robot.data.joint_vel[:, self._leg_joint_ids]
        wheel_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids]
        pitch, yaw, pitch_rate, yaw_rate = self._base_pitch_yaw_state()
        leg_lengths, leg_length_rates, _, _ = self._virtual_leg_state()
        wheel_contacts = self._wheel_contacts().float()

        self._single_obs[:] = torch.cat(
            (
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                pitch.unsqueeze(-1),
                yaw.unsqueeze(-1),
                pitch_rate.unsqueeze(-1),
                yaw_rate.unsqueeze(-1),
                leg_pos_rel,
                leg_vel,
                wheel_vel,
                leg_lengths,
                leg_length_rates,
                wheel_contacts,
                self.commands,
                self.previous_actions,
            ),
            dim=-1,
        )
        self._single_obs[:] = torch.nan_to_num(self._single_obs, nan=0.0, posinf=self.cfg.max_obs_abs, neginf=-self.cfg.max_obs_abs)
        self._single_obs[:] = torch.clamp(self._single_obs, -self.cfg.max_obs_abs, self.cfg.max_obs_abs)
        self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
        self._obs_history[:, -1, :] = self._single_obs
        return {"policy": self._obs_history.reshape(self.num_envs, -1)}

    def _get_rewards(self) -> torch.Tensor:
        gravity_xy_error = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        upright_reward = torch.exp(-gravity_xy_error / 0.10)

        base_height = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        height_error = torch.square(base_height - self.cfg.target_base_height)
        height_reward = torch.exp(-height_error / 0.01)
        leg_lengths, leg_length_rates, _, _ = self._virtual_leg_state()
        leg_length_error = torch.sum(torch.square(leg_lengths - self.cfg.target_leg_length), dim=1)
        leg_length_reward = torch.exp(-leg_length_error / 0.0025)
        leg_length_symmetry_penalty = torch.square(leg_lengths[:, 0] - leg_lengths[:, 1])
        leg_length_rate_penalty = torch.clamp(
            torch.sum(torch.square(leg_length_rates), dim=1),
            max=self.cfg.max_leg_length_rate_penalty,
        )
        wheel_contacts = self._wheel_contacts()
        wheel_contact_reward = torch.all(wheel_contacts, dim=1).float()
        wheel_air_penalty = torch.sum((~wheel_contacts).float(), dim=1)

        ang_vel_xy_penalty = torch.clamp(
            torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1),
            max=self.cfg.max_ang_vel_penalty,
        )
        joint_vel_penalty = torch.clamp(
            torch.sum(torch.square(self.robot.data.joint_vel[:, self._leg_joint_ids]), dim=1),
            max=self.cfg.max_joint_vel_penalty,
        )
        wheel_vel_penalty = torch.clamp(
            torch.sum(torch.square(self.robot.data.joint_vel[:, self._wheel_joint_ids]), dim=1),
            max=self.cfg.max_wheel_vel_penalty,
        )
        action_rate_penalty = torch.sum(torch.square(self.actions - self.previous_actions), dim=1)

        reward = (
            self.cfg.reward_alive
            + self.cfg.reward_upright * upright_reward
            + self.cfg.reward_height * height_reward
            + self.cfg.reward_leg_length * leg_length_reward
            + self.cfg.reward_wheel_contact * wheel_contact_reward
            - self.cfg.penalty_ang_vel_xy * ang_vel_xy_penalty
            - self.cfg.penalty_joint_vel * joint_vel_penalty
            - self.cfg.penalty_wheel_vel * wheel_vel_penalty
            - self.cfg.penalty_leg_length_symmetry * leg_length_symmetry_penalty
            - self.cfg.penalty_leg_length_rate * leg_length_rate_penalty
            - self.cfg.penalty_wheel_air * wheel_air_penalty
            - self.cfg.penalty_action_rate * action_rate_penalty
            - self.cfg.penalty_termination * self.reset_terminated.float()
        )

        reward = torch.nan_to_num(reward, nan=-self.cfg.max_reward_abs, posinf=self.cfg.max_reward_abs, neginf=-self.cfg.max_reward_abs)
        reward = torch.clamp(reward, -self.cfg.max_reward_abs, self.cfg.max_reward_abs)
        self._episode_reward_sums += reward
        self.extras["log"] = {
            "reward_alive": self.cfg.reward_alive,
            "reward_upright": torch.mean(self.cfg.reward_upright * upright_reward),
            "reward_height": torch.mean(self.cfg.reward_height * height_reward),
            "reward_leg_length": torch.mean(self.cfg.reward_leg_length * leg_length_reward),
            "reward_wheel_contact": torch.mean(self.cfg.reward_wheel_contact * wheel_contact_reward),
            "penalty_ang_vel_xy": torch.mean(self.cfg.penalty_ang_vel_xy * ang_vel_xy_penalty),
            "penalty_joint_vel": torch.mean(self.cfg.penalty_joint_vel * joint_vel_penalty),
            "penalty_wheel_vel": torch.mean(self.cfg.penalty_wheel_vel * wheel_vel_penalty),
            "penalty_leg_length_symmetry": torch.mean(self.cfg.penalty_leg_length_symmetry * leg_length_symmetry_penalty),
            "penalty_leg_length_rate": torch.mean(self.cfg.penalty_leg_length_rate * leg_length_rate_penalty),
            "penalty_wheel_air": torch.mean(self.cfg.penalty_wheel_air * wheel_air_penalty),
            "penalty_action_rate": torch.mean(self.cfg.penalty_action_rate * action_rate_penalty),
            "penalty_termination": torch.mean(self.cfg.penalty_termination * self.reset_terminated.float()),
            "termination_rate": torch.mean(self.reset_terminated.float()),
            "time_out_rate": torch.mean(self.reset_time_outs.float()),
            "mean_leg_length_left": torch.mean(leg_lengths[:, 0]),
            "mean_leg_length_right": torch.mean(leg_lengths[:, 1]),
            "wheel_contact_left": torch.mean(wheel_contacts[:, 0].float()),
            "wheel_contact_right": torch.mean(wheel_contacts[:, 1].float()),
            "episode_reward": torch.mean(self._episode_reward_sums),
        }
        done_debug = self.extras.get("debug_dones", {})
        if done_debug:
            self.extras["log"].update(
                {
                    "done_tilt_rate": torch.mean(done_debug["tilt_bad"].float()),
                    "done_height_rate": torch.mean(done_debug["height_bad"].float()),
                    "done_root_ang_vel_rate": torch.mean(done_debug["root_ang_vel_bad"].float()),
                    "done_root_lin_vel_rate": torch.mean(done_debug["root_lin_vel_bad"].float()),
                    "done_joint_vel_rate": torch.mean(done_debug["joint_vel_bad"].float()),
                    "done_invalid_rate": torch.mean(done_debug["invalid"].float()),
                    "done_past_grace_rate": torch.mean(done_debug["past_grace"].float()),
                }
            )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        tilt_bad = torch.linalg.norm(self.robot.data.projected_gravity_b[:, :2], dim=1) > torch.sin(
            torch.tensor(self.cfg.max_tilt_rad, device=self.device)
        )
        height = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        height_bad = (height < self.cfg.min_base_height) | (height > self.cfg.max_base_height)
        root_ang_vel_bad = torch.linalg.norm(self.robot.data.root_ang_vel_b, dim=1) > self.cfg.max_root_ang_vel
        root_lin_vel_bad = torch.linalg.norm(self.robot.data.root_lin_vel_w, dim=1) > self.cfg.max_root_lin_vel
        joint_vel_bad = torch.max(torch.abs(self.robot.data.joint_vel), dim=1).values > self.cfg.max_joint_vel
        invalid = (
            torch.isnan(self.robot.data.root_state_w).any(dim=1)
            | torch.isnan(self.robot.data.joint_pos).any(dim=1)
            | torch.isnan(self.robot.data.joint_vel).any(dim=1)
            | torch.isinf(self.robot.data.root_state_w).any(dim=1)
            | torch.isinf(self.robot.data.joint_pos).any(dim=1)
            | torch.isinf(self.robot.data.joint_vel).any(dim=1)
        )
        past_grace = self.episode_length_buf > self.cfg.termination_grace_steps
        terminated = invalid | (past_grace & (tilt_bad | height_bad | root_ang_vel_bad | root_lin_vel_bad | joint_vel_bad))
        self.extras["debug_dones"] = {
            "episode_length": self.episode_length_buf.clone(),
            "height": height.clone(),
            "tilt": torch.linalg.norm(self.robot.data.projected_gravity_b[:, :2], dim=1).clone(),
            "root_ang_vel": torch.linalg.norm(self.robot.data.root_ang_vel_b, dim=1).clone(),
            "root_lin_vel": torch.linalg.norm(self.robot.data.root_lin_vel_w, dim=1).clone(),
            "joint_vel": torch.max(torch.abs(self.robot.data.joint_vel), dim=1).values.clone(),
            "tilt_bad": tilt_bad.clone(),
            "height_bad": height_bad.clone(),
            "root_ang_vel_bad": root_ang_vel_bad.clone(),
            "root_lin_vel_bad": root_lin_vel_bad.clone(),
            "joint_vel_bad": joint_vel_bad.clone(),
            "invalid": invalid.clone(),
            "past_grace": past_grace.clone(),
            "terminated": terminated.clone(),
            "time_out": time_out.clone(),
        }
        return terminated, time_out

    def _base_pitch_yaw_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        pitch = wrap_to_pi(pitch)
        yaw = wrap_to_pi(yaw)
        pitch_rate = self.robot.data.root_ang_vel_b[:, 1]
        yaw_rate = self.robot.data.root_ang_vel_b[:, 2]
        return pitch, yaw, pitch_rate, yaw_rate

    def _wheel_contacts(self) -> torch.Tensor:
        left_force = self._contact_force(self.left_wheel_contact)
        right_force = self._contact_force(self.right_wheel_contact)
        return torch.stack((left_force, right_force), dim=-1) > self.cfg.wheel_contact_force_threshold

    def _contact_force(self, sensor) -> torch.Tensor:
        forces = sensor.data.net_forces_w
        if forces is None or forces.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        forces = forces.reshape(self.num_envs, -1, 3)
        return torch.linalg.norm(forces, dim=-1).max(dim=1).values

    def _virtual_leg_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.robot.data.joint_pos[:, self._leg_joint_ids]
        dq = self.robot.data.joint_vel[:, self._leg_joint_ids]

        left_length, left_length_rate, left_phi, left_phi_rate = self._one_virtual_leg_state(
            q[:, 0], q[:, 1], dq[:, 0], dq[:, 1]
        )
        right_length, right_length_rate, right_phi, right_phi_rate = self._one_virtual_leg_state(
            q[:, 2], q[:, 3], dq[:, 2], dq[:, 3]
        )
        return (
            torch.stack((left_length, right_length), dim=-1),
            torch.stack((left_length_rate, right_length_rate), dim=-1),
            torch.stack((left_phi, right_phi), dim=-1),
            torch.stack((left_phi_rate, right_phi_rate), dim=-1),
        )

    def _one_virtual_leg_state(
        self, q_front: torch.Tensor, q_rear: torch.Tensor, dq_front: torch.Tensor, dq_rear: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v_ij_s, v_ij_z = 0.096749440646263, 0.00674587310762054
        v_il_s, v_il_z = -0.0963551866275421, 0.0109403315543567
        v_ip_s, v_ip_z = -0.21357099037258, 0.0245217656620774
        l_jm = 0.115
        l_lm = 0.115
        l_pw = 0.258
        eps = 1.0e-9

        l_il = (v_il_s * v_il_s + v_il_z * v_il_z) ** 0.5
        l_ip = (v_ip_s * v_ip_s + v_ip_z * v_ip_z) ** 0.5
        k_a = l_ip / l_il
        k_b = l_pw / l_lm

        j_s, j_z = self._rotate(q_front, v_ij_s, v_ij_z)
        l_s, l_z = self._rotate(q_rear, v_il_s, v_il_z)
        p_s = k_a * l_s
        p_z = k_a * l_z

        delta_s = j_s - l_s
        delta_z = j_z - l_z
        rho_sq = torch.clamp(delta_s * delta_s + delta_z * delta_z, min=eps)
        c0 = l_lm * l_lm + rho_sq - l_jm * l_jm
        a0 = 2.0 * l_lm * delta_s
        b0 = 2.0 * l_lm * delta_z
        discriminant = torch.clamp(a0 * a0 + b0 * b0 - c0 * c0, min=0.0)
        phi2 = wrap_to_pi(2.0 * torch.atan2(b0 - torch.sqrt(discriminant), a0 + c0))
        m_s = l_s + l_lm * torch.cos(phi2)
        m_z = l_z + l_lm * torch.sin(phi2)

        w_s = p_s + k_b * (m_s - l_s)
        w_z = p_z + k_b * (m_z - l_z)
        leg_length = torch.sqrt(torch.clamp(w_s * w_s + w_z * w_z, min=eps))
        phi0 = torch.atan2(-w_z, w_s)

        u_s = m_s - j_s
        u_z = m_z - j_z
        v_s = m_s - l_s
        v_z = m_z - l_z
        delta_m = u_s * v_z - u_z * v_s
        delta_m_sign = torch.where(delta_m >= 0.0, torch.ones_like(delta_m), -torch.ones_like(delta_m))
        delta_m = torch.where(torch.abs(delta_m) < eps, delta_m_sign * eps, delta_m)

        b1_front = -u_s * j_z + u_z * j_s
        m_s_front = v_z * b1_front / delta_m
        m_z_front = -v_s * b1_front / delta_m

        b2_rear = -v_s * l_z + v_z * l_s
        m_s_rear = -u_z * b2_rear / delta_m
        m_z_rear = u_s * b2_rear / delta_m

        w_s_front = k_b * m_s_front
        w_z_front = k_b * m_z_front
        w_s_rear = k_b * m_s_rear - (k_a - k_b) * l_z
        w_z_rear = k_b * m_z_rear + (k_a - k_b) * l_s

        d_l_front = (w_s * w_s_front + w_z * w_z_front) / leg_length
        d_l_rear = (w_s * w_s_rear + w_z * w_z_rear) / leg_length
        d_phi_front = (w_z * w_s_front - w_s * w_z_front) / (leg_length * leg_length)
        d_phi_rear = (w_z * w_s_rear - w_s * w_z_rear) / (leg_length * leg_length)

        leg_length_rate = d_l_front * dq_front + d_l_rear * dq_rear
        phi0_rate = d_phi_front * dq_front + d_phi_rear * dq_rear
        return leg_length, leg_length_rate, phi0, phi0_rate

    @staticmethod
    def _rotate(angle: torch.Tensor, vector_s: float, vector_z: float) -> tuple[torch.Tensor, torch.Tensor]:
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        return cos_angle * vector_s - sin_angle * vector_z, sin_angle * vector_s + cos_angle * vector_z

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        num_resets = len(env_ids)
        if self.cfg.reset_leg_joint_pos_noise > 0.0:
            joint_pos[:, self._leg_joint_ids] += self._sample_uniform(
                (num_resets, len(self._leg_joint_ids)),
                -self.cfg.reset_leg_joint_pos_noise,
                self.cfg.reset_leg_joint_pos_noise,
            )
        if self.cfg.reset_leg_joint_vel_noise > 0.0:
            joint_vel[:, self._leg_joint_ids] += self._sample_uniform(
                (num_resets, len(self._leg_joint_ids)),
                -self.cfg.reset_leg_joint_vel_noise,
                self.cfg.reset_leg_joint_vel_noise,
            )
        if self.cfg.reset_root_lin_vel_noise > 0.0:
            root_state[:, 7:10] += self._sample_uniform(
                (num_resets, 3),
                -self.cfg.reset_root_lin_vel_noise,
                self.cfg.reset_root_lin_vel_noise,
            )
        if self.cfg.reset_root_ang_vel_noise > 0.0:
            root_state[:, 10:13] += self._sample_uniform(
                (num_resets, 3),
                -self.cfg.reset_root_ang_vel_noise,
                self.cfg.reset_root_ang_vel_noise,
            )

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self._leg_position_targets[env_ids] = joint_pos[:, self._leg_joint_ids]
        self._wheel_velocity_targets[env_ids] = 0.0
        self._passive_position_targets[env_ids] = joint_pos[:, self._passive_joint_ids]
        self._obs_history[env_ids] = 0.0
        self._episode_reward_sums[env_ids] = 0.0
        self.commands[env_ids, 0] = self.cfg.target_base_height

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot.set_joint_position_target(joint_pos[:, self._leg_joint_ids], joint_ids=self._leg_joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos[:, self._passive_joint_ids], joint_ids=self._passive_joint_ids, env_ids=env_ids)
        self.robot.set_joint_velocity_target(
            torch.zeros((len(env_ids), len(self._wheel_joint_ids)), device=self.device),
            joint_ids=self._wheel_joint_ids,
            env_ids=env_ids,
        )

    def _sample_uniform(self, shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
        return torch.empty(shape, device=self.device).uniform_(low, high)
