from __future__ import annotations

import math
from pathlib import Path

import torch
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


TROT_FOOT_PHASE_OFFSETS = (0.0, 0.0, 0.5, 0.5)
TROT_F_MIN = 1.4
TROT_F_MAX = 3.2
TROT_F_GAIN = 0.8
TROT_GATE_SHARPNESS = 10.0


def normalized_contact_forces(
    env,
    sensor_cfg: SceneEntityCfg,
    force_range: tuple[float, float] = (0.0, 50.0),
) -> torch.Tensor:
    """Return contact forces scaled like DreamWaQ privileged observations."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].reshape(env.num_envs, -1)
    force_min, force_max = force_range
    scale = 2.0 / (force_max - force_min)
    shift = 0.5 * (force_max + force_min)
    return (forces - shift) * scale


def base_pose_fall(
    env,
    min_base_height: float = 0.08,
    max_pitch_projected: float = 0.85,
    max_roll_projected: float = 0.75,
    grace_time_s: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base is clearly fallen, without reading contact forces."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])
    low_base = root_pos[:, 2] < min_base_height
    bad_attitude = (gravity_xy[:, 0] > max_pitch_projected) | (gravity_xy[:, 1] > max_roll_projected)
    past_grace = env.episode_length_buf.to(torch.float32) * env.step_dt > grace_time_s
    return past_grace & (low_base | bad_attitude)


def trot_phase(
    env,
    command_name: str = "base_velocity",
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    command_threshold: float = 0.05,
):
    """Return sin/cos phase generated from command-dependent trot frequency."""
    phase = _trot_phase_scalar(env, command_name=command_name, f_min=f_min, f_max=f_max, gain=gain)
    active = _active_command_mask(env, command_name, command_threshold)
    phase = torch.where(active > 0.0, phase, torch.zeros_like(phase))
    angle = 2.0 * math.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)


def _trot_phase_scalar(
    env,
    command_name: str = "base_velocity",
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
):
    command = env.command_manager.get_command(command_name)
    planar_speed = torch.linalg.norm(command[:, :2], dim=1)
    yaw_speed = 0.35 * command[:, 2].abs()
    frequency = torch.clamp(f_min + gain * (planar_speed + yaw_speed), min=f_min, max=f_max)
    return torch.remainder(env.episode_length_buf.to(torch.float32) * env.step_dt * frequency, 1.0)


def _trot_swing_stance_gates(
    env,
    command_name: str,
    f_min: float,
    f_max: float,
    gain: float,
    phase_offsets: tuple[float, ...],
    gate_sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = _trot_phase_scalar(env, command_name=command_name, f_min=f_min, f_max=f_max, gain=gain)
    offsets = torch.tensor(phase_offsets, dtype=torch.float32, device=phase.device)
    leg_phase = torch.remainder(phase.unsqueeze(1) + offsets.unsqueeze(0), 1.0)
    swing_gate = torch.sigmoid(gate_sharpness * torch.sin(2.0 * math.pi * leg_phase))
    return swing_gate, 1.0 - swing_gate


def _active_command_mask(env, command_name: str, command_threshold: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(command[:, :3], dim=1) > command_threshold).to(torch.float32)


def _foot_contact(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return torch.max(torch.linalg.norm(net_forces, dim=-1), dim=1)[0] > threshold


def trot_phase_contact_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    contact_threshold: float = 1.0,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Reward stance feet in contact and swing feet out of contact for a trot phase."""
    del asset_cfg
    swing_gate, stance_gate = _trot_swing_stance_gates(
        env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness
    )
    contact = _foot_contact(env, sensor_cfg, contact_threshold).to(torch.float32)
    active = _active_command_mask(env, command_name, command_threshold)
    phase_score = stance_gate * contact + swing_gate * (1.0 - contact)
    return active * torch.mean(phase_score, dim=1)


def swing_foot_clearance_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    target_height: float = 0.08,
    std: float = 0.35,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Reward swing feet for clearing the ground instead of skimming it."""
    asset = env.scene[asset_cfg.name]
    swing_gate, _ = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    active = _active_command_mask(env, command_name, command_threshold).unsqueeze(1)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    clearance_deficit = torch.clamp(target_height - foot_z, min=0.0) / target_height
    weighted_error = active * swing_gate * torch.square(clearance_deficit)
    normalizer = torch.clamp(torch.sum(active * swing_gate, dim=1), min=1.0)
    return active.squeeze(1) * torch.exp(-torch.sum(weighted_error, dim=1) / (normalizer * std))


def swing_foot_drag_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    contact_threshold: float = 1.0,
    clearance: float = 0.055,
    velocity_scale: float = 3.0,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Penalize swing feet that touch or skim the ground while moving forward."""
    asset = env.scene[asset_cfg.name]
    swing_gate, _ = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    active = _active_command_mask(env, command_name, command_threshold).unsqueeze(1)
    contact = _foot_contact(env, sensor_cfg, contact_threshold).to(torch.float32)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    foot_planar_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    clearance_deficit = torch.clamp(clearance - foot_z, min=0.0) / clearance
    skim = clearance_deficit * torch.tanh(velocity_scale * foot_planar_vel)
    return torch.sum(active * swing_gate * (contact + skim), dim=1)


def phase_swing_foot_drag_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    clearance: float = 0.055,
    velocity_scale: float = 3.0,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Penalize swing feet that skim the ground without reading foot contact forces."""
    asset = env.scene[asset_cfg.name]
    swing_gate, _ = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    active = _active_command_mask(env, command_name, command_threshold).unsqueeze(1)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    foot_planar_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    clearance_deficit = torch.clamp(clearance - foot_z, min=0.0) / clearance
    skim = clearance_deficit * torch.tanh(velocity_scale * foot_planar_vel)
    return torch.sum(active * swing_gate * skim, dim=1)


def phase_swing_foot_lift_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    min_height: float = 0.012,
    target_height: float = 0.042,
    foot_radius: float = 0.012,
    velocity_scale: float = 3.0,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Reward phase-swing feet for getting clearly above the floor."""
    asset = env.scene[asset_cfg.name]
    swing_gate, _ = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    active = _active_command_mask(env, command_name, command_threshold).unsqueeze(1)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    foot_clearance = torch.clamp(foot_z - foot_radius, min=0.0)
    target_clearance = max(target_height - foot_radius, 1.0e-6)
    min_clearance = min(max(min_height - foot_radius, 0.0), target_clearance)
    early_lift = 0.35 * torch.clamp(foot_clearance / max(min_clearance, 1.0e-6), 0.0, 1.0)
    high_lift = 0.65 * torch.clamp(
        (foot_clearance - min_clearance) / max(target_clearance - min_clearance, 1.0e-6),
        0.0,
        1.0,
    )
    lift_score = torch.clamp(early_lift + high_lift, 0.0, 1.0)
    foot_planar_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    moving_weight = torch.tanh(velocity_scale * foot_planar_vel)
    weighted_score = active * swing_gate * lift_score
    normalizer = torch.clamp(torch.sum(active * swing_gate, dim=1), min=1.0)
    return active.squeeze(1) * torch.sum(weighted_score * moving_weight, dim=1) / normalizer


def phase_swing_foot_up_velocity_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    target_up_speed: float = 0.18,
    foot_radius: float = 0.012,
    max_clearance: float = 0.060,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Reward swing feet for moving upward while they are still near the floor."""
    asset = env.scene[asset_cfg.name]
    swing_gate, _ = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    active = _active_command_mask(env, command_name, command_threshold).unsqueeze(1)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    foot_clearance = torch.clamp(foot_z - foot_radius, min=0.0)
    near_floor = torch.clamp(1.0 - foot_clearance / max(max_clearance, 1.0e-6), min=0.0, max=1.0)
    up_speed = torch.clamp(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2], min=0.0, max=target_up_speed)
    up_score = up_speed / max(target_up_speed, 1.0e-6)
    normalizer = torch.clamp(torch.sum(active * swing_gate, dim=1), min=1.0)
    return torch.sum(active * swing_gate * near_floor * up_score, dim=1) / normalizer


def phase_swing_foot_low_height_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    foot_radius: float = 0.012,
    clearance: float = 0.018,
    velocity_scale: float = 6.0,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Penalize phase-swing feet that move while staying close to the floor."""
    asset = env.scene[asset_cfg.name]
    swing_gate, _ = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    active = _active_command_mask(env, command_name, command_threshold).unsqueeze(1)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    foot_clearance = torch.clamp(foot_z - foot_radius, min=0.0)
    low_height = torch.clamp((clearance - foot_clearance) / max(clearance, 1.0e-6), min=0.0, max=1.0)
    foot_planar_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    moving = torch.tanh(velocity_scale * foot_planar_vel)
    normalizer = torch.clamp(torch.sum(active * swing_gate, dim=1), min=1.0)
    return torch.sum(active * swing_gate * low_height * moving, dim=1) / normalizer


def phase_swing_foot_low_height_pose_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    foot_radius: float = 0.012,
    clearance: float = 0.020,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Penalize low phase-swing feet even when the policy tries to slow-slide them."""
    asset = env.scene[asset_cfg.name]
    swing_gate, _ = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    active = _active_command_mask(env, command_name, command_threshold).unsqueeze(1)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    foot_clearance = torch.clamp(foot_z - foot_radius, min=0.0)
    low_height = torch.clamp((clearance - foot_clearance) / max(clearance, 1.0e-6), min=0.0, max=1.0)
    normalizer = torch.clamp(torch.sum(active * swing_gate, dim=1), min=1.0)
    return torch.sum(active * swing_gate * low_height, dim=1) / normalizer


def moving_foot_low_height_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    foot_radius: float = 0.012,
    clearance: float = 0.018,
    velocity_scale: float = 6.0,
) -> torch.Tensor:
    """Penalize moving feet that stay close to the floor, independent of phase."""
    asset = env.scene[asset_cfg.name]
    active = _active_command_mask(env, command_name, command_threshold).unsqueeze(1)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    foot_clearance = torch.clamp(foot_z - foot_radius, min=0.0)
    low_height = torch.clamp((clearance - foot_clearance) / max(clearance, 1.0e-6), min=0.0, max=1.0)
    foot_planar_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    moving = torch.tanh(velocity_scale * foot_planar_vel)
    return torch.mean(active * low_height * moving, dim=1)


def commanded_feet_air_time_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    threshold: float = 0.08,
    max_air_time: float = 0.22,
) -> torch.Tensor:
    """Reward nonzero command steps with useful air time, including yaw-only commands."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    useful_air_time = torch.clamp(last_air_time, max=max_air_time) - threshold
    command = env.command_manager.get_command(command_name)
    active = (torch.linalg.norm(command[:, :3], dim=1) > command_threshold).to(torch.float32)
    return active * torch.sum(useful_air_time * first_contact, dim=1)


def contact_foot_slide_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    contact_threshold: float = 0.5,
) -> torch.Tensor:
    """Penalize any contacted foot sliding laterally under a nonzero command."""
    asset = env.scene[asset_cfg.name]
    active = _active_command_mask(env, command_name, command_threshold).unsqueeze(1)
    contact = _foot_contact(env, sensor_cfg, contact_threshold).to(torch.float32)
    foot_planar_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    return torch.mean(active * contact * foot_planar_vel, dim=1)


def stand_still_foot_vel_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize foot jitter when the commanded base velocity is zero."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.norm(command[:, :3], dim=1) < command_threshold).to(torch.float32)
    foot_planar_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    return standing * torch.sum(torch.square(foot_planar_vel), dim=1)


def yaw_or_backward_low_foot_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_backward_command: float = 0.05,
    min_yaw_command_abs: float = 0.12,
    foot_radius: float = 0.012,
    clearance: float = 0.016,
    velocity_scale: float = 6.0,
) -> torch.Tensor:
    """Penalize low moving feet during backup and yaw commands."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    active = ((command[:, 0] < -min_backward_command) | (torch.abs(command[:, 2]) > min_yaw_command_abs)).to(
        torch.float32
    )
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    foot_clearance = torch.clamp(foot_z - foot_radius, min=0.0)
    low_height = torch.clamp((clearance - foot_clearance) / max(clearance, 1.0e-6), min=0.0, max=1.0)
    foot_planar_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    moving = torch.tanh(velocity_scale * foot_planar_vel)
    return active * torch.mean(low_height * moving, dim=1)


def yaw_or_backward_swing_foot_lift_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_backward_command: float = 0.05,
    min_yaw_command_abs: float = 0.12,
    foot_radius: float = 0.012,
    min_height: float = 0.024,
    target_height: float = 0.058,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Reward clear swing feet specifically for backing-up and yaw commands."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    active = ((command[:, 0] < -min_backward_command) | (torch.abs(command[:, 2]) > min_yaw_command_abs)).to(
        torch.float32
    )
    swing_gate, _ = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    foot_clearance = torch.clamp(foot_z - foot_radius, min=0.0)
    min_clearance = max(min_height - foot_radius, 0.0)
    target_clearance = max(target_height - foot_radius, min_clearance + 1.0e-6)
    lift_score = torch.clamp((foot_clearance - min_clearance) / (target_clearance - min_clearance), 0.0, 1.0)
    normalizer = torch.clamp(torch.sum(swing_gate, dim=1), min=1.0)
    return active * torch.sum(swing_gate * lift_score, dim=1) / normalizer


def yaw_or_backward_swing_foot_low_height_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_backward_command: float = 0.05,
    min_yaw_command_abs: float = 0.12,
    foot_radius: float = 0.012,
    clearance: float = 0.026,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Penalize backing-up/yaw swing feet that remain near the floor."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    active = ((command[:, 0] < -min_backward_command) | (torch.abs(command[:, 2]) > min_yaw_command_abs)).to(
        torch.float32
    )
    swing_gate, _ = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    foot_clearance = torch.clamp(foot_z - foot_radius, min=0.0)
    low_height = torch.clamp((clearance - foot_clearance) / max(clearance, 1.0e-6), min=0.0, max=1.0)
    normalizer = torch.clamp(torch.sum(swing_gate, dim=1), min=1.0)
    return active * torch.sum(swing_gate * low_height, dim=1) / normalizer


def swing_foot_height_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    max_height: float = 0.075,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Penalize exaggerated swing height after the foot has enough clearance."""
    asset = env.scene[asset_cfg.name]
    swing_gate, _ = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    active = _active_command_mask(env, command_name, command_threshold).unsqueeze(1)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    excess = torch.clamp(foot_z - max_height, min=0.0) / max_height
    normalizer = torch.clamp(torch.sum(active * swing_gate, dim=1), min=1.0)
    return torch.sum(active * swing_gate * torch.square(excess), dim=1) / normalizer


def stance_foot_slip_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    contact_threshold: float = 1.0,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Penalize planted feet sliding across the ground."""
    asset = env.scene[asset_cfg.name]
    _, stance_gate = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    active = _active_command_mask(env, command_name, command_threshold).unsqueeze(1)
    stance_weight = torch.where(active > 0.0, stance_gate, torch.ones_like(stance_gate))
    contact = _foot_contact(env, sensor_cfg, contact_threshold).to(torch.float32)
    foot_planar_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    return torch.sum(stance_weight * contact * foot_planar_vel, dim=1)


def phase_stance_foot_slip_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Penalize phase-stance feet sliding without reading foot contact forces."""
    asset = env.scene[asset_cfg.name]
    _, stance_gate = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    active = _active_command_mask(env, command_name, command_threshold).unsqueeze(1)
    stance_weight = torch.where(active > 0.0, stance_gate, torch.ones_like(stance_gate))
    foot_planar_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    normalizer = torch.clamp(torch.sum(stance_weight, dim=1), min=1.0)
    return torch.sum(stance_weight * foot_planar_vel, dim=1) / normalizer


def trot_pair_timing_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward FL/RR and FR/RL diagonal pairs for synchronized trot contacts."""
    contact = _foot_contact(env, sensor_cfg, contact_threshold).to(torch.float32)
    active = _active_command_mask(env, command_name, command_threshold)
    diagonal_sync = 1.0 - 0.5 * (torch.abs(contact[:, 0] - contact[:, 1]) + torch.abs(contact[:, 2] - contact[:, 3]))
    off_diagonal_async = 0.25 * (
        torch.abs(contact[:, 0] - contact[:, 2])
        + torch.abs(contact[:, 0] - contact[:, 3])
        + torch.abs(contact[:, 1] - contact[:, 2])
        + torch.abs(contact[:, 1] - contact[:, 3])
    )
    return active * diagonal_sync * off_diagonal_async


def stand_still_joint_deviation_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize joint motion away from default pose when the velocity command is zero."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.norm(command[:, :3], dim=1) < command_threshold).to(torch.float32)
    joint_error = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids])
    return standing * torch.sum(joint_error, dim=1)


def stand_still_action_penalty(
    env,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize residual actions when the velocity command is zero."""
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.norm(command[:, :3], dim=1) < command_threshold).to(torch.float32)
    return standing * torch.sum(torch.square(env.action_manager.action), dim=1)


def stand_still_joint_vel_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize joint chatter when the commanded base velocity is zero."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.norm(command[:, :3], dim=1) < command_threshold).to(torch.float32)
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return standing * torch.sum(torch.square(joint_vel), dim=1)


def joint_deviation_l2(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    moving_scale: float = 1.0,
    standing_scale: float = 2.0,
    joint_weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Penalize large posture excursions from the nominal standing pose."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    standing = torch.linalg.norm(command[:, :3], dim=1) < command_threshold
    scale = torch.where(
        standing,
        torch.full_like(command[:, 0], standing_scale),
        torch.full_like(command[:, 0], moving_scale),
    )
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    if joint_weights is not None:
        weights = torch.tensor(joint_weights, dtype=joint_error.dtype, device=joint_error.device)
        if weights.numel() != joint_error.shape[1]:
            raise ValueError(f"joint_weights has {weights.numel()} values, expected {joint_error.shape[1]}.")
        joint_error = joint_error * weights.unsqueeze(0)
    return scale * torch.sum(torch.square(joint_error), dim=1)


def commanded_backward_joint_deviation_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_backward_command: float = 0.06,
    allowances: float | tuple[float, ...] = 0.35,
) -> torch.Tensor:
    """Penalize front-leg over-folding specifically while backing up."""
    asset = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    active = (command_x < -min_backward_command).to(torch.float32)

    joint_error = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    allowance_values = torch.as_tensor(allowances, dtype=joint_error.dtype, device=joint_error.device)
    if allowance_values.ndim == 0:
        allowance_values = allowance_values.expand(joint_error.shape[1])
    if allowance_values.numel() != joint_error.shape[1]:
        raise ValueError(f"allowances has {allowance_values.numel()} values, expected {joint_error.shape[1]}.")
    excess = torch.clamp(joint_error - allowance_values.unsqueeze(0), min=0.0)
    return active * torch.sum(torch.square(excess), dim=1)


def pitch_abs_l1(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute base pitch angle."""
    asset = env.scene[asset_cfg.name]
    _, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.abs(pitch)


def terrain_levels_vel_public_init_guard(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """IsaacLab velocity curriculum with the first pre-reset update skipped.

    DreamWaQ/legged_gym computes the game-inspired terrain curriculum from reset-time
    root states that are already aligned with each environment origin. IsaacLab calls
    curriculum terms before reset events, so the first reset can see construction-time
    root states and incorrectly advance every environment by one level.
    """
    terrain = env.scene.terrain
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if not getattr(env, "_gogo_public_like_terrain_curriculum_started", False):
        env._gogo_public_like_terrain_curriculum_started = True
        if terrain_levels is None:
            return torch.zeros((), device=env.device)
        return torch.mean(terrain_levels.float())
    if getattr(terrain, "terrain_origins", None) is None:
        return torch.zeros((), device=env.device)

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def motor_power_percent(
    env,
    max_joint_power: float = 250.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return per-joint mechanical power usage normalized to a configured motor limit."""
    asset = env.scene[asset_cfg.name]
    joint_power = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids])
    return torch.clamp(joint_power / max_joint_power, 0.0, 1.0)


def joint_power_l1(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return summed absolute mechanical joint power."""
    asset = env.scene[asset_cfg.name]
    joint_power = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids])
    return torch.sum(joint_power, dim=1)


def paper_penalty_schedule_multiplier(
    env,
    half_life_iterations: float = 500.0,
    ppo_steps_per_iteration: int = 24,
) -> torch.Tensor:
    """Ramp policy penalties using the half-life schedule from the sim-to-real paper."""
    device = torch.device(getattr(env, "device", "cpu"))
    num_envs = int(getattr(env, "num_envs", 1))
    if half_life_iterations <= 0.0 or ppo_steps_per_iteration <= 0:
        return torch.ones(num_envs, dtype=torch.float32, device=device)

    step_counter = getattr(env, "common_step_counter", 0)
    step_counter = torch.as_tensor(step_counter, dtype=torch.float32, device=device)
    iteration = step_counter / float(ppo_steps_per_iteration)
    scale = 1.0 - torch.pow(torch.tensor(0.5, dtype=torch.float32, device=device), iteration / float(half_life_iterations))
    return scale.reshape(1).expand(num_envs)


def paper_equivalent_energy_proxy(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    torque_limit: float = 33.5,
    velocity_limit: float = 21.0,
    torque_weight: float = 0.7,
    positive_power_weight: float = 0.3,
    clip_max: float = 2.0,
) -> torch.Tensor:
    """Normalized A/B equivalent energy proxy for A1 without motor electrical constants."""
    asset = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    torque_norm = torque / max(float(torque_limit), 1.0e-6)
    velocity_norm = velocity / max(float(velocity_limit), 1.0e-6)
    torque_cost = torch.mean(torch.square(torque_norm), dim=1)
    positive_power_cost = torch.mean(torch.clamp(torque_norm * velocity_norm, min=0.0), dim=1)
    proxy = float(torque_weight) * torque_cost + float(positive_power_weight) * positive_power_cost
    if clip_max > 0.0:
        proxy = torch.clamp(proxy, min=0.0, max=float(clip_max))
    return proxy


def paper_scheduled_foot_touchdown_velocity(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    history_length: int = 3,
    half_life_iterations: float = 500.0,
    ppo_steps_per_iteration: int = 24,
) -> torch.Tensor:
    """Penalize maximum foot speed over the last steps when a foot touches down."""
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    speed = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)
    history_length = max(int(history_length), 1)
    if history_length > 1:
        attr_name = "_gogo_paper_ftd_speed_history"
        history = getattr(env, attr_name, None)
        expected_shape = (speed.shape[0], history_length, speed.shape[1])
        if history is None or tuple(history.shape) != expected_shape or history.device != speed.device:
            history = speed.unsqueeze(1).repeat(1, history_length, 1)
        else:
            history = torch.cat((history[:, 1:, :], speed.unsqueeze(1)), dim=1)
        setattr(env, attr_name, history.detach())
        speed = torch.max(history, dim=1).values

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].to(dtype=speed.dtype)
    schedule = paper_penalty_schedule_multiplier(
        env,
        half_life_iterations=half_life_iterations,
        ppo_steps_per_iteration=ppo_steps_per_iteration,
    ).to(device=speed.device, dtype=speed.dtype)
    return torch.sum(speed * first_contact, dim=1) * schedule


def paper_collision_indicator(
    env,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Single paper-style collision indicator for joint limits or thigh contacts."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    below_limit = joint_pos < joint_limits[..., 0]
    above_limit = joint_pos > joint_limits[..., 1]
    joint_limit_contact = torch.any(below_limit | above_limit, dim=1)

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    thigh_contact = torch.max(torch.linalg.norm(net_contact_forces, dim=-1), dim=1)[0] > threshold
    thigh_contact = torch.any(thigh_contact, dim=1)
    return (joint_limit_contact | thigh_contact).to(dtype=joint_pos.dtype)


def power_distribution_variance(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize uneven motor power usage as in DreamWaQ."""
    asset = env.scene[asset_cfg.name]
    joint_power = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids])
    return torch.var(joint_power, dim=1, unbiased=False)


def computed_torque_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_limit: float = 1.0,
) -> torch.Tensor:
    """Penalize actuator torque demand before simulation effort clipping."""
    asset = env.scene[asset_cfg.name]
    torque = asset.data.computed_torque[:, asset_cfg.joint_ids]
    if normalize_by_limit > 0.0:
        torque = torque / normalize_by_limit
    return torch.mean(torch.square(torque), dim=1)


def computed_torque_limit_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_limit: float = 0.42,
    hard_limit: float = 0.60,
    max_excess_ratio: float = 4.0,
    joint_weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Penalize torque demand above a soft limit before effort clipping hides it."""
    asset = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.computed_torque[:, asset_cfg.joint_ids])
    scale = max(float(hard_limit), 1e-6)
    excess = torch.clamp((torque - float(soft_limit)) / scale, min=0.0, max=float(max_excess_ratio))
    penalty = torch.square(excess)
    if joint_weights is not None:
        weights = torch.as_tensor(joint_weights, dtype=penalty.dtype, device=penalty.device)
        if weights.numel() != penalty.shape[1]:
            raise ValueError(f"joint_weights has {weights.numel()} values, expected {penalty.shape[1]}.")
        penalty = penalty * weights.unsqueeze(0)
    return torch.mean(penalty, dim=1)


def applied_torque_soft_limit_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_limit: float = 0.48,
    hard_limit: float = 0.60,
    max_excess_ratio: float = 1.0,
    joint_weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Penalize actual clipped actuator torque as it approaches the hardware limit."""
    asset = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    scale = max(float(hard_limit) - float(soft_limit), 1e-6)
    excess = torch.clamp((torque - float(soft_limit)) / scale, min=0.0, max=float(max_excess_ratio))
    penalty = torch.square(excess)
    if joint_weights is not None:
        weights = torch.as_tensor(joint_weights, dtype=penalty.dtype, device=penalty.device)
        if weights.numel() != penalty.shape[1]:
            raise ValueError(f"joint_weights has {weights.numel()} values, expected {penalty.shape[1]}.")
        penalty = penalty * weights.unsqueeze(0)
    return torch.mean(penalty, dim=1)


def applied_torque_near_limit_fraction(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.54,
) -> torch.Tensor:
    """Return the fraction of joints currently close to actuator saturation."""
    asset = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    return torch.mean((torque >= float(threshold)).to(dtype=torch.float32), dim=1)


def applied_torque_saturation_termination(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.595,
    max_saturated_joints: int = 6,
    grace_time_s: float = 0.35,
) -> torch.Tensor:
    """Terminate when too many joints hit the actuator limit at the same instant."""
    asset = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    saturated_joints = torch.sum(torque >= float(threshold), dim=1)
    past_grace = env.episode_length_buf.to(torch.float32) * env.step_dt > float(grace_time_s)
    return past_grace & (saturated_joints >= int(max_saturated_joints))


def applied_torque_any_saturation_termination(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.599,
    grace_time_s: float = 0.50,
) -> torch.Tensor:
    """Terminate when any joint reaches hard saturation after a short startup grace period."""
    asset = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    past_grace = env.episode_length_buf.to(torch.float32) * env.step_dt > float(grace_time_s)
    return past_grace & torch.any(torque >= float(threshold), dim=1)


def dreamwaq_foot_clearance_l2(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    desired_clearance: float = 0.08,
) -> torch.Tensor:
    """DreamWaQ Table-I foot-clearance penalty using terrain-relative foot height.

    This implements sum_k (p_des_f,z,k - p_f,z,k)^2 * ||v_f,xy,k||.
    The desired foot height is approximated as the nearest height-scan terrain
    sample plus a fixed clearance. This term is reward-only privileged
    information and does not change the proprioceptive actor observation.
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    ray_hits_w = sensor.data.ray_hits_w

    foot_xy = foot_pos_w[..., :2]
    ray_xy = ray_hits_w[..., :2]
    nearest_ray_ids = torch.argmin(
        torch.sum(torch.square(foot_xy.unsqueeze(2) - ray_xy.unsqueeze(1)), dim=-1),
        dim=-1,
    )
    ray_z = ray_hits_w[..., 2].unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1)
    terrain_z = torch.gather(ray_z, dim=2, index=nearest_ray_ids.unsqueeze(-1)).squeeze(-1)

    desired_foot_z = terrain_z + float(desired_clearance)
    foot_height_error = torch.square(desired_foot_z - foot_pos_w[..., 2])
    foot_speed_xy = torch.linalg.norm(foot_vel_w[..., :2], dim=-1)
    return torch.sum(foot_height_error * foot_speed_xy, dim=1)


def action_smoothness_l2(env) -> torch.Tensor:
    """Second-order action smoothness penalty."""
    current_action = env.action_manager.action
    previous_action = getattr(env.action_manager, "prev_action", None)
    if previous_action is None:
        return torch.zeros(current_action.shape[0], device=current_action.device)
    previous_previous_action = getattr(env, "_gogo_prev_prev_action", None)
    if previous_previous_action is None or previous_previous_action.shape != current_action.shape:
        previous_previous_action = previous_action
    value = torch.sum(torch.square(current_action - 2.0 * previous_action + previous_previous_action), dim=1)
    env._gogo_prev_prev_action = previous_action.detach().clone()
    return value


def base_lin_acc(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root linear acceleration expressed in the root frame."""
    asset = env.scene[asset_cfg.name]
    root_acc_w = asset.data.body_lin_acc_w[:, 0, :]
    return math_utils.quat_apply_inverse(asset.data.root_quat_w, root_acc_w)


def last_action_with_reset_seed(env) -> torch.Tensor:
    """Return recorded handoff action on the first reset step, then normal last actions."""
    action = env.action_manager.action
    seed = getattr(env, "_gogo_reset_last_action", None)
    use_seed = getattr(env, "_gogo_use_reset_last_action", None)
    if seed is None or use_seed is None:
        return action
    first_step = env.episode_length_buf == 0
    use_recorded_action = use_seed & first_step
    return torch.where(use_recorded_action.unsqueeze(1), seed, action)


def reset_stair_climb_root_state(
    env,
    env_ids: torch.Tensor,
    first_riser_x: float,
    step_height_range: tuple[float, float],
    approach_probability: float = 0.65,
    approach_x_range: tuple[float, float] = (-0.20, 0.18),
    near_step_x_range: tuple[float, float] = (0.58, 0.88),
    y_range: tuple[float, float] = (-0.06, 0.06),
    yaw_range: tuple[float, float] = (-0.08, 0.08),
    base_height: float = 0.31,
    step_base_clearance: float = 0.29,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    joint_position_scale_range: tuple[float, float] = (0.98, 1.02),
    joint_velocity_range: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset root either on the approach or near the first step for staged stair learning."""
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    num_envs = len(env_ids)
    device = asset.device

    stage_sample = torch.rand(num_envs, device=device)
    use_approach = stage_sample < approach_probability
    approach_x = math_utils.sample_uniform(
        approach_x_range[0], approach_x_range[1], (num_envs,), device=device
    )
    near_step_x = math_utils.sample_uniform(
        near_step_x_range[0], near_step_x_range[1], (num_envs,), device=device
    )
    local_x = torch.where(use_approach, approach_x, near_step_x)
    local_y = math_utils.sample_uniform(y_range[0], y_range[1], (num_envs,), device=device)
    local_z = torch.where(
        use_approach,
        torch.full((num_envs,), base_height, device=device),
        _terrain_level_step_height(env, step_height_range)[env_ids] + step_base_clearance,
    )
    positions = env.scene.env_origins[env_ids] + torch.stack((local_x, local_y, local_z), dim=-1)

    yaw = math_utils.sample_uniform(yaw_range[0], yaw_range[1], (num_envs,), device=device)
    zeros = torch.zeros_like(yaw)
    orientations = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

    if velocity_range is None:
        velocity_range = {}
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=device)
    velocities = root_states[:, 7:13] + math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (num_envs, 6), device=device
    )

    asset.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos *= math_utils.sample_uniform(
        joint_position_scale_range[0], joint_position_scale_range[1], joint_pos.shape, device=device
    )
    joint_vel += math_utils.sample_uniform(
        joint_velocity_range[0], joint_velocity_range[1], joint_vel.shape, device=device
    )
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    _set_reset_last_action_seed(env, env_ids, None)


_SWITCH_STATE_CACHE: dict[str, dict[str, torch.Tensor]] = {}


def _set_reset_last_action_seed(env, env_ids: torch.Tensor, last_action: torch.Tensor | None) -> None:
    action_dim = env.action_manager.total_action_dim
    if not hasattr(env, "_gogo_reset_last_action"):
        env._gogo_reset_last_action = torch.zeros((env.num_envs, action_dim), dtype=torch.float32, device=env.device)
        env._gogo_use_reset_last_action = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if last_action is None:
        env._gogo_reset_last_action[env_ids] = 0.0
        env._gogo_use_reset_last_action[env_ids] = False
        return

    if last_action.shape[-1] != action_dim:
        raise ValueError(f"Recorded last_action has dim {last_action.shape[-1]}, expected {action_dim}.")
    env._gogo_reset_last_action[env_ids] = last_action.to(device=env.device, dtype=torch.float32)
    env._gogo_use_reset_last_action[env_ids] = True


def reset_stair_climb_from_switch_state_dataset(
    env,
    env_ids: torch.Tensor,
    dataset_path: str,
    x_jitter: tuple[float, float] = (-0.03, 0.03),
    y_jitter: tuple[float, float] = (-0.03, 0.03),
    yaw_jitter: tuple[float, float] = (-0.06, 0.06),
    velocity_scale_range: tuple[float, float] = (0.85, 1.15),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the robot to recorded approach states so finisher training sees real switch conditions."""
    asset = env.scene[asset_cfg.name]
    dataset = _load_switch_state_dataset(dataset_path, device=asset.device)
    num_envs = len(env_ids)
    sample_ids = torch.randint(dataset["root_pos_local"].shape[0], (num_envs,), device=asset.device)

    root_pos_local = dataset["root_pos_local"][sample_ids].clone()
    root_pos_local[:, 0] += math_utils.sample_uniform(x_jitter[0], x_jitter[1], (num_envs,), device=asset.device)
    root_pos_local[:, 1] += math_utils.sample_uniform(y_jitter[0], y_jitter[1], (num_envs,), device=asset.device)

    yaw_delta = math_utils.sample_uniform(yaw_jitter[0], yaw_jitter[1], (num_envs,), device=asset.device)
    zeros = torch.zeros_like(yaw_delta)
    yaw_offset = math_utils.quat_from_euler_xyz(zeros, zeros, yaw_delta)
    root_quat = math_utils.quat_mul(yaw_offset, dataset["root_quat_w"][sample_ids])

    root_vel = dataset["root_vel_w"][sample_ids].clone()
    velocity_scale = math_utils.sample_uniform(
        velocity_scale_range[0], velocity_scale_range[1], (num_envs, 1), device=asset.device
    )
    root_vel *= velocity_scale

    root_pos = env.scene.env_origins[env_ids] + root_pos_local
    asset.write_root_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(
        dataset["joint_pos"][sample_ids],
        dataset["joint_vel"][sample_ids],
        env_ids=env_ids,
    )
    recorded_last_action = dataset.get("last_action", None)
    _set_reset_last_action_seed(
        env,
        env_ids,
        None if recorded_last_action is None else recorded_last_action[sample_ids],
    )


def reset_stair_climb_mixed_switch_and_near_step(
    env,
    env_ids: torch.Tensor,
    dataset_path: str,
    switch_probability: float,
    first_riser_x: float,
    step_height_range: tuple[float, float],
    near_step_x_range: tuple[float, float] = (0.70, 0.90),
    y_range: tuple[float, float] = (-0.06, 0.06),
    yaw_range: tuple[float, float] = (-0.08, 0.08),
    step_base_clearance: float = 0.29,
    x_jitter: tuple[float, float] = (-0.03, 0.03),
    y_jitter: tuple[float, float] = (-0.03, 0.03),
    yaw_jitter: tuple[float, float] = (-0.06, 0.06),
    velocity_scale_range: tuple[float, float] = (0.85, 1.15),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset from a mix of real switch states and the easier near-step finisher distribution."""
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = asset.device
    use_switch = torch.rand(num_envs, device=device) < switch_probability

    if use_switch.any():
        switch_env_ids = env_ids[use_switch]
        reset_stair_climb_from_switch_state_dataset(
            env,
            switch_env_ids,
            dataset_path=dataset_path,
            x_jitter=x_jitter,
            y_jitter=y_jitter,
            yaw_jitter=yaw_jitter,
            velocity_scale_range=velocity_scale_range,
            asset_cfg=asset_cfg,
        )

    if (~use_switch).any():
        near_env_ids = env_ids[~use_switch]
        root_states = asset.data.default_root_state[near_env_ids].clone()
        near_count = len(near_env_ids)
        local_x = math_utils.sample_uniform(near_step_x_range[0], near_step_x_range[1], (near_count,), device=device)
        local_y = math_utils.sample_uniform(y_range[0], y_range[1], (near_count,), device=device)
        local_z = _terrain_level_step_height(env, step_height_range)[near_env_ids] + step_base_clearance
        positions = env.scene.env_origins[near_env_ids] + torch.stack((local_x, local_y, local_z), dim=-1)
        yaw = math_utils.sample_uniform(yaw_range[0], yaw_range[1], (near_count,), device=device)
        zeros = torch.zeros_like(yaw)
        orientations = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        velocities = torch.zeros_like(root_states[:, 7:13])
        asset.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=near_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=near_env_ids)
        _set_reset_last_action_seed(env, near_env_ids, None)


def reset_stair_climb_mixed_approach_switch_and_near_step(
    env,
    env_ids: torch.Tensor,
    dataset_path: str,
    approach_probability: float,
    switch_probability: float,
    first_riser_x: float,
    step_height_range: tuple[float, float],
    approach_x_range: tuple[float, float] = (-0.20, 0.18),
    near_step_x_range: tuple[float, float] = (0.70, 0.90),
    y_range: tuple[float, float] = (-0.06, 0.06),
    yaw_range: tuple[float, float] = (-0.08, 0.08),
    base_height: float = 0.31,
    step_base_clearance: float = 0.29,
    x_jitter: tuple[float, float] = (-0.03, 0.03),
    y_jitter: tuple[float, float] = (-0.03, 0.03),
    yaw_jitter: tuple[float, float] = (-0.06, 0.06),
    velocity_scale_range: tuple[float, float] = (0.85, 1.15),
    approach_velocity_range: dict[str, tuple[float, float]] | None = None,
    near_velocity_range: dict[str, tuple[float, float]] | None = None,
    joint_position_scale_range: tuple[float, float] = (0.98, 1.02),
    joint_velocity_range: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset from approach starts, recorded handoff states, and near-step finisher states."""
    if approach_probability < 0.0 or switch_probability < 0.0 or approach_probability + switch_probability > 1.0:
        raise ValueError("approach_probability and switch_probability must be non-negative and sum to <= 1.")

    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = asset.device
    stage_sample = torch.rand(num_envs, device=device)
    use_approach = stage_sample < approach_probability
    use_switch = (stage_sample >= approach_probability) & (
        stage_sample < approach_probability + switch_probability
    )
    use_near = ~(use_approach | use_switch)

    if use_approach.any():
        reset_stair_climb_root_state(
            env,
            env_ids[use_approach],
            first_riser_x=first_riser_x,
            step_height_range=step_height_range,
            approach_probability=1.0,
            approach_x_range=approach_x_range,
            near_step_x_range=near_step_x_range,
            y_range=y_range,
            yaw_range=yaw_range,
            base_height=base_height,
            step_base_clearance=step_base_clearance,
            velocity_range=approach_velocity_range,
            joint_position_scale_range=joint_position_scale_range,
            joint_velocity_range=joint_velocity_range,
            asset_cfg=asset_cfg,
        )

    if use_switch.any():
        reset_stair_climb_from_switch_state_dataset(
            env,
            env_ids[use_switch],
            dataset_path=dataset_path,
            x_jitter=x_jitter,
            y_jitter=y_jitter,
            yaw_jitter=yaw_jitter,
            velocity_scale_range=velocity_scale_range,
            asset_cfg=asset_cfg,
        )

    if use_near.any():
        reset_stair_climb_root_state(
            env,
            env_ids[use_near],
            first_riser_x=first_riser_x,
            step_height_range=step_height_range,
            approach_probability=0.0,
            approach_x_range=approach_x_range,
            near_step_x_range=near_step_x_range,
            y_range=y_range,
            yaw_range=yaw_range,
            base_height=base_height,
            step_base_clearance=step_base_clearance,
            velocity_range=near_velocity_range,
            joint_position_scale_range=joint_position_scale_range,
            joint_velocity_range=joint_velocity_range,
            asset_cfg=asset_cfg,
        )


def reset_random_terrain_level_then_stair_climb_mixed_approach(
    env,
    env_ids: torch.Tensor,
    dataset_path: str,
    approach_probability: float,
    switch_probability: float,
    first_riser_x: float,
    step_height_range: tuple[float, float],
    min_level: int = 0,
    max_level: int = 2,
    approach_x_range: tuple[float, float] = (-0.20, 0.18),
    near_step_x_range: tuple[float, float] = (0.70, 0.90),
    y_range: tuple[float, float] = (-0.06, 0.06),
    yaw_range: tuple[float, float] = (-0.08, 0.08),
    base_height: float = 0.31,
    step_base_clearance: float = 0.29,
    x_jitter: tuple[float, float] = (-0.03, 0.03),
    y_jitter: tuple[float, float] = (-0.03, 0.03),
    yaw_jitter: tuple[float, float] = (-0.06, 0.06),
    velocity_scale_range: tuple[float, float] = (0.85, 1.15),
    approach_velocity_range: dict[str, tuple[float, float]] | None = None,
    near_velocity_range: dict[str, tuple[float, float]] | None = None,
    joint_position_scale_range: tuple[float, float] = (0.98, 1.02),
    joint_velocity_range: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Randomize terrain rows before applying the staged stair reset."""
    terrain = env.scene.terrain
    levels = getattr(terrain, "terrain_levels", None)
    origins = getattr(terrain, "terrain_origins", None)
    types = getattr(terrain, "terrain_types", None)
    if levels is not None and origins is not None and types is not None:
        max_valid = int(origins.shape[0]) - 1
        lo = max(0, min(int(min_level), max_valid))
        hi = max(lo, min(int(max_level), max_valid))
        sampled_levels = torch.randint(lo, hi + 1, (len(env_ids),), device=env.device, dtype=levels.dtype)
        levels[env_ids] = sampled_levels
        terrain.env_origins[env_ids] = origins[levels[env_ids], types[env_ids]]

    reset_stair_climb_mixed_approach_switch_and_near_step(
        env,
        env_ids,
        dataset_path=dataset_path,
        approach_probability=approach_probability,
        switch_probability=switch_probability,
        first_riser_x=first_riser_x,
        step_height_range=step_height_range,
        approach_x_range=approach_x_range,
        near_step_x_range=near_step_x_range,
        y_range=y_range,
        yaw_range=yaw_range,
        base_height=base_height,
        step_base_clearance=step_base_clearance,
        x_jitter=x_jitter,
        y_jitter=y_jitter,
        yaw_jitter=yaw_jitter,
        velocity_scale_range=velocity_scale_range,
        approach_velocity_range=approach_velocity_range,
        near_velocity_range=near_velocity_range,
        joint_position_scale_range=joint_position_scale_range,
        joint_velocity_range=joint_velocity_range,
        asset_cfg=asset_cfg,
    )


def _load_switch_state_dataset(dataset_path: str, device: str) -> dict[str, torch.Tensor]:
    path = str(Path(dataset_path).expanduser().resolve())
    cached = _SWITCH_STATE_CACHE.get(path)
    if cached is None or next(iter(cached.values())).device != torch.device(device):
        loaded = torch.load(path, weights_only=False, map_location=device)
        cached = {key: value.to(device) for key, value in loaded.items() if isinstance(value, torch.Tensor)}
        _SWITCH_STATE_CACHE[path] = cached
    return cached


def stair_goal_state(
    env,
    target_x: float,
    target_z: float,
    step_height_range: tuple[float, float] | None = None,
    num_steps: int = 3,
    base_clearance: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Privileged critic observation: remaining forward distance and relative base height."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    remaining_x = (target_x - root_pos[:, 0]).unsqueeze(1)
    height = root_pos[:, 2].unsqueeze(1)
    if step_height_range is None:
        target_height = torch.full_like(height, target_z)
    else:
        target_height = (float(num_steps) * _terrain_level_step_height(env, step_height_range) + base_clearance).unsqueeze(1)
    return torch.cat((remaining_x, height, target_height), dim=1)


def stair_handoff_state(
    env,
    target_x: float,
    x_scale: float = 1.0,
    y_scale: float = 0.5,
    yaw_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Policy observation for the approach controller: remaining x, lateral offset, and yaw error."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    remaining_x = torch.clamp((target_x - root_pos[:, 0]) / x_scale, min=-0.35, max=1.20)
    lateral_y = torch.clamp(root_pos[:, 1] / y_scale, min=-1.0, max=1.0)
    yaw_error = torch.clamp(yaw / yaw_scale, min=-1.0, max=1.0)
    return torch.stack((remaining_x, lateral_y, yaw_error), dim=1)


def _terrain_level_step_height(env, step_height_range: tuple[float, float]) -> torch.Tensor:
    """Approximate the per-env stair height from the terrain curriculum level."""
    min_height, max_height = step_height_range
    terrain = env.scene.terrain
    levels = getattr(terrain, "terrain_levels", None)
    origins = getattr(terrain, "terrain_origins", None)
    if levels is None or origins is None:
        return torch.full((env.num_envs,), min_height, dtype=torch.float32, device=env.device)

    num_rows = max(int(origins.shape[0]), 1)
    difficulty = torch.clamp(levels.to(torch.float32) / float(num_rows), min=0.0, max=1.0)
    return min_height + difficulty * (max_height - min_height)


def stair_forward_progress_reward(
    env,
    target_x: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward normalized forward progress toward the top platform."""
    asset = env.scene[asset_cfg.name]
    root_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return torch.clamp(root_x / target_x, min=0.0, max=1.0)


def stair_forward_velocity_reward(
    env,
    target_x: float,
    max_speed: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward velocity until the robot reaches the top platform."""
    asset = env.scene[asset_cfg.name]
    root_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    active = (root_x < target_x).to(torch.float32)
    forward_speed = torch.clamp(asset.data.root_lin_vel_w[:, 0], min=0.0, max=max_speed) / max_speed
    return active * forward_speed


def forward_velocity_reward(
    env,
    command_name: str = "base_velocity",
    min_command_x: float = 0.05,
    max_speed: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward body velocity when the command asks for forward motion."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    commanded = (command[:, 0] > min_command_x).to(torch.float32)
    forward_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0, max=max_speed) / max_speed
    return commanded * forward_speed


def commanded_stuck_penalty(
    env,
    command_name: str = "base_velocity",
    min_command_x: float = 0.12,
    min_speed_x: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize standing still when a forward command is active."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    commanded = command[:, 0] > min_command_x
    slow = asset.data.root_lin_vel_b[:, 0] < min_speed_x
    return (commanded & slow).to(torch.float32)


def commanded_planar_velocity_reward(
    env,
    command_name: str = "base_velocity",
    min_command_norm: float = 0.05,
    max_speed: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward body XY velocity along the commanded planar direction."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy = command[:, :2]
    command_norm = torch.linalg.norm(command_xy, dim=1)
    direction = command_xy / torch.clamp(command_norm.unsqueeze(1), min=1.0e-6)
    speed_along_command = torch.sum(asset.data.root_lin_vel_b[:, :2] * direction, dim=1)
    active = (command_norm > min_command_norm).to(torch.float32)
    return active * torch.clamp(speed_along_command, min=0.0, max=max_speed) / max_speed


def commanded_yaw_velocity_reward(
    env,
    command_name: str = "base_velocity",
    min_command_abs: float = 0.08,
    max_yaw_rate: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward yaw velocity in the commanded yaw direction."""
    asset = env.scene[asset_cfg.name]
    yaw_command = env.command_manager.get_command(command_name)[:, 2]
    yaw_rate_along_command = asset.data.root_ang_vel_b[:, 2] * torch.sign(yaw_command)
    active = (torch.abs(yaw_command) > min_command_abs).to(torch.float32)
    return active * torch.clamp(yaw_rate_along_command, min=0.0, max=max_yaw_rate) / max_yaw_rate


def _phase_swing_clearance_quality(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    foot_radius: float,
    clearance: float,
    f_min: float,
    f_max: float,
    gain: float,
    phase_offsets: tuple[float, ...],
    gate_sharpness: float,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    swing_gate, _ = _trot_swing_stance_gates(env, command_name, f_min, f_max, gain, phase_offsets, gate_sharpness)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    foot_clearance = torch.clamp(foot_z - foot_radius, min=0.0)
    clearance_score = torch.clamp(foot_clearance / max(clearance, 1.0e-6), min=0.0, max=1.0)
    normalizer = torch.clamp(torch.sum(swing_gate, dim=1), min=1.0)
    return torch.sum(swing_gate * clearance_score, dim=1) / normalizer


def _low_foot_slide_score(
    env,
    asset_cfg: SceneEntityCfg,
    foot_radius: float,
    clearance: float,
    velocity_scale: float,
    foot_weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    foot_clearance = torch.clamp(foot_z - foot_radius, min=0.0)
    low_height = torch.clamp((clearance - foot_clearance) / max(clearance, 1.0e-6), min=0.0, max=1.0)
    foot_planar_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    moving = torch.tanh(velocity_scale * foot_planar_vel)
    slide = low_height * moving
    if foot_weights is None:
        return torch.mean(slide, dim=1)
    weights = torch.tensor(foot_weights, dtype=slide.dtype, device=slide.device)
    if weights.numel() != slide.shape[1]:
        raise ValueError(f"foot_weights has {weights.numel()} values, expected {slide.shape[1]}.")
    weights = weights / torch.clamp(torch.mean(weights), min=1.0e-6)
    return torch.mean(slide * weights.unsqueeze(0), dim=1)


def gait_gated_commanded_planar_velocity_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_command_norm: float = 0.05,
    max_speed: float = 0.55,
    foot_radius: float = 0.012,
    clearance: float = 0.024,
    min_gate: float = 0.05,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Reward commanded planar speed only when phase-swing feet have useful clearance."""
    velocity_score = commanded_planar_velocity_reward(
        env,
        command_name=command_name,
        min_command_norm=min_command_norm,
        max_speed=max_speed,
    )
    clearance_quality = _phase_swing_clearance_quality(
        env, asset_cfg, command_name, foot_radius, clearance, f_min, f_max, gain, phase_offsets, gate_sharpness
    )
    gate = min_gate + (1.0 - min_gate) * clearance_quality
    return velocity_score * gate


def gait_gated_commanded_yaw_velocity_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_command_abs: float = 0.08,
    max_yaw_rate: float = 0.75,
    foot_radius: float = 0.012,
    clearance: float = 0.024,
    min_gate: float = 0.05,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
) -> torch.Tensor:
    """Reward commanded yaw speed only when phase-swing feet are not skimming the floor."""
    velocity_score = commanded_yaw_velocity_reward(
        env,
        command_name=command_name,
        min_command_abs=min_command_abs,
        max_yaw_rate=max_yaw_rate,
    )
    clearance_quality = _phase_swing_clearance_quality(
        env, asset_cfg, command_name, foot_radius, clearance, f_min, f_max, gain, phase_offsets, gate_sharpness
    )
    gate = min_gate + (1.0 - min_gate) * clearance_quality
    return velocity_score * gate


def low_clearance_velocity_progress_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_planar_command_norm: float = 0.05,
    max_speed: float = 0.55,
    min_yaw_command_abs: float = 0.08,
    max_yaw_rate: float = 0.75,
    foot_radius: float = 0.012,
    clearance: float = 0.026,
    velocity_scale: float = 8.0,
    f_min: float = TROT_F_MIN,
    f_max: float = TROT_F_MAX,
    gain: float = TROT_F_GAIN,
    phase_offsets: tuple[float, ...] = TROT_FOOT_PHASE_OFFSETS,
    gate_sharpness: float = TROT_GATE_SHARPNESS,
    foot_weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Penalize making commanded progress by sliding feet close to the ground."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    command_xy = command[:, :2]
    command_norm = torch.linalg.norm(command_xy, dim=1)
    direction = command_xy / torch.clamp(command_norm.unsqueeze(1), min=1.0e-6)
    planar_speed_along_command = torch.sum(asset.data.root_lin_vel_b[:, :2] * direction, dim=1)
    planar_score = torch.clamp(planar_speed_along_command, min=0.0, max=max_speed) / max_speed
    planar_active = command_norm > min_planar_command_norm

    yaw_command = command[:, 2]
    yaw_rate_along_command = asset.data.root_ang_vel_b[:, 2] * torch.sign(yaw_command)
    yaw_score = torch.clamp(yaw_rate_along_command, min=0.0, max=max_yaw_rate) / max_yaw_rate
    yaw_active = torch.abs(yaw_command) > min_yaw_command_abs

    clearance_quality = _phase_swing_clearance_quality(
        env, asset_cfg, command_name, foot_radius, clearance, f_min, f_max, gain, phase_offsets, gate_sharpness
    )
    slide_score = _low_foot_slide_score(env, asset_cfg, foot_radius, clearance, velocity_scale, foot_weights)
    progress_score = torch.maximum(
        torch.where(planar_active, planar_score, torch.zeros_like(planar_score)),
        torch.where(yaw_active, yaw_score, torch.zeros_like(yaw_score)),
    )
    active = (planar_active | yaw_active).to(torch.float32)
    return active * progress_score * torch.clamp(1.0 - clearance_quality, min=0.0, max=1.0) * (0.5 + slide_score)


def commanded_velocity_stuck_penalty(
    env,
    command_name: str = "base_velocity",
    min_planar_command_norm: float = 0.08,
    min_planar_speed: float = 0.035,
    min_yaw_command_abs: float = 0.14,
    min_yaw_rate: float = 0.07,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize staying nearly still when planar or yaw velocity is commanded."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    command_xy = command[:, :2]
    command_norm = torch.linalg.norm(command_xy, dim=1)
    direction = command_xy / torch.clamp(command_norm.unsqueeze(1), min=1.0e-6)
    planar_speed_along_command = torch.sum(asset.data.root_lin_vel_b[:, :2] * direction, dim=1)
    planar_stuck = (command_norm > min_planar_command_norm) & (planar_speed_along_command < min_planar_speed)

    yaw_command = command[:, 2]
    yaw_rate_along_command = asset.data.root_ang_vel_b[:, 2] * torch.sign(yaw_command)
    yaw_stuck = (torch.abs(yaw_command) > min_yaw_command_abs) & (yaw_rate_along_command < min_yaw_rate)

    return (planar_stuck | yaw_stuck).to(torch.float32)


def commanded_forward_velocity_reward(
    env,
    command_name: str = "base_velocity",
    min_command_x: float = 0.05,
    max_speed: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward speed directly when the command asks for forward motion."""
    asset = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    active = (command_x > min_command_x).to(torch.float32)
    speed_score = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0, max=max_speed) / max_speed
    command_scale = torch.clamp(command_x / max(max_speed, 1.0e-6), min=0.0, max=1.0)
    return active * speed_score * (0.35 + 0.65 * command_scale)


def stair_height_progress_reward(
    env,
    target_z: float,
    gate_x: float = 0.5,
    base_z: float = 0.35,
    step_height_range: tuple[float, float] | None = None,
    num_steps: int = 3,
    target_base_clearance: float = 0.30,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.75,
    max_roll_projected: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward lifting the base as the robot climbs."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    near_stair = (root_pos[:, 0] > gate_x).to(torch.float32)
    if step_height_range is None:
        target_height = torch.full_like(root_pos[:, 2], target_z)
    else:
        target_height = float(num_steps) * _terrain_level_step_height(env, step_height_range) + target_base_clearance
    height_progress = torch.clamp(
        (root_pos[:, 2] - base_z) / torch.clamp(target_height - base_z, min=1.0e-3),
        min=0.0,
        max=1.0,
    )
    return near_stair * height_progress * _upright_no_trunk_contact_gate(
        env,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
    )


def stair_foot_placement_reward(
    env,
    asset_cfg: SceneEntityCfg,
    foot_indices: tuple[int, ...],
    edge_x: float,
    target_height: float | None = None,
    step_height_range: tuple[float, float] | None = None,
    step_index: int = 1,
    x_margin: float = 0.08,
    z_margin: float = 0.04,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.80,
    max_roll_projected: float = 0.50,
) -> torch.Tensor:
    """Reward selected feet for getting onto the next tread."""
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    feet = foot_pos[:, list(foot_indices), :]
    x_ok = torch.sigmoid(12.0 * (feet[:, :, 0] - (edge_x + x_margin)))
    if step_height_range is None:
        target_height_tensor = torch.full((feet.shape[0],), float(target_height or 0.0), device=feet.device)
    else:
        target_height_tensor = float(step_index) * _terrain_level_step_height(env, step_height_range)
    z_ok = torch.sigmoid(18.0 * (feet[:, :, 2] - (target_height_tensor.unsqueeze(1) - z_margin)))
    centered = torch.exp(-torch.square(feet[:, :, 1] / 0.45))
    return torch.mean(x_ok * z_ok * centered, dim=1) * _upright_no_trunk_contact_gate(
        env,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
    )


def stair_foot_contact_on_step_reward(
    env,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    foot_indices: tuple[int, ...],
    edge_x: float,
    target_height: float | None = None,
    step_height_range: tuple[float, float] | None = None,
    step_index: int = 1,
    x_margin: float = 0.03,
    z_margin: float = 0.04,
    z_upper_margin: float = 0.10,
    lateral_std: float = 0.45,
    contact_threshold: float = 1.0,
    trunk_sensor_cfg: SceneEntityCfg | None = None,
    trunk_contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.80,
    max_roll_projected: float = 0.50,
) -> torch.Tensor:
    """Reward selected feet for contacting the top of a tread."""
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    feet = foot_pos[:, list(foot_indices), :]
    contact = _foot_contact(env, contact_sensor_cfg, contact_threshold).to(torch.float32)[:, list(foot_indices)]

    if step_height_range is None:
        target_height_tensor = torch.full((feet.shape[0],), float(target_height or 0.0), device=feet.device)
    else:
        target_height_tensor = float(step_index) * _terrain_level_step_height(env, step_height_range)
    tread_height = target_height_tensor.unsqueeze(1)

    x_ok = torch.sigmoid(14.0 * (feet[:, :, 0] - (edge_x + x_margin)))
    z_above_tread = torch.sigmoid(28.0 * (feet[:, :, 2] - (tread_height - z_margin)))
    z_near_tread = torch.sigmoid(28.0 * ((tread_height + z_upper_margin) - feet[:, :, 2]))
    centered = torch.exp(-torch.square(feet[:, :, 1] / lateral_std))
    score = x_ok * z_above_tread * z_near_tread * centered * contact
    return torch.mean(score, dim=1) * _upright_no_trunk_contact_gate(
        env,
        asset_cfg=asset_cfg,
        sensor_cfg=trunk_sensor_cfg,
        contact_threshold=trunk_contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
    )


def stair_base_on_step_reward(
    env,
    first_riser_x: float,
    step_depth: float,
    step_index: int,
    step_height_range: tuple[float, float],
    base_clearance: float = 0.20,
    x_fraction: float = 0.60,
    lateral_std: float = 0.45,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.72,
    max_roll_projected: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward bringing the trunk over a specific stair tread instead of only touching it with front feet."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    step_height = _terrain_level_step_height(env, step_height_range)
    target_x = first_riser_x + (float(step_index) - 1.0 + x_fraction) * step_depth
    target_z = float(step_index) * step_height + base_clearance

    x_score = torch.sigmoid(8.0 * (root_pos[:, 0] - target_x))
    z_score = torch.sigmoid(14.0 * (root_pos[:, 2] - target_z))
    centered = torch.exp(-torch.square(root_pos[:, 1] / lateral_std))
    return x_score * z_score * centered * _upright_no_trunk_contact_gate(
        env,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
    )


def stair_base_height_over_step_reward(
    env,
    first_riser_x: float,
    step_index: int,
    step_height_range: tuple[float, float],
    base_clearance: float = 0.28,
    gate_margin: float = -0.15,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.72,
    max_roll_projected: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward raising the trunk to a viable climbing height near or past a riser."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    step_height = _terrain_level_step_height(env, step_height_range)
    target_z = float(step_index) * step_height + base_clearance
    near_or_past_riser = torch.sigmoid(10.0 * (root_pos[:, 0] - (first_riser_x + gate_margin)))
    height_score = torch.sigmoid(18.0 * (root_pos[:, 2] - target_z))
    return near_or_past_riser * height_score * _upright_no_trunk_contact_gate(
        env,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
    )


def stair_rear_feet_forward_reward(
    env,
    asset_cfg: SceneEntityCfg,
    edge_x: float,
    foot_indices: tuple[int, ...] = (2, 3),
    x_margin: float = -0.05,
    lateral_std: float = 0.45,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.80,
    max_roll_projected: float = 0.50,
) -> torch.Tensor:
    """Reward rear feet for moving forward to follow the body over the first riser."""
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    rear_feet = foot_pos[:, list(foot_indices), :]
    x_score = torch.sigmoid(10.0 * (rear_feet[:, :, 0] - (edge_x + x_margin)))
    centered = torch.exp(-torch.square(rear_feet[:, :, 1] / lateral_std))
    return torch.mean(x_score * centered, dim=1) * _upright_no_trunk_contact_gate(
        env,
        asset_cfg=SceneEntityCfg("robot"),
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
    )


def stair_rear_feet_min_x_progress_reward(
    env,
    asset_cfg: SceneEntityCfg,
    first_riser_x: float,
    foot_indices: tuple[int, ...] = (2, 3),
    start_margin: float = -0.22,
    target_margin: float = -0.02,
    body_gate_margin: float = -0.12,
    lateral_std: float = 0.45,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.80,
    max_roll_projected: float = 0.50,
) -> torch.Tensor:
    """Reward both rear feet, not just the average rear foot, for catching up to the first riser."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    rear_feet = foot_pos[:, list(foot_indices), :]
    rear_min_x = torch.min(rear_feet[:, :, 0], dim=1).values
    start_x = first_riser_x + start_margin
    target_x = first_riser_x + target_margin
    progress = torch.clamp((rear_min_x - start_x) / max(target_x - start_x, 1.0e-3), min=0.0, max=1.0)
    body_gate = torch.sigmoid(12.0 * (root_pos[:, 0] - (first_riser_x + body_gate_margin)))
    centered = torch.exp(-torch.square(torch.mean(rear_feet[:, :, 1], dim=1) / lateral_std))
    return progress * body_gate * centered * _upright_no_trunk_contact_gate(
        env,
        asset_cfg=SceneEntityCfg("robot"),
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
    )


def stair_rear_feet_xz_contact_reward(
    env,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    first_riser_x: float,
    step_height_range: tuple[float, float],
    foot_indices: tuple[int, ...] = (2, 3),
    step_index: int = 1,
    foot_x_margin: float = -0.02,
    foot_z_margin: float = 0.00,
    x_std: float = 0.12,
    z_std: float = 0.07,
    lateral_std: float = 0.28,
    contact_threshold: float = 0.5,
    min_contacts_for_bonus: int = 1,
    body_gate_margin: float = -0.08,
    trunk_sensor_cfg: SceneEntityCfg | None = None,
    trunk_contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.76,
    max_roll_projected: float = 0.50,
    aggregation: str = "mean",
) -> torch.Tensor:
    """Reward rear feet only when they move forward and lift to the stair tread together."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    rear_feet = foot_pos[:, list(foot_indices), :]
    step_height = float(step_index) * _terrain_level_step_height(env, step_height_range)

    target_x = first_riser_x + foot_x_margin
    x_score = torch.sigmoid((rear_feet[:, :, 0] - target_x) / x_std)
    z_score = torch.sigmoid((rear_feet[:, :, 2] - (step_height.unsqueeze(1) - foot_z_margin)) / z_std)
    lateral_score = torch.exp(-torch.square(rear_feet[:, :, 1] / lateral_std))
    per_foot_score = x_score * z_score * lateral_score
    if aggregation == "mean":
        foot_score = torch.mean(per_foot_score, dim=1)
    elif aggregation == "min":
        foot_score = torch.min(per_foot_score, dim=1).values
    else:
        raise ValueError(f"Unsupported rear foot aggregation: {aggregation!r}.")

    contact = _foot_contact(env, contact_sensor_cfg, contact_threshold).to(torch.float32)[:, list(foot_indices)]
    contact_score = torch.mean(contact, dim=1)
    contact_count = torch.sum(contact, dim=1)
    contact_bonus = (contact_count >= min_contacts_for_bonus).to(torch.float32)
    body_gate = torch.sigmoid(12.0 * (root_pos[:, 0] - (first_riser_x + body_gate_margin)))
    return foot_score * (0.35 + 0.45 * contact_score + 0.20 * contact_bonus) * body_gate * _upright_no_trunk_contact_gate(
        env,
        asset_cfg=SceneEntityCfg("robot"),
        sensor_cfg=trunk_sensor_cfg,
        contact_threshold=trunk_contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
    )


def stair_rear_feet_low_after_base_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    first_riser_x: float,
    step_height_range: tuple[float, float],
    foot_indices: tuple[int, ...] = (2, 3),
    base_x_margin: float = 0.08,
    foot_x_margin: float = -0.01,
    foot_z_margin: float = 0.015,
    x_std: float = 0.08,
    z_std: float = 0.035,
    lateral_std: float = 0.22,
    max_pitch_projected: float = 0.72,
    max_roll_projected: float = 0.48,
) -> torch.Tensor:
    """Penalize leaving rear feet below the first tread after the base has moved onto it."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    rear_feet = foot_pos[:, list(foot_indices), :]
    step_height = _terrain_level_step_height(env, step_height_range)

    base_gate = torch.sigmoid((root_pos[:, 0] - (first_riser_x + base_x_margin)) / x_std)
    rear_forward = torch.sigmoid((rear_feet[:, :, 0] - (first_riser_x + foot_x_margin)) / x_std)
    rear_low = torch.sigmoid((step_height.unsqueeze(1) - foot_z_margin - rear_feet[:, :, 2]) / z_std)
    centered = torch.exp(-torch.square(rear_feet[:, :, 1] / lateral_std))
    rear_low_score = torch.mean(rear_forward * rear_low * centered, dim=1)

    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])
    upright = (gravity_xy[:, 0] < max_pitch_projected) & (gravity_xy[:, 1] < max_roll_projected)
    return base_gate * rear_low_score * upright.to(torch.float32)


def stair_near_goal_heading_reward(
    env,
    target_x: float,
    step_height_range: tuple[float, float],
    target_base_clearance: float = 0.24,
    num_steps: int = 1,
    x_fraction: float = 0.74,
    min_height_fraction: float = 0.60,
    yaw_std: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward staying pointed forward once the robot is near the stair completion state."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    target_height = float(num_steps) * _terrain_level_step_height(env, step_height_range) + target_base_clearance
    near_goal = (root_pos[:, 0] > x_fraction * target_x) & (root_pos[:, 2] > min_height_fraction * target_height)
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return near_goal.to(torch.float32) * torch.exp(-torch.square(yaw / yaw_std))


def stair_base_with_feet_on_step_reward(
    env,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    foot_indices: tuple[int, ...],
    first_riser_x: float,
    step_depth: float,
    step_height_range: tuple[float, float],
    step_index: int = 1,
    base_clearance: float = 0.24,
    base_x_fraction: float = 0.35,
    foot_x_margin: float = 0.02,
    foot_z_margin: float = 0.05,
    contact_threshold: float = 0.5,
    trunk_sensor_cfg: SceneEntityCfg | None = None,
    trunk_contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.70,
    max_roll_projected: float = 0.48,
) -> torch.Tensor:
    """Reward getting the trunk and selected feet onto a stair tread together."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    feet = foot_pos[:, list(foot_indices), :]
    contact = _foot_contact(env, contact_sensor_cfg, contact_threshold).to(torch.float32)[:, list(foot_indices)]

    step_height = _terrain_level_step_height(env, step_height_range)
    target_base_x = first_riser_x + (float(step_index) - 1.0 + base_x_fraction) * step_depth
    target_base_z = float(step_index) * step_height + base_clearance
    base_x_score = torch.sigmoid(10.0 * (root_pos[:, 0] - target_base_x))
    base_z_score = torch.sigmoid(18.0 * (root_pos[:, 2] - target_base_z))

    foot_x_score = torch.sigmoid(14.0 * (feet[:, :, 0] - (first_riser_x + foot_x_margin)))
    foot_z_score = torch.sigmoid(24.0 * (feet[:, :, 2] - (float(step_index) * step_height.unsqueeze(1) - foot_z_margin)))
    foot_score = torch.mean(foot_x_score * foot_z_score * contact, dim=1)
    return base_x_score * base_z_score * foot_score * _upright_no_trunk_contact_gate(
        env,
        asset_cfg=asset_cfg,
        sensor_cfg=trunk_sensor_cfg,
        contact_threshold=trunk_contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
    )


def stair_base_with_feet_pose_on_step_reward(
    env,
    asset_cfg: SceneEntityCfg,
    foot_indices: tuple[int, ...],
    first_riser_x: float,
    step_depth: float,
    step_height_range: tuple[float, float],
    step_index: int = 1,
    base_clearance: float = 0.23,
    base_x_fraction: float = 0.32,
    foot_x_margin: float = -0.04,
    foot_z_margin: float = 0.08,
    trunk_sensor_cfg: SceneEntityCfg | None = None,
    trunk_contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.78,
    max_roll_projected: float = 0.55,
) -> torch.Tensor:
    """Reward a soft stage where the trunk and feet are positioned for a stair completion."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    feet = foot_pos[:, list(foot_indices), :]

    step_height = _terrain_level_step_height(env, step_height_range)
    target_base_x = first_riser_x + (float(step_index) - 1.0 + base_x_fraction) * step_depth
    target_base_z = float(step_index) * step_height + base_clearance
    base_x_score = torch.sigmoid(9.0 * (root_pos[:, 0] - target_base_x))
    base_z_score = torch.sigmoid(14.0 * (root_pos[:, 2] - target_base_z))

    foot_x_score = torch.sigmoid(10.0 * (feet[:, :, 0] - (first_riser_x + foot_x_margin)))
    foot_z_score = torch.sigmoid(16.0 * (feet[:, :, 2] - (float(step_index) * step_height.unsqueeze(1) - foot_z_margin)))
    foot_score = torch.mean(foot_x_score * foot_z_score, dim=1)
    return base_x_score * base_z_score * foot_score * _upright_no_trunk_contact_gate(
        env,
        asset_cfg=asset_cfg,
        sensor_cfg=trunk_sensor_cfg,
        contact_threshold=trunk_contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
    )


def stair_low_base_at_riser_penalty(
    env,
    first_riser_x: float,
    step_height_range: tuple[float, float],
    min_base_clearance: float = 0.23,
    x_window: tuple[float, float] = (-0.05, 0.45),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize pushing over the first riser while the trunk stays below a climbing posture."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    step_height = _terrain_level_step_height(env, step_height_range)
    lower_x = first_riser_x + x_window[0]
    upper_x = first_riser_x + x_window[1]
    in_window = torch.sigmoid(18.0 * (root_pos[:, 0] - lower_x)) * torch.sigmoid(18.0 * (upper_x - root_pos[:, 0]))
    low_base = torch.sigmoid(18.0 * (step_height + min_base_clearance - root_pos[:, 2]))
    return in_window * low_base


def stair_low_base_at_riser_failure(
    env,
    first_riser_x: float,
    step_height_range: tuple[float, float],
    min_base_clearance: float = 0.25,
    x_window: tuple[float, float] = (-0.05, 0.42),
    grace_time_s: float = 0.7,
    max_speed_x: float | None = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot drives into the riser with the base still too low."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    step_height = _terrain_level_step_height(env, step_height_range)
    lower_x = first_riser_x + x_window[0]
    upper_x = first_riser_x + x_window[1]
    in_window = (root_pos[:, 0] > lower_x) & (root_pos[:, 0] < upper_x)
    low_base = root_pos[:, 2] < step_height + min_base_clearance
    past_grace = env.episode_length_buf.to(torch.float32) * env.step_dt > grace_time_s
    if max_speed_x is None:
        slow_or_stopped = torch.ones_like(low_base)
    else:
        slow_or_stopped = asset.data.root_lin_vel_w[:, 0] < max_speed_x
    return in_window & low_base & past_grace & slow_or_stopped


def stair_goal_reward(
    env,
    target_x: float,
    target_z: float,
    xy_std: float = 0.8,
    z_std: float = 0.6,
    step_height_range: tuple[float, float] | None = None,
    num_steps: int = 3,
    target_base_clearance: float = 0.30,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.65,
    max_roll_projected: float = 0.40,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward proximity to a pose on the upper platform."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    x_error = torch.clamp(target_x - root_pos[:, 0], min=0.0)
    if step_height_range is None:
        target_height = torch.full_like(root_pos[:, 2], target_z)
    else:
        target_height = float(num_steps) * _terrain_level_step_height(env, step_height_range) + target_base_clearance
    z_error = torch.clamp(target_height - root_pos[:, 2], min=0.0)
    return torch.exp(-torch.square(x_error / xy_std) - torch.square(z_error / z_std)) * _upright_no_trunk_contact_gate(
        env,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
    )


def stair_lateral_drift_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize drifting away from the stair centerline."""
    asset = env.scene[asset_cfg.name]
    root_y = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return torch.square(root_y)


def stair_yaw_drift_penalty(
    env,
    yaw_allowance: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize turning away from the stair direction."""
    asset = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    yaw_excess = torch.clamp(torch.abs(yaw) - yaw_allowance, min=0.0)
    return torch.square(yaw_excess)


def stair_bad_yaw(
    env,
    max_yaw: float = 1.15,
    gate_x: float = 0.55,
    grace_time_s: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot turns sideways after it has engaged the stair."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    past_grace = env.episode_length_buf.to(torch.float32) * env.step_dt > grace_time_s
    near_stair = root_pos[:, 0] > gate_x
    return near_stair & past_grace & (torch.abs(yaw) > max_yaw)


def stair_tilt_penalty(
    env,
    pitch_allowance: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll and excessive pitch while allowing the body to lean during climbing."""
    asset = env.scene[asset_cfg.name]
    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])
    pitch_excess = torch.clamp(gravity_xy[:, 0] - pitch_allowance, min=0.0)
    roll = gravity_xy[:, 1]
    return torch.square(pitch_excess) + 2.0 * torch.square(roll)


def stair_stuck_penalty(
    env,
    command_name: str = "base_velocity",
    min_command_x: float = 0.2,
    min_speed_x: float = 0.05,
    before_goal_x: float = 2.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize commanded forward motion with almost no forward progress."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    root_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    commanded = command[:, 0] > min_command_x
    slow = asset.data.root_lin_vel_w[:, 0] < min_speed_x
    before_goal = root_x < before_goal_x
    return (commanded & slow & before_goal).to(torch.float32)


def stair_near_goal_timeout_penalty(
    env,
    target_x: float,
    step_height_range: tuple[float, float],
    target_base_clearance: float = 0.24,
    num_steps: int = 1,
    min_progress_fraction: float = 0.72,
    min_height_fraction: float = 0.62,
    grace_time_s: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lingering near the stage goal without satisfying the success condition."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    target_height = float(num_steps) * _terrain_level_step_height(env, step_height_range) + target_base_clearance
    near_x = root_pos[:, 0] > min_progress_fraction * target_x
    near_z = root_pos[:, 2] > min_height_fraction * target_height
    past_grace = env.episode_length_buf.to(torch.float32) * env.step_dt > grace_time_s
    return (near_x & near_z & past_grace).to(torch.float32)


def stair_final_approach_reward(
    env,
    target_x: float,
    step_height_range: tuple[float, float],
    target_base_clearance: float = 0.24,
    num_steps: int = 1,
    start_fraction: float = 0.72,
    min_height_fraction: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward closing the last gap to the stage target instead of lingering near it."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    target_height = float(num_steps) * _terrain_level_step_height(env, step_height_range) + target_base_clearance
    start_x = start_fraction * target_x
    x_progress = torch.clamp((root_pos[:, 0] - start_x) / max(target_x - start_x, 1.0e-3), min=0.0, max=1.0)
    height_gate = torch.sigmoid(16.0 * (root_pos[:, 2] - min_height_fraction * target_height))
    return x_progress * height_gate


def trunk_contact_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize trunk contacts without immediately terminating the episode."""
    contact = _foot_contact(env, sensor_cfg, threshold).to(torch.float32)
    return torch.sum(contact, dim=1)


def _upright_no_trunk_contact_gate(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None,
    contact_threshold: float,
    max_pitch_projected: float,
    max_roll_projected: float,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])
    upright = (gravity_xy[:, 0] < max_pitch_projected) & (gravity_xy[:, 1] < max_roll_projected)
    if sensor_cfg is None:
        no_trunk_contact = torch.ones_like(upright)
    else:
        no_trunk_contact = ~torch.any(_foot_contact(env, sensor_cfg, contact_threshold), dim=1)
    return (upright & no_trunk_contact).to(torch.float32)


def stair_success_reward(
    env,
    target_x: float,
    target_z: float,
    step_height_range: tuple[float, float] | None = None,
    num_steps: int = 3,
    target_base_clearance: float = 0.30,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.55,
    max_roll_projected: float = 0.55,
    max_yaw: float | None = None,
    max_abs_y: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse reward when the base reaches the upper platform."""
    return stair_climb_success(
        env,
        target_x=target_x,
        target_z=target_z,
        step_height_range=step_height_range,
        num_steps=num_steps,
        target_base_clearance=target_base_clearance,
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
        max_yaw=max_yaw,
        max_abs_y=max_abs_y,
        asset_cfg=asset_cfg,
    ).to(torch.float32)


def stair_success_with_rear_feet_reward(
    env,
    target_x: float,
    target_z: float,
    first_riser_x: float,
    step_height_range: tuple[float, float] | None = None,
    num_steps: int = 1,
    target_base_clearance: float = 0.24,
    foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rear_foot_indices: tuple[int, ...] = (2, 3),
    foot_contact_sensor_cfg: SceneEntityCfg | None = None,
    rear_contact_threshold: float = 0.5,
    min_rear_contacts: int = 1,
    foot_x_margin: float = 0.0,
    foot_z_margin: float = 0.06,
    foot_z_upper_margin: float = 0.16,
    foot_lateral_tolerance: float = 0.65,
    trunk_sensor_cfg: SceneEntityCfg | None = None,
    trunk_contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.55,
    max_roll_projected: float = 0.55,
    max_yaw: float | None = None,
    max_abs_y: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse reward for reaching the stair with the base and rear feet in a stable pose."""
    return stair_climb_success_with_rear_feet(
        env,
        target_x=target_x,
        target_z=target_z,
        first_riser_x=first_riser_x,
        step_height_range=step_height_range,
        num_steps=num_steps,
        target_base_clearance=target_base_clearance,
        foot_asset_cfg=foot_asset_cfg,
        rear_foot_indices=rear_foot_indices,
        foot_contact_sensor_cfg=foot_contact_sensor_cfg,
        rear_contact_threshold=rear_contact_threshold,
        min_rear_contacts=min_rear_contacts,
        foot_x_margin=foot_x_margin,
        foot_z_margin=foot_z_margin,
        foot_z_upper_margin=foot_z_upper_margin,
        foot_lateral_tolerance=foot_lateral_tolerance,
        trunk_sensor_cfg=trunk_sensor_cfg,
        trunk_contact_threshold=trunk_contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
        max_yaw=max_yaw,
        max_abs_y=max_abs_y,
        asset_cfg=asset_cfg,
    ).to(torch.float32)


def stair_front_feet_on_step_success_reward(
    env,
    target_x: float,
    first_riser_x: float,
    step_height_range: tuple[float, float],
    step_index: int = 1,
    target_base_clearance: float = 0.15,
    foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    front_foot_indices: tuple[int, ...] = (0, 1),
    foot_contact_sensor_cfg: SceneEntityCfg | None = None,
    front_contact_threshold: float = 0.5,
    min_front_feet: int = 1,
    min_front_contacts: int = 1,
    foot_x_margin: float = -0.01,
    foot_z_margin: float = 0.01,
    foot_z_upper_margin: float = 0.13,
    foot_lateral_tolerance: float = 0.34,
    trunk_sensor_cfg: SceneEntityCfg | None = None,
    trunk_contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.76,
    max_roll_projected: float = 0.52,
    max_yaw: float | None = None,
    max_abs_y: float | None = None,
    min_steps: int = 4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse reward for the edge-commit subtask: front feet must actually touch the tread."""
    return stair_front_feet_on_step_success(
        env,
        target_x=target_x,
        first_riser_x=first_riser_x,
        step_height_range=step_height_range,
        step_index=step_index,
        target_base_clearance=target_base_clearance,
        foot_asset_cfg=foot_asset_cfg,
        front_foot_indices=front_foot_indices,
        foot_contact_sensor_cfg=foot_contact_sensor_cfg,
        front_contact_threshold=front_contact_threshold,
        min_front_feet=min_front_feet,
        min_front_contacts=min_front_contacts,
        foot_x_margin=foot_x_margin,
        foot_z_margin=foot_z_margin,
        foot_z_upper_margin=foot_z_upper_margin,
        foot_lateral_tolerance=foot_lateral_tolerance,
        trunk_sensor_cfg=trunk_sensor_cfg,
        trunk_contact_threshold=trunk_contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
        max_yaw=max_yaw,
        max_abs_y=max_abs_y,
        min_steps=min_steps,
        asset_cfg=asset_cfg,
    ).to(torch.float32)


def stair_climb_success(
    env,
    target_x: float,
    target_z: float,
    step_height_range: tuple[float, float] | None = None,
    num_steps: int = 3,
    target_base_clearance: float = 0.30,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.55,
    max_roll_projected: float = 0.55,
    max_yaw: float | None = None,
    max_abs_y: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate successfully once the robot reaches the upper platform in a stable pose."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    if step_height_range is None:
        target_height = torch.full_like(root_pos[:, 2], target_z)
    else:
        target_height = float(num_steps) * _terrain_level_step_height(env, step_height_range) + target_base_clearance
    stable_pitch = torch.abs(asset.data.projected_gravity_b[:, 0]) < max_pitch_projected
    stable_roll = torch.abs(asset.data.projected_gravity_b[:, 1]) < max_roll_projected
    if max_yaw is None:
        stable_yaw = torch.ones_like(stable_roll)
    else:
        _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        stable_yaw = torch.abs(yaw) < max_yaw
    if max_abs_y is None:
        centered = torch.ones_like(stable_roll)
    else:
        centered = torch.abs(root_pos[:, 1]) <= max_abs_y
    if sensor_cfg is None:
        no_trunk_contact = torch.ones_like(stable_roll)
    else:
        no_trunk_contact = ~torch.any(_foot_contact(env, sensor_cfg, contact_threshold), dim=1)
    return (
        (root_pos[:, 0] >= target_x)
        & (root_pos[:, 2] >= target_height)
        & stable_pitch
        & stable_roll
        & stable_yaw
        & centered
        & no_trunk_contact
    )


def stair_front_feet_on_step_success(
    env,
    target_x: float,
    first_riser_x: float,
    step_height_range: tuple[float, float],
    step_index: int = 1,
    target_base_clearance: float = 0.15,
    foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    front_foot_indices: tuple[int, ...] = (0, 1),
    foot_contact_sensor_cfg: SceneEntityCfg | None = None,
    front_contact_threshold: float = 0.5,
    min_front_feet: int = 1,
    min_front_contacts: int = 1,
    foot_x_margin: float = -0.01,
    foot_z_margin: float = 0.01,
    foot_z_upper_margin: float = 0.13,
    foot_lateral_tolerance: float = 0.34,
    trunk_sensor_cfg: SceneEntityCfg | None = None,
    trunk_contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.76,
    max_roll_projected: float = 0.52,
    max_yaw: float | None = None,
    max_abs_y: float | None = None,
    min_steps: int = 4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the body crosses the first riser and front feet contact the first tread."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    step_height = float(step_index) * _terrain_level_step_height(env, step_height_range)
    min_base_z = step_height + target_base_clearance

    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])
    stable_pitch = gravity_xy[:, 0] < max_pitch_projected
    stable_roll = gravity_xy[:, 1] < max_roll_projected
    if max_yaw is None:
        stable_yaw = torch.ones_like(stable_roll)
    else:
        _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        stable_yaw = torch.abs(yaw) < max_yaw
    if max_abs_y is None:
        centered = torch.ones_like(stable_roll)
    else:
        centered = torch.abs(root_pos[:, 1]) <= max_abs_y
    if trunk_sensor_cfg is None:
        no_trunk_contact = torch.ones_like(stable_roll)
    else:
        no_trunk_contact = ~torch.any(_foot_contact(env, trunk_sensor_cfg, trunk_contact_threshold), dim=1)

    foot_asset = env.scene[foot_asset_cfg.name]
    foot_pos = foot_asset.data.body_pos_w[:, foot_asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    front_feet = foot_pos[:, list(front_foot_indices), :]
    foot_x_ok = front_feet[:, :, 0] >= first_riser_x + foot_x_margin
    foot_z_ok = (front_feet[:, :, 2] >= step_height.unsqueeze(1) - foot_z_margin) & (
        front_feet[:, :, 2] <= step_height.unsqueeze(1) + foot_z_upper_margin
    )
    foot_y_ok = torch.abs(front_feet[:, :, 1]) <= foot_lateral_tolerance
    front_pose_count = torch.sum((foot_x_ok & foot_z_ok & foot_y_ok).to(torch.int64), dim=1)
    front_pose_ok = front_pose_count >= min_front_feet

    if foot_contact_sensor_cfg is None or min_front_contacts <= 0:
        front_contact_ok = torch.ones_like(front_pose_ok)
    else:
        contact = _foot_contact(env, foot_contact_sensor_cfg, front_contact_threshold).to(torch.int64)
        contact_count = torch.sum(contact[:, list(front_foot_indices)], dim=1)
        front_contact_ok = contact_count >= min_front_contacts

    return (
        (env.episode_length_buf >= min_steps)
        & (root_pos[:, 0] >= target_x)
        & (root_pos[:, 2] >= min_base_z)
        & stable_pitch
        & stable_roll
        & stable_yaw
        & centered
        & no_trunk_contact
        & front_pose_ok
        & front_contact_ok
    )


def stair_climb_success_with_rear_feet(
    env,
    target_x: float,
    target_z: float,
    first_riser_x: float,
    step_height_range: tuple[float, float] | None = None,
    num_steps: int = 1,
    target_base_clearance: float = 0.24,
    foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rear_foot_indices: tuple[int, ...] = (2, 3),
    foot_contact_sensor_cfg: SceneEntityCfg | None = None,
    rear_contact_threshold: float = 0.5,
    min_rear_contacts: int = 1,
    foot_x_margin: float = 0.0,
    foot_z_margin: float = 0.06,
    foot_z_upper_margin: float = 0.16,
    foot_lateral_tolerance: float = 0.65,
    trunk_sensor_cfg: SceneEntityCfg | None = None,
    trunk_contact_threshold: float = 1.0,
    max_pitch_projected: float = 0.55,
    max_roll_projected: float = 0.55,
    max_yaw: float | None = None,
    max_abs_y: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate successfully only when the base and rear feet have completed the step."""
    base_success = stair_climb_success(
        env,
        target_x=target_x,
        target_z=target_z,
        step_height_range=step_height_range,
        num_steps=num_steps,
        target_base_clearance=target_base_clearance,
        sensor_cfg=trunk_sensor_cfg,
        contact_threshold=trunk_contact_threshold,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
        max_yaw=max_yaw,
        max_abs_y=max_abs_y,
        asset_cfg=asset_cfg,
    )

    foot_asset = env.scene[foot_asset_cfg.name]
    foot_pos = foot_asset.data.body_pos_w[:, foot_asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    rear_feet = foot_pos[:, list(rear_foot_indices), :]
    if step_height_range is None:
        tread_height = torch.clamp(
            torch.full_like(rear_feet[:, 0, 2], target_z - target_base_clearance),
            min=0.0,
        )
    else:
        tread_height = float(num_steps) * _terrain_level_step_height(env, step_height_range)

    rear_x_ok = rear_feet[:, :, 0] >= first_riser_x + foot_x_margin
    rear_z_ok = (rear_feet[:, :, 2] >= tread_height.unsqueeze(1) - foot_z_margin) & (
        rear_feet[:, :, 2] <= tread_height.unsqueeze(1) + foot_z_upper_margin
    )
    rear_lateral_ok = torch.abs(rear_feet[:, :, 1]) <= foot_lateral_tolerance
    rear_pose_ok = torch.all(rear_x_ok & rear_z_ok & rear_lateral_ok, dim=1)

    if foot_contact_sensor_cfg is None or min_rear_contacts <= 0:
        rear_contact_ok = torch.ones_like(base_success)
    else:
        contact = _foot_contact(env, foot_contact_sensor_cfg, rear_contact_threshold).to(torch.int64)
        rear_contact_count = torch.sum(contact[:, list(rear_foot_indices)], dim=1)
        rear_contact_ok = rear_contact_count >= min_rear_contacts

    return base_success & rear_pose_ok & rear_contact_ok


def stair_bridge_success(
    env,
    target_x: float,
    min_base_height: float = 0.24,
    max_abs_y: float | None = None,
    max_pitch_projected: float = 0.35,
    max_roll_projected: float = 0.28,
    max_yaw: float = 0.45,
    min_steps: int = 8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate successfully when the robot reaches a stable pre-climb handoff pose."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])
    centered = torch.ones_like(root_pos[:, 0], dtype=torch.bool)
    if max_abs_y is not None:
        centered = torch.abs(root_pos[:, 1]) <= max_abs_y
    return (
        (env.episode_length_buf >= min_steps)
        & (root_pos[:, 0] >= target_x)
        & centered
        & (root_pos[:, 2] >= min_base_height)
        & (gravity_xy[:, 0] < max_pitch_projected)
        & (gravity_xy[:, 1] < max_roll_projected)
        & (torch.abs(yaw) < max_yaw)
    )


def stair_bridge_success_reward(
    env,
    target_x: float,
    min_base_height: float = 0.24,
    max_abs_y: float | None = None,
    max_pitch_projected: float = 0.35,
    max_roll_projected: float = 0.28,
    max_yaw: float = 0.45,
    min_steps: int = 8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse reward for a stable pre-climb bridge handoff."""
    return stair_bridge_success(
        env,
        target_x=target_x,
        min_base_height=min_base_height,
        max_abs_y=max_abs_y,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
        max_yaw=max_yaw,
        min_steps=min_steps,
        asset_cfg=asset_cfg,
    ).to(torch.float32)


def stair_bridge_handoff_reward(
    env,
    target_x: float,
    target_base_height: float = 0.32,
    x_std: float = 0.18,
    y_std: float | None = None,
    height_std: float = 0.12,
    yaw_std: float = 0.22,
    pitch_std: float = 0.24,
    roll_std: float = 0.20,
    max_speed: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for reaching a stable bridge handoff state before the final climb."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])
    x_error = torch.clamp(target_x - root_pos[:, 0], min=0.0)
    height_error = root_pos[:, 2] - target_base_height
    speed = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    y_term = torch.zeros_like(x_error)
    if y_std is not None:
        y_term = torch.square(root_pos[:, 1] / y_std)
    pose_score = torch.exp(
        -torch.square(x_error / x_std)
        - y_term
        - torch.square(height_error / height_std)
        - torch.square(yaw / yaw_std)
        - torch.square(gravity_xy[:, 0] / pitch_std)
        - torch.square(gravity_xy[:, 1] / roll_std)
    )
    speed_gate = torch.clamp(speed / max_speed, min=0.0, max=1.0)
    return pose_score * (0.35 + 0.65 * speed_gate)


def stair_bridge_forward_band_reward(
    env,
    start_x: float,
    target_x: float,
    max_speed: float = 0.45,
    yaw_allowance: float = 0.35,
    roll_allowance: float = 0.32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward controlled forward movement in the bridge band while preserving heading."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    roll = torch.abs(asset.data.projected_gravity_b[:, 1])
    in_band = ((root_pos[:, 0] >= start_x) & (root_pos[:, 0] <= target_x + 0.08)).to(torch.float32)
    progress = torch.clamp((root_pos[:, 0] - start_x) / max(target_x - start_x, 1.0e-3), min=0.0, max=1.0)
    speed = torch.clamp(asset.data.root_lin_vel_w[:, 0], min=0.0, max=max_speed) / max_speed
    heading_gate = torch.exp(-torch.square(torch.clamp(torch.abs(yaw) - yaw_allowance, min=0.0) / 0.25))
    roll_gate = torch.exp(-torch.square(torch.clamp(roll - roll_allowance, min=0.0) / 0.20))
    return in_band * (0.55 * progress + 0.45 * speed) * heading_gate * roll_gate


def stair_bridge_xz_progress_reward(
    env,
    start_x: float,
    target_x: float,
    target_base_height: float,
    base_z: float = 0.27,
    y_std: float = 0.12,
    yaw_std: float = 0.22,
    pitch_allowance: float = 0.55,
    roll_allowance: float = 0.32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward progress only when forward motion and base lift happen together."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])

    x_progress = torch.clamp((root_pos[:, 0] - start_x) / max(target_x - start_x, 1.0e-3), min=0.0, max=1.0)
    z_progress = torch.clamp(
        (root_pos[:, 2] - base_z) / max(target_base_height - base_z, 1.0e-3),
        min=0.0,
        max=1.0,
    )
    center_gate = torch.exp(-torch.square(root_pos[:, 1] / y_std))
    yaw_gate = torch.exp(-torch.square(yaw / yaw_std))
    pitch_gate = torch.exp(-torch.square(torch.clamp(gravity_xy[:, 0] - pitch_allowance, min=0.0) / 0.18))
    roll_gate = torch.exp(-torch.square(torch.clamp(gravity_xy[:, 1] - roll_allowance, min=0.0) / 0.16))
    return x_progress * z_progress * center_gate * yaw_gate * pitch_gate * roll_gate


def stair_bridge_height_hold_reward(
    env,
    x_gate: float,
    target_x: float,
    target_base_height: float,
    height_std: float = 0.045,
    y_std: float = 0.11,
    yaw_std: float = 0.18,
    pitch_allowance: float = 0.55,
    roll_allowance: float = 0.32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward carrying the lifted base forward instead of dropping after the initial pop-up."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])

    in_band = torch.sigmoid(18.0 * (root_pos[:, 0] - x_gate)) * torch.sigmoid(18.0 * (target_x + 0.05 - root_pos[:, 0]))
    x_score = torch.clamp((root_pos[:, 0] - x_gate) / max(target_x - x_gate, 1.0e-3), min=0.0, max=1.0)
    height_score = torch.exp(-torch.square(torch.clamp(target_base_height - root_pos[:, 2], min=0.0) / height_std))
    center_gate = torch.exp(-torch.square(root_pos[:, 1] / y_std))
    yaw_gate = torch.exp(-torch.square(yaw / yaw_std))
    pitch_gate = torch.exp(-torch.square(torch.clamp(gravity_xy[:, 0] - pitch_allowance, min=0.0) / 0.18))
    roll_gate = torch.exp(-torch.square(torch.clamp(gravity_xy[:, 1] - roll_allowance, min=0.0) / 0.16))
    return in_band * (0.35 + 0.65 * x_score) * height_score * center_gate * yaw_gate * pitch_gate * roll_gate


def stair_handoff_success(
    env,
    target_x: float,
    max_x: float | None = None,
    min_base_height: float = 0.27,
    max_abs_y: float = 0.18,
    max_pitch_projected: float = 0.28,
    max_roll_projected: float = 0.22,
    max_yaw: float = 0.25,
    min_steps: int = 12,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate successfully at a stable pre-climb handoff pose before the riser."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])
    x_ok = root_pos[:, 0] >= target_x
    if max_x is not None:
        x_ok &= root_pos[:, 0] <= max_x
    if sensor_cfg is None:
        no_trunk_contact = torch.ones_like(x_ok)
    else:
        no_trunk_contact = ~torch.any(_foot_contact(env, sensor_cfg, contact_threshold), dim=1)
    return (
        (env.episode_length_buf >= min_steps)
        & x_ok
        & (root_pos[:, 2] >= min_base_height)
        & (torch.abs(root_pos[:, 1]) <= max_abs_y)
        & (gravity_xy[:, 0] < max_pitch_projected)
        & (gravity_xy[:, 1] < max_roll_projected)
        & (torch.abs(yaw) < max_yaw)
        & no_trunk_contact
    )


def stair_handoff_success_reward(
    env,
    target_x: float,
    max_x: float | None = None,
    min_base_height: float = 0.27,
    max_abs_y: float = 0.18,
    max_pitch_projected: float = 0.28,
    max_roll_projected: float = 0.22,
    max_yaw: float = 0.25,
    min_steps: int = 12,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse reward for reaching a stable handoff pose before the stair policy takes over."""
    return stair_handoff_success(
        env,
        target_x=target_x,
        max_x=max_x,
        min_base_height=min_base_height,
        max_abs_y=max_abs_y,
        max_pitch_projected=max_pitch_projected,
        max_roll_projected=max_roll_projected,
        max_yaw=max_yaw,
        min_steps=min_steps,
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        asset_cfg=asset_cfg,
    ).to(torch.float32)


def stair_handoff_pose_reward(
    env,
    target_x: float,
    target_base_height: float = 0.31,
    x_std: float = 0.22,
    y_std: float = 0.16,
    height_std: float = 0.10,
    yaw_std: float = 0.20,
    pitch_std: float = 0.22,
    roll_std: float = 0.18,
    max_speed: float = 0.38,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense score for approaching the switch point in a centered, upright pose."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])
    x_error = root_pos[:, 0] - target_x
    height_error = root_pos[:, 2] - target_base_height
    pose_score = torch.exp(
        -torch.square(x_error / x_std)
        - torch.square(root_pos[:, 1] / y_std)
        - torch.square(height_error / height_std)
        - torch.square(yaw / yaw_std)
        - torch.square(gravity_xy[:, 0] / pitch_std)
        - torch.square(gravity_xy[:, 1] / roll_std)
    )
    speed_score = torch.clamp(asset.data.root_lin_vel_w[:, 0], min=0.0, max=max_speed) / max_speed
    return pose_score * (0.35 + 0.65 * speed_score)


def stair_handoff_progress_reward(
    env,
    start_x: float,
    target_x: float,
    max_speed: float = 0.38,
    yaw_allowance: float = 0.12,
    roll_allowance: float = 0.22,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward controlled approach progress while preserving a climbable heading."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])
    active = (root_pos[:, 0] <= target_x + 0.06).to(torch.float32)
    progress = torch.clamp((root_pos[:, 0] - start_x) / max(target_x - start_x, 1.0e-3), min=0.0, max=1.0)
    speed = torch.clamp(asset.data.root_lin_vel_w[:, 0], min=0.0, max=max_speed) / max_speed
    yaw_gate = torch.exp(-torch.square(torch.clamp(torch.abs(yaw) - yaw_allowance, min=0.0) / 0.20))
    roll_gate = torch.exp(-torch.square(torch.clamp(gravity_xy[:, 1] - roll_allowance, min=0.0) / 0.18))
    return active * (0.6 * progress + 0.4 * speed) * yaw_gate * roll_gate


def stair_handoff_overshoot_penalty(
    env,
    first_riser_x: float,
    margin: float = -0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize driving the trunk into or beyond the riser before a valid handoff."""
    asset = env.scene[asset_cfg.name]
    root_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return torch.sigmoid(28.0 * (root_x - (first_riser_x + margin)))


def stair_handoff_overshoot(
    env,
    overshoot_x: float,
    grace_time_s: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate approach episodes that pass the intended handoff zone."""
    asset = env.scene[asset_cfg.name]
    root_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    past_grace = env.episode_length_buf.to(torch.float32) * env.step_dt > grace_time_s
    return past_grace & (root_x > overshoot_x)


def stair_fall(
    env,
    min_base_height: float = 0.22,
    max_lateral_offset: float = 1.15,
    max_pitch_projected: float = 0.88,
    max_roll_projected: float = 0.68,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if the robot has fallen low or left the useful stair corridor."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    gravity_xy = torch.abs(asset.data.projected_gravity_b[:, :2])
    bad_attitude = (gravity_xy[:, 0] > max_pitch_projected) | (gravity_xy[:, 1] > max_roll_projected)
    return (root_pos[:, 2] < min_base_height) | (torch.abs(root_pos[:, 1]) > max_lateral_offset) | bad_attitude


def trunk_contact_failure(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    grace_time_s: float = 0.4,
) -> torch.Tensor:
    """Terminate episodes where the trunk hits the terrain instead of the feet climbing."""
    contact = torch.any(_foot_contact(env, sensor_cfg, threshold), dim=1)
    past_grace = env.episode_length_buf.to(torch.float32) * env.step_dt > grace_time_s
    return contact & past_grace


def stair_climb_terrain_levels(
    env,
    env_ids,
    target_x: float,
    target_z: float,
    step_height_range: tuple[float, float] | None = None,
    num_steps: int = 3,
    target_base_clearance: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Advance terrain difficulty on successful climbs and lower it after poor progress."""
    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        return torch.zeros((), device=env.device)

    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    if step_height_range is None:
        target_height = torch.full_like(root_pos[:, 2], target_z)
    else:
        all_target_heights = float(num_steps) * _terrain_level_step_height(env, step_height_range) + target_base_clearance
        target_height = all_target_heights[env_ids]
    move_up = (root_pos[:, 0] >= target_x) & (root_pos[:, 2] >= target_height)
    move_down = (root_pos[:, 0] < 0.35 * target_x) & (~move_up)
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
