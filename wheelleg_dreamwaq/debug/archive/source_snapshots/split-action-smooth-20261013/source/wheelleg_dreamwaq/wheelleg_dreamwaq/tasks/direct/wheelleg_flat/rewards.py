from __future__ import annotations

from dataclasses import dataclass

import torch

from wheelleg_dreamwaq.kinematics.virtual_leg import phi0_symmetry_error
from wheelleg_dreamwaq.schemas.frames import upright_cosine_from_projected_gravity

from .state import WheelLegState


REWARD_SCHEMA_VERSION = "RewardSchemaV2"


@dataclass(frozen=True)
class RewardWeights:
    tracking_vx: float = 1.0
    tracking_vx_enhance: float = 1.0
    tracking_yaw_rate: float = 1.0
    tracking_yaw_rate_enhance: float = 1.0
    tracking_base_height: float = 1.0
    lin_vel_z: float = -1.0
    ang_vel_xy: float = -0.20
    orientation: float = -100.0
    dof_vel: float = -5.0e-5
    dof_acc: float = -2.5e-7
    torques: float = -1.0e-4
    action_rate: float = -0.01
    action_smooth: float = -0.01
    dof_pos_limits: float = -1.0
    phi0_symmetry: float = -1.0
    termination: float = -5.0
    tracking_sigma: float = 0.25
    height_sigma: float = 0.001
    soft_leg_limit: float = 0.95


def compute_reward_terms(
    state: WheelLegState,
    weights: RewardWeights,
    terminated: torch.Tensor,
) -> dict[str, torch.Tensor]:
    vx_error = torch.square(state.command[:, 0] - state.root_com_linear_velocity[:, 0])
    yaw_error = torch.square(state.command[:, 1] - state.root_angular_velocity[:, 2])
    height_error = torch.square(state.command[:, 2] - state.base_height[:, 0])
    soft_lower = state.joint_position.new_full((4,), -weights.soft_leg_limit)
    soft_upper = state.joint_position.new_full((4,), weights.soft_leg_limit)
    lower_violation = torch.clamp(soft_lower - state.joint_position[:, :4], min=0.0)
    upper_violation = torch.clamp(state.joint_position[:, :4] - soft_upper, min=0.0)
    upright_cosine = upright_cosine_from_projected_gravity(state.projected_gravity)
    return {
        "tracking_vx": torch.exp(-vx_error / weights.tracking_sigma),
        "tracking_vx_enhance": torch.exp(-vx_error / (weights.tracking_sigma * 10.0)) - 1.0,
        "tracking_yaw_rate": torch.exp(-yaw_error / weights.tracking_sigma),
        "tracking_yaw_rate_enhance": torch.exp(-yaw_error / (weights.tracking_sigma * 10.0)) - 1.0,
        "tracking_base_height": torch.exp(-height_error / weights.height_sigma),
        "lin_vel_z": torch.square(state.root_com_linear_velocity[:, 2]),
        "ang_vel_xy": torch.sum(torch.square(state.root_angular_velocity[:, :2]), dim=-1),
        "orientation": 1.0 - upright_cosine,
        "dof_vel": torch.sum(torch.square(state.joint_velocity[:, :4]), dim=-1),
        "dof_acc": torch.sum(torch.square(state.joint_acceleration), dim=-1),
        "torques": torch.sum(torch.square(state.applied_torque), dim=-1),
        "action_rate": torch.sum(
            torch.square(state.last_applied_action - state.previous_applied_action), dim=-1
        ),
        "action_smooth": torch.sum(
            torch.square(
                state.last_applied_action
                - 2.0 * state.previous_applied_action
                + state.previous_previous_applied_action
            ),
            dim=-1,
        ),
        "dof_pos_limits": torch.sum(lower_violation + upper_violation, dim=-1),
        "phi0_symmetry": phi0_symmetry_error(state.virtual_leg_phi0_true),
        "termination": terminated.float(),
    }


def compute_reward(
    state: WheelLegState,
    weights: RewardWeights,
    *,
    control_dt: float,
    terminated: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    terms = compute_reward_terms(state, weights, terminated)
    total = torch.zeros_like(terminated, dtype=state.joint_position.dtype)
    weighted: dict[str, torch.Tensor] = {}
    for name, term in terms.items():
        value = getattr(weights, name) * control_dt * term
        weighted[name] = value
        total += value
    if not torch.isfinite(total).all():
        raise RuntimeError("Reward contains NaN or Inf")
    return total, weighted
