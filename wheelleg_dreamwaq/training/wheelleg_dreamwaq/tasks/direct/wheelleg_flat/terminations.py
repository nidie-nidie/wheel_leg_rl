from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from wheelleg_dreamwaq.schemas.frames import upright_cosine_from_projected_gravity

from .state import WheelLegState


@dataclass(frozen=True)
class TerminationLimits:
    min_base_height: float = 0.10
    max_base_height: float = 0.40
    max_tilt_rad: float = 0.80
    max_root_linear_velocity: float = 20.0
    max_root_angular_velocity: float = 35.0
    max_joint_velocity: float = 80.0
    grace_steps: int = 10


def compute_dones(
    state: WheelLegState,
    episode_length: torch.Tensor,
    *,
    max_episode_length: int,
    limits: TerminationLimits,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    truncated = episode_length >= max_episode_length - 1
    height = state.base_height[:, 0]
    height_bad = (height < limits.min_base_height) | (height > limits.max_base_height)
    upright_cosine = upright_cosine_from_projected_gravity(state.projected_gravity)
    tilt = torch.acos(upright_cosine)
    tilt_bad = upright_cosine < math.cos(limits.max_tilt_rad)
    root_linear_bad = torch.linalg.vector_norm(state.root_com_linear_velocity, dim=-1) > limits.max_root_linear_velocity
    root_angular_bad = torch.linalg.vector_norm(state.root_angular_velocity, dim=-1) > limits.max_root_angular_velocity
    joint_velocity_bad = torch.amax(torch.abs(state.joint_velocity), dim=-1) > limits.max_joint_velocity
    invalid = torch.zeros_like(truncated)
    for value in vars(state).values():
        if isinstance(value, torch.Tensor):
            invalid |= ~torch.isfinite(value).reshape(value.shape[0], -1).all(dim=-1)
    past_grace = episode_length > limits.grace_steps
    height_terminated = past_grace & height_bad
    tilt_terminated = past_grace & tilt_bad
    root_linear_terminated = past_grace & root_linear_bad
    root_angular_terminated = past_grace & root_angular_bad
    joint_velocity_terminated = past_grace & joint_velocity_bad
    terminated = (
        invalid
        | height_terminated
        | tilt_terminated
        | root_linear_terminated
        | root_angular_terminated
        | joint_velocity_terminated
    )
    diagnostics = {
        "height": height,
        "tilt": tilt,
        "upright_cosine": upright_cosine,
        "height_bad": height_bad,
        "tilt_bad": tilt_bad,
        "root_linear_bad": root_linear_bad,
        "root_angular_bad": root_angular_bad,
        "joint_velocity_bad": joint_velocity_bad,
        "invalid": invalid,
        "height_terminated": height_terminated,
        "tilt_terminated": tilt_terminated,
        "root_linear_terminated": root_linear_terminated,
        "root_angular_terminated": root_angular_terminated,
        "joint_velocity_terminated": joint_velocity_terminated,
    }
    return terminated, truncated, diagnostics
