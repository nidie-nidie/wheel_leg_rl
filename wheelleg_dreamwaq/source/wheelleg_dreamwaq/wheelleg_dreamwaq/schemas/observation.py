from __future__ import annotations

import torch


ACTOR_OBS_SCHEMA_VERSION = "ActorObsV1"
CRITIC_OBS_SCHEMA_VERSION = "CriticObsV1"
ACTOR_OBS_DIM = 25
CRITIC_OBS_DIM = 41


class ActorObsSlices:
    ANGULAR_VELOCITY = slice(0, 3)
    PROJECTED_GRAVITY = slice(3, 6)
    COMMAND = slice(6, 9)
    LEG_POSITION_ERROR = slice(9, 13)
    JOINT_VELOCITY = slice(13, 19)
    PREVIOUS_ACTION = slice(19, 25)


class CriticObsSlices:
    ACTOR_OBS = slice(0, 25)
    ROOT_LINEAR_VELOCITY = slice(25, 28)
    BASE_HEIGHT = slice(28, 29)
    JOINT_ACCELERATION = slice(29, 35)
    APPLIED_TORQUE = slice(35, 41)


def _require_last_dim(name: str, values: torch.Tensor, dimension: int) -> None:
    if values.shape[-1] != dimension:
        raise ValueError(f"{name} must have last dimension {dimension}, got {values.shape[-1]}")


def assemble_actor_obs(
    *,
    angular_velocity: torch.Tensor,
    projected_gravity: torch.Tensor,
    command: torch.Tensor,
    leg_position_error: torch.Tensor,
    joint_velocity: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    fields = (
        ("angular_velocity", angular_velocity, 3),
        ("projected_gravity", projected_gravity, 3),
        ("command", command, 3),
        ("leg_position_error", leg_position_error, 4),
        ("joint_velocity", joint_velocity, 6),
        ("previous_action", previous_action, 6),
    )
    for name, values, dimension in fields:
        _require_last_dim(name, values, dimension)
    result = torch.cat([values for _, values, _ in fields], dim=-1)
    _require_last_dim("ActorObsV1", result, ACTOR_OBS_DIM)
    return result


def assemble_critic_obs(
    *,
    actor_obs: torch.Tensor,
    root_linear_velocity: torch.Tensor,
    base_height: torch.Tensor,
    joint_acceleration: torch.Tensor,
    applied_torque: torch.Tensor,
) -> torch.Tensor:
    fields = (
        ("actor_obs", actor_obs, 25),
        ("root_linear_velocity", root_linear_velocity, 3),
        ("base_height", base_height, 1),
        ("joint_acceleration", joint_acceleration, 6),
        ("applied_torque", applied_torque, 6),
    )
    for name, values, dimension in fields:
        _require_last_dim(name, values, dimension)
    result = torch.cat([values for _, values, _ in fields], dim=-1)
    _require_last_dim("CriticObsV1", result, CRITIC_OBS_DIM)
    return result
