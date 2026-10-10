from __future__ import annotations

import torch


ACTOR_OBS_SCHEMA_VERSION = "ActorObsV1"
CRITIC_OBS_SCHEMA_VERSION = "CriticObsV1"
HISTORY_LAYOUT_VERSION = "HistoryLayoutV1"
VELOCITY_TARGET_VERSION = "VelocityTargetV1"
PROPRIO_RECONSTRUCTION_TARGET_VERSION = "ProprioReconstructionTargetV1"
ACTOR_OBS_DIM = 25
CRITIC_OBS_DIM = 41
HISTORY_LENGTH = 5
HISTORY_OBS_DIM = HISTORY_LENGTH * ACTOR_OBS_DIM
VELOCITY_TARGET_DIM = 3
PROPRIO_RECONSTRUCTION_TARGET_DIM = 16


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


PROPRIO_RECONSTRUCTION_SOURCE_SLICES = (
    slice(0, 6),
    slice(9, 19),
)


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


def initialize_frame_major_history(current_observation: torch.Tensor) -> torch.Tensor:
    """Fill all five history frames with one current ActorObsV1 observation."""

    if current_observation.ndim != 2:
        raise ValueError(f"current observation must be rank 2, got rank {current_observation.ndim}")
    _require_last_dim("ActorObsV1", current_observation, ACTOR_OBS_DIM)
    return current_observation.unsqueeze(1).expand(-1, HISTORY_LENGTH, -1).clone().contiguous()


def flatten_frame_major_history(history: torch.Tensor) -> torch.Tensor:
    """Flatten canonical ``[N,5,25]`` history without changing frame order."""

    if history.ndim != 3 or tuple(history.shape[1:]) != (HISTORY_LENGTH, ACTOR_OBS_DIM):
        raise ValueError(
            f"history must have shape [N,{HISTORY_LENGTH},{ACTOR_OBS_DIM}], got {tuple(history.shape)}"
        )
    return history.contiguous().reshape(history.shape[0], HISTORY_OBS_DIM)


def append_frame_major_history(
    history: torch.Tensor,
    next_observation: torch.Tensor,
    dones: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append one frame and reset completed environments to five copies of the reset observation."""

    if history.ndim != 3 or tuple(history.shape[1:]) != (HISTORY_LENGTH, ACTOR_OBS_DIM):
        raise ValueError(
            f"history must have shape [N,{HISTORY_LENGTH},{ACTOR_OBS_DIM}], got {tuple(history.shape)}"
        )
    if next_observation.ndim != 2 or tuple(next_observation.shape) != (history.shape[0], ACTOR_OBS_DIM):
        raise ValueError(
            f"next observation must have shape [{history.shape[0]},{ACTOR_OBS_DIM}], "
            f"got {tuple(next_observation.shape)}"
        )
    advanced = torch.cat((history[:, 1:], next_observation.unsqueeze(1)), dim=1)
    if dones is None:
        return advanced.contiguous()
    done_mask = dones.to(dtype=torch.bool).reshape(-1)
    if done_mask.shape[0] != history.shape[0]:
        raise ValueError(f"dones must have {history.shape[0]} elements, got {done_mask.shape[0]}")
    if done_mask.any():
        reset_history = initialize_frame_major_history(next_observation)
        advanced = torch.where(done_mask.view(-1, 1, 1), reset_history, advanced)
    return advanced.contiguous()


def extract_velocity_target(critic_observation: torch.Tensor) -> torch.Tensor:
    if critic_observation.ndim != 2:
        raise ValueError(f"critic observation must be rank 2, got rank {critic_observation.ndim}")
    _require_last_dim("CriticObsV1", critic_observation, CRITIC_OBS_DIM)
    target = critic_observation[:, CriticObsSlices.ROOT_LINEAR_VELOCITY]
    _require_last_dim("VelocityTargetV1", target, VELOCITY_TARGET_DIM)
    return target


def extract_proprio_reconstruction_target(critic_observation: torch.Tensor) -> torch.Tensor:
    if critic_observation.ndim != 2:
        raise ValueError(f"critic observation must be rank 2, got rank {critic_observation.ndim}")
    _require_last_dim("CriticObsV1", critic_observation, CRITIC_OBS_DIM)
    target = torch.cat(
        [critic_observation[:, source_slice] for source_slice in PROPRIO_RECONSTRUCTION_SOURCE_SLICES],
        dim=-1,
    )
    _require_last_dim("ProprioReconstructionTargetV1", target, PROPRIO_RECONSTRUCTION_TARGET_DIM)
    return target
