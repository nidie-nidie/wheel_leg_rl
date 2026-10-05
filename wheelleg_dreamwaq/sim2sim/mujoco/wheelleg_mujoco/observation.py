from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .contract import AdapterContract
from .model_map import ModelMap


@dataclass(frozen=True)
class KinematicState:
    angular_velocity_control: np.ndarray
    projected_gravity_control: np.ndarray
    joint_position_control: np.ndarray
    joint_velocity_control: np.ndarray
    com_linear_velocity_control: np.ndarray
    base_height_m: float


def collect_kinematic_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    model_map: ModelMap,
    contract: AdapterContract,
) -> KinematicState:
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBodyCom(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        model_map.base_body_id,
    )
    linear_world = jacobian_position @ data.qvel
    angular_world = jacobian_rotation @ data.qvel
    rotation_world_from_body = np.asarray(data.xmat[model_map.base_body_id], dtype=np.float64).reshape(3, 3)
    rotation_body_from_world = rotation_world_from_body.T
    linear_body = rotation_body_from_world @ linear_world
    angular_body = rotation_body_from_world @ angular_world
    gravity_body = rotation_body_from_world @ np.array((0.0, 0.0, -1.0), dtype=np.float64)
    state = KinematicState(
        angular_velocity_control=contract.r_control_from_mujoco @ angular_body,
        projected_gravity_control=contract.r_control_from_mujoco @ gravity_body,
        joint_position_control=model_map.joint_position_control(data),
        joint_velocity_control=model_map.joint_velocity_control(data),
        com_linear_velocity_control=contract.r_control_from_mujoco @ linear_body,
        base_height_m=float(data.xipos[model_map.base_body_id, 2]),
    )
    fields = (
        state.angular_velocity_control,
        state.projected_gravity_control,
        state.joint_position_control,
        state.joint_velocity_control,
        state.com_linear_velocity_control,
        np.asarray((state.base_height_m,)),
    )
    if not all(np.isfinite(field).all() for field in fields):
        raise FloatingPointError("MuJoCo kinematic state contains NaN or Inf")
    return state


def _normalize_command(command: np.ndarray, contract: AdapterContract) -> np.ndarray:
    if command.shape != (3,):
        raise ValueError(f"CommandV1 must have shape (3,), got {command.shape}")
    normalized = np.array(
        (
            command[0] / contract.normalization["vx_max_abs"],
            command[1] / contract.normalization["yaw_rate_max_abs"],
            (command[2] - contract.normalization["nominal_base_height"])
            / contract.normalization["height_command_span"],
        ),
        dtype=np.float64,
    )
    return np.clip(normalized, -1.0, 1.0)


def build_actor_observation(
    state: KinematicState,
    command: np.ndarray,
    previous_action: np.ndarray,
    contract: AdapterContract,
) -> np.ndarray:
    command = np.asarray(command, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    if previous_action.shape != (6,):
        raise ValueError(f"ActionV1 must have shape (6,), got {previous_action.shape}")
    observation = np.concatenate(
        (
            np.clip(
                state.angular_velocity_control * contract.normalization["angular_velocity_scale"],
                -contract.normalization["angular_velocity_clip"],
                contract.normalization["angular_velocity_clip"],
            ),
            np.clip(
                state.projected_gravity_control,
                -contract.normalization["projected_gravity_clip"],
                contract.normalization["projected_gravity_clip"],
            ),
            _normalize_command(command, contract),
            np.clip(
                (state.joint_position_control[:4] - contract.q_nominal)
                * contract.normalization["leg_position_scale"],
                -contract.normalization["leg_position_clip"],
                contract.normalization["leg_position_clip"],
            ),
            np.clip(
                state.joint_velocity_control * contract.normalization["joint_velocity_scale"],
                -contract.normalization["joint_velocity_clip"],
                contract.normalization["joint_velocity_clip"],
            ),
            np.clip(previous_action, -contract.action_clip, contract.action_clip),
        )
    ).astype(np.float32)
    if observation.shape != (25,) or not np.isfinite(observation).all():
        raise FloatingPointError(f"Invalid ActorObsV1: shape={observation.shape}")
    return observation
