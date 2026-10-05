from __future__ import annotations

import math

import mujoco
import numpy as np

from .contract import AdapterContract
from .model_map import ModelMap
from .observation import KinematicState, collect_kinematic_state


VIRTUAL_LEG_JOINTS = (("jIJ", "jwheel_left"), ("jAB", "jwheel_right"))
LOOP_SITE_PAIRS = (
    ("site_cf_gh_a1", "site_cf_gh_b1"),
    ("site_cf_gh_a2", "site_cf_gh_b2"),
    ("site_kn_op_a1", "site_kn_op_b1"),
    ("site_kn_op_a2", "site_kn_op_b2"),
    ("site_ec_ag_a1", "site_ec_ag_b1"),
    ("site_ec_ag_a2", "site_ec_ag_b2"),
    ("site_mk_io_a1", "site_mk_io_b1"),
    ("site_mk_io_a2", "site_mk_io_b2"),
)


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model is missing {name}")
    return object_id


def _wrap_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _roll_pitch_yaw(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(abs(pitch) - math.pi / 2.0) < 1.0e-9:
        roll = math.atan2(-rotation[0, 1], rotation[1, 1])
        yaw = 0.0
    else:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return roll, pitch, yaw


def _virtual_leg_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    model_map: ModelMap,
    contract: AdapterContract,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_world_from_body = np.asarray(data.xmat[model_map.base_body_id], dtype=np.float64).reshape(3, 3)
    lengths = []
    angles = []
    for hip_name, wheel_name in VIRTUAL_LEG_JOINTS:
        hip_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, hip_name)
        wheel_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, wheel_name)
        vector_world = data.xanchor[wheel_id] - data.xanchor[hip_id]
        vector_body = rotation_world_from_body.T @ vector_world
        s_w = float(vector_body[1])
        d_w = float(-vector_body[2])
        lengths.append(math.hypot(s_w, d_w))
        angles.append(math.atan2(d_w, s_w))
    return np.asarray(lengths), np.asarray(angles)


def _max_loop_error(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    errors = []
    for first, second in LOOP_SITE_PAIRS:
        first_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, first)
        second_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, second)
        errors.append(np.linalg.norm(data.site_xpos[first_id] - data.site_xpos[second_id]))
    return float(max(errors))


def collect_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    model_map: ModelMap,
    contract: AdapterContract,
    state: KinematicState | None = None,
) -> dict[str, float]:
    if state is None:
        state = collect_kinematic_state(model, data, model_map, contract)
    rotation_world_from_body = np.asarray(data.xmat[model_map.base_body_id], dtype=np.float64).reshape(3, 3)
    rotation_control_world_from_body = (
        contract.r_control_from_mujoco @ rotation_world_from_body @ contract.r_control_from_mujoco.T
    )
    roll, pitch, yaw = _roll_pitch_yaw(rotation_control_world_from_body)
    lengths, angles = _virtual_leg_metrics(model, data, model_map, contract)
    tilt = math.acos(float(np.clip(-state.projected_gravity_control[2], -1.0, 1.0)))
    metrics = {
        "base_height_m": state.base_height_m,
        "roll_rad": roll,
        "pitch_rad": pitch,
        "yaw_rad": yaw,
        "vx_mps": float(state.com_linear_velocity_control[0]),
        "yaw_rate_rad_s": float(state.angular_velocity_control[2]),
        "phi0_left_rad": float(angles[0]),
        "phi0_right_rad": float(angles[1]),
        "phi0_delta_abs_rad": abs(_wrap_to_pi(float(angles[0] - angles[1]))),
        "l0_left_m": float(lengths[0]),
        "l0_right_m": float(lengths[1]),
        "max_loop_closure_error_m": _max_loop_error(model, data),
        "tilt_rad": tilt,
        "root_linear_speed_mps": float(np.linalg.norm(state.com_linear_velocity_control)),
        "root_angular_speed_rad_s": float(np.linalg.norm(state.angular_velocity_control)),
        "max_controlled_joint_speed_rad_s": float(np.max(np.abs(state.joint_velocity_control))),
        "max_hinge_speed_rad_s": float(np.max(np.abs(data.qvel[6:]))),
    }
    if not np.isfinite(np.asarray(list(metrics.values()), dtype=np.float64)).all():
        raise FloatingPointError("MuJoCo metrics contain NaN or Inf")
    return metrics
