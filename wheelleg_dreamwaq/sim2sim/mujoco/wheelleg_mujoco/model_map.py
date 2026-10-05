from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


CANONICAL_JOINT_ORDER = (
    "jIJ",
    "jIO",
    "jAB",
    "jAG",
    "jwheel_left",
    "jwheel_right",
)
CONTROLLED_JOINT_SIGNS = np.array((1.0, 1.0, 1.0, 1.0, 1.0, -1.0), dtype=np.float64)
ACTUATOR_NAMES = (
    "Left_front_joint_act",
    "Left_rear_joint_act",
    "Right_front_joint_act",
    "Right_rear_joint_act",
    "Left_Wheel_act",
    "Right_Wheel_act",
)


def _required_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model is missing required {object_type.name}: {name}")
    return object_id


@dataclass(frozen=True)
class ModelMap:
    joint_names: tuple[str, ...]
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    actuator_ids: np.ndarray
    base_body_id: int
    free_joint_id: int
    reset_keyframe_id: int

    def joint_position_mujoco(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.qpos[self.qpos_addresses], dtype=np.float64)

    def joint_velocity_mujoco(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.qvel[self.dof_addresses], dtype=np.float64)

    def joint_position_control(self, data: mujoco.MjData) -> np.ndarray:
        return self.joint_position_mujoco(data) * CONTROLLED_JOINT_SIGNS

    def joint_velocity_control(self, data: mujoco.MjData) -> np.ndarray:
        return self.joint_velocity_mujoco(data) * CONTROLLED_JOINT_SIGNS


def build_model_map(model: mujoco.MjModel) -> ModelMap:
    joint_ids = np.asarray(
        [_required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in CANONICAL_JOINT_ORDER],
        dtype=np.int32,
    )
    if any(model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE for joint_id in joint_ids):
        raise ValueError("Every ActionV1 joint must be a MuJoCo hinge")
    actuator_ids = np.asarray(
        [_required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES],
        dtype=np.int32,
    )
    if not np.array_equal(model.actuator_trnid[actuator_ids, 0], joint_ids):
        raise ValueError("Canonical actuator-to-joint mapping does not match ActionV1")
    free_joint_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    if model.jnt_type[free_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("base_free is not a MuJoCo free joint")
    return ModelMap(
        joint_names=CANONICAL_JOINT_ORDER,
        joint_ids=joint_ids,
        qpos_addresses=np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32),
        dof_addresses=np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32),
        actuator_ids=actuator_ids,
        base_body_id=_required_id(model, mujoco.mjtObj.mjOBJ_BODY, "base"),
        free_joint_id=free_joint_id,
        reset_keyframe_id=_required_id(model, mujoco.mjtObj.mjOBJ_KEY, "contract_v4_reset"),
    )
