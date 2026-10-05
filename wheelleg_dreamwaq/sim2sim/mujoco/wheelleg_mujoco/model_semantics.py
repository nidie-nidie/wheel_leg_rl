from __future__ import annotations

import hashlib
import json

import mujoco

from .model_map import ACTUATOR_NAMES, CANONICAL_JOINT_ORDER


DYNAMICS_SEMANTICS_VERSION = "MujocoDynamicsSemanticsV1"


def stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    value = mujoco.mj_id2name(model, object_type, index)
    return value if value is not None else f"unnamed_{object_type.name}_{index}"


def compiled_model_semantics(model: mujoco.MjModel, reset_keyframe: str = "contract_v4_reset") -> dict:
    joints = []
    for joint_id in range(model.njnt):
        dof_start = int(model.jnt_dofadr[joint_id])
        dof_count = 6 if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE else 1
        joints.append(
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                "type": int(model.jnt_type[joint_id]),
                "qpos_address": int(model.jnt_qposadr[joint_id]),
                "dof_address": dof_start,
                "axis": model.jnt_axis[joint_id].tolist(),
                "position": model.jnt_pos[joint_id].tolist(),
                "limited": bool(model.jnt_limited[joint_id]),
                "range": model.jnt_range[joint_id].tolist(),
                "stiffness": float(model.jnt_stiffness[joint_id]),
                "damping": model.dof_damping[dof_start : dof_start + dof_count].tolist(),
                "armature": model.dof_armature[dof_start : dof_start + dof_count].tolist(),
                "frictionloss": model.dof_frictionloss[dof_start : dof_start + dof_count].tolist(),
            }
        )

    bodies = []
    for body_id in range(model.nbody):
        bodies.append(
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "parent_id": int(model.body_parentid[body_id]),
                "mass": float(model.body_mass[body_id]),
                "inertia": model.body_inertia[body_id].tolist(),
                "inertial_position": model.body_ipos[body_id].tolist(),
                "inertial_quaternion": model.body_iquat[body_id].tolist(),
            }
        )

    actuators = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuators.append(
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
                "joint": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                "transmission_type": int(model.actuator_trntype[actuator_id]),
                "gear": model.actuator_gear[actuator_id].tolist(),
                "control_limited": bool(model.actuator_ctrllimited[actuator_id]),
                "control_range": model.actuator_ctrlrange[actuator_id].tolist(),
            }
        )

    equalities = []
    for equality_id in range(model.neq):
        equalities.append(
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id),
                "type": int(model.eq_type[equality_id]),
                "object1_id": int(model.eq_obj1id[equality_id]),
                "object2_id": int(model.eq_obj2id[equality_id]),
                "active_initial": bool(model.eq_active0[equality_id]),
                "solref": model.eq_solref[equality_id].tolist(),
                "solimp": model.eq_solimp[equality_id].tolist(),
                "data": model.eq_data[equality_id].tolist(),
            }
        )

    contact_pairs = []
    for pair_id in range(model.npair):
        contact_pairs.append(
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_PAIR, pair_id),
                "geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[pair_id])),
                "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[pair_id])),
                "dimension": int(model.pair_dim[pair_id]),
                "friction": model.pair_friction[pair_id].tolist(),
                "solref": model.pair_solref[pair_id].tolist(),
                "solreffriction": model.pair_solreffriction[pair_id].tolist(),
                "solimp": model.pair_solimp[pair_id].tolist(),
            }
        )

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, reset_keyframe)
    if key_id < 0:
        raise ValueError(f"MuJoCo model is missing reset keyframe {reset_keyframe!r}")
    hinge_positions = {}
    for joint in joints:
        if joint["type"] == int(mujoco.mjtJoint.mjJNT_HINGE):
            hinge_positions[joint["name"]] = float(model.key_qpos[key_id, joint["qpos_address"]])

    return {
        "schema_version": DYNAMICS_SEMANTICS_VERSION,
        "dimensions": {
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "neq": int(model.neq),
            "npair": int(model.npair),
        },
        "options": {
            "timestep": float(model.opt.timestep),
            "gravity": model.opt.gravity.tolist(),
            "solver": int(model.opt.solver),
            "iterations": int(model.opt.iterations),
            "line_search_iterations": int(model.opt.ls_iterations),
            "tolerance": float(model.opt.tolerance),
        },
        "controlled_joint_order": list(CANONICAL_JOINT_ORDER),
        "controlled_actuator_order": list(ACTUATOR_NAMES),
        "joints": joints,
        "bodies": bodies,
        "actuators": actuators,
        "equalities": equalities,
        "contact_pairs": contact_pairs,
        "reset": {
            "keyframe": reset_keyframe,
            "base_position": model.key_qpos[key_id, :3].tolist(),
            "base_quaternion": model.key_qpos[key_id, 3:7].tolist(),
            "hinge_positions": hinge_positions,
            "qvel": model.key_qvel[key_id].tolist(),
        },
    }
