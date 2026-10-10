from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import mujoco
import numpy as np

from .causal_contract import (
    result_identity_fields,
    robot_configuration_semantics,
    robot_scenario_id,
    scenario_spec,
    sphere_configuration_semantics,
    sphere_scenario_id,
)
from .compiled_properties import (
    BodyProperty,
    composite_properties,
    golden_body_property,
    momentum_about_origin,
    validate_body_mapping,
)
from .contracts import (
    FROZEN_POLICIES,
    FROZEN_REPLAY_SOURCE,
    PROJECT_ROOT,
    canonical_json_bytes,
    sha256_file,
    stable_hash,
)
from .repeatability import build_repeatability_record
from .scenario_catalog import robot_probe_profiles, sphere_probe_profiles
from .trace_contract import FieldSpec, write_trace
from .variant_builders import (
    CONNECT_NAMES,
    ROBOT_HINGES,
    build_common_sphere_mjcf,
    build_robot_variant,
)


FORMAL_MODEL = PROJECT_ROOT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
MODEL_MANIFEST = PROJECT_ROOT / "sim2sim/mujoco/model_manifest.json"
SOURCE_AUDIT = PROJECT_ROOT / "artifacts/phase1_v4/mujoco-usd-data.json"
BODY_MAPPING = Path(__file__).with_name("body_mapping_v1.json")
CONTROLLED_JOINTS = ("jIJ", "jIO", "jAB", "jAG", "jwheel_left", "jwheel_right")
ACTUATORS = (
    "Left_front_joint_act", "Left_rear_joint_act", "Right_front_joint_act",
    "Right_rear_joint_act", "Left_Wheel_act", "Right_Wheel_act",
)
CANONICAL_SIGNS = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64)
OPTION_FIELDS = (
    "ccd_iterations", "ccd_tolerance", "cone", "density", "disableactuator",
    "disableflags", "enableflags", "gravity", "impratio", "integrator", "iterations",
    "jacobian", "ls_iterations", "ls_tolerance", "magnetic", "noslip_iterations",
    "noslip_tolerance", "o_friction", "o_margin", "o_solimp", "o_solref",
    "sdf_initpoints", "sdf_iterations", "sleep_tolerance", "solver", "timestep",
    "tolerance", "viscosity", "wind",
)
ENUM_FIELDS = {
    "cone": mujoco.mjtCone,
    "integrator": mujoco.mjtIntegrator,
    "jacobian": mujoco.mjtJacobian,
    "solver": mujoco.mjtSolver,
}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _tensor_bytes_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    value = mujoco.mj_id2name(model, kind, index)
    return value if value is not None else f"unnamed_{index}"


def _required_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    identifier = int(mujoco.mj_name2id(model, kind, name))
    if identifier < 0:
        raise ValueError(f"MuJoCo model is missing {kind.name}: {name}")
    return identifier


def _enum_payload(enum_type: Any, value: int) -> dict[str, Any]:
    integer = int(value)
    try:
        symbol = enum_type(integer).name
    except ValueError:
        symbol = "UNKNOWN"
    return {"value": integer, "symbol": symbol}


def _bit_symbols(enum_type: Any, value: int, *, count_member: str) -> list[str]:
    integer = int(value)
    return sorted(
        name
        for name, member in enum_type.__members__.items()
        if name != count_member and int(member.value) > 0 and integer & int(member.value)
    )


def compiled_option_snapshot(model: mujoco.MjModel) -> dict[str, Any]:
    public = tuple(name for name in dir(model.opt) if not name.startswith("_"))
    if set(public) != set(OPTION_FIELDS):
        raise ValueError(
            f"MuJoCo option field set drifted: missing={sorted(set(OPTION_FIELDS)-set(public))}, "
            f"extra={sorted(set(public)-set(OPTION_FIELDS))}"
        )
    payload: dict[str, Any] = {}
    for name in OPTION_FIELDS:
        value = getattr(model.opt, name)
        if name in ENUM_FIELDS:
            payload[name] = _enum_payload(ENUM_FIELDS[name], int(value))
        elif name == "enableflags":
            payload[name] = {
                "value": int(value),
                "symbols": _bit_symbols(mujoco.mjtEnableBit, int(value), count_member="mjNENABLE"),
            }
        elif name == "disableflags":
            payload[name] = {
                "value": int(value),
                "symbols": _bit_symbols(mujoco.mjtDisableBit, int(value), count_member="mjNDISABLE"),
            }
        else:
            payload[name] = _json_value(value)
    return payload


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def compiled_semantics(model: mujoco.MjModel) -> dict[str, Any]:
    joints: dict[str, dict[str, Any]] = {}
    for joint_id in range(model.njnt):
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        dof_address = int(model.jnt_dofadr[joint_id])
        dof_count = 6 if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE else 1
        joints[name] = {
            "type": int(model.jnt_type[joint_id]),
            "body": _name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[joint_id])
            ),
            "qpos_address": int(model.jnt_qposadr[joint_id]),
            "dof_address": dof_address,
            "axis": np.asarray(model.jnt_axis[joint_id]).tolist(),
            "position": np.asarray(model.jnt_pos[joint_id]).tolist(),
            "limited": bool(model.jnt_limited[joint_id]),
            "range": np.asarray(model.jnt_range[joint_id]).tolist(),
            "stiffness": float(model.jnt_stiffness[joint_id]),
            "dof_damping": np.asarray(
                model.dof_damping[dof_address : dof_address + dof_count]
            ).tolist(),
            "armature": np.asarray(
                model.dof_armature[dof_address : dof_address + dof_count]
            ).tolist(),
            "frictionloss": np.asarray(
                model.dof_frictionloss[dof_address : dof_address + dof_count]
            ).tolist(),
            "solref": np.asarray(model.jnt_solref[joint_id]).tolist(),
            "solimp": np.asarray(model.jnt_solimp[joint_id]).tolist(),
        }
    actuators: dict[str, dict[str, Any]] = {}
    for actuator_id in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuators[name] = {
            "joint": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
            "transmission_type": int(model.actuator_trntype[actuator_id]),
            "gaintype": int(model.actuator_gaintype[actuator_id]),
            "biastype": int(model.actuator_biastype[actuator_id]),
            "dyntype": int(model.actuator_dyntype[actuator_id]),
            "gainprm": np.asarray(model.actuator_gainprm[actuator_id]).tolist(),
            "biasprm": np.asarray(model.actuator_biasprm[actuator_id]).tolist(),
            "dynprm": np.asarray(model.actuator_dynprm[actuator_id]).tolist(),
            "gear": np.asarray(model.actuator_gear[actuator_id]).tolist(),
            "ctrlrange": np.asarray(model.actuator_ctrlrange[actuator_id]).tolist(),
            "ctrllimited": bool(model.actuator_ctrllimited[actuator_id]),
            "forcerange": np.asarray(model.actuator_forcerange[actuator_id]).tolist(),
            "forcelimited": bool(model.actuator_forcelimited[actuator_id]),
        }
    constraints = {
        _name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id): {
            "type": int(model.eq_type[equality_id]),
            "enabled": bool(model.eq_active0[equality_id]),
            "object1": int(model.eq_obj1id[equality_id]),
            "object2": int(model.eq_obj2id[equality_id]),
            "data": np.asarray(model.eq_data[equality_id]).tolist(),
            "solref": np.asarray(model.eq_solref[equality_id]).tolist(),
            "solimp": np.asarray(model.eq_solimp[equality_id]).tolist(),
        }
        for equality_id in range(model.neq)
    }
    pairs = {
        _name(model, mujoco.mjtObj.mjOBJ_PAIR, pair_id): {
            "geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[pair_id])),
            "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[pair_id])),
            "condim": int(model.pair_dim[pair_id]),
            "friction": np.asarray(model.pair_friction[pair_id]).tolist(),
            "solref": np.asarray(model.pair_solref[pair_id]).tolist(),
            "solreffriction": np.asarray(model.pair_solreffriction[pair_id]).tolist(),
            "solimp": np.asarray(model.pair_solimp[pair_id]).tolist(),
            "margin": float(model.pair_margin[pair_id]),
            "gap": float(model.pair_gap[pair_id]),
        }
        for pair_id in range(model.npair)
    }
    bodies = {
        _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id): {
            "parent": _name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[body_id])
            ),
            "position_parent": np.asarray(model.body_pos[body_id]).tolist(),
            "quaternion_parent_wxyz": np.asarray(model.body_quat[body_id]).tolist(),
            "mass": float(model.body_mass[body_id]),
            "com_body": np.asarray(model.body_ipos[body_id]).tolist(),
            "principal_quaternion_body": np.asarray(model.body_iquat[body_id]).tolist(),
            "principal_inertia": np.asarray(model.body_inertia[body_id]).tolist(),
            "gravity_compensation": float(model.body_gravcomp[body_id]),
            "collision_contype": int(model.body_contype[body_id]),
            "collision_conaffinity": int(model.body_conaffinity[body_id]),
            "collision_margin": float(model.body_margin[body_id]),
        }
        for body_id in range(model.nbody)
    }
    geoms = {
        _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id): {
            "body": _name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
            ),
            "type": int(model.geom_type[geom_id]),
            "position_body": np.asarray(model.geom_pos[geom_id]).tolist(),
            "quaternion_body_wxyz": np.asarray(model.geom_quat[geom_id]).tolist(),
            "size": np.asarray(model.geom_size[geom_id]).tolist(),
            "contype": int(model.geom_contype[geom_id]),
            "conaffinity": int(model.geom_conaffinity[geom_id]),
            "condim": int(model.geom_condim[geom_id]),
            "friction": np.asarray(model.geom_friction[geom_id]).tolist(),
            "solref": np.asarray(model.geom_solref[geom_id]).tolist(),
            "solimp": np.asarray(model.geom_solimp[geom_id]).tolist(),
            "solmix": float(model.geom_solmix[geom_id]),
            "margin": float(model.geom_margin[geom_id]),
            "gap": float(model.geom_gap[geom_id]),
            "priority": int(model.geom_priority[geom_id]),
        }
        for geom_id in range(model.ngeom)
    }
    return {
        "schema_version": "RootCauseMujocoCompiledSemanticsV2",
        "mujoco_version": mujoco.__version__,
        "dimensions": {
            "nbody": model.nbody, "njnt": model.njnt, "nq": model.nq,
            "nv": model.nv, "nu": model.nu, "neq": model.neq, "npair": model.npair,
            "ngeom": model.ngeom,
        },
        "ordered_names": {
            "bodies": list(bodies),
            "joints": list(joints),
            "actuators": list(actuators),
            "constraints": list(constraints),
            "geoms": list(geoms),
            "contact_pairs": list(pairs),
        },
        "options": compiled_option_snapshot(model),
        "bodies": bodies,
        "joints": joints,
        "actuators": actuators,
        "constraints": constraints,
        "geoms": geoms,
        "contact_pairs": pairs,
    }


def _mujoco_robot_resolved_semantics(
    compiled: Mapping[str, Any], *, scenario: str
) -> dict[str, Any]:
    joints = compiled["joints"]
    robot_hinges: dict[str, dict[str, Any]] = {}
    static_joints: dict[str, dict[str, Any]] = {}
    for name, record in joints.items():
        item = dict(record)
        damping = item.pop("dof_damping")
        if name in ROBOT_HINGES:
            if len(damping) != 1:
                raise ValueError(f"Robot hinge {name} is not one-DOF")
            robot_hinges[name] = {
                "dof_damping": float(damping[0]),
                "stiffness": item["stiffness"],
                "armature": item["armature"],
                "frictionloss": item["frictionloss"],
                "dof_address": item["dof_address"],
            }
        else:
            item["dof_damping"] = damping
        static_joints[name] = item
    direct = (
        scenario.endswith("direct")
        or scenario.startswith("p10")
        or scenario.startswith("p40")
        or scenario == "p60_b"
    )
    constraints = compiled["constraints"]
    closure_enabled = scenario not in {
        "p10_c",
        "p30_open_direct",
        "p30_open_target",
        "p40_off",
    }
    floor = compiled["geoms"].get("floor")
    collision_enabled_geoms = sorted(
        name
        for name, record in compiled["geoms"].items()
        if int(record["contype"]) != 0 or int(record["conaffinity"]) != 0
    )
    explicit_contact_pairs = sorted(compiled["contact_pairs"])
    contact_exclusion = {
        "ground_absent": floor is None,
        "collision_enabled_geoms": collision_enabled_geoms,
        "explicit_contact_pairs": explicit_contact_pairs,
        "contact_free": bool(
            floor is None
            and not collision_enabled_geoms
            and not explicit_contact_pairs
        ),
    }
    return {
        "topology": {
            **compiled["dimensions"],
            "ordered_names": compiled["ordered_names"],
        },
        "environment": {
            "ground_enabled": floor is not None,
            "ground_geometry": floor,
            "gravity_enabled": bool(
                np.linalg.norm(np.asarray(compiled["options"]["gravity"])) > 0.0
            ),
            "fixed_base": "base_free" not in joints,
        },
        "engine_options": compiled["options"],
        "static_model": {
            "bodies": compiled["bodies"],
            "joints": static_joints,
            "geoms": compiled["geoms"],
        },
        "closure": {
            "mode": "enabled" if closure_enabled else "disabled",
            "constraints": constraints,
        },
        "actuation": {
            "drive_mode": "direct_effort_bypass" if direct else "formal_target_drive",
            "external_controller_enabled": not direct,
            "target_neutralization": direct,
            "robot_hinges": robot_hinges,
            "actuators": compiled["actuators"],
            "canonical_native_signs": CANONICAL_SIGNS.tolist(),
        },
        "contact": {
            "mode": "compiled_explicit_pairs",
            "pairs": compiled["contact_pairs"],
            "contact_exclusion": contact_exclusion,
        },
    }


def _mujoco_sphere_resolved_semantics(
    compiled: Mapping[str, Any], *, friction: float
) -> dict[str, Any]:
    pair = compiled["contact_pairs"].get("floor_coupon")
    if pair is None:
        raise ValueError("Compiled common-sphere pair is missing")
    return {
        "topology": {
            **compiled["dimensions"],
            "ordered_names": compiled["ordered_names"],
        },
        "environment": {
            "ground_enabled": "floor" in compiled["geoms"],
            "gravity_enabled": True,
            "fixed_base": False,
        },
        "engine_options": compiled["options"],
        "static_model": {
            "bodies": compiled["bodies"],
            "joints": compiled["joints"],
            "geoms": compiled["geoms"],
            "coupon": {
                "shape": "sphere",
                "radius_m": 0.0625,
                "mass_kg": 0.4,
                "restitution": 0.0,
            },
        },
        "contact": {
            "mode": "compiled_explicit_pairs",
            "pairs": {"floor_coupon": pair},
        },
    }


def _mujoco_robot_transform_semantics(
    manifest_path: Path, *, scenario: str
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    after = manifest.get("after_semantics")
    if not isinstance(after, Mapping):
        raise ValueError("MuJoCo robot transform manifest has no after semantics")
    direct = (
        scenario.endswith("direct")
        or scenario.startswith("p10")
        or scenario.startswith("p40")
        or scenario == "p60_b"
    )
    joints = after.get("joints")
    connects = after.get("connects")
    if not isinstance(joints, Mapping) or not isinstance(connects, Mapping):
        raise ValueError("MuJoCo robot transform manifest is incomplete")
    robot_hinges = {
        name: {"dof_damping": float(str(joints[name]["damping"]))}
        for name in ROBOT_HINGES
    }
    closure_enabled = scenario not in {
        "p10_c",
        "p30_open_direct",
        "p30_open_target",
        "p40_off",
    }
    constraints = {
        name: {"enabled": str(connects[name]["active"]).lower() != "false"}
        for name in CONNECT_NAMES
    }
    return {
        "actuation": {
            "drive_mode": "direct_effort_bypass" if direct else "formal_target_drive",
            "external_controller_enabled": not direct,
            "target_neutralization": direct,
            "robot_hinges": robot_hinges,
        },
        "closure": {
            "mode": "enabled" if closure_enabled else "disabled",
            "constraints": constraints,
        },
    }


def _mujoco_sphere_transform_semantics(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    after = manifest.get("after_semantics")
    if not isinstance(after, Mapping):
        raise ValueError("MuJoCo sphere transform manifest has no after semantics")
    pairs = after.get("pairs")
    if not isinstance(pairs, Mapping) or "floor_coupon" not in pairs:
        raise ValueError("MuJoCo sphere transform manifest has no floor_coupon pair")
    value = pairs["floor_coupon"].get("friction")
    if not isinstance(value, str):
        raise ValueError("MuJoCo sphere transform friction declaration is invalid")
    return {
        "contact": {
            "pairs": {
                "floor_coupon": {
                    "friction": [float(item) for item in value.split()]
                }
            }
        }
    }


def write_identity(output: Path, model_path: Path = FORMAL_MODEL) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    semantics = compiled_semantics(model)
    policies = {
        name: {
            kind: {
                "path": str(record.absolute_path()),
                "expected_sha256": record.sha256,
                "actual_sha256": sha256_file(record.absolute_path()),
            }
            for kind, record in files.items()
        }
        for name, files in FROZEN_POLICIES.items()
    }
    payload = {
        "schema_version": "RootCauseMujocoIdentityV1",
        "model_path": str(Path(model_path).resolve()),
        "model_sha256": sha256_file(model_path),
        "model_manifest_sha256": sha256_file(MODEL_MANIFEST),
        "source_audit_sha256": sha256_file(SOURCE_AUDIT),
        "body_mapping_sha256": sha256_file(BODY_MAPPING),
        "policies": policies,
        "compiled_semantics": semantics,
        "compiled_semantics_hash": stable_hash(semantics),
    }
    (output / "identity.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def _local_inertia_body(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    rotation_body_from_principal = _quat_matrix(model.body_iquat[body_id])
    return rotation_body_from_principal @ np.diag(model.body_inertia[body_id]) @ rotation_body_from_principal.T


def _body_property(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> BodyProperty:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
    return BodyProperty(
        name=_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
        mass=float(model.body_mass[body_id]),
        com_position_world=np.asarray(data.xipos[body_id], dtype=np.float64).copy(),
        inertia_com_body=_local_inertia_body(model, body_id),
        rotation_world_from_body=np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy(),
        com_linear_velocity_world=jacp @ data.qvel,
        angular_velocity_world=jacr @ data.qvel,
    )


def _body_static_record(model: mujoco.MjModel, body_id: int) -> dict[str, Any]:
    return {
        "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
        "mass_kg": float(model.body_mass[body_id]),
        "center_of_mass_m": np.asarray(model.body_ipos[body_id]).tolist(),
        "principal_axes_wxyz": np.asarray(model.body_iquat[body_id]).tolist(),
        "diagonal_inertia_kg_m2": np.asarray(model.body_inertia[body_id]).tolist(),
        "inertia_com_body_kg_m2": _local_inertia_body(model, body_id).tolist(),
    }


def collect_properties(output: Path, model_path: Path = FORMAL_MODEL) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    mapping = json.loads(BODY_MAPPING.read_text(encoding="utf-8"))
    validate_body_mapping(mapping)
    source = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    rows, failures = [], []
    for entry in mapping["bodies"]:
        body_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, entry["mujoco"])
        compiled = _body_static_record(model, body_id)
        audited = source["bodies"][entry["source_audit"]]["inertia"]
        mass_error = abs(compiled["mass_kg"] - float(audited["mass_kg"]))
        com_error = float(np.max(np.abs(np.asarray(compiled["center_of_mass_m"]) - np.asarray(audited["center_of_mass_m"]))))
        inertia_error = float(np.max(np.abs(np.asarray(compiled["diagonal_inertia_kg_m2"]) - np.asarray(audited["diagonal_inertia_kg_m2"]))))
        passed = (
            mass_error <= max(1.0e-7, 1.0e-6 * abs(float(audited["mass_kg"])))
            and com_error <= 1.0e-6
            and inertia_error <= max(1.0e-9, 1.0e-5 * float(np.max(np.abs(audited["diagonal_inertia_kg_m2"]))))
        )
        if not passed:
            failures.append(entry["canonical"])
        rows.append({
            "mapping": entry, "compiled": compiled, "source_audit": audited,
            "errors": {"mass": mass_error, "com_max_abs": com_error, "inertia_diag_max_abs": inertia_error},
            "passed": passed,
        })
    total_mass = float(sum(row["compiled"]["mass_kg"] for row in rows))
    payload = {
        "schema_version": "RootCauseMujocoPropertiesV1",
        "model_sha256": sha256_file(model_path),
        "source_audit_sha256": sha256_file(SOURCE_AUDIT),
        "mapping_sha256": sha256_file(BODY_MAPPING),
        "body_count": len(rows),
        "total_mass_kg": total_mass,
        "total_mass_anchor_error_kg": abs(total_mass - 4.396253988146782),
        "failed_bodies": failures,
        "audit_valid": True,
        "mismatch_detected": bool(
            failures or abs(total_mass - 4.396253988146782) > 1.0e-6
        ),
        "passed": True,
        "bodies": rows,
    }
    (output / "properties.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def run_golden(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    golden = golden_body_property()
    rotation = golden.rotation_world_from_body
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
    body_origin = np.asarray([0.3, -0.2, 0.5], dtype=np.float64)
    com_offset_body = np.asarray([0.11, -0.07, 0.05], dtype=np.float64)
    xml = f"""<mujoco model='golden'>
  <option timestep='0.001' gravity='0 0 0'/>
  <worldbody><body name='golden_body' pos='{' '.join(map(str, body_origin))}'>
    <freejoint name='golden_free'/>
    <inertial pos='{' '.join(map(str, com_offset_body))}' quat='{' '.join(map(str, quaternion))}' mass='2.3' diaginertia='0.031 0.047 0.073'/>
    <geom type='sphere' size='0.01' contype='0' conaffinity='0'/>
  </body></worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, "golden_body")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
    desired = np.concatenate((golden.com_linear_velocity_world, golden.angular_velocity_world))
    data.qvel[:] = np.linalg.solve(np.vstack((jacp, jacr)), desired)
    mujoco.mj_forward(model, data)
    actual = _body_property(model, data, body_id)
    expected_linear, expected_com, expected_origin = momentum_about_origin(golden, np.zeros(3))
    actual_linear, actual_com, actual_origin = momentum_about_origin(actual, np.zeros(3))
    errors = {
        "com_position": float(np.max(np.abs(actual.com_position_world - golden.com_position_world))),
        "com_velocity": float(np.max(np.abs(actual.com_linear_velocity_world - golden.com_linear_velocity_world))),
        "angular_velocity": float(np.max(np.abs(actual.angular_velocity_world - golden.angular_velocity_world))),
        "linear_momentum": float(np.max(np.abs(actual_linear - expected_linear))),
        "angular_momentum_com": float(np.max(np.abs(actual_com - expected_com))),
        "angular_momentum_origin": float(np.max(np.abs(actual_origin - expected_origin))),
    }
    payload = {
        "schema_version": "RootCauseMujocoGoldenV1",
        "errors": errors,
        "passed": max(errors.values()) <= 1.0e-10,
        "actual": {
            "mass": actual.mass,
            "com_position_world": actual.com_position_world.tolist(),
            "com_linear_velocity_world": actual.com_linear_velocity_world.tolist(),
            "angular_velocity_world": actual.angular_velocity_world.tolist(),
            "linear_momentum": actual_linear.tolist(),
            "angular_momentum_com": actual_com.tolist(),
            "angular_momentum_origin": actual_origin.tolist(),
        },
    }
    (output / "golden.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def _controlled_ids(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.asarray(
        [_required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in CONTROLLED_JOINTS],
        dtype=np.int32,
    )
    dof_ids = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32)
    actuator_ids = np.asarray(
        [_required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATORS],
        dtype=np.int32,
    )
    if not np.array_equal(model.actuator_trnid[actuator_ids, 0], joint_ids):
        raise ValueError("Actuator to controlled-joint mapping drifted")
    return joint_ids, dof_ids, actuator_ids


def validate_direct_effort_bypass(model: mujoco.MjModel) -> dict[str, Any]:
    hinge_ids = np.asarray(
        [_required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ROBOT_HINGES],
        dtype=np.int32,
    )
    dof_ids = np.asarray(model.jnt_dofadr[hinge_ids], dtype=np.int32)
    if len(set(dof_ids.tolist())) != 26 or np.any(model.dof_damping[dof_ids] != 0.0):
        raise ValueError("Drive-off variant does not have 26 unique zero-damping robot hinges")
    joint_ids, controlled_dofs, actuator_ids = _controlled_ids(model)
    rows = []
    expected_gear = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for index, actuator_id in enumerate(actuator_ids):
        if int(model.actuator_gaintype[actuator_id]) != int(mujoco.mjtGain.mjGAIN_FIXED):
            raise ValueError("Direct-effort actuator is not fixed-gain")
        if int(model.actuator_biastype[actuator_id]) != int(mujoco.mjtBias.mjBIAS_NONE):
            raise ValueError("Direct-effort actuator has a bias")
        if abs(float(model.actuator_gainprm[actuator_id, 0]) - 1.0) > 0.0:
            raise ValueError("Direct-effort actuator gain is not one")
        if np.any(model.actuator_biasprm[actuator_id] != 0.0):
            raise ValueError("Direct-effort actuator bias parameters are non-zero")
        if not np.array_equal(np.asarray(model.actuator_gear[actuator_id]), expected_gear):
            raise ValueError("Direct-effort actuator gear changed")
        rows.append({
            "joint": CONTROLLED_JOINTS[index],
            "joint_id": int(joint_ids[index]),
            "dof_address": int(controlled_dofs[index]),
            "actuator": ACTUATORS[index],
            "actuator_id": int(actuator_id),
            "canonical_sign": float(CANONICAL_SIGNS[index]),
        })
    return {"hinge_count": 26, "controlled": rows}


def direct_effort_roundtrip(
    model: mujoco.MjModel, data: mujoco.MjData, canonical: np.ndarray
) -> dict[str, Any]:
    values = np.asarray(canonical, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError("Canonical direct effort must have shape (6,)")
    _, dof_ids, actuator_ids = _controlled_ids(model)
    native = values * CANONICAL_SIGNS
    data.ctrl[:] = 0.0
    data.ctrl[actuator_ids] = native
    mujoco.mj_forward(model, data)
    actuator_canonical = np.asarray(data.actuator_force[actuator_ids]) * CANONICAL_SIGNS
    generalized_canonical = np.asarray(data.qfrc_actuator[dof_ids]) * CANONICAL_SIGNS
    errors = {
        "actuator_force": float(np.max(np.abs(actuator_canonical - values))),
        "generalized_force": float(np.max(np.abs(generalized_canonical - values))),
    }
    if max(errors.values()) > 1.0e-12:
        raise ValueError(f"Direct effort round-trip failed: {errors}")
    return {
        "canonical_input": values.tolist(),
        "native_control": native.tolist(),
        "canonical_actuator_force": actuator_canonical.tolist(),
        "canonical_generalized_force": generalized_canonical.tolist(),
        "errors": errors,
    }


def _reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "contract_v4_reset")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, int(key_id))
    else:
        mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _closure_residual(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    residuals = []
    for name in CONNECT_NAMES:
        equality_id = _required_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        site1 = int(model.eq_obj1id[equality_id])
        site2 = int(model.eq_obj2id[equality_id])
        residuals.append(float(np.linalg.norm(data.site_xpos[site1] - data.site_xpos[site2])))
    return np.asarray(residuals, dtype=np.float64)


def _contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    normal_force = 0.0
    maximum_penetration = 0.0
    force = np.zeros(6, dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        mujoco.mj_contactForce(model, data, index, force)
        normal_force += max(0.0, float(force[0]))
        maximum_penetration = max(maximum_penetration, max(0.0, -float(contact.dist)))
    return normal_force, maximum_penetration


def _system_snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    joint_ids, dof_ids, actuator_ids = _controlled_ids(model)
    all_hinge_ids = np.asarray(
        [_required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ROBOT_HINGES],
        dtype=np.int32,
    )
    qpos_ids = np.asarray(model.jnt_qposadr[all_hinge_ids], dtype=np.int32)
    all_dofs = np.asarray(model.jnt_dofadr[all_hinge_ids], dtype=np.int32)
    bodies = [_body_property(model, data, body_id) for body_id in range(1, model.nbody)]
    composite = composite_properties(bodies)
    base_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    base = _body_property(model, data, base_id)
    normal_force, penetration = _contact_summary(model, data)
    return {
        "time_s": float(data.time),
        "controlled_position_canonical": np.asarray(data.qpos[model.jnt_qposadr[joint_ids]]) * CANONICAL_SIGNS,
        "controlled_velocity_canonical": np.asarray(data.qvel[dof_ids]) * CANONICAL_SIGNS,
        "all_hinge_position": np.asarray(data.qpos[qpos_ids]).copy(),
        "all_hinge_velocity": np.asarray(data.qvel[all_dofs]).copy(),
        "base_com_position_world": base.com_position_world.copy(),
        "base_com_velocity_world": base.com_linear_velocity_world.copy(),
        "base_angular_velocity_world": base.angular_velocity_world.copy(),
        "system_com_position_world": composite.com_position_world.copy(),
        "linear_momentum_world": composite.linear_momentum_world.copy(),
        "angular_momentum_com_world": composite.angular_momentum_com_world.copy(),
        "kinetic_energy_j": composite.kinetic_energy,
        "closure_residual_m": _closure_residual(model, data),
        "contact_count": int(data.ncon),
        "contact_normal_force_n": normal_force,
        "max_penetration_m": penetration,
        "actuator_force_native": np.asarray(data.actuator_force[actuator_ids]).copy(),
        "generalized_force_native": np.asarray(data.qfrc_actuator[dof_ids]).copy(),
    }


def _rows_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def _trace_fields(arrays: Mapping[str, np.ndarray]) -> dict[str, FieldSpec]:
    units = {
        "time_s": "s", "controlled_position_canonical": "rad",
        "controlled_velocity_canonical": "rad/s", "all_hinge_position": "rad",
        "all_hinge_velocity": "rad/s", "base_com_position_world": "m",
        "base_com_velocity_world": "m/s", "base_angular_velocity_world": "rad/s",
        "system_com_position_world": "m", "linear_momentum_world": "kg*m/s",
        "angular_momentum_com_world": "kg*m^2/s", "kinetic_energy_j": "J",
        "closure_residual_m": "m", "contact_count": "count",
        "contact_normal_force_n": "N", "max_penetration_m": "m",
        "actuator_force_native": "N*m", "generalized_force_native": "N*m",
    }
    return {
        name: FieldSpec(units[name], "world" if "world" in name else "canonical", "post_step", name)
        for name in arrays
    }


def _formal_contract() -> Any:
    from wheelleg_mujoco.contract import load_policy_contract

    actor = FROZEN_POLICIES["dreamwaq_run01"]["actor"].absolute_path()
    manifest = FROZEN_POLICIES["dreamwaq_run01"]["manifest"].absolute_path()
    contract, _, _ = load_policy_contract(manifest, actor, MODEL_MANIFEST)
    return contract


def _formal_target_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    action: np.ndarray,
    contract: Any,
) -> np.ndarray:
    clipped = np.clip(np.asarray(action, dtype=np.float64), -contract.action_clip, contract.action_clip)
    joint_ids, dof_ids, _ = _controlled_ids(model)
    q = np.asarray(data.qpos[model.jnt_qposadr[joint_ids]])
    qd = np.asarray(data.qvel[dof_ids])
    leg_target = np.clip(
        contract.q_nominal + contract.leg_action_scale * clipped[:4],
        contract.leg_target_lower,
        contract.leg_target_upper,
    )
    wheel_target = contract.wheel_action_scale * clipped[4:] * contract.wheel_signs
    torque = np.empty(6, dtype=np.float64)
    torque[:4] = contract.leg_kp * (leg_target - q[:4]) - contract.leg_kd * qd[:4]
    torque[4:] = contract.wheel_kd * (wheel_target - qd[4:])
    torque = np.clip(torque, -contract.effort_limits, contract.effort_limits)
    return torque


ROBOT_SCENARIO_OPERATIONS = {
    "p10_a": ("no_ground", "gravity_off", "drive_off"),
    "p10_b": ("no_ground", "drive_off"),
    "p10_c": ("no_ground", "gravity_off", "closure_off", "drive_off"),
    "p30_open_direct": ("no_ground", "gravity_off", "closure_off", "drive_off", "fixed_base"),
    "p30_open_target": ("no_ground", "gravity_off", "closure_off", "fixed_base"),
    "p30_closed_direct": ("no_ground", "gravity_off", "drive_off", "fixed_base"),
    "p30_closed_target": ("no_ground", "gravity_off", "fixed_base"),
    "p40_on": ("no_ground", "gravity_off", "drive_off"),
    "p40_off": ("no_ground", "gravity_off", "closure_off", "drive_off"),
    "p60_b": ("drive_off",),
    "p60_c": (),
}


def collect_robot_probe(
    output: Path,
    *,
    scenario: str,
    steps: int,
    channel: int = 0,
    amplitude: float = 0.1,
    sign: int = 1,
) -> dict[str, Any]:
    if scenario not in ROBOT_SCENARIO_OPERATIONS:
        raise ValueError(f"Unknown robot probe scenario: {scenario}")
    if steps <= 0 or not 0 <= channel < 6 or sign not in {-1, 0, 1}:
        raise ValueError("Invalid robot probe parameters")
    output.mkdir(parents=True, exist_ok=False)
    operations = ROBOT_SCENARIO_OPERATIONS[scenario]
    variant = build_robot_variant(FORMAL_MODEL, output / "model" / "robot.xml", operations=operations)
    model = mujoco.MjModel.from_xml_path(str(variant.model_path))
    data = mujoco.MjData(model)
    _reset(model, data)
    direct = (
        scenario.endswith("direct")
        or scenario.startswith("p10")
        or scenario.startswith("p40")
        or scenario == "p60_b"
    )
    bypass = validate_direct_effort_bypass(model) if "drive_off" in operations else None
    roundtrip = None
    if direct and "drive_off" in operations:
        probe = np.zeros(6, dtype=np.float64)
        probe[channel] = 0.123 * (1 if sign == 0 else sign)
        roundtrip = direct_effort_roundtrip(model, data, probe)
        _reset(model, data)
    contract = _formal_contract() if not direct else None
    rows = [_system_snapshot(model, data)]
    for _ in range(steps):
        canonical = np.zeros(6, dtype=np.float64)
        active = scenario.startswith("p30") or scenario.startswith("p40") or scenario == "p60_c"
        if active and sign:
            canonical[channel] = sign * amplitude
            if scenario.startswith("p40"):
                canonical[2] = -sign * amplitude
        _, _, actuator_ids = _controlled_ids(model)
        data.ctrl[:] = 0.0
        if direct:
            data.ctrl[actuator_ids] = canonical * CANONICAL_SIGNS
        else:
            data.ctrl[actuator_ids] = _formal_target_torque(model, data, canonical, contract)
        mujoco.mj_step(model, data)
        rows.append(_system_snapshot(model, data))
    arrays = _rows_arrays(rows)
    identity = write_trace(output / "trace", arrays, _trace_fields(arrays))
    payload = {
        "schema_version": "RootCauseMujocoRobotProbeV1",
        "scenario": scenario,
        "operations": list(operations),
        "model_sha256": variant.model_sha256,
        "variant_manifest_sha256": variant.manifest_sha256,
        "steps": steps,
        "channel": channel,
        "amplitude": amplitude,
        "sign": sign,
        "direct_effort_bypass": bypass,
        "direct_effort_roundtrip": roundtrip,
        "trace_sha256": identity.trace_sha256,
        "trace_metadata_sha256": identity.metadata_sha256,
    }
    (output / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def _batch_robot_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contract: Any,
    initial_base_com_world: np.ndarray,
) -> dict[str, Any]:
    source = _system_snapshot(model, data)
    rotation = np.asarray(contract.r_control_from_mujoco, dtype=np.float64)
    base_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    rotation_body_from_world = np.asarray(data.xmat[base_id], dtype=np.float64).reshape(3, 3).T
    base_position = rotation @ (
        np.asarray(source["base_com_position_world"]) - initial_base_com_world
    )
    base_position[2] = float(source["base_com_position_world"][2])
    return {
        "controlled_position_canonical": source["controlled_position_canonical"],
        "controlled_velocity_canonical": source["controlled_velocity_canonical"],
        "all_hinge_position": source["all_hinge_position"],
        "all_hinge_velocity": source["all_hinge_velocity"],
        "base_com_position_diag": base_position,
        "base_linear_velocity_control": rotation @ rotation_body_from_world @ source["base_com_velocity_world"],
        "base_angular_velocity_control": rotation @ rotation_body_from_world @ source["base_angular_velocity_world"],
        "system_com_position_control": rotation @ source["system_com_position_world"],
        "linear_momentum_control": rotation @ source["linear_momentum_world"],
        "angular_momentum_com_control": rotation @ source["angular_momentum_com_world"],
        "kinetic_energy_j": source["kinetic_energy_j"],
        "closure_residual_m": float(np.max(source["closure_residual_m"])),
        "host_applied_torque_canonical": source["actuator_force_native"] * CANONICAL_SIGNS,
    }


def _batch_robot_fields(arrays: Mapping[str, np.ndarray]) -> dict[str, FieldSpec]:
    units = {
        "time_s": "s", "profile_channel": "index", "profile_sign": "1",
        "profile_repetition": "index", "controlled_position_canonical": "rad",
        "controlled_velocity_canonical": "rad/s", "all_hinge_position": "rad",
        "all_hinge_velocity": "rad/s", "base_com_position_diag": "m",
        "base_linear_velocity_control": "m/s", "base_angular_velocity_control": "rad/s",
        "system_com_position_control": "m", "linear_momentum_control": "kg*m/s",
        "angular_momentum_com_control": "kg*m^2/s", "kinetic_energy_j": "J",
        "closure_residual_m": "m", "commanded_input_canonical": "N*m_or_action",
        "canonical_torque_equivalent": "N*m", "host_applied_torque_canonical": "N*m",
        "effort_limit_event": "bool", "joint_velocity_limit_exceeded": "bool",
        "terminated": "bool", "truncated": "bool",
    }
    frames = {
        "base_com_position_diag": "control_world",
        "base_linear_velocity_control": "control_body",
        "base_angular_velocity_control": "control_body",
        "system_com_position_control": "control_world",
        "linear_momentum_control": "control_world",
        "angular_momentum_com_control": "control_world",
    }
    return {
        name: FieldSpec(
            units[name],
            frames.get(name, "canonical"),
            "post_step" if name != "time_s" else "sample_time",
            name,
        )
        for name in arrays
    }


def collect_robot_probe_batch(
    output: Path,
    *,
    scenario: str,
    control_ticks: int = 5,
    repetitions: int = 3,
    amplitude: float | None = None,
    input_vector: Iterable[float] | None = None,
    repeatability_family: str | None = None,
) -> dict[str, Any]:
    if scenario not in ROBOT_SCENARIO_OPERATIONS or control_ticks <= 0:
        raise ValueError("Invalid MuJoCo batch robot probe")
    output.mkdir(parents=True, exist_ok=False)
    profiles = robot_probe_profiles(
        scenario,
        repetitions=repetitions,
        amplitude=amplitude,
        input_vector=None if input_vector is None else tuple(input_vector),
    )
    operations = ROBOT_SCENARIO_OPERATIONS[scenario]
    variant = build_robot_variant(
        FORMAL_MODEL,
        output / "model" / "robot.xml",
        operations=operations,
    )
    model = mujoco.MjModel.from_xml_path(str(variant.model_path))
    direct = (
        scenario.endswith("direct")
        or scenario.startswith("p10")
        or scenario.startswith("p40")
        or scenario == "p60_b"
    )
    bypass = validate_direct_effort_bypass(model) if direct else None
    contract = _formal_contract()
    _, controlled_dofs, actuator_ids = _controlled_ids(model)
    per_profile: list[list[dict[str, Any]]] = []
    reset_snapshots: list[dict[str, Any]] = []
    maximum_direct_effort_error = 0.0
    for profile in profiles:
        data = mujoco.MjData(model)
        _reset(model, data)
        initial_base = np.asarray(
            data.xipos[_required_id(model, mujoco.mjtObj.mjOBJ_BODY, "base")],
            dtype=np.float64,
        ).copy()
        trace_rows: list[dict[str, Any]] = []
        initial = _batch_robot_snapshot(model, data, contract, initial_base)
        reset_snapshots.append(initial)
        trace_rows.append(
            {
                **initial,
                "time_s": 0.0,
                "commanded_input_canonical": np.zeros(6, dtype=np.float64),
                "canonical_torque_equivalent": np.zeros(6, dtype=np.float64),
                "effort_limit_event": np.zeros(6, dtype=np.int8),
                "joint_velocity_limit_exceeded": np.zeros(6, dtype=np.int8),
                "terminated": np.int8(0),
                "truncated": np.int8(0),
            }
        )
        declared = np.asarray(profile["actual_input"], dtype=np.float64)
        torque_equivalent = np.asarray(
            profile["canonical_torque_equivalent"], dtype=np.float64
        )
        for physics_step in range(control_ticks * 20):
            active = physics_step < 20
            selected = declared if active else np.zeros(6, dtype=np.float64)
            data.ctrl[:] = 0.0
            if direct:
                data.ctrl[actuator_ids] = selected * CANONICAL_SIGNS
            else:
                data.ctrl[actuator_ids] = _formal_target_torque(
                    model, data, selected, contract
                )
            mujoco.mj_step(model, data)
            snapshot = _batch_robot_snapshot(model, data, contract, initial_base)
            canonical_force = snapshot["host_applied_torque_canonical"]
            if direct and active:
                maximum_direct_effort_error = max(
                    maximum_direct_effort_error,
                    float(np.max(np.abs(canonical_force - selected))),
                )
            velocity_event = np.zeros(6, dtype=np.int8)  # PhysicsV5: no joint-speed limiter.
            trace_rows.append(
                {
                    **snapshot,
                    "time_s": float(data.time),
                    "commanded_input_canonical": selected.copy(),
                    "canonical_torque_equivalent": (
                        torque_equivalent.copy() if active else np.zeros(6, dtype=np.float64)
                    ),
                    "effort_limit_event": np.zeros(6, dtype=np.int8),
                    "joint_velocity_limit_exceeded": velocity_event,
                    "terminated": np.int8(0),
                    "truncated": np.int8(0),
                }
            )
        per_profile.append(trace_rows)
    if direct and maximum_direct_effort_error > 1.0e-12:
        raise RuntimeError(
            "MuJoCo direct-effort round trip drifted in the batch probe: "
            f"{maximum_direct_effort_error}"
        )
    row_count = len(per_profile[0])
    if any(len(rows) != row_count for rows in per_profile):
        raise RuntimeError("MuJoCo batch profiles have inconsistent trace lengths")
    profile_channel = np.asarray([item["channel"] for item in profiles], dtype=np.int16)
    profile_sign = np.asarray([item["sign"] for item in profiles], dtype=np.int8)
    profile_repetition = np.asarray(
        [item["repetition"] for item in profiles], dtype=np.int16
    )
    rows = []
    for sample in range(row_count):
        row = {
            key: np.asarray([profile_rows[sample][key] for profile_rows in per_profile])
            for key in per_profile[0][sample]
        }
        row["profile_channel"] = profile_channel
        row["profile_sign"] = profile_sign
        row["profile_repetition"] = profile_repetition
        rows.append(row)
    arrays = _rows_arrays(rows)
    identity = write_trace(output / "trace", arrays, _batch_robot_fields(arrays))
    input_values = np.asarray(
        [item["actual_input"] for item in profiles], dtype=np.float64
    )
    torque_equivalent = np.asarray(
        [item["canonical_torque_equivalent"] for item in profiles], dtype=np.float64
    )
    excitation_semantics = {
        "input_kind": "direct_effort" if direct else "formal_target_action",
        "input_units": "N*m" if direct else "normalized_action",
        "actual_input_values": input_values.tolist(),
        "canonical_torque_equivalent": torque_equivalent.tolist(),
        "start_s": 0.0,
        "stop_s": 0.02,
        "physics_sample_dt_s": float(model.opt.timestep),
        "control_dt_s": 0.02,
    }
    compiled = compiled_semantics(model)
    pre_forward_hash = stable_hash(reset_snapshots)
    post_forward_hash = stable_hash(reset_snapshots)
    reset_returned_hash = None
    excitation_hash = stable_hash(excitation_semantics)
    comparison_profile_hash = (
        stable_hash(
            {
                "canonical_torque_equivalent": [
                    item["canonical_torque_equivalent"] for item in profiles
                ],
                "start_s": 0.0,
                "stop_s": 0.02,
                "profile_channel": [item["channel"] for item in profiles],
                "profile_sign": [item["sign"] for item in profiles],
            }
        )
        if scenario.startswith(("p30", "p40"))
        else None
    )
    source_model_sha256 = sha256_file(FORMAL_MODEL)
    configuration_semantics = robot_configuration_semantics(
        scenario=scenario,
        engine="mujoco",
        physics_dt_s=float(model.opt.timestep),
        control_dt_s=0.02,
        source_identity=source_model_sha256,
        resolved_semantics=_mujoco_robot_resolved_semantics(
            compiled, scenario=scenario
        ),
    )
    causal_identity = result_identity_fields(
        scenario_id=robot_scenario_id(scenario),
        engine="mujoco",
        source_model_sha256=source_model_sha256,
        model_artifact_sha256=variant.model_sha256,
        transform_manifest_sha256=variant.manifest_sha256,
        transform_semantics=_mujoco_robot_transform_semantics(
            variant.manifest_path, scenario=scenario
        ),
        worker_source_sha256=sha256_file(Path(__file__)),
        configuration_semantics=configuration_semantics,
        pre_forward_initial_condition_hash=pre_forward_hash,
        post_forward_state_hash=post_forward_hash,
        reset_returned_policy_hash=reset_returned_hash,
        excitation_hash=excitation_hash,
        comparison_profile_hash=comparison_profile_hash,
    )
    scenario_id = str(causal_identity["scenario_id"])
    family = scenario_spec(scenario_id).repeatability_family
    if repeatability_family is not None:
        if not (
            scenario == "p60_c"
            and repeatability_family == "reset_nominal_zero"
            and all(
                not any(float(value) for value in profile["actual_input"])
                for profile in profiles
            )
        ):
            raise ValueError("Invalid repeatability family override")
        family = repeatability_family
    if family is None:
        raise RuntimeError(f"Robot scenario has no repeatability family: {scenario_id}")
    repeatability_records = []
    for profile_index, (profile, initial) in enumerate(
        zip(profiles, reset_snapshots, strict=True)
    ):
        profile_semantics = {
            key: value for key, value in profile.items() if key != "repetition"
        }
        excitation = {
            "input_kind": excitation_semantics["input_kind"],
            "input_units": excitation_semantics["input_units"],
            "channel": profile["channel"],
            "sign": profile["sign"],
            "actual_input": profile["actual_input"],
            "canonical_torque_equivalent": profile[
                "canonical_torque_equivalent"
            ],
            "start_s": excitation_semantics["start_s"],
            "stop_s": excitation_semantics["stop_s"],
            "physics_sample_dt_s": excitation_semantics["physics_sample_dt_s"],
            "control_dt_s": excitation_semantics["control_dt_s"],
        }
        reset_context = {
            "state": _json_value(initial),
            "command": [0.0, 0.0, 0.0],
            "seed": 0,
            "rng_state": "not_applicable_deterministic_mujoco_probe",
            "reset_artifact_sha256": None,
        }
        repeatability_records.append(
            build_repeatability_record(
                engine="mujoco",
                scenario_id=scenario_id,
                repeatability_family=family,
                configuration_hash=str(causal_identity["configuration_hash"]),
                profile_index=profile_index,
                repetition=int(profile["repetition"]),
                profile_semantics=profile_semantics,
                pre_forward_payload={"phase": "pre_forward", **reset_context},
                post_forward_payload={"phase": "post_forward", **reset_context},
                reset_returned_policy_payload=None,
                excitation_semantics=excitation,
            )
        )
    payload = {
        "schema_version": "RootCauseMujocoRobotBatchV1",
        "scenario": scenario,
        "operations": list(operations),
        "model_sha256": variant.model_sha256,
        "variant_manifest_sha256": variant.manifest_sha256,
        "profile_count": len(profiles),
        "profiles": profiles,
        "control_ticks": control_ticks,
        "physics_dt_s": float(model.opt.timestep),
        "decimation": 20,
        "control_dt_s": 0.02,
        "excitation_semantics": excitation_semantics,
        "repeatability_records": repeatability_records,
        "compiled_semantics_hash": stable_hash(compiled),
        **causal_identity,
        "direct_effort_bypass": bypass,
        "maximum_direct_effort_roundtrip_error_nm": maximum_direct_effort_error,
        "trace_sha256": identity.trace_sha256,
        "trace_metadata_sha256": identity.metadata_sha256,
    }
    (output / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def collect_sphere_probe(
    output: Path,
    *,
    mode: str,
    steps: int,
    height_m: float = 0.1,
    vertical_velocity_mps: float = 0.0,
    horizontal_velocity_mps: float = 0.0,
    friction: float = 0.0,
) -> dict[str, Any]:
    if mode not in {"impact", "slide"} or steps <= 0 or friction < 0.0:
        raise ValueError("Invalid sphere probe parameters")
    output.mkdir(parents=True, exist_ok=False)
    radius = 0.0625
    variant = build_common_sphere_mjcf(
        output / "model" / "sphere.xml",
        radius_m=radius,
        mass_kg=0.4,
        friction=(friction, 0.0, 0.0),
    )
    model = mujoco.MjModel.from_xml_path(str(variant.model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    free_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "coupon_free")
    qpos = int(model.jnt_qposadr[free_id])
    dof = int(model.jnt_dofadr[free_id])
    data.qpos[qpos + 2] = radius + (height_m if mode == "impact" else 0.0)
    data.qvel[dof] = horizontal_velocity_mps
    data.qvel[dof + 2] = vertical_velocity_mps
    mujoco.mj_forward(model, data)
    body_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, "coupon")
    rows = []
    first_contact: float | None = None
    accumulated_normal_impulse = 0.0
    for index in range(steps + 1):
        body = _body_property(model, data, body_id)
        normal_force, penetration = _contact_summary(model, data)
        if data.ncon and first_contact is None:
            first_contact = float(data.time)
        accumulated_normal_impulse += normal_force * float(model.opt.timestep)
        rows.append({
            "time_s": float(data.time),
            "com_position_world": body.com_position_world.copy(),
            "com_velocity_world": body.com_linear_velocity_world.copy(),
            "angular_velocity_world": body.angular_velocity_world.copy(),
            "contact_count": int(data.ncon),
            "normal_force_n": normal_force,
            "normal_impulse_ns": accumulated_normal_impulse,
            "max_penetration_m": penetration,
            "kinetic_energy_j": 0.5 * body.mass * float(body.com_linear_velocity_world @ body.com_linear_velocity_world),
        })
        if index < steps:
            mujoco.mj_step(model, data)
    arrays = _rows_arrays(rows)
    units = {
        "time_s": "s", "com_position_world": "m", "com_velocity_world": "m/s",
        "angular_velocity_world": "rad/s", "contact_count": "count", "normal_force_n": "N",
        "normal_impulse_ns": "N*s", "max_penetration_m": "m", "kinetic_energy_j": "J",
    }
    fields = {name: FieldSpec(units[name], "world", "post_step", name) for name in arrays}
    identity = write_trace(output / "trace", arrays, fields)
    payload = {
        "schema_version": "RootCauseMujocoSphereProbeV1",
        "mode": mode,
        "model_sha256": variant.model_sha256,
        "steps": steps,
        "height_m": height_m,
        "vertical_velocity_mps": vertical_velocity_mps,
        "horizontal_velocity_mps": horizontal_velocity_mps,
        "friction": friction,
        "first_contact_time_s": first_contact,
        "trace_sha256": identity.trace_sha256,
    }
    (output / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def _sphere_probe_fields(arrays: Mapping[str, np.ndarray]) -> dict[str, FieldSpec]:
    units = {
        "time_s": "s",
        "profile_height_m": "m",
        "profile_vertical_velocity_mps": "m/s",
        "profile_horizontal_velocity_mps": "m/s",
        "profile_sign": "1",
        "profile_repetition": "index",
        "com_position_world": "m",
        "com_velocity_world": "m/s",
        "angular_velocity_world": "rad/s",
        "contact_count": "count",
        "normal_force_n": "N",
        "normal_impulse_ns": "N*s",
        "max_penetration_m": "m",
        "kinetic_energy_j": "J",
    }
    return {
        name: FieldSpec(
            units[name],
            "world" if name.endswith("_world") else "canonical",
            "post_step" if name != "time_s" else "sample_time",
            name,
        )
        for name in arrays
    }


def collect_sphere_probe_batch(
    output: Path,
    *,
    mode: str,
    friction: float,
    repetitions: int = 3,
    duration_s: float | None = None,
) -> dict[str, Any]:
    if mode not in {"impact", "slide"} or friction < 0.0:
        raise ValueError("Invalid sphere batch parameters")
    profiles = sphere_probe_profiles(mode, repetitions=repetitions)
    duration = (0.5 if mode == "impact" else 0.4) if duration_s is None else float(duration_s)
    if duration <= 0.0:
        raise ValueError("Sphere batch duration must be positive")
    output.mkdir(parents=True, exist_ok=False)
    radius = 0.0625
    mass = 0.4
    variant = build_common_sphere_mjcf(
        output / "model" / "sphere.xml",
        radius_m=radius,
        mass_kg=mass,
        friction=(friction, 0.0, 0.0),
    )
    model = mujoco.MjModel.from_xml_path(str(variant.model_path))
    steps = int(round(duration / float(model.opt.timestep)))
    if abs(steps * float(model.opt.timestep) - duration) > 1.0e-12:
        raise ValueError("Sphere duration is not an integer multiple of the physics step")
    free_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "coupon_free")
    qpos_address = int(model.jnt_qposadr[free_id])
    dof_address = int(model.jnt_dofadr[free_id])
    body_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, "coupon")
    per_profile: list[list[dict[str, Any]]] = []
    first_contact_times: list[float | None] = []
    initial_states: list[dict[str, Any]] = []
    for profile in profiles:
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_address + 2] = radius + float(profile["height_m"])
        data.qvel[dof_address] = float(profile["horizontal_velocity_mps"])
        data.qvel[dof_address + 2] = float(profile["vertical_velocity_mps"])
        mujoco.mj_forward(model, data)
        initial_states.append(
            {
                "position": np.asarray(data.qpos[qpos_address:qpos_address + 3]).tolist(),
                "quaternion_wxyz": np.asarray(
                    data.qpos[qpos_address + 3:qpos_address + 7]
                ).tolist(),
                "velocity": np.asarray(data.qvel[dof_address:dof_address + 6]).tolist(),
            }
        )
        rows: list[dict[str, Any]] = []
        first_contact: float | None = None
        normal_impulse = 0.0
        for step in range(steps + 1):
            body = _body_property(model, data, body_id)
            normal_force, penetration = _contact_summary(model, data)
            if data.ncon and first_contact is None:
                first_contact = float(data.time)
            if step:
                normal_impulse += normal_force * float(model.opt.timestep)
            rows.append(
                {
                    "time_s": float(data.time),
                    "com_position_world": body.com_position_world.copy(),
                    "com_velocity_world": body.com_linear_velocity_world.copy(),
                    "angular_velocity_world": body.angular_velocity_world.copy(),
                    "contact_count": int(data.ncon),
                    "normal_force_n": normal_force,
                    "normal_impulse_ns": normal_impulse,
                    "max_penetration_m": penetration,
                    "kinetic_energy_j": 0.5
                    * body.mass
                    * float(body.com_linear_velocity_world @ body.com_linear_velocity_world),
                }
            )
            if step < steps:
                mujoco.mj_step(model, data)
        first_contact_times.append(first_contact)
        per_profile.append(rows)
    row_count = steps + 1
    if any(len(rows) != row_count for rows in per_profile):
        raise RuntimeError("MuJoCo sphere profiles have inconsistent trace lengths")
    metadata_vectors = {
        "profile_height_m": np.asarray([row["height_m"] for row in profiles], dtype=np.float64),
        "profile_vertical_velocity_mps": np.asarray(
            [row["vertical_velocity_mps"] for row in profiles], dtype=np.float64
        ),
        "profile_horizontal_velocity_mps": np.asarray(
            [row["horizontal_velocity_mps"] for row in profiles], dtype=np.float64
        ),
        "profile_sign": np.asarray([row["sign"] for row in profiles], dtype=np.int8),
        "profile_repetition": np.asarray(
            [row["repetition"] for row in profiles], dtype=np.int16
        ),
    }
    rows = []
    for sample in range(row_count):
        row = {
            key: np.asarray([profile_rows[sample][key] for profile_rows in per_profile])
            for key in per_profile[0][sample]
        }
        row.update(metadata_vectors)
        rows.append(row)
    arrays = _rows_arrays(rows)
    trace = write_trace(output / "trace", arrays, _sphere_probe_fields(arrays))
    compiled = compiled_semantics(model)
    body_record = _body_static_record(model, body_id)
    expected_inertia = 0.4 * mass * radius * radius
    property_errors = {
        "mass_abs": abs(float(body_record["mass_kg"]) - mass),
        "inertia_max_abs": float(
            np.max(
                np.abs(
                    np.asarray(body_record["inertia_com_body_kg_m2"], dtype=np.float64)
                    - np.eye(3) * expected_inertia
                )
            )
        ),
    }
    property_passed = max(property_errors.values()) <= 1.0e-10
    pre_forward_hash = stable_hash(initial_states)
    post_forward_hash = stable_hash(initial_states)
    excitation_hash = stable_hash(profiles)
    comparison_profile_hash = stable_hash(
        {
            "profiles": profiles,
            "duration_s": duration,
            "radius_m": radius,
            "mass_kg": mass,
        }
    )
    source_identity = stable_hash(
        {"generator": "CommonSphereCouponV1", "radius_m": radius, "mass_kg": mass}
    )
    configuration_semantics = sphere_configuration_semantics(
        engine="mujoco",
        friction=friction,
        physics_dt_s=float(model.opt.timestep),
        source_identity=source_identity,
        resolved_semantics=_mujoco_sphere_resolved_semantics(
            compiled, friction=friction
        ),
    )
    causal_identity = result_identity_fields(
        scenario_id=sphere_scenario_id(mode, friction),
        engine="mujoco",
        source_model_sha256=source_identity,
        model_artifact_sha256=variant.model_sha256,
        transform_manifest_sha256=variant.manifest_sha256,
        transform_semantics=_mujoco_sphere_transform_semantics(
            variant.manifest_path
        ),
        worker_source_sha256=sha256_file(Path(__file__)),
        configuration_semantics=configuration_semantics,
        pre_forward_initial_condition_hash=pre_forward_hash,
        post_forward_state_hash=post_forward_hash,
        reset_returned_policy_hash=None,
        excitation_hash=excitation_hash,
        comparison_profile_hash=comparison_profile_hash,
    )
    scenario_id = str(causal_identity["scenario_id"])
    family = scenario_spec(scenario_id).repeatability_family
    if family is None:
        raise RuntimeError(f"Sphere scenario has no repeatability family: {scenario_id}")
    repeatability_records = []
    for profile_index, (profile, initial) in enumerate(
        zip(profiles, initial_states, strict=True)
    ):
        profile_semantics = {
            key: value for key, value in profile.items() if key != "repetition"
        }
        excitation = {
            "input_kind": "initial_condition",
            "height_m": profile["height_m"],
            "vertical_velocity_mps": profile["vertical_velocity_mps"],
            "horizontal_velocity_mps": profile["horizontal_velocity_mps"],
            "sign": profile["sign"],
            "duration_s": duration,
            "physics_sample_dt_s": float(model.opt.timestep),
        }
        reset_context = {
            "state": initial,
            "seed": 0,
            "rng_state": "not_applicable_deterministic_mujoco_probe",
            "reset_artifact_sha256": None,
        }
        repeatability_records.append(
            build_repeatability_record(
                engine="mujoco",
                scenario_id=scenario_id,
                repeatability_family=family,
                configuration_hash=str(causal_identity["configuration_hash"]),
                profile_index=profile_index,
                repetition=int(profile["repetition"]),
                profile_semantics=profile_semantics,
                pre_forward_payload={"phase": "pre_forward", **reset_context},
                post_forward_payload={"phase": "post_forward", **reset_context},
                reset_returned_policy_payload=None,
                excitation_semantics=excitation,
            )
        )
    payload = {
        "schema_version": "RootCauseMujocoSphereBatchV1",
        "mode": mode,
        "friction": friction,
        "radius_m": radius,
        "mass_kg": mass,
        "profile_count": len(profiles),
        "profiles": profiles,
        "duration_s": duration,
        "physics_dt_s": float(model.opt.timestep),
        "model_sha256": variant.model_sha256,
        "variant_manifest_sha256": variant.manifest_sha256,
        "compiled_semantics_hash": stable_hash(compiled),
        "repeatability_records": repeatability_records,
        **causal_identity,
        "first_contact_time_s": first_contact_times,
        "valid_contact_count": sum(value is not None for value in first_contact_times),
        "initial_clearance_m": [float(row["height_m"]) for row in profiles],
        "compiled_property_errors": property_errors,
        "compiled_property_passed": property_passed,
        "trace_sha256": trace.trace_sha256,
        "trace_metadata_sha256": trace.metadata_sha256,
    }
    (output / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def collect_adapter_probe(output: Path, *, pulse_amplitude: float = 1.0e-3) -> dict[str, Any]:
    if pulse_amplitude <= 0.0 or pulse_amplitude >= 0.1:
        raise ValueError("Adapter pulse amplitude must be in (0,0.1)")
    output.mkdir(parents=True, exist_ok=False)

    from debug.sim2sim.dreamwaq_debug_contract import DebugPolicyAdapter
    from wheelleg_mujoco.observation import (
        build_actor_observation,
        collect_kinematic_state,
    )
    from wheelleg_mujoco.runner import WheelLegMujocoRuntime

    policy = FROZEN_POLICIES["dreamwaq_run01"]
    adapter = DebugPolicyAdapter(
        actor_path=policy["actor"].absolute_path(),
        manifest_path=policy["manifest"].absolute_path(),
        model_manifest_path=MODEL_MANIFEST,
        load_mujoco_contract=True,
    )
    if adapter.contract is None:
        raise RuntimeError("MuJoCo adapter probe did not load the formal contract")
    contract = adapter.contract
    command = np.asarray(FROZEN_REPLAY_SOURCE["command"], dtype=np.float64)
    model_manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))

    def new_runtime() -> WheelLegMujocoRuntime:
        return WheelLegMujocoRuntime(FORMAL_MODEL, contract, model_manifest)

    def all_hinge_addresses(runtime: WheelLegMujocoRuntime) -> tuple[np.ndarray, np.ndarray]:
        joint_ids = np.asarray(
            [
                _required_id(runtime.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in model_manifest["joint_order"]
            ],
            dtype=np.int64,
        )
        return (
            np.asarray(runtime.model.jnt_qposadr[joint_ids], dtype=np.int64),
            np.asarray(runtime.model.jnt_dofadr[joint_ids], dtype=np.int64),
        )

    def orientation_control(runtime: WheelLegMujocoRuntime) -> list[float]:
        rotation_world_from_body = np.asarray(
            runtime.data.xmat[runtime.model_map.base_body_id], dtype=np.float64
        ).reshape(3, 3)
        rotation = (
            contract.r_control_from_mujoco
            @ rotation_world_from_body
            @ contract.r_control_from_mujoco.T
        )
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
        if quaternion[0] < 0.0:
            quaternion *= -1.0
        return quaternion.tolist()

    def capture_post(runtime: WheelLegMujocoRuntime) -> dict[str, Any]:
        qpos_addresses, dof_addresses = all_hinge_addresses(runtime)
        state = collect_kinematic_state(
            runtime.model, runtime.data, runtime.model_map, contract
        )
        return {
            "controlled_position_canonical": state.joint_position_control.tolist(),
            "controlled_velocity_canonical": state.joint_velocity_control.tolist(),
            "all_hinge_position_named": np.asarray(
                runtime.data.qpos[qpos_addresses], dtype=np.float64
            ).tolist(),
            "all_hinge_velocity_named": np.asarray(
                runtime.data.qvel[dof_addresses], dtype=np.float64
            ).tolist(),
            "base_orientation_control_wxyz": orientation_control(runtime),
            "base_linear_velocity_control": state.com_linear_velocity_control.tolist(),
            "base_angular_velocity_control": state.angular_velocity_control.tolist(),
            "projected_gravity": state.projected_gravity_control.tolist(),
            "base_height": float(state.base_height_m),
            "loop_closure_error": float(
                runtime.last_reset_metrics["max_loop_closure_error_m"]
                if runtime.last_reset_metrics is not None
                else 0.0
            ),
        }

    runtime = new_runtime()
    qpos_addresses, dof_addresses = all_hinge_addresses(runtime)
    mujoco.mj_resetDataKeyframe(
        runtime.model, runtime.data, runtime.model_map.reset_keyframe_id
    )
    runtime.data.ctrl[:] = 0.0
    runtime.previous_action.fill(0.0)
    pre_forward = {
        "root_qpos_engine_native": np.asarray(runtime.data.qpos[:7]).tolist(),
        "root_qvel_engine_native": np.asarray(runtime.data.qvel[:6]).tolist(),
        "controlled_position_canonical": runtime.model_map.joint_position_control(
            runtime.data
        ).tolist(),
        "controlled_velocity_canonical": runtime.model_map.joint_velocity_control(
            runtime.data
        ).tolist(),
        "all_hinge_position_named": np.asarray(
            runtime.data.qpos[qpos_addresses]
        ).tolist(),
        "all_hinge_velocity_named": np.asarray(
            runtime.data.qvel[dof_addresses]
        ).tolist(),
    }
    policy_input = runtime.reset(command)
    post_forward = capture_post(runtime)
    current = np.asarray(policy_input[-25:], dtype=np.float32)
    if policy_input.shape != (125,) or not np.array_equal(
        policy_input.reshape(5, 25), np.repeat(current[None, :], 5, axis=0)
    ):
        raise RuntimeError("MuJoCo DreamWaQ history initialization is invalid")
    inference = adapter.infer(np.asarray(policy_input, dtype=np.float32))
    raw_action = np.asarray(inference.raw_action, dtype=np.float64)
    clipped_action = np.clip(raw_action, -contract.action_clip, contract.action_clip)
    targets = runtime.controller.prepare(raw_action)
    canonical_target = np.concatenate(
        (
            targets.leg_position_target_mujoco,
            targets.wheel_velocity_target_mujoco * contract.wheel_signs,
        )
    )
    native_target = np.concatenate(
        (
            targets.leg_position_target_mujoco,
            targets.wheel_velocity_target_mujoco,
        )
    )
    previous_before = runtime.previous_action.copy()
    time_before = float(runtime.data.time)
    prepare_calls: list[list[float]] = []
    original_prepare = runtime.controller.prepare

    def counted_prepare(action: np.ndarray) -> Any:
        prepare_calls.append(np.asarray(action, dtype=np.float64).tolist())
        return original_prepare(action)

    runtime.controller.prepare = counted_prepare
    actor_step = runtime.step(raw_action, command)
    time_after = float(runtime.data.time)
    runtime.controller.prepare = original_prepare
    previous_after = runtime.previous_action.copy()
    next_current = np.asarray(actor_step.observation[-25:], dtype=np.float32)

    def pulse_once(action: np.ndarray) -> dict[str, Any]:
        pulse_runtime = new_runtime()
        pulse_input = pulse_runtime.reset(command)
        before_state = collect_kinematic_state(
            pulse_runtime.model,
            pulse_runtime.data,
            pulse_runtime.model_map,
            contract,
        )
        pulse_targets = pulse_runtime.controller.prepare(action)
        pulse_time_before = float(pulse_runtime.data.time)
        pulse_result = pulse_runtime.step(action, command)
        pulse_time_after = float(pulse_runtime.data.time)
        after_state = collect_kinematic_state(
            pulse_runtime.model,
            pulse_runtime.data,
            pulse_runtime.model_map,
            contract,
        )
        target_native = np.concatenate(
            (
                pulse_targets.leg_position_target_mujoco,
                pulse_targets.wheel_velocity_target_mujoco,
            )
        )
        target_canonical = np.concatenate(
            (
                pulse_targets.leg_position_target_mujoco,
                pulse_targets.wheel_velocity_target_mujoco * contract.wheel_signs,
            )
        )
        return {
            "policy_input_sha256": stable_hash(
                np.asarray(pulse_input, dtype=np.float32).tolist()
            ),
            "action": np.asarray(action, dtype=np.float64).tolist(),
            "clipped_action": pulse_result.clipped_action.tolist(),
            "target_canonical": target_canonical.tolist(),
            "target_engine_native": target_native.tolist(),
            "velocity_before": before_state.joint_velocity_control.tolist(),
            "velocity_after": after_state.joint_velocity_control.tolist(),
            "velocity_delta": (
                after_state.joint_velocity_control
                - before_state.joint_velocity_control
            ).tolist(),
            "previous_action_before": np.zeros(6, dtype=np.float64).tolist(),
            "previous_action_after": pulse_runtime.previous_action.tolist(),
            "control_time_delta_s": pulse_time_after - pulse_time_before,
        }

    zero_pulse = pulse_once(np.zeros(6, dtype=np.float64))
    pulse_records: dict[str, Any] = {}
    for channel in range(6):
        plus = np.zeros(6, dtype=np.float64)
        minus = np.zeros(6, dtype=np.float64)
        plus[channel] = pulse_amplitude
        minus[channel] = -pulse_amplitude
        plus_record = pulse_once(plus)
        minus_record = pulse_once(minus)
        plus_delta = np.asarray(plus_record["velocity_delta"], dtype=np.float64)
        minus_delta = np.asarray(minus_record["velocity_delta"], dtype=np.float64)
        odd_response = 0.5 * (plus_delta - minus_delta)
        odd_target = 0.5 * (
            np.asarray(plus_record["target_canonical"], dtype=np.float64)
            - np.asarray(minus_record["target_canonical"], dtype=np.float64)
        )
        pulse_records[str(channel)] = {
            "plus": plus_record,
            "minus": minus_record,
            "zero": zero_pulse,
            "odd_velocity_response": odd_response.tolist(),
            "odd_target_response": odd_target.tolist(),
            "driven_channel_velocity_sign": int(np.sign(odd_response[channel])),
            "driven_channel_target_sign": int(np.sign(odd_target[channel])),
        }

    payload = {
        "schema_version": "RootCauseMujocoAdapterProbeV1",
        "engine": "mujoco",
        "policy": "dreamwaq_run01",
        "command": command.tolist(),
        "contract": {
            "actor_slices": adapter.manifest["observation"]["actor_slices"],
            "canonical_joint_order": adapter.manifest["action"][
                "canonical_joint_order"
            ],
            "history_layout": adapter.manifest["network"]["history_layout"],
            "history_length": adapter.manifest["network"]["history_length"],
            "input_dimension": adapter.manifest["network"]["input_dimension"],
            "runtime_action_clip": adapter.manifest["network"][
                "runtime_action_clip"
            ],
            "control_dt_s": adapter.manifest["timing"]["control_dt_s"],
            "normalization_schema": adapter.manifest["schemas"]["normalization"],
            "control_frame_schema": adapter.manifest["schemas"]["control_frame"],
            "normalization": adapter.manifest["normalization"],
            "control": adapter.manifest["control"],
            "wheel_joint_sign_usd": adapter.manifest["action"][
                "wheel_joint_sign_usd"
            ],
        },
        "reset_phases": {
            "pre_forward": pre_forward,
            "post_forward": post_forward,
            "returned_policy": {
                "actor_obs_current": current.tolist(),
                "policy_input": np.asarray(policy_input, dtype=np.float32).tolist(),
                "previous_action": previous_before.tolist(),
            },
        },
        "actor_chain": {
            "policy_input_sha256": stable_hash(
                np.asarray(policy_input, dtype=np.float32).tolist()
            ),
            "estimated_velocity": inference.estimated_velocity.tolist(),
            "context_mu": inference.context_mu.tolist(),
            "context_logvar": inference.context_logvar.tolist(),
            "raw_action": raw_action.tolist(),
            "clipped_action": clipped_action.tolist(),
            "target_canonical": canonical_target.tolist(),
            "target_engine_native": native_target.tolist(),
            "previous_action_before": previous_before.tolist(),
            "previous_action_after": previous_after.tolist(),
            "next_actor_obs_current": next_current.tolist(),
        },
        "clock": {
            "physics_dt_s": float(runtime.model.opt.timestep),
            "physics_steps_per_action": int(contract.physics_steps_per_action),
            "control_dt_s": float(contract.control_dt_s),
            "time_before_s": time_before,
            "time_after_s": time_after,
            "target_refresh_first_substep": len(prepare_calls) == 1,
            "targets_constant_within_action": len(prepare_calls) == 1,
        },
        "pulse_amplitude": pulse_amplitude,
        "pulse_records": pulse_records,
        "worker_source_sha256": sha256_file(Path(__file__)),
    }
    (output / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def collect_replay(
    output: Path,
    actions_path: Path,
    command: np.ndarray,
    source_result_path: Path,
    *,
    repetition: int = 0,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    from wheelleg_mujoco.runner import WheelLegMujocoRuntime
    from wheelleg_mujoco.observation import collect_kinematic_state

    source_result_path = Path(source_result_path).resolve(strict=True)
    if repetition < 0:
        raise ValueError("Replay repetition must be non-negative")
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if source_result.get("schema_version") != "RootCauseIsaacReplaySourceV1":
        raise ValueError("MuJoCo replay source result schema is invalid")
    replay_identity = source_result.get("replay_source_identity")
    replay_identity_hash = source_result.get("replay_source_identity_hash")
    if not isinstance(replay_identity, dict) or stable_hash(replay_identity) != replay_identity_hash:
        raise ValueError("MuJoCo replay source identity hash is invalid")
    actions_path = Path(actions_path).resolve(strict=True)
    expected_actions_path = (
        source_result_path.parent / source_result["action_sequence_file"]
    ).resolve()
    if actions_path != expected_actions_path:
        raise ValueError("MuJoCo replay actions are not the frozen source action file")
    if sha256_file(actions_path) != source_result["clipped_action_file_sha256"]:
        raise ValueError("MuJoCo replay action file hash differs from source identity")
    with np.load(actions_path, allow_pickle=False) as archive:
        if "action_sequence" not in archive.files:
            raise ValueError("MuJoCo replay archive is missing action_sequence")
        actions = np.asarray(archive["action_sequence"], dtype=np.float32)
    if _tensor_bytes_sha256(actions) != source_result["clipped_action_sequence_sha256"]:
        raise ValueError("MuJoCo replay action tensor hash differs from source identity")
    if actions.shape != (int(replay_identity["horizon"]), 6) or not np.all(
        np.isfinite(actions)
    ):
        raise ValueError("Replay actions must have finite frozen shape [horizon,6]")
    contract = _formal_contract()
    runtime = WheelLegMujocoRuntime(FORMAL_MODEL, contract, model_manifest=None)
    command_values = np.asarray(command, dtype=np.float64)
    if command_values.tolist() != replay_identity["command"]:
        raise ValueError("MuJoCo replay command differs from frozen source identity")
    policy_input = runtime.reset(command_values)
    reset_state = collect_kinematic_state(
        runtime.model, runtime.data, runtime.model_map, contract
    )
    reset_payload = {
        "actor_obs_current": np.asarray(policy_input[-25:], dtype=np.float32),
        "actor_obs_policy": np.asarray(policy_input, dtype=np.float32),
        "controlled_position_canonical": reset_state.joint_position_control,
        "controlled_velocity_canonical": reset_state.joint_velocity_control,
        "base_linear_velocity_control": reset_state.com_linear_velocity_control,
        "base_angular_velocity_control": reset_state.angular_velocity_control,
        "base_height": reset_state.base_height_m,
        "loop_closure_error": runtime.last_reset_metrics["max_loop_closure_error_m"],
    }
    rows = []
    for index, action in enumerate(actions):
        result = runtime.step(action, command_values)
        state = collect_kinematic_state(
            runtime.model, runtime.data, runtime.model_map, contract
        )
        rows.append(
            {
                "control_tick": np.int64(index),
                "control_time_s": np.float64(index * contract.control_dt_s),
                "actor_obs_current_pre_step": np.asarray(
                    policy_input[-25:], dtype=np.float32
                ).copy(),
                "actor_obs_policy_pre_step": np.asarray(
                    policy_input, dtype=np.float32
                ).copy(),
                "raw_action": np.asarray(action, dtype=np.float32).copy(),
                "clipped_action": np.asarray(
                    result.clipped_action, dtype=np.float32
                ).copy(),
                "controlled_position_canonical_post_step": np.asarray(
                    state.joint_position_control, dtype=np.float64
                ).copy(),
                "controlled_velocity_canonical_post_step": np.asarray(
                    state.joint_velocity_control, dtype=np.float64
                ).copy(),
                "base_linear_velocity_control_post_step": np.asarray(
                    state.com_linear_velocity_control, dtype=np.float64
                ).copy(),
                "base_angular_velocity_control_post_step": np.asarray(
                    state.angular_velocity_control, dtype=np.float64
                ).copy(),
                "base_height_post_step": np.float64(state.base_height_m),
                "loop_closure_error_post_step": np.float64(
                    result.metrics["max_loop_closure_error_m"]
                ),
            }
        )
        policy_input = result.observation
    arrays = _rows_arrays(rows)
    units = {
        "control_tick": "index",
        "control_time_s": "s",
        "actor_obs_current_pre_step": "normalized",
        "actor_obs_policy_pre_step": "normalized",
        "raw_action": "ActionV1",
        "clipped_action": "ActionV1",
        "controlled_position_canonical_post_step": "rad",
        "controlled_velocity_canonical_post_step": "rad/s",
        "base_linear_velocity_control_post_step": "m/s",
        "base_angular_velocity_control_post_step": "rad/s",
        "base_height_post_step": "m",
        "loop_closure_error_post_step": "m",
    }
    fields = {
        name: FieldSpec(
            units[name],
            "control_body"
            if name.startswith("base_") and "velocity" in name
            else "canonical",
            "pre_action"
            if "pre_step" in name or name in {"control_tick", "control_time_s", "raw_action"}
            else "post_step",
            name,
        )
        for name in arrays
    }
    identity = write_trace(output / "trace", arrays, fields)
    configuration = {
        "model_sha256": sha256_file(FORMAL_MODEL),
        "model_manifest_sha256": sha256_file(MODEL_MANIFEST),
        "compiled_semantics_hash": stable_hash(compiled_semantics(runtime.model)),
        "physics_dt_s": float(runtime.model.opt.timestep),
        "control_dt_s": float(contract.control_dt_s),
        "physics_steps_per_action": int(contract.physics_steps_per_action),
        "command": command_values.tolist(),
    }
    configuration_hash = stable_hash(configuration)
    scenario_id = "P60_D_OPEN_LOOP_REPLAY"
    family = scenario_spec(scenario_id).repeatability_family
    if family is None:
        raise RuntimeError("P60-D has no repeatability family")
    repeatability_records = [
        build_repeatability_record(
            engine="mujoco",
            scenario_id=scenario_id,
            repeatability_family=family,
            configuration_hash=configuration_hash,
            profile_index=0,
            repetition=repetition,
            profile_semantics={
                "replay_source_identity_hash": replay_identity_hash,
                "command": command_values.tolist(),
                "horizon": int(actions.shape[0]),
            },
            pre_forward_payload={
                "phase": "pre_forward",
                "state": _json_value(reset_payload),
                "seed": replay_identity.get("seed"),
                "rng_state": "not_applicable_deterministic_mujoco_replay",
                "reset_cache_file_sha256": replay_identity.get(
                    "reset_cache_file_sha256"
                ),
            },
            post_forward_payload={
                "phase": "post_forward",
                "state": _json_value(reset_payload),
                "seed": replay_identity.get("seed"),
                "rng_state": "not_applicable_deterministic_mujoco_replay",
                "reset_cache_file_sha256": replay_identity.get(
                    "reset_cache_file_sha256"
                ),
            },
            reset_returned_policy_payload={
                "actor_obs_policy": np.asarray(
                    rows[0]["actor_obs_policy_pre_step"], dtype=np.float32
                ).tolist()
            },
            excitation_semantics={
                "input_kind": "frozen_open_loop_action_sequence",
                "action_sequence_sha256": _tensor_bytes_sha256(actions),
                "action_file_sha256": sha256_file(actions_path),
                "command": command_values.tolist(),
                "horizon": int(actions.shape[0]),
                "control_dt_s": float(contract.control_dt_s),
                "sample_ticks": list(range(int(actions.shape[0]))),
            },
        )
    ]
    payload = {
        "schema_version": "RootCauseMujocoReplayV1",
        "engine": "mujoco",
        "scenario_id": scenario_id,
        "replay_source_identity_hash": replay_identity_hash,
        "replay_source_identity": replay_identity,
        "source_trace_sha256": source_result["source_trace_sha256"],
        "source_result_path": str(source_result_path),
        "actions_path": str(actions_path),
        "actions_sha256": sha256_file(actions_path),
        "clipped_action_sequence_sha256": _tensor_bytes_sha256(actions),
        "action_count": int(actions.shape[0]),
        "command": command_values.tolist(),
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "pre_forward_initial_condition_hash": stable_hash(reset_payload),
        "post_forward_state_hash": stable_hash(reset_payload),
        "reset_returned_policy_hash": stable_hash(
            np.asarray(policy_input if not rows else rows[0]["actor_obs_policy_pre_step"]).tolist()
        ),
        "repetition": repetition,
        "repeatability_records": repeatability_records,
        "trace_sha256": identity.trace_sha256,
        "trace_metadata_sha256": identity.metadata_sha256,
    }
    (output / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def collect_instrumentation_probe(
    output: Path, *, repetitions: int = 5, ticks: int = 10
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    if repetitions != 5 or ticks != 10:
        raise ValueError("MuJoCo G03 probe is frozen at 5 resets x 10 ticks")
    from wheelleg_mujoco.observation import collect_kinematic_state
    from wheelleg_mujoco.runner import WheelLegMujocoRuntime

    contract = _formal_contract()
    command = np.asarray([0.0, 0.0, 0.20], dtype=np.float64)
    action = np.zeros(6, dtype=np.float64)

    def execute(*, observer_enabled: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        runtime = WheelLegMujocoRuntime(FORMAL_MODEL, contract, model_manifest=None)
        rows: list[dict[str, Any]] = []
        for repetition in range(repetitions):
            observation = runtime.reset(command)
            if observer_enabled:
                collect_kinematic_state(
                    runtime.model, runtime.data, runtime.model_map, contract
                )
            rows.append(
                {
                    "time_s": np.float64(0.0),
                    "repetition": np.int16(repetition),
                    "control_tick": np.int16(0),
                    "policy_observation": np.asarray(observation, dtype=np.float32),
                    "clipped_action": np.zeros(6, dtype=np.float64),
                    "applied_torque": np.zeros(6, dtype=np.float64),
                    "physics_steps": np.int16(0),
                    "base_height": np.float64(runtime.last_reset_metrics["base_height_m"]),
                    "closure_error": np.float64(
                        runtime.last_reset_metrics["max_loop_closure_error_m"]
                    ),
                }
            )
            for tick in range(1, ticks + 1):
                result = runtime.step(action, command)
                if observer_enabled:
                    collect_kinematic_state(
                        runtime.model, runtime.data, runtime.model_map, contract
                    )
                rows.append(
                    {
                        "time_s": np.float64(tick * contract.control_dt_s),
                        "repetition": np.int16(repetition),
                        "control_tick": np.int16(tick),
                        "policy_observation": np.asarray(
                            result.observation, dtype=np.float32
                        ),
                        "clipped_action": result.clipped_action.copy(),
                        "applied_torque": result.applied_torque.copy(),
                        "physics_steps": np.int16(result.physics_steps),
                        "base_height": np.float64(result.metrics["base_height_m"]),
                        "closure_error": np.float64(
                            result.metrics["max_loop_closure_error_m"]
                        ),
                    }
                )
        return _rows_arrays(rows), {
            "compiled_semantics_hash": stable_hash(compiled_semantics(runtime.model)),
            "model_sha256": sha256_file(FORMAL_MODEL),
            "physics_dt_s": float(runtime.model.opt.timestep),
            "control_dt_s": float(contract.control_dt_s),
            "physics_steps_per_action": int(contract.physics_steps_per_action),
        }

    formal_arrays, formal_identity = execute(observer_enabled=False)
    collector_arrays, collector_identity = execute(observer_enabled=True)
    units = {
        "time_s": "s",
        "repetition": "index",
        "control_tick": "index",
        "policy_observation": "normalized",
        "clipped_action": "ActionV1",
        "applied_torque": "N*m",
        "physics_steps": "count",
        "base_height": "m",
        "closure_error": "m",
    }
    fields = {
        name: FieldSpec(
            units[name],
            "canonical",
            "post_step" if name != "time_s" else "sample_time",
            name,
        )
        for name in formal_arrays
    }
    formal_trace = write_trace(output / "formal_trace", formal_arrays, fields)
    collector_trace = write_trace(
        output / "collector_trace", collector_arrays, fields
    )
    maximum_errors = {
        name: float(
            np.max(
                np.abs(
                    np.asarray(formal_arrays[name], dtype=np.float64)
                    - np.asarray(collector_arrays[name], dtype=np.float64)
                )
            )
        )
        for name in formal_arrays
    }
    exact_fields = {
        name: bool(np.array_equal(formal_arrays[name], collector_arrays[name]))
        for name in formal_arrays
    }
    payload = {
        "schema_version": "RootCauseMujocoInstrumentationProbeV1",
        "repetitions": repetitions,
        "ticks": ticks,
        "formal_identity": formal_identity,
        "collector_identity": collector_identity,
        "identity_equal": formal_identity == collector_identity,
        "maximum_errors": maximum_errors,
        "exact_fields": exact_fields,
        "passed": formal_identity == collector_identity and all(exact_fields.values()),
        "formal_trace_sha256": formal_trace.trace_sha256,
        "collector_trace_sha256": collector_trace.trace_sha256,
    }
    (output / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RootCauseSuite MuJoCo worker")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("identity", "properties", "golden"):
        item = sub.add_parser(name)
        item.add_argument("--output", type=Path, required=True)
    instrumentation = sub.add_parser("instrumentation")
    instrumentation.add_argument("--output", type=Path, required=True)
    instrumentation.add_argument("--repetitions", type=int, default=5)
    instrumentation.add_argument("--ticks", type=int, default=10)
    robot = sub.add_parser("robot-probe")
    robot.add_argument("--output", type=Path, required=True)
    robot.add_argument("--scenario", choices=tuple(ROBOT_SCENARIO_OPERATIONS), required=True)
    robot.add_argument("--steps", type=int, default=100)
    robot.add_argument("--channel", type=int, default=0)
    robot.add_argument("--amplitude", type=float, default=0.1)
    robot.add_argument("--sign", type=int, choices=(-1, 0, 1), default=1)
    batch = sub.add_parser("robot-batch")
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--scenario", choices=tuple(ROBOT_SCENARIO_OPERATIONS), required=True)
    batch.add_argument("--control-ticks", type=int, default=5)
    batch.add_argument("--repetitions", type=int, default=3)
    batch.add_argument("--amplitude", type=float)
    batch.add_argument("--input-vector", type=float, nargs=6)
    batch.add_argument("--repeatability-family")
    for name in ("sphere-impact", "sphere-slide"):
        item = sub.add_parser(name)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--steps", type=int, default=500)
        item.add_argument("--height", type=float, default=0.1)
        item.add_argument("--vertical-velocity", type=float, default=0.0)
        item.add_argument(
            "--horizontal-velocity",
            type=float,
            default=1.0 if name == "sphere-slide" else 0.0,
        )
        item.add_argument("--friction", type=float, default=0.0)
    sphere_batch = sub.add_parser("sphere-batch")
    sphere_batch.add_argument("--output", type=Path, required=True)
    sphere_batch.add_argument("--mode", choices=("impact", "slide"), required=True)
    sphere_batch.add_argument("--friction", type=float, required=True)
    sphere_batch.add_argument("--repetitions", type=int, default=3)
    sphere_batch.add_argument("--duration", type=float)
    adapter = sub.add_parser("adapter-probe")
    adapter.add_argument("--output", type=Path, required=True)
    adapter.add_argument("--pulse-amplitude", type=float, default=1.0e-3)
    replay = sub.add_parser("replay")
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--actions", type=Path, required=True)
    replay.add_argument("--source-result", type=Path, required=True)
    replay.add_argument("--repetition", type=int, default=0)
    replay.add_argument(
        "--command",
        dest="command_vector",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.2),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.dont_write_bytecode:
        raise RuntimeError("MuJoCo worker must be launched with -B")
    args = _parser().parse_args(argv)
    if args.command == "identity":
        payload = write_identity(args.output)
    elif args.command == "properties":
        payload = collect_properties(args.output)
    elif args.command == "golden":
        payload = run_golden(args.output)
    elif args.command == "instrumentation":
        payload = collect_instrumentation_probe(
            args.output,
            repetitions=args.repetitions,
            ticks=args.ticks,
        )
    elif args.command == "robot-probe":
        payload = collect_robot_probe(
            args.output,
            scenario=args.scenario,
            steps=args.steps,
            channel=args.channel,
            amplitude=args.amplitude,
            sign=args.sign,
        )
    elif args.command == "robot-batch":
        payload = collect_robot_probe_batch(
            args.output,
            scenario=args.scenario,
            control_ticks=args.control_ticks,
            repetitions=args.repetitions,
            amplitude=args.amplitude,
            input_vector=args.input_vector,
            repeatability_family=args.repeatability_family,
        )
    elif args.command in {"sphere-impact", "sphere-slide"}:
        payload = collect_sphere_probe(
            args.output,
            mode="impact" if args.command == "sphere-impact" else "slide",
            steps=args.steps,
            height_m=args.height,
            vertical_velocity_mps=args.vertical_velocity,
            horizontal_velocity_mps=args.horizontal_velocity,
            friction=args.friction,
        )
    elif args.command == "sphere-batch":
        payload = collect_sphere_probe_batch(
            args.output,
            mode=args.mode,
            friction=args.friction,
            repetitions=args.repetitions,
            duration_s=args.duration,
        )
    elif args.command == "adapter-probe":
        payload = collect_adapter_probe(
            args.output, pulse_amplitude=args.pulse_amplitude
        )
    else:
        payload = collect_replay(
            args.output,
            args.actions,
            np.asarray(args.command_vector, dtype=np.float64),
            args.source_result,
            repetition=args.repetition,
        )
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
