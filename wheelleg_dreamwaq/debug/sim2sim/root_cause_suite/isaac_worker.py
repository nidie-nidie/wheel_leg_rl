from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

from .bootstrap_runtime import finalize_bootstrap
from .contracts import (
    EvidenceIntegrityError,
    FROZEN_POLICIES,
    FROZEN_REPLAY_SOURCE,
    canonical_json_bytes,
    sha256_file,
    stable_hash,
)
from .causal_contract import (
    result_identity_fields,
    robot_configuration_semantics,
    robot_scenario_id,
    scenario_spec,
    sphere_configuration_semantics,
    sphere_scenario_id,
)
from .isaac_bootstrap import (
    kit_argument_string,
    prepare_kit_paths,
    root_cause_launcher_class,
    snapshot_file_handlers,
    validate_frozen_isaac_sources,
    validate_isaaclab_handlers,
    validate_kit_core,
    validate_resolved_kit_state,
    validate_source_contracts,
)
from .python_write_guard import PythonWriteGuard
from .repeatability import build_repeatability_record
from .scenario_catalog import robot_probe_profiles, sphere_probe_profiles
from .trace_contract import FieldSpec, load_verified_trace, write_trace


PHYSX_CFG_FIELDS = (
    "solver_type",
    "solve_articulation_contact_last",
    "min_position_iteration_count",
    "max_position_iteration_count",
    "min_velocity_iteration_count",
    "max_velocity_iteration_count",
    "enable_ccd",
    "enable_stabilization",
    "enable_external_forces_every_iteration",
    "enable_enhanced_determinism",
    "bounce_threshold_velocity",
    "friction_offset_threshold",
    "friction_correlation_distance",
    "gpu_max_rigid_contact_count",
    "gpu_max_rigid_patch_count",
    "gpu_found_lost_pairs_capacity",
    "gpu_found_lost_aggregate_pairs_capacity",
    "gpu_total_aggregate_pairs_capacity",
    "gpu_collision_stack_size",
    "gpu_heap_capacity",
    "gpu_temp_buffer_capacity",
    "gpu_max_num_partitions",
    "gpu_max_soft_body_contacts",
    "gpu_max_particle_contacts",
)
SIMULATION_PHYSICS_FIELDS = (
    "physics_prim_path",
    "device",
    "dt",
    "render_interval",
    "gravity",
    "enable_scene_query_support",
    "use_fabric",
    "create_stage_in_memory",
)


def _isaac_engine_options(sim_cfg: Any, *, resolved_physics_scene: Any) -> dict[str, Any]:
    payload = sim_cfg.to_dict()
    physx = payload.get("physx")
    if not isinstance(physx, dict) or set(physx) != set(PHYSX_CFG_FIELDS):
        raise RuntimeError(
            "Isaac PhysxCfg field set drifted: "
            f"actual={sorted(physx) if isinstance(physx, dict) else None}, "
            f"expected={sorted(PHYSX_CFG_FIELDS)}"
        )
    missing = [name for name in SIMULATION_PHYSICS_FIELDS if name not in payload]
    if missing:
        raise RuntimeError(f"Isaac SimulationCfg physics fields are missing: {missing}")
    return {
        "simulation_cfg": {
            name: payload[name] for name in SIMULATION_PHYSICS_FIELDS
        },
        "physx_cfg": {name: physx[name] for name in PHYSX_CFG_FIELDS},
        "physics_material": payload["physics_material"],
        "resolved_physics_scene": resolved_physics_scene,
    }


def _without_drive_values(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _without_drive_values(value)
            for key, value in payload.items()
            if key not in {"stiffness", "damping"}
        }
    if isinstance(payload, list):
        return [_without_drive_values(value) for value in payload]
    return payload


def _isaac_robot_resolved_semantics(
    env: Any, *, scenario: str, identity: Any
) -> dict[str, Any]:
    compiled = identity.compiled
    hinges = compiled["robot_hinges"]
    actuation_hinges = {
        name: {
            "stiffness": record["stiffness"],
            "damping": record["damping"],
            "armature": record["armature"],
            "position_limits": record["position_limits"],
            "velocity_limit": record["velocity_limit"],
            "effort_limit": record["effort_limit"],
            "resolved_sources_equal": bool(
                record["stiffness"] == record["articulation_data_stiffness"]
                and record["damping"] == record["articulation_data_damping"]
                and record["armature"] == record["articulation_data_armature"]
            ),
        }
        for name, record in hinges.items()
    }
    if not all(
        record["resolved_sources_equal"] for record in actuation_hinges.values()
    ):
        raise RuntimeError("Isaac drive values disagree across resolved runtime views")
    direct = scenario in DIRECT_EFFORT_SCENARIOS
    closure_enabled = scenario not in {
        "p10_c",
        "p30_open_direct",
        "p30_open_target",
        "p40_off",
    }
    ground_cfg = env.cfg.ground.to_dict() if env._root_cause_variant.ground_enabled else None
    closure = compiled["closure_constraints"]
    self_collisions_enabled = compiled["articulation"].get(
        "enabled_self_collisions"
    )
    contact_exclusion = {
        "ground_absent": ground_cfg is None,
        "self_collisions_enabled": self_collisions_enabled,
        "contact_free": bool(
            ground_cfg is None and self_collisions_enabled is False
        ),
    }
    return {
        "topology": compiled["topology"],
        "environment": {
            "ground_enabled": bool(env._root_cause_variant.ground_enabled),
            "ground_geometry": ground_cfg,
            "gravity_enabled": bool(env._root_cause_variant.gravity_enabled),
            "fixed_base": bool(env._root_cause_variant.fixed_base),
        },
        "engine_options": _isaac_engine_options(
            env.cfg.sim, resolved_physics_scene=compiled["physics_scene"]
        ),
        "static_model": {
            "asset_bundle_hash": identity.variant["asset_bundle_hash"],
            "bodies": compiled["bodies"],
            "joints": {
                name: {
                    key: value
                    for key, value in record.items()
                    if key
                    not in {
                        "stiffness",
                        "damping",
                        "articulation_data_stiffness",
                        "articulation_data_damping",
                    }
                }
                for name, record in hinges.items()
            },
            "articulation": compiled["articulation"],
        },
        "closure": {
            "mode": "enabled" if closure_enabled else "disabled",
            "constraints": closure,
        },
        "actuation": {
            "drive_mode": "direct_effort_bypass" if direct else "formal_target_drive",
            "external_controller_enabled": not direct,
            "target_neutralization": direct,
            "robot_hinges": actuation_hinges,
            "actuator_configs_without_drive_values": _without_drive_values(
                compiled["actuator_configs"]
            ),
        },
        "contact": {
            "mode": "asset_and_runtime_materials",
            "pairs": {},
            "runtime_material_properties": compiled["material_properties"],
            "ground": ground_cfg,
            "contact_exclusion": contact_exclusion,
        },
    }


def _isaac_robot_transform_semantics(*, scenario: str, identity: Any) -> dict[str, Any]:
    direct = scenario in DIRECT_EFFORT_SCENARIOS
    closure_enabled = scenario not in {
        "p10_c",
        "p30_open_direct",
        "p30_open_target",
        "p40_off",
    }
    return {
        "actuation": {
            "drive_mode": "direct_effort_bypass" if direct else "formal_target_drive",
            "external_controller_enabled": not direct,
            "target_neutralization": direct,
            "robot_hinges": {
                name: {
                    "stiffness": record["stiffness"],
                    "damping": record["damping"],
                }
                for name, record in identity.compiled["robot_hinges"].items()
            },
        },
        "closure": {
            "mode": "enabled" if closure_enabled else "disabled",
            "constraints": {
                name: {"enabled": record["enabled"]}
                for name, record in identity.compiled[
                    "closure_constraints"
                ].items()
            },
        },
    }


def _without_physics_material(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _without_physics_material(value)
            for key, value in payload.items()
            if key != "physics_material"
        }
    if isinstance(payload, list):
        return [_without_physics_material(value) for value in payload]
    return payload


def _isaac_sphere_resolved_semantics(
    *,
    sim: Any,
    sim_cfg: Any,
    scene_cfg: Any,
    coupon: Any,
    material: Any,
    radius_m: float,
    mass_kg: float,
) -> dict[str, Any]:
    import numpy as np

    from .isaac_probe_env import _prim_semantics

    stage = sim.get_initial_stage()
    physics_scene = stage.GetPrimAtPath(sim_cfg.physics_prim_path)
    ground_prim = stage.GetPrimAtPath(scene_cfg.ground.prim_path)
    coupon_path = "/World/envs/env_0/Coupon"
    coupon_prim = stage.GetPrimAtPath(coupon_path)
    missing = [
        path
        for path, prim in (
            (sim_cfg.physics_prim_path, physics_scene),
            (scene_cfg.ground.prim_path, ground_prim),
            (coupon_path, coupon_prim),
        )
        if not prim.IsValid()
    ]
    if missing:
        raise RuntimeError(f"Isaac common-sphere compiled prims are missing: {missing}")

    masses = coupon.root_physx_view.get_masses().detach().cpu().numpy()
    inertias = coupon.root_physx_view.get_inertias().detach().cpu().numpy()
    coms = coupon.root_physx_view.get_coms().detach().cpu().numpy()
    materials = (
        coupon.root_physx_view.get_material_properties().detach().cpu().numpy()
    )
    expected_material = np.asarray(
        [material.static_friction, material.dynamic_friction, material.restitution],
        dtype=np.float64,
    )
    material_rows = np.asarray(materials, dtype=np.float64).reshape(-1, 3)
    material_error = float(np.max(np.abs(material_rows - expected_material)))
    if material_error > 1.0e-6:
        raise RuntimeError(
            "Isaac common-sphere runtime material differs from the frozen material: "
            f"max_abs={material_error}"
        )

    return {
        "topology": {
            "body_count": 1,
            "joint_count": 0,
            "ground_prim_path": str(scene_cfg.ground.prim_path),
            "coupon_prim_path": coupon_path,
            "contact_pair_names": ["floor_coupon"],
            "environment_count": int(scene_cfg.num_envs),
            "coupon_shape_count": int(material_rows.shape[0] // scene_cfg.num_envs),
        },
        "environment": {
            "ground_enabled": True,
            "gravity_enabled": True,
            "fixed_base": False,
            "gravity": list(sim_cfg.gravity),
            "environment_spacing_m": float(scene_cfg.env_spacing),
            "replicate_physics": bool(scene_cfg.replicate_physics),
            "clone_in_fabric": bool(scene_cfg.clone_in_fabric),
        },
        "engine_options": _isaac_engine_options(
            sim_cfg, resolved_physics_scene=_prim_semantics(physics_scene)
        ),
        "static_model": {
            "ground": {
                "prim": _prim_semantics(ground_prim),
                "configuration_without_material": _without_physics_material(
                    scene_cfg.ground.to_dict()
                ),
            },
            "coupon": {
                "prim": _prim_semantics(coupon_prim),
                "configuration_without_material": _without_physics_material(
                    scene_cfg.coupon.to_dict()
                ),
                "radius_m": float(radius_m),
                "mass_kg": float(mass_kg),
                "compiled_masses": masses.tolist(),
                "compiled_inertias": inertias.tolist(),
                "compiled_centers_of_mass": coms.tolist(),
            },
            "contact_sensor": scene_cfg.contact.to_dict(),
        },
        "contact": {
            "mode": "shared_explicit_material",
            "pairs": {
                "floor_coupon": {
                    "material": {
                        "static_friction": float(material.static_friction),
                        "dynamic_friction": float(material.dynamic_friction),
                        "restitution": float(material.restitution),
                        "friction_combine_mode": material.friction_combine_mode,
                        "restitution_combine_mode": material.restitution_combine_mode,
                        "compliant_contact_stiffness": float(
                            material.compliant_contact_stiffness
                        ),
                        "compliant_contact_damping": float(
                            material.compliant_contact_damping
                        ),
                    },
                    "runtime_material_verified": True,
                    "runtime_material_shape": list(materials.shape),
                }
            },
        },
    }


def _isaac_sphere_transform_semantics(*, friction: float) -> dict[str, Any]:
    return {
        "contact": {
            "pairs": {
                "floor_coupon": {
                    "material": {
                        "static_friction": float(friction),
                        "dynamic_friction": float(friction),
                    }
                }
            }
        }
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RootCauseSuite Isaac worker")
    parser.add_argument(
        "command",
        choices=(
            "identity",
            "one-tick",
            "properties",
            "golden",
            "robot-probe",
            "sphere-probe",
            "adapter-probe",
            "instrumentation-probe",
            "replay-source",
            "replay",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--scenario",
        choices=(
            "p10_a",
            "p10_b",
            "p10_c",
            "p30_open_direct",
            "p30_open_target",
            "p30_closed_direct",
            "p30_closed_target",
            "p40_on",
            "p40_off",
            "p60_b",
            "p60_c",
        ),
    )
    parser.add_argument("--control-ticks", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--amplitude", type=float)
    parser.add_argument("--input-vector", type=float, nargs=6)
    parser.add_argument("--reset-cache", type=Path)
    parser.add_argument("--repeatability-family")
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--sphere-mode", choices=("impact", "slide"))
    parser.add_argument("--friction", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--pulse-amplitude", type=float, default=1.0e-3)
    parser.add_argument(
        "--instrumentation-mode",
        choices=("formal", "debug", "contact", "system_observer"),
    )
    parser.add_argument("--instrumentation-repetitions", type=int, default=5)
    parser.add_argument("--instrumentation-ticks", type=int, default=10)
    return parser


ISAAC_ROBOT_VARIANTS: dict[str, dict[str, bool]] = {
    "p10_a": {
        "ground_enabled": False,
        "gravity_enabled": False,
        "closure_enabled": True,
        "drive_enabled": False,
        "fixed_base": False,
    },
    "p10_b": {
        "ground_enabled": False,
        "gravity_enabled": True,
        "closure_enabled": True,
        "drive_enabled": False,
        "fixed_base": False,
    },
    "p10_c": {
        "ground_enabled": False,
        "gravity_enabled": False,
        "closure_enabled": False,
        "drive_enabled": False,
        "fixed_base": False,
    },
    "p30_open_direct": {
        "ground_enabled": False,
        "gravity_enabled": False,
        "closure_enabled": False,
        "drive_enabled": False,
        "fixed_base": True,
    },
    "p30_open_target": {
        "ground_enabled": False,
        "gravity_enabled": False,
        "closure_enabled": False,
        "drive_enabled": True,
        "fixed_base": True,
    },
    "p30_closed_direct": {
        "ground_enabled": False,
        "gravity_enabled": False,
        "closure_enabled": True,
        "drive_enabled": False,
        "fixed_base": True,
    },
    "p30_closed_target": {
        "ground_enabled": False,
        "gravity_enabled": False,
        "closure_enabled": True,
        "drive_enabled": True,
        "fixed_base": True,
    },
    "p40_on": {
        "ground_enabled": False,
        "gravity_enabled": False,
        "closure_enabled": True,
        "drive_enabled": False,
        "fixed_base": False,
    },
    "p40_off": {
        "ground_enabled": False,
        "gravity_enabled": False,
        "closure_enabled": False,
        "drive_enabled": False,
        "fixed_base": False,
    },
    "p60_b": {
        "ground_enabled": True,
        "gravity_enabled": True,
        "closure_enabled": True,
        "drive_enabled": False,
        "fixed_base": False,
    },
    "p60_c": {
        "ground_enabled": True,
        "gravity_enabled": True,
        "closure_enabled": True,
        "drive_enabled": True,
        "fixed_base": False,
    },
}


DIRECT_EFFORT_SCENARIOS = {
    "p10_a",
    "p10_b",
    "p10_c",
    "p30_open_direct",
    "p30_closed_direct",
    "p40_on",
    "p40_off",
    "p60_b",
}


def _configure_run_environment(run_root: Path) -> dict[str, str]:
    cache = run_root / "runtime_cache"
    paths = {
        "TEMP": cache / "temp",
        "TMP": cache / "temp",
        "CUDA_CACHE_PATH": cache / "cuda",
        "NV_COMPUTE_CACHE_PATH": cache / "cuda",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "TRITON_CACHE_DIR": cache / "triton",
        "MPLCONFIGDIR": cache / "matplotlib",
    }
    for path in set(paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        os.environ[name] = str(path.resolve())
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    return {name: os.environ[name] for name in sorted(paths)}


def _plugin_names(carb_module: Any) -> list[str]:
    result = []
    for plugin in carb_module.get_framework().get_plugins():
        implementation = getattr(plugin, "impl", None)
        name = getattr(implementation, "name", None)
        if name is None:
            name = getattr(plugin, "name", None)
        if name is not None:
            result.append(str(name))
    return sorted(set(result))


def _resolved_kit_paths(carb_module: Any, paths: Any) -> dict[str, str]:
    settings = carb_module.settings.get_settings()
    tokens = carb_module.tokens.get_tokens_interface()

    def setting(name: str, fallback: Path) -> str:
        value = settings.get_as_string(name)
        return str(fallback) if not value else tokens.resolve(value)

    return {
        "log": setting("/log/file", paths.logs / "kit.log"),
        "data": tokens.resolve("${data}"),
        "cache": tokens.resolve("${cache}"),
        "config": setting("/app/userConfigPath", paths.user_config),
        "dump": setting("/crashreporter/dumpDir", paths.crash_dump),
        "texture_cache": setting(
            "/rtx-transient/resourcemanager/localTextureCachePath", paths.texture_cache
        ),
    }


def _to_numpy(value: Any, *, remove_env: bool = True) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if remove_env and array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array).copy()


def _rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if not rows:
        raise ValueError("Cannot serialize an empty Isaac trace")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows[1:]):
        raise ValueError("Isaac trace rows do not share one fixed schema")
    arrays: dict[str, Any] = {}
    for key in keys:
        values = [np.asarray(row[key]) for row in rows]
        shapes = [value.shape for value in values]
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError(f"Isaac trace field {key!r} has inconsistent shapes: {shapes}")
        arrays[key] = np.stack(values) if values[0].ndim else np.asarray(values)
    return arrays


RESET_UNAVAILABLE_SCHEMA_VERSION = "RootCauseResetUnavailableV1"
RESET_CONTACT_FIELDS = frozenset(
    {
        "wheel_contact_active",
        "wheel_normal_force_n",
        "wheel_normal_impulse_ns",
        "wheel_normal_force_world",
        "wheel_friction_force_world",
        "unexpected_contact",
    }
)


def _reset_unavailable_marker(
    array: Any, *, reason: str, sentinel: str
) -> dict[str, Any]:
    return {
        "schema_version": RESET_UNAVAILABLE_SCHEMA_VERSION,
        "availability": "unavailable",
        "reason": reason,
        "sentinel": sentinel,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


def _serialize_reset_state_arrays(arrays: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    missing = sorted(RESET_CONTACT_FIELDS - arrays.keys())
    if missing:
        raise EvidenceIntegrityError(
            f"Reset state is missing frozen contact fields: {missing}"
        )
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    active = normalized["wheel_contact_active"]
    unexpected = normalized["unexpected_contact"]
    if active.dtype != np.dtype(np.int8) or unexpected.dtype != np.dtype(np.int8):
        raise EvidenceIntegrityError(
            "Reset contact integer sentinels must retain exact int8 dtype"
        )
    if active.ndim < 1 or active.shape[-1:] != (2,):
        raise EvidenceIntegrityError(
            f"wheel_contact_active shape drifted: {active.shape}"
        )
    prefix = active.shape[:-1]
    expected_shapes = {
        "wheel_normal_force_n": (*prefix, 2),
        "wheel_normal_impulse_ns": (*prefix, 2),
        "wheel_normal_force_world": (*prefix, 2, 3),
        "wheel_friction_force_world": (*prefix, 2, 3),
        "unexpected_contact": prefix,
    }
    for name, expected_shape in expected_shapes.items():
        value = normalized[name]
        if value.shape != expected_shape:
            raise EvidenceIntegrityError(
                f"Reset contact field {name} shape drifted: "
                f"actual={value.shape}, expected={expected_shape}"
            )
    float_fields = (
        "wheel_normal_force_n",
        "wheel_normal_impulse_ns",
        "wheel_normal_force_world",
        "wheel_friction_force_world",
    )
    for name in float_fields:
        if normalized[name].dtype != np.dtype(np.float32):
            raise EvidenceIntegrityError(
                f"Reset contact field {name} must retain exact float32 dtype"
            )

    disabled = bool(np.all(active == -1))
    if np.any(active == -1) and not disabled:
        raise EvidenceIntegrityError(
            "wheel_contact_active contains a mixed disabled-observer sentinel"
        )
    normal_fields = (
        "wheel_normal_force_n",
        "wheel_normal_impulse_ns",
        "wheel_normal_force_world",
    )
    if disabled:
        for name in normal_fields:
            if not np.isnan(normalized[name]).all():
                raise EvidenceIntegrityError(
                    f"Disabled contact observer field {name} is not exact all-NaN"
                )
    else:
        if not np.isin(active, (0, 1)).all():
            raise EvidenceIntegrityError(
                "Enabled wheel_contact_active values must be exact 0/1"
            )
        for name in normal_fields:
            if not np.isfinite(normalized[name]).all():
                raise EvidenceIntegrityError(
                    f"Enabled contact observer field {name} contains NaN/Inf"
                )
    friction = normalized["wheel_friction_force_world"]
    if not np.isnan(friction).all():
        raise EvidenceIntegrityError(
            "wheel_friction_force_world is not the frozen all-NaN unavailable sentinel"
        )
    if not np.all(unexpected == -1):
        raise EvidenceIntegrityError(
            "unexpected_contact is not the frozen exact -1 unavailable sentinel"
        )

    payload: dict[str, Any] = {}
    for name in sorted(normalized):
        value = normalized[name]
        if name not in RESET_CONTACT_FIELDS:
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise EvidenceIntegrityError(
                    f"Authoritative reset field {name} contains NaN/Inf"
                )
            if not (
                np.issubdtype(value.dtype, np.floating)
                or np.issubdtype(value.dtype, np.integer)
                or np.issubdtype(value.dtype, np.bool_)
            ):
                raise EvidenceIntegrityError(
                    f"Authoritative reset field {name} has unsupported dtype {value.dtype}"
                )
            payload[name] = value.tolist()
            continue
        if name == "wheel_friction_force_world":
            payload[name] = _reset_unavailable_marker(
                value,
                reason="isaaclab_friction_force_unavailable",
                sentinel="all_nan",
            )
        elif name == "unexpected_contact":
            payload[name] = _reset_unavailable_marker(
                value,
                reason="debug_unexpected_contact_observer_unavailable",
                sentinel="all_negative_one",
            )
        elif disabled:
            payload[name] = _reset_unavailable_marker(
                value,
                reason="debug_contact_observer_disabled",
                sentinel=(
                    "all_negative_one"
                    if name == "wheel_contact_active"
                    else "all_nan"
                ),
            )
        else:
            payload[name] = value.tolist()
    return payload


def _phase_hash_payload(state: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    arrays = {
        name: np.asarray(_to_numpy(value)) for name, value in state.items()
    }
    return _serialize_reset_state_arrays(arrays)


def _capture_authoritative_identity_array(name: str, value: Any) -> Any:
    import numpy as np

    array = np.asarray(_to_numpy(value, remove_env=False))
    if array.ndim == 0:
        raise EvidenceIntegrityError(
            f"Authoritative identity field {name} has no environment axis"
        )
    return array.copy()


def _canonical_float32_identity_value(
    name: str, value: Any, *, expected_shape: tuple[int, ...]
) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(_to_numpy(value, remove_env=False))
    if array.shape != expected_shape:
        raise EvidenceIntegrityError(
            f"Authoritative robot reset field {name} must have exact shape "
            f"{expected_shape}, got {array.shape}"
        )
    if array.dtype.str != "<f4":
        raise EvidenceIntegrityError(
            f"Authoritative robot reset field {name} must have exact "
            f"little-endian float32 dtype <f4, got {array.dtype.str}"
        )
    if not np.isfinite(array).all():
        raise EvidenceIntegrityError(
            f"Authoritative robot reset field {name} contains NaN/Inf"
        )
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _repeatability_profile_state(
    state: dict[str, Any],
    profile_index: int,
    *,
    phase: str,
    root_com_position_control: Any,
    command: Any | None = None,
) -> dict[str, Any]:
    import numpy as np

    if phase not in {"pre_forward", "post_forward"}:
        raise ValueError(f"Unknown robot reset phase: {phase}")

    def profile_row(
        name: str, value: Any, *, expected_shape: tuple[int, ...]
    ) -> dict[str, Any]:
        array = np.asarray(_to_numpy(value, remove_env=False))
        if array.ndim == 0 or profile_index >= array.shape[0]:
            raise RuntimeError(
                f"Robot reset field {name} has no profile row {profile_index}"
            )
        return _canonical_float32_identity_value(
            name, array[profile_index], expected_shape=expected_shape
        )

    required = (
        "base_orientation_control_wxyz",
        "base_linear_velocity_control",
        "base_angular_velocity_control",
        "all_hinge_position_named",
        "all_hinge_velocity_named",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise EvidenceIntegrityError(
            f"Robot reset phase is missing authoritative fields: {missing}"
        )
    payload = {
        "root_com_position_control": profile_row(
            "root_com_position_control",
            root_com_position_control,
            expected_shape=(3,),
        ),
        "root_orientation_control_wxyz": profile_row(
            "base_orientation_control_wxyz",
            state["base_orientation_control_wxyz"],
            expected_shape=(4,),
        ),
        "root_linear_velocity_control": profile_row(
            "base_linear_velocity_control",
            state["base_linear_velocity_control"],
            expected_shape=(3,),
        ),
        "root_angular_velocity_control": profile_row(
            "base_angular_velocity_control",
            state["base_angular_velocity_control"],
            expected_shape=(3,),
        ),
        "all_hinge_position_named": profile_row(
            "all_hinge_position_named",
            state["all_hinge_position_named"],
            expected_shape=(26,),
        ),
        "all_hinge_velocity_named": profile_row(
            "all_hinge_velocity_named",
            state["all_hinge_velocity_named"],
            expected_shape=(26,),
        ),
    }
    if phase == "pre_forward":
        if command is None:
            raise EvidenceIntegrityError(
                "Pre-forward robot reset identity requires the sampled command"
            )
        payload["command"] = _canonical_float32_identity_value(
            "command", command, expected_shape=(3,)
        )
    else:
        if "loop_closure_error" not in state:
            raise EvidenceIntegrityError(
                "Post-forward robot reset identity requires closure residuals"
            )
        payload["closure_residual_m"] = profile_row(
            "loop_closure_error",
            state["loop_closure_error"],
            expected_shape=(2,),
        )
    return payload


def _repeatability_returned_policy_state(
    current: Any, previous_action: Any
) -> dict[str, Any]:
    import numpy as np

    current_array = np.asarray(_to_numpy(current, remove_env=False))
    previous_action_array = np.asarray(
        _to_numpy(previous_action, remove_env=False)
    )
    return {
        "actor_obs_policy_returned": _canonical_float32_identity_value(
            "actor_obs_policy_returned", current_array, expected_shape=(25,)
        ),
        "previous_action": _canonical_float32_identity_value(
            "previous_action", previous_action_array, expected_shape=(6,)
        ),
        "initialized_history": _canonical_float32_identity_value(
            "initialized_history",
            np.repeat(current_array[None, :], 5, axis=0),
            expected_shape=(5, 25),
        ),
    }


def _reset_phase_identity_payload(
    *,
    phase: str,
    state_name: str,
    state: Any,
    reset_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in {"pre_forward", "post_forward"}:
        raise ValueError(f"Unknown reset identity phase: {phase}")
    if state_name not in {"state", "states"}:
        raise ValueError(f"Unknown reset identity state field: {state_name}")
    payload = {"phase": phase, state_name: state}
    if phase == "pre_forward":
        if reset_identity is None:
            raise EvidenceIntegrityError(
                "Pre-forward identity requires seed, RNG and reset artifact identity"
            )
        overlap = set(payload) & set(reset_identity)
        if overlap:
            raise EvidenceIntegrityError(
                f"Pre-forward reset identity overwrites phase fields: {sorted(overlap)}"
            )
        payload.update(reset_identity)
    elif reset_identity is not None:
        raise EvidenceIntegrityError(
            "Post-forward identity must not include pre-forward reset identity"
        )
    return payload


def _serial_robot_profiles(
    profiles: list[dict[str, Any]], *, repetitions: int
) -> list[dict[str, Any]]:
    if repetitions < 1 or len(profiles) % repetitions != 0:
        raise EvidenceIntegrityError(
            "Robot profile matrix cannot be partitioned into exact repetitions"
        )
    serial: list[dict[str, Any]] = []
    for start in range(0, len(profiles), repetitions):
        group = profiles[start : start + repetitions]
        actual_repetitions = [int(profile.get("repetition", -1)) for profile in group]
        if actual_repetitions != list(range(repetitions)):
            raise EvidenceIntegrityError(
                "Robot profile group does not contain exact repetitions 0..N-1"
            )
        semantics = [
            {key: value for key, value in profile.items() if key != "repetition"}
            for profile in group
        ]
        if any(candidate != semantics[0] for candidate in semantics[1:]):
            raise EvidenceIntegrityError(
                "Robot profile group changes excitation semantics across repetitions"
            )
        serial.append(semantics[0])
    return serial


def _merge_serial_repetition_arrays(
    repetition_arrays: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    if not repetition_arrays:
        raise EvidenceIntegrityError("Serial robot probe has no repetition arrays")
    field_names = set(repetition_arrays[0])
    if any(set(arrays) != field_names for arrays in repetition_arrays[1:]):
        raise EvidenceIntegrityError(
            "Serial robot repetition traces expose different field sets"
        )
    merged: dict[str, Any] = {}
    for name in sorted(field_names):
        values = [np.asarray(arrays[name]) for arrays in repetition_arrays]
        reference = values[0]
        if reference.ndim < 2:
            raise EvidenceIntegrityError(
                f"Serial robot trace field {name} has no profile axis"
            )
        if any(
            value.shape != reference.shape or value.dtype != reference.dtype
            for value in values[1:]
        ):
            raise EvidenceIntegrityError(
                f"Serial robot trace field {name} changed shape or dtype"
            )
        stacked = np.stack(values, axis=2)
        merged[name] = stacked.reshape(
            reference.shape[0],
            reference.shape[1] * len(values),
            *reference.shape[2:],
        )
    return merged


def _normalized_serial_physics_time_s(
    *,
    absolute_time_s: float,
    origin_s: float,
    tick: int,
    substep_index: int,
    decimation: int,
    physics_dt_s: float,
) -> float:
    import math

    if tick < 0 or decimation < 1 or not 0 <= substep_index < decimation:
        raise ValueError("Invalid serial robot physics schedule index")
    absolute = float(absolute_time_s)
    origin = float(origin_s)
    physics_dt = float(physics_dt_s)
    if not all(math.isfinite(value) for value in (absolute, origin, physics_dt)):
        raise EvidenceIntegrityError(
            "Isaac serial robot probe physics clock inputs must be finite"
        )
    if physics_dt <= 0.0:
        raise ValueError("Serial robot physics dt must be positive")
    relative_physics_time_s = absolute - origin
    expected_physics_time_s = float(
        (tick * decimation + substep_index + 1) * physics_dt
    )
    if not all(
        math.isfinite(value)
        for value in (relative_physics_time_s, expected_physics_time_s)
    ):
        raise EvidenceIntegrityError(
            "Isaac serial robot probe derived physics clock must be finite"
        )
    if abs(relative_physics_time_s - expected_physics_time_s) > 1.0e-10:
        raise RuntimeError(
            "Isaac serial robot probe physics clock drifted: "
            f"actual={relative_physics_time_s}, expected={expected_physics_time_s}"
        )
    return expected_physics_time_s


def _one_tick_fields(arrays: dict[str, Any]) -> dict[str, FieldSpec]:
    units = {
        "time_s": "s",
        "active_joint_position_canonical": "rad",
        "active_joint_velocity_canonical": "rad/s",
        "all_hinge_position_named": "rad",
        "all_hinge_velocity_named": "rad/s",
        "base_com_position_engine_world": "m",
        "base_linear_velocity_control": "m/s",
        "base_angular_velocity_control": "rad/s",
        "loop_closure_error": "m",
    }
    frames = {
        "base_com_position_engine_world": "usd_world",
        "base_linear_velocity_control": "control_body",
        "base_angular_velocity_control": "control_body",
    }
    return {
        name: FieldSpec(
            unit=units[name],
            frame=frames.get(name, "canonical"),
            phase="post_step",
            reference=name,
        )
        for name in arrays
    }


def _robot_profiles(
    scenario: str,
    *,
    repetitions: int,
    amplitude: float | None,
    input_vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    return robot_probe_profiles(
        scenario,
        repetitions=repetitions,
        amplitude=amplitude,
        input_vector=input_vector,
    )


def _robot_probe_fields(arrays: dict[str, Any]) -> dict[str, FieldSpec]:
    units = {
        "time_s": "s",
        "profile_channel": "index",
        "profile_sign": "1",
        "profile_repetition": "index",
        "controlled_position_canonical": "rad",
        "controlled_velocity_canonical": "rad/s",
        "all_hinge_position": "rad",
        "all_hinge_velocity": "rad/s",
        "base_com_position_diag": "m",
        "base_linear_velocity_control": "m/s",
        "base_angular_velocity_control": "rad/s",
        "system_com_position_control": "m",
        "linear_momentum_control": "kg*m/s",
        "angular_momentum_com_control": "kg*m^2/s",
        "kinetic_energy_j": "J",
        "closure_residual_m": "m",
        "commanded_input_canonical": "N*m_or_action",
        "canonical_torque_equivalent": "N*m",
        "host_applied_torque_canonical": "N*m",
        "effort_limit_event": "bool",
        "joint_velocity_limit_exceeded": "bool",
        "terminated": "bool",
        "truncated": "bool",
    }
    frames = {
        "base_com_position_diag": "diagnostic_world",
        "base_linear_velocity_control": "control_body",
        "base_angular_velocity_control": "control_body",
        "system_com_position_control": "control_world",
        "linear_momentum_control": "control_world",
        "angular_momentum_com_control": "control_world",
    }
    return {
        name: FieldSpec(
            unit=units[name],
            frame=frames.get(name, "canonical"),
            phase="sample_time" if name == "time_s" else "post_step",
            reference=name,
        )
        for name in arrays
    }


def _quaternion_geodesic_wxyz(left: Any, right: Any) -> float:
    import numpy as np

    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    lhs /= np.linalg.norm(lhs)
    rhs /= np.linalg.norm(rhs)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(lhs, rhs))), 0.0, 1.0)))


def _quaternion_matrix_wxyz(value: Any) -> Any:
    import numpy as np

    quaternion = np.asarray(value, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _collect_properties(output: Path, *, run_root: Path, device: str) -> dict[str, Any]:
    import numpy as np

    from isaaclab.sim import SimulationContext

    from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
    from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG

    from .compiled_properties import validate_body_mapping
    from .isaac_probe_env import RootCauseIsaacProbeEnv, make_probe_env_cfg
    from .variant_builders import IsaacVariantSpec

    if SimulationContext.instance() is not None:
        raise RuntimeError("Isaac properties probe requires no pre-existing SimulationContext")
    project_root = Path(__file__).resolve().parents[3]
    source_path = project_root / "artifacts" / "phase1_v4" / "mujoco-usd-data.json"
    mapping_path = Path(__file__).with_name("body_mapping_v1.json")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    validate_body_mapping(mapping)
    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    if source.get("asset_bundle_hash") != asset_report.bundle_hash:
        raise RuntimeError("P20 source audit does not match the active AssetBundleV2")

    formal_config_hash_before = stable_hash(WHEELLEG_CFG.to_dict())
    spec = IsaacVariantSpec(
        ground_enabled=False,
        gravity_enabled=False,
        closure_enabled=True,
        drive_enabled=False,
        fixed_base=False,
    )
    cfg = make_probe_env_cfg(
        spec,
        device=device,
        num_envs=1,
        enable_contact_sensors=False,
        episode_length_s=10.02,
    )
    cfg.sim.save_logs_to_file = True
    log_directory = (run_root / "runtime_cache" / "isaaclab" / "logs").resolve()
    log_directory.mkdir(parents=True, exist_ok=True)
    cfg.sim.log_dir = str(log_directory)
    env = RootCauseIsaacProbeEnv(
        cfg,
        variant=spec,
        direct_effort=True,
        enable_contact_sensors=False,
    )
    try:
        model_manifest = json.loads(
            (project_root / "sim2sim" / "mujoco" / "model_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        env.configure_debug_joint_order(list(model_manifest["joint_order"]))
        env.reset(seed=0)
        compiled_by_name = {
            record["name"]: record for record in env.compiled_body_property_records()
        }
        rows: list[dict[str, Any]] = []
        failures: list[str] = []
        for entry in mapping["bodies"]:
            compiled = dict(compiled_by_name[entry["isaac"]])
            inertia_matrix = np.asarray(
                compiled["inertia_com_body_kg_m2"], dtype=np.float64
            )
            audited = source["bodies"][entry["source_audit"]]["inertia"]
            compiled_axes = _quaternion_matrix_wxyz(compiled["principal_axes_wxyz"])
            compiled_principal = compiled_axes.T @ inertia_matrix @ compiled_axes
            compiled["diagonal_inertia_kg_m2"] = np.diag(compiled_principal).tolist()
            compiled["principal_off_diagonal_max_abs"] = float(
                np.max(np.abs(compiled_principal - np.diag(np.diag(compiled_principal))))
            )
            audited_axes = _quaternion_matrix_wxyz(audited["principal_axes_wxyz"])
            audited_inertia_matrix = (
                audited_axes
                @ np.diag(np.asarray(audited["diagonal_inertia_kg_m2"], dtype=np.float64))
                @ audited_axes.T
            )
            mass_error = abs(compiled["mass_kg"] - float(audited["mass_kg"]))
            com_error = float(
                np.max(
                    np.abs(
                        np.asarray(compiled["center_of_mass_m"], dtype=np.float64)
                        - np.asarray(audited["center_of_mass_m"], dtype=np.float64)
                    )
                )
            )
            inertia_error = float(
                np.max(np.abs(inertia_matrix - audited_inertia_matrix))
            )
            orientation_error = _quaternion_geodesic_wxyz(
                compiled["principal_axes_wxyz"], audited["principal_axes_wxyz"]
            )
            passed = (
                mass_error <= max(1.0e-7, 1.0e-6 * abs(float(audited["mass_kg"])))
                and com_error <= 1.0e-6
                and inertia_error
                <= max(
                    1.0e-9,
                    1.0e-5
                    * float(np.max(np.abs(audited["diagonal_inertia_kg_m2"]))),
                )
                and orientation_error <= 1.0e-6
            )
            if not passed:
                failures.append(entry["canonical"])
            rows.append(
                {
                    "mapping": entry,
                    "compiled": compiled,
                    "source_audit": audited,
                    "errors": {
                        "mass": mass_error,
                        "com_max_abs": com_error,
                        "inertia_element_max_abs": inertia_error,
                        "orientation_geodesic_rad": orientation_error,
                    },
                    "passed": passed,
                }
            )
        total_mass = float(sum(row["compiled"]["mass_kg"] for row in rows))
        composite = env.compiled_composite()
        payload = {
            "schema_version": "RootCauseIsaacPropertiesV1",
            "source_audit_sha256": sha256_file(source_path),
            "mapping_sha256": sha256_file(mapping_path),
            "asset_bundle_hash": asset_report.bundle_hash,
            "body_count": len(rows),
            "total_mass_kg": total_mass,
            "total_mass_anchor_error_kg": abs(total_mass - 4.396253988146782),
            "composite": {
                "mass_kg": composite.mass,
                "system_com_position_world_m": composite.com_position_world.tolist(),
                "inertia_com_world_kg_m2": composite.inertia_com_world.tolist(),
            },
            "failed_bodies": failures,
            "audit_valid": True,
            "mismatch_detected": bool(
                failures or abs(total_mass - 4.396253988146782) > 1.0e-6
            ),
            "passed": True,
            "bodies": rows,
            "formal_config_hash_before": formal_config_hash_before,
            "formal_config_hash_after": stable_hash(WHEELLEG_CFG.to_dict()),
        }
        if payload["formal_config_hash_before"] != payload["formal_config_hash_after"]:
            raise RuntimeError("Formal WHEELLEG_CFG changed during the properties probe")
        (output / "properties.json").write_bytes(canonical_json_bytes(payload) + b"\n")
        return payload
    finally:
        env.close()


def _collect_golden(output: Path, *, device: str) -> dict[str, Any]:
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext

    from .compiled_properties import BodyProperty, golden_body_property, momentum_about_origin

    if SimulationContext.instance() is not None:
        raise RuntimeError("Isaac golden requires no pre-existing SimulationContext")
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=device, gravity=(0.0, 0.0, 0.0))
    )
    cfg = RigidObjectCfg(
        prim_path="/World/GoldenBody",
        spawn=sim_utils.SphereCfg(
            radius=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.3),
        ),
    )
    body = RigidObject(cfg)
    sim.reset()
    golden = golden_body_property()
    indices = torch.tensor([0], dtype=torch.int32)
    masses = body.root_physx_view.get_masses()
    masses[0, 0] = golden.mass
    body.root_physx_view.set_masses(masses, indices)
    inertias = body.root_physx_view.get_inertias()
    inertia_link = (
        golden.rotation_world_from_body
        @ golden.inertia_com_body
        @ golden.rotation_world_from_body.T
    )
    inertias[0, :] = torch.as_tensor(
        inertia_link.reshape(-1), dtype=inertias.dtype, device=inertias.device
    )
    body.root_physx_view.set_inertias(inertias, indices)
    coms = body.root_physx_view.get_coms()
    coms[0, :3] = torch.tensor([0.11, -0.07, 0.05], dtype=coms.dtype)
    principal_wxyz = math_utils.quat_from_angle_axis(
        torch.tensor([0.61], device=device),
        torch.tensor([[1.0, 2.0, 3.0]], device=device)
        / torch.linalg.norm(torch.tensor([[1.0, 2.0, 3.0]], device=device), dim=-1, keepdim=True),
    )[0]
    coms[0, 3:7] = math_utils.convert_quat(principal_wxyz, to="xyzw").to(coms.device)
    body.root_physx_view.set_coms(coms, indices)
    root_com_state = torch.zeros((1, 13), dtype=torch.float32, device=device)
    root_com_state[0, :3] = torch.as_tensor(golden.com_position_world, device=device)
    root_com_state[0, 3:7] = principal_wxyz
    root_com_state[0, 7:10] = torch.as_tensor(golden.com_linear_velocity_world, device=device)
    root_com_state[0, 10:13] = torch.as_tensor(golden.angular_velocity_world, device=device)
    body.write_root_com_state_to_sim(root_com_state)
    sim.forward()
    body.update(0.0)
    actual_state = body.data.root_com_state_w[0].detach().cpu().numpy()
    actual_inertia = body.root_physx_view.get_inertias()[0].detach().cpu().numpy().reshape(3, 3)
    actual = BodyProperty(
        name="golden_body",
        mass=float(body.root_physx_view.get_masses()[0, 0].item()),
        com_position_world=actual_state[:3],
        inertia_com_body=actual_inertia,
        rotation_world_from_body=math_utils.matrix_from_quat(
            body.data.root_link_quat_w.to(torch.float64)
        )[0]
        .detach()
        .cpu()
        .numpy(),
        com_linear_velocity_world=actual_state[7:10],
        angular_velocity_world=actual_state[10:13],
    )
    expected_linear, expected_com, expected_origin = momentum_about_origin(
        golden, np.zeros(3)
    )
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
        "schema_version": "RootCauseIsaacGoldenV1",
        "errors": errors,
        "passed": max(errors.values()) <= 1.0e-6,
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
            unit=units[name],
            frame=frames.get(name, "canonical"),
            phase="post_step" if name != "time_s" else "sample_time",
            reference=name,
        )
        for name in arrays
    }


def _sphere_probe_fields(arrays: dict[str, Any]) -> dict[str, FieldSpec]:
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
        "raw_normal_force_n": "N",
        "normal_impulse_ns": "N*s",
        "max_penetration_m": "m",
        "kinetic_energy_j": "J",
    }
    return {
        name: FieldSpec(
            unit=units[name],
            frame="world" if name.endswith("_world") else "canonical",
            phase="post_step" if name != "time_s" else "sample_time",
            reference=name,
        )
        for name in arrays
    }


def _collect_sphere_probe(
    output: Path,
    *,
    device: str,
    mode: str,
    friction: float,
    repetitions: int,
    duration_s: float | None,
) -> dict[str, Any]:
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sim import SimulationContext

    if SimulationContext.instance() is not None:
        raise RuntimeError("Isaac sphere probe requires no pre-existing SimulationContext")
    if mode not in {"impact", "slide"} or friction < 0.0:
        raise ValueError("Invalid Isaac sphere probe parameters")
    profiles = sphere_probe_profiles(mode, repetitions=repetitions)
    duration = (0.5 if mode == "impact" else 0.4) if duration_s is None else float(duration_s)
    physics_dt = 0.005
    steps = int(round(duration / physics_dt))
    if duration <= 0.0 or abs(steps * physics_dt - duration) > 1.0e-12:
        raise ValueError("Sphere duration must be a positive multiple of 5 ms")
    radius = 0.0625
    mass = 0.4
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
        friction_combine_mode="min",
        restitution_combine_mode="min",
    )
    sim_cfg = sim_utils.SimulationCfg(
        dt=physics_dt,
        render_interval=1,
        device=device,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = SimulationContext(sim_cfg)
    scene_cfg = InteractiveSceneCfg(
        num_envs=len(profiles),
        env_spacing=1.0,
        lazy_sensor_update=False,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=material,
        ),
    )
    scene_cfg.coupon = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Coupon",
        spawn=sim_utils.SphereCfg(
            radius=radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            physics_material=material,
            activate_contact_sensors=True,
        ),
    )
    scene_cfg.contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Coupon",
        update_period=0.0,
        history_length=1,
        track_contact_points=False,
        track_friction_forces=False,
        max_contact_data_count_per_prim=4,
        debug_vis=False,
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    coupon = scene["coupon"]
    contact = scene["contact"]
    root_state = coupon.data.default_root_state.clone()
    root_state[:, :3] = scene.env_origins
    root_state[:, 2] += torch.as_tensor(
        [radius + float(row["height_m"]) for row in profiles],
        device=device,
        dtype=root_state.dtype,
    )
    root_state[:, 3:7] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=device, dtype=root_state.dtype
    )
    root_state[:, 7] = torch.as_tensor(
        [float(row["horizontal_velocity_mps"]) for row in profiles],
        device=device,
        dtype=root_state.dtype,
    )
    root_state[:, 8:10] = 0.0
    root_state[:, 9] = torch.as_tensor(
        [float(row["vertical_velocity_mps"]) for row in profiles],
        device=device,
        dtype=root_state.dtype,
    )
    root_state[:, 10:13] = 0.0
    coupon.write_root_state_to_sim(root_state)
    sim.forward()
    scene.update(0.0)
    contact.update(0.0, force_recompute=True)
    post_forward_root_state = coupon.data.root_state_w.detach().cpu().numpy().tolist()
    mass_runtime = coupon.root_physx_view.get_masses().detach().cpu().numpy().reshape(-1)
    inertia_runtime = coupon.root_physx_view.get_inertias().detach().cpu().numpy().reshape(-1, 3, 3)
    expected_inertia = 0.4 * mass * radius * radius
    property_errors = {
        "mass_max_abs": float(np.max(np.abs(mass_runtime - mass))),
        "inertia_max_abs": float(
            np.max(
                np.abs(
                    inertia_runtime
                    - np.broadcast_to(np.eye(3) * expected_inertia, inertia_runtime.shape)
                )
            )
        ),
    }
    first_contact = np.full(len(profiles), np.nan, dtype=np.float64)
    normal_impulse = np.zeros(len(profiles), dtype=np.float64)
    raw_initial_normal_force: list[float] | None = None
    filtered_initial_normal_force: list[float] | None = None
    rows: list[dict[str, Any]] = []
    profile_vectors = {
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

    def capture(time_s: float, *, integrate: bool) -> None:
        state = coupon.data.root_com_state_w.detach().cpu().numpy()
        force_vectors = contact.data.net_forces_w.detach().cpu().numpy().reshape(len(profiles), -1, 3)
        raw_normal_force = np.linalg.norm(force_vectors.sum(axis=1), axis=-1)
        normal_force = raw_normal_force.copy()
        if mode == "impact" and time_s == 0.0:
            # The sensor still contains the pre-teleport spawn contact until the first
            # physics step. Geometry is already above the plane for every impact row.
            normal_force = np.zeros_like(normal_force)
        nonlocal raw_initial_normal_force, filtered_initial_normal_force
        if time_s == 0.0:
            raw_initial_normal_force = raw_normal_force.tolist()
            filtered_initial_normal_force = normal_force.tolist()
        active = normal_force > 1.0e-6
        newly_active = active & ~np.isfinite(first_contact)
        first_contact[newly_active] = time_s
        if integrate:
            normal_impulse[:] += normal_force * physics_dt
        relative_position = state[:, :3] - scene.env_origins.detach().cpu().numpy()
        rows.append(
            {
                "time_s": np.full(len(profiles), time_s, dtype=np.float64),
                **profile_vectors,
                "com_position_world": relative_position,
                "com_velocity_world": state[:, 7:10],
                "angular_velocity_world": state[:, 10:13],
                "contact_count": active.astype(np.int16),
                "normal_force_n": normal_force,
                "raw_normal_force_n": raw_normal_force,
                "normal_impulse_ns": normal_impulse.copy(),
                "max_penetration_m": np.maximum(0.0, radius - relative_position[:, 2]),
                "kinetic_energy_j": 0.5 * mass * np.sum(np.square(state[:, 7:10]), axis=-1),
            }
        )

    capture(0.0, integrate=False)
    for step in range(1, steps + 1):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(physics_dt)
        capture(step * physics_dt, integrate=True)
    arrays = _rows_to_arrays(rows)
    trace = write_trace(output / "trace", arrays, _sphere_probe_fields(arrays))
    initial_states = {
        "root_state": root_state.detach().cpu().numpy().tolist(),
        "profiles": profiles,
    }
    configuration = {
        "physics_dt_s": physics_dt,
        "gravity": [0.0, 0.0, -9.81],
        "radius_m": radius,
        "mass_kg": mass,
        "friction": friction,
        "friction_combine_mode": "min",
        "restitution": 0.0,
    }
    pre_forward_hash = stable_hash(initial_states)
    post_forward_hash = stable_hash(
        {"root_state": post_forward_root_state, "profiles": profiles}
    )
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
    transform_payload = {
        "schema_version": "RootCauseIsaacSphereVariantV1",
        "source_identity": source_identity,
        "friction": friction,
        "configuration": configuration,
    }
    causal_identity = result_identity_fields(
        scenario_id=sphere_scenario_id(mode, friction),
        engine="isaac",
        source_model_sha256=source_identity,
        model_artifact_sha256=stable_hash(transform_payload),
        transform_manifest_sha256=stable_hash(
            {"schema_version": transform_payload["schema_version"], "friction": friction}
        ),
        transform_semantics=_isaac_sphere_transform_semantics(friction=friction),
        worker_source_sha256=sha256_file(Path(__file__)),
        configuration_semantics=sphere_configuration_semantics(
            engine="isaac",
            friction=friction,
            physics_dt_s=physics_dt,
            source_identity=source_identity,
            resolved_semantics=_isaac_sphere_resolved_semantics(
                sim=sim,
                sim_cfg=sim_cfg,
                scene_cfg=scene_cfg,
                coupon=coupon,
                material=material,
                radius_m=radius,
                mass_kg=mass,
            ),
        ),
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
    origin_values = scene.env_origins.detach().cpu().numpy()
    pre_root_values = root_state.detach().cpu().numpy()
    post_root_values = np.asarray(post_forward_root_state, dtype=np.float64)
    repeatability_records = []
    for profile_index, profile in enumerate(profiles):
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
            "physics_sample_dt_s": physics_dt,
        }

        def canonical_root(values: Any) -> dict[str, Any]:
            row = np.asarray(values[profile_index], dtype=np.float64)
            return {
                "position_relative": (row[:3] - origin_values[profile_index]).tolist(),
                "quaternion_wxyz": row[3:7].tolist(),
                "linear_velocity_world": row[7:10].tolist(),
                "angular_velocity_world": row[10:13].tolist(),
            }

        common_reset = {
            "seed": 0,
            "rng_state": "not_applicable_deterministic_isaac_sphere_probe",
            "reset_artifact_sha256": None,
        }
        repeatability_records.append(
            build_repeatability_record(
                engine="isaac",
                scenario_id=scenario_id,
                repeatability_family=family,
                configuration_hash=str(causal_identity["configuration_hash"]),
                profile_index=profile_index,
                repetition=int(profile["repetition"]),
                profile_semantics=profile_semantics,
                pre_forward_payload={
                    "phase": "pre_forward",
                    "state": canonical_root(pre_root_values),
                    **common_reset,
                },
                post_forward_payload={
                    "phase": "post_forward",
                    "state": canonical_root(post_root_values),
                    **common_reset,
                },
                reset_returned_policy_payload=None,
                excitation_semantics=excitation,
            )
        )
    relative_initial_position = (
        np.asarray(post_forward_root_state, dtype=np.float64)[:, :3]
        - scene.env_origins.detach().cpu().numpy()
    )
    initial_clearance = (relative_initial_position[:, 2] - radius).tolist()
    property_passed = max(property_errors.values()) <= 1.0e-6
    payload = {
        "schema_version": "RootCauseIsaacSphereProbeV1",
        "mode": mode,
        "friction": friction,
        "radius_m": radius,
        "mass_kg": mass,
        "profile_count": len(profiles),
        "profiles": profiles,
        "duration_s": duration,
        "physics_dt_s": physics_dt,
        "repeatability_records": repeatability_records,
        **causal_identity,
        "first_contact_time_s": [
            None if not np.isfinite(value) else float(value) for value in first_contact
        ],
        "valid_contact_count": int(np.isfinite(first_contact).sum()),
        "initial_clearance_m": initial_clearance,
        "raw_initial_normal_force_n": raw_initial_normal_force,
        "filtered_initial_normal_force_n": filtered_initial_normal_force,
        "compiled_property_errors": property_errors,
        "compiled_property_passed": property_passed,
        "trace_sha256": trace.trace_sha256,
        "trace_metadata_sha256": trace.metadata_sha256,
    }
    (output / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def _tensor_bytes_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(_to_numpy(value, remove_env=False))
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


def _randomization_rng_identity(env: Any) -> dict[str, Any]:
    state = env.get_randomization_rng_state()
    streams = state.get("streams")
    if not isinstance(streams, dict):
        raise RuntimeError("Environment randomization RNG state is missing")
    payload = {
        "seed_derivation_version": state.get("seed_derivation_version"),
        "streams": {
            name: {
                "device": record.get("device"),
                "state_sha256": _tensor_bytes_sha256(record.get("state")),
            }
            for name, record in sorted(streams.items())
        },
    }
    payload["identity_hash"] = stable_hash(payload)
    return payload


def _load_frozen_replay_inputs() -> dict[str, Any]:
    import torch

    from wheelleg_dreamwaq.schemas.isaac_evaluation import (
        EVALUATION_ACTION_STEPS,
        EVALUATION_ENV_COUNT,
        EVALUATION_SEED,
        FORMAL_SCENARIOS,
        build_evaluation_reset_cache_identity,
    )
    from wheelleg_dreamwaq.schemas.randomization import (
        validate_closed_chain_reset_cache_artifact,
    )

    frozen = FROZEN_REPLAY_SOURCE
    cache_record = frozen["reset_cache"]
    summary_record = frozen["evaluation_summary"]
    cache_path = cache_record.absolute_path()
    summary_path = summary_record.absolute_path()
    for label, record in (("reset cache", cache_record), ("evaluation summary", summary_record)):
        actual = sha256_file(record.absolute_path())
        if actual != record.sha256:
            raise ValueError(f"Frozen replay {label} hash drifted: {actual} != {record.sha256}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    embedded_report_hash = summary.get("report_hash")
    if embedded_report_hash != stable_hash(
        {key: value for key, value in summary.items() if key != "report_hash"}
    ):
        raise ValueError("Frozen Isaac evaluation summary embedded hash is invalid")
    if summary.get("evaluation_contract_hash") != frozen["evaluation_contract_hash"]:
        raise ValueError("Frozen Isaac evaluation contract hash drifted")
    cache = validate_closed_chain_reset_cache_artifact(
        torch.load(cache_path, map_location="cpu", weights_only=False)
    )
    cache_identity = build_evaluation_reset_cache_identity(
        path=str(cache_path.resolve()),
        file_sha256=sha256_file(cache_path),
        cache_payload=cache,
    )
    expected_cache = {
        "file_sha256": cache_record.sha256,
        "tensor_sha256": frozen["reset_cache_tensor_sha256"],
        "identity_hash": frozen["reset_cache_identity_hash"],
        "cache_schema_version": frozen["reset_cache_schema"],
        "relaxation_algorithm_version": frozen["reset_cache_relaxation_algorithm"],
        "root_height_algorithm_version": frozen["reset_cache_root_height_algorithm"],
        "num_envs": frozen["environment_count"],
    }
    for field, expected in expected_cache.items():
        if cache_identity.get(field) != expected:
            raise ValueError(
                f"Frozen replay cache identity mismatch for {field}: "
                f"{cache_identity.get(field)!r} != {expected!r}"
            )
    if EVALUATION_ENV_COUNT != frozen["environment_count"] or EVALUATION_SEED != frozen["seed"]:
        raise ValueError("Isaac evaluation constants drifted from the frozen replay source")
    if EVALUATION_ACTION_STEPS != frozen["horizon"]:
        raise ValueError("Isaac evaluation horizon drifted from the frozen replay source")
    if FORMAL_SCENARIOS[0] != (frozen["scenario_name"], frozen["command"]):
        raise ValueError("Formal scenario row zero drifted from nominal_stand")
    if summary.get("reset_cache") != cache_identity:
        raise ValueError("Evaluation summary and frozen reset-cache identity differ")
    return {
        "cache": cache,
        "cache_identity": cache_identity,
        "cache_path": cache_path.resolve(),
        "summary_path": summary_path.resolve(),
        "summary": summary,
        "formal_scenarios": FORMAL_SCENARIOS,
    }


def _create_formal_replay_env(
    *,
    run_root: Path,
    device: str,
    cache: dict[str, Any],
) -> tuple[Any, dict[str, Any], Any]:
    import torch

    from debug.sim2sim.isaac_debug_env import WheelLegSim2SimDebugEnv, make_debug_env_cfg
    from wheelleg_dreamwaq.schemas.isaac_evaluation import (
        EVALUATION_ENV_COUNT,
        EVALUATION_SEED,
        FORMAL_SCENARIOS,
    )
    from wheelleg_dreamwaq.schemas.randomization import NOMINAL_EVALUATION_PROFILE_V1

    cfg = make_debug_env_cfg(device=device, enable_contact_sensors=False)
    cfg.scene.num_envs = EVALUATION_ENV_COUNT
    cfg.seed = EVALUATION_SEED
    cfg.randomization = NOMINAL_EVALUATION_PROFILE_V1
    cfg.episode_length_s = 10.0
    cfg.sim.save_logs_to_file = True
    log_directory = (run_root / "runtime_cache" / "isaaclab" / "logs").resolve()
    log_directory.mkdir(parents=True, exist_ok=True)
    cfg.sim.log_dir = str(log_directory)
    env = WheelLegSim2SimDebugEnv(
        cfg,
        enable_contact_sensors=False,
        closed_chain_reset_cache=cache,
    )
    model_manifest = json.loads(
        (Path(__file__).resolve().parents[3] / "sim2sim" / "mujoco" / "model_manifest.json")
        .read_text(encoding="utf-8")
    )
    env.configure_debug_joint_order(list(model_manifest["joint_order"]))
    env.reset(seed=EVALUATION_SEED)
    commands = torch.tensor(
        [command for _, command in FORMAL_SCENARIOS],
        dtype=torch.float32,
        device=env.device,
    )
    env._commands.copy_(commands)
    observations = env._get_observations()
    env.obs_buf = observations
    expected_commands = cfg.normalization.normalize_command(commands)
    if not torch.allclose(
        observations["policy"][:, 6:9], expected_commands, rtol=0.0, atol=1.0e-6
    ):
        env.close()
        raise RuntimeError("Frozen formal commands are absent from the replay first observation")
    reset = env._debug_reset_snapshot
    if reset is None:
        env.close()
        raise RuntimeError("Replay environment did not expose reset phases")
    reset_identity = {
        "pre_forward_initial_condition_hash": stable_hash(
            _phase_hash_payload(reset["reset_written_pre_forward"])
        ),
        "post_forward_state_hash": stable_hash(
            _phase_hash_payload(reset["reset_forwarded_post_forward"])
        ),
        "reset_returned_policy_hash": stable_hash(
            _to_numpy(observations["policy"], remove_env=False).tolist()
        ),
    }
    return env, reset_identity, observations


def _replay_trace_fields(arrays: dict[str, Any]) -> dict[str, FieldSpec]:
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
        "terminated": "bool",
        "truncated": "bool",
        "next_actor_obs_current_returned": "normalized",
    }
    frames = {
        "base_linear_velocity_control_post_step": "control_body",
        "base_angular_velocity_control_post_step": "control_body",
    }
    return {
        name: FieldSpec(
            unit=units[name],
            frame=frames.get(name, "canonical"),
            phase=("pre_action" if "pre_step" in name or name in {"control_tick", "raw_action"} else "post_step"),
            reference=name,
        )
        for name in arrays
    }


def _collect_isaac_replay(
    output: Path,
    *,
    run_root: Path,
    device: str,
    source_mode: bool,
    actions_path: Path | None,
    source_result_path: Path | None,
    repetition: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from debug.sim2sim.dreamwaq_debug_contract import DebugPolicyAdapter
    from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG

    frozen_inputs = _load_frozen_replay_inputs()
    if repetition < 0:
        raise ValueError("Replay repetition must be non-negative")
    frozen = FROZEN_REPLAY_SOURCE
    policy = FROZEN_POLICIES["dreamwaq_run01"]
    for record in policy.values():
        if sha256_file(record.absolute_path()) != record.sha256:
            raise ValueError(f"Frozen DreamWaQ replay policy hash drifted: {record.relative_path}")
    source_result = None
    replay_actions = None
    replay_environment_actions = None
    if source_mode:
        if actions_path is not None or source_result_path is not None:
            raise ValueError("replay-source does not accept existing actions or a source result")
    else:
        if actions_path is None or source_result_path is None:
            raise ValueError("replay requires --actions and --source-result")
        source_result_path = source_result_path.resolve(strict=True)
        source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
        if source_result.get("schema_version") != "RootCauseIsaacReplaySourceV1":
            raise ValueError("Fresh replay source result schema is invalid")
        actions_path = actions_path.resolve(strict=True)
        if actions_path != (source_result_path.parent / source_result["action_sequence_file"]).resolve():
            raise ValueError("Fresh replay actions are not the frozen source action file")
        if sha256_file(actions_path) != source_result["clipped_action_file_sha256"]:
            raise ValueError("Fresh replay action file hash differs from source identity")
        with np.load(actions_path, allow_pickle=False) as archive:
            replay_actions = np.asarray(archive["action_sequence"], dtype=np.float32)
            replay_environment_actions = np.asarray(
                archive["environment_action_sequence"], dtype=np.float32
            )
        if _tensor_bytes_sha256(replay_actions) != source_result["clipped_action_sequence_sha256"]:
            raise ValueError("Fresh replay action tensor hash differs from source identity")
        if replay_actions.shape != (frozen["horizon"], 6):
            raise ValueError("Fresh replay row-zero action shape is invalid")
        if replay_environment_actions.shape != (
            frozen["horizon"], frozen["environment_count"], 6
        ):
            raise ValueError("Fresh replay environment action shape is invalid")

    formal_config_hash_before = stable_hash(WHEELLEG_CFG.to_dict())
    env, reset_identity, observations = _create_formal_replay_env(
        run_root=run_root,
        device=device,
        cache=frozen_inputs["cache"],
    )
    try:
        adapter = DebugPolicyAdapter(
            actor_path=policy["actor"].absolute_path(),
            manifest_path=policy["manifest"].absolute_path(),
            model_manifest_path=(
                Path(__file__).resolve().parents[3]
                / "sim2sim"
                / "mujoco"
                / "model_manifest.json"
            ),
            load_mujoco_contract=False,
        )
        current = _to_numpy(observations["policy"], remove_env=False).astype(
            np.float32, copy=False
        )
        history = np.repeat(current[:, None, :], 5, axis=1).reshape(
            frozen["environment_count"], 125
        )
        rng_identity = _randomization_rng_identity(env)
        rows: list[dict[str, Any]] = []
        clipped_rows: list[np.ndarray] = []
        environment_clipped_rows: list[np.ndarray] = []
        for tick in range(frozen["horizon"]):
            pre_current = _to_numpy(env.obs_buf["policy"], remove_env=False).astype(
                np.float32, copy=False
            )
            if not np.array_equal(pre_current, history[:, -25:]):
                raise RuntimeError("Replay current observation differs from latest history frame")
            if source_mode:
                with torch.inference_mode():
                    raw_batch = (
                        adapter.actor(torch.from_numpy(history))
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                selected = raw_batch
            else:
                assert replay_environment_actions is not None
                selected = replay_environment_actions[tick]
                raw_batch = selected
            next_observations, _, terminated, truncated, _ = env.step(
                torch.from_numpy(np.asarray(selected, dtype=np.float32)).to(env.device)
            )
            terminal = env._debug_terminal_state
            if terminal is None or env._debug_action_clipped is None or env._debug_next_obs is None:
                raise RuntimeError("Replay environment omitted terminal diagnostics")
            clipped = _to_numpy(env._debug_action_clipped, remove_env=False).astype(
                np.float32, copy=False
            )
            expected_clipped = np.clip(selected, -1.0, 1.0).astype(np.float32)
            if not np.array_equal(clipped, expected_clipped):
                raise RuntimeError("Isaac replay runtime clipping differs from ActionV1")
            next_current = _to_numpy(env._debug_next_obs, remove_env=False).astype(
                np.float32, copy=False
            )
            terminated_np = _to_numpy(terminated, remove_env=False).astype(np.int8)
            truncated_np = _to_numpy(truncated, remove_env=False).astype(np.int8)
            rows.append(
                {
                    "control_tick": np.int64(tick),
                    "control_time_s": np.float64(tick * 0.02),
                    "actor_obs_current_pre_step": pre_current[0],
                    "actor_obs_policy_pre_step": history[0],
                    "raw_action": np.asarray(raw_batch[0], dtype=np.float32),
                    "clipped_action": clipped[0],
                    "controlled_position_canonical_post_step": _to_numpy(
                        terminal["active_joint_position_canonical"], remove_env=False
                    )[0],
                    "controlled_velocity_canonical_post_step": _to_numpy(
                        terminal["active_joint_velocity_canonical"], remove_env=False
                    )[0],
                    "base_linear_velocity_control_post_step": _to_numpy(
                        terminal["base_linear_velocity_control"], remove_env=False
                    )[0],
                    "base_angular_velocity_control_post_step": _to_numpy(
                        terminal["base_angular_velocity_control"], remove_env=False
                    )[0],
                    "base_height_post_step": np.float64(
                        _to_numpy(terminal["base_height"], remove_env=False)[0]
                    ),
                    "loop_closure_error_post_step": _to_numpy(
                        terminal["loop_closure_error"], remove_env=False
                    )[0],
                    "terminated": terminated_np[0],
                    "truncated": truncated_np[0],
                    "next_actor_obs_current_returned": next_current[0],
                }
            )
            clipped_rows.append(clipped[0].copy())
            environment_clipped_rows.append(clipped.copy())
            done = (terminated_np | truncated_np).astype(bool)
            advanced = np.empty_like(history.reshape(frozen["environment_count"], 5, 25))
            previous = history.reshape(frozen["environment_count"], 5, 25)
            advanced[:, :-1] = previous[:, 1:]
            advanced[:, -1] = next_current
            if np.any(done):
                advanced[done] = np.repeat(next_current[done, None, :], 5, axis=1)
            history = advanced.reshape(frozen["environment_count"], 125)
            env.obs_buf = next_observations
            if terminated_np[0] or (truncated_np[0] and tick != frozen["horizon"] - 1):
                raise RuntimeError(f"Frozen row-zero replay ended early at control tick {tick}")
        if not rows[-1]["truncated"] or rows[-1]["terminated"]:
            raise RuntimeError("Frozen row-zero replay did not timeout exactly at action 499")
        arrays = _rows_to_arrays(rows)
        trace = write_trace(output / "trace", arrays, _replay_trace_fields(arrays))
        clipped_sequence = np.asarray(clipped_rows, dtype=np.float32)
        environment_sequence = np.asarray(environment_clipped_rows, dtype=np.float32)
        action_path = output / "actions.npz"
        if source_mode:
            np.savez(
                action_path,
                action_sequence=clipped_sequence,
                environment_action_sequence=environment_sequence,
            )
            action_file_sha = sha256_file(action_path)
            action_tensor_sha = _tensor_bytes_sha256(clipped_sequence)
        else:
            assert actions_path is not None and source_result is not None
            if not np.array_equal(clipped_sequence, replay_actions):
                raise RuntimeError("Fresh Isaac replay altered the frozen clipped actions")
            action_file_sha = source_result["clipped_action_file_sha256"]
            action_tensor_sha = source_result["clipped_action_sequence_sha256"]
        configuration = {
            "formal_config_hash": formal_config_hash_before,
            "evaluation_contract_hash": frozen["evaluation_contract_hash"],
            "environment_count": frozen["environment_count"],
            "command_rows": [list(command) for _, command in frozen_inputs["formal_scenarios"]],
            "physics_dt_s": float(env.physics_dt),
            "decimation": int(env.cfg.decimation),
            "control_dt_s": float(env.step_dt),
        }
        identity_payload = {
            "source_scenario_id": frozen["source_scenario_id"],
            "generator_actor_sha256": policy["actor"].sha256,
            "generator_manifest_sha256": policy["manifest"].sha256,
            "evaluation_summary_sha256": FROZEN_REPLAY_SOURCE["evaluation_summary"].sha256,
            "evaluation_contract_hash": frozen["evaluation_contract_hash"],
            "reset_cache_file_sha256": FROZEN_REPLAY_SOURCE["reset_cache"].sha256,
            "reset_cache_tensor_sha256": frozen["reset_cache_tensor_sha256"],
            "reset_cache_identity_hash": frozen["reset_cache_identity_hash"],
            "cache_schema_version": frozen_inputs["cache_identity"][
                "cache_schema_version"
            ],
            "relaxation_algorithm_version": frozen_inputs["cache_identity"][
                "relaxation_algorithm_version"
            ],
            "root_height_algorithm_version": frozen_inputs["cache_identity"][
                "root_height_algorithm_version"
            ],
            "scenario_name": frozen["scenario_name"],
            "environment_count": frozen["environment_count"],
            "environment_index": frozen["environment_index"],
            "command": list(frozen["command"]),
            "seed": frozen["seed"],
            "repetition_index": frozen["repetition_index"],
            "horizon": frozen["horizon"],
            "configuration_hash": stable_hash(configuration),
            **reset_identity,
            "rng_identity_hash": rng_identity["identity_hash"],
            "source_trace_sha256": (
                trace.trace_sha256 if source_mode else source_result["source_trace_sha256"]
            ),
            "clipped_action_file_sha256": action_file_sha,
            "clipped_action_sequence_sha256": action_tensor_sha,
            "action_count": int(clipped_sequence.shape[0]),
        }
        if source_mode:
            replay_identity_hash = stable_hash(identity_payload)
            equivalence = None
            schema_version = "RootCauseIsaacReplaySourceV1"
            repeatability_records: list[dict[str, Any]] = []
        else:
            replay_identity_hash = str(source_result["replay_source_identity_hash"])
            expected_identity = dict(source_result["replay_source_identity"])
            if stable_hash(expected_identity) != replay_identity_hash:
                raise ValueError("Frozen replay source identity hash is invalid")
            for field in (
                "configuration_hash",
                "pre_forward_initial_condition_hash",
                "post_forward_state_hash",
                "reset_returned_policy_hash",
                "rng_identity_hash",
            ):
                if identity_payload[field] != expected_identity[field]:
                    raise ValueError(f"Fresh replay identity differs for {field}")
            source_trace = load_verified_trace(source_result_path.parent / "trace")
            comparison_fields = (
                "actor_obs_current_pre_step",
                "actor_obs_policy_pre_step",
                "clipped_action",
                "controlled_position_canonical_post_step",
                "controlled_velocity_canonical_post_step",
                "base_linear_velocity_control_post_step",
                "base_angular_velocity_control_post_step",
                "base_height_post_step",
                "loop_closure_error_post_step",
            )
            maximum = {
                field: float(
                    np.max(
                        np.abs(
                            np.asarray(arrays[field], dtype=np.float64)
                            - np.asarray(source_trace.arrays[field], dtype=np.float64)
                        )
                    )
                )
                for field in comparison_fields
            }
            discrete_match = all(
                np.array_equal(arrays[field], source_trace.arrays[field])
                for field in ("terminated", "truncated")
            )
            equivalence = {
                "maximum_abs_error": maximum,
                "discrete_match": discrete_match,
                "effective_tolerance": 1.0e-6,
                "passed": max(maximum.values(), default=0.0) <= 1.0e-6 and discrete_match,
                "source_trace_sha256": source_trace.identity.trace_sha256,
                "fresh_trace_sha256": trace.trace_sha256,
            }
            if not equivalence["passed"]:
                raise RuntimeError(f"Fresh Isaac replay equivalence failed: {equivalence}")
            schema_version = "RootCauseIsaacFreshReplayV1"
            scenario_id = "P60_D_OPEN_LOOP_REPLAY"
            family = scenario_spec(scenario_id).repeatability_family
            if family is None:
                raise RuntimeError("P60-D has no repeatability family")
            repeatability_records = [
                build_repeatability_record(
                    engine="isaac",
                    scenario_id=scenario_id,
                    repeatability_family=family,
                    configuration_hash=str(identity_payload["configuration_hash"]),
                    profile_index=0,
                    repetition=repetition,
                    profile_semantics={
                        "replay_source_identity_hash": replay_identity_hash,
                        "command": list(frozen["command"]),
                        "horizon": frozen["horizon"],
                    },
                    pre_forward_payload={
                        "phase_hash": reset_identity[
                            "pre_forward_initial_condition_hash"
                        ],
                        "seed": frozen["seed"],
                        "rng_identity_hash": rng_identity["identity_hash"],
                        "reset_cache_file_sha256": FROZEN_REPLAY_SOURCE[
                            "reset_cache"
                        ].sha256,
                    },
                    post_forward_payload={
                        "phase_hash": reset_identity["post_forward_state_hash"],
                        "seed": frozen["seed"],
                        "rng_identity_hash": rng_identity["identity_hash"],
                        "reset_cache_file_sha256": FROZEN_REPLAY_SOURCE[
                            "reset_cache"
                        ].sha256,
                    },
                    reset_returned_policy_payload={
                        "phase_hash": reset_identity["reset_returned_policy_hash"]
                    },
                    excitation_semantics={
                        "input_kind": "frozen_open_loop_action_sequence",
                        "action_sequence_sha256": action_tensor_sha,
                        "action_file_sha256": action_file_sha,
                        "command": list(frozen["command"]),
                        "horizon": frozen["horizon"],
                        "control_dt_s": float(env.step_dt),
                        "sample_ticks": list(range(frozen["horizon"])),
                    },
                )
            ]
        payload = {
            "schema_version": schema_version,
            "engine": "isaac",
            "scenario_id": (
                None if source_mode else "P60_D_OPEN_LOOP_REPLAY"
            ),
            "replay_source_identity_hash": replay_identity_hash,
            "replay_source_identity": identity_payload if source_mode else source_result["replay_source_identity"],
            "configuration": configuration,
            "configuration_hash": identity_payload["configuration_hash"],
            "reset_identity": reset_identity,
            "rng_identity": rng_identity,
            "source_trace_sha256": identity_payload["source_trace_sha256"],
            "trace_sha256": trace.trace_sha256,
            "trace_metadata_sha256": trace.metadata_sha256,
            "action_sequence_file": "actions.npz" if source_mode else str(actions_path),
            "clipped_action_file_sha256": action_file_sha,
            "clipped_action_sequence_sha256": action_tensor_sha,
            "action_count": int(clipped_sequence.shape[0]),
            "first_action": clipped_sequence[0].tolist(),
            "fresh_replay_equivalence": equivalence,
            "repetition": repetition,
            "repeatability_records": repeatability_records,
            "formal_config_hash_before": formal_config_hash_before,
            "formal_config_hash_after": stable_hash(WHEELLEG_CFG.to_dict()),
        }
        if payload["formal_config_hash_before"] != payload["formal_config_hash_after"]:
            raise RuntimeError("Formal WHEELLEG_CFG changed during replay collection")
        (output / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
        return payload
    finally:
        env.close()


def _collect_robot_probe(
    output: Path,
    *,
    run_root: Path,
    device: str,
    scenario: str,
    control_ticks: int,
    repetitions: int,
    amplitude: float | None,
    input_vector: list[float] | None,
    reset_cache_path: Path | None,
    repeatability_family: str | None,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from isaaclab.sim import SimulationContext

    from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG
    from wheelleg_dreamwaq.schemas.frames import transform_usd_vector_to_control

    from .isaac_probe_env import RootCauseIsaacProbeEnv, make_probe_env_cfg
    from .variant_builders import IsaacVariantSpec, ROBOT_HINGES

    if SimulationContext.instance() is not None:
        raise RuntimeError("Isaac probe requires no pre-existing SimulationContext")
    if scenario not in ISAAC_ROBOT_VARIANTS:
        raise ValueError(f"Unknown Isaac robot scenario: {scenario}")
    if control_ticks <= 0:
        raise ValueError("Robot probe control tick count must be positive")
    profiles = _robot_profiles(
        scenario,
        repetitions=repetitions,
        amplitude=amplitude,
        input_vector=input_vector,
    )
    serial_profiles = _serial_robot_profiles(profiles, repetitions=repetitions)
    reset_artifact = None
    if reset_cache_path is not None:
        from wheelleg_dreamwaq.schemas.randomization import (
            validate_closed_chain_reset_cache_artifact,
        )

        reset_artifact = validate_closed_chain_reset_cache_artifact(
            torch.load(reset_cache_path.resolve(strict=True), map_location="cpu", weights_only=False)
        )
        if not scenario.startswith("p60"):
            raise ValueError("Frozen evaluation reset cache is only valid for P60 probes")
    spec = IsaacVariantSpec(**ISAAC_ROBOT_VARIANTS[scenario])
    direct_effort = scenario in DIRECT_EFFORT_SCENARIOS
    formal_config_hash_before = stable_hash(WHEELLEG_CFG.to_dict())
    cfg = make_probe_env_cfg(
        spec,
        device=device,
        num_envs=len(serial_profiles),
        enable_contact_sensors=False,
        episode_length_s=max(10.02, (control_ticks + 1) * 0.02),
    )
    cfg.sim.save_logs_to_file = True
    log_directory = (run_root / "runtime_cache" / "isaaclab" / "logs").resolve()
    log_directory.mkdir(parents=True, exist_ok=True)
    cfg.sim.log_dir = str(log_directory)
    model_manifest = json.loads(
        (Path(__file__).resolve().parents[3] / "sim2sim" / "mujoco" / "model_manifest.json")
        .read_text(encoding="utf-8")
    )
    env = RootCauseIsaacProbeEnv(
        cfg,
        variant=spec,
        direct_effort=direct_effort,
        reset_artifact=reset_artifact,
        replicate_reset_row_zero=(
            reset_artifact is not None and len(serial_profiles) != 1
        ),
        enable_contact_sensors=False,
    )
    try:
        env.configure_debug_joint_order(list(model_manifest["joint_order"]))
        variant_identity = env.validate_variant_identity()
        reset_rng_state = env.get_randomization_rng_state()
        profile_channel = np.asarray(
            [item["channel"] for item in serial_profiles], dtype=np.int16
        )
        profile_sign = np.asarray(
            [item["sign"] for item in serial_profiles], dtype=np.int8
        )
        input_values = np.asarray(
            [item["actual_input"] for item in serial_profiles], dtype=np.float32
        )
        torque_equivalent = np.asarray(
            [item["canonical_torque_equivalent"] for item in serial_profiles],
            dtype=np.float32,
        )

        def env_array(value: Any) -> np.ndarray:
            array = np.asarray(_to_numpy(value, remove_env=False))
            if (
                array.ndim >= 2
                and array.shape[0] == 1
                and array.shape[1] == len(serial_profiles)
            ):
                array = array[0]
            return array

        def row_from_state(
            *,
            time_s: float,
            state: dict[str, Any],
            system: dict[str, Any],
            commanded: np.ndarray,
            host_torque: np.ndarray,
            effort_event: np.ndarray,
            velocity_event: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray,
            repetition: int,
        ) -> dict[str, Any]:
            origins = env.scene.env_origins
            system_position = transform_usd_vector_to_control(
                system["system_com_position_world"] - origins
            )
            return {
                "time_s": np.full(len(serial_profiles), time_s, dtype=np.float64),
                "profile_channel": profile_channel,
                "profile_sign": profile_sign,
                "profile_repetition": np.full(
                    len(serial_profiles), repetition, dtype=np.int16
                ),
                "controlled_position_canonical": _to_numpy(
                    state["active_joint_position_canonical"], remove_env=False
                ),
                "controlled_velocity_canonical": _to_numpy(
                    state["active_joint_velocity_canonical"], remove_env=False
                ),
                "all_hinge_position": _to_numpy(
                    state["all_hinge_position_named"], remove_env=False
                ),
                "all_hinge_velocity": _to_numpy(
                    state["all_hinge_velocity_named"], remove_env=False
                ),
                "base_com_position_diag": _to_numpy(
                    state["base_com_position_diag"], remove_env=False
                ),
                "base_linear_velocity_control": _to_numpy(
                    state["base_linear_velocity_control"], remove_env=False
                ),
                "base_angular_velocity_control": _to_numpy(
                    state["base_angular_velocity_control"], remove_env=False
                ),
                "system_com_position_control": _to_numpy(
                    system_position, remove_env=False
                ),
                "linear_momentum_control": _to_numpy(
                    transform_usd_vector_to_control(system["linear_momentum_world"]),
                    remove_env=False,
                ),
                "angular_momentum_com_control": _to_numpy(
                    transform_usd_vector_to_control(system["angular_momentum_com_world"]),
                    remove_env=False,
                ),
                "kinetic_energy_j": _to_numpy(
                    system["kinetic_energy_j"], remove_env=False
                ),
                "closure_residual_m": np.max(
                    _to_numpy(state["loop_closure_error"], remove_env=False), axis=-1
                ),
                "commanded_input_canonical": commanded,
                "canonical_torque_equivalent": torque_equivalent if time_s <= 0.020000001 else np.zeros_like(torque_equivalent),
                "host_applied_torque_canonical": host_torque,
                "effort_limit_event": effort_event,
                "joint_velocity_limit_exceeded": velocity_event,
                "terminated": terminated,
                "truncated": truncated,
            }

        maximum_direct_effort_error = 0.0
        repetition_runs: list[dict[str, Any]] = []
        repetition_arrays: list[dict[str, Any]] = []
        for repetition in range(repetitions):
            env.set_randomization_rng_state(reset_rng_state)
            rng_before_reset = _randomization_rng_identity(env)
            observations, _ = env.reset(seed=0)
            rng_after_reset = _randomization_rng_identity(env)
            reset = env._debug_reset_snapshot
            if reset is None:
                raise RuntimeError("Isaac robot probe did not expose reset phases")
            returned_observation = _capture_authoritative_identity_array(
                "returned_observation",
                reset["reset_returned_actor_obs_policy"],
            )
            if returned_observation.shape != (len(serial_profiles), 25):
                raise RuntimeError(
                    "Robot reset returned observation has no exact [profile,25] layout"
                )
            commands = _capture_authoritative_identity_array(
                "commands", env._commands
            )
            previous_action = _capture_authoritative_identity_array(
                "previous_action", env._previous_action
            )
            physics_time_origin_s = float(env._sim_step_counter * env.physics_dt)
            origins = env.scene.env_origins
            pre_state = reset["reset_written_pre_forward"]
            post_reset_state = reset["reset_forwarded_post_forward"]
            pre_root_position_control = _capture_authoritative_identity_array(
                "pre_root_position_control",
                transform_usd_vector_to_control(
                    pre_state["base_com_position_engine_world"] - origins
                ),
            )
            post_root_position_control = _capture_authoritative_identity_array(
                "post_root_position_control",
                transform_usd_vector_to_control(
                    post_reset_state["base_com_position_engine_world"] - origins
                ),
            )
            rows = [
                row_from_state(
                    time_s=0.0,
                    state=post_reset_state,
                    system=env.capture_system_state(),
                    commanded=np.zeros_like(input_values),
                    host_torque=np.zeros_like(input_values),
                    effort_event=np.zeros_like(input_values, dtype=np.int8),
                    velocity_event=np.zeros_like(input_values, dtype=np.int8),
                    terminated=np.zeros(len(serial_profiles), dtype=np.int8),
                    truncated=np.zeros(len(serial_profiles), dtype=np.int8),
                    repetition=repetition,
                )
            ]
            for tick in range(control_ticks):
                selected = input_values if tick == 0 else np.zeros_like(input_values)
                action = torch.as_tensor(
                    selected, dtype=torch.float32, device=env.device
                )
                _, _, terminated, truncated, _ = env.step(action)
                if len(env._debug_substeps) != env.cfg.decimation:
                    raise RuntimeError(
                        "Isaac robot probe returned the wrong substep count"
                    )
                if len(env._root_cause_substep_system) != env.cfg.decimation:
                    raise RuntimeError(
                        "Isaac robot probe returned the wrong system-state count"
                    )
                for substep_index, (source, system) in enumerate(
                    zip(
                        env._debug_substeps,
                        env._root_cause_substep_system,
                        strict=True,
                    )
                ):
                    post_state = {
                        name: source[f"{name}_post_step"]
                        for name in (
                            "active_joint_position_canonical",
                            "active_joint_velocity_canonical",
                            "all_hinge_position_named",
                            "all_hinge_velocity_named",
                            "base_com_position_diag",
                            "base_linear_velocity_control",
                            "base_angular_velocity_control",
                        )
                    }
                    post_state["loop_closure_error"] = source[
                        "loop_closure_error_post_step"
                    ]
                    host = _to_numpy(
                        source["isaac_host_pd_torque_estimate_canonical"],
                        remove_env=False,
                    )
                    if direct_effort and tick == 0:
                        maximum_direct_effort_error = max(
                            maximum_direct_effort_error,
                            float(np.max(np.abs(host - selected))),
                        )
                    is_last = substep_index == env.cfg.decimation - 1
                    expected_physics_time_s = _normalized_serial_physics_time_s(
                        absolute_time_s=float(source["physics_time_s"]),
                        origin_s=physics_time_origin_s,
                        tick=tick,
                        substep_index=substep_index,
                        decimation=int(env.cfg.decimation),
                        physics_dt_s=float(env.physics_dt),
                    )
                    rows.append(
                        row_from_state(
                            time_s=expected_physics_time_s,
                            state=post_state,
                            system=system,
                            commanded=selected.copy(),
                            host_torque=host,
                            effort_event=env_array(
                                source["effort_limit_event"]
                            ).astype(np.int8),
                            velocity_event=env_array(
                                source["joint_velocity_limit_exceeded"]
                            ).astype(np.int8),
                            terminated=(
                                _to_numpy(terminated, remove_env=False).astype(
                                    np.int8
                                )
                                if is_last
                                else np.zeros(
                                    len(serial_profiles), dtype=np.int8
                                )
                            ),
                            truncated=(
                                _to_numpy(truncated, remove_env=False).astype(
                                    np.int8
                                )
                                if is_last
                                else np.zeros(
                                    len(serial_profiles), dtype=np.int8
                                )
                            ),
                            repetition=repetition,
                        )
                    )
            repetition_arrays.append(_rows_to_arrays(rows))
            repetition_runs.append(
                {
                    "reset": reset,
                    "returned_observation": returned_observation.copy(),
                    "previous_action": previous_action,
                    "commands": commands,
                    "pre_root_position_control": pre_root_position_control,
                    "post_root_position_control": post_root_position_control,
                    "rng_before_reset": rng_before_reset,
                    "rng_after_reset": rng_after_reset,
                }
            )
        if direct_effort and maximum_direct_effort_error > 1.0e-6:
            raise RuntimeError(
                "Isaac direct-effort host buffer differs from the declared effort: "
                f"{maximum_direct_effort_error}"
            )
        arrays = _merge_serial_repetition_arrays(repetition_arrays)
        trace = write_trace(output / "trace", arrays, _robot_probe_fields(arrays))
        excitation_semantics = {
            "input_kind": "direct_effort" if direct_effort else "formal_target_action",
            "input_units": "N*m" if direct_effort else "normalized_action",
            "actual_input_values": input_values.tolist(),
            "canonical_torque_equivalent": torque_equivalent.tolist(),
            "start_s": 0.0,
            "stop_s": 0.02,
            "physics_sample_dt_s": float(env.physics_dt),
            "control_dt_s": float(env.step_dt),
        }
        reset_artifact_sha = (
            None if reset_cache_path is None else sha256_file(reset_cache_path)
        )

        def common_reset_identity(run: dict[str, Any]) -> dict[str, Any]:
            return {
                "seed": 0,
                "rng_state_before_reset_hash": run["rng_before_reset"][
                    "identity_hash"
                ],
                "rng_state_after_reset_hash": run["rng_after_reset"][
                    "identity_hash"
                ],
                "reset_artifact_sha256": reset_artifact_sha,
                "reset_artifact_environment_row": (
                    None if reset_cache_path is None else 0
                ),
            }

        def phase_state(
            run: dict[str, Any], profile_index: int, phase: str
        ) -> dict[str, Any]:
            reset_snapshot = run["reset"]
            if phase == "pre_forward":
                return _repeatability_profile_state(
                    reset_snapshot["reset_written_pre_forward"],
                    profile_index,
                    phase=phase,
                    root_com_position_control=run["pre_root_position_control"],
                    command=run["commands"][profile_index],
                )
            return _repeatability_profile_state(
                reset_snapshot["reset_forwarded_post_forward"],
                profile_index,
                phase=phase,
                root_com_position_control=run["post_root_position_control"],
            )

        first_run = repetition_runs[0]
        first_common_reset = common_reset_identity(first_run)
        pre_forward_hash = stable_hash(
            _reset_phase_identity_payload(
                phase="pre_forward",
                state_name="states",
                state=[
                    phase_state(first_run, index, "pre_forward")
                    for index in range(len(serial_profiles))
                ],
                reset_identity=first_common_reset,
            )
        )
        post_forward_hash = stable_hash(
            _reset_phase_identity_payload(
                phase="post_forward",
                state_name="states",
                state=[
                    phase_state(first_run, index, "post_forward")
                    for index in range(len(serial_profiles))
                ],
            )
        )
        reset_returned_hash = stable_hash(
            {
                "profiles": [
                    _repeatability_returned_policy_state(
                        first_run["returned_observation"][index],
                        first_run["previous_action"][index],
                    )
                    for index in range(len(serial_profiles))
                ]
            }
        )
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
        source_model_sha256 = str(variant_identity.variant["asset_bundle_hash"])
        transform_payload = {
            "schema_version": "RootCauseIsaacRobotVariantV1",
            "scenario": scenario,
            "variant": variant_identity.variant,
            "closure_joints": list(variant_identity.closure_joints),
            "drive": variant_identity.drive,
        }
        causal_identity = result_identity_fields(
            scenario_id=robot_scenario_id(scenario),
            engine="isaac",
            source_model_sha256=source_model_sha256,
            model_artifact_sha256=stable_hash(transform_payload),
            transform_manifest_sha256=stable_hash(
                {
                    "schema_version": transform_payload["schema_version"],
                    "scenario": scenario,
                    "variant": variant_identity.variant,
                }
            ),
            transform_semantics=_isaac_robot_transform_semantics(
                scenario=scenario, identity=variant_identity
            ),
            worker_source_sha256=sha256_file(Path(__file__)),
            configuration_semantics=robot_configuration_semantics(
                scenario=scenario,
                engine="isaac",
                physics_dt_s=float(env.physics_dt),
                control_dt_s=float(env.step_dt),
                source_identity=source_model_sha256,
                resolved_semantics=_isaac_robot_resolved_semantics(
                    env, scenario=scenario, identity=variant_identity
                ),
            ),
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
            raise RuntimeError(
                f"Robot scenario has no repeatability family: {scenario_id}"
            )
        repeatability_records = []
        for profile_index, profile in enumerate(profiles):
            repetition = int(profile["repetition"])
            serial_profile_index = profile_index // repetitions
            run = repetition_runs[repetition]
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
                "physics_sample_dt_s": excitation_semantics[
                    "physics_sample_dt_s"
                ],
                "control_dt_s": excitation_semantics["control_dt_s"],
            }
            common_reset = common_reset_identity(run)
            repeatability_records.append(
                build_repeatability_record(
                    engine="isaac",
                    scenario_id=scenario_id,
                    repeatability_family=family,
                    configuration_hash=str(causal_identity["configuration_hash"]),
                    profile_index=profile_index,
                    repetition=repetition,
                    profile_semantics=profile_semantics,
                    pre_forward_payload=_reset_phase_identity_payload(
                        phase="pre_forward",
                        state_name="state",
                        state=phase_state(
                            run, serial_profile_index, "pre_forward"
                        ),
                        reset_identity=common_reset,
                    ),
                    post_forward_payload=_reset_phase_identity_payload(
                        phase="post_forward",
                        state_name="state",
                        state=phase_state(
                            run, serial_profile_index, "post_forward"
                        ),
                    ),
                    reset_returned_policy_payload=_repeatability_returned_policy_state(
                        run["returned_observation"][serial_profile_index],
                        run["previous_action"][serial_profile_index],
                    ),
                    excitation_semantics=excitation,
                )
            )
        payload = {
            "schema_version": "RootCauseIsaacRobotProbeV1",
            "scenario": scenario,
            "variant": variant_identity.variant,
            "closure_joints": list(variant_identity.closure_joints),
            "drive": variant_identity.drive,
            "compiled_semantics_hash": stable_hash(variant_identity.compiled),
            "profile_count": len(profiles),
            "serial_environment_count": len(serial_profiles),
            "serial_fresh_reset_repetitions": repetitions,
            "profiles": profiles,
            "control_ticks": control_ticks,
            "physics_dt_s": float(env.physics_dt),
            "decimation": int(env.cfg.decimation),
            "control_dt_s": float(env.step_dt),
            "excitation_semantics": excitation_semantics,
            "repeatability_records": repeatability_records,
            **causal_identity,
            "maximum_direct_effort_host_error_nm": maximum_direct_effort_error,
            "returned_policy_observation_hash": stable_hash(
                first_run["returned_observation"].tolist()
            ),
            "reset_cache_path": (
                None if reset_cache_path is None else str(reset_cache_path.resolve(strict=True))
            ),
            "reset_cache_sha256": (
                None if reset_cache_path is None else sha256_file(reset_cache_path)
            ),
            "reset_cache_environment_rows": (
                None
                if reset_cache_path is None
                else [0] * len(profiles)
            ),
            "trace_sha256": trace.trace_sha256,
            "trace_metadata_sha256": trace.metadata_sha256,
            "formal_config_hash_before": formal_config_hash_before,
            "formal_config_hash_after": stable_hash(WHEELLEG_CFG.to_dict()),
        }
        if payload["formal_config_hash_before"] != payload["formal_config_hash_after"]:
            raise RuntimeError("Formal WHEELLEG_CFG changed during the Isaac robot probe")
        (output / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
        return payload
    finally:
        env.close()


def _collect_one_tick(output: Path, *, run_root: Path, device: str) -> dict[str, Any]:
    import torch

    from isaaclab.sim import SimulationContext

    from debug.sim2sim.isaac_debug_env import WheelLegSim2SimDebugEnv, make_debug_env_cfg
    from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG
    from wheelleg_dreamwaq.schemas.randomization import NOMINAL_EVALUATION_PROFILE_V1

    if SimulationContext.instance() is not None:
        raise RuntimeError("Isaac probe requires no pre-existing SimulationContext")
    formal_config_hash_before = stable_hash(WHEELLEG_CFG.to_dict())
    cfg = make_debug_env_cfg(device=device, enable_contact_sensors=False)
    cfg.randomization = NOMINAL_EVALUATION_PROFILE_V1
    cfg.sim.save_logs_to_file = True
    log_directory = (run_root / "runtime_cache" / "isaaclab" / "logs").resolve()
    log_directory.mkdir(parents=True, exist_ok=True)
    cfg.sim.log_dir = str(log_directory)
    model_manifest = json.loads(
        (Path(__file__).resolve().parents[3] / "sim2sim" / "mujoco" / "model_manifest.json")
        .read_text(encoding="utf-8")
    )
    env = WheelLegSim2SimDebugEnv(cfg, enable_contact_sensors=False)
    try:
        env.configure_debug_joint_order(list(model_manifest["joint_order"]))
        handlers = validate_isaaclab_handlers(
            snapshot_file_handlers(), expected_directory=log_directory
        )
        observations, _ = env.reset(seed=0)
        reset = env._debug_reset_snapshot
        if reset is None:
            raise RuntimeError("Isaac debug environment did not expose reset phases")
        action = torch.zeros((1, 6), dtype=torch.float32, device=env.device)
        env.step(action)
        rows: list[dict[str, Any]] = []
        initial = reset["reset_forwarded_post_forward"]
        rows.append(
            {
                "time_s": 0.0,
                "active_joint_position_canonical": _to_numpy(
                    initial["active_joint_position_canonical"]
                ),
                "active_joint_velocity_canonical": _to_numpy(
                    initial["active_joint_velocity_canonical"]
                ),
                "all_hinge_position_named": _to_numpy(initial["all_hinge_position_named"]),
                "all_hinge_velocity_named": _to_numpy(initial["all_hinge_velocity_named"]),
                "base_com_position_engine_world": _to_numpy(
                    initial["base_com_position_engine_world"]
                ),
                "base_linear_velocity_control": _to_numpy(
                    initial["base_linear_velocity_control"]
                ),
                "base_angular_velocity_control": _to_numpy(
                    initial["base_angular_velocity_control"]
                ),
                "loop_closure_error": _to_numpy(initial["loop_closure_error"]),
            }
        )
        for source in env._debug_substeps:
            rows.append(
                {
                    "time_s": float(source["physics_time_s"]),
                    "active_joint_position_canonical": _to_numpy(
                        source["active_joint_position_canonical_post_step"]
                    ),
                    "active_joint_velocity_canonical": _to_numpy(
                        source["active_joint_velocity_canonical_post_step"]
                    ),
                    "all_hinge_position_named": _to_numpy(
                        source["all_hinge_position_named_post_step"]
                    ),
                    "all_hinge_velocity_named": _to_numpy(
                        source["all_hinge_velocity_named_post_step"]
                    ),
                    "base_com_position_engine_world": _to_numpy(
                        source["base_com_position_engine_world_post_step"]
                    ),
                    "base_linear_velocity_control": _to_numpy(
                        source["base_linear_velocity_control_post_step"]
                    ),
                    "base_angular_velocity_control": _to_numpy(
                        source["base_angular_velocity_control_post_step"]
                    ),
                    "loop_closure_error": _to_numpy(source["loop_closure_error_post_step"]),
                }
            )
        arrays = _rows_to_arrays(rows)
        trace = write_trace(output / "one_tick_trace", arrays, _one_tick_fields(arrays))
        payload = {
            "schema_version": "RootCauseIsaacOneTickV1",
            "physics_dt_s": float(env.physics_dt),
            "decimation": int(env.cfg.decimation),
            "control_dt_s": float(env.step_dt),
            "logger_handlers": handlers,
            "formal_config_hash_before": formal_config_hash_before,
            "formal_config_hash_after": stable_hash(WHEELLEG_CFG.to_dict()),
            "reset_pre_forward_hash": stable_hash(
                _phase_hash_payload(reset["reset_written_pre_forward"])
            ),
            "reset_post_forward_hash": stable_hash(
                _phase_hash_payload(reset["reset_forwarded_post_forward"])
            ),
            "reset_returned_policy_hash": stable_hash(
                _to_numpy(reset["reset_returned_actor_obs_policy"]).tolist()
            ),
            "returned_policy_observation": _to_numpy(observations["policy"]).tolist(),
            "trace_sha256": trace.trace_sha256,
            "trace_metadata_sha256": trace.metadata_sha256,
        }
        if payload["formal_config_hash_before"] != payload["formal_config_hash_after"]:
            raise RuntimeError("Formal WHEELLEG_CFG changed during the Isaac probe")
        (output / "one_tick.json").write_bytes(canonical_json_bytes(payload) + b"\n")
        return payload
    finally:
        env.close()


def _collect_adapter_probe(
    output: Path,
    *,
    run_root: Path,
    device: str,
    pulse_amplitude: float,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from isaaclab.sim import SimulationContext

    from debug.sim2sim.dreamwaq_debug_contract import DebugPolicyAdapter
    from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG
    from wheelleg_dreamwaq.schemas.randomization import (
        validate_closed_chain_reset_cache_artifact,
    )

    from .adapter_gate import G02_THRESHOLDS
    from .isaac_probe_env import RootCauseIsaacProbeEnv, make_probe_env_cfg
    from .variant_builders import IsaacVariantSpec

    if pulse_amplitude <= 0.0 or pulse_amplitude >= 0.1:
        raise ValueError("Adapter pulse amplitude must be in (0,0.1)")
    if SimulationContext.instance() is not None:
        raise RuntimeError("Adapter probe requires no pre-existing SimulationContext")

    formal_config_hash_before = stable_hash(WHEELLEG_CFG.to_dict())
    spec = IsaacVariantSpec(**ISAAC_ROBOT_VARIANTS["p60_c"])
    cfg = make_probe_env_cfg(
        spec,
        device=device,
        num_envs=14,
        enable_contact_sensors=False,
        episode_length_s=10.02,
    )
    cfg.sim.save_logs_to_file = True
    log_directory = (run_root / "runtime_cache" / "isaaclab" / "logs").resolve()
    log_directory.mkdir(parents=True, exist_ok=True)
    cfg.sim.log_dir = str(log_directory)

    model_manifest_path = (
        Path(__file__).resolve().parents[3] / "sim2sim" / "mujoco" / "model_manifest.json"
    )
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    policy = FROZEN_POLICIES["dreamwaq_run01"]
    adapter = DebugPolicyAdapter(
        actor_path=policy["actor"].absolute_path(),
        manifest_path=policy["manifest"].absolute_path(),
        model_manifest_path=model_manifest_path,
        load_mujoco_contract=False,
    )
    reset_cache_path = FROZEN_REPLAY_SOURCE["reset_cache"].absolute_path()
    if sha256_file(reset_cache_path) != FROZEN_REPLAY_SOURCE["reset_cache"].sha256:
        raise RuntimeError("Frozen adapter reset cache hash drifted")
    reset_artifact = validate_closed_chain_reset_cache_artifact(
        torch.load(reset_cache_path, map_location="cpu", weights_only=False)
    )

    def reset_state_payload(
        state: dict[str, Any], *, environment_index: int, post_forward: bool
    ) -> dict[str, Any]:
        def row(name: str) -> np.ndarray:
            values = np.asarray(
                _to_numpy(state[name], remove_env=False), dtype=np.float64
            )
            if values.shape[0] != cfg.scene.num_envs:
                raise RuntimeError(
                    f"Isaac adapter state {name} has no environment axis"
                )
            return np.asarray(values[environment_index]).copy()

        payload = {
            "controlled_position_canonical": row(
                "active_joint_position_canonical"
            ).tolist(),
            "controlled_velocity_canonical": row(
                "active_joint_velocity_canonical"
            ).tolist(),
            "all_hinge_position_named": row("all_hinge_position_named").tolist(),
            "all_hinge_velocity_named": row("all_hinge_velocity_named").tolist(),
        }
        if post_forward:
            closure = row("loop_closure_error").reshape(-1)
            payload.update(
                {
                    "base_orientation_control_wxyz": row(
                        "base_orientation_control_wxyz"
                    ).tolist(),
                    "base_linear_velocity_control": row(
                        "base_linear_velocity_control"
                    ).tolist(),
                    "base_angular_velocity_control": row(
                        "base_angular_velocity_control"
                    ).tolist(),
                    "projected_gravity": row("projected_gravity").tolist(),
                    "base_height": float(row("base_height").reshape(-1)[0]),
                    "loop_closure_error": float(
                        np.max(np.abs(closure), initial=0.0)
                    ),
                    "loop_closure_error_per_side": closure.tolist(),
                }
            )
        return payload

    def reset_identity(
        snapshot: dict[str, Any], current: np.ndarray, *, environment_index: int
    ) -> str:
        return stable_hash(
            {
                "pre_forward": reset_state_payload(
                    snapshot["reset_written_pre_forward"],
                    environment_index=environment_index,
                    post_forward=False,
                ),
                "post_forward": reset_state_payload(
                    snapshot["reset_forwarded_post_forward"],
                    environment_index=environment_index,
                    post_forward=True,
                ),
                "returned_actor_obs_current": current.tolist(),
            }
        )

    env = RootCauseIsaacProbeEnv(
        cfg,
        variant=spec,
        direct_effort=False,
        reset_artifact=reset_artifact,
        replicate_reset_row_zero=True,
        enable_contact_sensors=False,
    )
    try:
        env.configure_debug_joint_order(list(model_manifest["joint_order"]))
        variant_identity = env.validate_variant_identity()
        handlers = validate_isaaclab_handlers(
            snapshot_file_handlers(), expected_directory=log_directory
        )
        observations, _ = env.reset(seed=0)
        reset = env._debug_reset_snapshot
        if reset is None:
            raise RuntimeError("Isaac adapter probe did not expose reset phases")
        current_matrix = np.asarray(
            _to_numpy(observations["policy"], remove_env=False), dtype=np.float32
        )
        if current_matrix.shape != (14, 25):
            raise RuntimeError(
                "Isaac adapter probe ActorObsV1 matrix is not [14,25]"
            )
        current = current_matrix[0].copy()
        reset_observation_replica_error = float(
            np.max(np.abs(current_matrix - current[None, :]), initial=0.0)
        )
        if reset_observation_replica_error > G02_THRESHOLDS[
            "actor_observation_max_abs"
        ]:
            raise RuntimeError(
                "Isaac parallel adapter profiles did not receive one reset observation: "
                f"value={reset_observation_replica_error}"
            )
        policy_input = adapter.initialize_policy_input(current)
        if policy_input.shape != (125,) or not np.array_equal(
            policy_input.reshape(5, 25), np.repeat(current[None, :], 5, axis=0)
        ):
            raise RuntimeError("Isaac DreamWaQ history initialization is invalid")
        baseline_reset_identity = reset_identity(
            reset, current, environment_index=0
        )
        baseline_reset_payload = {
            "pre_forward": reset_state_payload(
                reset["reset_written_pre_forward"],
                environment_index=0,
                post_forward=False,
            ),
            "post_forward": reset_state_payload(
                reset["reset_forwarded_post_forward"],
                environment_index=0,
                post_forward=True,
            ),
            "returned_actor_obs_current": current.tolist(),
        }

        def reset_match_metrics(environment_index: int) -> dict[str, float]:
            candidate = {
                "pre_forward": reset_state_payload(
                    reset["reset_written_pre_forward"],
                    environment_index=environment_index,
                    post_forward=False,
                ),
                "post_forward": reset_state_payload(
                    reset["reset_forwarded_post_forward"],
                    environment_index=environment_index,
                    post_forward=True,
                ),
            }
            metrics: dict[str, float] = {}
            for phase, threshold_position, threshold_velocity in (
                (
                    "pre_forward",
                    G02_THRESHOLDS["reset_pre_position_max_abs"],
                    G02_THRESHOLDS["reset_pre_velocity_max_abs"],
                ),
                (
                    "post_forward",
                    G02_THRESHOLDS["reset_post_position_max_abs"],
                    G02_THRESHOLDS["reset_post_velocity_max_abs"],
                ),
            ):
                for field, threshold in (
                    ("controlled_position_canonical", threshold_position),
                    ("controlled_velocity_canonical", threshold_velocity),
                    ("all_hinge_position_named", threshold_position),
                    ("all_hinge_velocity_named", threshold_velocity),
                ):
                    value = float(
                        np.max(
                            np.abs(
                                np.asarray(candidate[phase][field], dtype=np.float64)
                                - np.asarray(
                                    baseline_reset_payload[phase][field],
                                    dtype=np.float64,
                                )
                            ),
                            initial=0.0,
                        )
                    )
                    metrics[f"{phase}.{field}"] = value
                    if value > threshold:
                        raise RuntimeError(
                            "Isaac parallel adapter reset state differs across profiles: "
                            f"environment={environment_index}, field={phase}.{field}, "
                            f"value={value}, threshold={threshold}"
                        )
            observation_error = float(
                np.max(
                    np.abs(current_matrix[environment_index] - current),
                    initial=0.0,
                )
            )
            metrics["returned_actor_obs_current"] = observation_error
            if observation_error > G02_THRESHOLDS["actor_observation_max_abs"]:
                raise RuntimeError(
                    "Isaac parallel adapter observation differs across profiles: "
                    f"environment={environment_index}, value={observation_error}"
                )
            return metrics

        all_reset_match_metrics = [reset_match_metrics(index) for index in range(14)]
        inference = adapter.infer(policy_input)
        raw_action = np.asarray(inference.raw_action, dtype=np.float64)
        actions = np.zeros((14, 6), dtype=np.float64)
        actions[0] = raw_action
        profile_layout = [
            {"environment_index": 0, "role": "actor"},
            {"environment_index": 1, "role": "zero"},
        ]
        for channel in range(6):
            plus_index = 2 + 2 * channel
            minus_index = plus_index + 1
            actions[plus_index, channel] = pulse_amplitude
            actions[minus_index, channel] = -pulse_amplitude
            profile_layout.extend(
                (
                    {
                        "environment_index": plus_index,
                        "role": "pulse_plus",
                        "channel": channel,
                    },
                    {
                        "environment_index": minus_index,
                        "role": "pulse_minus",
                        "channel": channel,
                    },
                )
            )
        reset_post_state = reset["reset_forwarded_post_forward"]
        velocity_before_matrix = np.asarray(
            _to_numpy(
                reset_post_state["active_joint_velocity_canonical"],
                remove_env=False,
            ),
            dtype=np.float64,
        )
        time_before = float(env._sim_step_counter * env.physics_dt)
        next_observations, _, terminated, truncated, _ = env.step(
            torch.as_tensor(actions, dtype=torch.float32, device=env.device)
        )
        if bool(torch.any(terminated).item()) or bool(torch.any(truncated).item()):
            raise RuntimeError("Isaac adapter parallel step terminated unexpectedly")
        if len(env._debug_substeps) != int(env.cfg.decimation):
            raise RuntimeError("Isaac adapter probe captured the wrong substep count")
        terminal = env._debug_terminal_state
        if terminal is None:
            raise RuntimeError("Isaac adapter probe terminal state is unavailable")
        first_substep = env._debug_substeps[0]
        canonical_target_matrix = np.asarray(
            _to_numpy(
                first_substep["target_command_canonical"], remove_env=False
            ),
            dtype=np.float64,
        )
        native_target_matrix = np.asarray(
            _to_numpy(
                first_substep["target_command_engine_native"], remove_env=False
            ),
            dtype=np.float64,
        )
        clipped_action_matrix = np.asarray(
            _to_numpy(env._debug_action_clipped, remove_env=False), dtype=np.float64
        )
        next_current_matrix = np.asarray(
            _to_numpy(next_observations["policy"], remove_env=False),
            dtype=np.float32,
        )
        velocity_after_matrix = np.asarray(
            _to_numpy(
                terminal["active_joint_velocity_canonical"], remove_env=False
            ),
            dtype=np.float64,
        )
        expected_shapes = {
            "canonical target": (14, 6),
            "native target": (14, 6),
            "clipped action": (14, 6),
            "next observation": (14, 25),
            "velocity before": (14, 6),
            "velocity after": (14, 6),
        }
        actual_shapes = {
            "canonical target": canonical_target_matrix.shape,
            "native target": native_target_matrix.shape,
            "clipped action": clipped_action_matrix.shape,
            "next observation": next_current_matrix.shape,
            "velocity before": velocity_before_matrix.shape,
            "velocity after": velocity_after_matrix.shape,
        }
        if actual_shapes != expected_shapes:
            raise RuntimeError(
                f"Isaac adapter parallel evidence shape mismatch: {actual_shapes}"
            )
        time_after = float(env._sim_step_counter * env.physics_dt)
        targets_constant = all(
            np.array_equal(
                np.asarray(
                    _to_numpy(
                        row["target_command_canonical"], remove_env=False
                    )
                ),
                canonical_target_matrix,
            )
            and np.array_equal(
                np.asarray(
                    _to_numpy(
                        row["target_command_engine_native"], remove_env=False
                    )
                ),
                native_target_matrix,
            )
            for row in env._debug_substeps
        )
        canonical_target = canonical_target_matrix[0].copy()
        native_target = native_target_matrix[0].copy()
        clipped_action = clipped_action_matrix[0].copy()
        next_current = next_current_matrix[0].copy()

        def profile_record(environment_index: int) -> dict[str, Any]:
            profile_current = current_matrix[environment_index]
            return {
                "policy_input_sha256": stable_hash(
                    adapter.initialize_policy_input(profile_current).tolist()
                ),
                "reset_identity_hash": reset_identity(
                    reset,
                    profile_current,
                    environment_index=environment_index,
                ),
                "baseline_reset_identity_hash": baseline_reset_identity,
                "reset_within_frozen_tolerance_of_baseline": True,
                "reset_match_metrics": all_reset_match_metrics[environment_index],
                "action": actions[environment_index].tolist(),
                "clipped_action": clipped_action_matrix[environment_index].tolist(),
                "target_canonical": canonical_target_matrix[environment_index].tolist(),
                "target_engine_native": native_target_matrix[environment_index].tolist(),
                "velocity_before": velocity_before_matrix[environment_index].tolist(),
                "velocity_after": velocity_after_matrix[environment_index].tolist(),
                "velocity_delta": (
                    velocity_after_matrix[environment_index]
                    - velocity_before_matrix[environment_index]
                ).tolist(),
                "previous_action_before": profile_current[19:25].tolist(),
                "previous_action_after": next_current_matrix[
                    environment_index, 19:25
                ].tolist(),
                "control_time_delta_s": time_after - time_before,
            }

        zero_pulse = profile_record(1)
        pulse_records: dict[str, Any] = {}
        for channel in range(6):
            plus_record = profile_record(2 + 2 * channel)
            minus_record = profile_record(3 + 2 * channel)
            plus_delta = np.asarray(plus_record["velocity_delta"], dtype=np.float64)
            minus_delta = np.asarray(minus_record["velocity_delta"], dtype=np.float64)
            odd_velocity = 0.5 * (plus_delta - minus_delta)
            odd_target = 0.5 * (
                np.asarray(plus_record["target_canonical"], dtype=np.float64)
                - np.asarray(minus_record["target_canonical"], dtype=np.float64)
            )
            pulse_records[str(channel)] = {
                "plus": plus_record,
                "minus": minus_record,
                "zero": zero_pulse,
                "odd_velocity_response": odd_velocity.tolist(),
                "odd_target_response": odd_target.tolist(),
                "driven_channel_velocity_sign": int(np.sign(odd_velocity[channel])),
                "driven_channel_target_sign": int(np.sign(odd_target[channel])),
            }

        manifest = adapter.manifest
        payload = {
            "schema_version": "RootCauseIsaacAdapterProbeV1",
            "engine": "isaac",
            "policy": "dreamwaq_run01",
            "command": np.asarray(
                _to_numpy(env._commands, remove_env=False), dtype=np.float64
            )[0].tolist(),
            "profile_layout": profile_layout,
            "contract": {
                "actor_slices": manifest["observation"]["actor_slices"],
                "canonical_joint_order": manifest["action"]["canonical_joint_order"],
                "history_layout": manifest["network"]["history_layout"],
                "history_length": manifest["network"]["history_length"],
                "input_dimension": manifest["network"]["input_dimension"],
                "runtime_action_clip": manifest["network"]["runtime_action_clip"],
                "control_dt_s": manifest["timing"]["control_dt_s"],
                "normalization_schema": manifest["schemas"]["normalization"],
                "control_frame_schema": manifest["schemas"]["control_frame"],
                "normalization": manifest["normalization"],
                "control": manifest["control"],
                "wheel_joint_sign_usd": manifest["action"]["wheel_joint_sign_usd"],
            },
            "reset_phases": {
                "pre_forward": reset_state_payload(
                    reset["reset_written_pre_forward"],
                    environment_index=0,
                    post_forward=False,
                ),
                "post_forward": reset_state_payload(
                    reset["reset_forwarded_post_forward"],
                    environment_index=0,
                    post_forward=True,
                ),
                "returned_policy": {
                    "actor_obs_current": current.tolist(),
                    "policy_input": policy_input.tolist(),
                    "previous_action": current[19:25].tolist(),
                },
            },
            "reset_identity_hash": baseline_reset_identity,
            "reset_cache_sha256": sha256_file(reset_cache_path),
            "reset_cache_tensor_sha256": reset_artifact["tensor_sha256"],
            "variant": variant_identity.variant,
            "reset_observation_replica_max_abs": reset_observation_replica_error,
            "actor_chain": {
                "policy_input_sha256": stable_hash(policy_input.tolist()),
                "estimated_velocity": inference.estimated_velocity.tolist(),
                "context_mu": inference.context_mu.tolist(),
                "context_logvar": inference.context_logvar.tolist(),
                "raw_action": raw_action.tolist(),
                "clipped_action": clipped_action.tolist(),
                "target_canonical": canonical_target.tolist(),
                "target_engine_native": native_target.tolist(),
                "previous_action_before": current[19:25].tolist(),
                "previous_action_after": next_current[19:25].tolist(),
                "next_actor_obs_current": next_current.tolist(),
            },
            "clock": {
                "physics_dt_s": float(env.physics_dt),
                "physics_steps_per_action": int(env.cfg.decimation),
                "control_dt_s": float(env.step_dt),
                "time_before_s": time_before,
                "time_after_s": time_after,
                "target_refresh_first_substep": bool(
                    first_substep["substep_index"] == 0
                ),
                "targets_constant_within_action": targets_constant,
            },
            "pulse_amplitude": pulse_amplitude,
            "pulse_records": pulse_records,
            "logger_handlers": handlers,
            "formal_config_hash_before": formal_config_hash_before,
            "formal_config_hash_after": stable_hash(WHEELLEG_CFG.to_dict()),
            "worker_source_sha256": sha256_file(Path(__file__)),
        }
        if payload["formal_config_hash_before"] != payload["formal_config_hash_after"]:
            raise RuntimeError("Formal WHEELLEG_CFG changed during the adapter probe")
        (output / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
        return payload
    finally:
        env.close()


def _collect_instrumentation_probe(
    output: Path,
    *,
    run_root: Path,
    device: str,
    mode: str,
    repetitions: int,
    ticks: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from isaaclab.sim import SimulationContext

    from debug.sim2sim.isaac_debug_env import (
        WheelLegSim2SimDebugEnv,
        make_debug_env_cfg,
    )
    from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG
    from wheelleg_dreamwaq.schemas.action import controlled_joint_feedback_usd_to_control
    from wheelleg_dreamwaq.schemas.frames import transform_usd_vector_to_control
    from wheelleg_dreamwaq.schemas.randomization import NOMINAL_EVALUATION_PROFILE_V1
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv

    from .isaac_probe_env import RootCauseIsaacProbeEnv

    if SimulationContext.instance() is not None:
        raise RuntimeError("Instrumentation probe requires no pre-existing SimulationContext")
    if repetitions != 5 or ticks != 10:
        raise ValueError("G03 instrumentation probe is frozen at 5 resets x 10 ticks")
    if mode not in {"formal", "debug", "contact", "system_observer"}:
        raise ValueError(f"Unknown instrumentation mode: {mode}")

    class SystemObserverDebugEnv(WheelLegSim2SimDebugEnv):
        def _debug_capture_direct_state(self) -> dict[str, torch.Tensor]:
            state = super()._debug_capture_direct_state()
            RootCauseIsaacProbeEnv.capture_system_state(self)
            return state

    formal_config_hash_before = stable_hash(WHEELLEG_CFG.to_dict())
    contact_enabled = mode == "contact"
    cfg = make_debug_env_cfg(device=device, enable_contact_sensors=contact_enabled)
    cfg.randomization = NOMINAL_EVALUATION_PROFILE_V1
    cfg.sim.save_logs_to_file = True
    log_directory = (run_root / "runtime_cache" / "isaaclab" / "logs").resolve()
    log_directory.mkdir(parents=True, exist_ok=True)
    cfg.sim.log_dir = str(log_directory)

    if mode == "formal":
        env = WheelLegFlatEnv(cfg)
    elif mode == "system_observer":
        env = SystemObserverDebugEnv(cfg, enable_contact_sensors=False)
    else:
        env = WheelLegSim2SimDebugEnv(
            cfg, enable_contact_sensors=contact_enabled
        )
    try:
        if mode != "formal":
            model_manifest = json.loads(
                (
                    Path(__file__).resolve().parents[3]
                    / "sim2sim"
                    / "mujoco"
                    / "model_manifest.json"
                ).read_text(encoding="utf-8")
            )
            env.configure_debug_joint_order(list(model_manifest["joint_order"]))
        rows: list[dict[str, Any]] = []
        read_rng_unchanged = True

        def capture(
            *,
            repetition: int,
            tick: int,
            observation: Any,
            terminated: Any,
            truncated: Any,
        ) -> None:
            nonlocal read_rng_unchanged
            rng_before = _randomization_rng_identity(env)["identity_hash"]
            if mode == "system_observer":
                RootCauseIsaacProbeEnv.capture_system_state(env)
            elif mode in {"debug", "contact"}:
                env._debug_capture_direct_state()
            robot = env.robot
            q = controlled_joint_feedback_usd_to_control(
                robot.data.joint_pos[:, env._controlled_joint_ids]
            )
            qd = controlled_joint_feedback_usd_to_control(
                robot.data.joint_vel[:, env._controlled_joint_ids]
            )
            linear = transform_usd_vector_to_control(robot.data.root_link_lin_vel_b)
            angular = transform_usd_vector_to_control(robot.data.root_link_ang_vel_b)
            rng_after = _randomization_rng_identity(env)["identity_hash"]
            read_rng_unchanged = read_rng_unchanged and rng_before == rng_after
            rows.append(
                {
                    "time_s": np.float64(tick * env.step_dt),
                    "repetition": np.int16(repetition),
                    "control_tick": np.int16(tick),
                    "actor_observation": _to_numpy(
                        observation["policy"], remove_env=False
                    )[0],
                    "controlled_position_canonical": _to_numpy(
                        q, remove_env=False
                    )[0],
                    "controlled_velocity_canonical": _to_numpy(
                        qd, remove_env=False
                    )[0],
                    "base_linear_velocity_control": _to_numpy(
                        linear, remove_env=False
                    )[0],
                    "base_angular_velocity_control": _to_numpy(
                        angular, remove_env=False
                    )[0],
                    "episode_length": np.int64(env.episode_length_buf[0].item()),
                    "common_step_counter": np.int64(env.common_step_counter),
                    "sim_step_counter": np.int64(env._sim_step_counter),
                    "terminated": np.int8(terminated),
                    "truncated": np.int8(truncated),
                }
            )

        for repetition in range(repetitions):
            observation, _ = env.reset(seed=0)
            capture(
                repetition=repetition,
                tick=0,
                observation=observation,
                terminated=0,
                truncated=0,
            )
            for tick in range(1, ticks + 1):
                observation, _, terminated, truncated, _ = env.step(
                    torch.zeros((1, 6), dtype=torch.float32, device=env.device)
                )
                capture(
                    repetition=repetition,
                    tick=tick,
                    observation=observation,
                    terminated=int(terminated[0].item()),
                    truncated=int(truncated[0].item()),
                )
        arrays = _rows_to_arrays(rows)
        units = {
            "time_s": "s",
            "repetition": "index",
            "control_tick": "index",
            "actor_observation": "normalized",
            "controlled_position_canonical": "rad",
            "controlled_velocity_canonical": "rad/s",
            "base_linear_velocity_control": "m/s",
            "base_angular_velocity_control": "rad/s",
            "episode_length": "tick",
            "common_step_counter": "tick",
            "sim_step_counter": "tick",
            "terminated": "bool",
            "truncated": "bool",
        }
        fields = {
            name: FieldSpec(
                units[name],
                "control_body" if name.startswith("base_") else "canonical",
                "post_step" if name != "time_s" else "sample_time",
                name,
            )
            for name in arrays
        }
        trace = write_trace(output / "trace", arrays, fields)
        view = env.robot.root_physx_view
        controlled = env._controlled_joint_ids
        compiled = {
            "stiffness": view.get_dof_stiffnesses()[0, controlled]
            .detach()
            .cpu()
            .tolist(),
            "damping": view.get_dof_dampings()[0, controlled]
            .detach()
            .cpu()
            .tolist(),
            "armature": view.get_dof_armatures()[0, controlled]
            .detach()
            .cpu()
            .tolist(),
            "physics_dt_s": float(env.physics_dt),
            "control_dt_s": float(env.step_dt),
            "decimation": int(env.cfg.decimation),
        }
        payload = {
            "schema_version": "RootCauseIsaacInstrumentationProbeV1",
            "mode": mode,
            "repetitions": repetitions,
            "ticks": ticks,
            "read_rng_unchanged": read_rng_unchanged,
            "compiled_properties": compiled,
            "compiled_properties_hash": stable_hash(compiled),
            "formal_config_hash_before": formal_config_hash_before,
            "formal_config_hash_after": stable_hash(WHEELLEG_CFG.to_dict()),
            "trace_sha256": trace.trace_sha256,
            "trace_metadata_sha256": trace.metadata_sha256,
        }
        (output / "instrumentation.json").write_bytes(
            canonical_json_bytes(payload) + b"\n"
        )
        return payload
    finally:
        env.close()


def run_identity(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    environment = _configure_run_environment(run_root)
    kit_paths = prepare_kit_paths(args.kit_root.resolve(), run_root=run_root)
    sources = validate_frozen_isaac_sources()
    kit_core = validate_kit_core()
    source_contracts = validate_source_contracts()

    guard = PythonWriteGuard(run_root)
    simulation_app = None
    fast_shutdown = args.command != "identity"
    with ExitStack() as stack:
        stack.enter_context(guard)
        output.mkdir(parents=True, exist_ok=False)
        pre_handlers = snapshot_file_handlers()

        from isaaclab.app import AppLauncher

        Launcher = root_cause_launcher_class(AppLauncher)
        launcher = Launcher(
            {
                "headless": True,
                "enable_cameras": False,
                "device": args.device,
                "fast_shutdown": fast_shutdown,
                "kit_args": kit_argument_string(kit_paths),
            }
        )
        simulation_app = launcher.app

        import carb
        import isaacsim  # noqa: F401
        import torch

        try:
            resolved = _resolved_kit_paths(carb, kit_paths)
            settings = carb.settings.get_settings()
            crash_enabled = bool(settings.get_as_bool("/crashreporter/enabled"))
            kit_state = validate_resolved_kit_state(
                kit_paths,
                resolved=resolved,
                crashreporter_enabled=crash_enabled,
                loaded_plugins=_plugin_names(carb),
            )
            prelaunch = dict(getattr(launcher, "_root_cause_prelaunch_config", {}))
            if prelaunch.get("enable_crashreporter") is not False:
                raise RuntimeError(
                    "Launcher did not preserve the pre-instantiation crash reporter proof"
                )
            if prelaunch.get("fast_shutdown") is not fast_shutdown:
                raise RuntimeError("Isaac worker shutdown mode changed before SimulationApp creation")
            probe = None
            if args.command == "one-tick":
                probe = _collect_one_tick(output, run_root=run_root, device=args.device)
            elif args.command == "properties":
                probe = _collect_properties(
                    output, run_root=run_root, device=args.device
                )
            elif args.command == "golden":
                probe = _collect_golden(output, device=args.device)
            elif args.command == "robot-probe":
                if args.scenario is None:
                    raise ValueError("robot-probe requires --scenario")
                probe = _collect_robot_probe(
                    output,
                    run_root=run_root,
                    device=args.device,
                    scenario=args.scenario,
                    control_ticks=args.control_ticks,
                    repetitions=args.repetitions,
                    amplitude=args.amplitude,
                    input_vector=args.input_vector,
                    reset_cache_path=args.reset_cache,
                    repeatability_family=args.repeatability_family,
                )
            elif args.command == "sphere-probe":
                if args.sphere_mode is None or args.friction is None:
                    raise ValueError("sphere-probe requires --sphere-mode and --friction")
                probe = _collect_sphere_probe(
                    output,
                    device=args.device,
                    mode=args.sphere_mode,
                    friction=args.friction,
                    repetitions=args.repetitions,
                    duration_s=args.duration,
                )
            elif args.command == "adapter-probe":
                probe = _collect_adapter_probe(
                    output,
                    run_root=run_root,
                    device=args.device,
                    pulse_amplitude=args.pulse_amplitude,
                )
            elif args.command == "instrumentation-probe":
                if args.instrumentation_mode is None:
                    raise ValueError(
                        "instrumentation-probe requires --instrumentation-mode"
                    )
                probe = _collect_instrumentation_probe(
                    output,
                    run_root=run_root,
                    device=args.device,
                    mode=args.instrumentation_mode,
                    repetitions=args.instrumentation_repetitions,
                    ticks=args.instrumentation_ticks,
                )
            elif args.command in {"replay-source", "replay"}:
                probe = _collect_isaac_replay(
                    output,
                    run_root=run_root,
                    device=args.device,
                    source_mode=args.command == "replay-source",
                    actions_path=args.actions,
                    source_result_path=args.source_result,
                    repetition=args.repetition,
                )
            payload = {
                "schema_version": "RootCauseIsaacIdentityV1",
                "python": sys.version,
                "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": args.device,
                "environment": environment,
                "frozen_sources": sources,
                "kit_core": kit_core,
                "source_contracts": source_contracts,
                "kit_state": kit_state,
                "prelaunch_config": prelaunch,
                "pre_handlers": pre_handlers,
                "probe": probe,
            }
        except BaseException:
            if fast_shutdown:
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()
                finalize_bootstrap(1)
                os._exit(1)
            simulation_app.close()
            raise

        if not fast_shutdown:
            simulation_app.close()
        payload.update(
            {
                "created_lifetime_handlers": list(guard.created_lifetime_handlers),
                "post_handlers": snapshot_file_handlers(),
                "write_guard": {
                    "installed": guard.installed,
                    "probe_count": guard.probe_count,
                    "capability_manifest": guard.capability_manifest,
                    "ledger": list(guard.ledger),
                    "ledger_cutoff": "immediately_before_identity_write",
                },
            }
        )
        (output / "identity.json").write_bytes(canonical_json_bytes(payload) + b"\n")
        payload["identity_sha256"] = sha256_file(output / "identity.json")
        if fast_shutdown:
            finalize_bootstrap(0)
            print(json.dumps(payload, sort_keys=True), flush=True)
            simulation_app.close(skip_cleanup=True)
        return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not sys.flags.dont_write_bytecode:
        raise RuntimeError("Isaac worker must be launched with -B")
    payload = run_identity(args)
    if args.command == "identity":
        print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
