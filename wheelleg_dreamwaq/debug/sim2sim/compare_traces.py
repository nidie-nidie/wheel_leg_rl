from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .trace_schema import (
    SCHEMA_VERSION,
    build_file_hashes,
    load_json,
    load_npz,
    sha256_array,
    sha256_file,
    stable_payload_hash,
    validate_control_trace,
    validate_substep_trace,
    write_json,
)


MATERIAL_THRESHOLDS = {
    "actor_obs_policy_pre_step": 1.0e-6,
    "previous_action_before_inference": 1.0e-6,
    "actor_output_raw": 1.0e-6,
    "action_clipped": 1.0e-6,
    "target_command_canonical": 1.0e-6,
    "active_joint_position_canonical_post_step_pre_reset": 1.0e-3,
    "active_joint_velocity_canonical_post_step_pre_reset": 1.0e-2,
    "all_hinge_position_named_post_step_pre_reset": 1.0e-3,
    "all_hinge_velocity_named_post_step_pre_reset": 1.0e-2,
    "base_com_position_diag_post_step_pre_reset": 1.0e-3,
    "base_linear_velocity_control_post_step_pre_reset": 2.0e-2,
    "base_angular_velocity_control_post_step_pre_reset": 2.0e-2,
    "projected_gravity_post_step_pre_reset": 1.0e-3,
    "base_height_post_step_pre_reset": 1.0e-3,
    "virtual_leg_phi0_post_step_pre_reset": 1.0e-3,
    "virtual_leg_length_post_step_pre_reset": 1.0e-4,
    "loop_closure_error_post_step_pre_reset": 1.0e-4,
}

EXACT_EVENT_FIELDS = (
    "effort_limit_event",
    "joint_velocity_limit_exceeded",
    "native_terminated_int8",
    "native_truncated_int8",
    "next_obs_is_reset_int8",
)

IDENTITY_METADATA_FIELDS = (
    "schema_version",
    "scenario_hash",
    "frozen_files",
    "scenario_variant",
    "action_sequence_identity",
    "command",
    "random_seed",
    "requested_control_ticks",
    "inference_dtype",
    "actor_sha256",
    "policy_manifest_sha256",
    "model_manifest_sha256",
    "model_xml_sha256",
    "control_dt_s",
    "canonical_joint_order",
    "canonical_from_engine_native",
    "all_hinge_order",
    "r_diag_from_engine_world",
)

RESET_SIGNALS = {
    "active_joint_position_canonical": 1.0e-6,
    "active_joint_velocity_canonical": 1.0e-8,
    "base_com_position_diag": 1.0e-5,
    "projected_gravity": 1.0e-6,
    "virtual_leg_phi0": 1.0e-5,
    "virtual_leg_length": 1.0e-6,
    "actor_obs_policy_rebuilt": 1.0e-6,
}


def current_engine_source_paths(
    project_root: str | Path,
    engine: str,
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    debug_root = root / "debug/sim2sim"
    common = {
        "trace_schema": debug_root / "trace_schema.py",
        "comparison": debug_root / "compare_traces.py",
        "stand_scenario": debug_root / "stand_scenario.py",
        "pitch_torque_pulse": debug_root / "pitch_torque_pulse.py",
    }
    if engine == "isaac_sim":
        formal_task = root / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat"
        schemas = root / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas"
        return {
            "collector": debug_root / "collect_isaac_trace.py",
            "debug_environment": debug_root / "isaac_debug_env.py",
            **common,
            "formal_environment": formal_task / "env.py",
            "formal_environment_config": formal_task / "env_cfg.py",
            "formal_observation_adapter": formal_task / "observations.py",
            "formal_action_adapter": formal_task / "control.py",
            "formal_action_schema": schemas / "action.py",
            "formal_frame_schema": schemas / "frames.py",
            "formal_normalization_schema": schemas / "normalization.py",
        }
    if engine == "mujoco":
        mujoco_root = root / "sim2sim/mujoco/wheelleg_mujoco"
        return {
            "collector": debug_root / "collect_mujoco_trace.py",
            **common,
            "mujoco_observation_adapter": mujoco_root / "observation.py",
            "mujoco_action_adapter": mujoco_root / "control.py",
            "mujoco_contract": mujoco_root / "contract.py",
            "mujoco_runtime": mujoco_root / "runner.py",
        }
    raise ValueError(f"Unsupported engine identity: {engine!r}")


def quaternion_geodesic_error(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs_wxyz, dtype=np.float64)
    rhs = np.asarray(rhs_wxyz, dtype=np.float64)
    lhs = lhs / np.linalg.norm(lhs, axis=-1, keepdims=True)
    rhs = rhs / np.linalg.norm(rhs, axis=-1, keepdims=True)
    dot = np.abs(np.sum(lhs * rhs, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def wrapped_angle_difference(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    delta = np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)
    return np.arctan2(np.sin(delta), np.cos(delta))


def _per_tick_max(error: np.ndarray) -> np.ndarray:
    array = np.asarray(error, dtype=np.float64)
    if array.ndim == 1:
        return np.abs(array)
    return np.nanmax(np.abs(array).reshape(array.shape[0], -1), axis=1)


def first_threshold_crossing(
    error: np.ndarray,
    *,
    numeric_limit: float,
    material_limit: float,
    persistent_ticks: int = 3,
) -> tuple[int | None, int | None, int | None]:
    per_tick = _per_tick_max(error)

    def first(mask: np.ndarray) -> int | None:
        indices = np.flatnonzero(mask)
        return int(indices[0]) if indices.size else None

    numeric = first(per_tick > numeric_limit)
    material_mask = per_tick > material_limit
    material = first(material_mask)
    persistent = None
    if persistent_ticks > 0 and material_mask.size >= persistent_ticks:
        runs = np.convolve(material_mask.astype(np.int8), np.ones(persistent_ticks, dtype=np.int8), mode="valid")
        starts = np.flatnonzero(runs == persistent_ticks)
        persistent = int(starts[0]) if starts.size else None
    return numeric, material, persistent


def _first_persistent_mask(mask: np.ndarray, persistent_ticks: int = 3) -> int | None:
    values = np.asarray(mask, dtype=bool)
    if persistent_ticks <= 0 or values.size < persistent_ticks:
        return None
    runs = np.convolve(
        values.astype(np.int8), np.ones(persistent_ticks, dtype=np.int8), mode="valid"
    )
    starts = np.flatnonzero(runs == persistent_ticks)
    return int(starts[0]) if starts.size else None


def summarize_signal(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    numeric_limit: float,
    material_limit: float,
) -> dict[str, Any]:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Signal shapes differ: {left.shape} != {right.shape}")
    difference = left - right
    return summarize_error(
        difference,
        shape=left.shape,
        numeric_limit=numeric_limit,
        material_limit=material_limit,
    )


def summarize_error(
    error: np.ndarray,
    *,
    shape: tuple[int, ...] | None = None,
    numeric_limit: float,
    material_limit: float,
) -> dict[str, Any]:
    difference = np.asarray(error, dtype=np.float64)
    numeric, material, persistent = first_threshold_crossing(
        difference,
        numeric_limit=numeric_limit,
        material_limit=material_limit,
    )
    return {
        "shape": list(shape if shape is not None else difference.shape),
        "max_abs_error": float(np.nanmax(np.abs(difference))),
        "rms_error": float(np.sqrt(np.nanmean(np.square(difference)))),
        "numeric_limit": float(numeric_limit),
        "material_limit": float(material_limit),
        "first_numeric_tick": numeric,
        "first_material_tick": material,
        "first_persistent_tick": persistent,
    }


def _default_numeric_limit(*arrays: np.ndarray) -> float:
    epsilon = 0.0
    scale = 1.0
    for value in arrays:
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            epsilon = max(epsilon, float(np.finfo(array.dtype).eps))
        if array.size:
            scale = max(scale, float(np.nanmax(np.abs(array))))
    return 32.0 * (epsilon or float(np.finfo(np.float64).eps)) * scale


def natural_reproduction_limit(samples: np.ndarray) -> dict[str, float]:
    values = np.asarray(samples)
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("Natural envelope requires at least two repeated runs")
    maximum_pairwise = 0.0
    for first in range(values.shape[0]):
        for second in range(first + 1, values.shape[0]):
            maximum_pairwise = max(maximum_pairwise, float(np.nanmax(np.abs(values[first] - values[second]))))
    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.dtype(np.float64)
    scale = max(1.0, float(np.nanmax(np.abs(values))))
    numeric_floor = 32.0 * float(np.finfo(dtype).eps) * scale
    return {
        "observed_envelope": maximum_pairwise,
        "dtype_epsilon": float(np.finfo(dtype).eps),
        "scale": scale,
        "numeric_floor": numeric_floor,
        "equivalence_limit": max(maximum_pairwise, numeric_floor),
    }


def compare_control_traces(
    isaac: Mapping[str, np.ndarray],
    mujoco: Mapping[str, np.ndarray],
    *,
    numeric_limits: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    limits = dict(numeric_limits or {})
    signals: dict[str, dict[str, Any]] = {}
    common_rows = min(len(isaac["control_tick"]), len(mujoco["control_tick"]))
    for name, material_limit in MATERIAL_THRESHOLDS.items():
        if name not in isaac or name not in mujoco:
            continue
        lhs = np.asarray(isaac[name])[:common_rows]
        rhs = np.asarray(mujoco[name])[:common_rows]
        numeric_limit = float(limits.get(name, _default_numeric_limit(lhs, rhs)))
        if name == "virtual_leg_phi0_post_step_pre_reset":
            signals[name] = summarize_error(
                wrapped_angle_difference(lhs, rhs),
                shape=lhs.shape,
                numeric_limit=numeric_limit,
                material_limit=material_limit,
            )
        else:
            signals[name] = summarize_signal(
                lhs,
                rhs,
                numeric_limit=numeric_limit,
                material_limit=material_limit,
            )
    orientation_name = "base_orientation_control_wxyz_post_step_pre_reset"
    if orientation_name in isaac and orientation_name in mujoco:
        error = quaternion_geodesic_error(
            np.asarray(isaac[orientation_name])[:common_rows],
            np.asarray(mujoco[orientation_name])[:common_rows],
        )
        numeric_limit = float(
            limits.get(
                orientation_name,
                _default_numeric_limit(
                    np.asarray(isaac[orientation_name])[:common_rows],
                    np.asarray(mujoco[orientation_name])[:common_rows],
                ),
            )
        )
        numeric, material, persistent = first_threshold_crossing(
            error,
            numeric_limit=numeric_limit,
            material_limit=np.deg2rad(0.5),
        )
        signals[orientation_name] = {
            "shape": list(error.shape),
            "max_abs_error": float(np.nanmax(error)),
            "rms_error": float(np.sqrt(np.nanmean(np.square(error)))),
            "numeric_limit": numeric_limit,
            "material_limit": float(np.deg2rad(0.5)),
            "first_numeric_tick": numeric,
            "first_material_tick": material,
            "first_persistent_tick": persistent,
        }
    for name in EXACT_EVENT_FIELDS:
        if name not in isaac or name not in mujoco:
            continue
        lhs = np.asarray(isaac[name])[:common_rows]
        rhs = np.asarray(mujoco[name])[:common_rows]
        if lhs.shape != rhs.shape:
            raise ValueError(f"Event shapes differ for {name}: {lhs.shape} != {rhs.shape}")
        mismatch = np.any(lhs != rhs, axis=tuple(range(1, lhs.ndim))) if lhs.ndim > 1 else lhs != rhs
        indices = np.flatnonzero(mismatch)
        signals[name] = {
            "shape": list(lhs.shape),
            "exact_match": not bool(indices.size),
            "first_numeric_tick": int(indices[0]) if indices.size else None,
            "first_material_tick": int(indices[0]) if indices.size else None,
            "first_persistent_tick": _first_persistent_mask(mismatch),
        }

    common_name = "common_diagnostic_flags_int8"
    if common_name in isaac and common_name in mujoco:
        lhs = np.asarray(isaac[common_name])[:common_rows]
        rhs = np.asarray(mujoco[common_name])[:common_rows]
        if lhs.shape != rhs.shape:
            raise ValueError(f"Event shapes differ for {common_name}: {lhs.shape} != {rhs.shape}")
        known = (lhs >= 0) & (rhs >= 0)
        component_mismatch = (lhs != rhs) & known
        mismatch = np.any(component_mismatch, axis=1)
        indices = np.flatnonzero(mismatch)
        signals[common_name] = {
            "shape": list(lhs.shape),
            "exact_match_where_known": not bool(indices.size),
            "unknown_component_count": int(np.count_nonzero(~known)),
            "first_numeric_tick": int(indices[0]) if indices.size else None,
            "first_material_tick": int(indices[0]) if indices.size else None,
            "first_persistent_tick": _first_persistent_mask(mismatch),
        }

    if "mujoco_velocity_guard_active" in isaac or "mujoco_velocity_guard_active" in mujoco:
        signals["mujoco_velocity_guard_active"] = {
            "cross_engine_evaluable": False,
            "reason": "The guard exists only in the explicit MuJoCo actuator path",
            "first_numeric_tick": None,
            "first_material_tick": None,
            "first_persistent_tick": None,
        }

    def first_global(key: str) -> dict[str, Any] | None:
        candidates = [
            (metrics[key], name)
            for name, metrics in signals.items()
            if metrics.get(key) is not None
        ]
        first = min(candidates) if candidates else None
        return {"control_tick": first[0], "signal": first[1]} if first else None

    return {
        "schema_version": "Sim2SimComparisonV1",
        "common_control_ticks": common_rows,
        "signals": signals,
        "first_numeric_divergence": first_global("first_numeric_tick"),
        "first_material_divergence": first_global("first_material_tick"),
        "first_persistent_divergence": first_global("first_persistent_tick"),
    }


def _nested_value(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def compare_reset_snapshots(
    isaac: Mapping[str, Any], mujoco: Mapping[str, Any]
) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    for phase in ("reset_forwarded_post_forward", "reset_returned_to_policy"):
        phase_signals: dict[str, Any] = {}
        names = dict(RESET_SIGNALS)
        if phase == "reset_returned_to_policy":
            names = {"actor_obs_policy_returned": 1.0e-6, "previous_action": 0.0}
        for name, limit in names.items():
            lhs = _nested_value(isaac, (phase, name))
            rhs = _nested_value(mujoco, (phase, name))
            if lhs is None or rhs is None:
                continue
            left = np.asarray(lhs, dtype=np.float64)
            right = np.asarray(rhs, dtype=np.float64)
            if left.shape != right.shape:
                phase_signals[name] = {
                    "shape_match": False,
                    "isaac_shape": list(left.shape),
                    "mujoco_shape": list(right.shape),
                }
                continue
            difference = (
                np.abs(wrapped_angle_difference(left, right))
                if name == "virtual_leg_phi0"
                else np.abs(left - right)
            )
            phase_signals[name] = {
                "shape_match": True,
                "max_abs_error": float(np.max(difference)) if difference.size else 0.0,
                "limit": float(limit),
                "within_limit": bool(np.all(difference <= limit)),
            }
        if phase == "reset_forwarded_post_forward":
            lhs = _nested_value(isaac, (phase, "base_orientation_control_wxyz"))
            rhs = _nested_value(mujoco, (phase, "base_orientation_control_wxyz"))
            if lhs is not None and rhs is not None:
                left = np.asarray(lhs, dtype=np.float64).reshape(1, 4)
                right = np.asarray(rhs, dtype=np.float64).reshape(1, 4)
                error = float(quaternion_geodesic_error(left, right)[0])
                phase_signals["base_orientation_control_wxyz"] = {
                    "shape_match": True,
                    "geodesic_error_rad": error,
                    "limit": 1.0e-5,
                    "within_limit": error <= 1.0e-5,
                }
        phases[phase] = phase_signals
    failures = [
        f"{phase}.{name}"
        for phase, signals in phases.items()
        for name, metrics in signals.items()
        if not metrics.get("shape_match", True) or not metrics.get("within_limit", True)
    ]
    return {"phases": phases, "failures": failures, "passed": not failures}


def compare_metadata_identity(
    isaac: Mapping[str, Any], mujoco: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {}
    for name in IDENTITY_METADATA_FIELDS:
        isaac_value = isaac.get(name)
        mujoco_value = mujoco.get(name)
        checks[name] = {
            "match": isaac_value is not None and mujoco_value is not None and isaac_value == mujoco_value,
            "isaac": isaac_value,
            "mujoco": mujoco_value,
        }
    failures = [name for name, result in checks.items() if not result["match"]]
    return {"checks": checks, "failures": failures, "passed": not failures}


def check_internal_torque_invariants(
    isaac_substeps: Mapping[str, np.ndarray], mujoco_substeps: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    isaac_reference = np.asarray(isaac_substeps["pd_torque_effort_clipped_canonical"])
    isaac_host = np.asarray(isaac_substeps["isaac_host_pd_torque_estimate_canonical"])
    isaac_error = np.abs(isaac_reference - isaac_host)

    mujoco_reference = np.asarray(mujoco_substeps["pd_torque_effort_clipped_engine_native"])
    mujoco_guard = np.asarray(mujoco_substeps["mujoco_velocity_guard_active"]).astype(bool)
    mujoco_expected = np.where(mujoco_guard, 0.0, mujoco_reference)
    mujoco_commanded = np.asarray(mujoco_substeps["mujoco_commanded_torque_engine_native"])
    mujoco_error = np.abs(mujoco_expected - mujoco_commanded)
    return {
        "isaac_host_vs_effort_clipped": {
            "max_abs_error": float(np.max(isaac_error)),
            "rms_error": float(np.sqrt(np.mean(np.square(isaac_error)))),
        },
        "mujoco_commanded_vs_effort_clipped_and_guard": {
            "max_abs_error": float(np.max(mujoco_error)),
            "rms_error": float(np.sqrt(np.mean(np.square(mujoco_error)))),
        },
    }


def _candidate_evidence(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not summary["identity"]["passed"]:
        candidates.append({"candidate": "identity_or_input_error", "evidence": summary["identity"]["failures"]})
    if not summary["reset"]["passed"]:
        candidates.append({"candidate": "reset_or_coordinate_error", "evidence": summary["reset"]["failures"]})
    first = summary["control"]["first_material_divergence"]
    if first is not None:
        name = first["signal"]
        if name.startswith(("actor_obs", "previous_action", "actor_output")):
            candidate = "observation_or_closed_loop_amplification"
        elif name.startswith(("action_clipped", "target_command")):
            candidate = "action_adapter_error"
        elif name.startswith("all_hinge"):
            candidate = "closed_chain_or_passive_dynamics_difference"
        else:
            candidate = "actuator_or_physics_response_difference"
        candidates.append({"candidate": candidate, "evidence": first})
    if not summary["contact"]["evaluable"]:
        candidates.append(
            {
                "candidate": "contact_difference",
                "evidence": "not_evaluable_in_uninstrumented_baseline",
            }
        )
    return candidates


def load_verified_engine_run(
    engine_directory: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    directory = Path(engine_directory).resolve()
    metadata_path = directory / "metadata.json"
    reset_path = directory / "reset_snapshot.json"
    control_path = directory / "control_trace.npz"
    substep_path = directory / "substep_trace.npz"
    hashes_path = directory / "file_hashes.json"
    metadata = load_json(metadata_path)
    reset = load_json(reset_path)
    expected_hashes = load_json(hashes_path)

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported metadata schema in {directory}: {metadata.get('schema_version')!r}")
    if reset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported reset schema in {directory}: {reset.get('schema_version')!r}")
    required_outputs = {
        "metadata": metadata_path,
        "reset_snapshot": reset_path,
        "control_trace": control_path,
        "substep_trace": substep_path,
    }
    missing_hashes = sorted(set(required_outputs) - set(expected_hashes))
    if missing_hashes:
        raise ValueError(f"Run hash manifest is missing required outputs: {missing_hashes}")
    hash_mismatches = {
        name: {"expected": expected_hashes[name], "actual": sha256_file(path)}
        for name, path in required_outputs.items()
        if expected_hashes[name] != sha256_file(path)
    }
    if hash_mismatches:
        raise ValueError(f"Run artifact hash mismatch: {hash_mismatches}")

    engine = metadata.get("engine")
    source_paths = current_engine_source_paths(
        project_root or Path(__file__).resolve().parents[2],
        str(engine),
    )
    required_sources = tuple(source_paths)
    missing_sources = sorted(set(required_sources) - set(expected_hashes))
    if missing_sources:
        raise ValueError(f"Run hash manifest is missing required source hashes: {missing_sources}")
    malformed_hashes = [
        name
        for name in required_sources
        if not isinstance(expected_hashes[name], str) or len(expected_hashes[name]) != 64
    ]
    if malformed_hashes:
        raise ValueError(f"Run hash manifest contains malformed source hashes: {malformed_hashes}")
    current_source_mismatches = {
        name: {"collected": expected_hashes[name], "current": sha256_file(path)}
        for name, path in source_paths.items()
        if expected_hashes[name] != sha256_file(path)
    }
    if current_source_mismatches:
        raise ValueError(f"Run source files differ from the current workspace: {current_source_mismatches}")
    source_hashes = (
        metadata.get("formal_source_sha256")
        if engine == "isaac_sim"
        else metadata.get("mujoco_source_sha256")
    )
    if not isinstance(source_hashes, Mapping):
        raise ValueError("Run metadata is missing its source hash mapping")
    source_mismatches = {
        name: {"metadata": digest, "manifest": expected_hashes.get(name)}
        for name, digest in source_hashes.items()
        if expected_hashes.get(name) != digest
    }
    if source_mismatches:
        raise ValueError(f"Run source hash identity mismatch: {source_mismatches}")

    action_identity = metadata.get("action_sequence_identity")
    if not isinstance(action_identity, Mapping):
        raise ValueError("Run metadata is missing action_sequence_identity")
    content_hash = action_identity.get("content_sha256")
    if content_hash is not None:
        embedded_path = directory / "input_action_sequence.npz"
        expected_embedded_hash = expected_hashes.get("input_action_sequence")
        if expected_embedded_hash is None or not embedded_path.is_file():
            raise ValueError("Run is missing its embedded action sequence")
        actual_embedded_hash = sha256_file(embedded_path)
        if actual_embedded_hash != expected_embedded_hash:
            raise ValueError("Embedded action sequence file hash mismatch")
        embedded = load_npz(embedded_path)
        if set(embedded) != {"action_sequence"}:
            raise ValueError("Embedded action sequence must contain only action_sequence")
        sequence = embedded["action_sequence"]
        if sha256_array(sequence) != content_hash:
            raise ValueError("Embedded action sequence content hash mismatch")
        if int(action_identity.get("rows", -1)) != len(sequence):
            raise ValueError("Embedded action sequence row count mismatch")

    control = load_npz(control_path)
    substeps = load_npz(substep_path)
    validate_control_trace(control)
    try:
        physics_steps = int(metadata["physics_steps_per_action"])
        continuity_atol = float(metadata["substep_continuity_atol"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Run metadata is missing a valid substep validation contract") from exc
    validate_substep_trace(
        substeps,
        physics_steps_per_action=physics_steps,
        continuity_atol=continuity_atol,
    )
    completed = int(metadata.get("completed_control_ticks", -1))
    if completed != len(control["control_tick"]):
        raise ValueError(
            f"completed_control_ticks {completed} does not match trace rows {len(control['control_tick'])}"
        )
    return {
        "metadata": metadata,
        "reset": reset,
        "control": control,
        "substeps": substeps,
        "file_hashes": expected_hashes,
    }


def write_comparison(run_directory: str | Path) -> dict[str, Any]:
    root = Path(run_directory)
    isaac_run = load_verified_engine_run(root / "isaac")
    mujoco_run = load_verified_engine_run(root / "mujoco")
    isaac = isaac_run["control"]
    mujoco = mujoco_run["control"]
    isaac_substeps = isaac_run["substeps"]
    mujoco_substeps = mujoco_run["substeps"]
    isaac_metadata = isaac_run["metadata"]
    mujoco_metadata = mujoco_run["metadata"]
    isaac_reset = isaac_run["reset"]
    mujoco_reset = mujoco_run["reset"]
    contact_evaluable = (
        isaac_metadata.get("contact_observation_mode") == "instrumented_equivalent"
        and mujoco_metadata.get("contact_observation_mode") == "native"
    )
    summary = {
        "schema_version": "Sim2SimComparisonV1",
        "input_artifact_hashes": {
            "isaac": {
                name: isaac_run["file_hashes"][name]
                for name in ("metadata", "reset_snapshot", "control_trace", "substep_trace")
            },
            "mujoco": {
                name: mujoco_run["file_hashes"][name]
                for name in ("metadata", "reset_snapshot", "control_trace", "substep_trace")
            },
        },
        "identity": compare_metadata_identity(isaac_metadata, mujoco_metadata),
        "reset": compare_reset_snapshots(isaac_reset, mujoco_reset),
        "control": compare_control_traces(isaac, mujoco),
        "internal_torque_invariants": check_internal_torque_invariants(
            isaac_substeps, mujoco_substeps
        ),
        "contact": {
            "evaluable": contact_evaluable,
            "isaac_mode": isaac_metadata.get("contact_observation_mode"),
            "mujoco_mode": mujoco_metadata.get("contact_observation_mode"),
            "reason": None if contact_evaluable else "Isaac contact instrumentation has not passed its equivalence gate",
        },
    }
    summary["evidence_ranked_candidates"] = _candidate_evidence(summary)
    summary["comparison_hash"] = stable_payload_hash(summary)
    output = root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    with (output / "per_signal_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("signal", "max_abs_error", "rms_error", "first_material_tick", "first_persistent_tick"))
        for name, metrics in sorted(summary["control"]["signals"].items()):
            writer.writerow(
                (
                    name,
                    metrics.get("max_abs_error"),
                    metrics.get("rms_error"),
                    metrics.get("first_material_tick"),
                    metrics.get("first_persistent_tick"),
                )
            )
    write_json(
        output / "file_hashes.json",
        build_file_hashes(
            {
                "summary": output / "summary.json",
                "per_signal_metrics": output / "per_signal_metrics.csv",
                "comparison": Path(__file__),
                "trace_schema": Path(__file__).with_name("trace_schema.py"),
            }
        ),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen Isaac and MuJoCo Sim2Sim debug traces.")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    summary = write_comparison(args.run_directory.resolve())
    print(json.dumps(summary["control"]["first_material_divergence"], sort_keys=True))


if __name__ == "__main__":
    main()
