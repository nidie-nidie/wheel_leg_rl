from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .compare_traces import (
    load_verified_engine_run,
    quaternion_geodesic_error,
    wrapped_angle_difference,
)
from .trace_schema import (
    build_file_hashes,
    load_json,
    load_npz,
    sha256_file,
    stable_payload_hash,
    write_json,
)


FLOAT_FIELDS = (
    "actor_obs_policy_pre_step",
    "actor_output_raw",
    "action_clipped",
    "active_joint_position_canonical_post_step_pre_reset",
    "active_joint_velocity_canonical_post_step_pre_reset",
    "all_hinge_position_named_post_step_pre_reset",
    "all_hinge_velocity_named_post_step_pre_reset",
    "base_com_position_engine_world_post_step_pre_reset",
    "base_com_position_diag_post_step_pre_reset",
    "base_linear_velocity_control_post_step_pre_reset",
    "base_angular_velocity_control_post_step_pre_reset",
    "projected_gravity_post_step_pre_reset",
    "base_height_post_step_pre_reset",
    "virtual_leg_length_post_step_pre_reset",
    "virtual_leg_phi0_post_step_pre_reset",
    "loop_closure_error_post_step_pre_reset",
    "next_actor_obs_policy_returned",
    "previous_action_before_inference",
)
EXACT_FIELDS = ("control_tick", "native_terminated_int8", "native_truncated_int8")
ORIENTATION_FIELD = "base_orientation_control_wxyz_post_step_pre_reset"
WRAPPED_ANGLE_FIELDS = ("virtual_leg_phi0_post_step_pre_reset",)

SHARED_IDENTITY_FIELDS = (
    "engine",
    "engine_version",
    "isaac_lab_version",
    "isaac_lab_extension_version",
    "isaac_lab_commit",
    "python_version",
    "numpy_version",
    "torch_version",
    "scenario_hash",
    "frozen_files",
    "scenario_variant",
    "action_sequence_identity",
    "command",
    "random_seed",
    "requested_control_ticks",
    "actor_sha256",
    "policy_manifest_sha256",
    "model_manifest_sha256",
    "model_xml_sha256",
    "control_dt_s",
    "physics_dt_s",
    "physics_steps_per_action",
    "inference_device",
    "inference_dtype",
    "simulation_device",
    "physics_pipeline",
    "canonical_joint_order",
    "canonical_from_engine_native",
    "all_hinge_order",
    "r_diag_from_engine_world",
    "solver_position_iterations",
    "solver_velocity_iterations",
    "clone_in_fabric",
    "use_fabric",
    "render_mode",
    "contact_observation_mode",
    "contact_processing_state",
    "formal_source_sha256",
)


def independent_run_report(
    directories: list[Path],
    metadata: list[dict[str, Any]],
    *,
    trace_filename: str,
) -> dict[str, Any]:
    resolved = [str(path.resolve()).casefold() for path in directories]
    collection_ids = [item.get("collection_id") for item in metadata]
    trace_hashes = [sha256_file(path.resolve() / trace_filename) for path in directories]
    metadata_hashes = [sha256_file(path.resolve() / "metadata.json") for path in directories]

    def duplicates(values: list[Any]) -> list[Any]:
        return sorted({value for value in values if values.count(value) > 1}, key=str)

    failures: list[str] = []
    duplicate_directories = duplicates(resolved)
    duplicate_collection_ids = duplicates(collection_ids)
    duplicate_trace_hashes = duplicates(trace_hashes)
    duplicate_metadata_hashes = duplicates(metadata_hashes)
    if duplicate_directories:
        failures.append("duplicate_directories")
    if any(value is None for value in collection_ids):
        failures.append("missing_collection_id")
    if duplicate_collection_ids:
        failures.append("duplicate_collection_ids")
    if duplicate_metadata_hashes:
        failures.append("duplicate_metadata_hashes")
    return {
        "run_count": len(directories),
        "collection_ids": collection_ids,
        "trace_hashes": trace_hashes,
        "metadata_hashes": metadata_hashes,
        "duplicate_directories": duplicate_directories,
        "duplicate_collection_ids": duplicate_collection_ids,
        "duplicate_trace_hashes": duplicate_trace_hashes,
        "duplicate_metadata_hashes": duplicate_metadata_hashes,
        "failures": failures,
        "passed": not failures,
    }


def _current_production_source_paths(project_root: Path) -> dict[str, Path]:
    debug_root = project_root / "debug/sim2sim"
    formal_task = project_root / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat"
    schemas = project_root / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas"
    return {
        "collector": debug_root / "collect_isaac_production_reference.py",
        "debug_environment": debug_root / "isaac_debug_env.py",
        "trace_schema": debug_root / "trace_schema.py",
        "stand_scenario": debug_root / "stand_scenario.py",
        "equivalence_comparison": debug_root / "compare_isaac_equivalence.py",
        "formal_environment": formal_task / "env.py",
        "formal_environment_config": formal_task / "env_cfg.py",
        "formal_observation_adapter": formal_task / "observations.py",
        "formal_action_adapter": formal_task / "control.py",
        "formal_action_schema": schemas / "action.py",
        "formal_frame_schema": schemas / "frames.py",
        "formal_normalization_schema": schemas / "normalization.py",
    }


def _pairwise_max(samples: np.ndarray, error: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    maximum = 0.0
    for first in range(samples.shape[0]):
        for second in range(first + 1, samples.shape[0]):
            maximum = max(maximum, float(np.max(error(samples[first], samples[second]))))
    return maximum


def _absolute_error(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64))


def _numeric_floor(samples: np.ndarray) -> float:
    dtype = samples.dtype if np.issubdtype(samples.dtype, np.floating) else np.dtype(np.float64)
    scale = max(1.0, float(np.max(np.abs(samples))))
    return 32.0 * float(np.finfo(dtype).eps) * scale


def _field_report(
    production: np.ndarray,
    debug: np.ndarray,
    *,
    error: Callable[[np.ndarray, np.ndarray], np.ndarray] = _absolute_error,
) -> dict[str, Any]:
    production_envelope = _pairwise_max(production, error)
    debug_envelope = _pairwise_max(debug, error)
    numerical_floor = max(_numeric_floor(production), _numeric_floor(debug))
    limit = max(production_envelope, debug_envelope, numerical_floor)
    cross_max = 0.0
    for production_run in production:
        for debug_run in debug:
            cross_max = max(cross_max, float(np.max(error(production_run, debug_run))))
    return {
        "production_envelope": production_envelope,
        "debug_envelope": debug_envelope,
        "numeric_floor": numerical_floor,
        "equivalence_limit": limit,
        "cross_path_max_error": cross_max,
        "passed": cross_max <= limit,
    }


def compare_equivalence_runs(
    production_runs: list[dict[str, np.ndarray]],
    debug_runs: list[dict[str, np.ndarray]],
    *,
    float_fields: tuple[str, ...] = FLOAT_FIELDS,
) -> dict[str, Any]:
    if len(production_runs) < 2 or len(debug_runs) < 2:
        raise ValueError("Equivalence comparison requires at least two runs per path")
    report: dict[str, Any] = {"schema_version": "IsaacCollectorEquivalenceV1", "signals": {}}
    failures: list[str] = []
    for name in float_fields:
        production = np.stack([run[name] for run in production_runs])
        debug = np.stack([run[name] for run in debug_runs])
        if production.shape != debug.shape:
            raise ValueError(f"Equivalence shape mismatch for {name}: {production.shape} != {debug.shape}")
        error = wrapped_angle_difference if name in WRAPPED_ANGLE_FIELDS else _absolute_error
        metrics = _field_report(production, debug, error=error)
        report["signals"][name] = metrics
        if not metrics["passed"]:
            failures.append(name)

    production_orientation = np.stack([run[ORIENTATION_FIELD] for run in production_runs])
    debug_orientation = np.stack([run[ORIENTATION_FIELD] for run in debug_runs])
    orientation_metrics = _field_report(
        production_orientation,
        debug_orientation,
        error=quaternion_geodesic_error,
    )
    report["signals"][ORIENTATION_FIELD] = orientation_metrics
    if not orientation_metrics["passed"]:
        failures.append(ORIENTATION_FIELD)

    for name in EXACT_FIELDS:
        production = np.stack([run[name] for run in production_runs])
        debug = np.stack([run[name] for run in debug_runs])
        exact = bool(np.array_equal(production, debug))
        report["signals"][name] = {"exact_match": exact, "passed": exact}
        if not exact:
            failures.append(name)
    report["production_runs"] = len(production_runs)
    report["debug_runs"] = len(debug_runs)
    report["failures"] = failures
    report["passed"] = not failures
    return report


def _load_verified_production_run(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = directory.resolve()
    trace_path = root / "reference_trace.npz"
    metadata_path = root / "metadata.json"
    hashes_path = root / "file_hashes.json"
    metadata = load_json(metadata_path)
    expected_hashes = load_json(hashes_path)
    if metadata.get("schema_version") != "IsaacProductionEquivalenceReferenceV1":
        raise ValueError(f"Unsupported production-reference schema in {root}")
    required = {
        "reference_trace": trace_path,
        "metadata": metadata_path,
    }
    missing = sorted(set(required) - set(expected_hashes))
    if missing:
        raise ValueError(f"Production reference hash manifest is missing outputs: {missing}")
    mismatches = {
        name: {"expected": expected_hashes[name], "actual": sha256_file(path)}
        for name, path in required.items()
        if expected_hashes[name] != sha256_file(path)
    }
    if mismatches:
        raise ValueError(f"Production reference artifact hash mismatch: {mismatches}")
    project_root = Path(__file__).resolve().parents[2]
    source_paths = _current_production_source_paths(project_root)
    missing_sources = sorted(set(source_paths) - set(expected_hashes))
    if missing_sources:
        raise ValueError(f"Production reference source hashes are missing: {missing_sources}")
    source_mismatches = {
        name: {"collected": expected_hashes[name], "current": sha256_file(path)}
        for name, path in source_paths.items()
        if expected_hashes[name] != sha256_file(path)
    }
    if source_mismatches:
        raise ValueError(f"Production source files differ from the current workspace: {source_mismatches}")
    formal_source_hashes = metadata.get("formal_source_sha256")
    if not isinstance(formal_source_hashes, dict):
        raise ValueError("Production metadata is missing formal_source_sha256")
    formal_mismatches = {
        name: {"metadata": digest, "manifest": expected_hashes.get(name)}
        for name, digest in formal_source_hashes.items()
        if expected_hashes.get(name) != digest
    }
    if formal_mismatches:
        raise ValueError(f"Production formal source hash identity mismatch: {formal_mismatches}")

    trace = load_npz(trace_path)
    required_fields = set(FLOAT_FIELDS) | set(EXACT_FIELDS) | {ORIENTATION_FIELD}
    missing_fields = sorted(required_fields - set(trace))
    if missing_fields:
        raise ValueError(f"Production reference trace is missing fields: {missing_fields}")
    rows = int(metadata.get("completed_control_ticks", -1))
    for name in required_fields:
        if len(trace[name]) != rows:
            raise ValueError(f"Production reference field {name} has the wrong row count")
    return trace, metadata


def _identity_report(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    failures: list[str] = []
    for name in SHARED_IDENTITY_FIELDS:
        values = [item.get(name) for item in metadata]
        serialized = [json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values]
        passed = all(value is not None for value in values) and len(set(serialized)) == 1
        checks[name] = {"passed": passed, "values": values if not passed else [values[0]]}
        if not passed:
            failures.append(name)
    return {"checks": checks, "failures": failures, "passed": not failures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare production and debug Isaac trajectories.")
    parser.add_argument("--production", type=Path, nargs="+", required=True)
    parser.add_argument("--debug", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if not args.allow_smoke and (len(args.production) < 5 or len(args.debug) < 5):
        parser.error("The formal gate requires at least five runs per path")
    if len(args.production) != len(args.debug):
        parser.error("Production and debug run counts must match")

    production_loaded = [_load_verified_production_run(path) for path in args.production]
    debug_loaded = [load_verified_engine_run(path) for path in args.debug]
    production = [item[0] for item in production_loaded]
    debug = [item["control"] for item in debug_loaded]
    production_metadata = [item[1] for item in production_loaded]
    debug_metadata = [item["metadata"] for item in debug_loaded]
    production_independence = independent_run_report(
        args.production,
        production_metadata,
        trace_filename="reference_trace.npz",
    )
    debug_independence = independent_run_report(
        args.debug,
        debug_metadata,
        trace_filename="control_trace.npz",
    )
    identity = _identity_report(production_metadata + debug_metadata)
    observer_failures = [
        *(
            f"production[{index}]"
            for index, item in enumerate(production_metadata)
            if item.get("observer_mode") != "production_unmodified_step"
            or item.get("step_override") is not False
        ),
        *(
            f"debug[{index}]"
            for index, item in enumerate(debug_metadata)
            if item.get("observer_mode") != "debug_step_subclass"
        ),
    ]
    report = compare_equivalence_runs(production, debug)
    report["input_artifact_hashes"] = {
        "production": [
            {
                "metadata": sha256_file(path.resolve() / "metadata.json"),
                "reference_trace": sha256_file(path.resolve() / "reference_trace.npz"),
            }
            for path in args.production
        ],
        "debug": [
            {
                "metadata": sha256_file(path.resolve() / "metadata.json"),
                "control_trace": sha256_file(path.resolve() / "control_trace.npz"),
                "substep_trace": sha256_file(path.resolve() / "substep_trace.npz"),
            }
            for path in args.debug
        ],
    }
    report["identity"] = identity
    report["independence"] = {
        "production": production_independence,
        "debug": debug_independence,
    }
    report["observer_failures"] = observer_failures
    report["passed"] = (
        report["passed"]
        and identity["passed"]
        and production_independence["passed"]
        and debug_independence["passed"]
        and not observer_failures
    )
    report["gate_mode"] = "smoke" if args.allow_smoke else "formal"
    report["formal_gate_passed"] = (
        report["passed"] and not args.allow_smoke and len(production) >= 5
    )
    report["report_hash"] = stable_payload_hash(report)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", report)
    write_json(
        output / "file_hashes.json",
        build_file_hashes(
            {
                "summary": output / "summary.json",
                "comparison": Path(__file__),
                "trace_schema": Path(__file__).with_name("trace_schema.py"),
            }
        ),
    )
    print(json.dumps({"passed": report["passed"], "failures": report["failures"]}, sort_keys=True))
    gate_passed = report["passed"] if args.allow_smoke else report["formal_gate_passed"]
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
