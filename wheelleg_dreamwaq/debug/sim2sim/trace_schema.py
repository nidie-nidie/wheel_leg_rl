from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "Sim2SimDebugTraceV1"
CONTROLLED_JOINT_ORDER = ("jIJ", "jIO", "jAB", "jAG", "jwheel_left", "jwheel_right")
CONTROLLED_SIGNS = np.asarray((1.0, 1.0, 1.0, 1.0, 1.0, -1.0), dtype=np.float64)
DIAGNOSTIC_FLAG_ORDER = (
    "height",
    "tilt",
    "root_linear_velocity",
    "root_angular_velocity",
    "joint_velocity",
    "loop_closure",
    "virtual_leg_length",
    "unexpected_contact",
)


def _six_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1] != 6:
        raise ValueError(f"Expected six controlled values, got shape {array.shape}")
    return array


def engine_native_to_canonical(values: np.ndarray) -> np.ndarray:
    array = _six_values(values)
    return array * CONTROLLED_SIGNS.astype(array.dtype, copy=False)


def canonical_to_engine_native(values: np.ndarray) -> np.ndarray:
    return engine_native_to_canonical(values)


def stable_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def stable_payload_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest().upper()


def sha256_array(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(
        stable_json_bytes(
            {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        )
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def create_run_directory(base: str | Path, run_id: str) -> Path:
    if not run_id or any(character in run_id for character in '<>:"/\\|?*'):
        raise ValueError(f"Invalid run id: {run_id!r}")
    output = Path(base) / run_id
    output.mkdir(parents=True, exist_ok=False)
    return output


def _numeric_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError(f"NPZ field {name!r} must use a numeric dtype, got {array.dtype}")
    return array


def save_npz(path: str | Path, arrays: Mapping[str, Any]) -> Path:
    if not arrays:
        raise ValueError("Refusing to save an empty trace")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = {name: _numeric_array(name, value) for name, value in sorted(arrays.items())}
    np.savez_compressed(output, **normalized)
    return output


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def generated_action_sequence_identity(
    values: np.ndarray,
    *,
    source: str,
    key: str,
) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "source": source,
        "file_sha256": None,
        "sidecar_sha256": None,
        "key": key,
        "content_sha256": sha256_array(array),
        "rows": int(array.shape[0]),
        "source_control_trace_sha256": None,
    }


def closed_loop_action_identity() -> dict[str, Any]:
    return {
        "source": "policy_closed_loop",
        "file_sha256": None,
        "sidecar_sha256": None,
        "key": None,
        "content_sha256": None,
        "rows": None,
        "source_control_trace_sha256": None,
    }


def load_action_sequence(
    path: str | Path,
    key: str,
    *,
    requested_ticks: int,
    require_replay_sidecar: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f"Action sequence key {key!r} is not present in {source}")
        sequence = archive[key].copy()
    if sequence.ndim != 2 or sequence.shape[1] != 6:
        raise ValueError(f"Action sequence must have shape (N, 6), got {sequence.shape}")
    if sequence.dtype != np.float32:
        raise TypeError(f"Action sequence must use float32, got {sequence.dtype}")
    if len(sequence) < requested_ticks:
        raise ValueError(
            f"Action sequence has {len(sequence)} rows for {requested_ticks} requested ticks"
        )
    if not np.isfinite(sequence).all():
        raise ValueError("Action sequence contains NaN or Inf")

    file_sha256 = sha256_file(source)
    identity: dict[str, Any] = {
        "source": "npz",
        "file_sha256": file_sha256,
        "sidecar_sha256": None,
        "key": key,
        "content_sha256": sha256_array(sequence),
        "rows": int(len(sequence)),
        "source_control_trace_sha256": None,
    }
    if require_replay_sidecar:
        sidecar_path = source.with_suffix(".json")
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"Replay action sidecar is missing: {sidecar_path}")
        sidecar = load_json(sidecar_path)
        expected = {
            "schema_version": "IsaacPolicyReplayActionsV1",
            "key": key,
            "output_sha256": file_sha256,
        }
        mismatches = {
            name: {"expected": value, "actual": sidecar.get(name)}
            for name, value in expected.items()
            if sidecar.get(name) != value
        }
        if int(sidecar.get("control_ticks", -1)) != len(sequence):
            mismatches["control_ticks"] = {
                "expected": int(len(sequence)),
                "actual": sidecar.get("control_ticks"),
            }
        source_trace_hash = sidecar.get("source_control_trace_sha256")
        if not isinstance(source_trace_hash, str) or len(source_trace_hash) != 64:
            mismatches["source_control_trace_sha256"] = {
                "expected": "64-character SHA256",
                "actual": source_trace_hash,
            }
        if mismatches:
            raise ValueError(f"Replay action sidecar mismatch: {mismatches}")
        identity.update(
            {
                "source": "isaac_policy_replay_npz",
                "sidecar_sha256": sha256_file(sidecar_path),
                "source_control_trace_sha256": source_trace_hash,
            }
        )
    return sequence, identity


def build_file_hashes(paths: Mapping[str, str | Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def verify_file_hashes(paths: Mapping[str, str | Path], expected: Mapping[str, str]) -> None:
    actual = build_file_hashes(paths)
    if actual != dict(expected):
        mismatches = {
            name: {"expected": expected.get(name), "actual": actual.get(name)}
            for name in sorted(set(actual) | set(expected))
            if actual.get(name) != expected.get(name)
        }
        raise ValueError(f"File hash mismatch: {mismatches}")


def _require_rows(trace: Mapping[str, np.ndarray], fields: Mapping[str, int]) -> int:
    missing = sorted(set(fields) - set(trace))
    if missing:
        raise ValueError(f"Trace is missing required fields: {missing}")
    rows: int | None = None
    for name, width in fields.items():
        array = _numeric_array(name, trace[name])
        expected_ndim = 1 if width == 1 else 2
        if array.ndim != expected_ndim or (width != 1 and array.shape[1] != width):
            raise ValueError(f"Trace field {name!r} has invalid shape {array.shape}; expected (*,{width})")
        current_rows = array.shape[0]
        if rows is None:
            rows = current_rows
        elif current_rows != rows:
            raise ValueError(f"Trace field {name!r} has {current_rows} rows; expected {rows}")
    return int(rows or 0)


def validate_substep_trace(
    trace: Mapping[str, np.ndarray],
    *,
    physics_steps_per_action: int,
    continuity_atol: float,
) -> None:
    if physics_steps_per_action <= 0:
        raise ValueError("physics_steps_per_action must be positive")
    fields = {
        "control_tick": 1,
        "substep_index": 1,
        "active_joint_position_canonical_pre_step": 6,
        "active_joint_velocity_canonical_pre_step": 6,
        "active_joint_position_canonical_post_step": 6,
        "active_joint_velocity_canonical_post_step": 6,
        "all_hinge_position_named_pre_step": 26,
        "all_hinge_velocity_named_pre_step": 26,
        "all_hinge_position_named_post_step": 26,
        "all_hinge_velocity_named_post_step": 26,
    }
    rows = _require_rows(trace, fields)
    if rows == 0 or rows % physics_steps_per_action:
        raise ValueError(f"Substep row count {rows} is not divisible by {physics_steps_per_action}")
    expected_substeps = np.tile(np.arange(physics_steps_per_action), rows // physics_steps_per_action)
    expected_ticks = np.repeat(np.arange(rows // physics_steps_per_action), physics_steps_per_action)
    if not np.array_equal(np.asarray(trace["substep_index"]), expected_substeps):
        raise ValueError("Substep indices are missing, duplicated, or out of order")
    if not np.array_equal(np.asarray(trace["control_tick"]), expected_ticks):
        raise ValueError("Control ticks are missing, duplicated, or out of order")

    continuity_pairs = (
        ("active_joint_position_canonical_post_step", "active_joint_position_canonical_pre_step"),
        ("active_joint_velocity_canonical_post_step", "active_joint_velocity_canonical_pre_step"),
        ("all_hinge_position_named_post_step", "all_hinge_position_named_pre_step"),
        ("all_hinge_velocity_named_post_step", "all_hinge_velocity_named_pre_step"),
    )
    for post_name, pre_name in continuity_pairs:
        post = np.asarray(trace[post_name])[:-1]
        following_pre = np.asarray(trace[pre_name])[1:]
        if not np.allclose(post, following_pre, rtol=0.0, atol=continuity_atol, equal_nan=True):
            maximum = float(np.nanmax(np.abs(post - following_pre)))
            raise ValueError(f"substep continuity failed for {post_name}: max gap {maximum:.9g}")


def validate_control_trace(
    trace: Mapping[str, np.ndarray],
    *,
    policy_observation_dimension: int = 25,
) -> None:
    if policy_observation_dimension not in (25, 125):
        raise ValueError("policy_observation_dimension must be 25 or 125")
    fields = {
        "control_tick": 1,
        "actor_obs_policy_pre_step": policy_observation_dimension,
        "actor_output_raw": 6,
        "action_clipped": 6,
        "target_command_canonical": 6,
        "target_command_engine_native": 6,
        "active_joint_position_canonical_post_step_pre_reset": 6,
        "active_joint_velocity_canonical_post_step_pre_reset": 6,
        "common_diagnostic_flags_int8": 8,
        "next_actor_obs_policy_returned": policy_observation_dimension,
        "next_obs_is_reset_int8": 1,
    }
    rows = _require_rows(trace, fields)
    if not np.array_equal(np.asarray(trace["control_tick"]), np.arange(rows)):
        raise ValueError("Control ticks are missing, duplicated, or out of order")
    canonical = np.asarray(trace["target_command_canonical"])
    native = np.asarray(trace["target_command_engine_native"])
    if not np.allclose(canonical_to_engine_native(canonical), native, rtol=0.0, atol=0.0):
        raise ValueError("Canonical/native target round-trip failed")
