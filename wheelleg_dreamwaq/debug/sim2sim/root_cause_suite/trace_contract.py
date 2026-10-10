from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import SCHEMA_VERSION, canonical_json_bytes, sha256_file, stable_hash


TRACE_SCHEMA_VERSION = "RootCauseTraceV1"
COMMON_TIMES_MS = np.asarray([0, 5, 10, 15, 20, 40, 100, 200, 400], dtype=np.float64)


@dataclass(frozen=True)
class FieldSpec:
    unit: str
    frame: str
    phase: str
    reference: str
    availability: str = "available"
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("unit", "frame", "phase", "reference"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"FieldSpec.{name} must be a non-empty string")
        if self.availability not in {"available", "unavailable"}:
            raise ValueError(f"Unknown availability: {self.availability}")
        if self.availability == "available" and self.unavailable_reason is not None:
            raise ValueError("Available fields cannot have an unavailable reason")
        if self.availability == "unavailable" and not self.unavailable_reason:
            raise ValueError("Unavailable fields require an explicit reason")


@dataclass(frozen=True)
class Availability:
    available: bool
    value: Any
    reason: str | None = None


@dataclass(frozen=True)
class TraceIdentity:
    trace_sha256: str
    metadata_sha256: str
    schema_hash: str
    row_count: int


@dataclass(frozen=True)
class VerifiedTrace:
    arrays: Mapping[str, np.ndarray]
    fields: Mapping[str, FieldSpec]
    identity: TraceIdentity
    metadata: Mapping[str, Any]


def normalize_availability(
    value: object, *, sentinel: object | None = None
) -> Availability:
    if isinstance(value, Availability):
        return value
    if sentinel is not None:
        try:
            matches = bool(np.array_equal(value, sentinel))
        except (TypeError, ValueError):
            matches = value == sentinel
        if matches:
            return Availability(False, None, "sentinel")
    if value is None:
        return Availability(False, None, "explicit_none")
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError("Empty arrays are not an availability marker")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("NaN or Inf cannot be inferred as unavailable")
    return Availability(True, value, None)


def validate_common_times(times_s: np.ndarray) -> None:
    times = np.asarray(times_s, dtype=np.float64)
    expected = COMMON_TIMES_MS / 1000.0
    if times.ndim != 1 or times.shape != expected.shape:
        raise ValueError(f"Common times must have shape {expected.shape}")
    if not np.allclose(times, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError("Common time samples do not match the frozen schedule")


def _validate_arrays(
    arrays: Mapping[str, np.ndarray], fields: Mapping[str, FieldSpec]
) -> tuple[dict[str, np.ndarray], int]:
    if not arrays:
        raise ValueError("A trace must contain at least one array")
    if set(arrays) != set(fields):
        missing = sorted(set(arrays) - set(fields))
        extra = sorted(set(fields) - set(arrays))
        raise ValueError(f"Trace field metadata mismatch: missing={missing}, extra={extra}")
    normalized: dict[str, np.ndarray] = {}
    row_count: int | None = None
    for name in sorted(arrays):
        if not isinstance(name, str) or not name:
            raise ValueError("Trace field names must be non-empty strings")
        field = fields[name]
        if not isinstance(field, FieldSpec):
            raise TypeError(f"Field metadata for {name} is not FieldSpec")
        value = np.asarray(arrays[name])
        if value.ndim == 0:
            value = value.reshape(1)
        if value.dtype == object:
            raise ValueError(f"Object arrays are forbidden: {name}")
        if field.availability == "available":
            if value.shape[0] == 0:
                raise ValueError(f"Available trace field is empty: {name}")
            if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
                raise ValueError(f"Available trace field contains NaN/Inf: {name}")
            if row_count is None:
                row_count = int(value.shape[0])
            elif value.shape[0] != row_count:
                raise ValueError(f"Inconsistent trace row count for {name}")
        elif value.shape[0] != 0:
            raise ValueError(f"Unavailable trace field must use an empty array: {name}")
        normalized[name] = value
    if row_count is None:
        raise ValueError("A trace must contain at least one available field")
    return normalized, row_count


def write_trace(
    directory: Path,
    arrays: Mapping[str, np.ndarray],
    fields: Mapping[str, FieldSpec],
) -> TraceIdentity:
    normalized, row_count = _validate_arrays(arrays, fields)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    npz_path = destination / "trace.npz"
    np.savez(npz_path, **{name: normalized[name] for name in sorted(normalized)})
    trace_sha256 = sha256_file(npz_path)
    field_payload = {name: asdict(fields[name]) for name in sorted(fields)}
    schema_hash = stable_hash(field_payload)
    metadata: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "run_schema_version": SCHEMA_VERSION,
        "trace_file": npz_path.name,
        "trace_sha256": trace_sha256,
        "row_count": row_count,
        "array_shapes": {name: list(normalized[name].shape) for name in sorted(normalized)},
        "array_dtypes": {name: str(normalized[name].dtype) for name in sorted(normalized)},
        "fields": field_payload,
        "schema_hash": schema_hash,
    }
    metadata_path = destination / "trace.json"
    metadata_path.write_bytes(canonical_json_bytes(metadata) + b"\n")
    return TraceIdentity(
        trace_sha256=trace_sha256,
        metadata_sha256=sha256_file(metadata_path),
        schema_hash=schema_hash,
        row_count=row_count,
    )


def load_verified_trace(directory: Path) -> VerifiedTrace:
    source = Path(directory)
    metadata_path = source / "trace.json"
    npz_path = source / "trace.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError("Trace schema version mismatch")
    actual_trace_hash = sha256_file(npz_path)
    if actual_trace_hash != metadata.get("trace_sha256"):
        raise ValueError("Trace NPZ hash mismatch")
    raw_fields = metadata.get("fields")
    if not isinstance(raw_fields, dict):
        raise ValueError("Trace metadata fields are missing")
    fields = {name: FieldSpec(**payload) for name, payload in raw_fields.items()}
    if stable_hash(raw_fields) != metadata.get("schema_hash"):
        raise ValueError("Trace schema hash mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    normalized, row_count = _validate_arrays(arrays, fields)
    if row_count != int(metadata.get("row_count", -1)):
        raise ValueError("Trace row count mismatch")
    shapes = {name: list(value.shape) for name, value in sorted(normalized.items())}
    dtypes = {name: str(value.dtype) for name, value in sorted(normalized.items())}
    if shapes != metadata.get("array_shapes") or dtypes != metadata.get("array_dtypes"):
        raise ValueError("Trace array layout does not match sidecar")
    identity = TraceIdentity(
        trace_sha256=actual_trace_hash,
        metadata_sha256=sha256_file(metadata_path),
        schema_hash=metadata["schema_hash"],
        row_count=row_count,
    )
    return VerifiedTrace(normalized, fields, identity, metadata)
