from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .trace_contract import VerifiedTrace, load_verified_trace
from .contracts import FROZEN_POLICIES, PROJECT_ROOT, sha256_file, stable_hash
from .repeatability import frozen_repeat_envelope


COMMON_TIMES_MS = np.asarray([0, 5, 10, 15, 20, 40, 100, 200, 400], dtype=np.float64)


def _frozen_envelope(
    snapshot: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    field: str,
    *,
    indices: np.ndarray | None = None,
    observed: float,
) -> float:
    if snapshot is None and result is None:
        return float(observed)
    if snapshot is None or result is None:
        raise ValueError("Frozen repeatability requires both snapshot and worker result")
    records = result.get("repeatability_records")
    if not isinstance(records, list):
        raise ValueError("Worker result has no repeatability records")
    selected = records
    if indices is not None:
        selected = [records[int(index)] for index in np.flatnonzero(indices)]
    return frozen_repeat_envelope(snapshot, selected, field)


def effective_tolerance(material_floor: float, repeat_envelope: float) -> float:
    if material_floor < 0.0 or repeat_envelope < 0.0:
        raise ValueError("Tolerances must be non-negative")
    return float(max(material_floor, 5.0 * repeat_envelope))


def odd_even_response(
    plus: np.ndarray, zero: np.ndarray, minus: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    plus_array = np.asarray(plus, dtype=np.float64)
    zero_array = np.asarray(zero, dtype=np.float64)
    minus_array = np.asarray(minus, dtype=np.float64)
    if plus_array.shape != zero_array.shape or plus_array.shape != minus_array.shape:
        raise ValueError("Plus, zero, and minus responses must have identical shapes")
    odd = 0.5 * (plus_array - minus_array)
    even = 0.5 * (plus_array + minus_array) - zero_array
    return odd, even


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.size == 0:
        raise ValueError("RMSE inputs must be non-empty with identical shapes")
    return float(np.sqrt(np.mean(np.square(lhs - rhs))))


def normalized_rmse(
    left: np.ndarray, right: np.ndarray, *, minimum_scale: float
) -> float:
    if minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be positive")
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.size == 0:
        raise ValueError("Normalized RMSE inputs must be non-empty with identical shapes")
    scale = max(
        float(minimum_scale),
        float(np.sqrt(np.mean(np.square(lhs)))),
        float(np.sqrt(np.mean(np.square(rhs)))),
    )
    return rmse(lhs, rhs) / scale


def response_mismatch_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_repeat_envelope: float,
    right_repeat_envelope: float,
    minimum_scale: float = 1.0,
    material_floor: float = 1.0e-2,
) -> dict[str, float | bool]:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.size == 0:
        raise ValueError("Response inputs must be non-empty with identical shapes")
    scale = max(
        float(minimum_scale),
        float(np.sqrt(np.mean(np.square(lhs)))),
        float(np.sqrt(np.mean(np.square(rhs)))),
    )
    mismatch_rmse = rmse(lhs, rhs)
    repeat_envelope = max(float(left_repeat_envelope), float(right_repeat_envelope))
    tolerance = effective_tolerance(material_floor, repeat_envelope)
    return {
        "rmse": mismatch_rmse,
        "normalized_rmse": mismatch_rmse / scale,
        "normalization_scale": scale,
        "max_abs": float(np.max(np.abs(lhs - rhs))),
        "repeat_envelope": repeat_envelope,
        "normalized_repeat_envelope": repeat_envelope / scale,
        "effective_tolerance": tolerance,
        "normalized_effective_tolerance": tolerance / scale,
        "material": bool(mismatch_rmse > tolerance),
    }


def first_persistent_divergence(mask: np.ndarray, *, width: int = 3) -> int | None:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("Divergence mask must be one-dimensional")
    if width <= 0:
        raise ValueError("Persistence width must be positive")
    if values.size < width:
        return None
    for index in range(values.size - width + 1):
        if bool(np.all(values[index : index + width])):
            return index
    return None


def explanation_ratio(
    mismatch_before: float, mismatch_after: float, *, tolerance: float
) -> float | None:
    if mismatch_before < 0.0 or mismatch_after < 0.0 or tolerance < 0.0:
        raise ValueError("Mismatch and tolerance values must be non-negative")
    if mismatch_before <= tolerance:
        return None
    return float(1.0 - mismatch_after / mismatch_before)


def sample_common_times(
    times_s: np.ndarray,
    values: np.ndarray,
    *,
    common_times_ms: np.ndarray = COMMON_TIMES_MS,
) -> np.ndarray:
    times = np.asarray(times_s, dtype=np.float64)
    samples = np.asarray(values, dtype=np.float64)
    targets = np.asarray(common_times_ms, dtype=np.float64) / 1000.0
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("Source times must be a strictly increasing vector")
    if samples.ndim == 0 or samples.shape[0] != times.size:
        raise ValueError("Value rows must match source times")
    if targets.ndim != 1 or np.any(np.diff(targets) <= 0.0):
        raise ValueError("Target times must be strictly increasing")
    if targets[0] < times[0] - 1.0e-12 or targets[-1] > times[-1] + 1.0e-12:
        raise ValueError("Common times fall outside the source trace")
    flat = samples.reshape(samples.shape[0], -1)
    interpolated = np.column_stack(
        [np.interp(targets, times, flat[:, column]) for column in range(flat.shape[1])]
    )
    return interpolated.reshape((targets.size,) + samples.shape[1:])


def rms_gain(baseline: np.ndarray, perturbed: np.ndarray, feature_rms: float) -> float:
    if feature_rms <= 1.0e-6:
        raise ValueError("Feature RMS does not pass the C70 eligibility floor")
    base = np.asarray(baseline, dtype=np.float64)
    changed = np.asarray(perturbed, dtype=np.float64)
    if base.shape != changed.shape or base.size == 0:
        raise ValueError("Action vectors must be non-empty with identical shapes")
    return float(np.sqrt(np.mean(np.square(changed - base))) / feature_rms)


def median_valid(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not finite else float(np.median(np.asarray(finite)))


def _trace_time_vector(trace: VerifiedTrace) -> np.ndarray:
    values = np.asarray(trace.arrays["time_s"], dtype=np.float64)
    if values.ndim == 1:
        return values
    if values.ndim != 2 or not np.allclose(values, values[:, :1], rtol=0.0, atol=1.0e-12):
        raise ValueError("Probe time must be one shared vector across profiles")
    return values[:, 0]


def _profile_vector(trace: VerifiedTrace, name: str) -> np.ndarray:
    values = np.asarray(trace.arrays[name])
    if values.ndim == 1:
        return values
    if values.ndim != 2 or not np.array_equal(values, np.broadcast_to(values[:1], values.shape)):
        raise ValueError(f"Profile field {name} changes over time")
    return values[0]


def sample_probe_field(
    trace: VerifiedTrace,
    field: str,
    *,
    common_times_ms: np.ndarray,
) -> np.ndarray:
    if field not in trace.arrays:
        raise KeyError(field)
    return sample_common_times(
        _trace_time_vector(trace),
        np.asarray(trace.arrays[field]),
        common_times_ms=np.asarray(common_times_ms, dtype=np.float64),
    )


def p30_odd_response(
    trace: VerifiedTrace,
    *,
    field: str = "controlled_velocity_canonical",
    common_times_ms: np.ndarray = np.asarray([0, 5, 10, 15, 20, 40, 100]),
) -> tuple[np.ndarray, float]:
    channels = _profile_vector(trace, "profile_channel").astype(np.int64)
    signs = _profile_vector(trace, "profile_sign").astype(np.int64)
    repetitions = _profile_vector(trace, "profile_repetition").astype(np.int64)
    values = sample_probe_field(trace, field, common_times_ms=common_times_ms)
    responses = []
    repeat_spreads: list[float] = []
    for channel in range(6):
        means: dict[int, np.ndarray] = {}
        for sign in (1, 0, -1):
            mask = (channels == channel) & (signs == sign)
            selected = values[:, mask]
            selected_repetitions = repetitions[mask]
            if selected.shape[1] < 3 or len(set(selected_repetitions.tolist())) != selected.shape[1]:
                raise ValueError(f"P30 channel {channel} sign {sign} lacks unique repetitions")
            means[sign] = np.mean(selected, axis=1)
            repeat_spreads.append(float(np.max(np.ptp(selected, axis=1))))
        odd, _ = odd_even_response(means[1], means[0], means[-1])
        responses.append(odd)
    return np.stack(responses, axis=1), max(repeat_spreads, default=0.0)


def compare_p30_trace_directories(
    isaac_directory: Path,
    mujoco_directory: Path,
    *,
    repeatability_snapshot: Mapping[str, Any] | None = None,
    isaac_result: Mapping[str, Any] | None = None,
    mujoco_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    isaac = load_verified_trace(Path(isaac_directory))
    mujoco = load_verified_trace(Path(mujoco_directory))
    isaac_odd, isaac_repeat = p30_odd_response(isaac)
    mujoco_odd, mujoco_repeat = p30_odd_response(mujoco)
    isaac_frozen = _frozen_envelope(
        repeatability_snapshot,
        isaac_result,
        "controlled_velocity_canonical",
        observed=isaac_repeat,
    )
    mujoco_frozen = _frozen_envelope(
        repeatability_snapshot,
        mujoco_result,
        "controlled_velocity_canonical",
        observed=mujoco_repeat,
    )
    metrics = response_mismatch_metrics(
        isaac_odd,
        mujoco_odd,
        left_repeat_envelope=isaac_frozen,
        right_repeat_envelope=mujoco_frozen,
    )
    return {
        "schema_version": "RootCauseP30ComparisonV1",
        "common_times_ms": [0, 5, 10, 15, 20, 40, 100],
        **metrics,
        "isaac_repeat_envelope": isaac_frozen,
        "mujoco_repeat_envelope": mujoco_frozen,
        "isaac_observed_repeat_envelope": isaac_repeat,
        "mujoco_observed_repeat_envelope": mujoco_repeat,
        "isaac_odd_response": isaac_odd.tolist(),
        "mujoco_odd_response": mujoco_odd.tolist(),
        "isaac_trace_sha256": isaac.identity.trace_sha256,
        "mujoco_trace_sha256": mujoco.identity.trace_sha256,
    }


def p40_odd_response(
    trace: VerifiedTrace,
    *,
    field: str = "all_hinge_velocity",
    common_times_ms: np.ndarray = np.asarray([0, 5, 10, 15, 20, 40, 100]),
) -> tuple[np.ndarray, float]:
    channels = _profile_vector(trace, "profile_channel").astype(np.int64)
    signs = _profile_vector(trace, "profile_sign").astype(np.int64)
    repetitions = _profile_vector(trace, "profile_repetition").astype(np.int64)
    if not np.all(channels == -1):
        raise ValueError("P40 expects the frozen symmetric multi-joint profile")
    values = sample_probe_field(trace, field, common_times_ms=common_times_ms)
    means: dict[int, np.ndarray] = {}
    repeat_spreads: list[float] = []
    for sign in (1, 0, -1):
        mask = signs == sign
        selected = values[:, mask]
        selected_repetitions = repetitions[mask]
        if selected.shape[1] < 3 or len(set(selected_repetitions.tolist())) != selected.shape[1]:
            raise ValueError(f"P40 sign {sign} lacks unique repetitions")
        means[sign] = np.mean(selected, axis=1)
        repeat_spreads.append(float(np.max(np.ptp(selected, axis=1))))
    odd, _ = odd_even_response(means[1], means[0], means[-1])
    return odd, max(repeat_spreads, default=0.0)


def compare_p40_trace_directories(
    isaac_on_directory: Path,
    isaac_off_directory: Path,
    mujoco_on_directory: Path,
    mujoco_off_directory: Path,
    *,
    repeatability_snapshot: Mapping[str, Any] | None = None,
    results: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    traces = {
        "isaac_on": load_verified_trace(Path(isaac_on_directory)),
        "isaac_off": load_verified_trace(Path(isaac_off_directory)),
        "mujoco_on": load_verified_trace(Path(mujoco_on_directory)),
        "mujoco_off": load_verified_trace(Path(mujoco_off_directory)),
    }
    responses: dict[str, np.ndarray] = {}
    repeat_envelopes: dict[str, float] = {}
    for name, trace in traces.items():
        responses[name], repeat_envelopes[name] = p40_odd_response(trace)
    frozen_envelopes = {
        name: _frozen_envelope(
            repeatability_snapshot,
            None if results is None else results.get(name),
            "all_hinge_velocity",
            observed=repeat_envelopes[name],
        )
        for name in traces
    }

    on_metrics = response_mismatch_metrics(
        responses["isaac_on"],
        responses["mujoco_on"],
        left_repeat_envelope=frozen_envelopes["isaac_on"],
        right_repeat_envelope=frozen_envelopes["mujoco_on"],
    )
    off_metrics = response_mismatch_metrics(
        responses["isaac_off"],
        responses["mujoco_off"],
        left_repeat_envelope=frozen_envelopes["isaac_off"],
        right_repeat_envelope=frozen_envelopes["mujoco_off"],
    )
    isaac_closure_effect = responses["isaac_on"] - responses["isaac_off"]
    mujoco_closure_effect = responses["mujoco_on"] - responses["mujoco_off"]
    closure_effect_mismatch = normalized_rmse(
        isaac_closure_effect, mujoco_closure_effect, minimum_scale=1.0
    )
    return {
        "schema_version": "RootCauseP40ComparisonV1",
        "common_times_ms": [0, 5, 10, 15, 20, 40, 100],
        "closure_on": on_metrics,
        "closure_off": off_metrics,
        "closure_effect_normalized_rmse": closure_effect_mismatch,
        "closure_effect_rmse": rmse(isaac_closure_effect, mujoco_closure_effect),
        "closure_effect_max_abs": float(
            np.max(np.abs(isaac_closure_effect - mujoco_closure_effect))
        ),
        "explanation_ratio": explanation_ratio(
            float(on_metrics["normalized_rmse"]),
            float(off_metrics["normalized_rmse"]),
            tolerance=float(on_metrics["normalized_effective_tolerance"]),
        ),
        "repeat_envelopes": frozen_envelopes,
        "observed_repeat_envelopes": repeat_envelopes,
        "repeat_envelope": max(frozen_envelopes.values(), default=0.0),
        "responses": {name: value.tolist() for name, value in responses.items()},
        "closure_effects": {
            "isaac": isaac_closure_effect.tolist(),
            "mujoco": mujoco_closure_effect.tolist(),
        },
        "trace_sha256": {
            name: trace.identity.trace_sha256 for name, trace in traces.items()
        },
    }


def _median_profile_trace(values: np.ndarray) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[1] < 3:
        raise ValueError("Repeatability analysis requires a profile axis with at least three rows")
    return np.median(array, axis=1), float(np.max(np.ptp(array, axis=1)))


def compare_p10_trace_directories(
    isaac_directory: Path,
    mujoco_directory: Path,
    *,
    repeatability_snapshot: Mapping[str, Any] | None = None,
    isaac_result: Mapping[str, Any] | None = None,
    mujoco_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    common = np.asarray([0, 5, 10, 15, 20, 40, 100], dtype=np.float64)
    isaac = load_verified_trace(Path(isaac_directory))
    mujoco = load_verified_trace(Path(mujoco_directory))
    signals: dict[str, Any] = {}
    maximum_repeat = 0.0
    for field in (
        "linear_momentum_control",
        "angular_momentum_com_control",
        "kinetic_energy_j",
        "all_hinge_velocity",
        "closure_residual_m",
        "system_com_position_control",
    ):
        isaac_values, isaac_repeat = _median_profile_trace(
            sample_probe_field(isaac, field, common_times_ms=common)
        )
        mujoco_values, mujoco_repeat = _median_profile_trace(
            sample_probe_field(mujoco, field, common_times_ms=common)
        )
        if field in {
            "all_hinge_velocity",
            "closure_residual_m",
            "system_com_position_control",
        }:
            isaac_frozen = _frozen_envelope(
                repeatability_snapshot,
                isaac_result,
                field,
                observed=isaac_repeat,
            )
            mujoco_frozen = _frozen_envelope(
                repeatability_snapshot,
                mujoco_result,
                field,
                observed=mujoco_repeat,
            )
        else:
            isaac_frozen = isaac_repeat
            mujoco_frozen = mujoco_repeat
        maximum_repeat = max(maximum_repeat, isaac_frozen, mujoco_frozen)
        signals[field] = {
            "rmse": rmse(isaac_values, mujoco_values),
            "max_abs": float(np.max(np.abs(isaac_values - mujoco_values))),
            "isaac_repeat_envelope": isaac_frozen,
            "mujoco_repeat_envelope": mujoco_frozen,
            "isaac_observed_repeat_envelope": isaac_repeat,
            "mujoco_observed_repeat_envelope": mujoco_repeat,
            "isaac": isaac_values.tolist(),
            "mujoco": mujoco_values.tolist(),
        }
    mass = 4.396253988146782
    isaac_momentum = np.asarray(signals["linear_momentum_control"]["isaac"])
    mujoco_momentum = np.asarray(signals["linear_momentum_control"]["mujoco"])
    duration = (common[-1] - common[0]) / 1000.0
    isaac_accel_z = float((isaac_momentum[-1, 2] - isaac_momentum[0, 2]) / mass / duration)
    mujoco_accel_z = float((mujoco_momentum[-1, 2] - mujoco_momentum[0, 2]) / mass / duration)
    return {
        "schema_version": "RootCauseP10ComparisonV1",
        "common_times_ms": common.astype(int).tolist(),
        "signals": signals,
        "maximum_repeat_envelope": maximum_repeat,
        "com_acceleration_z": {
            "isaac": isaac_accel_z,
            "mujoco": mujoco_accel_z,
            "absolute_difference": abs(isaac_accel_z - mujoco_accel_z),
        },
        "isaac_trace_sha256": isaac.identity.trace_sha256,
        "mujoco_trace_sha256": mujoco.identity.trace_sha256,
    }


P50_IMPACT_OFFSETS_MS = np.asarray(
    [-20, -15, -10, -5, 0, 5, 10, 15, 20, 40, 60, 80, 100],
    dtype=np.float64,
)
P50_SLIDE_TIMES_MS = np.asarray(
    [0, 5, 10, 15, 20, 40, 60, 80, 100, 150, 200, 300, 400],
    dtype=np.float64,
)


def _sphere_profile_vectors(trace: VerifiedTrace) -> dict[str, np.ndarray]:
    return {
        name: _profile_vector(trace, name)
        for name in (
            "profile_height_m",
            "profile_vertical_velocity_mps",
            "profile_horizontal_velocity_mps",
            "profile_sign",
            "profile_repetition",
        )
    }


def _first_contact_times(trace: VerifiedTrace) -> np.ndarray:
    times = _trace_time_vector(trace)
    contact = np.asarray(trace.arrays["contact_count"])
    if contact.ndim != 2 or contact.shape[0] != times.size:
        raise ValueError("Sphere contact trace must have shape [time, profile]")
    result = np.full(contact.shape[1], np.nan, dtype=np.float64)
    for profile in range(contact.shape[1]):
        indices = np.flatnonzero(contact[:, profile] > 0)
        if indices.size:
            result[profile] = times[int(indices[0])]
    return result


def _contact_aligned_velocity_z(
    trace: VerifiedTrace,
    *,
    offsets_ms: np.ndarray = P50_IMPACT_OFFSETS_MS,
) -> tuple[np.ndarray, np.ndarray]:
    times = _trace_time_vector(trace)
    velocity = np.asarray(trace.arrays["com_velocity_world"], dtype=np.float64)
    if velocity.ndim != 3 or velocity.shape[0] != times.size or velocity.shape[2] != 3:
        raise ValueError("Sphere COM velocity must have shape [time, profile, 3]")
    contact_times = _first_contact_times(trace)
    aligned = np.full((len(offsets_ms), velocity.shape[1]), np.nan, dtype=np.float64)
    for profile, contact_time in enumerate(contact_times):
        if not np.isfinite(contact_time):
            continue
        targets = contact_time + np.asarray(offsets_ms, dtype=np.float64) / 1000.0
        if targets[0] < times[0] - 1.0e-12 or targets[-1] > times[-1] + 1.0e-12:
            continue
        aligned[:, profile] = np.interp(targets, times, velocity[:, profile, 2])
    return aligned, contact_times


def _impact_condition_summary(
    trace: VerifiedTrace,
    *,
    repeatability_snapshot: Mapping[str, Any] | None = None,
    worker_result: Mapping[str, Any] | None = None,
) -> dict[tuple[float, float], dict[str, Any]]:
    profiles = _sphere_profile_vectors(trace)
    aligned, contacts = _contact_aligned_velocity_z(trace)
    result: dict[tuple[float, float], dict[str, Any]] = {}
    keys = sorted(
        {
            (float(height), float(vertical))
            for height, vertical in zip(
                profiles["profile_height_m"],
                profiles["profile_vertical_velocity_mps"],
                strict=True,
            )
        }
    )
    for key in keys:
        mask = np.isclose(profiles["profile_height_m"], key[0]) & np.isclose(
            profiles["profile_vertical_velocity_mps"], key[1]
        )
        repetitions = profiles["profile_repetition"][mask].astype(np.int64)
        values = aligned[:, mask]
        event_values = contacts[mask]
        valid = (
            values.shape[1] >= 3
            and len(set(repetitions.tolist())) == values.shape[1]
            and np.isfinite(values).all()
            and np.isfinite(event_values).all()
        )
        observed_velocity_repeat = (
            float(np.max(np.ptp(values, axis=1))) if valid else None
        )
        observed_contact_repeat = (
            float(np.ptp(event_values)) if valid else None
        )
        velocity_repeat = (
            _frozen_envelope(
                repeatability_snapshot,
                worker_result,
                "com_velocity_world",
                indices=mask,
                observed=float(observed_velocity_repeat),
            )
            if valid
            else None
        )
        contact_repeat = (
            _frozen_envelope(
                repeatability_snapshot,
                worker_result,
                "first_contact_time_s",
                indices=mask,
                observed=float(observed_contact_repeat),
            )
            if valid
            else None
        )
        result[key] = {
            "valid": bool(valid),
            "median_velocity_z": np.median(values, axis=1) if valid else None,
            "velocity_repeat_envelope": velocity_repeat,
            "observed_velocity_repeat_envelope": observed_velocity_repeat,
            "median_contact_time_s": (
                float(np.median(event_values)) if valid else None
            ),
            "contact_time_repeat_envelope_s": contact_repeat,
            "observed_contact_time_repeat_envelope_s": observed_contact_repeat,
        }
    return result


def compare_p50_impact_trace_directories(
    isaac_directory: Path,
    mujoco_directory: Path,
    *,
    repeatability_snapshot: Mapping[str, Any] | None = None,
    isaac_result: Mapping[str, Any] | None = None,
    mujoco_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    isaac = load_verified_trace(Path(isaac_directory))
    mujoco = load_verified_trace(Path(mujoco_directory))
    summaries = {
        "isaac": _impact_condition_summary(
            isaac,
            repeatability_snapshot=repeatability_snapshot,
            worker_result=isaac_result,
        ),
        "mujoco": _impact_condition_summary(
            mujoco,
            repeatability_snapshot=repeatability_snapshot,
            worker_result=mujoco_result,
        ),
    }
    if set(summaries["isaac"]) != set(summaries["mujoco"]):
        raise ValueError("Isaac and MuJoCo impact conditions differ")
    pre_mask = P50_IMPACT_OFFSETS_MS < 0.0
    post_mask = ~pre_mask
    conditions: list[dict[str, Any]] = []
    material_count = 0
    for key in sorted(summaries["isaac"]):
        left = summaries["isaac"][key]
        right = summaries["mujoco"][key]
        valid = bool(left["valid"] and right["valid"])
        if valid:
            left_velocity = np.asarray(left["median_velocity_z"])
            right_velocity = np.asarray(right["median_velocity_z"])
            left_repeat = float(left["velocity_repeat_envelope"])
            right_repeat = float(right["velocity_repeat_envelope"])
            precontact = response_mismatch_metrics(
                left_velocity[pre_mask],
                right_velocity[pre_mask],
                left_repeat_envelope=left_repeat,
                right_repeat_envelope=right_repeat,
                material_floor=0.02,
            )
            postcontact = response_mismatch_metrics(
                left_velocity[post_mask],
                right_velocity[post_mask],
                left_repeat_envelope=left_repeat,
                right_repeat_envelope=right_repeat,
                material_floor=0.02,
            )
            event_repeat = max(
                float(left["contact_time_repeat_envelope_s"]),
                float(right["contact_time_repeat_envelope_s"]),
            )
            event_tolerance = effective_tolerance(0.005, event_repeat)
            event_delta = abs(
                float(left["median_contact_time_s"])
                - float(right["median_contact_time_s"])
            )
            event_guard = event_delta <= event_tolerance
            material = bool(postcontact["material"] and not precontact["material"] and event_guard)
            material_count += int(material)
        else:
            precontact = None
            postcontact = None
            event_delta = None
            event_tolerance = None
            event_guard = False
            material = False
        conditions.append(
            {
                "height_m": key[0],
                "vertical_velocity_mps": key[1],
                "valid": valid,
                "precontact": precontact,
                "postcontact": postcontact,
                "contact_time_delta_s": event_delta,
                "contact_time_effective_tolerance_s": event_tolerance,
                "contact_event_guard": event_guard,
                "material_postcontact_failure": material,
                "isaac": {
                    name: value.tolist() if isinstance(value, np.ndarray) else value
                    for name, value in left.items()
                },
                "mujoco": {
                    name: value.tolist() if isinstance(value, np.ndarray) else value
                    for name, value in right.items()
                },
            }
        )
    return {
        "schema_version": "RootCauseP50ImpactComparisonV1",
        "contact_relative_offsets_ms": P50_IMPACT_OFFSETS_MS.astype(int).tolist(),
        "conditions": conditions,
        "valid_condition_count": sum(int(row["valid"]) for row in conditions),
        "material_condition_count": material_count,
        "normal_primary_supported": material_count >= 2,
        "isaac_trace_sha256": isaac.identity.trace_sha256,
        "mujoco_trace_sha256": mujoco.identity.trace_sha256,
    }


def _slide_odd_response(trace: VerifiedTrace) -> tuple[np.ndarray, float]:
    profiles = _sphere_profile_vectors(trace)
    signs = profiles["profile_sign"].astype(np.int64)
    repetitions = profiles["profile_repetition"].astype(np.int64)
    velocity = sample_probe_field(
        trace, "com_velocity_world", common_times_ms=P50_SLIDE_TIMES_MS
    )[..., 0]
    means: dict[int, np.ndarray] = {}
    spreads: list[float] = []
    for sign in (1, 0, -1):
        mask = signs == sign
        selected = velocity[:, mask]
        selected_repetitions = repetitions[mask]
        if selected.shape[1] < 3 or len(set(selected_repetitions.tolist())) != selected.shape[1]:
            raise ValueError(f"P50 slide sign {sign} lacks unique repetitions")
        means[sign] = np.median(selected, axis=1)
        spreads.append(float(np.max(np.ptp(selected, axis=1))))
    odd, _ = odd_even_response(means[1], means[0], means[-1])
    return odd, max(spreads, default=0.0)


def compare_p50_slide_trace_directories(
    isaac_nominal_directory: Path,
    isaac_zero_directory: Path,
    mujoco_nominal_directory: Path,
    mujoco_zero_directory: Path,
    *,
    repeatability_snapshot: Mapping[str, Any] | None = None,
    results: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    traces = {
        "isaac_nominal": load_verified_trace(Path(isaac_nominal_directory)),
        "isaac_zero": load_verified_trace(Path(isaac_zero_directory)),
        "mujoco_nominal": load_verified_trace(Path(mujoco_nominal_directory)),
        "mujoco_zero": load_verified_trace(Path(mujoco_zero_directory)),
    }
    responses: dict[str, np.ndarray] = {}
    repeats: dict[str, float] = {}
    for name, trace in traces.items():
        responses[name], repeats[name] = _slide_odd_response(trace)
    frozen_repeats = {
        name: _frozen_envelope(
            repeatability_snapshot,
            None if results is None else results.get(name),
            "com_velocity_world",
            observed=repeats[name],
        )
        for name in traces
    }
    nominal = response_mismatch_metrics(
        responses["isaac_nominal"],
        responses["mujoco_nominal"],
        left_repeat_envelope=frozen_repeats["isaac_nominal"],
        right_repeat_envelope=frozen_repeats["mujoco_nominal"],
        material_floor=0.02,
    )
    zero = response_mismatch_metrics(
        responses["isaac_zero"],
        responses["mujoco_zero"],
        left_repeat_envelope=frozen_repeats["isaac_zero"],
        right_repeat_envelope=frozen_repeats["mujoco_zero"],
        material_floor=0.02,
    )
    ratio = explanation_ratio(
        float(nominal["normalized_rmse"]),
        float(zero["normalized_rmse"]),
        tolerance=float(nominal["normalized_effective_tolerance"]),
    )
    return {
        "schema_version": "RootCauseP50SlideComparisonV1",
        "common_times_ms": P50_SLIDE_TIMES_MS.astype(int).tolist(),
        "nominal": nominal,
        "zero_friction": zero,
        "explanation_ratio": ratio,
        "tangential_primary_supported_before_normal_gate": bool(
            nominal["material"] and not zero["material"] and ratio is not None and ratio >= 0.70
        ),
        "repeat_envelopes": frozen_repeats,
        "observed_repeat_envelopes": repeats,
        "responses": {name: value.tolist() for name, value in responses.items()},
        "trace_sha256": {
            name: trace.identity.trace_sha256 for name, trace in traces.items()
        },
    }


P60_COMMON_TIMES_MS = np.asarray([0, 5, 10, 15, 20, 40, 100], dtype=np.float64)
C70_TICKS = (0, 1, 2, 3, 4, 5)
C70_FEATURE_GROUPS = {
    "angular_velocity": (0, 3),
    "projected_gravity": (3, 6),
    "command": (6, 9),
    "leg_position_error": (9, 13),
    "joint_velocity": (13, 19),
    "previous_action": (19, 25),
}


def _median_robot_signal(
    trace: VerifiedTrace,
    field: str,
    *,
    component: int | None = None,
) -> tuple[np.ndarray, float]:
    sampled = sample_probe_field(
        trace,
        field,
        common_times_ms=P60_COMMON_TIMES_MS,
    )
    if component is not None:
        sampled = sampled[..., component]
    return _median_profile_trace(sampled)


def compare_p60_robot_trace_directories(
    isaac_directory: Path,
    mujoco_directory: Path,
    *,
    scenario: str,
    repeatability_snapshot: Mapping[str, Any] | None = None,
    isaac_result: Mapping[str, Any] | None = None,
    mujoco_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if scenario not in {"p60_b", "p60_c"}:
        raise ValueError("P60 robot comparison only supports p60_b or p60_c")
    isaac = load_verified_trace(Path(isaac_directory))
    mujoco = load_verified_trace(Path(mujoco_directory))
    field = (
        "base_angular_velocity_control"
        if scenario == "p60_b"
        else "controlled_velocity_canonical"
    )
    component = 1 if scenario == "p60_b" else None
    isaac_signal, isaac_repeat = _median_robot_signal(
        isaac, field, component=component
    )
    mujoco_signal, mujoco_repeat = _median_robot_signal(
        mujoco, field, component=component
    )
    isaac_frozen = _frozen_envelope(
        repeatability_snapshot,
        isaac_result,
        field,
        observed=isaac_repeat,
    )
    mujoco_frozen = _frozen_envelope(
        repeatability_snapshot,
        mujoco_result,
        field,
        observed=mujoco_repeat,
    )
    metrics = response_mismatch_metrics(
        isaac_signal,
        mujoco_signal,
        left_repeat_envelope=isaac_frozen,
        right_repeat_envelope=mujoco_frozen,
    )
    return {
        "schema_version": "RootCauseP60RobotComparisonV1",
        "scenario": scenario,
        "field": field,
        "component": component,
        "common_times_ms": P60_COMMON_TIMES_MS.astype(int).tolist(),
        **metrics,
        "isaac_repeat_envelope": isaac_frozen,
        "mujoco_repeat_envelope": mujoco_frozen,
        "isaac_observed_repeat_envelope": isaac_repeat,
        "mujoco_observed_repeat_envelope": mujoco_repeat,
        "isaac_signal": isaac_signal.tolist(),
        "mujoco_signal": mujoco_signal.tolist(),
        "isaac_trace_sha256": isaac.identity.trace_sha256,
        "mujoco_trace_sha256": mujoco.identity.trace_sha256,
    }


def compare_p60_replay_trace_directories(
    isaac_directory: Path,
    mujoco_directory: Path,
    *,
    replay_source_identity_hash: str,
    repeatability_snapshot: Mapping[str, Any] | None = None,
    isaac_result: Mapping[str, Any] | None = None,
    mujoco_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    isaac = load_verified_trace(Path(isaac_directory))
    mujoco = load_verified_trace(Path(mujoco_directory))
    rows = np.asarray(C70_TICKS, dtype=np.int64)
    if isaac.identity.row_count <= rows[-1] or mujoco.identity.row_count <= rows[-1]:
        raise ValueError("P60 replay traces do not cover the frozen 0..100 ms ticks")
    isaac_signal = np.asarray(
        isaac.arrays["base_angular_velocity_control_post_step"], dtype=np.float64
    )[rows, 1]
    mujoco_signal = np.asarray(
        mujoco.arrays["base_angular_velocity_control_post_step"], dtype=np.float64
    )[rows, 1]
    isaac_repeat = _frozen_envelope(
        repeatability_snapshot,
        isaac_result,
        "base_angular_velocity_control_post_step",
        observed=0.0,
    )
    mujoco_repeat = _frozen_envelope(
        repeatability_snapshot,
        mujoco_result,
        "base_angular_velocity_control_post_step",
        observed=0.0,
    )
    metrics = response_mismatch_metrics(
        isaac_signal,
        mujoco_signal,
        left_repeat_envelope=isaac_repeat,
        right_repeat_envelope=mujoco_repeat,
    )
    return {
        "schema_version": "RootCauseP60ReplayComparisonV1",
        "replay_source_identity_hash": replay_source_identity_hash,
        "policy_ticks": list(C70_TICKS),
        "times_ms": [20 * tick for tick in C70_TICKS],
        "field": "base_angular_velocity_control_post_step[1]",
        **metrics,
        "isaac_signal": isaac_signal.tolist(),
        "mujoco_signal": mujoco_signal.tolist(),
        "isaac_trace_sha256": isaac.identity.trace_sha256,
        "mujoco_trace_sha256": mujoco.identity.trace_sha256,
    }


def _geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("Geometric mean requires finite positive values")
    return float(np.exp(np.mean(np.log(array))))


def _classify_c70_groups(
    groups: dict[str, dict[str, Any]],
    *,
    saturation_fraction_difference: float,
) -> dict[str, Any]:
    eligible = [row for row in groups.values() if row.get("eligible")]
    ratios = [float(row["gain_ratio"]) for row in eligible]
    differences = [float(row["gain_difference"]) for row in eligible]
    gain_ratio = median_valid(ratios)
    gain_difference = median_valid(differences)
    directions_hold = bool(eligible) and all(
        all(
            float(row["signs"][sign]["dreamwaq_gain"])
            > float(row["signs"][sign]["phase1r_gain"])
            for sign in ("plus", "minus")
        )
        for row in eligible
    )
    amplifier = bool(
        len(eligible) >= 2
        and directions_hold
        and gain_ratio is not None
        and gain_ratio >= 1.50
        and gain_difference is not None
        and gain_difference >= 0.25
    )
    not_distinguished = bool(
        len(eligible) >= 2
        and all(0.80 <= ratio <= 1.25 for ratio in ratios)
        and gain_difference is not None
        and abs(gain_difference) < 0.10
        and abs(saturation_fraction_difference) < 0.05
    )
    classification = (
        "amplifier"
        if amplifier
        else "not_distinguished"
        if not_distinguished
        else "inconclusive"
    )
    return {
        "classification": classification,
        "eligible_group_count": len(eligible),
        "eligible_groups": [name for name, row in groups.items() if row.get("eligible")],
        "gain_ratio": gain_ratio,
        "gain_difference": gain_difference,
        "dreamwaq_gain_higher_both_signs": directions_hold,
        "saturation_fraction_difference": saturation_fraction_difference,
        "threshold_proximity_1_25_to_1_50": bool(
            gain_ratio is not None and 1.25 <= gain_ratio < 1.50
        ),
    }


def _policy_output_payload(output: Any, *, policy_kind: str) -> dict[str, Any]:
    return {
        "raw_action": np.asarray(output.raw_action, dtype=np.float64).tolist(),
        "clipped_action": np.clip(output.raw_action, -1.0, 1.0).tolist(),
        "saturated": (np.abs(output.raw_action) > 1.0 + 1.0e-6).tolist(),
        "estimated_velocity": (
            np.asarray(output.estimated_velocity, dtype=np.float64).tolist()
            if policy_kind == "dreamwaq"
            else None
        ),
        "context_mu": (
            np.asarray(output.context_mu, dtype=np.float64).tolist()
            if policy_kind == "dreamwaq"
            else None
        ),
        "context_logvar": (
            np.asarray(output.context_logvar, dtype=np.float64).tolist()
            if policy_kind == "dreamwaq"
            else None
        ),
        "cenet_availability": (
            "available" if policy_kind == "dreamwaq" else "unavailable_policy_has_no_cenet"
        ),
    }


def _c70_reduction(
    *,
    anchor_history: np.ndarray,
    error_history: np.ndarray,
    dreamwaq: Any,
    phase1r: Any,
    reduction_mode: str,
) -> dict[str, Any]:
    if reduction_mode not in {"full_history", "latest_frame_only"}:
        raise ValueError("Unknown C70 reduction mode")
    groups: dict[str, dict[str, Any]] = {}
    detail: list[dict[str, Any]] = []
    saturation = {"dreamwaq": [], "phase1r": []}
    first_saturation = {"dreamwaq": None, "phase1r": None}
    for group_name, (start, stop) in C70_FEATURE_GROUPS.items():
        sign_rows: dict[str, dict[str, Any]] = {}
        for sign_name, sign_value in (("plus", 1.0), ("minus", -1.0)):
            gains = {"dreamwaq": [], "phase1r": []}
            valid_ticks: list[int] = []
            for tick_index, tick in enumerate(C70_TICKS):
                baseline_history = np.asarray(anchor_history[tick_index], dtype=np.float32)
                measured_error = np.asarray(error_history[tick_index], dtype=np.float32)
                baseline_current = baseline_history.reshape(5, 25)[-1].copy()
                latest_error = measured_error.reshape(5, 25)[-1]
                full_masked = np.zeros((5, 25), dtype=np.float32)
                if reduction_mode == "full_history":
                    full_masked[:, start:stop] = measured_error.reshape(5, 25)[:, start:stop]
                    dream_feature = full_masked[:, start:stop]
                else:
                    full_masked[-1, start:stop] = latest_error[start:stop]
                    dream_feature = latest_error[start:stop]
                ppo_masked = np.zeros(25, dtype=np.float32)
                ppo_masked[start:stop] = latest_error[start:stop]
                dream_feature_rms = float(np.sqrt(np.mean(np.square(dream_feature))))
                ppo_feature_rms = float(
                    np.sqrt(np.mean(np.square(latest_error[start:stop])))
                )
                if dream_feature_rms <= 1.0e-6 or ppo_feature_rms <= 1.0e-6:
                    detail.append(
                        {
                            "group": group_name,
                            "sign": sign_name,
                            "tick": tick,
                            "eligible": False,
                            "reason": "feature_rms_below_floor",
                            "dreamwaq_feature_rms": dream_feature_rms,
                            "phase1r_feature_rms": ppo_feature_rms,
                        }
                    )
                    continue
                dream_baseline = dreamwaq.infer(baseline_history)
                ppo_baseline = phase1r.infer(baseline_current)
                dream_input = baseline_history + sign_value * full_masked.reshape(125)
                ppo_input = baseline_current + sign_value * ppo_masked
                dream_changed = dreamwaq.infer(dream_input.astype(np.float32, copy=False))
                ppo_changed = phase1r.infer(ppo_input.astype(np.float32, copy=False))
                dream_gain = rms_gain(
                    dream_baseline.raw_action,
                    dream_changed.raw_action,
                    dream_feature_rms,
                )
                ppo_gain = rms_gain(
                    ppo_baseline.raw_action,
                    ppo_changed.raw_action,
                    ppo_feature_rms,
                )
                gains["dreamwaq"].append(dream_gain)
                gains["phase1r"].append(ppo_gain)
                valid_ticks.append(tick)
                for policy_name, output in (
                    ("dreamwaq", dream_changed),
                    ("phase1r", ppo_changed),
                ):
                    saturated = np.abs(output.raw_action) > 1.0 + 1.0e-6
                    saturation[policy_name].extend(saturated.tolist())
                    if np.any(saturated) and first_saturation[policy_name] is None:
                        first_saturation[policy_name] = tick
                detail.append(
                    {
                        "group": group_name,
                        "sign": sign_name,
                        "tick": tick,
                        "eligible": True,
                        "dreamwaq_feature_rms": dream_feature_rms,
                        "phase1r_feature_rms": ppo_feature_rms,
                        "dreamwaq_gain": dream_gain,
                        "phase1r_gain": ppo_gain,
                        "dreamwaq_baseline": _policy_output_payload(
                            dream_baseline, policy_kind="dreamwaq"
                        ),
                        "dreamwaq_perturbed": _policy_output_payload(
                            dream_changed, policy_kind="dreamwaq"
                        ),
                        "phase1r_baseline": _policy_output_payload(
                            ppo_baseline, policy_kind="ppo"
                        ),
                        "phase1r_perturbed": _policy_output_payload(
                            ppo_changed, policy_kind="ppo"
                        ),
                    }
                )
            dream_median = median_valid(gains["dreamwaq"])
            ppo_median = median_valid(gains["phase1r"])
            sign_rows[sign_name] = {
                "valid_ticks": valid_ticks,
                "valid_tick_count": len(valid_ticks),
                "dreamwaq_gain": dream_median,
                "phase1r_gain": ppo_median,
            }
        eligible = all(
            sign_rows[sign]["valid_tick_count"] >= 3
            and sign_rows[sign]["dreamwaq_gain"] is not None
            and sign_rows[sign]["phase1r_gain"] is not None
            for sign in ("plus", "minus")
        )
        if eligible:
            ratios = [
                float(sign_rows[sign]["dreamwaq_gain"])
                / max(float(sign_rows[sign]["phase1r_gain"]), 1.0e-6)
                for sign in ("plus", "minus")
            ]
            differences = [
                float(sign_rows[sign]["dreamwaq_gain"])
                - float(sign_rows[sign]["phase1r_gain"])
                for sign in ("plus", "minus")
            ]
            ratio = _geometric_mean(ratios)
            difference = float(np.median(differences))
        else:
            ratio = None
            difference = None
        groups[group_name] = {
            "eligible": eligible,
            "signs": sign_rows,
            "gain_ratio": ratio,
            "gain_difference": difference,
        }
    saturation_fraction = {
        name: float(np.mean(values)) if values else 0.0
        for name, values in saturation.items()
    }
    saturation_difference = (
        saturation_fraction["dreamwaq"] - saturation_fraction["phase1r"]
    )
    classification = _classify_c70_groups(
        groups,
        saturation_fraction_difference=saturation_difference,
    )
    first_support = bool(
        first_saturation["dreamwaq"] is not None
        and (
            first_saturation["phase1r"] is None
            or int(first_saturation["dreamwaq"])
            <= int(first_saturation["phase1r"]) - 1
        )
    )
    return {
        "reduction_mode": reduction_mode,
        "groups": groups,
        "detail": detail,
        "saturation_fraction": saturation_fraction,
        "first_saturation_tick": first_saturation,
        "saturation_amplifier": bool(
            saturation_difference >= 0.10 or first_support
        ),
        **classification,
    }


def analyze_c70_checkpoint_sensitivity(
    isaac_result_path: Path,
    mujoco_result_path: Path,
) -> dict[str, Any]:
    import json

    from debug.sim2sim.dreamwaq_debug_contract import DebugPolicyAdapter

    isaac_result_path = Path(isaac_result_path).resolve(strict=True)
    mujoco_result_path = Path(mujoco_result_path).resolve(strict=True)
    isaac_result = json.loads(isaac_result_path.read_text(encoding="utf-8"))
    mujoco_result = json.loads(mujoco_result_path.read_text(encoding="utf-8"))
    identity_hash = isaac_result.get("replay_source_identity_hash")
    if not identity_hash or mujoco_result.get("replay_source_identity_hash") != identity_hash:
        raise ValueError("C70 replay source identities differ")
    if stable_hash(isaac_result["replay_source_identity"]) != identity_hash:
        raise ValueError("C70 Isaac replay identity is invalid")
    if stable_hash(mujoco_result["replay_source_identity"]) != identity_hash:
        raise ValueError("C70 MuJoCo replay identity is invalid")
    isaac = load_verified_trace(isaac_result_path.parent / "trace")
    mujoco = load_verified_trace(mujoco_result_path.parent / "trace")
    if isaac.identity.trace_sha256 != isaac_result["source_trace_sha256"]:
        raise ValueError("C70 Isaac trace is not the frozen replay source trace")
    if mujoco.identity.trace_sha256 != mujoco_result["trace_sha256"]:
        raise ValueError("C70 MuJoCo trace hash differs from its result")
    isaac_history = np.asarray(
        isaac.arrays["actor_obs_policy_pre_step"], dtype=np.float32
    )[list(C70_TICKS)]
    mujoco_history = np.asarray(
        mujoco.arrays["actor_obs_policy_pre_step"], dtype=np.float32
    )[list(C70_TICKS)]
    if isaac_history.shape != (6, 125) or mujoco_history.shape != (6, 125):
        raise ValueError("C70 replay observations must have shape [6,125]")
    error_history = isaac_history - mujoco_history
    model_manifest = PROJECT_ROOT / "sim2sim/mujoco/model_manifest.json"
    adapters = {
        name: DebugPolicyAdapter(
            actor_path=records["actor"].absolute_path(),
            manifest_path=records["manifest"].absolute_path(),
            model_manifest_path=model_manifest,
            load_mujoco_contract=False,
        )
        for name, records in FROZEN_POLICIES.items()
    }
    if adapters["dreamwaq_run01"].policy_kind != "dreamwaq":
        raise RuntimeError("Frozen DreamWaQ adapter kind changed")
    if adapters["phase1r_run03"].policy_kind != "ppo":
        raise RuntimeError("Frozen Phase 1R adapter kind changed")
    anchors: dict[str, Any] = {}
    for anchor_name, anchor_history in (
        ("isaac", isaac_history),
        ("mujoco", mujoco_history),
    ):
        full = _c70_reduction(
            anchor_history=anchor_history,
            error_history=error_history,
            dreamwaq=adapters["dreamwaq_run01"],
            phase1r=adapters["phase1r_run03"],
            reduction_mode="full_history",
        )
        latest = _c70_reduction(
            anchor_history=anchor_history,
            error_history=error_history,
            dreamwaq=adapters["dreamwaq_run01"],
            phase1r=adapters["phase1r_run03"],
            reduction_mode="latest_frame_only",
        )
        flip = {full["classification"], latest["classification"]} == {
            "amplifier",
            "not_distinguished",
        }
        classification = "inconclusive" if flip else full["classification"]
        anchors[anchor_name] = {
            "classification": classification,
            "reason": "history_reduction_sensitive" if flip else None,
            "trace_sha256": (
                isaac.identity.trace_sha256
                if anchor_name == "isaac"
                else mujoco.identity.trace_sha256
            ),
            "observation_sequence_sha256": stable_hash(anchor_history.tolist()),
            "baseline_history": anchor_history.tolist(),
            "full_history": full,
            "latest_frame_only": latest,
        }
    anchor_classes = {row["classification"] for row in anchors.values()}
    if len(anchor_classes) == 1 and next(iter(anchor_classes)) in {
        "amplifier",
        "not_distinguished",
    }:
        checkpoint_role = next(iter(anchor_classes))
        reason = None
    else:
        checkpoint_role = "inconclusive"
        reason = "baseline_anchor_sensitive"
    return {
        "schema_version": "RootCauseC70CheckpointSensitivityV1",
        "replay_source_identity_hash": identity_hash,
        "ticks": list(C70_TICKS),
        "times_ms": [20 * tick for tick in C70_TICKS],
        "feature_groups": {
            name: [start, stop] for name, (start, stop) in C70_FEATURE_GROUPS.items()
        },
        "policies": {
            name: {
                "actor_path": records["actor"].relative_path,
                "actor_sha256": sha256_file(records["actor"].absolute_path()),
                "manifest_path": records["manifest"].relative_path,
                "manifest_sha256": sha256_file(records["manifest"].absolute_path()),
            }
            for name, records in FROZEN_POLICIES.items()
        },
        "isaac_trace_sha256": isaac.identity.trace_sha256,
        "mujoco_trace_sha256": mujoco.identity.trace_sha256,
        "error_history_sha256": stable_hash(error_history.tolist()),
        "anchors": anchors,
        "checkpoint_role": checkpoint_role,
        "reason": reason,
    }
