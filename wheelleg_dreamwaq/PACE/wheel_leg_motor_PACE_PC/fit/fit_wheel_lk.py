from __future__ import annotations

import argparse
import pathlib
from typing import Mapping, MutableMapping, Optional

import numpy as np
from scipy.optimize import lsq_linear

from pace_raw.decoder import decode_file
from pace_raw.normalize import NormalizedDataset, normalize_session
from pace_raw.quality import require_fit_quality
from validation.crossval import aggregate_metrics, error_metrics

from .common import (
    COMMAND_LK_TORQUE,
    COMMAND_LK_VELOCITY,
    STAGE_LK_TORQUE,
    STAGE_LK_VELOCITY,
    DynamicsSeries,
    delayed,
    fit_and_validation_roles,
    smooth_direction,
    wheel_series,
    segment_rows,
    series_segments,
    validate_delay_period,
)
from .model_manifest import (
    create_model_manifest,
    load_model_manifest,
    replace_motor_result,
    validate_model_manifest,
    write_model_manifest,
)


LK_ARMATURE_BOUNDS = (1.0e-5, 5.0)
LK_VISCOUS_FRICTION_BOUNDS = (0.0, 20.0)
LK_COULOMB_FRICTION_BOUNDS = (0.0, 10.0)
LK_VELOCITY_GAIN_BOUNDS = (0.01, 2.0)
LK_VELOCITY_TIME_CONSTANT_BOUNDS = (0.001, 2.0)
LK_FRICTION_TRANSITION_VELOCITY = 0.10
LK_EFFORT_LIMIT_NM = 2.41


def _mode_rows(
    dataset: NormalizedDataset,
    mode: int,
    role: str,
    stage_id: Optional[int] = None,
) -> list:
    rows = []
    for row in dataset.rows:
        if str(row["role"]) != role:
            continue
        if stage_id is not None and int(row["stage_id"]) != stage_id:
            continue
        if all(int(row[f"m{index}_command_mode"]) == mode for index in (4, 5)):
            rows.append(row)
    return rows


def _combine_design(segments, delay_steps, design_function):
    pieces = [design_function(segment, delay_steps, validate=False) for segment in series_segments(segments)]
    combined = tuple(np.concatenate([piece[index] for piece in pieces]) for index in range(3))
    _validate_design(*combined, label="LK")
    return combined


def _saturation(series):
    return np.concatenate([segment.saturated for segment in series_segments(series)])


def _validate_design(design, target, valid, *, label):
    if np.count_nonzero(valid) < 32:
        raise ValueError(f"{label} fitting requires at least 32 valid samples across its segments")
    if np.linalg.matrix_rank(design[valid]) < design.shape[1]:
        raise ValueError(f"{label} trajectory does not excite the model parameters")


def _torque_design(series: DynamicsSeries, delay_steps: int, *, validate=True):
    if not isinstance(series, DynamicsSeries):
        return _combine_design(series, delay_steps, _torque_design)
    command, delay_valid = delayed(series.command, delay_steps)
    design = np.column_stack(
        [
            series.acceleration,
            series.velocity,
            smooth_direction(series.velocity, LK_FRICTION_TRANSITION_VELOCITY),
        ]
    )
    finite = np.isfinite(command) & np.all(np.isfinite(design), axis=1)
    delayed_source_valid = delayed(series.valid, delay_steps)[0]
    valid = series.valid & delayed_source_valid & delay_valid & finite
    if validate:
        _validate_design(design, command, valid, label="LK torque")
    return design, command, valid


def _fit_torque(series: DynamicsSeries, delay_steps: int):
    design, target, valid = _torque_design(series, delay_steps)
    result = lsq_linear(
        design[valid],
        target[valid],
        bounds=(
            [
                LK_ARMATURE_BOUNDS[0],
                LK_VISCOUS_FRICTION_BOUNDS[0],
                LK_COULOMB_FRICTION_BOUNDS[0],
            ],
            [
                LK_ARMATURE_BOUNDS[1],
                LK_VISCOUS_FRICTION_BOUNDS[1],
                LK_COULOMB_FRICTION_BOUNDS[1],
            ],
        ),
        lsmr_tol="auto",
    )
    if not result.success:
        raise RuntimeError(f"bounded LK torque fit failed: {result.message}")
    residual = design[valid] @ result.x - target[valid]
    return {
        "armature": float(result.x[0]),
        "viscous_friction": float(result.x[1]),
        "static_friction": float(result.x[2]),
        "dynamic_friction": float(result.x[2]),
        "friction_transition_velocity_rad_s": LK_FRICTION_TRANSITION_VELOCITY,
        "delay_steps": int(delay_steps),
    }, residual, valid


def _torque_residual(series: DynamicsSeries, parameters: Mapping[str, float], delay_steps: int):
    design, target, valid = _torque_design(series, delay_steps)
    values = np.asarray(
        [
            parameters["armature"],
            parameters["viscous_friction"],
            parameters["dynamic_friction"],
        ],
        dtype=np.float64,
    )
    return design[valid] @ values - target[valid], valid


def _velocity_design(series: DynamicsSeries, delay_steps: int, *, validate=True):
    if not isinstance(series, DynamicsSeries):
        return _combine_design(series, delay_steps, _velocity_design)
    command, delay_valid = delayed(series.command, delay_steps)
    design = np.column_stack([command, -series.acceleration])
    target = series.velocity
    finite = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    delayed_source_valid = delayed(series.valid, delay_steps)[0]
    valid = series.valid & delayed_source_valid & delay_valid & finite
    if validate:
        _validate_design(design, target, valid, label="LK velocity")
    return design, target, valid


def _fit_velocity(series: DynamicsSeries, delay_steps: int):
    design, target, valid = _velocity_design(series, delay_steps)
    result = lsq_linear(
        design[valid],
        target[valid],
        bounds=(
            [LK_VELOCITY_GAIN_BOUNDS[0], LK_VELOCITY_TIME_CONSTANT_BOUNDS[0]],
            [LK_VELOCITY_GAIN_BOUNDS[1], LK_VELOCITY_TIME_CONSTANT_BOUNDS[1]],
        ),
        lsmr_tol="auto",
    )
    if not result.success:
        raise RuntimeError(f"bounded LK velocity fit failed: {result.message}")
    residual = design[valid] @ result.x - target[valid]
    return {
        "command_gain": float(result.x[0]),
        "time_constant_s": float(result.x[1]),
        "delay_steps": int(delay_steps),
    }, residual, valid


def _velocity_residual(series: DynamicsSeries, parameters: Mapping[str, float], delay_steps: int):
    design, target, valid = _velocity_design(series, delay_steps)
    values = np.asarray(
        [parameters["command_gain"], parameters["time_constant_s"]],
        dtype=np.float64,
    )
    return design[valid] @ values - target[valid], valid


def _fit_with_delay(series: DynamicsSeries, fitter, max_delay_steps: int):
    results = []
    scores = []
    for delay_steps in range(max_delay_steps + 1):
        parameters, residual, valid = fitter(series, delay_steps)
        results.append((parameters, residual, valid))
        scores.append(float(np.mean(residual * residual)))
    selected = int(np.argmin(scores))
    return selected, results[selected]


def _validation_delay(series: DynamicsSeries, parameters, residual_function, max_delay_steps: int):
    scores = []
    for delay_steps in range(max_delay_steps + 1):
        residual, _ = residual_function(series, parameters, delay_steps)
        scores.append(float(np.mean(residual * residual)))
    return int(np.argmin(scores))


def fit_wheel_lk(dataset, model=None, max_delay_steps=10):
    require_fit_quality(dataset)
    if max_delay_steps < 0:
        raise ValueError("max_delay_steps cannot be negative")
    if model is None:
        model = create_model_manifest(dataset)
    else:
        validate_model_manifest(model, require_complete=False)
        from .model_manifest import require_matching_source
        require_matching_source(model, dataset)
    model["fit_backend"]["wheel_lk"] = {
        "name": "segmented_torque_and_velocity_identification",
        "primary_report": "velocity_mode",
        "torque_metric_quantity": "effective_torque_balance_nm",
        "velocity_metric_quantity": "wheel_velocity_rad_s",
        "max_delay_steps": max_delay_steps,
        "held_out_validation_available": True,
    }
    all_validation, velocity_validation_metrics = [], []
    for motor_index in (4, 5):
        mode_results = {}
        fit_result = {"role": "FIT", "method": "segmented_bounded_mode_specific_regression"}
        validation_result = {"available": True, "role": "VALIDATION"}
        for mode_name, mode, command_key, fitter, residual_function in (
            ("torque", COMMAND_LK_TORQUE, "cmd_torque_nm", _fit_torque, _torque_residual),
            ("velocity", COMMAND_LK_VELOCITY, "cmd_velocity_rad_s", _fit_velocity, _velocity_residual),
        ):
            category = STAGE_LK_TORQUE if mode == COMMAND_LK_TORQUE else STAGE_LK_VELOCITY
            fit_rows = segment_rows(dataset.rows, motor_index, stage_id=category, role="FIT", mode=mode)
            held_rows = segment_rows(dataset.rows, motor_index, stage_id=category, role="VALIDATION", mode=mode)
            fit_ids = {segment[0]["segment_id"] for segment in fit_rows}
            held_ids = {segment[0]["segment_id"] for segment in held_rows}
            if fit_ids & held_ids:
                raise ValueError("FIT and VALIDATION segment IDs overlap")
            fit_series = [wheel_series(segment, motor_index, command_key) for segment in fit_rows]
            held_series = [wheel_series(segment, motor_index, command_key) for segment in held_rows]
            sample_period = validate_delay_period([fit_series, held_series])
            delay, (parameters, fit_error, fit_valid) = _fit_with_delay(fit_series, fitter, max_delay_steps)
            parameters.update(delay_s=delay * sample_period, delay_semantics="command_before_response",
                              identification_sample_period_s=sample_period)
            fit_metrics = error_metrics(fit_error, delay, delay, _saturation(fit_series)[fit_valid])
            estimated = _validation_delay(held_series, parameters, residual_function, max_delay_steps)
            held_error, held_valid = residual_function(held_series, parameters, delay)
            metrics = error_metrics(held_error, delay, estimated, _saturation(held_series)[held_valid])
            all_validation.append(metrics)
            if mode_name == "velocity":
                velocity_validation_metrics.append(metrics)
            mode_results[mode_name + "_mode"] = parameters
            fit_result[mode_name + "_mode"] = {"segment_ids": sorted(fit_ids), "metrics": fit_metrics}
            validation_result[mode_name + "_mode"] = {"segment_ids": sorted(held_ids), **metrics}
        actuator = {"type": "lk9025_dual_mode_v1", **mode_results, "effort_limit_nm": LK_EFFORT_LIMIT_NM,
                    "parameter_semantics": "effective_joint_space_initializer"}
        replace_motor_result(model, motor_index, actuator, fit_result, validation_result)
    model["aggregate_metrics"]["wheel_lk_validation"] = aggregate_metrics(all_validation)
    model["aggregate_metrics"]["wheel_velocity_validation"] = aggregate_metrics(velocity_validation_metrics)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit both LK9025 wheel actuator modes")
    parser.add_argument("raw_session", type=pathlib.Path)
    parser.add_argument("input_manifest", type=pathlib.Path)
    parser.add_argument("output_manifest", type=pathlib.Path)
    parser.add_argument("--max-delay-steps", type=int, default=10)
    args = parser.parse_args()
    dataset = normalize_session(decode_file(args.raw_session))
    model = load_model_manifest(args.input_manifest, require_complete=False)
    model = fit_wheel_lk(dataset, model=model, max_delay_steps=args.max_delay_steps)
    write_model_manifest(model, args.output_manifest)


if __name__ == "__main__":
    main()
