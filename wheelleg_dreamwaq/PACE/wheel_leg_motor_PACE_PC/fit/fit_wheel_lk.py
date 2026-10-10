from __future__ import annotations

import argparse
import pathlib
from typing import Mapping, MutableMapping, Optional

import numpy as np
from scipy.optimize import lsq_linear

from pace_raw.decoder import decode_file
from pace_raw.normalize import NormalizedDataset, normalize_session
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


def _torque_design(series: DynamicsSeries, delay_steps: int):
    command, delay_valid = delayed(series.command, delay_steps)
    design = np.column_stack(
        [
            series.acceleration,
            series.velocity,
            smooth_direction(series.velocity, LK_FRICTION_TRANSITION_VELOCITY),
        ]
    )
    finite = np.isfinite(command) & np.all(np.isfinite(design), axis=1)
    valid = series.valid & delay_valid & finite
    if np.count_nonzero(valid) < 32:
        raise ValueError("LK torque fitting requires at least 32 valid samples")
    if np.linalg.matrix_rank(design[valid]) < design.shape[1]:
        raise ValueError("LK torque trajectory does not excite inertia and friction")
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


def _velocity_design(series: DynamicsSeries, delay_steps: int):
    command, delay_valid = delayed(series.command, delay_steps)
    design = np.column_stack([command, -series.acceleration])
    target = series.velocity
    finite = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    valid = series.valid & delay_valid & finite
    if np.count_nonzero(valid) < 32:
        raise ValueError("LK velocity fitting requires at least 32 valid samples")
    if np.linalg.matrix_rank(design[valid]) < design.shape[1]:
        raise ValueError("LK velocity trajectory does not excite gain and time constant")
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


def fit_wheel_lk(
    dataset: NormalizedDataset,
    model: Optional[MutableMapping[str, object]] = None,
    max_delay_steps: int = 10,
) -> MutableMapping[str, object]:
    header = dataset.decoded.header
    if header is None:
        raise ValueError("LK fitting requires a session header")
    fit_role, validation_role = fit_and_validation_roles(header.session_type)
    commissioning = header.session_type == 1
    torque_fit_rows = _mode_rows(
        dataset,
        COMMAND_LK_TORQUE,
        fit_role,
        STAGE_LK_TORQUE if commissioning else None,
    )
    velocity_fit_rows = _mode_rows(
        dataset,
        COMMAND_LK_VELOCITY,
        fit_role,
        STAGE_LK_VELOCITY if commissioning else None,
    )
    if not torque_fit_rows or not velocity_fit_rows:
        raise ValueError("LK fitting requires separate torque-mode and velocity-mode stages")
    torque_validation_rows = [] if commissioning else _mode_rows(
        dataset, COMMAND_LK_TORQUE, validation_role
    )
    velocity_validation_rows = [] if commissioning else _mode_rows(
        dataset, COMMAND_LK_VELOCITY, validation_role
    )
    if not commissioning and (not torque_validation_rows or not velocity_validation_rows):
        raise ValueError("a final LK model requires held-out validation data for both command modes")

    if model is None:
        model = create_model_manifest(dataset)
    else:
        validate_model_manifest(model, require_complete=False)
    model["fit_backend"]["wheel_lk"] = {
        "name": "separate_torque_and_velocity_identification",
        "torque_metric_quantity": "effective_torque_balance_nm",
        "velocity_metric_quantity": "wheel_velocity_rad_s",
        "max_delay_steps": int(max_delay_steps),
        "held_out_validation_available": not commissioning,
    }

    validation_metric_sets = []
    for motor_index in (4, 5):
        torque_series = wheel_series(torque_fit_rows, motor_index, "cmd_torque_nm")
        velocity_series = wheel_series(
            velocity_fit_rows, motor_index, "cmd_velocity_rad_s"
        )
        torque_delay, (torque_parameters, torque_error, torque_valid) = _fit_with_delay(
            torque_series, _fit_torque, max_delay_steps
        )
        velocity_delay, (velocity_parameters, velocity_error, velocity_valid) = _fit_with_delay(
            velocity_series, _fit_velocity, max_delay_steps
        )
        torque_fit_metrics = error_metrics(
            torque_error,
            torque_delay,
            torque_delay,
            torque_series.saturated[torque_valid],
        )
        velocity_fit_metrics = error_metrics(
            velocity_error,
            velocity_delay,
            velocity_delay,
            velocity_series.saturated[velocity_valid],
        )

        if commissioning:
            validation_result = {
                "available": False,
                "reason": "commissioning session has no independent LK holdout stage",
            }
        else:
            torque_held_out = wheel_series(
                torque_validation_rows, motor_index, "cmd_torque_nm"
            )
            velocity_held_out = wheel_series(
                velocity_validation_rows, motor_index, "cmd_velocity_rad_s"
            )
            torque_estimated_delay = _validation_delay(
                torque_held_out,
                torque_parameters,
                _torque_residual,
                max_delay_steps,
            )
            velocity_estimated_delay = _validation_delay(
                velocity_held_out,
                velocity_parameters,
                _velocity_residual,
                max_delay_steps,
            )
            torque_held_error, torque_held_valid = _torque_residual(
                torque_held_out, torque_parameters, torque_delay
            )
            velocity_held_error, velocity_held_valid = _velocity_residual(
                velocity_held_out, velocity_parameters, velocity_delay
            )
            torque_validation = error_metrics(
                torque_held_error,
                torque_delay,
                torque_estimated_delay,
                torque_held_out.saturated[torque_held_valid],
            )
            velocity_validation = error_metrics(
                velocity_held_error,
                velocity_delay,
                velocity_estimated_delay,
                velocity_held_out.saturated[velocity_held_valid],
            )
            validation_metric_sets.extend([torque_validation, velocity_validation])
            validation_result = {
                "available": True,
                "role": validation_role,
                "torque_mode": torque_validation,
                "velocity_mode": velocity_validation,
            }

        actuator_model = {
            "type": "lk9025_dual_mode_v1",
            "torque_mode": torque_parameters,
            "velocity_mode": velocity_parameters,
            "effort_limit_nm": LK_EFFORT_LIMIT_NM,
            "parameter_semantics": "effective_joint_space_initializer",
        }
        fit_result = {
            "role": fit_role,
            "method": "bounded_mode_specific_regression",
            "torque_mode": {
                "stage_id": STAGE_LK_TORQUE,
                "metrics": torque_fit_metrics,
            },
            "velocity_mode": {
                "stage_id": STAGE_LK_VELOCITY,
                "metrics": velocity_fit_metrics,
            },
        }
        replace_motor_result(
            model,
            motor_index,
            actuator_model,
            fit_result,
            validation_result,
        )

    model["aggregate_metrics"]["wheel_lk_validation"] = (
        aggregate_metrics(validation_metric_sets)
        if validation_metric_sets
        else {
            "available": False,
            "reason": "commissioning session provides provisional LK fits only",
        }
    )
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
