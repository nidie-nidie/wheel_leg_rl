from __future__ import annotations

import argparse
import pathlib
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import lsq_linear

from pace_raw.decoder import decode_file
from pace_raw.normalize import NormalizedDataset, normalize_session
from validation.crossval import aggregate_metrics, error_metrics

from .common import (
    STAGE_DM_FIT,
    STAGE_DM_VALIDATION,
    DynamicsSeries,
    delayed,
    dm_series,
    fit_and_validation_roles,
    require_dm_baseline,
    select_rows,
    smooth_direction,
)
from .model_manifest import (
    create_model_manifest,
    load_model_manifest,
    replace_motor_result,
    validate_model_manifest,
    write_model_manifest,
)


PACE_ARMATURE_BOUNDS = (1.0e-5, 1.0)
PACE_VISCOUS_FRICTION_BOUNDS = (0.0, 7.0)
PACE_COULOMB_FRICTION_BOUNDS = (0.0, 0.5)
PACE_ENCODER_BIAS_BOUNDS = (-0.1, 0.1)
PACE_FRICTION_TRANSITION_VELOCITY = 0.05


def _design(
    series: DynamicsSeries,
    delay_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    command, delay_valid = delayed(series.command, delay_steps)
    direction = smooth_direction(series.velocity, PACE_FRICTION_TRANSITION_VELOCITY)
    target = series.kp * (command - series.position) - series.kd * series.velocity
    design = np.column_stack(
        [
            series.acceleration,
            series.velocity,
            direction,
            -series.kp,
        ]
    )
    finite = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    valid = series.valid & delay_valid & finite
    if np.count_nonzero(valid) < 32:
        raise ValueError("DM fitting requires at least 32 valid dynamic samples")
    if np.linalg.matrix_rank(design[valid]) < design.shape[1]:
        raise ValueError("DM fitting trajectory does not excite the full PACE parameter family")
    return design, target, valid


def _fit_channel(series: DynamicsSeries, delay_steps: int) -> Tuple[dict, np.ndarray, np.ndarray]:
    design, target, valid = _design(series, delay_steps)
    result = lsq_linear(
        design[valid],
        target[valid],
        bounds=(
            [
                PACE_ARMATURE_BOUNDS[0],
                PACE_VISCOUS_FRICTION_BOUNDS[0],
                PACE_COULOMB_FRICTION_BOUNDS[0],
                PACE_ENCODER_BIAS_BOUNDS[0],
            ],
            [
                PACE_ARMATURE_BOUNDS[1],
                PACE_VISCOUS_FRICTION_BOUNDS[1],
                PACE_COULOMB_FRICTION_BOUNDS[1],
                PACE_ENCODER_BIAS_BOUNDS[1],
            ],
        ),
        lsmr_tol="auto",
    )
    if not result.success:
        raise RuntimeError(f"bounded DM fit failed: {result.message}")
    predicted = design[valid] @ result.x
    residual = predicted - target[valid]
    parameters = {
        "armature": float(result.x[0]),
        "viscous_friction": float(result.x[1]),
        "static_friction": float(result.x[2]),
        "dynamic_friction": float(result.x[2]),
        "encoder_bias_rad": float(result.x[3]),
        "friction_transition_velocity_rad_s": PACE_FRICTION_TRANSITION_VELOCITY,
        "delay_steps": int(delay_steps),
    }
    return parameters, residual, valid


def _residual_for_parameters(
    series: DynamicsSeries,
    parameters: Mapping[str, float],
    delay_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    design, target, valid = _design(series, delay_steps)
    values = np.asarray(
        [
            parameters["armature"],
            parameters["viscous_friction"],
            parameters["dynamic_friction"],
            parameters["encoder_bias_rad"],
        ],
        dtype=np.float64,
    )
    return design[valid] @ values - target[valid], valid


def fit_leg_series(
    fit_series: Sequence[DynamicsSeries],
    validation_series: Sequence[DynamicsSeries],
    max_delay_steps: int = 10,
) -> Tuple[int, Sequence[dict], Sequence[dict], Sequence[dict]]:
    if len(fit_series) != 4 or len(validation_series) != 4:
        raise ValueError("leg fitting requires exactly four DM channels")
    candidates = []
    candidate_parameters = []
    for delay_steps in range(max_delay_steps + 1):
        parameters = []
        channel_mse = []
        for series in fit_series:
            fitted, residual, _ = _fit_channel(series, delay_steps)
            parameters.append(fitted)
            channel_mse.append(float(np.mean(residual * residual)))
        candidates.append(float(np.mean(channel_mse)))
        candidate_parameters.append(parameters)
    delay_steps = int(np.argmin(candidates))
    parameters = candidate_parameters[delay_steps]

    validation_delay_score = []
    for candidate_delay in range(max_delay_steps + 1):
        channel_mse = []
        for series, fitted in zip(validation_series, parameters):
            residual, _ = _residual_for_parameters(series, fitted, candidate_delay)
            channel_mse.append(float(np.mean(residual * residual)))
        validation_delay_score.append(float(np.mean(channel_mse)))
    estimated_validation_delay = int(np.argmin(validation_delay_score))

    fit_metrics = []
    validation_metrics = []
    for fitted_series, held_out_series, fitted in zip(
        fit_series, validation_series, parameters
    ):
        fit_error, fit_valid = _residual_for_parameters(fitted_series, fitted, delay_steps)
        held_out_error, held_out_valid = _residual_for_parameters(
            held_out_series, fitted, delay_steps
        )
        fit_metrics.append(
            error_metrics(
                fit_error,
                delay_steps,
                delay_steps,
                fitted_series.saturated[fit_valid],
            )
        )
        validation_metrics.append(
            error_metrics(
                held_out_error,
                delay_steps,
                estimated_validation_delay,
                held_out_series.saturated[held_out_valid],
            )
        )
    return delay_steps, parameters, fit_metrics, validation_metrics


def fit_leg_dm(
    dataset: NormalizedDataset,
    model: Optional[MutableMapping[str, object]] = None,
    max_delay_steps: int = 10,
) -> MutableMapping[str, object]:
    header = dataset.decoded.header
    if header is None:
        raise ValueError("DM fitting requires a session header")
    fit_role, validation_role = fit_and_validation_roles(header.session_type)
    fit_rows = select_rows(dataset.rows, STAGE_DM_FIT, fit_role)
    validation_rows = select_rows(dataset.rows, STAGE_DM_VALIDATION, validation_role)
    for index in range(4):
        require_dm_baseline(fit_rows, index)
        require_dm_baseline(validation_rows, index)
    fit_data = [dm_series(fit_rows, index) for index in range(4)]
    validation_data = [dm_series(validation_rows, index) for index in range(4)]
    delay_steps, parameters, fit_metrics, validation_metrics = fit_leg_series(
        fit_data, validation_data, max_delay_steps=max_delay_steps
    )

    if model is None:
        model = create_model_manifest(dataset)
    else:
        validate_model_manifest(model, require_complete=False)
    model["fit_backend"]["leg_dm"] = {
        "name": "pace_parameter_family_local_initializer",
        "parameter_layout": "armature[4], viscous[4], static_dynamic_friction[4], encoder_bias[4], global_delay[1]",
        "formal_optimizer": "pace_sim2real.CMAESOptimizer",
        "metric_quantity": "effective_torque_balance_nm",
        "max_delay_steps": int(max_delay_steps),
        "selected_global_delay_steps": delay_steps,
    }
    for index in range(4):
        actuator_model = {
            "type": "pace_dm_pd_v1",
            **parameters[index],
            "controller": {
                "kp": 20.0,
                "kd": 0.6,
                "dq_des_rad_s": 0.0,
                "torque_feedforward_nm": 0.0,
            },
            "effort_limit_nm": 54.0,
            "parameter_semantics": "effective_joint_space_initializer",
        }
        fit_result = {
            "stage_id": STAGE_DM_FIT,
            "role": fit_role,
            "method": "bounded_torque_balance",
            "metrics": fit_metrics[index],
        }
        validation_result = {
            "stage_id": STAGE_DM_VALIDATION,
            "role": validation_role,
            "metric_quantity": "effective_torque_balance_nm",
            **validation_metrics[index],
        }
        replace_motor_result(
            model,
            index,
            actuator_model,
            fit_result,
            validation_result,
        )
    model["aggregate_metrics"]["leg_dm_validation"] = aggregate_metrics(
        validation_metrics
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the four DM8009 PACE actuator channels")
    parser.add_argument("raw_session", type=pathlib.Path)
    parser.add_argument("output_manifest", type=pathlib.Path)
    parser.add_argument("--merge", type=pathlib.Path)
    parser.add_argument("--max-delay-steps", type=int, default=10)
    args = parser.parse_args()
    dataset = normalize_session(decode_file(args.raw_session))
    model = load_model_manifest(args.merge, require_complete=False) if args.merge else None
    model = fit_leg_dm(dataset, model=model, max_delay_steps=args.max_delay_steps)
    write_model_manifest(model, args.output_manifest, require_complete=False)


if __name__ == "__main__":
    main()
