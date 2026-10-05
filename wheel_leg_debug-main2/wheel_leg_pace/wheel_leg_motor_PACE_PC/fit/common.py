from __future__ import annotations

import dataclasses
from typing import List, Mapping, Sequence, Tuple

import numpy as np


COMMAND_DM_MIT = 1
COMMAND_LK_TORQUE = 5
COMMAND_LK_VELOCITY = 6
STAGE_DM_FIT = 2
STAGE_DM_VALIDATION = 3
STAGE_LK_TORQUE = 4
STAGE_LK_VELOCITY = 5
DM_BASELINE_KP = 20.0
DM_BASELINE_KD = 0.6
DM_VELOCITY_QUANTUM = 90.0 / 4095.0
DM_TORQUE_QUANTUM = 108.0 / 4095.0


@dataclasses.dataclass(frozen=True)
class DynamicsSeries:
    time_s: np.ndarray
    command: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    saturated: np.ndarray
    valid: np.ndarray
    sample_period_s: float


def fit_and_validation_roles(session_type: int) -> Tuple[str, str]:
    if session_type == 1:
        return "PROVISIONAL_FIT", "VALIDATION"
    if session_type == 2:
        return "FIT", "VALIDATION"
    raise ValueError(f"unsupported source session type {session_type}")


def select_rows(
    rows: Sequence[Mapping[str, object]],
    stage_id: int,
    role: str,
) -> List[Mapping[str, object]]:
    selected = [
        row
        for row in rows
        if int(row["stage_id"]) == stage_id and str(row["role"]) == role
    ]
    if not selected:
        raise ValueError(f"no rows found for stage {stage_id} and role {role}")
    return selected


def _numeric(rows: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def _boolean(rows: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.bool_)


def _sample_period(time_s: np.ndarray) -> float:
    if time_s.size < 3:
        raise ValueError("at least three samples are required")
    differences = np.diff(time_s)
    positive = differences[differences > 0.0]
    if positive.size == 0:
        raise ValueError("sample timestamps are not increasing")
    period = float(np.median(positive))
    if np.any(np.abs(positive - period) > max(1.0e-7, period * 0.05)):
        raise ValueError("selected fitting rows are not uniformly sampled")
    return period


def _derivative_validity(time_s: np.ndarray, eligible: np.ndarray, period: float) -> np.ndarray:
    valid = eligible.copy()
    valid[0] = False
    valid[-1] = False
    before = np.empty_like(valid)
    after = np.empty_like(valid)
    before[0] = False
    before[1:] = eligible[:-1]
    after[-1] = False
    after[:-1] = eligible[1:]
    valid &= before & after
    dt = np.diff(time_s)
    gap = np.abs(dt - period) > max(1.0e-7, period * 0.05)
    valid[:-1] &= ~gap
    valid[1:] &= ~gap
    return valid


def dm_series(rows: Sequence[Mapping[str, object]], motor_index: int) -> DynamicsSeries:
    prefix = f"m{motor_index}_"
    time_s = _numeric(rows, "time_s")
    period = _sample_period(time_s)
    eligible = _boolean(rows, prefix + "fit_eligible")
    position = _numeric(rows, prefix + "fb_q_rad")
    velocity = _numeric(rows, prefix + "fb_dq_rad_s")
    acceleration = np.gradient(velocity, time_s)
    valid = _derivative_validity(time_s, eligible, period)
    return DynamicsSeries(
        time_s=time_s,
        command=_numeric(rows, prefix + "cmd_q_rad"),
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        kp=_numeric(rows, prefix + "kp"),
        kd=_numeric(rows, prefix + "kd"),
        saturated=_boolean(rows, prefix + "saturated"),
        valid=valid,
        sample_period_s=period,
    )


def wheel_series(
    rows: Sequence[Mapping[str, object]],
    motor_index: int,
    command_key: str,
) -> DynamicsSeries:
    prefix = f"m{motor_index}_"
    time_s = _numeric(rows, "time_s")
    period = _sample_period(time_s)
    eligible = _boolean(rows, prefix + "fit_eligible")
    position = _numeric(rows, prefix + "fb_position_rad")
    velocity = _numeric(rows, prefix + "fb_velocity_rad_s")
    acceleration = np.gradient(velocity, time_s)
    valid = _derivative_validity(time_s, eligible, period)
    return DynamicsSeries(
        time_s=time_s,
        command=_numeric(rows, prefix + command_key),
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        kp=np.zeros_like(time_s),
        kd=np.zeros_like(time_s),
        saturated=_boolean(rows, prefix + "saturated"),
        valid=valid,
        sample_period_s=period,
    )


def delayed(values: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    if steps < 0:
        raise ValueError("delay steps cannot be negative")
    output = np.empty_like(values)
    valid = np.ones(values.shape, dtype=np.bool_)
    if steps == 0:
        output[:] = values
        return output, valid
    output[:steps] = values[0]
    output[steps:] = values[:-steps]
    valid[:steps] = False
    return output, valid


def smooth_direction(velocity: np.ndarray, transition_velocity: float = 0.05) -> np.ndarray:
    if transition_velocity <= 0.0:
        raise ValueError("transition velocity must be positive")
    return np.tanh(velocity / transition_velocity)


def require_dm_baseline(rows: Sequence[Mapping[str, object]], motor_index: int) -> None:
    prefix = f"m{motor_index}_"
    modes = {int(row[prefix + "command_mode"]) for row in rows}
    if modes != {COMMAND_DM_MIT}:
        raise ValueError(f"DM{motor_index} baseline data is not MIT position mode")
    kp = _numeric(rows, prefix + "kp")
    kd = _numeric(rows, prefix + "kd")
    dq = _numeric(rows, prefix + "cmd_dq_rad_s")
    tau = _numeric(rows, prefix + "cmd_tau_nm")
    if np.max(np.abs(kp - DM_BASELINE_KP)) > (500.0 / 4095.0 + 1.0e-6):
        raise ValueError(f"DM{motor_index} baseline Kp is not {DM_BASELINE_KP}")
    if np.max(np.abs(kd - DM_BASELINE_KD)) > (5.0 / 4095.0 + 1.0e-6):
        raise ValueError(f"DM{motor_index} baseline Kd is not {DM_BASELINE_KD}")
    if np.max(np.abs(dq)) > (1.5 * DM_VELOCITY_QUANTUM):
        raise ValueError(f"DM{motor_index} baseline dq_des is not zero")
    if np.max(np.abs(tau)) > (1.5 * DM_TORQUE_QUANTUM):
        raise ValueError(f"DM{motor_index} baseline torque feed-forward is not zero")
