"""Exported command practice validation and independent physical-response statistics."""
from __future__ import annotations

import numbers
import numpy as np

PRACTICE_VERSION = "StopReverseCommandPracticeV1"
STAGE_PAYLOAD = {
    "version": PRACTICE_VERSION,
    "stage_end_control_steps": [100, 150, 250, 300, 400, 500],
    "factors": [1, 0, -1, 0, 1, 0],
    "control_dt_s": 0.02,
    "clock": "real_completed_control_steps_per_environment",
    "update_timing": "after_reward_before_next_observation",
    "height_behavior": "hold_reset_sample",
    "final_phase_behavior": "hold_last_factor",
}


def validate_command_practice(manifest: dict) -> dict | None:
    commands = manifest.get("command_sampling")
    if not isinstance(commands, dict):
        raise ValueError("Policy command sampling contract is missing")
    schedule = commands.get("practice_schedule", "disabled")
    payload = commands.get("practice_contract")
    if schedule == "disabled":
        if commands.get("hold_for_episode") is not True or payload is not None:
            raise ValueError("Disabled practice requires episode-held commands and a null payload")
        return None
    if schedule != PRACTICE_VERSION or commands.get("hold_for_episode") is not False:
        raise ValueError("Unknown or contradictory command practice profile")
    if payload != STAGE_PAYLOAD:
        raise ValueError("Exported command practice timeline differs from StopReverseCommandPracticeV1")
    weights = manifest.get("training_reward_weights", {})
    if weights.get("tracking_vx") != 2.0 or weights.get("tracking_vx_enhance") != 2.0:
        raise ValueError("Stop/reverse policy must explicitly record its doubled vx training weights")
    # Return independent arrays/lists through a fresh dictionary.
    return {key: list(value) if isinstance(value, list) else value for key, value in payload.items()}


def command_at_tick(initial: np.ndarray, completed_ticks: int, payload: dict) -> np.ndarray:
    if payload != STAGE_PAYLOAD:
        raise ValueError("Unsupported command practice payload")
    if isinstance(completed_ticks, bool) or not isinstance(completed_ticks, numbers.Integral) or completed_ticks < 0:
        raise ValueError("Completed ticks must be a non-negative integer")
    command = np.asarray(initial, dtype=np.float64).copy()
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError("Initial command must be a finite CommandV1")
    stage = int(np.searchsorted(payload["stage_end_control_steps"][:-1], completed_ticks, side="right"))
    command[:2] *= payload["factors"][stage]
    return command


def _held_band_time(rows: list[dict], field: str, target: float, start: int, dt: float) -> float | None:
    consecutive = 0
    for row in rows:
        if abs(row[field] - target) <= 0.15:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive == 10:
            return (row["tick"] - start + 1) * dt
    return None


def summarize_practice(name: str, initial: np.ndarray, rows: list[dict], failure_reason: str | None) -> dict:
    """Rows are contiguous, survived post-action samples; tick is zero-based."""
    initial = np.asarray(initial, dtype=np.float64)
    if initial.shape != (3,) or not np.isfinite(initial).all():
        raise ValueError("Initial command must be finite")
    if len(rows) > 500 or any(row["tick"] != index for index, row in enumerate(rows)):
        raise ValueError("Statistics require a contiguous valid prefix, excluding the failed sample")
    dt = STAGE_PAYLOAD["control_dt_s"]
    stages = []
    start = 0
    for index, end in enumerate(STAGE_PAYLOAD["stage_end_control_steps"]):
        selected = rows[start:min(end, len(rows))]
        target = command_at_tick(initial, start, STAGE_PAYLOAD)
        steady = [row for row in selected if (row["tick"] - start) * dt >= 0.5]
        complete = len(selected) == end - start
        vx_mae = float(np.mean([abs(row["vx_mps"] - target[0]) for row in steady])) if steady else None
        yaw_mae = float(np.mean([abs(row["yaw_rate_rad_s"] - target[1]) for row in steady])) if steady else None
        stopped = STAGE_PAYLOAD["factors"][index] == 0
        stop_distance = net_displacement = None
        if stopped and selected:
            positions = np.asarray([selected[0]["com_xy_before"], *[row["com_xy_after"] for row in selected]])
            stop_distance = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        axis = "vx_mps" if initial[0] != 0.0 else "yaw_rate_rad_s"
        axis_target = float(target[0] if initial[0] != 0.0 else target[1])
        sign_time = next(((row["tick"] - start + 1) * dt for row in selected
                          if axis_target != 0.0 and row[axis] * axis_target > 0.0), None)
        band_time = _held_band_time(selected, axis, axis_target, start, dt) if axis_target else None
        accepted = complete
        reasons = []
        if not complete:
            reasons.append("incomplete_stage")
        if stopped:
            if vx_mae is None or vx_mae > 0.10:
                reasons.append("stop_vx_mae")
            if yaw_mae is None or yaw_mae > 0.15:
                reasons.append("stop_yaw_mae")
        elif index in {2, 4}:
            if band_time is None or band_time > 1.0:
                reasons.append("reverse_response_time")
        accepted = accepted and not reasons
        stages.append({
            "stage_index": index, "start_tick": start, "end_tick": end,
            "target": target.tolist(), "valid_ticks": len(selected), "completed": complete,
            "steady_sample_ticks": len(steady), "steady_vx_mae_mps": vx_mae,
            "steady_yaw_mae_rad_s": yaw_mae, "stop_distance_m": stop_distance,
            "stop_net_displacement_m": net_displacement, "first_correct_sign_time_s": sign_time,
            "held_error_band_confirmation_time_s": band_time, "performance_accepted": accepted,
            "failures": reasons,
        })
        start = end
    completed = len(rows) == 500 and failure_reason is None
    return {"name": name, "initial_command": initial.tolist(), "valid_ticks": len(rows),
            "completed": completed, "failure_reason": failure_reason, "stages": stages,
            "performance_accepted": completed and all(stage["performance_accepted"] for stage in stages)}
