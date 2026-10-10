from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .trace_schema import load_json, load_npz, sha256_file, stable_payload_hash, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "MujocoIsolationTraceV1"
HORIZONS = (("0p10", 0.10), ("0p20", 0.20))


def _sample_at_or_before(
    time_s: np.ndarray,
    values: np.ndarray,
    target_s: float,
) -> tuple[float, float] | None:
    if float(time_s[-1]) + 1.0e-12 < target_s:
        return None
    indices = np.flatnonzero(time_s <= target_s + 1.0e-12)
    if not len(indices):
        return None
    index = int(indices[-1])
    return float(time_s[index]), float(values[index])


def _resolve_model_path(raw_path: str) -> Path:
    model_path = Path(raw_path)
    return model_path if model_path.is_absolute() else PROJECT_ROOT / model_path


def _validate_artifact(run: Path, metadata: dict[str, Any], trace: dict[str, np.ndarray]) -> None:
    hashes_path = run / "file_hashes.json"
    if not hashes_path.is_file():
        raise ValueError(f"Isolation artifact is missing file_hashes.json: {run}")
    expected_hashes = load_json(hashes_path)
    emitted_files = {
        "metadata": run / "metadata.json",
        "isolation_trace": run / "isolation_trace.npz",
    }
    for name, path in emitted_files.items():
        expected = expected_hashes.get(name)
        actual = sha256_file(path)
        if expected != actual:
            raise ValueError(f"Isolation artifact hash mismatch for {name}: {run}")

    scenario = metadata.get("scenario")
    if not isinstance(scenario, dict) or metadata.get("scenario_hash") != stable_payload_hash(scenario):
        raise ValueError(f"Isolation scenario hash mismatch: {run}")

    model_path = _resolve_model_path(str(metadata.get("model_path", "")))
    expected_model_hash = metadata.get("model_sha256")
    if not model_path.is_file() or sha256_file(model_path) != expected_model_hash:
        raise ValueError(f"Isolation model hash mismatch: {model_path}")
    if expected_hashes.get("model_xml") != expected_model_hash:
        raise ValueError(f"Isolation model hash metadata mismatch: {run}")

    rows = len(trace.get("control_time_s", ()))
    if rows <= 0:
        raise ValueError("Isolation trace is empty")
    inconsistent = sorted(name for name, values in trace.items() if np.asarray(values).shape[0] != rows)
    if inconsistent:
        raise ValueError(f"Isolation trace fields have inconsistent row counts: {inconsistent}")
    completed = int(metadata.get("completed_control_ticks", -1))
    if completed != rows:
        raise ValueError(f"completed_control_ticks={completed} does not match trace rows={rows}")
    requested = int(metadata.get("requested_control_ticks", -1))
    if requested < completed:
        raise ValueError("requested_control_ticks is smaller than completed_control_ticks")

    ticks = np.asarray(trace.get("control_tick"))
    if not np.array_equal(ticks, np.arange(rows, dtype=ticks.dtype)):
        raise ValueError("Isolation control ticks are missing, duplicated, or out of order")
    control_dt = float(metadata.get("control_dt_s", np.nan))
    physics_dt = float(metadata.get("physics_dt_s", np.nan))
    substeps = int(metadata.get("physics_steps_per_action", -1))
    if not np.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("Isolation control_dt_s must be positive")
    if substeps <= 0 or not np.isclose(physics_dt * substeps, control_dt, rtol=0.0, atol=1.0e-12):
        raise ValueError("Isolation physics and control clocks are inconsistent")
    expected_time = control_dt * np.arange(1, rows + 1, dtype=np.float64)
    if not np.allclose(trace["control_time_s"], expected_time, rtol=0.0, atol=1.0e-10):
        raise ValueError("Isolation control_time_s is not continuous at control_dt_s")


def _first_zero_crossing(
    time_s: np.ndarray,
    pitch_deg: np.ndarray,
    initial_pitch_deg: float,
) -> float | None:
    initial_sign = np.sign(initial_pitch_deg)
    if initial_sign == 0.0:
        return None
    indices = np.flatnonzero(initial_sign * pitch_deg <= 0.0)
    return None if not len(indices) else float(time_s[int(indices[0])])


def summarize_run(run_directory: str | Path) -> dict[str, Any]:
    run = Path(run_directory).resolve()
    metadata = load_json(run / "metadata.json")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported isolation schema in {run}")
    trace = load_npz(run / "isolation_trace.npz")
    required = {
        "control_tick",
        "base_rpy_control",
        "base_angular_velocity_control",
        "base_linear_velocity_control",
        "active_joint_velocity_canonical",
        "commanded_torque_canonical_mean",
        "wheel_contact_active",
        "loop_closure_error_m",
        "control_time_s",
    }
    missing = sorted(required - set(trace))
    if missing:
        raise ValueError(f"Isolation trace is missing fields: {missing}")
    _validate_artifact(run, metadata, trace)

    scenario = metadata["scenario"]
    pitch_deg = np.rad2deg(np.asarray(trace["base_rpy_control"])[:, 1])
    pitch_rate = np.asarray(trace["base_angular_velocity_control"])[:, 1]
    vx = np.asarray(trace["base_linear_velocity_control"])[:, 0]
    wheel_speed = np.asarray(trace["active_joint_velocity_canonical"])[:, 4:6]
    wheel_torque_mean = np.asarray(trace["commanded_torque_canonical_mean"])[:, 4:6]
    contact = np.asarray(trace["wheel_contact_active"])
    closure = np.asarray(trace["loop_closure_error_m"])
    initial_pitch_deg = float(metadata["initial_state"]["pitch_deg"])
    final_pitch_deg = round(float(pitch_deg[-1]), 12)
    time_s = np.asarray(trace["control_time_s"], dtype=np.float64)
    minimum_index = int(np.argmin(np.abs(pitch_deg)))
    initial_sign = np.sign(initial_pitch_deg)
    opposite_pitch = np.maximum(-initial_sign * pitch_deg, 0.0) if initial_sign else np.zeros_like(pitch_deg)

    summary: dict[str, Any] = {
        "run_directory": str(run),
        "scenario_kind": scenario["kind"],
        "initial_pitch_deg": initial_pitch_deg,
        "wheel_common_action": float(scenario.get("wheel_common_action", 0.0)),
        "pulse_channel": scenario.get("pulse_channel"),
        "wheel_pulse_mode": scenario.get("wheel_pulse_mode"),
        "pulse_amplitude": float(scenario.get("pulse_amplitude", 0.0)),
        "completed_control_ticks": int(metadata["completed_control_ticks"]),
        "stopped_reason": metadata.get("stopped_reason", "unknown"),
        "duration_s": float(time_s[-1]),
        "final_pitch_deg": final_pitch_deg,
        "final_abs_pitch_deg": abs(final_pitch_deg),
        "pitch_abs_reduction_deg": round(abs(initial_pitch_deg) - abs(final_pitch_deg), 12),
        "minimum_abs_pitch_deg": round(float(abs(pitch_deg[minimum_index])), 12),
        "minimum_abs_pitch_time_s": float(time_s[minimum_index]),
        "pitch_rate_at_minimum_abs_pitch_rad_s": float(pitch_rate[minimum_index]),
        "first_zero_crossing_time_s": _first_zero_crossing(time_s, pitch_deg, initial_pitch_deg),
        "max_opposite_signed_pitch_deg": round(float(np.max(opposite_pitch)), 12),
        "peak_abs_pitch_deg": round(float(np.max(np.abs(pitch_deg))), 12),
        "peak_abs_pitch_rate_rad_s": float(np.max(np.abs(pitch_rate))),
        "peak_abs_vx_mps": float(np.max(np.abs(vx))),
        "final_wheel_speed_common_rad_s": float(np.mean(wheel_speed[-1])),
        "peak_abs_wheel_speed_rad_s": float(np.max(np.abs(wheel_speed))),
        "peak_abs_tick_mean_wheel_torque_nm": float(np.max(np.abs(wheel_torque_mean))),
        "peak_abs_substep_wheel_torque_nm": None,
        "wheel_contact_fraction": float(np.mean(contact)),
        "max_loop_closure_error_m": float(np.max(closure)),
    }
    if "commanded_torque_canonical_peak_abs" in trace:
        summary["peak_abs_substep_wheel_torque_nm"] = float(
            np.max(np.asarray(trace["commanded_torque_canonical_peak_abs"])[:, 4:6])
        )
    for label, target_s in HORIZONS:
        pitch_sample = _sample_at_or_before(time_s, pitch_deg, target_s)
        rate_sample = _sample_at_or_before(time_s, pitch_rate, target_s)
        summary[f"sample_time_{label}_s"] = None if pitch_sample is None else pitch_sample[0]
        summary[f"pitch_deg_at_{label}_s"] = None if pitch_sample is None else round(pitch_sample[1], 12)
        summary[f"pitch_rate_rad_s_at_{label}_s"] = None if rate_sample is None else rate_sample[1]
        summary[f"pitch_abs_reduction_deg_at_{label}_s"] = (
            None
            if pitch_sample is None
            else round(abs(initial_pitch_deg) - abs(pitch_sample[1]), 12)
        )
    return summary


def _response_at_horizon(
    row: dict[str, Any],
    zero: dict[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    pitch = row[f"pitch_deg_at_{label}_s"]
    rate = row[f"pitch_rate_rad_s_at_{label}_s"]
    response = {
        "available": pitch is not None and rate is not None,
        "sample_time_s": row[f"sample_time_{label}_s"],
        "pitch_deg": pitch,
        "pitch_rate_rad_s": rate,
        "abs_pitch_difference_vs_zero_deg": None,
        "pitch_rate_difference_vs_zero_rad_s": None,
    }
    if response["available"] and zero is not None:
        zero_pitch = zero[f"pitch_deg_at_{label}_s"]
        zero_rate = zero[f"pitch_rate_rad_s_at_{label}_s"]
        if zero_pitch is not None and zero_rate is not None:
            response["abs_pitch_difference_vs_zero_deg"] = abs(float(pitch)) - abs(float(zero_pitch))
            response["pitch_rate_difference_vs_zero_rad_s"] = float(rate) - float(zero_rate)
    return response


def compare_pitch_runs(summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[float, list[dict[str, Any]]] = {}
    for summary in summaries:
        if summary["scenario_kind"] != "grounded_pitch":
            continue
        key = round(float(summary["initial_pitch_deg"]), 9)
        groups.setdefault(key, []).append(summary)

    comparisons = []
    for initial_pitch_deg, rows in sorted(groups.items()):
        zero_rows = [row for row in rows if abs(float(row["wheel_common_action"])) <= 1.0e-12]
        zero = zero_rows[0] if len(zero_rows) == 1 else None
        responses = []
        for row in sorted(rows, key=lambda item: float(item["wheel_common_action"])):
            responses.append(
                {
                    "wheel_common_action": float(row["wheel_common_action"]),
                    "duration_s": float(row["duration_s"]),
                    "stopped_reason": row["stopped_reason"],
                    "minimum_abs_pitch_deg": row["minimum_abs_pitch_deg"],
                    "minimum_abs_pitch_time_s": row["minimum_abs_pitch_time_s"],
                    "pitch_rate_at_minimum_abs_pitch_rad_s": row[
                        "pitch_rate_at_minimum_abs_pitch_rad_s"
                    ],
                    "first_zero_crossing_time_s": row["first_zero_crossing_time_s"],
                    "max_opposite_signed_pitch_deg": row["max_opposite_signed_pitch_deg"],
                    "at_0p10_s": _response_at_horizon(row, zero, "0p10"),
                    "at_0p20_s": _response_at_horizon(row, zero, "0p20"),
                }
            )
        comparisons.append(
            {
                "initial_pitch_deg": initial_pitch_deg,
                "run_count": len(rows),
                "zero_action_available": zero is not None,
                "responses": responses,
            }
        )
    return comparisons


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refusing to write an empty isolation summary")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MuJoCo isolation traces.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = [summarize_run(run) for run in args.runs]
    comparisons = compare_pitch_runs(summaries)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "run_summary.csv", summaries)
    payload = {
        "schema_version": "MujocoIsolationAnalysisV2",
        "runs": summaries,
        "pitch_response_groups": comparisons,
    }
    write_json(args.output / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
