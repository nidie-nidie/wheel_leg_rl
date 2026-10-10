from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .compare_traces import load_verified_engine_run
from .trace_schema import build_file_hashes, write_json


HORIZONS_S = (0.005, 0.02, 0.10, 0.20)
RUN_NAMES = {
    "isaac": {
        "zero": "isaac-zero",
        "positive": "isaac-plus4",
        "negative": "isaac-minus4",
    },
    "mujoco": {
        "zero": "mujoco-zero",
        "positive": "mujoco-plus4",
        "negative": "mujoco-minus4",
    },
}


def pitch_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError(f"Expected wxyz quaternion, got shape {values.shape}")
    w, x, y, z = np.moveaxis(values, -1, 0)
    return np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))


def centered_response(
    zero: np.ndarray | float,
    positive: np.ndarray | float,
    negative: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.asarray(zero, dtype=np.float64)
    plus = np.asarray(positive, dtype=np.float64)
    minus = np.asarray(negative, dtype=np.float64)
    return (plus - minus) / 2.0, (plus + minus) / 2.0 - baseline


def _sample_index(trace: dict[str, np.ndarray], target_time_s: float) -> tuple[int, float]:
    times = np.asarray(trace["physics_time_s"], dtype=np.float64)
    index = int(np.argmin(np.abs(times - target_time_s)))
    physics_dt = float(np.median(np.diff(times)))
    if abs(float(times[index]) - target_time_s) > physics_dt * 0.51:
        raise ValueError(f"No substep sample near {target_time_s:.6f} s")
    return index, float(times[index])


def _state(trace: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    return {
        "pitch_rad": np.asarray(
            pitch_from_wxyz(trace["base_orientation_control_wxyz_post_step"][index])
        ),
        "pitch_rate_rad_s": np.asarray(
            trace["base_angular_velocity_control_post_step"][index, 1]
        ),
        "vx_m_s": np.asarray(trace["base_linear_velocity_control_post_step"][index, 0]),
        "phi0_rad": np.asarray(trace["virtual_leg_phi0_post_step"][index]),
        "wheel_velocity_rad_s": np.asarray(
            trace["active_joint_velocity_canonical_post_step"][index, 4:6]
        ),
        "wheel_pd_torque_nm": np.asarray(
            trace["pd_torque_effort_clipped_canonical"][index, 4:6]
        ),
        "wheel_normal_force_n": np.asarray(trace["wheel_normal_force_n"][index]),
    }


def _json_value(value: np.ndarray | float) -> float | list[float]:
    array = np.asarray(value, dtype=np.float64)
    return float(array) if array.ndim == 0 else array.tolist()


def _validate_triplet(engine: str, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata = {name: run["metadata"] for name, run in runs.items()}
    if {item["engine"] for item in metadata.values()} != {
        "isaac_sim" if engine == "isaac" else "mujoco"
    }:
        raise ValueError(f"{engine} triplet contains a different engine")
    identities = {
        item["action_sequence_identity"]["content_sha256"] for item in metadata.values()
    }
    if len(identities) != 1:
        raise ValueError(f"{engine} triplet does not use one action sequence")
    if {tuple(item["command"]) for item in metadata.values()} != {(0.0, 0.0, 0.2)}:
        raise ValueError(f"{engine} triplet command differs from the stand experiment")

    zero = metadata["zero"]["base_pitch_torque_pulse"]
    positive = metadata["positive"]["base_pitch_torque_pulse"]
    negative = metadata["negative"]["base_pitch_torque_pulse"]
    if zero != {"magnitude_nm": 0.0, "start_tick": 0, "end_tick": 0}:
        raise ValueError(f"{engine} zero run contains a torque pulse")
    if positive["magnitude_nm"] <= 0.0 or negative["magnitude_nm"] != -positive["magnitude_nm"]:
        raise ValueError(f"{engine} torque magnitudes are not signed opposites")
    if (positive["start_tick"], positive["end_tick"]) != (
        negative["start_tick"],
        negative["end_tick"],
    ):
        raise ValueError(f"{engine} torque intervals differ")
    return positive


def _summarize_engine(engine: str, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pulse = _validate_triplet(engine, runs)
    control_dt = float(runs["positive"]["metadata"]["control_dt_s"])
    start_time = int(pulse["start_tick"]) * control_dt
    traces = {name: run["substeps"] for name, run in runs.items()}
    horizons: dict[str, Any] = {}
    for horizon in HORIZONS_S:
        target = start_time + horizon
        samples = {}
        actual_times = {}
        for name, trace in traces.items():
            index, actual_time = _sample_index(trace, target)
            samples[name] = _state(trace, index)
            actual_times[name] = actual_time
        signals = {}
        for signal in samples["zero"]:
            odd, even_bias = centered_response(
                samples["zero"][signal],
                samples["positive"][signal],
                samples["negative"][signal],
            )
            signals[signal] = {
                "odd_response": _json_value(odd),
                "even_bias": _json_value(even_bias),
            }
        horizons[f"{horizon:.3f}"] = {
            "target_time_s": target,
            "actual_times_s": actual_times,
            "signals": signals,
        }

    impulses = {}
    for name, trace in traces.items():
        mask = (trace["control_tick"] >= pulse["start_tick"]) & (
            trace["control_tick"] < pulse["end_tick"]
        )
        impulses[name] = np.sum(trace["wheel_normal_impulse_ns"][mask], axis=0)
    odd_impulse, impulse_bias = centered_response(
        impulses["zero"], impulses["positive"], impulses["negative"]
    )
    return {
        "pulse": pulse,
        "horizons": horizons,
        "normal_contact_impulse_during_pulse_ns": {
            "zero": impulses["zero"].tolist(),
            "positive": impulses["positive"].tolist(),
            "negative": impulses["negative"].tolist(),
            "odd_response": odd_impulse.tolist(),
            "even_bias": impulse_bias.tolist(),
        },
        "completed_control_ticks": {
            name: int(run["metadata"]["completed_control_ticks"])
            for name, run in runs.items()
        },
        "stopped_reason": {
            name: run["metadata"]["stopped_reason"] for name, run in runs.items()
        },
    }


def _relative_error(isaac: Any, mujoco: Any) -> dict[str, Any]:
    lhs = np.asarray(isaac, dtype=np.float64)
    rhs = np.asarray(mujoco, dtype=np.float64)
    absolute = np.abs(rhs - lhs)
    denominator = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0e-12)
    return {
        "absolute": _json_value(absolute),
        "relative": _json_value(absolute / denominator),
    }


def analyze_experiment(experiment_root: str | Path, output_directory: str | Path) -> dict[str, Any]:
    root = Path(experiment_root).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for engine, names in RUN_NAMES.items():
        loaded[engine] = {
            role: load_verified_engine_run(root / directory) for role, directory in names.items()
        }
    engines = {
        engine: _summarize_engine(engine, runs) for engine, runs in loaded.items()
    }
    cross_engine: dict[str, Any] = {}
    for horizon in ("0.020", "0.100", "0.200"):
        signal_metrics = {}
        isaac_signals = engines["isaac"]["horizons"][horizon]["signals"]
        mujoco_signals = engines["mujoco"]["horizons"][horizon]["signals"]
        for signal in isaac_signals:
            signal_metrics[signal] = _relative_error(
                isaac_signals[signal]["odd_response"],
                mujoco_signals[signal]["odd_response"],
            )
        cross_engine[horizon] = signal_metrics
    summary = {
        "schema_version": "CrossEnginePitchTorqueResponseV1",
        "experiment_root": str(root),
        "method": "odd=(positive-negative)/2; even_bias=(positive+negative)/2-zero",
        "engines": engines,
        "cross_engine_odd_response_error": cross_engine,
    }
    summary_path = write_json(output / "summary.json", summary)
    input_paths = {"analyzer": Path(__file__), "summary": summary_path}
    for engine, names in RUN_NAMES.items():
        for role, directory in names.items():
            run = root / directory
            prefix = f"{engine}_{role}"
            input_paths[f"{prefix}_metadata"] = run / "metadata.json"
            input_paths[f"{prefix}_control"] = run / "control_trace.npz"
            input_paths[f"{prefix}_substeps"] = run / "substep_trace.npz"
    write_json(output / "file_hashes.json", build_file_hashes(input_paths))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze matched cross-engine pitch torque runs.")
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze_experiment(args.experiment_root, args.output)


if __name__ == "__main__":
    main()
