from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


def error_metrics(
    error: Sequence[float],
    fitted_delay_steps: int,
    estimated_delay_steps: Optional[int] = None,
    saturated: Optional[Sequence[bool]] = None,
) -> dict:
    values = np.asarray(error, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cross-validation requires at least one finite error sample")
    absolute = np.abs(values)
    if saturated is None:
        saturation_rate = 0.0
    else:
        saturation = np.asarray(saturated, dtype=np.bool_)
        saturation_rate = float(np.mean(saturation)) if saturation.size else 0.0
    estimated = fitted_delay_steps if estimated_delay_steps is None else estimated_delay_steps
    return {
        "sample_count": int(values.size),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "p95_abs_error": float(np.percentile(absolute, 95.0)),
        "max_abs_error": float(np.max(absolute)),
        "estimated_delay_steps": int(estimated),
        "delay_error_steps": int(estimated - fitted_delay_steps),
        "saturation_rate": saturation_rate,
    }


def aggregate_metrics(metric_sets: Iterable[Mapping[str, float]]) -> dict:
    metrics = list(metric_sets)
    if not metrics:
        return {
            "channel_count": 0,
            "sample_count": 0,
            "rmse_mean": 0.0,
            "rmse_max": 0.0,
            "p95_abs_error_max": 0.0,
            "max_abs_error": 0.0,
            "saturation_rate_mean": 0.0,
        }
    return {
        "channel_count": len(metrics),
        "sample_count": int(sum(int(item["sample_count"]) for item in metrics)),
        "rmse_mean": float(np.mean([item["rmse"] for item in metrics])),
        "rmse_max": float(np.max([item["rmse"] for item in metrics])),
        "p95_abs_error_max": float(np.max([item["p95_abs_error"] for item in metrics])),
        "max_abs_error": float(np.max([item["max_abs_error"] for item in metrics])),
        "saturation_rate_mean": float(np.mean([item["saturation_rate"] for item in metrics])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fitted-model validation metrics")
    parser.add_argument("manifest", type=pathlib.Path)
    args = parser.parse_args()
    model = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = {}
    for motor in model.get("motors", []):
        output[motor.get("name", "unknown")] = motor.get("validation", {})
    output["aggregate"] = model.get("aggregate_metrics", {})
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
