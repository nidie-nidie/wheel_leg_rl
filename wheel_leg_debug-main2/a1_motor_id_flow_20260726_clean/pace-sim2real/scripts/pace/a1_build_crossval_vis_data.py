"""Build compact visualization data for fixed-mean A1 cross-validation replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch


DATASETS = (
    {
        "id": "064152",
        "group": "changba01-three",
        "label": "064152 0.1-10Hz amp .20/.35/.35",
        "traj": "replay_cv_064152_from_071950.pt",
        "report": "replay_cv_064152_from_071950.json",
    },
    {
        "id": "071554",
        "group": "changba01-three",
        "label": "071554 0.1-10Hz amp .20/.38/.38",
        "traj": "replay_cv_071554_from_071950.pt",
        "report": "replay_cv_071554_from_071950.json",
    },
    {
        "id": "071950",
        "group": "changba01-three",
        "label": "071950 fit source amp .22/.38/.38",
        "traj": "replay_cv_071950_from_071950.pt",
        "report": "replay_cv_071950_from_071950.json",
    },
    {
        "id": "062823",
        "group": "extra-two",
        "label": "062823 0.1-4Hz v20 a250",
        "traj": "replay_cv_062823_from_071950.pt",
        "report": "replay_cv_062823_from_071950.json",
    },
    {
        "id": "063144",
        "group": "extra-two",
        "label": "063144 0.1-8Hz v20 a800",
        "traj": "replay_cv_063144_from_071950.pt",
        "report": "replay_cv_063144_from_071950.json",
    },
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def build(root: Path, output: Path, stride: int) -> dict[str, Any]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    result: dict[str, Any] = {
        "schema": "a1-fixed-mean-crossval-vis/v1",
        "source": {
            "mean": "/home/changba01/pace-sim2real/logs/pace/a1/26_07_26_02-50-16/mean_199.pt",
            "task": "Isaac-Pace-A1-v0",
            "stride": stride,
            "series": "measured_encoder vs simulated_encoder, optional command_encoder",
        },
        "datasets": [],
    }
    datasets = result["datasets"]
    assert isinstance(datasets, list)
    for item in DATASETS:
        report = _load_json(root / item["report"])
        traj = torch.load(root / item["traj"], map_location="cpu", weights_only=False)
        time = traj["time"].double()
        count = int(time.numel())
        indices = list(range(0, count, stride))
        if indices[-1] != count - 1:
            indices.append(count - 1)
        index_tensor = torch.tensor(indices, dtype=torch.long)

        measured = traj["measured_encoder"][index_tensor].float()
        simulated = traj["simulated_encoder"][index_tensor].float()
        command = traj["command_encoder"][index_tensor].float()
        joint_order = list(traj["joint_order"])
        per_joint = report["encoder_frame_metrics"]["per_joint"]

        dataset: dict[str, Any] = {
            **item,
            "duration_s": round(float(time[-1] - time[0]), 3),
            "sample_count": count,
            "time": [round(float(value), 3) for value in time[index_tensor].tolist()],
            "metrics": {
                "rmse": round(
                    float(report["encoder_frame_metrics"]["aggregate_rmse_rad"]), 8
                ),
                "p95": round(
                    float(report["encoder_frame_metrics"]["aggregate_p95_rad"]), 8
                ),
                "max_abs": round(
                    float(report["encoder_frame_metrics"]["aggregate_max_abs_rad"]), 8
                ),
                "command_rmse": round(
                    float(
                        report["command_vs_measured_metrics"]["aggregate_rmse_rad"]
                    ),
                    8,
                ),
            },
            "joints": [],
        }
        joints = dataset["joints"]
        assert isinstance(joints, list)
        for joint_index, name in enumerate(joint_order):
            metrics = per_joint[joint_index]
            joints.append(
                {
                    "name": name,
                    "real": [
                        round(float(value), 5)
                        for value in measured[:, joint_index].tolist()
                    ],
                    "sim": [
                        round(float(value), 5)
                        for value in simulated[:, joint_index].tolist()
                    ],
                    "cmd": [
                        round(float(value), 5)
                        for value in command[:, joint_index].tolist()
                    ],
                    "rmse": round(float(metrics["rmse_rad"]), 8),
                    "p95": round(float(metrics["p95_rad"]), 8),
                    "maxAbs": round(float(metrics["max_abs_rad"]), 8),
                    "meanErr": round(float(metrics["mean_error_rad"]), 8),
                }
            )
        datasets.append(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":")), encoding="ascii")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data = build(args.root, args.output, args.stride)
    print(args.output)
    print(args.output.stat().st_size)
    for dataset in data["datasets"]:
        print(dataset["id"], dataset["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
