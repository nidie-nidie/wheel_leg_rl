"""Cross-validate one fitted A1 PACE mean by replaying fixed real chirp data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay A1 chirp data with one frozen fitted PACE mean."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import pace_sim2real.tasks  # noqa: F401, E402
from pace_sim2real.tasks.manager_based.pace.a1_mean import (  # noqa: E402
    A1_PACE_JOINT_ORDER,
    load_a1_mean,
)
from pace_sim2real.tasks.manager_based.pace.a1_replay import (  # noqa: E402
    inject_a1_actuator_before_make,
    make_a1_pace_actuator_cfg,
    replay_absolute_targets,
    validate_a1_pace_data,
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".json", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
            newline="\n",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _error_metrics(error: torch.Tensor) -> dict[str, Any]:
    absolute = error.abs()
    per_joint_rmse = torch.sqrt(torch.mean(error.square(), dim=0))
    per_joint_p95 = torch.quantile(absolute, 0.95, dim=0)
    per_joint_max_abs = torch.max(absolute, dim=0).values
    per_joint_mean = torch.mean(error, dim=0)
    per_joint_std = torch.std(error, dim=0)
    return {
        "aggregate_rmse_rad": float(torch.sqrt(torch.mean(error.square()))),
        "aggregate_p95_rad": float(torch.quantile(absolute.flatten(), 0.95)),
        "aggregate_max_abs_rad": float(torch.max(absolute)),
        "aggregate_mean_abs_rad": float(torch.mean(absolute)),
        "per_joint": [
            {
                "joint": name,
                "mean_error_rad": float(per_joint_mean[index]),
                "std_error_rad": float(per_joint_std[index]),
                "rmse_rad": float(per_joint_rmse[index]),
                "p95_rad": float(per_joint_p95[index]),
                "max_abs_rad": float(per_joint_max_abs[index]),
            }
            for index, name in enumerate(A1_PACE_JOINT_ORDER)
        ],
    }


def replay_once(args: argparse.Namespace) -> dict[str, Any]:
    data_path = args.data.resolve()
    mean_path = args.mean.resolve()
    output_path = args.output.resolve()
    trajectory_path = (
        args.trajectory_output.resolve() if args.trajectory_output is not None else None
    )
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not mean_path.is_file():
        raise FileNotFoundError(mean_path)

    data = torch.load(data_path, map_location="cpu", weights_only=True)
    validate_a1_pace_data(data)
    mean = load_a1_mean(mean_path)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    inject_a1_actuator_before_make(env_cfg, make_a1_pace_actuator_cfg(mean))
    env = gym.make(args.task, cfg=env_cfg)
    try:
        physical_samples = replay_absolute_targets(env, data, mean.encoder_bias)
    finally:
        env.close()

    measured = data["dof_pos"].detach().cpu().to(torch.float64)
    command = data["des_dof_pos"].detach().cpu().to(torch.float64)
    physical = physical_samples.detach().cpu().to(torch.float64)
    bias = torch.tensor(
        [mean.encoder_bias[name] for name in A1_PACE_JOINT_ORDER],
        dtype=torch.float64,
    )
    encoder = physical - bias
    encoder_error = encoder - measured
    physical_error = physical - measured
    command_error = command - measured

    report = {
        "schema_version": "a1_pace_crossval_replay/v1",
        "task": args.task,
        "data_path": str(data_path),
        "mean_path": str(mean_path),
        "data_sha256": _sha256_path(data_path),
        "mean_sha256": _sha256_path(mean_path),
        "joint_order": list(A1_PACE_JOINT_ORDER),
        "sample_count": int(measured.shape[0]),
        "duration_s": float(data["time"][-1].item() - data["time"][0].item()),
        "delay_steps": int(mean.delay_steps),
        "encoder_bias_rad": {
            name: float(mean.encoder_bias[name]) for name in A1_PACE_JOINT_ORDER
        },
        "encoder_frame_metrics": _error_metrics(encoder_error),
        "physical_frame_metrics": _error_metrics(physical_error),
        "command_vs_measured_metrics": _error_metrics(command_error),
    }
    _atomic_json(report, output_path)

    if trajectory_path is not None:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "time": data["time"].detach().cpu(),
                "joint_order": list(A1_PACE_JOINT_ORDER),
                "measured_encoder": measured.to(torch.float32),
                "command_encoder": command.to(torch.float32),
                "simulated_physical": physical.to(torch.float32),
                "simulated_encoder": encoder.to(torch.float32),
                "encoder_error": encoder_error.to(torch.float32),
            },
            trajectory_path,
        )

    return report


def main(argv: Sequence[str] | None = None) -> int:
    if argv is not None:
        raise RuntimeError("This script parses arguments before launching Isaac Sim.")
    try:
        report = replay_once(args_cli)
        print(json.dumps(report, sort_keys=True))
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
