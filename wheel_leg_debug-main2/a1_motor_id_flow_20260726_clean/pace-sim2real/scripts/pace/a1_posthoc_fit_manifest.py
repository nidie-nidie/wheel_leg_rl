#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import torch


MEAN_PATTERN = re.compile(r"^mean_(?P<index>[0-9]+)\.pt$")
SCHEMA_VERSION = "a1_pace_fit_manifest/v1"
TASK = "Isaac-Pace-A1-v0"
A1_PACE_JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def named(values: torch.Tensor) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(A1_PACE_JOINT_ORDER, values.detach().cpu().tolist(), strict=True)
    }


def parse_mean(mean: torch.Tensor) -> dict[str, Any]:
    if mean.shape != (49,) or not torch.is_floating_point(mean) or not bool(torch.isfinite(mean).all().item()):
        raise ValueError("A1 PACE mean must be a finite floating tensor with shape (49,)")
    return {
        "armature": named(mean[:12]),
        "viscous_friction": named(mean[12:24]),
        "coulomb_friction": named(mean[24:36]),
        "encoder_bias": named(mean[36:48]),
        "delay_steps": int(round(float(mean[48].item()))),
    }


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot read git HEAD for {path}: {result.stderr.strip()}")
    head = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", head):
        raise ValueError(f"invalid git HEAD for {path}: {head!r}")
    return head


def latest_run(log_root: Path) -> Path:
    runs = [path for path in log_root.iterdir() if path.is_dir() and not path.is_symlink()]
    if not runs:
        raise ValueError(f"no PACE runs found under {log_root}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def select_mean(run_dir: Path) -> tuple[Path, int]:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("mean_*.pt"):
        match = MEAN_PATTERN.fullmatch(path.name)
        if match:
            candidates.append((int(match.group("index")), path))
    if not candidates:
        raise ValueError(f"no mean_*.pt found in {run_dir}")
    return max(candidates, key=lambda item: item[0])[1], max(candidates)[0]


def require_complete(run_dir: Path, mean_index: int, *, max_iteration: int) -> None:
    if mean_index != max_iteration - 1:
        raise ValueError(
            f"fit is incomplete: selected mean index {mean_index}, expected {max_iteration - 1}"
        )
    progress_path = run_dir / "progress.pt"
    if not progress_path.is_file():
        raise ValueError(f"completed fit must contain progress.pt: {progress_path}")


def fit_metrics(run_dir: Path, mean_path: Path) -> dict[str, Any]:
    config = torch_load(run_dir / "config.pt")
    best = torch_load(run_dir / "best_trajectory.pt").detach().cpu().to(torch.float64)
    mean = torch_load(mean_path).detach().cpu().to(torch.float32)
    parsed = parse_mean(mean)
    real = config["dof_pos"].detach().cpu().to(torch.float64)
    desired = config["des_dof_pos"].detach().cpu().to(torch.float64)
    time = config["time"].detach().cpu().to(torch.float64)
    if best.shape != real.shape or desired.shape != real.shape or real.shape[1] != len(A1_PACE_JOINT_ORDER):
        raise ValueError("trajectory/config tensor shapes do not match the A1 contract")
    bias = torch.tensor(
        [parsed["encoder_bias"][name] for name in A1_PACE_JOINT_ORDER],
        dtype=torch.float64,
    )
    direct_error = best - real
    objective_error = best - real - bias.unsqueeze(0)
    command_error = real - desired

    def per_joint_rmse(error: torch.Tensor) -> list[float]:
        return torch.sqrt(torch.mean(error * error, dim=0)).tolist()

    objective_score = torch.mean(torch.sum(objective_error * objective_error, dim=1)).item()
    return {
        "sample_count": int(real.shape[0]),
        "duration_s": float(time[-1].item() - time[0].item()),
        "objective_score": float(objective_score),
        "objective_rmse_mean_rad": float((objective_score / len(A1_PACE_JOINT_ORDER)) ** 0.5),
        "objective_rmse_per_joint_rad": per_joint_rmse(objective_error),
        "direct_rmse_mean_rad": float(torch.sqrt(torch.mean(direct_error * direct_error)).item()),
        "direct_rmse_per_joint_rad": per_joint_rmse(direct_error),
        "real_command_rmse_mean_rad": float(torch.sqrt(torch.mean(command_error * command_error)).item()),
        "encoder_bias": parsed["encoder_bias"],
        "delay_steps": parsed["delay_steps"],
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = args.repository_root.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run(repository_root / "logs" / "pace" / "a1")
    mean_path, mean_index = select_mean(run_dir)
    if not args.allow_incomplete:
        require_complete(run_dir, mean_index, max_iteration=args.max_iteration)
    metrics = fit_metrics(run_dir, mean_path)
    status = "PASS"
    failures = []
    if metrics["objective_rmse_mean_rad"] > args.max_objective_rmse_rad:
        failures.append("objective_rmse")
    if metrics["direct_rmse_mean_rad"] > args.max_direct_rmse_rad:
        failures.append("direct_rmse")
    if failures:
        status = "FAIL"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "failure_reasons": failures,
        "physical_motion_authorized": False,
        "task": TASK,
        "num_envs": int(args.num_envs),
        "run_basename": run_dir.name,
        "mean_basename": mean_path.name,
        "mean_sha256": sha256_path(mean_path),
        "best_trajectory_basename": "best_trajectory.pt",
        "best_trajectory_sha256": sha256_path(run_dir / "best_trajectory.pt"),
        "config_basename": "config.pt",
        "config_sha256": sha256_path(run_dir / "config.pt"),
        "metrics": metrics,
        "acceptance_thresholds": {
            "max_objective_rmse_rad": float(args.max_objective_rmse_rad),
            "max_direct_rmse_rad": float(args.max_direct_rmse_rad),
        },
        "environment_contract": {
            "joint_order": list(A1_PACE_JOINT_ORDER),
            "fixture_pose": "upside_down_fixed_air",
            "fix_root_link": True,
            "kp": 25.0,
            "kd": 2.0,
            "dt_s": 0.002,
            "decimation": 1,
        },
        "repository_revisions": {
            "pace": git_head(repository_root),
            "isaaclab": git_head(args.isaaclab_root.resolve()),
            "a1_base": git_head(args.a1_base_root.resolve()),
            "gogo_learn": git_head(args.gogo_learn_root.resolve()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a hash-bound fit manifest for an existing A1 PACE run.")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iteration", type=int, default=200)
    parser.add_argument("--max-objective-rmse-rad", type=float, default=0.05)
    parser.add_argument("--max-direct-rmse-rad", type=float, default=0.08)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/changba01/IsaacLab"))
    parser.add_argument("--a1-base-root", type=Path, default=Path("/home/changba01/worktrees/A1_Base-a1-pace"))
    parser.add_argument("--gogo-learn-root", type=Path, default=Path("/home/changba01/worktrees/gogo-learn-a1-pace"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    atomic_json(args.output.resolve(), manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
