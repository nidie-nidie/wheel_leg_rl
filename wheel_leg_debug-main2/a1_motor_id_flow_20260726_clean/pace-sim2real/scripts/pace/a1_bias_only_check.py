#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cmaes
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Estimate A1 encoder bias with fixed fitted motor dynamics.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--task", default="Isaac-Pace-A1-v0")
parser.add_argument("--mean-file", type=Path, required=True)
parser.add_argument("--data-file", type=Path, action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--iterations", type=int, default=12)
parser.add_argument("--sigma", type=float, default=0.15)
parser.add_argument("--save-trajectories", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import pace_sim2real.tasks  # noqa: F401,E402
from pace_sim2real.tasks.manager_based.pace.a1_mean import A1_PACE_JOINT_ORDER  # noqa: E402


def _load_tensor_file(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu")
    if set(value) != {"time", "dof_pos", "des_dof_pos"}:
        raise ValueError(f"{path} does not match the A1 PACE data contract")
    return value


def _transform(normalized: torch.Tensor) -> torch.Tensor:
    return normalized * 0.1


def _inverse_transform(bias: torch.Tensor) -> torch.Tensor:
    return torch.clamp(bias / 0.1, -0.999, 0.999)


def _update_simulator(
    articulation: Any,
    joint_ids: torch.Tensor,
    initial_position: torch.Tensor,
    armature: torch.Tensor,
    viscous: torch.Tensor,
    friction: torch.Tensor,
    bias: torch.Tensor,
    delay_steps: int,
) -> None:
    env_ids = torch.arange(bias.shape[0], device=bias.device)
    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(viscous, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = viscous
    articulation.write_joint_dynamic_friction_coefficient_to_sim(0.0, joint_ids=joint_ids, env_ids=env_ids)
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction
    articulation.write_joint_dynamic_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_dynamic_friction_coeff[:, joint_ids] = friction
    articulation.write_joint_position_to_sim(initial_position + bias, joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim(torch.zeros_like(initial_position), joint_ids=joint_ids)
    delay = torch.full((bias.shape[0],), delay_steps, device=bias.device, dtype=torch.int64)
    for actuator in articulation.actuators.values():
        drive_indices = actuator.joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0)
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        actuator.update_encoder_bias(bias[:, drive_joint_idx])
        actuator.update_time_lags(delay)
        actuator.reset(env_ids)


def _evaluate_generation(
    env: Any,
    articulation: Any,
    joint_ids: torch.Tensor,
    initial_position: torch.Tensor,
    armature: torch.Tensor,
    viscous: torch.Tensor,
    friction: torch.Tensor,
    bias: torch.Tensor,
    delay_steps: int,
    measured: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    env.reset()
    _update_simulator(
        articulation,
        joint_ids,
        initial_position,
        armature,
        viscous,
        friction,
        bias,
        delay_steps,
    )
    per_joint_sse = torch.zeros((bias.shape[0], len(A1_PACE_JOINT_ORDER)), device=bias.device)
    actions = torch.zeros(env.action_space.shape, device=bias.device)
    for index in range(measured.shape[0]):
        sim_pos = articulation.data.joint_pos[:, joint_ids]
        error = sim_pos - measured[index, :].unsqueeze(0) - bias
        per_joint_sse += error * error
        actions.zero_()
        actions[:, joint_ids] = targets[index, :].unsqueeze(0)
        env.step(actions)
        if (index + 1) % 2000 == 0:
            print(f"[bias-check] step {index + 1}/{measured.shape[0]}", flush=True)
    per_joint_mse = per_joint_sse / measured.shape[0]
    objective = torch.sum(per_joint_mse, dim=1)
    return objective, per_joint_mse


def _fit_dataset(
    env: Any,
    articulation: Any,
    joint_ids: torch.Tensor,
    data_file: Path,
    mean: torch.Tensor,
    *,
    iterations: int,
    sigma: float,
) -> dict[str, Any]:
    device = torch.device(env.unwrapped.device)
    data = _load_tensor_file(data_file)
    measured = data["dof_pos"].to(device=device, dtype=torch.float32)
    targets = data["des_dof_pos"].to(device=device, dtype=torch.float32)
    initial_position = measured[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

    armature = mean[:12].to(device=device, dtype=torch.float32).unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    viscous = mean[12:24].to(device=device, dtype=torch.float32).unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    friction = mean[24:36].to(device=device, dtype=torch.float32).unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    initial_bias = mean[36:48].to(dtype=torch.float32)
    delay_steps = int(mean[48].to(torch.int).item())

    optimizer = cmaes.CMA(
        mean=_inverse_transform(initial_bias).numpy(),
        sigma=sigma,
        bounds=torch.stack(
            [
                torch.full((12,), -1.0, dtype=torch.float32),
                torch.full((12,), 1.0, dtype=torch.float32),
            ],
            dim=1,
        ).numpy(),
        seed=1,
        population_size=env.unwrapped.num_envs,
    )
    best_score = float("inf")
    best_bias: torch.Tensor | None = None
    best_per_joint: torch.Tensor | None = None

    for iteration in range(iterations):
        normalized = torch.tensor([optimizer.ask() for _ in range(env.unwrapped.num_envs)], device=device, dtype=torch.float32)
        bias = _transform(normalized)
        objective, per_joint_mse = _evaluate_generation(
            env,
            articulation,
            joint_ids,
            initial_position,
            armature,
            viscous,
            friction,
            bias,
            delay_steps,
            measured,
            targets,
        )
        solutions = [
            (normalized[index].detach().cpu().numpy(), float(objective[index].item()))
            for index in range(env.unwrapped.num_envs)
        ]
        optimizer.tell(solutions)
        min_index = int(torch.argmin(objective).item())
        min_score = float(objective[min_index].item())
        if min_score < best_score:
            best_score = min_score
            best_bias = bias[min_index].detach().cpu()
            best_per_joint = per_joint_mse[min_index].detach().cpu()
        print(
            f"[bias-check] {data_file.name} iteration {iteration + 1}/{iterations} "
            f"rmse={(min_score / 12.0) ** 0.5:.6f} best_rmse={(best_score / 12.0) ** 0.5:.6f}",
            flush=True,
        )

    assert best_bias is not None and best_per_joint is not None
    return {
        "data_file": str(data_file),
        "delay_steps_truncated": delay_steps,
        "iterations": iterations,
        "num_envs": env.unwrapped.num_envs,
        "bias": {
            name: float(value)
            for name, value in zip(A1_PACE_JOINT_ORDER, best_bias.tolist(), strict=True)
        },
        "per_joint_rmse": {
            name: float(value)
            for name, value in zip(A1_PACE_JOINT_ORDER, torch.sqrt(best_per_joint).tolist(), strict=True)
        },
        "objective_rmse_mean_rad": float((best_score / 12.0) ** 0.5),
    }


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    articulation = env.unwrapped.scene["robot"]
    joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in A1_PACE_JOINT_ORDER],
        device=env.unwrapped.device,
        dtype=torch.long,
    )
    mean = torch.load(args_cli.mean_file, map_location="cpu", weights_only=True).detach().cpu().to(torch.float32)
    results = [
        _fit_dataset(
            env,
            articulation,
            joint_ids,
            data_file.resolve(),
            mean,
            iterations=args_cli.iterations,
            sigma=args_cli.sigma,
        )
        for data_file in args_cli.data_file
    ]
    env.close()
    payload = {
        "mean_file": str(args_cli.mean_file.resolve()),
        "joint_order": list(A1_PACE_JOINT_ORDER),
        "results": results,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
