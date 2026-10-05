"""Run A1 PACE fitting with encoder-bias parameters fixed at zero.

This is the same replay/optimization loop as ``fit.py`` with one intentional
contract change: the 12 encoder-bias dimensions in the 49-value A1 mean are
not allowed to absorb a joint-coordinate offset.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PACE A1 fit with encoder_bias fixed to zero."
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4096,
        help="Number of environments to simulate.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Pace-A1-v0",
        help="Name of the registered PACE task.",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="a1_no_encoder_bias",
        help="PACE log subdirectory name under logs/pace/.",
    )
    parser.add_argument(
        "--bias-half-width",
        type=float,
        default=0.0,
        help=(
            "Half-width for encoder-bias bounds. Use 0 for exact zero bounds; "
            "use a tiny value such as 1e-9 only if the optimizer rejects "
            "zero-width bounds."
        ),
    )
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
from pace_sim2real import CMAESOptimizer  # noqa: E402
from pace_sim2real.utils import project_root  # noqa: E402


def _apply_zero_encoder_bias_bounds(env_cfg, half_width: float) -> None:
    if half_width < 0.0:
        raise ValueError("--bias-half-width must be non-negative")
    bounds = env_cfg.sim2real.bounds_params.clone()
    bounds[36:48, 0] = -half_width
    bounds[36:48, 1] = half_width
    env_cfg.sim2real.bounds_params = bounds
    env_cfg.sim2real.robot_name = args_cli.robot_name


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    _apply_zero_encoder_bias_bounds(env_cfg, args_cli.bias_half_width)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        print(
            "[INFO]: encoder_bias bounds fixed to "
            f"[{-args_cli.bias_half_width}, {args_cli.bias_half_width}]"
        )

        bounds_params = env_cfg.sim2real.bounds_params.to(env.unwrapped.device)
        articulation = env.unwrapped.scene["robot"]
        joint_order = env_cfg.sim2real.joint_order
        sim_joint_ids = torch.tensor(
            [articulation.joint_names.index(name) for name in joint_order],
            device=env.unwrapped.device,
        )

        data_file = project_root() / "data" / env_cfg.sim2real.data_dir
        log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name

        data = torch.load(data_file)
        time_data = data["time"].to(env.unwrapped.device)
        target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
        measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)

        initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(
            env.unwrapped.num_envs, 1
        )

        time_steps = time_data.shape[0]
        sim_dt = env.unwrapped.sim.cfg.dt

        opt = CMAESOptimizer(
            bounds=bounds_params,
            population_size=env.unwrapped.num_envs,
            log_dir=log_dir,
            joint_order=joint_order,
            max_iteration=env_cfg.sim2real.cmaes.max_iteration,
            data=data,
            device=env.unwrapped.device,
            epsilon=env_cfg.sim2real.cmaes.epsilon,
            sigma=env_cfg.sim2real.cmaes.sigma,
            save_interval=env_cfg.sim2real.cmaes.save_interval,
            save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process,
        )

        env.reset()
        opt.update_simulator(articulation, sim_joint_ids, initial_dof_pos)

        counter = 0
        while simulation_app.is_running():
            with torch.inference_mode():
                opt.tell(
                    env.unwrapped.scene.articulations["robot"].data.joint_pos[
                        :, sim_joint_ids
                    ],
                    measured_dof_pos[counter, :].unsqueeze(0).repeat(
                        env.unwrapped.num_envs, 1
                    ),
                )
                actions = torch.zeros(
                    env.action_space.shape, device=env.unwrapped.device
                )
                actions[:, sim_joint_ids] = target_dof_pos[counter, :].unsqueeze(
                    0
                ).repeat(env.unwrapped.num_envs, 1)
                env.step(actions)
                counter += 1
                if counter % 400 == 0:
                    print(
                        f"[INFO]: Step {counter * sim_dt:.1f} / "
                        f"{time_data[-1]:.1f} seconds "
                        f"({counter / time_steps * 100:.1f} %)"
                    )
                if counter >= time_steps:
                    print("[INFO]: Reached the end of the trajectory, evolving.")
                    counter = 0
                    opt.evolve()
                    if opt.finished():
                        break
                    env.reset()
                    opt.update_simulator(
                        env.unwrapped.scene["robot"], sim_joint_ids, initial_dof_pos
                    )
        opt.close()
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
