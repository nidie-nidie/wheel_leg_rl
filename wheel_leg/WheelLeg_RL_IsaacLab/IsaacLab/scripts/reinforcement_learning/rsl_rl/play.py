# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a trained RSL-RL policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play a trained policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video while playing.")
parser.add_argument("--video_length", type=int, default=1000, help="Number of simulation steps to record.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of simulation steps to play.")
parser.add_argument("--metrics_csv", type=str, default=None, help="Path to save headless rollout metrics as CSV.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_folder = os.path.join(os.path.dirname(resume_path), "videos", "play")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording play video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    steps = 0
    metrics_file = None
    metrics_writer = None

    if args_cli.metrics_csv:
        metrics_path = os.path.abspath(args_cli.metrics_csv)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        metrics_file = open(metrics_path, "w", newline="")
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(
            [
                "step",
                "reward",
                "done",
                "command_x_vel",
                "command_y_vel",
                "command_yaw_vel",
                "command_leg_length",
                "base_x_vel",
                "base_y_vel",
                "base_z_vel",
                "base_yaw_vel",
                "pitch",
                "roll",
                "yaw_total",
                "leg_length_left",
                "leg_length_right",
                "theta_left",
                "theta_right",
                "debug_raw_action_delta_all",
                "debug_raw_action_delta_leg_angle",
                "debug_raw_action_delta_leg_length",
                "debug_raw_action_delta_wheel",
                "debug_action_delta_all",
                "debug_action_delta_leg_angle",
                "debug_action_delta_leg_length",
                "debug_action_delta_wheel",
                "debug_action_limiter_error",
                "debug_leg_target_delta",
                "debug_wheel_target_delta",
                "debug_leg_target_error",
                "debug_wheel_target_error",
                "debug_leg_joint_vel_abs",
                "debug_wheel_joint_vel_abs",
                "debug_stand_raw_action_delta",
                "debug_stand_action_delta",
                "debug_stand_action_limiter_error",
                "debug_stand_leg_target_delta",
                "debug_stand_leg_target_error",
                "debug_stand_leg_joint_vel_abs",
                "debug_stand_wheel_joint_vel_abs",
            ]
        )

    def _scalar(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value[0].detach().cpu().item()

    try:
        while simulation_app.is_running() and steps < args_cli.num_steps:
            with torch.inference_mode():
                actions = policy(obs)
                obs, reward, dones, _ = env.step(actions)

            if metrics_writer is not None:
                base_env = env.unwrapped
                debug_log = getattr(base_env, "_debug_control_log", base_env._get_debug_control_log())
                metrics_writer.writerow(
                    [
                        steps,
                        _scalar(reward),
                        int(_scalar(dones)),
                        _scalar(base_env.commands[:, 0]),
                        _scalar(base_env.commands[:, 1]),
                        _scalar(base_env.commands[:, 2]),
                        _scalar(base_env.commands[:, 3]),
                        _scalar(base_env.wheellegrobot.data.root_lin_vel_b[:, 0]),
                        _scalar(base_env.wheellegrobot.data.root_lin_vel_b[:, 1]),
                        _scalar(base_env.wheellegrobot.data.root_lin_vel_b[:, 2]),
                        _scalar(base_env.Gyro[:, 2]),
                        _scalar(base_env.ins.Pitch.squeeze(1)),
                        _scalar(base_env.ins.Roll.squeeze(1)),
                        _scalar(base_env.ins.YawTotalAngle.squeeze(1)),
                        _scalar(base_env.leg_length_L.squeeze(1)),
                        _scalar(base_env.leg_length_R.squeeze(1)),
                        _scalar(base_env.theta_L.squeeze(1)),
                        _scalar(base_env.theta_R.squeeze(1)),
                        _scalar(debug_log["Debug/raw_action_delta_all"]),
                        _scalar(debug_log["Debug/raw_action_delta_leg_angle"]),
                        _scalar(debug_log["Debug/raw_action_delta_leg_length"]),
                        _scalar(debug_log["Debug/raw_action_delta_wheel"]),
                        _scalar(debug_log["Debug/action_delta_all"]),
                        _scalar(debug_log["Debug/action_delta_leg_angle"]),
                        _scalar(debug_log["Debug/action_delta_leg_length"]),
                        _scalar(debug_log["Debug/action_delta_wheel"]),
                        _scalar(debug_log["Debug/action_limiter_error"]),
                        _scalar(debug_log["Debug/leg_target_delta"]),
                        _scalar(debug_log["Debug/wheel_target_delta"]),
                        _scalar(debug_log["Debug/leg_target_error"]),
                        _scalar(debug_log["Debug/wheel_target_error"]),
                        _scalar(debug_log["Debug/leg_joint_vel_abs"]),
                        _scalar(debug_log["Debug/wheel_joint_vel_abs"]),
                        _scalar(debug_log["Debug/stand_raw_action_delta"]),
                        _scalar(debug_log["Debug/stand_action_delta"]),
                        _scalar(debug_log["Debug/stand_action_limiter_error"]),
                        _scalar(debug_log["Debug/stand_leg_target_delta"]),
                        _scalar(debug_log["Debug/stand_leg_target_error"]),
                        _scalar(debug_log["Debug/stand_leg_joint_vel_abs"]),
                        _scalar(debug_log["Debug/stand_wheel_joint_vel_abs"]),
                    ]
                )
            steps += 1
    finally:
        if metrics_file is not None:
            metrics_file.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
