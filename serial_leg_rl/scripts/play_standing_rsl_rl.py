#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = PROJECT_ROOT / "wheel_leg" / "WheelLeg_RL_IsaacLab" / "IsaacLab"
RSL_RL_SCRIPT_DIR = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
SERIAL_LEG_SOURCE = PROJECT_ROOT / "serial_leg_rl" / "source" / "serial_leg_rl"

sys.path.insert(0, SERIAL_LEG_SOURCE.as_posix())
sys.path.insert(0, RSL_RL_SCRIPT_DIR.as_posix())

import cli_args  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="Play the trained serial-leg standing policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video while playing.")
parser.add_argument("--video_length", type=int, default=1000, help="Number of simulation steps to record.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="SerialLeg-Standing-Direct-v0", help="Gym task id.")
parser.add_argument("--seed", type=int, default=None, help="Seed used by the environment and agent.")
parser.add_argument("--num_steps", type=int, default=3000, help="Number of simulation steps to play.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import serial_leg_rl.tasks  # noqa: F401,E402


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] 载入实验目录: {log_root_path}")
    print(f"[INFO] 载入 checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.unwrapped.sim.set_camera_view(eye=(-0.6, -2.4, 0.9), target=(0.02, -1.27, 0.15))
    if args_cli.video:
        video_folder = os.path.join(os.path.dirname(resume_path), "videos", "play")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 录制播放视频。")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    try:
        runner.load(resume_path, load_optimizer=False)
    except RuntimeError as err:
        if "size mismatch" in str(err):
            print("[ERROR] Checkpoint is not compatible with the current observation/action network shape.")
            print(f"[ERROR] Tried checkpoint: {resume_path}")
            print("[ERROR] This usually means the environment observation terms changed after that model was trained.")
            print("[ERROR] Use the newest checkpoint from the matching training run, or retrain from scratch.")
            env.close()
            raise SystemExit(2) from None
        raise
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    try:
        for _ in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
