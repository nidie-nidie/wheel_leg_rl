#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = PROJECT_ROOT / "wheel_leg" / "WheelLeg_RL_IsaacLab" / "IsaacLab"
SERIAL_LEG_SOURCE = PROJECT_ROOT / "serial_leg_rl" / "source" / "serial_leg_rl"

sys.path.insert(0, SERIAL_LEG_SOURCE.as_posix())

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="View the serial-leg standing environment without PPO.")
parser.add_argument("--task", type=str, default="SerialLeg-Standing-Direct-v0", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=3000, help="Number of environment steps to simulate.")
parser.add_argument("--print_interval", type=int, default=100, help="Print diagnostics every N environment steps.")
parser.add_argument("--real_time", action="store_true", help="Throttle stepping to roughly real time for GUI viewing.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import serial_leg_rl.tasks  # noqa: F401,E402


CLOSURE_POINT_PAIRS = (
    ("jIO_dummy_child_link1", "jMK_dummy_child1"),
    ("jIO_dummy_child_link2", "jMK_dummy_child2"),
    ("jAG_dummy_child_link1", "jEC_dummy_child_link1"),
    ("jAG_dummy_child_link2", "jEC_dummy_child_link2"),
)


@hydra_task_config(args_cli.task, None)
def main(env_cfg: DirectRLEnvCfg, _agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    base_env.sim.set_camera_view(eye=(-0.6, -2.4, 0.9), target=(0.02, -1.27, 0.15))
    actions = torch.zeros((base_env.num_envs, base_env.cfg.action_space), device=base_env.device)

    obs, _ = env.reset()
    del obs
    closure_point_ids: list[tuple[str, int, int]] = []
    for name_a, name_b in CLOSURE_POINT_PAIRS:
        try:
            body_ids, body_names = base_env.robot.find_bodies([name_a, name_b], preserve_order=True)
        except ValueError:
            body_ids = []
            body_names = []
        if len(body_ids) == 2:
            closure_point_ids.append((f"{body_names[0]}-{body_names[1]}", body_ids[0], body_ids[1]))

    print(
        "[VIEW] "
        f"max_episode_length={base_env.max_episode_length} "
        f"step_dt={base_env.step_dt:.5f} "
        f"episode_length_s={base_env.cfg.episode_length_s}"
    )

    try:
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break

            with torch.inference_mode():
                _obs, reward, terminated, truncated, _extras = env.step(actions)
            del _obs
            if args_cli.real_time:
                time.sleep(base_env.step_dt)

            done = terminated | truncated
            if step % args_cli.print_interval == 0 or torch.any(done):
                done_debug = base_env.extras.get("debug_dones", {})
                if done_debug:
                    episode_length = done_debug["episode_length"]
                    height = done_debug["height"]
                    tilt = done_debug["tilt"]
                    root_ang_vel = done_debug["root_ang_vel"]
                    root_lin_vel = done_debug["root_lin_vel"]
                    joint_vel = done_debug["joint_vel"]
                    tilt_bad = done_debug["tilt_bad"]
                    height_bad = done_debug["height_bad"]
                    root_ang_vel_bad = done_debug["root_ang_vel_bad"]
                    root_lin_vel_bad = done_debug["root_lin_vel_bad"]
                    joint_vel_bad = done_debug["joint_vel_bad"]
                    invalid = done_debug["invalid"]
                else:
                    episode_length = base_env.episode_length_buf
                    height = base_env.robot.data.root_pos_w[:, 2] - base_env.scene.env_origins[:, 2]
                    tilt = torch.linalg.norm(base_env.robot.data.projected_gravity_b[:, :2], dim=1)
                    root_ang_vel = torch.linalg.norm(base_env.robot.data.root_ang_vel_b, dim=1)
                    root_lin_vel = torch.linalg.norm(base_env.robot.data.root_lin_vel_w, dim=1)
                    joint_vel = torch.max(torch.abs(base_env.robot.data.joint_vel), dim=1).values
                    tilt_bad = tilt > torch.sin(torch.tensor(base_env.cfg.max_tilt_rad, device=base_env.device))
                    height_bad = (height < base_env.cfg.min_base_height) | (height > base_env.cfg.max_base_height)
                    root_ang_vel_bad = root_ang_vel > base_env.cfg.max_root_ang_vel
                    root_lin_vel_bad = root_lin_vel > base_env.cfg.max_root_lin_vel
                    joint_vel_bad = joint_vel > base_env.cfg.max_joint_vel
                    invalid = torch.zeros_like(tilt_bad)
                closure_text = "closure=unavailable"
                if closure_point_ids:
                    body_pos = base_env.robot.data.body_pos_w
                    distances = []
                    pair_texts = []
                    for pair_name, body_id_a, body_id_b in closure_point_ids:
                        distance = torch.linalg.norm(body_pos[:, body_id_a] - body_pos[:, body_id_b], dim=1)
                        distances.append(distance)
                        pair_texts.append(f"{pair_name}:{1000.0 * distance.max().item():.1f}mm")
                    stacked_distances = torch.stack(distances, dim=1)
                    closure_text = (
                        f"closure_max={1000.0 * stacked_distances.max().item():.1f}mm "
                        f"closure_pairs={';'.join(pair_texts)}"
                    )
                print(
                    "[VIEW] "
                    f"step={step} "
                    f"episode_step={int(episode_length.max().item())} "
                    f"reward={reward.mean().item():+.3f} "
                    f"height={height.mean().item():.3f} "
                    f"tilt={tilt.mean().item():.3f} "
                    f"root_ang_vel={root_ang_vel.max().item():.3f} "
                    f"root_lin_vel={root_lin_vel.max().item():.3f} "
                    f"max_joint_vel={joint_vel.max().item():.3f} "
                    f"terminated={int(terminated.sum().item())} "
                    f"timeout={int(truncated.sum().item())} "
                    f"done_count={int(done.sum().item())} "
                    f"reasons="
                    f"tilt:{int(tilt_bad.sum().item())},"
                    f"height:{int(height_bad.sum().item())},"
                    f"root_ang_vel:{int(root_ang_vel_bad.sum().item())},"
                    f"root_lin_vel:{int(root_lin_vel_bad.sum().item())},"
                    f"joint_vel:{int(joint_vel_bad.sum().item())},"
                    f"invalid:{int(invalid.sum().item())} "
                    f"{closure_text}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
