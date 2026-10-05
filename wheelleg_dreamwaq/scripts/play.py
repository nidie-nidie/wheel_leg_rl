from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import h5py  # noqa: F401
import tensordict  # noqa: F401
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAY_EXPERIENCE = PROJECT_ROOT / "apps" / "wheelleg_play_no_rtx.kit"

# On Windows, preload native extension DLLs before Kit changes the DLL search state.
from isaaclab.app import AppLauncher

from wheelleg_dreamwaq.training.runtime import validate_runtime

parser = argparse.ArgumentParser(description="Evaluate a trained WheelLeg Phase 1 PPO checkpoint.")
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--fixed-command",
    type=float,
    nargs=3,
    metavar=("VX", "YAW_RATE", "BASE_HEIGHT"),
    default=None,
    help="Hold one command for the complete evaluation instead of resampling commands.",
)
parser.add_argument("--real-time", action="store_true", help="Pace evaluation at the environment control period.")
parser.add_argument("--video-output", type=Path, default=None, help="Record evaluation to this MP4 file.")
parser.add_argument("--keyboard", action="store_true", help="Control the command from the Isaac Sim window.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
livestream = args_cli.livestream if args_cli.livestream >= 0 else int(os.environ.get("LIVESTREAM", 0))
headless = args_cli.headless or os.environ.get("HEADLESS", "0") == "1" or livestream in (1, 2)
if not headless and not args_cli.experience:
    args_cli.experience = str(PLAY_EXPERIENCE)
if args_cli.keyboard:
    if headless:
        parser.error("--keyboard requires the native Isaac Sim window")
    if args_cli.num_envs != 1:
        parser.error("--keyboard requires --num-envs 1")
    if args_cli.fixed_command is not None:
        parser.error("--keyboard cannot be combined with --fixed-command")
    if args_cli.video_output is not None:
        parser.error("--keyboard cannot be combined with --video-output")
    args_cli.real_time = True
if args_cli.video_output is not None:
    args_cli.enable_cameras = True
validate_runtime(PROJECT_ROOT, device=args_cli.device)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

if args_cli.keyboard:
    import carb
    import omni.appwindow

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.schemas.manifest import build_phase1_contract_from_configs
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.agents import WheelLegFlatPPORunnerCfg
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
from wheelleg_dreamwaq.training.checkpoint import validate_checkpoint_metadata


class KeyboardCommand:
    def __init__(self, direct_env: WheelLegFlatEnv):
        height_range = direct_env.cfg.commands.base_height
        self.command = direct_env._commands.new_tensor((0.0, 0.0, sum(height_range) / 2.0))
        self.reset_requested = False
        self.quit_requested = False
        self._ranges = direct_env.cfg.commands
        self._input = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("Isaac Sim did not create a native application window")
        self._keyboard = app_window.get_keyboard()
        if self._keyboard is None:
            raise RuntimeError("Isaac Sim native window did not expose a keyboard")
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)
        print(
            "[INFO] Keyboard: W/S forward/back, A/D yaw, Q/E height, "
            "Space stop, R reset, Esc exit. Click the viewport first.",
            flush=True,
        )

    def close(self) -> None:
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)

    def _on_event(self, event) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = event.input.name
        if key in ("W", "UP"):
            self.command[0] += 0.05
        elif key in ("S", "DOWN"):
            self.command[0] -= 0.05
        elif key in ("A", "LEFT"):
            self.command[1] += 0.10
        elif key in ("D", "RIGHT"):
            self.command[1] -= 0.10
        elif key == "Q":
            self.command[2] -= 0.005
        elif key == "E":
            self.command[2] += 0.005
        elif key == "SPACE":
            self.command[:2].zero_()
        elif key == "R":
            self.reset_requested = True
        elif key == "ESCAPE":
            self.quit_requested = True
        self.command[0].clamp_(*self._ranges.vx)
        self.command[1].clamp_(*self._ranges.yaw_rate)
        self.command[2].clamp_(*self._ranges.base_height)
        return True


def _latest_checkpoint() -> Path:
    latest_run_path = PROJECT_ROOT / "artifacts" / "phase1" / "latest_run.txt"
    if not latest_run_path.is_file():
        raise FileNotFoundError("No latest Phase 1 run is recorded; pass --checkpoint explicitly")
    run_dir = Path(latest_run_path.read_text(encoding="utf-8").strip())
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda item: int(item.stem.split("_")[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return checkpoints[-1]


def _fixed_command_tensor(direct_env: WheelLegFlatEnv) -> torch.Tensor | None:
    if args_cli.fixed_command is None:
        return None

    values = tuple(float(value) for value in args_cli.fixed_command)
    names_and_ranges = (
        ("vx", direct_env.cfg.commands.vx),
        ("yaw_rate", direct_env.cfg.commands.yaw_rate),
        ("base_height", direct_env.cfg.commands.base_height),
    )
    for value, (name, bounds) in zip(values, names_and_ranges, strict=True):
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"Fixed {name}={value} is outside the training range {bounds}")

    return torch.tensor(values, dtype=direct_env._commands.dtype, device=direct_env.device).unsqueeze(0)


def _apply_command(direct_env: WheelLegFlatEnv, command: torch.Tensor | None) -> None:
    if command is not None:
        direct_env._commands.copy_(command.reshape(1, 3).expand_as(direct_env._commands))


def main() -> None:
    if args_cli.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args_cli.steps < 0 or (args_cli.steps == 0 and not args_cli.keyboard):
        raise ValueError("--steps must be positive, or zero in keyboard mode")
    if args_cli.video_output is not None:
        if args_cli.num_envs != 1:
            raise ValueError("Video recording requires --num-envs 1")
        if args_cli.video_output.suffix.lower() != ".mp4":
            raise ValueError("--video-output must use the .mp4 suffix")

    checkpoint = args_cli.checkpoint.resolve() if args_cli.checkpoint else _latest_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    env_cfg = WheelLegFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    agent_cfg = WheelLegFlatPPORunnerCfg()
    agent_cfg.device = args_cli.device
    contract = build_phase1_contract_from_configs(
        asset_bundle_hash=asset_report.bundle_hash,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
    )
    checkpoint_metadata = validate_checkpoint_metadata(
        checkpoint,
        checkpoint.parent / "run_manifest.json",
        contract,
    )

    video_output = args_cli.video_output.resolve() if args_cli.video_output is not None else None
    if video_output is not None:
        env_cfg.viewer.eye = (0.90, -0.45, 0.55)
        env_cfg.viewer.lookat = (0.04, -1.27, 0.13)
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.resolution = (960, 540)
        env_cfg.ground.spawn.visual_material.diffuse_color = (0.04, 0.05, 0.06)

    direct_env = WheelLegFlatEnv(env_cfg, render_mode="rgb_array" if video_output is not None else None)
    evaluation_env = direct_env
    generated_video = None
    if video_output is not None:
        from pxr import UsdLux

        dome_light = UsdLux.DomeLight(direct_env.sim.stage.GetPrimAtPath("/World/Light"))
        dome_light.GetIntensityAttr().Set(500.0)
        for _ in range(5):
            direct_env.render(recompute=True)

        video_output.parent.mkdir(parents=True, exist_ok=True)
        evaluation_env = gym.wrappers.RecordVideo(
            direct_env,
            video_folder=str(video_output.parent),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.steps,
            name_prefix=video_output.stem,
            fps=round(1.0 / direct_env.step_dt),
            disable_logger=True,
        )
        generated_video = video_output.parent / f"{video_output.stem}-step-0.mp4"

    env = RslRlVecEnvWrapper(evaluation_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.device)
    fixed_command = _fixed_command_tensor(direct_env)
    keyboard = KeyboardCommand(direct_env) if args_cli.keyboard else None
    command = keyboard.command if keyboard is not None else fixed_command
    _apply_command(direct_env, command)
    observations = env.get_observations()

    if fixed_command is not None:
        vx, yaw_rate, base_height = fixed_command[0].tolist()
        print(
            f"[INFO] Fixed command: vx={vx:.3f} m/s, yaw_rate={yaw_rate:.3f} rad/s, "
            f"base_height={base_height:.3f} m",
            flush=True,
        )

    reward_sum = torch.zeros(args_cli.num_envs, device=env.device)
    reset_count = 0
    metric_keys = (
        "Tracking/vx_abs_error",
        "Tracking/yaw_rate_abs_error",
        "Tracking/base_height_abs_error",
        "Action/saturation_fraction",
        "Actuator/effort_saturation_fraction",
        "Joint/leg_soft_limit_fraction",
    )
    metric_sums = {key: 0.0 for key in metric_keys}
    termination_counts = {
        key: 0
        for key in (
            "invalid",
            "height_terminated",
            "tilt_terminated",
            "root_linear_terminated",
            "root_angular_terminated",
            "joint_velocity_terminated",
            "timeout",
        )
    }
    completed_steps = 0
    try:
        while simulation_app.is_running() and not (keyboard is not None and keyboard.quit_requested):
            if keyboard is not None and keyboard.reset_requested:
                env.reset()
                _apply_command(direct_env, command)
                observations = env.get_observations()
                keyboard.reset_requested = False

            step_start = time.perf_counter()
            with torch.inference_mode():
                actions = policy(observations)
                observations, rewards, dones, extras = env.step(actions)
                _apply_command(direct_env, command)
                if command is not None:
                    observations = env.get_observations()
            if not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
                raise RuntimeError("Evaluation produced NaN or Inf")
            completed_steps += 1
            reward_sum += rewards
            reset_count += int(dones.sum().item())
            log = extras.get("log", {})
            for key in metric_keys:
                metric_sums[key] += float(log[key].item())
            diagnostics = extras.get("termination_diagnostics", {})
            for key in termination_counts:
                values = extras["time_outs"] if key == "timeout" else diagnostics[key]
                termination_counts[key] += int(values.sum().item())

            if args_cli.steps and completed_steps >= args_cli.steps:
                break
            if args_cli.real_time:
                sleep_time = direct_env.step_dt - (time.perf_counter() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        try:
            if keyboard is not None:
                keyboard.close()
        finally:
            env.close()

    if completed_steps == 0:
        raise RuntimeError("Evaluation stopped before completing a simulation step")

    if video_output is not None:
        if generated_video is None or not generated_video.is_file():
            raise RuntimeError(f"Video recorder did not create the expected file: {generated_video}")
        generated_video.replace(video_output)

    report = {
        "checkpoint": str(checkpoint),
        "num_envs": args_cli.num_envs,
        "requested_steps": args_cli.steps,
        "completed_steps": completed_steps,
        "fixed_command": None if fixed_command is None else fixed_command[0].tolist(),
        "video_output": None if video_output is None else str(video_output),
        "mean_reward_per_step": float((reward_sum / completed_steps).mean().item()),
        "reset_count": reset_count,
        "finite": True,
        "contract_hash": checkpoint_metadata["contract_hash"],
        "mean_metrics": {
            key: value / completed_steps
            for key, value in metric_sums.items()
        },
        "termination_counts": termination_counts,
    }
    if args_cli.output is not None:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if exit_code == 0:
            simulation_app.close(skip_cleanup=True)
        else:
            os._exit(exit_code)
    raise SystemExit(exit_code)
