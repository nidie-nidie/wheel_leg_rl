from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run the WheelLeg Phase 0 zero-action dynamics smoke test.")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--root-height", type=float, default=None)
parser.add_argument("--q-nominal", type=float, nargs=4, default=None, metavar=("JIJ", "JIO", "JAB", "JAG"))
parser.add_argument("--fix-root", action="store_true", help="Fix the root only for Phase 0 pose calibration.")
parser.add_argument("--zero-gravity", action="store_true", help="Disable gravity only for Phase 0 pose calibration.")
parser.add_argument(
    "--disable-early-termination",
    action="store_true",
    help="Keep a failing free-root trajectory alive long enough to capture Phase 0 diagnostics.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "artifacts" / "phase0" / "zero-action-smoke.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.schemas.action import CANONICAL_JOINT_ORDER
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.terminations import TerminationLimits


def _tensor_range(values: torch.Tensor) -> dict[str, float]:
    return {"min": float(values.min().item()), "max": float(values.max().item())}


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("--num-envs and --steps must be positive")
    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    cfg = WheelLegFlatEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    if args_cli.root_height is not None:
        cfg.robot_cfg.init_state.pos = (0.0, 0.0, args_cli.root_height)
    if args_cli.fix_root:
        cfg.robot_cfg.spawn.articulation_props.fix_root_link = True
    if args_cli.fix_root or args_cli.disable_early_termination:
        cfg.termination = TerminationLimits(
            min_base_height=-10.0,
            max_base_height=10.0,
            max_tilt_rad=3.0,
            max_root_linear_velocity=1000.0,
            max_root_angular_velocity=1000.0,
            max_joint_velocity=1000.0,
            grace_steps=1_000_000,
        )
    if args_cli.zero_gravity:
        cfg.sim.gravity = (0.0, 0.0, 0.0)
    if args_cli.q_nominal is not None:
        cfg.q_nominal = tuple(args_cli.q_nominal)
        cfg.robot_cfg.init_state.joint_pos = dict(zip(CANONICAL_JOINT_ORDER[:4], args_cli.q_nominal, strict=True))
    env = WheelLegFlatEnv(cfg)
    observations, _ = env.reset()

    if tuple(env.robot.joint_names[index] for index in env._controlled_joint_ids) != CANONICAL_JOINT_ORDER:
        raise RuntimeError("Runtime controlled-joint order does not match ActionV1")
    if observations["policy"].shape != (args_cli.num_envs, 25):
        raise RuntimeError(f"Unexpected policy observation shape: {tuple(observations['policy'].shape)}")
    if observations["critic"].shape != (args_cli.num_envs, 41):
        raise RuntimeError(f"Unexpected critic observation shape: {tuple(observations['critic'].shape)}")

    actions = torch.zeros((args_cli.num_envs, 6), device=env.device)
    wheel_body_ids, wheel_body_names = env.robot.find_bodies(["jwheel_left", "jwheel_right"], preserve_order=True)
    diagnostic_joint_names = [
        "jIJ",
        "jIO",
        "jAB",
        "jAG",
        "jOP",
        "jGH",
        "jJM",
        "jBE",
        "jMK",
        "jEC",
        "jKN",
        "jCF",
    ]
    diagnostic_joint_ids, resolved_diagnostic_joint_names = env.robot.find_joints(
        diagnostic_joint_names, preserve_order=True
    )
    if resolved_diagnostic_joint_names != diagnostic_joint_names:
        raise RuntimeError(
            "Diagnostic joint order mismatch: "
            f"{resolved_diagnostic_joint_names} != {diagnostic_joint_names}"
        )
    initial_base_height = float(env._state.base_height[0, 0].item())
    initial_root_link_position = (
        env.robot.data.root_link_pos_w[0] - env.scene.env_origins[0]
    ).detach().cpu().tolist()
    initial_root_com_position = (
        env.robot.data.root_com_pos_w[0] - env.scene.env_origins[0]
    ).detach().cpu().tolist()
    initial_wheel_center_position = (
        env.robot.data.body_pos_w[0, wheel_body_ids] - env.scene.env_origins[0]
    ).detach().cpu().tolist()
    initial_wheel_center_height = (
        env.robot.data.body_pos_w[0, wheel_body_ids, 2] - env.scene.env_origins[0, 2]
    ).detach().cpu().tolist()
    startup_trace: list[dict[str, float]] = []
    reset_count = 0
    reward_min = float("inf")
    reward_max = float("-inf")
    height_min = float("inf")
    height_max = float("-inf")
    tilt_max = 0.0
    for step in range(args_cli.steps):
        observations, rewards, terminated, truncated, _ = env.step(actions)
        tensors = (observations["policy"], observations["critic"], rewards)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise RuntimeError("Zero-action smoke test produced NaN or Inf")
        reset_count += int((terminated | truncated).sum().item())
        reward_min = min(reward_min, float(rewards.min().item()))
        reward_max = max(reward_max, float(rewards.max().item()))
        height_min = min(height_min, float(env._state.base_height.min().item()))
        height_max = max(height_max, float(env._state.base_height.max().item()))
        tilt = torch.linalg.vector_norm(env._state.projected_gravity[:, :2], dim=-1)
        tilt_max = max(tilt_max, float(tilt.max().item()))
        if step < 25:
            wheel_center_height = (
                env.robot.data.body_pos_w[0, wheel_body_ids, 2] - env.scene.env_origins[0, 2]
            )
            diagnostics = env.extras["termination_diagnostics"]
            joint_positions = env.robot.data.joint_pos[0, diagnostic_joint_ids]
            joint_velocities = env.robot.data.joint_vel[0, diagnostic_joint_ids]
            startup_trace.append(
                {
                    "step": step + 1,
                    "base_height": float(env._state.base_height[0, 0].item()),
                    "root_link_position": (
                        env.robot.data.root_link_pos_w[0] - env.scene.env_origins[0]
                    ).detach().cpu().tolist(),
                    "root_com_position": (
                        env.robot.data.root_com_pos_w[0] - env.scene.env_origins[0]
                    ).detach().cpu().tolist(),
                    "root_link_quaternion_wxyz": env.robot.data.root_link_quat_w[0].detach().cpu().tolist(),
                    "root_link_linear_velocity_world": (
                        env.robot.data.root_link_lin_vel_w[0].detach().cpu().tolist()
                    ),
                    "root_com_linear_velocity_world": (
                        env.robot.data.root_com_lin_vel_w[0].detach().cpu().tolist()
                    ),
                    "root_link_angular_velocity_world": (
                        env.robot.data.root_link_ang_vel_w[0].detach().cpu().tolist()
                    ),
                    "wheel_center_position": (
                        env.robot.data.body_pos_w[0, wheel_body_ids] - env.scene.env_origins[0]
                    ).detach().cpu().tolist(),
                    "projected_gravity": env._state.projected_gravity[0].detach().cpu().tolist(),
                    "left_wheel_center_height": float(wheel_center_height[0].item()),
                    "right_wheel_center_height": float(wheel_center_height[1].item()),
                    "terminated": bool(terminated[0].item()),
                    "truncated": bool(truncated[0].item()),
                    "height_bad": bool(diagnostics["height_bad"][0].item()),
                    "tilt_bad": bool(diagnostics["tilt_bad"][0].item()),
                    "root_linear_bad": bool(diagnostics["root_linear_bad"][0].item()),
                    "root_angular_bad": bool(diagnostics["root_angular_bad"][0].item()),
                    "joint_velocity_bad": bool(diagnostics["joint_velocity_bad"][0].item()),
                    "joint_position": {
                        name: float(value.item())
                        for name, value in zip(diagnostic_joint_names, joint_positions, strict=True)
                    },
                    "joint_velocity": {
                        name: float(value.item())
                        for name, value in zip(diagnostic_joint_names, joint_velocities, strict=True)
                    },
                    "controlled_applied_torque": {
                        name: float(value.item())
                        for name, value in zip(
                            CANONICAL_JOINT_ORDER,
                            env.robot.data.applied_torque[0, env._controlled_joint_ids],
                            strict=True,
                        )
                    },
                }
            )

    report = {
        "asset_bundle_hash": asset_report.bundle_hash,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "control_dt": env.step_dt,
        "configured_root_height": cfg.robot_cfg.init_state.pos[2],
        "configured_q_nominal": list(cfg.q_nominal),
        "fixed_root_calibration": args_cli.fix_root,
        "zero_gravity_calibration": args_cli.zero_gravity,
        "early_termination_disabled": args_cli.disable_early_termination,
        "joint_names": env.robot.joint_names,
        "controlled_joint_ids": list(env._controlled_joint_ids),
        "controlled_joint_names": [env.robot.joint_names[index] for index in env._controlled_joint_ids],
        "body_names": env.robot.body_names,
        "default_root_state": env.robot.data.default_root_state[0].detach().cpu().tolist(),
        "default_controlled_joint_position": (
            env.robot.data.default_joint_pos[0, env._controlled_joint_ids].detach().cpu().tolist()
        ),
        "initial_base_height": initial_base_height,
        "initial_root_link_position": initial_root_link_position,
        "initial_root_com_position": initial_root_com_position,
        "wheel_body_names": wheel_body_names,
        "initial_wheel_center_position": initial_wheel_center_position,
        "initial_wheel_center_height": initial_wheel_center_height,
        "startup_trace": startup_trace,
        "policy_observation_shape": list(observations["policy"].shape),
        "critic_observation_shape": list(observations["critic"].shape),
        "reward": {"min": reward_min, "max": reward_max},
        "base_height": {"min": height_min, "max": height_max},
        "max_projected_gravity_xy_norm": tilt_max,
        "controlled_joint_position": _tensor_range(env._state.joint_position),
        "controlled_joint_velocity": _tensor_range(env._state.joint_velocity),
        "applied_torque": _tensor_range(env._state.applied_torque),
        "final_root_link_pose": torch.cat(
            (env.robot.data.root_link_pos_w[0] - env.scene.env_origins[0], env.robot.data.root_link_quat_w[0])
        ).detach().cpu().tolist(),
        "final_wheel_center_height": (
            env.robot.data.body_pos_w[0, wheel_body_ids, 2] - env.scene.env_origins[0, 2]
        ).detach().cpu().tolist(),
        "final_projected_gravity": env._state.projected_gravity[0].detach().cpu().tolist(),
        "final_joint_position_by_name": {
            name: float(env.robot.data.joint_pos[0, index].item())
            for index, name in enumerate(env.robot.joint_names)
        },
        "final_joint_velocity_by_name": {
            name: float(env.robot.data.joint_vel[0, index].item())
            for index, name in enumerate(env.robot.joint_names)
        },
        "reset_count": reset_count,
        "finite": True,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"[INFO] Zero-action smoke report written to: {args_cli.output.resolve()}", flush=True)


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
