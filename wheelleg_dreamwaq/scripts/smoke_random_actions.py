from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run the WheelLeg Phase 0 random-action dynamics smoke test.")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "artifacts" / "phase0" / "random-action-smoke.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.kinematics.virtual_leg import wrap_to_pi
from wheelleg_dreamwaq.schemas.action import (
    ACTION_DIM,
    CANONICAL_JOINT_ORDER,
    controlled_joint_feedback_usd_to_control,
)
from wheelleg_dreamwaq.schemas.frames import quat_rotate_inverse_wxyz, transform_usd_vector_to_control
from wheelleg_dreamwaq.schemas.observation import CriticObsSlices
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg

def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("--num-envs and --steps must be positive")
    torch.manual_seed(args_cli.seed)
    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    cfg = WheelLegFlatEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = WheelLegFlatEnv(cfg)
    observations, _ = env.reset()
    controlled_names = tuple(env.robot.joint_names[index] for index in env._controlled_joint_ids)
    if controlled_names != CANONICAL_JOINT_ORDER:
        raise RuntimeError(f"Runtime controlled-joint order does not match ActionV1: {controlled_names}")
    if observations["policy"].shape != (args_cli.num_envs, 25):
        raise RuntimeError(f"Unexpected policy observation shape: {tuple(observations['policy'].shape)}")
    if observations["critic"].shape != (args_cli.num_envs, 41):
        raise RuntimeError(f"Unexpected critic observation shape: {tuple(observations['critic'].shape)}")

    reset_count = 0
    reset_previous_action_leak_count = 0
    reward_min = float("inf")
    reward_max = float("-inf")
    max_abs_leg_position = 0.0
    max_abs_controlled_velocity = 0.0
    max_abs_all_joint_velocity = 0.0
    max_abs_applied_torque = 0.0
    max_abs_root_linear_velocity = 0.0
    max_abs_root_angular_velocity = 0.0
    max_critic_joint_acceleration_error = 0.0
    max_actor_joint_velocity_error = 0.0
    max_com_velocity_frame_error = 0.0
    max_loop_closure_position_error = 0.0
    max_virtual_leg_wheel_error = 0.0
    max_virtual_leg_length_error = 0.0
    max_virtual_leg_phi0_error_rad = 0.0
    max_phi0_delta_abs_rad = 0.0
    min_virtual_leg_length = float("inf")
    max_virtual_leg_length = float("-inf")
    action_saturation_count = 0
    sample_count = args_cli.num_envs * args_cli.steps * ACTION_DIM

    for _ in range(args_cli.steps):
        joint_velocity_before = controlled_joint_feedback_usd_to_control(
            env.robot.data.joint_vel[:, env._controlled_joint_ids]
        ).clone()
        actions = 2.0 * torch.rand((args_cli.num_envs, ACTION_DIM), device=env.device) - 1.0
        observations, rewards, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated
        joint_velocity_after = controlled_joint_feedback_usd_to_control(
            env.robot.data.joint_vel[:, env._controlled_joint_ids]
        )

        finite_tensors = (
            observations["policy"],
            observations["critic"],
            rewards,
            env.robot.data.joint_pos,
            env.robot.data.joint_vel,
            env.robot.data.root_com_pos_w,
            env.robot.data.root_com_lin_vel_w,
            env.robot.data.root_link_ang_vel_w,
            env._state.virtual_leg_length_true,
            env._state.virtual_leg_phi0_true,
            env._state.virtual_leg_length_fk,
            env._state.virtual_leg_phi0_fk,
            env._state.virtual_leg_wheel_error,
            env._state.loop_closure_position_error,
        )
        if not all(torch.isfinite(value).all() for value in finite_tensors):
            raise RuntimeError("Random-action smoke test produced NaN or Inf")

        reconstructed_com_velocity_b = quat_rotate_inverse_wxyz(
            env.robot.data.root_link_quat_w,
            env.robot.data.root_com_lin_vel_w,
        )
        body_velocity_error = torch.max(
            torch.abs(reconstructed_com_velocity_b - env.robot.data.root_com_lin_vel_b)
        )
        control_velocity_error = torch.max(
            torch.abs(
                transform_usd_vector_to_control(reconstructed_com_velocity_b)
                - env._state.root_com_linear_velocity
            )
        )
        frame_error = max(float(body_velocity_error.item()), float(control_velocity_error.item()))
        max_com_velocity_frame_error = max(max_com_velocity_frame_error, frame_error)
        if frame_error > 1.0e-5:
            raise RuntimeError(f"Root COM velocity frame conversion mismatch: {frame_error:.6g}")

        loop_closure_error = float(env._state.loop_closure_position_error.max().item())
        max_loop_closure_position_error = max(max_loop_closure_position_error, loop_closure_error)
        if loop_closure_error > 5.0e-3:
            raise RuntimeError(f"Loop-closure anchor error exceeded 5 mm: {loop_closure_error:.6g} m")

        wheel_error = float(env._state.virtual_leg_wheel_error.max().item())
        length_error = float(
            torch.abs(env._state.virtual_leg_length_fk - env._state.virtual_leg_length_true).max().item()
        )
        phi0_error = float(
            torch.abs(wrap_to_pi(env._state.virtual_leg_phi0_fk - env._state.virtual_leg_phi0_true)).max().item()
        )
        phi0_delta = float(
            torch.abs(
                wrap_to_pi(env._state.virtual_leg_phi0_true[:, 0] - env._state.virtual_leg_phi0_true[:, 1])
            ).max().item()
        )
        max_virtual_leg_wheel_error = max(max_virtual_leg_wheel_error, wheel_error)
        max_virtual_leg_length_error = max(max_virtual_leg_length_error, length_error)
        max_virtual_leg_phi0_error_rad = max(max_virtual_leg_phi0_error_rad, phi0_error)
        max_phi0_delta_abs_rad = max(max_phi0_delta_abs_rad, phi0_delta)
        min_virtual_leg_length = min(min_virtual_leg_length, float(env._state.virtual_leg_length_true.min().item()))
        max_virtual_leg_length = max(max_virtual_leg_length, float(env._state.virtual_leg_length_true.max().item()))

        if torch.any(env._leg_position_targets < actions.new_tensor(cfg.control.leg_target_lower) - 1.0e-6):
            raise RuntimeError("Leg position target crossed its lower software limit")
        if torch.any(env._leg_position_targets > actions.new_tensor(cfg.control.leg_target_upper) + 1.0e-6):
            raise RuntimeError("Leg position target crossed its upper software limit")
        if torch.any(torch.abs(env._wheel_velocity_targets) > cfg.control.wheel_action_scale + 1.0e-6):
            raise RuntimeError("Wheel velocity target crossed its software limit")

        active = ~done
        if active.any():
            expected_velocity = cfg.normalization.normalize_joint_velocity(joint_velocity_after[active])
            returned_velocity = observations["policy"][active, 13:19]
            velocity_error = torch.max(torch.abs(returned_velocity - expected_velocity))
            max_actor_joint_velocity_error = max(
                max_actor_joint_velocity_error,
                float(velocity_error.item()),
            )
            if velocity_error > 1.0e-5:
                raise RuntimeError(
                    "Actor joint velocity does not use canonical wheel feedback signs: "
                    f"max normalized error={float(velocity_error.item()):.6g}"
                )
            expected_acceleration = (joint_velocity_after[active] - joint_velocity_before[active]) / env.step_dt
            expected_normalized = cfg.normalization.normalize_joint_acceleration(expected_acceleration)
            returned_normalized = observations["critic"][active, CriticObsSlices.JOINT_ACCELERATION]
            acceleration_error = torch.max(torch.abs(returned_normalized - expected_normalized))
            max_critic_joint_acceleration_error = max(
                max_critic_joint_acceleration_error,
                float(acceleration_error.item()),
            )
            if acceleration_error > 1.0e-5:
                raise RuntimeError(
                    "Critic joint acceleration does not describe the returned non-reset transition: "
                    f"max normalized error={float(acceleration_error.item()):.6g}"
                )

        if done.any():
            returned_previous_action = observations["policy"][done, 19:25]
            leaks = torch.any(torch.abs(returned_previous_action) > 1.0e-6, dim=-1)
            reset_previous_action_leak_count += int(leaks.sum().item())
            if torch.any(torch.abs(observations["critic"][done, CriticObsSlices.JOINT_ACCELERATION]) > 1.0e-6):
                raise RuntimeError("Reset observations contain stale joint acceleration")
            if torch.any(torch.abs(observations["critic"][done, CriticObsSlices.APPLIED_TORQUE]) > 1.0e-6):
                raise RuntimeError("Reset observations contain stale applied torque")

        reset_count += int(done.sum().item())
        action_saturation_count += int((torch.abs(actions) > 0.999).sum().item())
        reward_min = min(reward_min, float(rewards.min().item()))
        reward_max = max(reward_max, float(rewards.max().item()))
        max_abs_leg_position = max(
            max_abs_leg_position,
            float(torch.abs(env.robot.data.joint_pos[:, env._leg_joint_ids]).max().item()),
        )
        max_abs_controlled_velocity = max(
            max_abs_controlled_velocity,
            float(torch.abs(env.robot.data.joint_vel[:, env._controlled_joint_ids]).max().item()),
        )
        max_abs_all_joint_velocity = max(
            max_abs_all_joint_velocity,
            float(torch.abs(env.robot.data.joint_vel).max().item()),
        )
        max_abs_applied_torque = max(
            max_abs_applied_torque,
            float(torch.abs(env.robot.data.applied_torque[:, env._controlled_joint_ids]).max().item()),
        )
        max_abs_root_linear_velocity = max(
            max_abs_root_linear_velocity,
            float(torch.linalg.vector_norm(env.robot.data.root_com_lin_vel_w, dim=-1).max().item()),
        )
        max_abs_root_angular_velocity = max(
            max_abs_root_angular_velocity,
            float(torch.linalg.vector_norm(env.robot.data.root_link_ang_vel_w, dim=-1).max().item()),
        )

    if reset_previous_action_leak_count:
        raise RuntimeError(
            f"Detected {reset_previous_action_leak_count} reset observations with leaked previous actions"
        )
    if max_abs_leg_position > 1.10:
        raise RuntimeError(f"Actual leg position crossed the Phase 0 guard band: {max_abs_leg_position:.6f} rad")

    report = {
        "asset_bundle_hash": asset_report.bundle_hash,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "control_dt": env.step_dt,
        "policy_observation_shape": list(observations["policy"].shape),
        "critic_observation_shape": list(observations["critic"].shape),
        "controlled_joint_names": list(controlled_names),
        "reset_count": reset_count,
        "reset_previous_action_leak_count": reset_previous_action_leak_count,
        "action_saturation_fraction": action_saturation_count / sample_count,
        "reward": {"min": reward_min, "max": reward_max},
        "max_abs_leg_position": max_abs_leg_position,
        "max_abs_controlled_velocity": max_abs_controlled_velocity,
        "max_abs_all_joint_velocity": max_abs_all_joint_velocity,
        "max_abs_applied_torque": max_abs_applied_torque,
        "max_abs_root_linear_velocity": max_abs_root_linear_velocity,
        "max_abs_root_angular_velocity": max_abs_root_angular_velocity,
        "max_critic_joint_acceleration_error": max_critic_joint_acceleration_error,
        "max_actor_joint_velocity_error": max_actor_joint_velocity_error,
        "max_com_velocity_frame_error": max_com_velocity_frame_error,
        "max_loop_closure_position_error": max_loop_closure_position_error,
        "max_virtual_leg_wheel_error": max_virtual_leg_wheel_error,
        "max_virtual_leg_length_error": max_virtual_leg_length_error,
        "max_virtual_leg_phi0_error_rad": max_virtual_leg_phi0_error_rad,
        "max_phi0_delta_abs_rad": max_phi0_delta_abs_rad,
        "virtual_leg_length_range": [min_virtual_leg_length, max_virtual_leg_length],
        "reward_terms": sorted(env._reward_sums),
        "leg_target_lower": list(cfg.control.leg_target_lower),
        "leg_target_upper": list(cfg.control.leg_target_upper),
        "wheel_velocity_target_limit": cfg.control.wheel_action_scale,
        "finite": True,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"[INFO] Random-action smoke report written to: {args_cli.output.resolve()}", flush=True)


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
