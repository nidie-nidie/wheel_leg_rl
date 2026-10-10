from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preflight PhysX projection for randomized closed-chain reset states.")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb

from isaaclab.sim import SimulationContext

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2
from wheelleg_dreamwaq.assets.wheelleg import PASSIVE_JOINT_NAMES
from wheelleg_dreamwaq.kinematics.virtual_leg import offset_leg_fk, virtual_leg_from_body_vector, wrap_to_pi
from wheelleg_dreamwaq.schemas.action import controlled_joint_feedback_usd_to_control
from wheelleg_dreamwaq.schemas.frames import quat_rotate_inverse_wxyz
from wheelleg_dreamwaq.schemas.physics import PHYSICS_SCHEMA_VERSION
from wheelleg_dreamwaq.schemas.randomization import (
    FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
    profile_contract_hash,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg


PHYSICAL_PASSIVE_BRANCH_JOINTS = ("jJM", "jMK", "jKN", "jOP", "jBE", "jEC", "jCF", "jGH")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _gravity_tuple(value) -> list[float]:
    return [float(value[index]) for index in range(3)]


def _metrics(
    env: WheelLegFlatEnv,
    *,
    passive_ids: list[int],
    branch_ids: list[int],
) -> dict:
    env.robot.data._physics_sim_view.update_articulations_kinematic()
    env.robot.update(env.physics_dt)
    root_quaternion = env.robot.data.root_link_quat_w
    hip_world = env._world_anchors(env._virtual_leg_hip_body_ids, env._virtual_leg_hip_local_anchors)
    wheel_world = env._world_anchors(env._virtual_leg_wheel_body_ids, env._virtual_leg_wheel_local_anchors)
    vector_world = wheel_world - hip_world
    vector_body = quat_rotate_inverse_wxyz(root_quaternion.unsqueeze(1).expand(-1, 2, -1), vector_world)
    true_length, true_phi0 = virtual_leg_from_body_vector(vector_body)

    controlled = controlled_joint_feedback_usd_to_control(
        env.robot.data.joint_pos[:, env._controlled_joint_ids]
    )
    left = offset_leg_fk(controlled[:, 0:2], env._virtual_leg_fk_geometries[0])
    right = offset_leg_fk(controlled[:, 2:4], env._virtual_leg_fk_geometries[1])
    fk_vector = torch.stack((left.wheel_vector_body, right.wheel_vector_body), dim=1)
    fk_length = torch.stack((left.length, right.length), dim=1)
    fk_phi0 = torch.stack((left.phi0, right.phi0), dim=1)
    planar = vector_body.clone()
    planar[..., 0] = 0.0

    loop_world = env._world_anchors(env._loop_body_ids, env._loop_local_anchors)
    loop_world = loop_world.reshape(env.num_envs, -1, 2, 3)
    loop_error = torch.linalg.vector_norm(loop_world[:, :, 0] - loop_world[:, :, 1], dim=-1)
    position = env.robot.data.joint_pos
    velocity = env.robot.data.joint_vel
    nominal = env.robot.data.default_joint_pos
    limits = env.robot.data.joint_pos_limits
    passive_raw_delta = position[:, passive_ids] - nominal[:, passive_ids]
    passive_wrapped_delta = torch.atan2(torch.sin(passive_raw_delta), torch.cos(passive_raw_delta))
    finite_limit = torch.isfinite(limits[..., 0]) & torch.isfinite(limits[..., 1])
    lower_margin = position - limits[..., 0]
    upper_margin = limits[..., 1] - position
    finite_margin = torch.minimum(lower_margin, upper_margin)[finite_limit]
    within_limits = (~finite_limit) | (
        (position >= limits[..., 0] - 1.0e-6) & (position <= limits[..., 1] + 1.0e-6)
    )
    branch_values = position[:, branch_ids]
    branch_signature = torch.sign(branch_values).to(dtype=torch.int8)
    nominal_branch_signature = torch.sign(nominal[:, branch_ids]).to(dtype=torch.int8)
    return {
        "loop_error_max_m": float(loop_error.max().item()),
        "wheel_error_max_m": float(torch.linalg.vector_norm(fk_vector - planar, dim=-1).max().item()),
        "length_error_max_m": float(torch.abs(fk_length - true_length).max().item()),
        "phi0_error_max_rad": float(torch.abs(wrap_to_pi(fk_phi0 - true_phi0)).max().item()),
        "active_reference_error_max_rad": float(
            torch.abs(controlled[:, :4] - env._q_reference).max().item()
        ),
        "joint_speed_max_rad_s": float(torch.abs(env.robot.data.joint_vel).max().item()),
        "all_joint_hard_limit_violation_count": int((~within_limits).sum().item()),
        "finite_joint_limit_margin_min_rad": (
            float(finite_margin.min().item()) if finite_margin.numel() else None
        ),
        "passive_raw_delta_max_abs_rad": float(torch.abs(passive_raw_delta).max().item()),
        "passive_wrapped_delta_max_abs_rad": float(torch.abs(passive_wrapped_delta).max().item()),
        "branch_signature_mismatch_count": int((branch_signature != nominal_branch_signature).sum().item()),
        "physical_passive_branch_signature": branch_signature.detach().cpu().tolist(),
        "joint_position_rad": position.detach().cpu().tolist(),
        "joint_velocity_rad_s": velocity.detach().cpu().tolist(),
        "passive_raw_delta_rad": passive_raw_delta.detach().cpu().tolist(),
        "passive_wrapped_delta_rad": passive_wrapped_delta.detach().cpu().tolist(),
    }


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("--num-envs and --steps must be positive")
    cfg = WheelLegFlatEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.randomization = FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    env = WheelLegFlatEnv(cfg)

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    passive_ids, passive_names = env.robot.find_joints(PASSIVE_JOINT_NAMES, preserve_order=True)
    branch_ids, branch_names = env.robot.find_joints(list(PHYSICAL_PASSIVE_BRANCH_JOINTS), preserve_order=True)
    if passive_names != PASSIVE_JOINT_NAMES or tuple(branch_names) != PHYSICAL_PASSIVE_BRANCH_JOINTS:
        raise RuntimeError("Passive joint order did not resolve exactly")
    joint_position = env.robot.data.default_joint_pos.clone()
    joint_position[:, env._leg_joint_ids] = env._q_reference
    wheel_nominal = joint_position[:, env._wheel_joint_ids].clone()
    joint_velocity = torch.zeros_like(joint_position)
    root_state = env.robot.data.default_root_state.clone()
    root_state[:, :3] += env.scene.env_origins
    root_state[:, 2] += 0.75
    root_state[:, 7:] = 0.0

    physics_view = SimulationContext.instance().physics_sim_view
    original_gravity = _gravity_tuple(physics_view.get_gravity())
    checkpoints = {0, 1, 5, 10, 25, 50, 100, args_cli.steps}
    rows = []
    failure = None
    counter_before = {
        "sim_step_counter": int(env._sim_step_counter),
        "common_step_counter": int(env.common_step_counter),
        "episode_length_max": int(env.episode_length_buf.max().item()),
    }
    gravity_zero_readback = None
    gravity_restored_readback = None
    try:
        physics_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))
        gravity_zero_readback = _gravity_tuple(physics_view.get_gravity())
        for step in range(args_cli.steps + 1):
            if step > 0:
                joint_position = env.robot.data.joint_pos.clone()
                joint_position[:, env._leg_joint_ids] = env._q_reference
                joint_position[:, env._wheel_joint_ids] = wheel_nominal
                joint_velocity.zero_()
            env.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
            env.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
            env.robot.write_joint_state_to_sim(joint_position, joint_velocity, None, env_ids)
            env.robot.set_joint_position_target(env._q_reference, joint_ids=env._leg_joint_ids)
            env.robot.set_joint_velocity_target(
                torch.zeros((env.num_envs, 2), device=env.device), joint_ids=env._wheel_joint_ids
            )
            env.scene.write_data_to_sim()
            if step > 0:
                env.sim.step(render=False)
            else:
                env.sim.forward()
            env.scene.update(env.physics_dt)
            if step in checkpoints:
                row = {
                    "step": step,
                    **_metrics(env, passive_ids=passive_ids, branch_ids=branch_ids),
                }
                rows.append(row)
                summary = {key: value for key, value in row.items() if not isinstance(value, list)}
                print(json.dumps(summary, sort_keys=True), flush=True)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        physics_view.set_gravity(carb.Float3(*cfg.sim.gravity))
        gravity_restored_readback = _gravity_tuple(physics_view.get_gravity())
        audit = env.randomization_audit
        report = {
            "schema_version": "BoundaryClampedPhysXRelaxationPreflightV2",
            "algorithm_version": "BoundaryClampedPhysXRelaxationV1",
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "failure": failure,
            "identity": {
                "script_sha256": _sha256(Path(__file__)),
                "asset_bundle_version": ASSET_BUNDLE_V2.version,
                "asset_bundle_hash": ASSET_BUNDLE_V2.bundle_hash,
                "randomization_profile_hash": profile_contract_hash(FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1),
                "realized_plan_hash": audit["realized_plan_hash"],
                "physics_schema_version": PHYSICS_SCHEMA_VERSION,
                "isaac_sim_version": importlib.metadata.version("isaacsim"),
                "isaac_lab_tag": "v2.3.2",
                "isaac_lab_commit": "37ddf626871758333d6ed89cf64ad702aef127d0",
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "projection_contract": {
                "sim_dt_s": float(env.physics_dt),
                "root_lift_world_z_m": 0.75,
                "gravity_during_projection": [0.0, 0.0, 0.0],
                "write_order": [
                    "read_previous_joint_position",
                    "overwrite_active_leg_reference",
                    "overwrite_wheel_nominal_position",
                    "zero_all_joint_velocity",
                    "write_root_pose_velocity",
                    "write_full_joint_state",
                    "set_active_leg_position_target",
                    "set_wheel_zero_velocity_target",
                    "scene_write_data_to_sim",
                    "sim_step",
                    "scene_update",
                ],
                "active_leg_realized_stiffness": env._randomization.realized_stiffness[:, :4].detach().cpu().tolist(),
                "active_leg_realized_damping": env._randomization.realized_damping[:, :4].detach().cpu().tolist(),
                "active_leg_realized_effort_limit": env._randomization.realized_effort_limit[:, :4].detach().cpu().tolist(),
                "passive_stiffness": 0.0,
                "passive_damping": 0.05,
                "passive_velocity_target": 0.0,
            },
            "gravity": {
                "before": original_gravity,
                "zero_readback": gravity_zero_readback,
                "restored_readback": gravity_restored_readback,
            },
            "environment_counters": {
                "before": counter_before,
                "after": {
                    "sim_step_counter": int(env._sim_step_counter),
                    "common_step_counter": int(env.common_step_counter),
                    "episode_length_max": int(env.episode_length_buf.max().item()),
                },
                "underlying_physics_time_advanced_s": args_cli.steps * float(env.physics_dt),
            },
            "joint_contract": {
                "joint_names": list(env.robot.joint_names),
                "passive_joint_names": passive_names,
                "physical_passive_branch_joint_names": branch_names,
                "default_joint_position_rad": env.robot.data.default_joint_pos.detach().cpu().tolist(),
                "joint_position_limits_rad": env.robot.data.joint_pos_limits.detach().cpu().tolist(),
                "nominal_physical_branch_signature": torch.sign(
                    env.robot.data.default_joint_pos[:, branch_ids]
                ).to(dtype=torch.int8).detach().cpu().tolist(),
            },
            "rows": rows,
        }
        if args_cli.output is not None:
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
