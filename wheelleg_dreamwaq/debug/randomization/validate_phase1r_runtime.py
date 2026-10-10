from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate the Phase 1R randomization and closed-chain reset runtime.")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--randomization-profile", choices=("fudan-v1", "nominal"), default="fudan-v1")
parser.add_argument("--root-height-alignment", choices=("none", "fk"), default="none")
parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_ROOT / "artifacts" / "debug" / "randomization" / "phase1r-runtime-gates.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from wheelleg_dreamwaq.kinematics.virtual_leg import (
    MAX_FK_LENGTH_ERROR_M,
    MAX_FK_PHI0_ERROR_RAD,
    MAX_FK_WHEEL_ERROR_M,
    MAX_LOOP_CLOSURE_ERROR_M,
)
from wheelleg_dreamwaq.schemas.action import controlled_joint_feedback_usd_to_control
from wheelleg_dreamwaq.schemas.frames import quat_rotate_wxyz
from wheelleg_dreamwaq.schemas.randomization import (
    FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
    NOMINAL_TRAINING_PROFILE_V1,
    compute_closed_chain_root_height_offset,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
from pxr import Usd, UsdGeom, UsdPhysics


PHYSICS_STEPS = 40
PENETRATION_LIMIT_M = 0.002
LEG_SPEED_LIMIT_RAD_S = 5.0
WHEEL_SPEED_LIMIT_RAD_S = 10.0
SATURATION_FRACTION_LIMIT = 0.05
SATURATION_CONSECUTIVE_LIMIT = 2


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.max(torch.abs(actual - expected)).item())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _actuator_sync(env: WheelLegFlatEnv) -> dict:
    ids = env._controlled_joint_ids
    expected = env._randomization
    physx_stiffness = env.robot.root_physx_view.get_dof_stiffnesses().to(env.device)[:, ids]
    physx_damping = env.robot.root_physx_view.get_dof_dampings().to(env.device)[:, ids]
    physx_effort = env.robot.root_physx_view.get_dof_max_forces().to(env.device)[:, ids]
    errors = {
        "physx_stiffness": _max_abs_error(physx_stiffness, expected.realized_stiffness),
        "physx_damping": _max_abs_error(physx_damping, expected.realized_damping),
        "physx_effort_limit": _max_abs_error(physx_effort, expected.realized_effort_limit),
        "data_stiffness": _max_abs_error(env.robot.data.joint_stiffness[:, ids], expected.realized_stiffness),
        "data_damping": _max_abs_error(env.robot.data.joint_damping[:, ids], expected.realized_damping),
        "data_effort_limit": _max_abs_error(env.robot.data.joint_effort_limits[:, ids], expected.realized_effort_limit),
        "controlled_effort_limit": _max_abs_error(env._controlled_effort_limits, expected.realized_effort_limit),
    }
    canonical_index = {joint_id: index for index, joint_id in enumerate(ids)}
    for actuator_name in ("legs", "wheels"):
        actuator = env.robot.actuators[actuator_name]
        joint_ids = actuator.joint_indices.tolist()
        columns = [canonical_index[int(joint_id)] for joint_id in joint_ids]
        errors[f"{actuator_name}_mirror_stiffness"] = _max_abs_error(
            actuator.stiffness, expected.realized_stiffness[:, columns]
        )
        errors[f"{actuator_name}_mirror_damping"] = _max_abs_error(
            actuator.damping, expected.realized_damping[:, columns]
        )
        errors[f"{actuator_name}_mirror_effort_limit_sim"] = _max_abs_error(
            actuator.effort_limit_sim, expected.realized_effort_limit[:, columns]
        )
        errors[f"{actuator_name}_mirror_effort_limit"] = _max_abs_error(
            actuator.effort_limit, expected.realized_effort_limit[:, columns]
        )
    max_error = max(errors.values())
    return {"max_abs_error": max_error, "errors": errors, "passed": max_error <= 1.0e-6}


def _wheel_collision_geometry(env: WheelLegFlatEnv) -> tuple[list[int], list[torch.Tensor], dict]:
    stage = env.sim.stage
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        useExtentsHint=False,
    )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    body_ids = []
    points_body = []
    details = {}
    for body_name in ("jwheel_left", "jwheel_right"):
        resolved_ids, resolved_names = env.robot.find_bodies([body_name], preserve_order=True)
        if resolved_names != [body_name] or len(resolved_ids) != 1:
            raise RuntimeError(f"Unable to resolve wheel body {body_name}")
        body_ids.append(resolved_ids[0])
        body_path = f"/World/envs/env_0/Robot/{body_name}"
        body_prim = stage.GetPrimAtPath(body_path)
        collision_prims = [
            prim
            for prim in Usd.PrimRange(body_prim)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        ]
        if len(collision_prims) != 1:
            raise RuntimeError(f"Expected one enabled collision prim for {body_name}, found {collision_prims}")
        collision_prim = collision_prims[0]
        mesh = UsdGeom.Mesh(collision_prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            raise RuntimeError(f"Wheel collision prim has no mesh points: {collision_prim.GetPath()}")
        mesh_to_body, _ = xform_cache.ComputeRelativeTransform(collision_prim, body_prim)
        transformed_points = [mesh_to_body.Transform(point) for point in points]
        points_tensor = torch.tensor(
            [[float(value) for value in point] for point in transformed_points],
            dtype=torch.float32,
            device=env.device,
        )
        points_body.append(points_tensor)
        aligned_range = cache.ComputeRelativeBound(collision_prim, body_prim).ComputeAlignedRange()
        minimum = [float(value) for value in aligned_range.GetMin()]
        maximum = [float(value) for value in aligned_range.GetMax()]
        radius = max(abs(minimum[2]), abs(maximum[2]))
        if radius <= 0.0 or radius > 1.0:
            raise RuntimeError(
                f"Invalid wheel collision bound for {body_name}: type={collision_prim.GetTypeName()!r}, "
                f"instance={collision_prim.IsInstance()}, min={minimum}, max={maximum}"
            )
        details[body_name] = {
            "collision_prim": str(collision_prim.GetPath()),
            "vertex_count": int(points_tensor.shape[0]),
            "body_local_aabb_min_m": minimum,
            "body_local_aabb_max_m": maximum,
            "vertical_radius_m": radius,
        }
    return body_ids, points_body, details


def _root_height_offset_from_fk(env: WheelLegFlatEnv) -> torch.Tensor:
    return compute_closed_chain_root_height_offset(
        env._q_reference,
        q_nominal=env._q_nominal,
        geometries=env._virtual_leg_fk_geometries,
    ).to(device=env.device)


def _write_reset_state(
    env: WheelLegFlatEnv,
    root_velocity: torch.Tensor,
    *,
    align_root_height: bool,
) -> torch.Tensor:
    env_ids = env.robot._ALL_INDICES
    env.robot.reset(env_ids)
    joint_position = env._q_reset_projected_env.clone()
    joint_velocity = torch.zeros_like(joint_position)
    root_state = env.robot.data.default_root_state.clone()
    root_state[:, :3] += env.scene.env_origins
    root_height_offset = (
        _root_height_offset_from_fk(env)
        if align_root_height
        else torch.zeros(env.num_envs, device=env.device)
    )
    root_state[:, 2] += root_height_offset
    root_state[:, 7:] = root_velocity
    env.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    env.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    env.robot.write_joint_state_to_sim(joint_position, joint_velocity, None, env_ids)
    env._canonical_action.zero_()
    env._previous_action.zero_()
    env._previous_previous_action.zero_()
    env._leg_position_targets.copy_(env._q_reference)
    env._wheel_velocity_targets.zero_()
    env.robot.set_joint_position_target(env._q_reference, joint_ids=env._leg_joint_ids)
    env.robot.set_joint_velocity_target(env._wheel_velocity_targets, joint_ids=env._wheel_joint_ids)
    env.robot.set_joint_velocity_target(
        torch.zeros((env.num_envs, len(env._passive_joint_ids)), device=env.device),
        joint_ids=env._passive_joint_ids,
    )
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)
    return root_height_offset


def _wheel_clearance(
    env: WheelLegFlatEnv,
    wheel_body_ids: list[int],
    wheel_collision_points_body: list[torch.Tensor],
) -> torch.Tensor:
    wheel_bottom_z = []
    for body_id, local_points in zip(
        wheel_body_ids,
        wheel_collision_points_body,
        strict=True,
    ):
        local = local_points.unsqueeze(0).expand(env.num_envs, -1, -1)
        quaternion = env.robot.data.body_link_quat_w[:, body_id].unsqueeze(1).expand(-1, local.shape[1], -1)
        world_points = env.robot.data.body_link_pos_w[:, body_id].unsqueeze(1) + quat_rotate_wxyz(
            quaternion, local
        )
        wheel_bottom_z.append(world_points[..., 2].min(dim=1).values)
    return torch.stack(wheel_bottom_z, dim=1) - env.scene.env_origins[:, 2:3]


def _run_gate(
    env: WheelLegFlatEnv,
    *,
    name: str,
    root_velocity: torch.Tensor,
    wheel_body_ids: list[int],
    wheel_collision_points_body: list[torch.Tensor],
    align_root_height: bool,
) -> dict:
    root_height_offset = _write_reset_state(
        env,
        root_velocity,
        align_root_height=align_root_height,
    )
    initial_controlled_position = controlled_joint_feedback_usd_to_control(
        env.robot.data.joint_pos[:, env._controlled_joint_ids]
    ).clone()
    initial_virtual_leg = env._virtual_leg_state(
        env.robot.data.root_link_quat_w,
        initial_controlled_position,
        enforce_guards=False,
    )
    initial_wheel_clearance = _wheel_clearance(
        env,
        wheel_body_ids,
        wheel_collision_points_body,
    )
    initial_root_height = env.robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    initial_metrics = env._closed_chain_metrics()
    initial_controlled_velocity = controlled_joint_feedback_usd_to_control(
        env.robot.data.joint_vel[:, env._controlled_joint_ids]
    )
    initial_penetration = torch.clamp(-initial_wheel_clearance, min=0.0)
    saturation_count = torch.zeros((env.num_envs, 6), dtype=torch.int32, device=env.device)
    consecutive = torch.zeros_like(saturation_count)
    max_consecutive = torch.zeros_like(saturation_count)
    per_env_wheel_penetration = initial_penetration.clone()
    per_env_wheel_penetration_step = torch.zeros(
        (env.num_envs, 2), dtype=torch.int32, device=env.device
    )
    maxima = {
        "loop_error_m": initial_metrics["loop_error_max_m"],
        "wheel_error_m": initial_metrics["wheel_error_max_m"],
        "length_error_m": initial_metrics["length_error_max_m"],
        "phi0_error_rad": initial_metrics["phi0_error_max_rad"],
        "wheel_penetration_m": float(initial_penetration.max().item()),
        "leg_speed_rad_s": float(torch.abs(initial_controlled_velocity[:, :4]).max().item()),
        "wheel_speed_rad_s": float(torch.abs(initial_controlled_velocity[:, 4:]).max().item()),
    }
    finite = (
        initial_metrics["finite"]
        and bool(torch.isfinite(initial_wheel_clearance).all().item())
        and bool(torch.isfinite(initial_controlled_velocity).all().item())
    )
    for step in range(1, PHYSICS_STEPS + 1):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        metrics = env._closed_chain_metrics()
        controlled_velocity = controlled_joint_feedback_usd_to_control(
            env.robot.data.joint_vel[:, env._controlled_joint_ids]
        )
        wheel_clearance = _wheel_clearance(
            env,
            wheel_body_ids,
            wheel_collision_points_body,
        )
        penetration = torch.clamp(-wheel_clearance, min=0.0)
        penetration_increased = penetration > per_env_wheel_penetration
        per_env_wheel_penetration = torch.maximum(per_env_wheel_penetration, penetration)
        per_env_wheel_penetration_step = torch.where(
            penetration_increased,
            torch.full_like(per_env_wheel_penetration_step, step),
            per_env_wheel_penetration_step,
        )
        torque = torch.abs(
            controlled_joint_feedback_usd_to_control(
                env.robot.data.applied_torque[:, env._controlled_joint_ids]
            )
        )
        saturated = torque >= 0.99 * env._controlled_effort_limits
        saturation_count += saturated.to(torch.int32)
        consecutive = torch.where(saturated, consecutive + 1, torch.zeros_like(consecutive))
        max_consecutive = torch.maximum(max_consecutive, consecutive)
        maxima["loop_error_m"] = max(maxima["loop_error_m"], metrics["loop_error_max_m"])
        maxima["wheel_error_m"] = max(maxima["wheel_error_m"], metrics["wheel_error_max_m"])
        maxima["length_error_m"] = max(maxima["length_error_m"], metrics["length_error_max_m"])
        maxima["phi0_error_rad"] = max(maxima["phi0_error_rad"], metrics["phi0_error_max_rad"])
        maxima["wheel_penetration_m"] = max(
            maxima["wheel_penetration_m"], float(penetration.max().item())
        )
        maxima["leg_speed_rad_s"] = max(
            maxima["leg_speed_rad_s"], float(torch.abs(controlled_velocity[:, :4]).max().item())
        )
        maxima["wheel_speed_rad_s"] = max(
            maxima["wheel_speed_rad_s"], float(torch.abs(controlled_velocity[:, 4:]).max().item())
        )
        finite = finite and metrics["finite"] and bool(torch.isfinite(torque).all().item())
    saturation_fraction = saturation_count.float() / PHYSICS_STEPS
    maximum_saturation_fraction = float(saturation_fraction.max().item())
    maximum_consecutive_saturation = int(max_consecutive.max().item())
    checks = {
        "finite": finite,
        "loop_closure": maxima["loop_error_m"] <= MAX_LOOP_CLOSURE_ERROR_M,
        "wheel_fk": maxima["wheel_error_m"] <= MAX_FK_WHEEL_ERROR_M,
        "length_fk": maxima["length_error_m"] <= MAX_FK_LENGTH_ERROR_M,
        "phi0_fk": maxima["phi0_error_rad"] <= MAX_FK_PHI0_ERROR_RAD,
        "wheel_penetration": maxima["wheel_penetration_m"] <= PENETRATION_LIMIT_M,
        "leg_speed": maxima["leg_speed_rad_s"] <= LEG_SPEED_LIMIT_RAD_S,
        "wheel_speed": maxima["wheel_speed_rad_s"] <= WHEEL_SPEED_LIMIT_RAD_S,
        "saturation_fraction": maximum_saturation_fraction <= SATURATION_FRACTION_LIMIT,
        "saturation_consecutive": maximum_consecutive_saturation <= SATURATION_CONSECUTIVE_LIMIT,
    }
    per_environment = []
    for env_id in range(env.num_envs):
        per_environment.append(
            {
                "env_id": env_id,
                "q_reference_control_rad": env._q_reference[env_id].detach().cpu().tolist(),
                "q_reset_control_rad": initial_controlled_position[env_id].detach().cpu().tolist(),
                "reference_offset_rad": env._randomization.reference_offset[env_id].detach().cpu().tolist(),
                "root_height_offset_m": float(root_height_offset[env_id].item()),
                "initial_root_height_m": float(initial_root_height[env_id].item()),
                "initial_leg_length_fk_m": initial_virtual_leg["length_fk"][env_id].detach().cpu().tolist(),
                "initial_phi0_fk_rad": initial_virtual_leg["phi0_fk"][env_id].detach().cpu().tolist(),
                "initial_wheel_clearance_m": initial_wheel_clearance[env_id].detach().cpu().tolist(),
                "max_wheel_penetration_m": per_env_wheel_penetration[env_id].detach().cpu().tolist(),
                "max_wheel_penetration_step": per_env_wheel_penetration_step[env_id].detach().cpu().tolist(),
                "wheel_friction": float(env._randomization.wheel_friction[env_id].item()),
                "controlled_stiffness": env._randomization.realized_stiffness[env_id].detach().cpu().tolist(),
                "controlled_damping": env._randomization.realized_damping[env_id].detach().cpu().tolist(),
                "controlled_effort_limit": env._randomization.realized_effort_limit[env_id]
                .detach()
                .cpu()
                .tolist(),
            }
        )
    return {
        "name": name,
        "physics_steps": PHYSICS_STEPS,
        "duration_s": PHYSICS_STEPS * env.physics_dt,
        "root_velocity_world_usd": root_velocity.detach().cpu().tolist(),
        "maxima": maxima,
        "per_environment": per_environment,
        "maximum_effort_saturation_fraction": maximum_saturation_fraction,
        "maximum_consecutive_saturation_steps": maximum_consecutive_saturation,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    if args_cli.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    cfg = WheelLegFlatEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.randomization = (
        FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
        if args_cli.randomization_profile == "fudan-v1"
        else NOMINAL_TRAINING_PROFILE_V1
    )
    env = WheelLegFlatEnv(cfg)
    try:
        counters_before = {
            "sim_step_counter": int(env._sim_step_counter),
            "common_step_counter": int(env.common_step_counter),
            "episode_length": env.episode_length_buf.detach().cpu().tolist(),
        }
        env.reset()
        counters_after = {
            "sim_step_counter": int(env._sim_step_counter),
            "common_step_counter": int(env.common_step_counter),
            "episode_length": env.episode_length_buf.detach().cpu().tolist(),
        }
        reset_counter_unchanged = counters_before == counters_after
        wheel_body_ids, wheel_collision_points_body, wheel_collision_geometry = _wheel_collision_geometry(env)
        zero_velocity = torch.zeros((env.num_envs, 6), device=env.device)
        gate_a = _run_gate(
            env,
            name="GateAZeroRootVelocity",
            root_velocity=zero_velocity,
            wheel_body_ids=wheel_body_ids,
            wheel_collision_points_body=wheel_collision_points_body,
            align_root_height=args_cli.root_height_alignment == "fk",
        )
        gate_b_velocity = env._randomization.sample_root_velocity(env.num_envs)
        gate_b = _run_gate(
            env,
            name="GateBRandomRootVelocity",
            root_velocity=gate_b_velocity,
            wheel_body_ids=wheel_body_ids,
            wheel_collision_points_body=wheel_collision_points_body,
            align_root_height=args_cli.root_height_alignment == "fk",
        )
        actuator_sync = _actuator_sync(env)
        report = {
            "schema_version": "Phase1RandomizationRuntimeGateV2",
            "script_sha256": _sha256_file(Path(__file__)),
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "randomization_profile": args_cli.randomization_profile,
            "root_height_alignment": args_cli.root_height_alignment,
            "cache": env.randomization_audit["closed_chain_reset_cache"],
            "actuator_sync": actuator_sync,
            "episode_reset_counter_unchanged": reset_counter_unchanged,
            "episode_reset_counters_before": counters_before,
            "episode_reset_counters_after": counters_after,
            "wheel_collision_geometry": wheel_collision_geometry,
            "gate_a": gate_a,
            "gate_b": gate_b,
        }
        report["passed"] = (
            actuator_sync["passed"] and reset_counter_unchanged and gate_a["passed"] and gate_b["passed"]
        )
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if not report["passed"]:
            raise RuntimeError(f"Phase 1R runtime gate failed; see {args_cli.output.resolve()}")
    finally:
        env.close()


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
