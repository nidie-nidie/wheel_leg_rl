"""Read back PhysicsV5 and test speed-cap removal without training."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--randomized", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from pxr import PhysxSchema, Usd, UsdPhysics

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2
from wheelleg_dreamwaq.schemas.physics import (
    PHYSICS_SCHEMA_VERSION,
    UNRESTRICTED_SIM_VELOCITY,
    unrestricted_velocity_policy,
)
from wheelleg_dreamwaq.schemas.randomization import FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import build_base_task_contract_from_configs
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg


class ProbeEnv(WheelLegFlatEnv):
    def _setup_scene(self) -> None:
        super()._setup_scene()
        self.speed_probes = {}
        for index, (label, limit) in enumerate((
            ("historical_caps", 100.0),
            ("unrestricted", UNRESTRICTED_SIM_VELOCITY),
        )):
            cfg = RigidObjectCfg(
                prim_path=f"/World/{label}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.2, 0.2, 0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        linear_damping=0.0,
                        angular_damping=0.0,
                        max_linear_velocity=limit,
                        max_angular_velocity=limit,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 100.0 + index)),
            )
            body = RigidObject(cfg)
            self.speed_probes[label] = body
            self.scene.rigid_objects[label] = body


def main() -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    cfg = WheelLegFlatEnvCfg()
    cfg.scene.num_envs = 8
    cfg.sim.device = args.device
    if args.randomized:
        cfg.randomization = FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    env = ProbeEnv(cfg)
    try:
        env.reset()
        joint_limits = env.robot.root_physx_view.get_dof_max_velocities()
        assert tuple(joint_limits.shape) == (8, 26)
        assert torch.equal(joint_limits, torch.full_like(joint_limits, UNRESTRICTED_SIM_VELOCITY))
        bodies = []
        root = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot")
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = PhysxSchema.PhysxRigidBodyAPI(prim)
                linear = api.GetMaxLinearVelocityAttr().Get()
                angular = api.GetMaxAngularVelocityAttr().Get()
                assert linear == UNRESTRICTED_SIM_VELOCITY, (str(prim.GetPath()), linear)
                assert angular == UNRESTRICTED_SIM_VELOCITY, (str(prim.GetPath()), angular)
                bodies.append({"path": str(prim.GetPath()), "linear_limit": linear, "angular_limit_deg_s": angular})
        assert len(bodies) == 27
        historical_cache_rejected = None
        if args.randomized:
            cache = env.closed_chain_reset_cache_artifact
            assert cache["identity"]["physics_schema_version"] == PHYSICS_SCHEMA_VERSION
            historical = deepcopy(cache)
            historical["identity"]["physics_schema_version"] = "PhysicsV4"
            try:
                env._load_closed_chain_reset_cache(historical)
            except RuntimeError as error:
                assert "identity mismatch" in str(error)
                historical_cache_rejected = True
            else:
                raise AssertionError("Historical PhysicsV4 cache was accepted")
        params = {
            "effort": env.robot.root_physx_view.get_dof_max_forces(),
            "damping": env.robot.root_physx_view.get_dof_dampings(),
            "armature": env.robot.root_physx_view.get_dof_armatures(),
        }
        for name, actuator in env.robot.actuators.items():
            if args.randomized and name != "passive":
                continue  # Realized active gains/efforts remain managed by the existing profile.
            group_cfg = cfg.robot_cfg.actuators[name]
            for parameter, expected in (
                ("effort", group_cfg.effort_limit_sim),
                ("damping", group_cfg.damping),
                ("armature", group_cfg.armature),
            ):
                indices = actuator.joint_indices
                if isinstance(indices, torch.Tensor):
                    indices = indices.to(params[parameter].device)
                actual = params[parameter][:, indices]
                assert torch.allclose(actual, torch.full_like(actual, expected), rtol=0., atol=1.e-6)
        for body in env.speed_probes.values():
            state = body.data.default_root_state.clone()
            state[:, 7] = 200.0
            state[:, 10] = 10.0
            body.write_root_state_to_sim(state)
        for _ in range(3):
            env.sim.step(render=False)
            for body in env.speed_probes.values():
                body.update(env.physics_dt)
        probe_results = {
            label: {
                "linear_speed_mps": float(body.data.root_com_lin_vel_w.norm(dim=-1)[0]),
                "angular_speed_rad_s": float(body.data.root_link_ang_vel_w.norm(dim=-1)[0]),
            }
            for label, body in env.speed_probes.items()
        }
        assert abs(probe_results["unrestricted"]["linear_speed_mps"] - 200.0) < 1.e-3
        assert abs(probe_results["unrestricted"]["angular_speed_rad_s"] - 10.0) < 1.e-3
        assert probe_results["historical_caps"]["linear_speed_mps"] < 150.0
        assert probe_results["historical_caps"]["angular_speed_rad_s"] < 5.0
        observations, _ = env.reset()
        max_loop_error = 0.0
        reset_count = 0
        generator = torch.Generator(device=env.device).manual_seed(20261011)
        with torch.inference_mode():
            for _ in range(args.steps):
                actions = 2.0 * torch.rand((8, 6), generator=generator, device=env.device) - 1.0
                observations, rewards, terminated, truncated, _ = env.step(actions)
                assert torch.isfinite(observations["policy"]).all()
                assert torch.isfinite(observations["critic"]).all()
                assert torch.isfinite(rewards).all()
                anchors = env._world_anchors(env._loop_body_ids, env._loop_local_anchors).reshape(8, -1, 2, 3)
                loop_error = torch.linalg.vector_norm(anchors[:, :, 0] - anchors[:, :, 1], dim=-1)
                max_loop_error = max(max_loop_error, float(loop_error.max()))
                reset_count += int((terminated | truncated).sum())
        assert max_loop_error <= 0.005
        report = {
            "physics_schema_version": PHYSICS_SCHEMA_VERSION,
            "velocity_limit_policy": unrestricted_velocity_policy(),
            "base_task_contract": build_base_task_contract_from_configs(
                asset_bundle_hash=ASSET_BUNDLE_V2.bundle_hash,
                env_cfg=cfg,
                clip_actions=1.0,
            ),
            "randomized": args.randomized,
            "historical_cache_rejected": historical_cache_rejected,
            "robot_bodies": bodies,
            "joint_velocity_limit_shape": list(joint_limits.shape),
            "joint_velocity_limits": joint_limits[0].tolist(),
            "motor_effort_damping_armature": {name: value[0].tolist() for name, value in params.items()},
            "free_body_probes": probe_results,
            "steps": args.steps,
            "policy_shape": list(observations["policy"].shape),
            "critic_shape": list(observations["critic"].shape),
            "max_loop_error_m": max_loop_error,
            "reset_count": reset_count,
            "finite": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        print("UNRESTRICTED_DYNAMICS_VERIFIED", args.output, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".error.txt").write_text(error, encoding="utf-8")
        print(error, flush=True)
        raise
    finally:
        simulation_app.close()
