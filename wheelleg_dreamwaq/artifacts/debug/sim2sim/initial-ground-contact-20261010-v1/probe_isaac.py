"""Read-only production physics probe; all new files stay in this artifact directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import tensordict  # Import before Kit on Windows, as in existing integration probes.

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reset-cache", type=Path, required=True)
parser.add_argument("--actor", type=Path, required=True)
parser.add_argument("--contact-sensors", action="store_true")
parser.add_argument("--actions-from", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.assets.paths import asset_root_v2
from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG
from wheelleg_dreamwaq.schemas.isaac_evaluation import EVALUATION_SEED, FORMAL_SCENARIOS
from wheelleg_dreamwaq.schemas.randomization import NOMINAL_EVALUATION_PROFILE_V1
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def protected():
    paths = list((PROJECT / "source").rglob("*.py"))
    paths += list((PROJECT / "scripts").glob("*.py"))
    paths += [asset_root_v2() / item.relative_path for item in ASSET_BUNDLE_V2.files]
    paths += [args.reset_cache, args.actor, PROJECT.parent / "docs/2026-10-03-wheelleg-dreamwaq-architecture.md"]
    paths += list((PROJECT / "sim2sim/mujoco/wheelleg_mujoco").glob("*.py"))
    paths += [PROJECT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"]
    return {str(path.resolve()): digest(path) for path in paths}


def cpu(value):
    return value.detach().cpu().numpy().copy()


class ContactProbeEnv(WheelLegFlatEnv):
    # Only add observers in a separate config. Production reset/step are inherited.
    def _setup_scene(self):
        super()._setup_scene()
        self.probe_sensors = []
        if args.contact_sensors:
            for side in ("left", "right"):
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/jwheel_{side}",
                    track_contact_points=False, track_friction_forces=False,
                    max_contact_data_count_per_prim=8,
                ))
                self.scene.sensors[f"probe_{side}"] = sensor
                self.probe_sensors.append(sensor)


def rotated(points, quat_xyzw):
    q = quat_xyzw[:3]
    return points + 2.0 * np.cross(q, np.cross(q, points) + quat_xyzw[3] * points)


def main():
    args.output.mkdir(parents=True, exist_ok=False)
    before = protected()
    verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    cache = torch.load(args.reset_cache, map_location="cpu", weights_only=False)
    cfg = WheelLegFlatEnvCfg()
    cfg.seed = EVALUATION_SEED
    cfg.scene.num_envs = 8
    cfg.sim.device = args.device
    cfg.randomization = NOMINAL_EVALUATION_PROFILE_V1
    if args.contact_sensors:
        cfg.robot_cfg = WHEELLEG_CFG.replace(spawn=WHEELLEG_CFG.spawn.replace(activate_contact_sensors=True))
    env = ContactProbeEnv(cfg, closed_chain_reset_cache=cache)
    stage = env.sim.get_initial_stage()
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    ground = stage.GetPrimAtPath(cfg.ground.prim_path)
    ground_range = bbox.ComputeWorldBound(ground).ComputeAlignedRange()
    ground_top = float(ground_range.GetMax()[2])
    expected_ground_top = cfg.ground.translation[2] + cfg.ground.spawn.size[2] / 2.0
    assert abs(ground_top - expected_ground_top) < 1e-6, (ground_top, expected_ground_top)
    shapes, geometry = [], []
    for side in ("left", "right"):
        body_path = f"/World/envs/env_0/Robot/jwheel_{side}"
        body = stage.GetPrimAtPath(body_path)
        colliders = [prim for prim in Usd.PrimRange(body)
                     if prim.HasAPI(UsdPhysics.CollisionAPI)
                     and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
        assert len(colliders) == 1
        prim = colliders[0]
        assert prim.IsA(UsdGeom.Mesh), str(prim.GetPath())
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        matrix, _ = xforms.ComputeRelativeTransform(prim, body)
        local = np.asarray([matrix.Transform(Gf.Vec3d(*point)) for point in points], dtype=np.float64)
        body_ids, names = env.robot.find_bodies([f"jwheel_{side}"], preserve_order=True)
        assert names == [f"jwheel_{side}"]
        shapes.append((body_ids[0], local))
        api = PhysxSchema.PhysxCollisionAPI(prim)
        attrs = {key: prim.GetAttribute(key).Get() for key in (
            "physxCollision:contactOffset", "physxCollision:restOffset",
            "physxConvexHullCollision:hullVertexLimit", "physxConvexHullCollision:minThickness")}
        geometry.append({"side": side, "body_id": body_ids[0], "collision_path": str(prim.GetPath()),
                         "approximation": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                         "point_count": len(points), "local_min": local.min(axis=0).tolist(),
                         "local_max": local.max(axis=0).tolist(), "authored_offsets": attrs})

    view = env.robot.root_physx_view
    offsets = {"contact_offsets": cpu(view.get_contact_offsets()).tolist(),
               "rest_offsets": cpu(view.get_rest_offsets()).tolist()}
    samples = []
    case = "constructor"
    step_index = 0

    def snapshot(phase):
        links = cpu(view.get_link_transforms())
        roots = cpu(view.get_root_transforms())
        gaps, centers = [], []
        for body_id, points in shapes:
            gap, center = [], []
            for row in links:
                pose = row[body_id]
                world = rotated(points, pose[3:7]) + pose[:3]
                gap.append(float(world[:, 2].min() - ground_top))
                center.append(pose[:3].tolist())
            gaps.append(gap)
            centers.append(center)
        force = None
        if env.probe_sensors:
            force = np.stack([cpu(sensor.contact_physx_view.get_net_contact_forces(dt=env.physics_dt))
                              .reshape(8, -1, 3).sum(axis=1) for sensor in env.probe_sensors], axis=1).tolist()
        return {"case": case, "phase": phase, "physics_step": step_index,
                "elapsed_s": step_index * env.physics_dt, "absolute_sim_time_s": float(env.sim.current_time),
                "root_pose_xyzw": roots.tolist(), "root_velocity_world": cpu(view.get_root_velocities()).tolist(),
                "wheel_body_centers_world": np.asarray(centers).transpose(1, 0, 2).tolist(),
                "wheel_collision_bottom_gap_m": np.asarray(gaps).T.tolist(),
                "normal_force_world_n": force,
                "force_freshness": "last_solved_physics_step" if step_index else "no_new_episode_physics_solve_yet",
                "joint_position": cpu(view.get_dof_positions()).tolist(),
                "joint_velocity": cpu(view.get_dof_velocities()).tolist()}

    samples.append(snapshot("constructor_after_cache_load"))
    original_step = env.sim.step

    def observed_step(*positional, **keywords):
        nonlocal step_index
        result = original_step(*positional, **keywords)
        step_index += 1
        samples.append(snapshot("post_physics_step"))
        return result

    env.sim.step = observed_step
    reset_before = float(env.sim.current_time)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)  # Same reset entry used by formal evaluation.
    reset_after = float(env.sim.current_time)
    assert step_index == 0 and reset_before == reset_after
    env._commands.copy_(torch.tensor([command for _, command in FORMAL_SCENARIOS], device=env.device))
    obs = wrapped.get_observations()
    history = torch.cat([obs["policy"]] * 5, dim=-1)
    if args.actions_from:
        actions_record = json.loads(args.actions_from.read_text(encoding="utf-8"))
        shared = torch.tensor(actions_record["shared_first_action_clipped"], device=env.device)
    else:
        actor = torch.jit.load(str(args.actor), map_location=env.device).eval()
        with torch.inference_mode():
            raw = actor(history)
        assert raw.shape == (8, 6) and torch.isfinite(raw).all()
        shared = raw.clamp(-1.0, 1.0)
        actions_record = {"shared_first_action_raw": cpu(raw).tolist(),
                          "shared_first_action_clipped": cpu(shared).tolist(),
                          "initial_policy_history": cpu(history).tolist()}
    (args.output / "actions.json").write_text(json.dumps(actions_record, indent=2), encoding="utf-8")
    for name, action in (("zero_action", torch.zeros((8, 6), device=env.device)), ("shared_first_action_hold", shared)):
        case = name
        step_index = 0
        start = float(env.sim.current_time)
        wrapped.reset()
        assert step_index == 0 and float(env.sim.current_time) == start
        env._commands.copy_(torch.tensor([command for _, command in FORMAL_SCENARIOS], device=env.device))
        samples.append(snapshot("reset_returned_before_first_action"))
        for _ in range(5):
            _, _, dones, _ = wrapped.step(action)
            assert not dones.any(), "Probe crossed an episode boundary"
    env.sim.step = original_step
    after = protected()
    assert before == after
    report = {"schema_version": "InitialGroundContactIsaacProbeV1", "debug_only": True,
              "formal_ranking_eligible": False, "contact_sensors": args.contact_sensors,
              "production_reset_and_step_inherited": True, "seed": cfg.seed, "num_envs": 8,
              "physics_dt_s": env.physics_dt, "reset_physics_time_advanced_s": reset_after - reset_before,
              "cache_sha256": digest(args.reset_cache), "actor_sha256": digest(args.actor),
              "ground_top_z_m": ground_top, "geometry": geometry, "compiled_shape_offsets": offsets,
              "normal_force_semantics": "PhysX net normal force; no fresh force solve exists at reset t=0",
              "gap_semantics": "Runtime PhysX link pose applied to authored mesh vertices; convex-hull support plane, excluding offsets and cooking tolerance",
              "sources_unchanged": before == after, "protected_source_sha256": after, "samples": samples}
    (args.output / "evidence.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("PROBE_COMPLETE", args.output, flush=True)
    wrapped.close()


try:
    main()
finally:
    app.close()
