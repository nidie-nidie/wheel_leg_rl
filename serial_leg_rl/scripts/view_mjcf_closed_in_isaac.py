#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF_PATH = PROJECT_ROOT / "wheel_leg_urdf4_mjcf" / "wheel_leg_urdf4" / "wheel_leg_urdf4_closed.xml"
DEFAULT_USD_DIR = Path("/tmp/serial_leg_rl_mjcf_assets")

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="Convert and view the closed-loop MJCF wheel-leg model in Isaac Sim.")
parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF_PATH, help="Closed-loop MJCF XML path.")
parser.add_argument("--usd_dir", type=Path, default=DEFAULT_USD_DIR, help="Directory for converted USD files.")
parser.add_argument("--num_steps", type=int, default=3000, help="Number of simulation steps.")
parser.add_argument("--real_time", action="store_true", help="Throttle stepping to roughly real time.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdPhysics  # noqa: E402


LOOP_JOINT_PAIRS = (
    ("left_loop_1", "jIO_dummy_child_link1", "jMK_dummy_child1"),
    ("left_loop_2", "jIO_dummy_child_link2", "jMK_dummy_child2"),
    ("right_loop_1", "jAG_dummy_child_link1", "jEC_dummy_child_link1"),
    ("right_loop_2", "jAG_dummy_child_link2", "jEC_dummy_child_link2"),
)


def _find_prim_path_by_name(stage, prim_name: str) -> Sdf.Path | None:
    matches = [prim.GetPath() for prim in stage.Traverse() if prim.GetName() == prim_name]
    if not matches:
        print(f"[MJCF_VIEW] missing prim for loop joint body: {prim_name}", flush=True)
        return None
    body_matches = [path for path in matches if "/joints/" not in path.pathString]
    if body_matches:
        matches = body_matches
    if len(matches) > 1:
        print(f"[MJCF_VIEW] multiple prims named {prim_name}; using {matches[0]}", flush=True)
    return matches[0]


def _add_loop_spherical_joints() -> None:
    stage = stage_utils.get_current_stage()
    stage.DefinePrim("/World/Robot/LoopJoints", "Xform")

    created = 0
    for joint_name, body0_name, body1_name in LOOP_JOINT_PAIRS:
        body0_path = _find_prim_path_by_name(stage, body0_name)
        body1_path = _find_prim_path_by_name(stage, body1_name)
        if body0_path is None or body1_path is None:
            continue

        joint_path = Sdf.Path(f"/World/Robot/LoopJoints/{joint_name}")
        joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([body0_path])
        joint.CreateBody1Rel().SetTargets([body1_path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateJointEnabledAttr().Set(True)
        created += 1
        print(f"[MJCF_VIEW] created spherical joint {joint_name}: {body0_path} <-> {body1_path}", flush=True)

    print(f"[MJCF_VIEW] isaac_loop_joints={created}/{len(LOOP_JOINT_PAIRS)}", flush=True)


def _disable_reduced_articulation() -> None:
    stage = stage_utils.get_current_stage()
    removed = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            removed += 1
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            removed += 1
    print(f"[MJCF_VIEW] removed_articulation_apis={removed}", flush=True)


def main() -> None:
    mjcf_path = args_cli.mjcf.resolve()
    usd_dir = args_cli.usd_dir.resolve()

    enable_extension("isaacsim.asset.importer.mjcf")
    for _ in range(10):
        if not simulation_app.is_running():
            return
        simulation_app.update()

    converter = MjcfConverter(
        MjcfConverterCfg(
            asset_path=mjcf_path.as_posix(),
            usd_dir=usd_dir.as_posix(),
            usd_file_name="wheel_leg_urdf4_closed.usd",
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=False,
            import_sites=True,
            self_collision=False,
        )
    )
    print(f"[MJCF_VIEW] mjcf={mjcf_path}", flush=True)
    print(f"[MJCF_VIEW] usd={converter.usd_path}", flush=True)

    sim_cfg = SimulationCfg(dt=1.0 / 200.0, device=args_cli.device or "cuda:0")
    sim = SimulationContext(sim_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    floor_cfg = sim_utils.CuboidCfg(
        size=(2.0, 2.0, 0.02),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
    )
    floor_cfg.func("/World/floor", floor_cfg, translation=(0.02, -1.27, -0.02))

    robot_cfg = sim_utils.UsdFileCfg(usd_path=converter.usd_path)
    robot_cfg.func("/World/Robot", robot_cfg, translation=(0.0, 0.0, 0.25))
    _disable_reduced_articulation()
    _add_loop_spherical_joints()

    sim.set_camera_view(eye=(-0.6, -2.6, 0.9), target=(0.02, -1.27, 0.12))
    sim.reset()

    for step in range(args_cli.num_steps):
        if not simulation_app.is_running():
            break
        sim.step()
        if step % 200 == 0:
            print(f"[MJCF_VIEW] step={step}", flush=True)
        if args_cli.real_time:
            time.sleep(sim_cfg.dt)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
