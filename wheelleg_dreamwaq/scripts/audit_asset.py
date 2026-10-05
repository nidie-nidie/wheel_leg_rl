from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Audit WheelLeg AssetBundleV1/V2 and the V2 reference load.")
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "artifacts" / "phase0" / "asset-audit.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from pxr import Usd, UsdGeom, UsdPhysics

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V1, ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.assets.asset_overrides import (
    assert_embedded_ground_absent,
    configure_wheel_only_collisions,
    is_wheel_collision_path,
)
from wheelleg_dreamwaq.assets.paths import asset_root, asset_root_v2
from wheelleg_dreamwaq.schemas.physics import SIM_DT_S


def _joint_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    return [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]


def _rigid_body_paths(stage: Usd.Stage) -> list[str]:
    return sorted(str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI))


def _physics_scene_paths(stage: Usd.Stage) -> list[str]:
    return sorted(str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene))


def _collision_paths(stage: Usd.Stage) -> tuple[list[str], list[str]]:
    all_paths: list[str] = []
    enabled_paths: list[str] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        all_paths.append(path)
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
            enabled_paths.append(path)
    return sorted(all_paths), sorted(enabled_paths)


def _mass_report(stage: Usd.Stage) -> dict:
    rows: list[dict] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        if mass is None:
            raise RuntimeError(f"Rigid body has no authored mass: {prim.GetPath()}")
        path = str(prim.GetPath())
        rows.append({"path": path, "mass_kg": float(mass), "dummy": "dummy" in path.lower()})
    dummy_rows = [row for row in rows if row["dummy"]]
    return {
        "total_mass_kg": sum(row["mass_kg"] for row in rows),
        "dummy_body_count": len(dummy_rows),
        "dummy_mass_kg": sum(row["mass_kg"] for row in dummy_rows),
        "bodies": rows,
    }


def _audit_source_stage(entry_path: Path, *, expected, expect_embedded_ground: bool) -> dict:
    stage = Usd.Stage.Open(str(entry_path))
    if stage is None:
        raise RuntimeError(f"Unable to open source USD: {entry_path}")

    joints = _joint_prims(stage)
    joint_paths = sorted(str(prim.GetPath()) for prim in joints)
    excluded_joint_paths = sorted(
        str(prim.GetPath())
        for prim in joints
        if bool(prim.GetAttribute("physics:excludeFromArticulation").Get())
    )
    loop_attribute_debug = {
        path: {
            attribute.GetName(): attribute.Get()
            for attribute in stage.GetPrimAtPath(path).GetAttributes()
            if "exclude" in attribute.GetName().lower() or "jointenabled" in attribute.GetName().lower()
        }
        for path in expected.loop_joint_paths
    }
    articulation_roots = sorted(
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    embedded_ground = stage.GetPrimAtPath("/wheel_leg_urdf4/GroundPlane/CollisionPlane")
    source_ground_enabled = (
        UsdPhysics.CollisionAPI(embedded_ground).GetCollisionEnabledAttr().Get()
        if embedded_ground.IsValid()
        else None
    )
    collision_paths, enabled_collision_paths = _collision_paths(stage)
    mass_report = _mass_report(stage)
    left_mirror_enabled = [path for path in enabled_collision_paths if path.startswith("/wheel_leg_urdf4/jMK/")]
    right_mirror_enabled = [path for path in enabled_collision_paths if path.startswith("/wheel_leg_urdf4/jEC/")]

    controlled_matches = {
        name: [path for path in joint_paths if path.rsplit("/", 1)[-1] == name]
        for name in expected.controlled_joints
    }
    report = {
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "root_prims": [str(prim.GetPath()) for prim in stage.GetPseudoRoot().GetChildren()],
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "articulation_roots": articulation_roots,
        "joint_count": len(joint_paths),
        "tree_joint_count": len(joint_paths) - len(excluded_joint_paths),
        "loop_joint_count": len(excluded_joint_paths),
        "joint_paths": joint_paths,
        "excluded_joint_paths": excluded_joint_paths,
        "loop_attribute_debug": loop_attribute_debug,
        "controlled_joint_matches": controlled_matches,
        "rigid_body_count": len(_rigid_body_paths(stage)),
        "rigid_body_paths": _rigid_body_paths(stage),
        "mass": mass_report,
        "embedded_ground_collision_enabled": source_ground_enabled,
        "embedded_ground_absent": not stage.GetPrimAtPath("/wheel_leg_urdf4/GroundPlane").IsValid(),
        "collision_paths": collision_paths,
        "enabled_collision_paths": enabled_collision_paths,
        "mirror_collision_asymmetry": {
            "jMK_enabled_paths": left_mirror_enabled,
            "jEC_enabled_paths": right_mirror_enabled,
        },
    }

    failures: list[str] = []
    if report["default_prim"] != expected.default_prim:
        failures.append("default prim mismatch")
    if tuple(report["root_prims"]) != expected.root_prims:
        failures.append("root prim list mismatch")
    if report["up_axis"] != expected.up_axis:
        failures.append("up axis mismatch")
    if report["meters_per_unit"] != expected.meters_per_unit:
        failures.append("meters-per-unit mismatch")
    if expected.articulation_root not in articulation_roots:
        failures.append("articulation root mismatch")
    if report["tree_joint_count"] != expected.tree_joint_count:
        failures.append("tree joint count mismatch")
    if report["loop_joint_count"] != expected.loop_joint_count:
        failures.append("loop joint count mismatch")
    if tuple(excluded_joint_paths) != tuple(sorted(expected.loop_joint_paths)):
        failures.append("loop joint paths mismatch")
    if report["rigid_body_count"] != expected.rigid_body_count:
        failures.append("rigid body count mismatch")
    if abs(mass_report["total_mass_kg"] - expected.total_mass_kg) > 1.0e-8:
        failures.append("total mass mismatch")
    if mass_report["dummy_body_count"] != expected.dummy_body_count:
        failures.append("dummy body count mismatch")
    if abs(mass_report["dummy_mass_kg"] - expected.dummy_mass_kg) > 1.0e-8:
        failures.append("dummy mass mismatch")
    if not all(len(matches) == 1 for matches in controlled_matches.values()):
        failures.append("controlled joint name mismatch")
    if expect_embedded_ground and source_ground_enabled is not True:
        failures.append("source embedded ground is not collision-enabled")
    if not expect_embedded_ground and report["embedded_ground_absent"] is not True:
        failures.append("AssetBundleV2 still contains embedded ground")
    if not left_mirror_enabled or right_mirror_enabled:
        failures.append("expected source jMK/jEC collision asymmetry is missing")
    if failures:
        raise RuntimeError(
            "Source USD audit failed: "
            + "; ".join(failures)
            + f"; joint_count={len(joint_paths)} excluded={excluded_joint_paths} attrs={loop_attribute_debug}"
        )
    return report


def _audit_reference_load_v2(entry_path: Path) -> dict:
    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=SIM_DT_S, device=args_cli.device))
    robot_prim_path = "/World/envs/env_0/Robot"
    cfg = sim_utils.UsdFileCfg(usd_path=entry_path.as_posix())
    cfg.func(robot_prim_path, cfg)
    stage = sim.stage

    assert_embedded_ground_absent(stage, robot_prim_path)
    disabled_robot_collisions, kept_robot_collisions = configure_wheel_only_collisions(stage, robot_prim_path)
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.update_stage()

    physics_scenes_before_reset = _physics_scene_paths(stage)
    embedded_ground_absent_before_reset = not stage.GetPrimAtPath(
        f"{robot_prim_path}/GroundPlane"
    ).IsValid()
    sim.reset()
    physics_scenes_after_reset = _physics_scene_paths(stage)
    embedded_ground_absent_after_reset = not stage.GetPrimAtPath(
        f"{robot_prim_path}/GroundPlane"
    ).IsValid()

    robot_physics_scenes = [path for path in physics_scenes_after_reset if path.startswith(robot_prim_path + "/")]
    _, enabled_collision_paths_after_reset = _collision_paths(stage)
    enabled_robot_collisions_after_reset = [
        path for path in enabled_collision_paths_after_reset if path.startswith(robot_prim_path + "/")
    ]
    report = {
        "robot_prim_path": robot_prim_path,
        "referenced_default_prim_valid": stage.GetPrimAtPath(robot_prim_path).IsValid(),
        "embedded_ground_absent_before_reset": embedded_ground_absent_before_reset,
        "embedded_ground_absent_after_reset": embedded_ground_absent_after_reset,
        "physics_scenes_before_reset": physics_scenes_before_reset,
        "physics_scenes_after_reset": physics_scenes_after_reset,
        "robot_namespace_physics_scenes": robot_physics_scenes,
        "global_ground_valid": stage.GetPrimAtPath("/World/Ground").IsValid(),
        "disabled_robot_collision_paths": disabled_robot_collisions,
        "kept_robot_collision_paths": kept_robot_collisions,
        "enabled_robot_collision_paths_after_reset": enabled_robot_collisions_after_reset,
    }
    if not report["referenced_default_prim_valid"]:
        raise RuntimeError("Referenced default prim did not load")
    if not embedded_ground_absent_before_reset or not embedded_ground_absent_after_reset:
        raise RuntimeError("AssetBundleV2 embedded ground appeared during reference load")
    if len(physics_scenes_after_reset) != 1:
        raise RuntimeError(f"Expected one PhysicsScene, found {physics_scenes_after_reset}")
    if robot_physics_scenes:
        raise RuntimeError(f"PhysicsScene leaked into robot namespace: {robot_physics_scenes}")
    if len(enabled_robot_collisions_after_reset) != 2 or not all(
        is_wheel_collision_path(path, robot_prim_path) for path in enabled_robot_collisions_after_reset
    ):
        raise RuntimeError(
            "Runtime collision override did not preserve exactly the two wheel collision shapes: "
            f"{enabled_robot_collisions_after_reset}"
        )
    return report


def main() -> None:
    v1_report = verify_asset_bundle()
    v2_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    source_v1 = _audit_source_stage(
        asset_root() / ASSET_BUNDLE_V1.entry_file,
        expected=ASSET_BUNDLE_V1,
        expect_embedded_ground=True,
    )
    source_v2 = _audit_source_stage(
        asset_root_v2() / ASSET_BUNDLE_V2.entry_file,
        expected=ASSET_BUNDLE_V2,
        expect_embedded_ground=False,
    )
    reference_v2 = _audit_reference_load_v2(asset_root_v2() / ASSET_BUNDLE_V2.entry_file)
    post_v1_report = verify_asset_bundle()
    if v1_report.bundle_hash != post_v1_report.bundle_hash:
        raise RuntimeError("Asset bundle changed during audit")

    output = {
        "asset_bundle_v1": asdict(v1_report),
        "asset_bundle_v2": asdict(v2_report),
        "source_v1": source_v1,
        "source_v2": source_v2,
        "reference_load_v2": reference_v2,
        "source_v1_files_unchanged": True,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"[INFO] Asset audit written to: {args_cli.output.resolve()}")


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
