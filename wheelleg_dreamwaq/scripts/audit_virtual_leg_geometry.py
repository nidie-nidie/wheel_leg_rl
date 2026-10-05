from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Audit WheelLeg virtual-leg joint-axis geometry from AssetBundleV2.")
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "artifacts" / "phase1_v4" / "usd-anchor-audit.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.assets.paths import asset_root_v2
from wheelleg_dreamwaq.kinematics.virtual_leg import VIRTUAL_LEG_KINEMATICS_VERSION


AXIS_VECTOR = {
    "X": Gf.Vec3d(1.0, 0.0, 0.0),
    "Y": Gf.Vec3d(0.0, 1.0, 0.0),
    "Z": Gf.Vec3d(0.0, 0.0, 1.0),
}


def _vec3(value) -> list[float]:
    return [float(value[index]) for index in range(3)]


def _quat(value) -> list[float]:
    imaginary = value.GetImaginary()
    return [float(value.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2])]


def _normalize(value: Gf.Vec3d) -> Gf.Vec3d:
    length = value.GetLength()
    if length <= 1.0e-12:
        raise RuntimeError("Joint axis has zero length")
    return value / length


def _source_layer(attribute) -> str:
    stack = attribute.GetPropertyStack(Usd.TimeCode.Default())
    if not stack:
        return ""
    return stack[0].layer.identifier


def _joint_prims_by_name(stage: Usd.Stage) -> dict[str, Usd.Prim]:
    result: dict[str, Usd.Prim] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        name = prim.GetName()
        if name in result:
            raise RuntimeError(f"Duplicate USD joint name: {name}")
        result[name] = prim
    return result


def _joint_record(stage: Usd.Stage, cache: UsdGeom.XformCache, prim: Usd.Prim) -> dict:
    joint = UsdPhysics.Joint(prim)
    body0_targets = joint.GetBody0Rel().GetTargets()
    body1_targets = joint.GetBody1Rel().GetTargets()
    if len(body0_targets) != 1 or len(body1_targets) != 1:
        raise RuntimeError(f"Joint must bind exactly two bodies: {prim.GetPath()}")

    axis_token = str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
    if axis_token not in AXIS_VECTOR:
        raise RuntimeError(f"Unsupported joint axis {axis_token!r}: {prim.GetPath()}")

    sides: list[dict] = []
    for index, (body_path, local_pos_attr, local_rot_attr) in enumerate(
        (
            (body0_targets[0], joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
            (body1_targets[0], joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()),
        )
    ):
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid() or not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Joint side is not a rigid body: {prim.GetPath()} body{index}={body_path}")
        local_anchor = local_pos_attr.Get()
        local_rotation = local_rot_attr.Get()
        body_matrix = cache.GetLocalToWorldTransform(body_prim)
        world_anchor = body_matrix.Transform(Gf.Vec3d(local_anchor))
        axis_body = Gf.Rotation(local_rotation).TransformDir(AXIS_VECTOR[axis_token])
        world_axis = _normalize(body_matrix.ExtractRotation().TransformDir(axis_body))
        sides.append(
            {
                "side": index,
                "body_path": str(body_path),
                "body_name": body_path.name,
                "local_anchor_m": _vec3(local_anchor),
                "local_rotation_wxyz": _quat(local_rotation),
                "axis_body": _vec3(_normalize(axis_body)),
                "world_anchor_m": _vec3(world_anchor),
                "world_axis": _vec3(world_axis),
                "anchor_source_layer": _source_layer(local_pos_attr),
                "rotation_source_layer": _source_layer(local_rot_attr),
            }
        )

    anchor_error = math.dist(sides[0]["world_anchor_m"], sides[1]["world_anchor_m"])
    axis_dot = sum(a * b for a, b in zip(sides[0]["world_axis"], sides[1]["world_axis"], strict=True))
    return {
        "joint_name": prim.GetName(),
        "joint_path": str(prim.GetPath()),
        "joint_type": prim.GetTypeName(),
        "axis_token": axis_token,
        "prim_source_layers": [spec.layer.identifier for spec in prim.GetPrimStack()],
        "sides": sides,
        "joint_anchor_error_m": anchor_error,
        "joint_axis_dot": axis_dot,
    }


def _select_side(record: dict, body_name: str) -> dict:
    matches = [side for side in record["sides"] if side["body_name"] == body_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {body_name!r} side for {record['joint_name']}, found "
            f"{[side['body_name'] for side in record['sides']]}"
        )
    return matches[0]


def _joint_midpoint(record: dict) -> Gf.Vec3d:
    lhs = record["sides"][0]["world_anchor_m"]
    rhs = record["sides"][1]["world_anchor_m"]
    return Gf.Vec3d(*(0.5 * (lhs[index] + rhs[index]) for index in range(3)))


def _fk_geometry(
    cache: UsdGeom.XformCache,
    stage: Usd.Stage,
    records: dict[str, dict],
    point_joints: dict[str, str],
) -> dict:
    base_matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ASSET_BUNDLE_V2.articulation_root))
    world_to_base = base_matrix.GetInverse()
    points = {
        name: world_to_base.Transform(_joint_midpoint(records[joint_name]))
        for name, joint_name in point_joints.items()
    }

    def vector(start: str, end: str) -> Gf.Vec3d:
        return points[end] - points[start]

    vectors = {
        "v_ij": vector("I", "J"),
        "v_il": vector("I", "L"),
        "v_ip": vector("I", "P"),
        "v_jm": vector("J", "M"),
        "v_ml": vector("M", "L"),
        "v_mk": vector("M", "K"),
        "v_kn": vector("K", "N"),
        "v_pn": vector("P", "N"),
        "v_pw": vector("P", "W"),
    }
    return {
        "point_joint_names": point_joints,
        "points_base_m": {name: _vec3(value) for name, value in points.items()},
        "vectors_s_z_m": {name.lower(): [float(value[1]), float(value[2])] for name, value in vectors.items()},
        "out_of_plane_x_m": {name.lower(): float(value[0]) for name, value in vectors.items()},
        "max_abs_out_of_plane_x_m": max(abs(float(value[0])) for value in vectors.values()),
    }


def _build_leg_record(
    stage: Usd.Stage,
    cache: UsdGeom.XformCache,
    records: dict[str, dict],
    *,
    hip_names: tuple[str, str],
    wheel_name: str,
    point_joints: dict[str, str],
) -> dict:
    hips = [records[name] for name in hip_names]
    base_sides = [_select_side(record, "base_link") for record in hips]
    hip_anchor_error = math.dist(base_sides[0]["world_anchor_m"], base_sides[1]["world_anchor_m"])
    hip_axis_dot = sum(
        a * b for a, b in zip(base_sides[0]["world_axis"], base_sides[1]["world_axis"], strict=True)
    )
    if hip_anchor_error > 1.0e-4:
        raise RuntimeError(f"Hip joints are not coaxial: {hip_names} anchor error={hip_anchor_error:.9g} m")
    if abs(hip_axis_dot) < 0.99999:
        raise RuntimeError(f"Hip joint axes are not aligned: {hip_names} dot={hip_axis_dot:.9g}")

    wheel_joint = records[wheel_name]
    wheel_side = _select_side(wheel_joint, wheel_name)
    selected_hip = dict(base_sides[0])
    selected_hip["joint_name"] = hip_names[0]
    selected_hip["joint_path"] = hips[0]["joint_path"]
    selected_wheel = dict(wheel_side)
    selected_wheel["joint_name"] = wheel_name
    selected_wheel["joint_path"] = wheel_joint["joint_path"]
    leg = {
        "hip_joint_names": list(hip_names),
        "wheel_joint_name": wheel_name,
        "hip_coaxial_anchor_error_m": hip_anchor_error,
        "hip_axis_dot": hip_axis_dot,
        "hip": selected_hip,
        "wheel": selected_wheel,
        "joint_records": [*hips, wheel_joint],
    }
    leg["fk_geometry"] = _fk_geometry(cache, stage, records, point_joints)
    return leg


def main() -> None:
    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    entry = asset_root_v2() / ASSET_BUNDLE_V2.entry_file
    stage = Usd.Stage.Open(str(entry))
    if stage is None:
        raise RuntimeError(f"Unable to open AssetBundleV2: {entry}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    joint_prims = _joint_prims_by_name(stage)
    required = {
        "jIJ",
        "jIO",
        "jAB",
        "jAG",
        "jwheel_left",
        "jwheel_right",
        "jJM",
        "jOP",
        "jMK",
        "jKN",
        "jBE",
        "jGH",
        "jEC",
        "jCF",
        *(Path(path).name for path in ASSET_BUNDLE_V2.loop_joint_paths),
    }
    missing = sorted(required - joint_prims.keys())
    if missing:
        raise RuntimeError(f"Required joints are missing from AssetBundleV2: {missing}")
    records = {name: _joint_record(stage, cache, joint_prims[name]) for name in sorted(required)}
    for record in records.values():
        if record["joint_anchor_error_m"] > 1.0e-4:
            raise RuntimeError(
                f"Joint local anchors disagree: {record['joint_name']} error={record['joint_anchor_error_m']:.9g} m"
            )
        if abs(record["joint_axis_dot"]) < 0.99999:
            raise RuntimeError(
                f"Joint local axes disagree: {record['joint_name']} dot={record['joint_axis_dot']:.9g}"
            )

    output = {
        "schema_version": VIRTUAL_LEG_KINEMATICS_VERSION,
        "asset_bundle_version": ASSET_BUNDLE_V2.version,
        "asset_bundle_hash": asset_report.bundle_hash,
        "asset_entry": str(entry.resolve()),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "coordinate_definition": {
            "vector": "wheel_axis_world - hip_axis_world, rotated into USD base_link frame",
            "s_w": "+base_link Y",
            "d_w": "-base_link Z",
            "l0": "hypot(s_w, d_w)",
            "phi0": "atan2(d_w, s_w)",
            "base_x": "lateral residual only; excluded from L0 and phi0",
        },
        "legs": {
            "left": _build_leg_record(
                stage,
                cache,
                records,
                hip_names=("jIJ", "jIO"),
                wheel_name="jwheel_left",
                point_joints={
                    "I": "jIJ",
                    "J": "jJM",
                    "L": "jIO_loop_closure",
                    "M": "jMK",
                    "K": "jKN",
                    "N": "jKN_loop_closure",
                    "P": "jOP",
                    "W": "jwheel_left",
                },
            ),
            "right": _build_leg_record(
                stage,
                cache,
                records,
                hip_names=("jAB", "jAG"),
                wheel_name="jwheel_right",
                point_joints={
                    "I": "jAB",
                    "J": "jBE",
                    "L": "jAG_loop_closure",
                    "M": "jEC",
                    "K": "jCF",
                    "N": "jCF_revolute_joint",
                    "P": "jGH",
                    "W": "jwheel_right",
                },
            ),
        },
        "loop_joints": [records[Path(path).name] for path in ASSET_BUNDLE_V2.loop_joint_paths],
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    print(f"[INFO] Virtual-leg anchor audit written to: {args_cli.output.resolve()}", flush=True)


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
