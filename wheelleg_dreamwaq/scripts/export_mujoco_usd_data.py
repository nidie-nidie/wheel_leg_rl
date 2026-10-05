from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_XML = (
    PROJECT_ROOT.parent
    / "wheel_leg_debug-main2"
    / "mujoco_control_extract"
    / "sim"
    / "models"
    / "wheel_leg_urdf4_self_mesh_all.xml"
)

parser = argparse.ArgumentParser(description="Export AssetBundleV2 dynamics for the isolated MuJoCo model.")
parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_ROOT / "artifacts" / "phase1_v4" / "mujoco-usd-data.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.assets.paths import asset_root_v2
from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG


SCHEMA_VERSION = "MujocoUsdDataV1"
AXIS_VECTOR = {
    "X": Gf.Vec3d(1.0, 0.0, 0.0),
    "Y": Gf.Vec3d(0.0, 1.0, 0.0),
    "Z": Gf.Vec3d(0.0, 0.0, 1.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _vec3(value) -> list[float]:
    return [float(value[index]) for index in range(3)]


def _quat(value) -> list[float]:
    imaginary = value.GetImaginary()
    return [float(value.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2])]


def _matrix_column_convention(matrix: Gf.Matrix4d) -> np.ndarray:
    return np.asarray([[float(matrix[row][column]) for column in range(4)] for row in range(4)]).T


def _quat_from_rotation(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion.tolist()


def _transform_record(matrix: np.ndarray) -> dict:
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-7):
        raise RuntimeError("USD body transform contains scale or shear")
    return {
        "matrix": matrix.tolist(),
        "position_m": matrix[:3, 3].tolist(),
        "quaternion_wxyz": _quat_from_rotation(rotation),
    }


def _finite_limit(value) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _aabb_record(minimum: np.ndarray, maximum: np.ndarray) -> dict:
    return {
        "minimum_m": minimum.tolist(),
        "maximum_m": maximum.tolist(),
        "center_m": ((minimum + maximum) * 0.5).tolist(),
        "size_m": (maximum - minimum).tolist(),
    }


def _has_collision_ancestor(prim: Usd.Prim, root_path: str) -> bool:
    current = prim
    while current.IsValid() and str(current.GetPath()).startswith(root_path):
        if current.HasAPI(UsdPhysics.CollisionAPI):
            return True
        current = current.GetParent()
    return False


def _visual_geometry_record(
    stage: Usd.Stage,
    root_prim: Usd.Prim,
    root_transform: np.ndarray,
) -> dict:
    root_path = str(root_prim.GetPath())
    root_inverse = np.linalg.inv(root_transform)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    mesh_records = []
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not str(prim.GetPath()).startswith(root_path):
            continue
        if _has_collision_ancestor(prim, root_path):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points:
            continue
        points_array = np.asarray([[float(point[index]) for index in range(3)] for point in points])
        root_from_mesh = root_inverse @ _matrix_column_convention(cache.GetLocalToWorldTransform(prim))
        root_points = points_array @ root_from_mesh[:3, :3].T + root_from_mesh[:3, 3]
        root_points *= meters_per_unit
        mesh_minimum = root_points.min(axis=0)
        mesh_maximum = root_points.max(axis=0)
        minimum = np.minimum(minimum, mesh_minimum)
        maximum = np.maximum(maximum, mesh_maximum)
        mesh_records.append(
            {
                "path": str(prim.GetPath()),
                "point_count": int(points_array.shape[0]),
                "root_aabb": _aabb_record(mesh_minimum, mesh_maximum),
            }
        )
    if not mesh_records:
        raise RuntimeError("AssetBundleV2 contains no non-collision visual meshes")
    return {
        "selection": "UsdGeom.Mesh prims without a CollisionAPI ancestor",
        "mesh_count": len(mesh_records),
        "meshes": mesh_records,
        "asset_root_aabb": _aabb_record(minimum, maximum),
    }


def _body_inertia(prim: Usd.Prim) -> dict:
    mass_api = UsdPhysics.MassAPI(prim)
    mass = mass_api.GetMassAttr().Get()
    center = mass_api.GetCenterOfMassAttr().Get()
    diagonal = mass_api.GetDiagonalInertiaAttr().Get()
    axes = mass_api.GetPrincipalAxesAttr().Get()
    if mass is None or center is None or diagonal is None or axes is None:
        raise RuntimeError(f"Rigid body has incomplete mass properties: {prim.GetPath()}")
    diagonal_values = _vec3(diagonal)
    if float(mass) <= 0.0 or min(diagonal_values) <= 0.0:
        raise RuntimeError(f"Rigid body has non-positive mass or inertia: {prim.GetPath()}")
    return {
        "mass_kg": float(mass),
        "center_of_mass_m": _vec3(center),
        "diagonal_inertia_kg_m2": diagonal_values,
        "principal_axes_wxyz": _quat(axes),
    }


def _tree_joint_records(stage: Usd.Stage) -> list[dict]:
    records = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        if bool(prim.GetAttribute("physics:excludeFromArticulation").Get()):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if len(body0) != 1 or len(body1) != 1:
            raise RuntimeError(f"Tree joint must bind exactly two bodies: {prim.GetPath()}")
        revolute = UsdPhysics.RevoluteJoint(prim)
        axis_token = str(revolute.GetAxisAttr().Get())
        if axis_token not in AXIS_VECTOR:
            raise RuntimeError(f"Unsupported revolute axis {axis_token!r}: {prim.GetPath()}")
        records.append(
            {
                "joint_name": prim.GetName(),
                "joint_path": str(prim.GetPath()),
                "parent_body_path": str(body0[0]),
                "child_body_path": str(body1[0]),
                "axis_token": axis_token,
                "lower_limit_rad": _finite_limit(revolute.GetLowerLimitAttr().Get()),
                "upper_limit_rad": _finite_limit(revolute.GetUpperLimitAttr().Get()),
                "parent_local_position_m": _vec3(joint.GetLocalPos0Attr().Get()),
                "parent_local_rotation_wxyz": _quat(joint.GetLocalRot0Attr().Get()),
                "child_local_position_m": _vec3(joint.GetLocalPos1Attr().Get()),
                "child_local_rotation_wxyz": _quat(joint.GetLocalRot1Attr().Get()),
            }
        )
    return records


def _source_base_position() -> list[float]:
    root = ET.parse(SOURCE_XML).getroot()
    base = root.find("./worldbody/body[@name='base']")
    if base is None:
        raise RuntimeError("Source MuJoCo XML is missing body 'base'")
    values = [float(value) for value in base.get("pos", "").split()]
    if len(values) != 3:
        raise RuntimeError("Source MuJoCo base body must define a 3D position")
    return values


def main() -> None:
    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    entry = asset_root_v2() / ASSET_BUNDLE_V2.entry_file
    stage = Usd.Stage.Open(str(entry))
    if stage is None:
        raise RuntimeError(f"Unable to open AssetBundleV2: {entry}")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_prim = stage.GetPrimAtPath(ASSET_BUNDLE_V2.default_prim)
    base_prim = stage.GetPrimAtPath(ASSET_BUNDLE_V2.articulation_root)
    root_transform = _matrix_column_convention(cache.GetLocalToWorldTransform(root_prim))

    body_prims = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(body_prims) != ASSET_BUNDLE_V2.rigid_body_count:
        raise RuntimeError(f"Expected 27 rigid bodies, found {len(body_prims)}")
    bodies_by_path = {str(prim.GetPath()): prim for prim in body_prims}
    joints = _tree_joint_records(stage)
    if len(joints) != ASSET_BUNDLE_V2.tree_joint_count:
        raise RuntimeError(f"Expected 26 articulation joints, found {len(joints)}")

    child_joint = {}
    for joint in joints:
        child = joint["child_body_path"]
        if child in child_joint:
            raise RuntimeError(f"Rigid body has multiple parent joints: {child}")
        child_joint[child] = joint
    if set(child_joint) != set(bodies_by_path) - {str(base_prim.GetPath())}:
        missing = sorted((set(bodies_by_path) - {str(base_prim.GetPath())}) - set(child_joint))
        extra = sorted(set(child_joint) - set(bodies_by_path))
        raise RuntimeError(f"USD articulation graph is not a single body0->body1 tree: missing={missing}, extra={extra}")

    body_records = {}
    total_mass = 0.0
    dummy_mass = 0.0
    for path, prim in sorted(bodies_by_path.items()):
        root_body = np.linalg.inv(root_transform) @ _matrix_column_convention(cache.GetLocalToWorldTransform(prim))
        joint = child_joint.get(path)
        parent_path = None if joint is None else joint["parent_body_path"]
        if parent_path is None:
            parent_body = np.eye(4)
        else:
            parent_prim = bodies_by_path.get(parent_path)
            if parent_prim is None:
                raise RuntimeError(f"Joint parent is not a rigid body: {joint['joint_name']} -> {parent_path}")
            parent_body = np.linalg.inv(root_transform) @ _matrix_column_convention(
                cache.GetLocalToWorldTransform(parent_prim)
            )
        parent_child = root_body if parent_path is None else np.linalg.inv(parent_body) @ root_body
        inertia = _body_inertia(prim)
        total_mass += inertia["mass_kg"]
        if "dummy" in prim.GetName().lower():
            dummy_mass += inertia["mass_kg"]
        record = {
            "body_name": prim.GetName(),
            "body_path": path,
            "parent_body_name": None if parent_path is None else Path(parent_path).name,
            "parent_body_path": parent_path,
            "root_transform_q0": _transform_record(root_body),
            "parent_child_transform_q0": _transform_record(parent_child),
            "inertia": inertia,
        }
        if joint is not None:
            local_rotation = Gf.Quatd(
                joint["child_local_rotation_wxyz"][0],
                Gf.Vec3d(*joint["child_local_rotation_wxyz"][1:]),
            )
            axis = Gf.Rotation(local_rotation).TransformDir(AXIS_VECTOR[joint["axis_token"]])
            axis /= axis.GetLength()
            record["joint"] = {
                **joint,
                "axis_child_body": _vec3(axis),
            }
        body_records[prim.GetName()] = record

    dummy_names = sorted(name for name in body_records if "dummy" in name.lower())
    if len(dummy_names) != ASSET_BUNDLE_V2.dummy_body_count:
        raise RuntimeError(f"Expected 12 dummy bodies, found {dummy_names}")
    if abs(total_mass - ASSET_BUNDLE_V2.total_mass_kg) > 1.0e-9:
        raise RuntimeError(f"USD mass audit mismatch: {total_mass} != {ASSET_BUNDLE_V2.total_mass_kg}")
    if abs(dummy_mass - ASSET_BUNDLE_V2.dummy_mass_kg) > 1.0e-9:
        raise RuntimeError(f"USD dummy mass audit mismatch: {dummy_mass} != {ASSET_BUNDLE_V2.dummy_mass_kg}")

    reset_joint_positions = {
        str(name): float(value) for name, value in WHEELLEG_CFG.init_state.joint_pos.items()
    }
    if set(reset_joint_positions) != {joint["joint_name"] for joint in joints}:
        missing = sorted({joint["joint_name"] for joint in joints} - set(reset_joint_positions))
        extra = sorted(set(reset_joint_positions) - {joint["joint_name"] for joint in joints})
        raise RuntimeError(f"ContractV4 reset joint set mismatch: missing={missing}, extra={extra}")
    source_base = _source_base_position()
    reset_base_position = [source_base[0], source_base[1], float(WHEELLEG_CFG.init_state.pos[2])]

    output = {
        "schema_version": SCHEMA_VERSION,
        "asset_bundle_version": ASSET_BUNDLE_V2.version,
        "asset_bundle_hash": asset_report.bundle_hash,
        "asset_entry_file": ASSET_BUNDLE_V2.entry_file,
        "source_mujoco_xml": {
            "name": SOURCE_XML.name,
            "sha256": _sha256(SOURCE_XML),
            "base_qpos0_position_m": source_base,
        },
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "visual_geometry": _visual_geometry_record(stage, root_prim, root_transform),
        "root_body_name": base_prim.GetName(),
        "root_body_path": str(base_prim.GetPath()),
        "total_mass_kg": total_mass,
        "dummy_mass_kg": dummy_mass,
        "dummy_body_names": dummy_names,
        "tree_joint_names": [joint["joint_name"] for joint in joints],
        "bodies": body_records,
        "contract_v4_reset": {
            "base_position_m": reset_base_position,
            "base_quaternion_wxyz": [float(value) for value in WHEELLEG_CFG.init_state.rot],
            "joint_positions_rad": reset_joint_positions,
            "joint_velocities_rad_s": {name: 0.0 for name in reset_joint_positions},
        },
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args_cli.output.resolve()),
        "body_count": len(body_records),
        "joint_count": len(joints),
        "dummy_count": len(dummy_names),
        "total_mass_kg": total_mass,
        "dummy_mass_kg": dummy_mass,
    }, indent=2), flush=True)


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
