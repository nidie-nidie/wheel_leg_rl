from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBPROJECT_ROOT = PROJECT_ROOT / "sim2sim" / "mujoco"
SOURCE_XML = (
    PROJECT_ROOT.parent
    / "wheel_leg_debug-main2"
    / "mujoco_control_extract"
    / "sim"
    / "models"
    / "wheel_leg_urdf4_self_mesh_all.xml"
)
SOURCE_MESH_ROOT = PROJECT_ROOT.parent / "wheel_leg_urdf4_mjcf" / "wheel_leg_urdf4" / "meshes"
USD_DATA_PATH = PROJECT_ROOT / "artifacts" / "phase1_v4" / "mujoco-usd-data.json"
MODEL_PATH = SUBPROJECT_ROOT / "models" / "wheel_leg_urdf4_v1.xml"
MESH_ROOT = MODEL_PATH.parent / "meshes"
MODEL_MANIFEST_PATH = SUBPROJECT_ROOT / "model_manifest.json"
DUMMY_AUDIT_PATH = PROJECT_ROOT / "artifacts" / "phase1_v4" / "mujoco-dummy-dynamics-audit.json"

sys.path.insert(0, str(SUBPROJECT_ROOT))
from wheelleg_mujoco.versions import (  # noqa: E402
    ACTION_ADAPTER_VERSION,
    MUJOCO_MODEL_VERSION,
    OBSERVATION_ADAPTER_VERSION,
)
from wheelleg_mujoco.model_semantics import compiled_model_semantics, stable_hash  # noqa: E402

MODEL_VERSION = MUJOCO_MODEL_VERSION
COLLISION_VERSION = "MujocoWheelOnlyCollisionV2"
DUMMY_VERSION = "MujocoDummyDynamicsV1"
PHYSICS_DT_S = 0.001
POLICY_STEPS = 20
CONTROL_DT_S = PHYSICS_DT_S * POLICY_STEPS
MUJOCO_STL_MAX_FACES = 200_000
BASE_STL_PART_FACES = 180_000
INERTIA_ROUNDING_RELATIVE_TOLERANCE = 1.0e-6
ACTIVE_LEG_JOINTS = ("jIJ", "jIO", "jAB", "jAG")
WHEEL_JOINTS = ("jwheel_left", "jwheel_right")
ACTUATOR_LIMITS = {
    "Left_front_joint_act": 18.0,
    "Left_rear_joint_act": 18.0,
    "Right_front_joint_act": 18.0,
    "Right_rear_joint_act": 18.0,
    "Left_Wheel_act": 9.0,
    "Right_Wheel_act": 9.0,
}
SOURCE_BODY_FROM_USD = {"base_link": "base"}
R_CONTROL_FROM_MUJOCO = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
VISUAL_AABB_TOLERANCE_M = 1.0e-3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _format(values) -> str:
    return " ".join(f"{0.0 if abs(float(value)) < 1.0e-15 else float(value):.17g}" for value in values)


def _parse_vector(text: str | None, length: int, *, default: tuple[float, ...]) -> np.ndarray:
    values = default if text is None else tuple(float(value) for value in text.split())
    if len(values) != length:
        raise RuntimeError(f"Expected {length} values, got {values}")
    return np.asarray(values, dtype=np.float64)


def _quaternion_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = lhs / np.linalg.norm(lhs)
    rhs = rhs / np.linalg.norm(rhs)
    return min(float(np.linalg.norm(lhs - rhs)), float(np.linalg.norm(lhs + rhs)))


def _quaternion_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise RuntimeError("Quaternion has zero length")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _pose_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quaternion_rotation(quaternion)
    matrix[:3, 3] = position
    return matrix


def _body_elements(worldbody: ET.Element) -> dict[str, ET.Element]:
    result = {}
    for body in worldbody.iter("body"):
        name = body.get("name")
        if not name or name in result:
            raise RuntimeError(f"Source XML body name is missing or duplicated: {name!r}")
        result[name] = body
    return result


def _parent_body_name(root: ET.Element, target: ET.Element) -> str | None:
    for parent in root.iter("body"):
        if target in list(parent):
            return parent.get("name")
    return None


def _mujoco_diagonal_inertia(body_name: str, source_values: list[float]) -> tuple[list[float], dict | None]:
    values = [float(value) for value in source_values]
    maximum_index = int(np.argmax(values))
    other_sum = sum(value for index, value in enumerate(values) if index != maximum_index)
    violation = values[maximum_index] - other_sum
    if violation <= 0.0:
        return values, None
    relative_violation = violation / values[maximum_index]
    if relative_violation > INERTIA_ROUNDING_RELATIVE_TOLERANCE:
        raise RuntimeError(
            f"USD inertia for {body_name} is physically invalid: relative violation={relative_violation:.17g}"
        )
    adjusted = list(values)
    adjusted[maximum_index] = math.nextafter(other_sum, 0.0)
    return adjusted, {
        "body_name": body_name,
        "reason": "USD float quantization crossed the rigid-body inertia triangle boundary",
        "source_diagonal_inertia_kg_m2": values,
        "mujoco_diagonal_inertia_kg_m2": adjusted,
        "absolute_correction_kg_m2": adjusted[maximum_index] - values[maximum_index],
        "relative_violation": relative_violation,
    }


def _set_inertial(body: ET.Element, body_name: str, record: dict, adjustments: list[dict]) -> None:
    inertia = record["inertia"]
    diagonal, adjustment = _mujoco_diagonal_inertia(body_name, inertia["diagonal_inertia_kg_m2"])
    if adjustment is not None:
        adjustments.append(adjustment)
    inertial = body.find("inertial")
    if inertial is None:
        inertial = ET.Element("inertial")
        body.insert(0, inertial)
    inertial.attrib.clear()
    inertial.set("pos", _format(inertia["center_of_mass_m"]))
    inertial.set("quat", _format(inertia["principal_axes_wxyz"]))
    inertial.set("mass", f"{float(inertia['mass_kg']):.17g}")
    inertial.set("diaginertia", _format(diagonal))


def _configure_joint(joint: ET.Element, name: str) -> None:
    joint.set("frictionloss", "0")
    if name in ACTIVE_LEG_JOINTS or name in WHEEL_JOINTS:
        joint.set("damping", "0")
        joint.set("armature", "0.05")
    else:
        joint.set("damping", "0.05")
        joint.set("armature", "0.005")


def _validate_main_body_geometry(worldbody: ET.Element, usd_data: dict) -> None:
    bodies = _body_elements(worldbody)
    for usd_name, record in usd_data["bodies"].items():
        if "dummy" in usd_name.lower() or usd_name == "base_link":
            continue
        body = bodies.get(usd_name)
        if body is None:
            raise RuntimeError(f"Source XML is missing USD body {usd_name!r}")
        expected_parent = record["parent_body_name"]
        expected_parent = SOURCE_BODY_FROM_USD.get(expected_parent, expected_parent)
        actual_parent = _parent_body_name(worldbody, body)
        if actual_parent != expected_parent:
            raise RuntimeError(f"Body parent mismatch for {usd_name}: {actual_parent} != {expected_parent}")
        actual_position = _parse_vector(body.get("pos"), 3, default=(0.0, 0.0, 0.0))
        expected_position = np.asarray(record["parent_child_transform_q0"]["position_m"])
        if not np.allclose(actual_position, expected_position, rtol=0.0, atol=2.0e-6):
            raise RuntimeError(f"Body transform mismatch for {usd_name}: {actual_position} != {expected_position}")
        actual_quaternion = _parse_vector(body.get("quat"), 4, default=(1.0, 0.0, 0.0, 0.0))
        expected_quaternion = np.asarray(record["parent_child_transform_q0"]["quaternion_wxyz"])
        if _quaternion_distance(actual_quaternion, expected_quaternion) > 2.0e-6:
            raise RuntimeError(
                f"Body quaternion mismatch for {usd_name}: {actual_quaternion} != {expected_quaternion}"
            )
        joint = body.find(f"joint[@name='{usd_name}']")
        if joint is None:
            raise RuntimeError(f"Source XML body {usd_name!r} has no same-named joint")
        actual_axis = _parse_vector(joint.get("axis"), 3, default=(0.0, 0.0, 1.0))
        expected_axis = np.asarray(record["joint"]["axis_child_body"])
        if not np.allclose(actual_axis, expected_axis, rtol=0.0, atol=2.0e-6):
            raise RuntimeError(f"Joint axis mismatch for {usd_name}: {actual_axis} != {expected_axis}")


def _split_binary_stl(source: Path, destination_root: Path) -> tuple[list[Path], dict]:
    expected_minimum_size = 84
    if source.stat().st_size < expected_minimum_size:
        raise RuntimeError(f"Binary STL is too small: {source}")
    with source.open("rb") as stream:
        header = stream.read(80)
        face_count_bytes = stream.read(4)
        if len(header) != 80 or len(face_count_bytes) != 4:
            raise RuntimeError(f"Binary STL header is incomplete: {source}")
        face_count = struct.unpack("<I", face_count_bytes)[0]
        expected_size = expected_minimum_size + 50 * face_count
        if source.stat().st_size != expected_size:
            raise RuntimeError(
                f"Binary STL size mismatch for {source}: {source.stat().st_size} != {expected_size}"
            )

        parts = []
        part_face_counts = []
        for part_index, face_offset in enumerate(range(0, face_count, BASE_STL_PART_FACES)):
            part_face_count = min(BASE_STL_PART_FACES, face_count - face_offset)
            if not 1 <= part_face_count <= MUJOCO_STL_MAX_FACES:
                raise RuntimeError(f"Invalid MuJoCo STL part size: {part_face_count}")
            destination = destination_root / f"base_link_part_{part_index:03d}.STL"
            part_header = f"WheelLeg base_link lossless part {part_index:03d}".encode("ascii").ljust(80, b"\0")
            triangle_bytes = stream.read(50 * part_face_count)
            if len(triangle_bytes) != 50 * part_face_count:
                raise RuntimeError(f"Binary STL triangle data ended early: {source}")
            with destination.open("wb") as output:
                output.write(part_header)
                output.write(struct.pack("<I", part_face_count))
                output.write(triangle_bytes)
            parts.append(destination)
            part_face_counts.append(part_face_count)
        if stream.read(1):
            raise RuntimeError(f"Binary STL contains unaccounted trailing data: {source}")

    source_aabb = _binary_stl_aabb(source)
    part_aabbs = [_binary_stl_aabb(path) for path in parts]
    part_minimum = np.min(np.asarray([record["minimum_m"] for record in part_aabbs]), axis=0)
    part_maximum = np.max(np.asarray([record["maximum_m"] for record in part_aabbs]), axis=0)
    combined_aabb = _aabb_record(part_minimum, part_maximum)
    max_abs_aabb_error = max(
        float(np.max(np.abs(np.asarray(source_aabb["minimum_m"]) - part_minimum))),
        float(np.max(np.abs(np.asarray(source_aabb["maximum_m"]) - part_maximum))),
    )
    if max_abs_aabb_error != 0.0:
        raise RuntimeError(f"Lossless STL partition changed the base local AABB: {max_abs_aabb_error}")

    return parts, {
        "source_file": source.name,
        "source_sha256": _sha256(source),
        "source_face_count": face_count,
        "part_face_limit": BASE_STL_PART_FACES,
        "part_face_counts": part_face_counts,
        "parts": [path.name for path in parts],
        "source_local_aabb": source_aabb,
        "parts_combined_local_aabb": combined_aabb,
        "max_abs_local_aabb_error_m": max_abs_aabb_error,
    }


def _aabb_record(minimum: np.ndarray, maximum: np.ndarray) -> dict:
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    return {
        "minimum_m": minimum.tolist(),
        "maximum_m": maximum.tolist(),
        "center_m": ((minimum + maximum) * 0.5).tolist(),
        "size_m": (maximum - minimum).tolist(),
    }


def _binary_stl_aabb(path: Path) -> dict:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise RuntimeError(f"Binary STL is too small: {path}")
    face_count = struct.unpack("<I", payload[80:84])[0]
    if len(payload) != 84 + 50 * face_count:
        raise RuntimeError(f"Binary STL size does not match its face count: {path}")
    records = np.frombuffer(payload, dtype=np.uint8, offset=84).reshape(face_count, 50)
    vertices = np.frombuffer(records[:, 12:48].copy().tobytes(), dtype="<f4").reshape(-1, 3)
    return _aabb_record(vertices.min(axis=0), vertices.max(axis=0))


def _binary_stl_vertices(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise RuntimeError(f"Binary STL is too small: {path}")
    face_count = struct.unpack("<I", payload[80:84])[0]
    if len(payload) != 84 + 50 * face_count:
        raise RuntimeError(f"Binary STL size does not match its face count: {path}")
    records = np.frombuffer(payload, dtype=np.uint8, offset=84).reshape(face_count, 50)
    return np.frombuffer(records[:, 12:48].copy().tobytes(), dtype="<f4").reshape(-1, 3)


def _visual_aabb_audit(root: ET.Element, usd_data: dict) -> dict:
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("Generated XML is missing asset or worldbody")
    meshes = {}
    for mesh in asset.findall("mesh"):
        name = mesh.get("name")
        filename = mesh.get("file")
        if not name or not filename:
            raise RuntimeError("Generated XML contains an incomplete mesh asset")
        scale = _parse_vector(mesh.get("scale"), 3, default=(1.0, 1.0, 1.0))
        meshes[name] = (_binary_stl_vertices(MESH_ROOT / filename) * scale, filename)

    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    geom_records = []

    def visit_body(body: ET.Element, parent_transform: np.ndarray) -> None:
        position = _parse_vector(body.get("pos"), 3, default=(0.0, 0.0, 0.0))
        quaternion = _parse_vector(body.get("quat"), 4, default=(1.0, 0.0, 0.0, 0.0))
        body_transform = parent_transform @ _pose_matrix(position, quaternion)
        nonlocal minimum, maximum
        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if mesh_name is None:
                continue
            vertices, filename = meshes[mesh_name]
            geom_position = _parse_vector(geom.get("pos"), 3, default=(0.0, 0.0, 0.0))
            geom_quaternion = _parse_vector(geom.get("quat"), 4, default=(1.0, 0.0, 0.0, 0.0))
            transform = body_transform @ _pose_matrix(geom_position, geom_quaternion)
            points = vertices @ transform[:3, :3].T + transform[:3, 3]
            geom_minimum = points.min(axis=0)
            geom_maximum = points.max(axis=0)
            minimum = np.minimum(minimum, geom_minimum)
            maximum = np.maximum(maximum, geom_maximum)
            geom_records.append(
                {
                    "body": body.get("name"),
                    "mesh": mesh_name,
                    "file": filename,
                    "root_aabb": _aabb_record(geom_minimum, geom_maximum),
                }
            )
        for child in body.findall("body"):
            visit_body(child, body_transform)

    source_base_position = np.asarray(
        usd_data["source_mujoco_xml"]["base_qpos0_position_m"],
        dtype=np.float64,
    )
    asset_root_from_mujoco_world = np.eye(4, dtype=np.float64)
    asset_root_from_mujoco_world[:3, 3] = -source_base_position
    for body in worldbody.findall("body"):
        visit_body(body, asset_root_from_mujoco_world)
    if not geom_records:
        raise RuntimeError("Generated MuJoCo XML contains no visual mesh geoms")

    mujoco_aabb = _aabb_record(minimum, maximum)
    usd_aabb = usd_data["visual_geometry"]["asset_root_aabb"]
    center_error = np.abs(np.asarray(mujoco_aabb["center_m"]) - np.asarray(usd_aabb["center_m"]))
    size_error = np.abs(np.asarray(mujoco_aabb["size_m"]) - np.asarray(usd_aabb["size_m"]))
    if np.max(center_error) > VISUAL_AABB_TOLERANCE_M or np.max(size_error) > VISUAL_AABB_TOLERANCE_M:
        raise RuntimeError(
            "USD/MuJoCo visual AABB mismatch exceeds 1 mm: "
            f"center_error={center_error.tolist()}, size_error={size_error.tolist()}"
        )
    return {
        "tolerance_m": VISUAL_AABB_TOLERANCE_M,
        "usd_asset_root_aabb": usd_aabb,
        "mujoco_q0_root_aabb": mujoco_aabb,
        "center_abs_error_m": center_error.tolist(),
        "size_abs_error_m": size_error.tolist(),
        "mujoco_mesh_geom_count": len(geom_records),
        "passed": True,
    }


def _replace_mesh_geoms(root: ET.Element, mesh_name: str, part_names: list[str]) -> None:
    replaced = 0
    for parent in root.iter():
        for child_index, child in enumerate(list(parent)):
            if child.tag != "geom" or child.get("mesh") != mesh_name:
                continue
            attributes = dict(child.attrib)
            parent.remove(child)
            for offset, part_name in enumerate(part_names):
                part_attributes = dict(attributes)
                part_attributes["mesh"] = part_name
                parent.insert(child_index + offset, ET.Element("geom", part_attributes))
            replaced += 1
    if replaced != 1:
        raise RuntimeError(f"Expected one geom using {mesh_name!r}, found {replaced}")


def _copy_meshes(root: ET.Element) -> tuple[list[Path], dict]:
    MESH_ROOT.mkdir(parents=True, exist_ok=True)
    required = []
    base_partition = None
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("Source XML has no asset element")
    for stale in MESH_ROOT.glob("base_link_part_*.STL"):
        stale.unlink()
    (MESH_ROOT / "base_link.STL").unlink(missing_ok=True)
    for mesh in list(asset.findall("mesh")):
        name = mesh.get("name")
        if name == "base_link_original":
            source = SOURCE_MESH_ROOT / "base_link.STL"
            if not source.is_file():
                raise RuntimeError(f"Required base mesh is missing: {source}")
            parts, base_partition = _split_binary_stl(source, MESH_ROOT)
            mesh_index = list(asset).index(mesh)
            asset.remove(mesh)
            part_names = []
            for offset, part in enumerate(parts):
                part_name = f"base_link_original_part_{offset:03d}"
                asset.insert(mesh_index + offset, ET.Element("mesh", name=part_name, file=part.name))
                part_names.append(part_name)
            _replace_mesh_geoms(root, name, part_names)
            required.extend(parts)
            continue
        filename = mesh.get("file")
        if not filename:
            raise RuntimeError(f"Mesh {name!r} has no file")
        source = SOURCE_MESH_ROOT / filename
        if not source.is_file() or source.stat().st_size <= 84:
            raise RuntimeError(f"Required non-placeholder mesh is missing: {source}")
        destination = MESH_ROOT / filename
        shutil.copy2(source, destination)
        required.append(destination)
    if base_partition is None:
        raise RuntimeError("Source XML has no base_link_original mesh")
    return required, base_partition


def _add_dummy_bodies(worldbody: ET.Element, usd_data: dict, adjustments: list[dict]) -> None:
    bodies = _body_elements(worldbody)
    for name in usd_data["dummy_body_names"]:
        record = usd_data["bodies"][name]
        parent_name = SOURCE_BODY_FROM_USD.get(record["parent_body_name"], record["parent_body_name"])
        parent = bodies.get(parent_name)
        if parent is None:
            raise RuntimeError(f"Dummy parent is missing from source XML: {name} -> {parent_name}")
        transform = record["parent_child_transform_q0"]
        body = ET.SubElement(
            parent,
            "body",
            name=name,
            pos=_format(transform["position_m"]),
            quat=_format(transform["quaternion_wxyz"]),
        )
        _set_inertial(body, name, record, adjustments)
        joint_record = record["joint"]
        attributes = {
            "name": name,
            "pos": _format(joint_record["child_local_position_m"]),
            "axis": _format(joint_record["axis_child_body"]),
        }
        lower = joint_record["lower_limit_rad"]
        upper = joint_record["upper_limit_rad"]
        if lower is not None and upper is not None:
            attributes.update({"limited": "true", "range": _format((lower, upper))})
        joint = ET.SubElement(body, "joint", **attributes)
        _configure_joint(joint, name)
        bodies[name] = body


def _configure_collisions(root: ET.Element) -> None:
    for geom in root.findall(".//geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    contact = root.find("contact")
    if contact is None:
        contact = ET.Element("contact")
        root.append(contact)
    for child in list(contact):
        contact.remove(child)
    pair_attributes = {
        "condim": "3",
        "friction": "1 1 0 0 0",
        "solref": "0.004 1",
        "solimp": "0.95 0.99 0.001 0.5 2",
    }
    ET.SubElement(
        contact,
        "pair",
        name="floor_left_wheel",
        geom1="floor",
        geom2="left_wheel_proxy",
        **pair_attributes,
    )
    ET.SubElement(
        contact,
        "pair",
        name="floor_right_wheel",
        geom1="floor",
        geom2="right_wheel_proxy",
        **pair_attributes,
    )


def _configure_actuators(root: ET.Element) -> None:
    actuators = root.findall("./actuator/motor")
    if {actuator.get("name") for actuator in actuators} != set(ACTUATOR_LIMITS):
        raise RuntimeError("Source XML actuator set does not match ActionV1")
    for actuator in actuators:
        limit = ACTUATOR_LIMITS[actuator.get("name")]
        actuator.set("gear", "1")
        actuator.set("ctrllimited", "true")
        actuator.set("ctrlrange", _format((-limit, limit)))


def _add_reset_keyframe(root: ET.Element, worldbody: ET.Element, usd_data: dict) -> list[str]:
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    keyframe = ET.Element("keyframe")
    root.append(keyframe)
    joint_order = [joint.get("name") for joint in worldbody.iter("joint")]
    if len(joint_order) != 26 or any(name is None for name in joint_order):
        raise RuntimeError(f"Expected 26 named hinge joints, found {joint_order}")
    reset = usd_data["contract_v4_reset"]
    positions = reset["joint_positions_rad"]
    if set(joint_order) != set(positions):
        raise RuntimeError("Generated MuJoCo hinge set does not match ContractV4 reset joint set")
    qpos = [*reset["base_position_m"], *reset["base_quaternion_wxyz"]]
    qpos.extend(float(positions[name]) for name in joint_order)
    ET.SubElement(keyframe, "key", name="contract_v4_reset", qpos=_format(qpos))
    return joint_order


def _compiled_dimensions(model: mujoco.MjModel) -> dict[str, int]:
    return {
        "nbody": model.nbody,
        "njnt": model.njnt,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "neq": model.neq,
        "npair": model.npair,
    }


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    value = mujoco.mj_id2name(model, object_type, index)
    if value is None:
        raise RuntimeError(f"Compiled MuJoCo object has no name: {object_type} index={index}")
    return value


def _audit_compiled_model(model: mujoco.MjModel, usd_data: dict, joint_order: list[str]) -> dict:
    expected_dimensions = {"nbody": 28, "njnt": 27, "nq": 33, "nv": 32, "nu": 6, "neq": 8, "npair": 2}
    dimensions = _compiled_dimensions(model)
    if dimensions != expected_dimensions:
        raise RuntimeError(f"Compiled MuJoCo dimensions mismatch: {dimensions} != {expected_dimensions}")
    total_mass = float(model.body_mass.sum())
    if abs(total_mass - float(usd_data["total_mass_kg"])) > 1.0e-6:
        raise RuntimeError(f"Compiled model mass mismatch: {total_mass} != {usd_data['total_mass_kg']}")
    if np.any(model.geom_contype != 0) or np.any(model.geom_conaffinity != 0):
        raise RuntimeError("Automatic MuJoCo geom collision masks are not fully disabled")
    pair_names = {_name(model, mujoco.mjtObj.mjOBJ_PAIR, index) for index in range(model.npair)}
    if pair_names != {"floor_left_wheel", "floor_right_wheel"}:
        raise RuntimeError(f"Unexpected explicit contact pairs: {pair_names}")
    if not np.allclose(model.pair_friction[:, :2], 1.0) or not np.allclose(model.pair_friction[:, 2:], 0.0):
        raise RuntimeError(f"Explicit pair friction mismatch: {model.pair_friction.tolist()}")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "contract_v4_reset")
    if key_id < 0:
        raise RuntimeError("ContractV4 reset keyframe is missing")

    reference = mujoco.MjData(model)
    mujoco.mj_resetData(model, reference)
    mujoco.mj_forward(model, reference)
    reset = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, reset, key_id)
    mujoco.mj_forward(model, reset)
    dummy_records = []
    for name in usd_data["dummy_body_names"]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if body_id < 0 or joint_id < 0:
            raise RuntimeError(f"Compiled dummy body/joint is missing: {name}")
        usd_inertia = usd_data["bodies"][name]["inertia"]
        if abs(float(model.body_mass[body_id]) - float(usd_inertia["mass_kg"])) > 1.0e-8:
            raise RuntimeError(f"Dummy mass mismatch for {name}")
        if not np.allclose(
            model.body_inertia[body_id],
            usd_inertia["diagonal_inertia_kg_m2"],
            rtol=1.0e-6,
            atol=1.0e-8,
        ):
            raise RuntimeError(f"Dummy inertia mismatch for {name}")
        dummy_records.append(
            {
                "name": name,
                "body_id": body_id,
                "joint_id": joint_id,
                "qpos_address": int(model.jnt_qposadr[joint_id]),
                "dof_address": int(model.jnt_dofadr[joint_id]),
                "mass_kg": float(model.body_mass[body_id]),
                "reference_world_position_m": reference.xpos[body_id].tolist(),
                "reference_world_quaternion_wxyz": reference.xquat[body_id].tolist(),
                "reset_world_position_m": reset.xpos[body_id].tolist(),
                "reset_world_quaternion_wxyz": reset.xquat[body_id].tolist(),
                "reset_joint_position_rad": float(reset.qpos[model.jnt_qposadr[joint_id]]),
                "usd": usd_data["bodies"][name],
            }
        )
    equality_constraints = []
    for index in range(model.neq):
        equality_constraints.append(
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index),
                "type": int(model.eq_type[index]),
                "solref": model.eq_solref[index].tolist(),
                "solimp": model.eq_solimp[index].tolist(),
            }
        )
    return {
        "schema_version": DUMMY_VERSION,
        "model_version": MODEL_VERSION,
        "compiled_dimensions": dimensions,
        "total_mass_kg": total_mass,
        "joint_order": joint_order,
        "keyframe_id": key_id,
        "dummy_bodies": dummy_records,
        "equality_constraints": equality_constraints,
        "solver": {
            "type": int(model.opt.solver),
            "iterations": int(model.opt.iterations),
            "line_search_iterations": int(model.opt.ls_iterations),
            "tolerance": float(model.opt.tolerance),
        },
    }


def main() -> None:
    if not SOURCE_XML.is_file() or not USD_DATA_PATH.is_file():
        raise FileNotFoundError("Source XML or generated USD dynamics JSON is missing")
    usd_data = json.loads(USD_DATA_PATH.read_text(encoding="utf-8"))
    if usd_data.get("schema_version") != "MujocoUsdDataV1":
        raise RuntimeError("Unexpected USD dynamics export schema")
    if usd_data["source_mujoco_xml"]["sha256"] != _sha256(SOURCE_XML):
        raise RuntimeError("Source MuJoCo XML hash changed after USD dynamics export")

    tree = ET.parse(SOURCE_XML)
    root = tree.getroot()
    root.set("model", "wheel_leg_urdf4_v1")
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("Source XML has no compiler element")
    compiler.set("angle", "radian")
    compiler.set("meshdir", "meshes")
    compiler.attrib.pop("texturedir", None)
    option = root.find("option")
    if option is None:
        raise RuntimeError("Source XML has no option element")
    option.set("timestep", f"{PHYSICS_DT_S:.17g}")
    option.set("gravity", "0 0 -9.81")

    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("Source XML has no asset element")
    for texture in list(asset.findall("texture")):
        if texture.get("type") == "skybox" or texture.get("file") == "desert.png":
            asset.remove(texture)
    copied_meshes, base_mesh_partition = _copy_meshes(root)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Source XML has no worldbody")
    _validate_main_body_geometry(worldbody, usd_data)
    inertia_adjustments = []
    _add_dummy_bodies(worldbody, usd_data, inertia_adjustments)
    bodies = _body_elements(worldbody)
    for usd_name, record in usd_data["bodies"].items():
        xml_name = SOURCE_BODY_FROM_USD.get(usd_name, usd_name)
        body = bodies.get(xml_name)
        if body is None:
            raise RuntimeError(f"Generated XML is missing USD body {usd_name!r}")
        _set_inertial(body, usd_name, record, inertia_adjustments)
    for joint in worldbody.iter("joint"):
        name = joint.get("name")
        if name is None:
            raise RuntimeError("Generated XML contains an unnamed hinge")
        _configure_joint(joint, name)
    _configure_collisions(root)
    _configure_actuators(root)
    joint_order = _add_reset_keyframe(root, worldbody, usd_data)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(MODEL_PATH, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    audit = _audit_compiled_model(model, usd_data, joint_order)
    visual_aabb_audit = _visual_aabb_audit(root, usd_data)
    dynamics_semantics = compiled_model_semantics(model)

    DUMMY_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUMMY_AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    mesh_manifest = [
        {"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(copied_meshes, key=lambda item: item.name)
    ]
    manifest = {
        "model_version": MODEL_VERSION,
        "dummy_dynamics_version": DUMMY_VERSION,
        "collision_policy_version": COLLISION_VERSION,
        "mujoco_version": mujoco.__version__,
        "source_xml": {"path": SOURCE_XML.name, "sha256": _sha256(SOURCE_XML)},
        "usd_data_sha256": _sha256(USD_DATA_PATH),
        "asset_bundle_version": usd_data["asset_bundle_version"],
        "asset_bundle_hash": usd_data["asset_bundle_hash"],
        "model_xml": {"path": MODEL_PATH.name, "sha256": _sha256(MODEL_PATH)},
        "meshes": mesh_manifest,
        "base_mesh_partition": base_mesh_partition,
        "visual_aabb_audit": visual_aabb_audit,
        "inertia_adjustments": inertia_adjustments,
        "dynamics_semantics": dynamics_semantics,
        "dynamics_semantics_hash": stable_hash(dynamics_semantics),
        "compiled_dimensions": audit["compiled_dimensions"],
        "total_mass_kg": audit["total_mass_kg"],
        "physics_dt_s": PHYSICS_DT_S,
        "physics_steps_per_policy_action": POLICY_STEPS,
        "control_dt_s": CONTROL_DT_S,
        "root_body": "base",
        "freejoint": "base_free",
        "reset_keyframe": "contract_v4_reset",
        "authored_q0_semantics": "model_reference_only; ContractV4 reset uses the named keyframe",
        "r_control_from_mujoco": R_CONTROL_FROM_MUJOCO,
        "com_kinematics": {
            "position_api": "data.xipos[base]",
            "velocity_api": "mj_jacBodyCom(base) @ data.qvel",
        },
        "ground": {
            "geom_name": "floor",
            "type": "plane",
            "top_z_m": 0.0,
            "owner": MODEL_VERSION,
        },
        "contact_pairs": ["floor_left_wheel", "floor_right_wheel"],
        "equality_constraints": audit["equality_constraints"],
        "solver": audit["solver"],
        "cross_engine_solver_note": (
            "MuJoCo equality/contact solver parameters are frozen for relative evaluation and are not "
            "a field-by-field equivalent of PhysX articulation solver iterations 96/4."
        ),
        "wheel_proxy": {"type": "sphere", "radius_m": 0.0625},
        "joint_order": joint_order,
        "actuator_effort_limits_nm": ACTUATOR_LIMITS,
        "adapters": {
            "observation": {
                "version": OBSERVATION_ADAPTER_VERSION,
                "implementation_file": "wheelleg_mujoco/observation.py",
                "implementation_sha256": _sha256(SUBPROJECT_ROOT / "wheelleg_mujoco" / "observation.py"),
            },
            "action": {
                "version": ACTION_ADAPTER_VERSION,
                "implementation_file": "wheelleg_mujoco/control.py",
                "implementation_sha256": _sha256(SUBPROJECT_ROOT / "wheelleg_mujoco" / "control.py"),
            },
        },
    }
    MODEL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "model": str(MODEL_PATH.resolve()),
        "manifest": str(MODEL_MANIFEST_PATH.resolve()),
        "dummy_audit": str(DUMMY_AUDIT_PATH.resolve()),
        "dimensions": audit["compiled_dimensions"],
        "total_mass_kg": audit["total_mass_kg"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
