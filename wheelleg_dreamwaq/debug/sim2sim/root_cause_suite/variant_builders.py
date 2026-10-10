from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET

from .contracts import canonical_json_bytes, sha256_file, stable_hash


ROBOT_HINGES = (
    "jIO", "jOP", "jwheel_left", "jIO_dummy_child_link1", "jIO_dummy_child_link2",
    "jAG", "jGH", "jwheel_right", "jAG_dummy_child_link1", "jAG_dummy_child_link2",
    "jIJ", "jJM", "jMK", "jKN", "jKN_dummy_child_link1", "jKN_dummy_child_link2",
    "jMK_dummy_child1", "jMK_dummy_child2", "jAB", "jBE", "jEC", "jCF",
    "jCF_dummy_child_link1", "jCF_dummy_child_link2", "jEC_dummy_child_link1",
    "jEC_dummy_child_link2",
)
CONNECT_NAMES = (
    "connect_cf_gh_1", "connect_cf_gh_2", "connect_kn_op_1", "connect_kn_op_2",
    "connect_ec_ag_1", "connect_ec_ag_2", "connect_mk_io_1", "connect_mk_io_2",
)
OPERATION_ORDER = ("no_ground", "gravity_off", "closure_off", "drive_off", "fixed_base")


@dataclass(frozen=True)
class VariantBuildResult:
    model_path: Path
    manifest_path: Path
    model_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class IsaacVariantSpec:
    ground_enabled: bool = True
    gravity_enabled: bool = True
    closure_enabled: bool = True
    drive_enabled: bool = True
    fixed_base: bool = False
    coupon: str | None = None
    friction: float | None = None

    def __post_init__(self) -> None:
        if self.coupon not in {None, "sphere"}:
            raise ValueError("Core V1 permits only the common sphere coupon")
        if self.friction is not None and self.coupon != "sphere":
            raise ValueError("Friction override is only valid for the common sphere coupon")
        if self.friction is not None and self.friction < 0.0:
            raise ValueError("Friction cannot be negative")

    def changed_factors(self) -> tuple[str, ...]:
        factors = []
        if not self.closure_enabled:
            factors.append("closure")
        if not self.drive_enabled:
            factors.append("drive")
        if self.fixed_base:
            factors.append("fixed_base")
        if self.friction is not None:
            factors.append("friction")
        if not self.gravity_enabled:
            factors.append("gravity")
        if not self.ground_enabled:
            factors.append("ground")
        if self.coupon is not None:
            factors.append("sphere_coupon")
        return tuple(sorted(factors))


def _require_single(root: ET.Element, path: str) -> ET.Element:
    items = root.findall(path)
    if len(items) != 1:
        raise ValueError(f"Expected one XML node at {path}, found {len(items)}")
    return items[0]


def _canonical_xml_semantics(root: ET.Element) -> dict[str, Any]:
    option = _require_single(root, "option")
    compiler = _require_single(root, "compiler")
    floor = root.find("worldbody/geom[@name='floor']")
    base = root.find("worldbody/body[@name='base']")
    if base is None:
        base = _require_single(root, "worldbody/body[@name='coupon']")
    joints = {
        node.get("name"): {
            "damping": node.get("damping"),
            "armature": node.get("armature"),
            "axis": node.get("axis"),
        }
        for node in root.findall("worldbody//joint")
    }
    connects = {
        node.get("name"): {
            "active": node.get("active", "true"),
            "site1": node.get("site1"),
            "site2": node.get("site2"),
            "solref": node.get("solref"),
            "solimp": node.get("solimp"),
        }
        for node in root.findall("equality/connect")
    }
    motors = {
        node.get("name"): dict(sorted(node.attrib.items()))
        for node in root.findall("actuator/motor")
    }
    pairs = {
        node.get("name"): dict(sorted(node.attrib.items()))
        for node in root.findall("contact/pair")
    }
    inertials = {
        body.get("name"): dict(sorted(inertial.attrib.items()))
        for body in root.findall(".//body")
        for inertial in body.findall("inertial")
    }
    return {
        "compiler_meshdir": compiler.get("meshdir"),
        "gravity": option.get("gravity"),
        "timestep": option.get("timestep"),
        "floor": None if floor is None else dict(sorted(floor.attrib.items())),
        "base": {
            "pos": base.get("pos"),
            "quat": base.get("quat", "1 0 0 0"),
            "freejoint": base.find("freejoint") is not None,
        },
        "joints": joints,
        "connects": connects,
        "motors": motors,
        "pairs": pairs,
        "inertials": inertials,
    }


def _write_tree(tree: ET.ElementTree, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True, short_empty_elements=True)


def _write_manifest(destination: Path, payload: Mapping[str, Any]) -> None:
    destination.write_bytes(canonical_json_bytes(dict(payload)) + b"\n")


def _normalize_operations(operations: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(operations)
    if len(requested) != len(set(requested)):
        raise ValueError("Variant operations must be unique")
    unknown = set(requested) - set(OPERATION_ORDER)
    if unknown:
        raise ValueError(f"Unknown variant operations: {sorted(unknown)}")
    if requested != tuple(item for item in OPERATION_ORDER if item in requested):
        raise ValueError("Variant operations must use the frozen order")
    return requested


def build_robot_variant(
    source_path: Path,
    output_path: Path,
    *,
    operations: Iterable[str],
) -> VariantBuildResult:
    source = Path(source_path).resolve(strict=True)
    output = Path(output_path).resolve()
    if output.exists() or output.parent.exists() and any(output.parent.iterdir()):
        if output.exists():
            raise FileExistsError(output)
    selected = _normalize_operations(operations)
    source_hash_before = sha256_file(source)
    tree = ET.parse(source)
    root = tree.getroot()
    before = _canonical_xml_semantics(root)
    compiler = _require_single(root, "compiler")
    meshdir = compiler.get("meshdir")
    if not meshdir:
        raise ValueError("Formal MJCF compiler lacks meshdir")
    mesh_path = Path(meshdir)
    if not mesh_path.is_absolute():
        mesh_path = (source.parent / mesh_path).resolve(strict=True)
    compiler.set("meshdir", mesh_path.as_posix())
    changes: list[dict[str, Any]] = [
        {"path": "/compiler/meshdir", "old": meshdir, "new": mesh_path.as_posix(), "factor": "artifact_path"}
    ]

    if "no_ground" in selected:
        worldbody = _require_single(root, "worldbody")
        floor = _require_single(root, "worldbody/geom[@name='floor']")
        worldbody.remove(floor)
        contact = _require_single(root, "contact")
        removed = [node.get("name") for node in list(contact)]
        for node in list(contact):
            contact.remove(node)
        changes.append({"path": "/ground/enabled", "old": True, "new": False, "factor": "ground"})
        changes.append({"path": "/contact/pairs", "old": removed, "new": [], "factor": "ground"})
    if "gravity_off" in selected:
        option = _require_single(root, "option")
        old = option.get("gravity")
        option.set("gravity", "0 0 0")
        changes.append({"path": "/option/gravity", "old": old, "new": "0 0 0", "factor": "gravity"})
    if "closure_off" in selected:
        connects = root.findall("equality/connect")
        if tuple(node.get("name") for node in connects) != CONNECT_NAMES:
            raise ValueError("Formal closure topology does not match the frozen eight constraints")
        for node in connects:
            old = node.get("active", "true")
            node.set("active", "false")
            changes.append({
                "path": f"/closure/constraints/{node.get('name')}/enabled",
                "old": old,
                "new": "false",
                "factor": "closure_mode",
            })
    if "drive_off" in selected:
        joints = root.findall("worldbody//joint")
        if tuple(node.get("name") for node in joints) != ROBOT_HINGES:
            raise ValueError("Formal robot hinge set does not match the frozen 26 names")
        for node in joints:
            old = node.get("damping")
            node.set("damping", "0")
            changes.append({
                "path": f"/joints/{node.get('name')}/damping",
                "old": old,
                "new": "0",
                "factor": "drive_mode",
            })
    if "fixed_base" in selected:
        base = _require_single(root, "worldbody/body[@name='base']")
        freejoint = _require_single(root, "worldbody/body[@name='base']/freejoint")
        key = _require_single(root, "keyframe/key[@name='contract_v4_reset']")
        qpos = [float(value) for value in key.get("qpos", "").split()]
        if len(qpos) != 33:
            raise ValueError("Formal reset keyframe must have 33 qpos values")
        base.set("pos", " ".join(format(value, ".17g") for value in qpos[:3]))
        base.set("quat", " ".join(format(value, ".17g") for value in qpos[3:7]))
        base.remove(freejoint)
        key.set("qpos", " ".join(format(value, ".17g") for value in qpos[7:]))
        changes.append({"path": "/base/freejoint", "old": True, "new": False, "factor": "fixed_base"})
        changes.append({"path": "/keyframe/qpos_length", "old": 33, "new": 26, "factor": "fixed_base"})

    after = _canonical_xml_semantics(root)
    _write_tree(tree, output)
    if sha256_file(source) != source_hash_before:
        raise RuntimeError("Formal MJCF changed while building a variant")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    payload = {
        "schema_version": "RootCauseMujocoVariantV1",
        "kind": "robot",
        "source_path": str(source),
        "source_sha256": source_hash_before,
        "model_path": str(output),
        "model_sha256": sha256_file(output),
        "operations": list(selected),
        "changes": changes,
        "before_semantics_hash": stable_hash(before),
        "after_semantics": after,
        "after_semantics_hash": stable_hash(after),
    }
    _write_manifest(manifest_path, payload)
    validate_robot_variant(output, payload)
    return VariantBuildResult(output, manifest_path, payload["model_sha256"], sha256_file(manifest_path))


def build_common_sphere_mjcf(
    output_path: Path,
    *,
    radius_m: float,
    mass_kg: float,
    friction: tuple[float, float, float],
) -> VariantBuildResult:
    if radius_m <= 0.0 or mass_kg <= 0.0 or any(value < 0.0 for value in friction):
        raise ValueError("Sphere parameters must be positive/non-negative")
    output = Path(output_path).resolve()
    inertia = 0.4 * mass_kg * radius_m * radius_m
    root = ET.Element("mujoco", {"model": "root_cause_common_sphere"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", {"timestep": "0.001", "gravity": "0 0 -9.81"})
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", {
        "name": "floor", "type": "plane", "pos": "0 0 0", "size": "0 0 0.25",
        "contype": "0", "conaffinity": "0",
    })
    body = ET.SubElement(world, "body", {"name": "coupon", "pos": "0 0 0.25"})
    ET.SubElement(body, "freejoint", {"name": "coupon_free"})
    ET.SubElement(body, "inertial", {
        "pos": "0 0 0", "quat": "1 0 0 0", "mass": format(mass_kg, ".17g"),
        "diaginertia": " ".join([format(inertia, ".17g")] * 3),
    })
    ET.SubElement(body, "geom", {
        "name": "coupon_geom", "type": "sphere", "size": format(radius_m, ".17g"),
        "contype": "0", "conaffinity": "0",
    })
    contact = ET.SubElement(root, "contact")
    friction5 = (friction[0], friction[0], friction[1], friction[2], friction[2])
    ET.SubElement(contact, "pair", {
        "name": "floor_coupon", "geom1": "floor", "geom2": "coupon_geom", "condim": "3",
        "friction": " ".join(format(value, ".17g") for value in friction5),
        "solref": "0.004 1", "solimp": "0.95 0.99 0.001 0.5 2",
    })
    tree = ET.ElementTree(root)
    _write_tree(tree, output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    semantics = _canonical_xml_semantics(root)
    payload = {
        "schema_version": "RootCauseMujocoVariantV1",
        "kind": "sphere_coupon",
        "source_path": None,
        "source_sha256": None,
        "model_path": str(output),
        "model_sha256": sha256_file(output),
        "operations": ["sphere_coupon"],
        "parameters": {"radius_m": radius_m, "mass_kg": mass_kg, "friction": list(friction)},
        "changes": [],
        "after_semantics": semantics,
        "after_semantics_hash": stable_hash(semantics),
    }
    _write_manifest(manifest_path, payload)
    validate_robot_variant(output, payload)
    return VariantBuildResult(output, manifest_path, payload["model_sha256"], sha256_file(manifest_path))


def validate_robot_variant(model_path: Path, manifest: Mapping[str, Any]) -> None:
    model = Path(model_path).resolve(strict=True)
    if manifest.get("schema_version") != "RootCauseMujocoVariantV1":
        raise ValueError("Variant manifest schema mismatch")
    if manifest.get("model_sha256") != sha256_file(model):
        raise ValueError("Variant model hash does not match manifest")
    root = ET.parse(model).getroot()
    semantics = _canonical_xml_semantics(root)
    if stable_hash(semantics) != manifest.get("after_semantics_hash"):
        raise ValueError("Variant semantic hash does not match manifest")
    if semantics != manifest.get("after_semantics"):
        raise ValueError("Variant semantics do not match manifest")
    if manifest.get("kind") == "robot":
        operations = _normalize_operations(manifest.get("operations", []))
        joints = semantics["joints"]
        if tuple(joints) != ROBOT_HINGES:
            raise ValueError("Robot variant hinge set mismatch")
        if "drive_off" in operations and any(float(joints[name]["damping"]) != 0.0 for name in ROBOT_HINGES):
            raise ValueError("Drive-off variant has non-zero damping")
        if "closure_off" in operations:
            if tuple(semantics["connects"]) != CONNECT_NAMES:
                raise ValueError("Closure-off topology changed")
            if any(item["active"] != "false" for item in semantics["connects"].values()):
                raise ValueError("Closure-off variant has active constraints")
        source = Path(str(manifest["source_path"]))
        if sha256_file(source) != manifest.get("source_sha256"):
            raise ValueError("Formal source hash drifted")
