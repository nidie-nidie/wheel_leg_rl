from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


MODELS = Path(__file__).resolve().parents[1] / "models"
SPHERE_MODEL = MODELS / "wheel_leg_urdf4_v1.xml"
MESH_MODEL = MODELS / "wheel_leg_urdf4_contact_test_wheel_mesh_vs_sphere.xml"
WHEEL_MESHES = {
    "left_wheel_proxy": "jwheel_left",
    "right_wheel_proxy": "jwheel_right",
}


def _proxy(root: ET.Element, name: str) -> ET.Element:
    geom = root.find(f".//geom[@name='{name}']")
    assert geom is not None
    return geom


def _semantic_tree(element: ET.Element) -> tuple:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_semantic_tree(child) for child in element),
    )


def _normalized_candidate(root: ET.Element) -> tuple:
    normalized = copy.deepcopy(root)
    normalized.set("model", "wheel_leg_urdf4_v1")
    for name in WHEEL_MESHES:
        geom = _proxy(normalized, name)
        geom.set("type", "sphere")
        geom.set("size", "0.0625")
        geom.attrib.pop("mesh", None)
    return _semantic_tree(normalized)


def test_mesh_contact_candidate_changes_only_wheel_collision_geometry() -> None:
    assert MESH_MODEL.is_file()
    sphere_root = ET.parse(SPHERE_MODEL).getroot()
    mesh_root = ET.parse(MESH_MODEL).getroot()

    assert mesh_root.get("model") == "wheel_leg_urdf4_contact_test_wheel_mesh_vs_sphere"
    for name, mesh_name in WHEEL_MESHES.items():
        sphere_geom = _proxy(sphere_root, name)
        mesh_geom = _proxy(mesh_root, name)
        assert sphere_geom.get("type") == "sphere"
        assert sphere_geom.get("size") == "0.0625"
        assert sphere_geom.get("mesh") is None
        assert mesh_geom.get("type") == "mesh"
        assert mesh_geom.get("mesh") == mesh_name
        assert mesh_geom.get("size") is None

    assert _normalized_candidate(mesh_root) == _semantic_tree(sphere_root)


def test_mesh_contact_candidate_compiles_with_the_same_model_dimensions() -> None:
    sphere = mujoco.MjModel.from_xml_path(str(SPHERE_MODEL))
    mesh = mujoco.MjModel.from_xml_path(str(MESH_MODEL))
    assert (mesh.nbody, mesh.njnt, mesh.nq, mesh.nv, mesh.nu, mesh.neq, mesh.npair) == (
        sphere.nbody,
        sphere.njnt,
        sphere.nq,
        sphere.nv,
        sphere.nu,
        sphere.neq,
        sphere.npair,
    )
    for name in WHEEL_MESHES:
        sphere_id = mujoco.mj_name2id(sphere, mujoco.mjtObj.mjOBJ_GEOM, name)
        mesh_id = mujoco.mj_name2id(mesh, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert sphere.geom_type[sphere_id] == mujoco.mjtGeom.mjGEOM_SPHERE
        assert mesh.geom_type[mesh_id] == mujoco.mjtGeom.mjGEOM_MESH
