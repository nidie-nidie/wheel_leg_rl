from __future__ import annotations

import shutil
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[5]
SOURCE_URDF_PATH = PROJECT_ROOT / "wheel_leg_urdf4" / "urdf" / "wheel_leg_urdf4.urdf"
GENERATED_ASSET_DIR = Path("/tmp/serial_leg_rl_isaac_assets")
SIM_MESH_DIR = GENERATED_ASSET_DIR / "meshes"
SIM_URDF_PATH = GENERATED_ASSET_DIR / "wheel_leg_urdf4_sim.urdf"


def prepare_sim_urdf() -> Path:
    GENERATED_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    SIM_MESH_DIR.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(SOURCE_URDF_PATH)
    root = tree.getroot()
    package_prefix = "package://wheel_leg_urdf4/"
    source_package_dir = PROJECT_ROOT / "wheel_leg_urdf4"

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename is None or not filename.startswith(package_prefix):
            continue
        relative_path = filename.removeprefix(package_prefix)
        source_mesh_path = source_package_dir / relative_path
        sim_mesh_path = SIM_MESH_DIR / (source_mesh_path.stem + source_mesh_path.suffix.lower())
        if not sim_mesh_path.exists() or source_mesh_path.stat().st_mtime_ns > sim_mesh_path.stat().st_mtime_ns:
            shutil.copy2(source_mesh_path, sim_mesh_path)
        mesh.set("filename", sim_mesh_path.as_posix())

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        if "dummy" not in link_name:
            continue
        for child in list(link):
            if child.tag in {"visual", "collision"}:
                link.remove(child)
        inertial = link.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(link, "inertial")
        origin = inertial.find("origin")
        if origin is None:
            origin = ET.SubElement(inertial, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")
        mass = inertial.find("mass")
        if mass is None:
            mass = ET.SubElement(inertial, "mass")
        mass.set("value", "0.001")
        inertia = inertial.find("inertia")
        if inertia is None:
            inertia = ET.SubElement(inertial, "inertia")
        inertia.set("ixx", "1e-6")
        inertia.set("ixy", "0")
        inertia.set("ixz", "0")
        inertia.set("iyy", "1e-6")
        inertia.set("iyz", "0")
        inertia.set("izz", "1e-6")

    tree.write(SIM_URDF_PATH, encoding="utf-8", xml_declaration=True)
    return SIM_URDF_PATH
