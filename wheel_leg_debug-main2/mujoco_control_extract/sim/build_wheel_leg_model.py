from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_XML = (
    ROOT.parent
    / "wheel_leg_urdf4"
    / "wheel_leg_urdf4"
    / "mujoco_export_obj"
    / "wheel_leg_urdf4_export_obj_closed_validated.xml"
)
SOURCE_MESH_DIR = SOURCE_XML.parent / "meshes"
ENV_XML = (
    ROOT.parent
    / "wheel_leg_urdf4"
    / "wheel_leg_urdf4"
    / "MJCF"
    / "env.xml"
)
ENV_DIR = ENV_XML.parent
OUTPUT_MODEL = ROOT / "sim" / "models" / "wheel_leg_urdf4_closed.xml"
OUTPUT_ASSET_DIR = ROOT / "sim" / "models" / "wheel_leg_urdf4_assets"


BASE_INERTIAL = {
    "pos": "0.0417103203611099 -1.27388449249164 0.0104466638996164",
    "mass": "2.95900767288669",
    "fullinertia": (
        "0.00564821484497837 "
        "0.00505242789744662 "
        "0.00757330965851875 "
        "-8.67327771195511E-07 "
        "4.43710335940438E-08 "
        "4.03121947642724E-06"
    ),
}

# Shift the floating base in world XY so the exported mechanism is centered
# around the MuJoCo world origin, and lift Z so the wheel mesh just touches
# the floor at the XML default pose.
BASE_WORLD_POS = "-0.0417103203611099 1.27388449249164 0.620724319674778"


ALIAS_SITES = {
    "jIO": [("IO-L", "-0.0963553 0.00167196 0.01094037")],
    "jMK": [("MK-L", "-0.0951965 -0.0008 0.0645185")],
    "jOP": [("OP-N", "-0.0373868 -0.01325 0.05169847")],
    "jKN": [("KN-N", "-0.117216 -0.001 0.0135814")],
    "jAG": [("AG-D", "-0.0963556 -0.00197196 0.01094037")],
    "jEC": [("EC-D", "-0.0951965 0 0.0645185")],
    "jGH": [("GH-F", "-0.0373871 0.01325 0.05169847")],
    "jCF": [("CF-F", "-0.117216 0.001 0.0135814")],
}


ROBOT_MATERIALS = {
    "base_mat": {"rgba": "0.2 0.2 0.25 1", "specular": "0.5", "shininess": "0.5"},
    "arm_blue": {"rgba": "0.3 0.5 0.8 1", "specular": "0.8", "shininess": "0.8"},
    "arm_red": {"rgba": "0.8 0.3 0.3 1", "specular": "0.8", "shininess": "0.8"},
    "arm_green": {"rgba": "0.3 0.8 0.4 1", "specular": "0.8", "shininess": "0.8"},
    "joint_mat": {"rgba": "0.7 0.7 0.75 1", "specular": "1.0", "shininess": "1.0"},
    "wheel_mat": {"rgba": "0.15 0.15 0.2 1", "specular": "0.3", "shininess": "0.2"},
    "white_mat": {"rgba": "0.95 0.95 0.97 1", "specular": "0.9", "shininess": "0.7"},
}


BODY_MATERIALS = {
    "base": "base_mat",
    "jAG": "arm_blue",
    "jGH": "joint_mat",
    "jwheel_right": "arm_blue",
    "jAB": "joint_mat",
    "jBE": "arm_blue",
    "jEC": "joint_mat",
    "jCF": "arm_blue",
    "jIJ": "joint_mat",
    "jJM": "arm_red",
    "jMK": "joint_mat",
    "jKN": "arm_red",
    "jIO": "arm_red",
    "jOP": "joint_mat",
    "jwheel_left": "arm_red",
}


def ensure_asset_dir() -> None:
    OUTPUT_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for entry in OUTPUT_ASSET_DIR.iterdir():
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
        elif entry.is_dir():
            shutil.rmtree(entry)

    for mesh in sorted(SOURCE_MESH_DIR.iterdir()):
        target = OUTPUT_ASSET_DIR / mesh.name
        os.symlink(mesh, target)

    skybox = ENV_DIR / "desert.png"
    if skybox.exists():
        os.symlink(skybox, OUTPUT_ASSET_DIR / skybox.name)


def add_child(parent: ET.Element, tag: str, **attrs: str) -> ET.Element:
    elem = ET.Element(tag)
    for key, value in attrs.items():
        elem.set(key, value)
    parent.append(elem)
    return elem


def clone_element(elem: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(elem, encoding="unicode"))


def parse_vec3(text: str | None, default: str = "0 0 0") -> list[float]:
    raw = text if text is not None else default
    return [float(value) for value in raw.split()]


def format_vec3(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def load_env_root() -> ET.Element:
    env_text = ENV_XML.read_text(encoding="utf-8")
    # The source env.xml has one missing space between attributes on the floor geom.
    env_text = env_text.replace('material="plane"condim="3"', 'material="plane" condim="3"')
    return ET.fromstring(env_text)


def replace_visual_with_env(root: ET.Element, env_root: ET.Element) -> None:
    env_visual = env_root.find("visual")
    if env_visual is None:
        return

    visual = root.find("visual")
    if visual is not None:
        root.remove(visual)

    insert_at = 0
    compiler = root.find("compiler")
    option = root.find("option")
    if option is not None:
        insert_at = list(root).index(option) + 1
    elif compiler is not None:
        insert_at = list(root).index(compiler) + 1

    root.insert(insert_at, clone_element(env_visual))


def merge_env_assets(root: ET.Element, env_root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.append(asset)

    for child in list(asset):
        if child.tag == "texture" and (
            child.get("type") == "skybox" or child.get("name") == "plane"
        ):
            asset.remove(child)
        elif child.tag == "material" and child.get("name") in {"plane", "box"}:
            asset.remove(child)

    env_asset = env_root.find("asset")
    if env_asset is None:
        return

    for child in env_asset:
        if child.tag not in {"texture", "material"}:
            continue
        cloned = clone_element(child)
        if cloned.tag == "texture" and cloned.get("file") == "desert.png":
            cloned.set("file", "desert.png")
        asset.append(cloned)


def merge_robot_materials(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.append(asset)

    for child in list(asset):
        if child.tag == "material" and child.get("name") in ROBOT_MATERIALS:
            asset.remove(child)

    for name, attrs in ROBOT_MATERIALS.items():
        material = ET.Element("material")
        material.set("name", name)
        for key, value in attrs.items():
            material.set(key, value)
        asset.append(material)


def apply_robot_materials(root: ET.Element) -> None:
    for body_name, material_name in BODY_MATERIALS.items():
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            continue
        for geom in body.findall("geom"):
            if geom.get("type") != "mesh":
                continue
            geom.set("material", material_name)
            if "rgba" in geom.attrib:
                del geom.attrib["rgba"]


def add_env_world(worldbody: ET.Element, env_root: ET.Element) -> None:
    env_worldbody = env_root.find("worldbody")
    if env_worldbody is None:
        return

    for child in env_worldbody:
        if child.tag == "geom" and child.get("name") == "floor":
            worldbody.append(
                ET.Comment("Comment out the floor geom below if you want the model to have no ground contact.")
            )
            worldbody.append(clone_element(child))
        elif child.tag == "light":
            worldbody.append(clone_element(child))


def collapse_dummy_bodies(parent: ET.Element) -> None:
    for child in list(parent):
        if child.tag != "body":
            continue

        collapse_dummy_bodies(child)

        body_name = child.get("name", "")
        if "dummy" not in body_name:
            continue

        body_pos = parse_vec3(child.get("pos"))
        for site in child.findall("site"):
            site_pos = parse_vec3(site.get("pos"))
            merged_pos = [body_pos[i] + site_pos[i] for i in range(3)]
            site.set("pos", format_vec3(merged_pos))
            child.remove(site)
            parent.append(site)

        parent.remove(child)


def build_model() -> None:
    tree = ET.parse(SOURCE_XML)
    root = tree.getroot()
    env_root = load_env_root()
    root.set("model", "wheel_leg_urdf4_closed")

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("meshdir", "wheel_leg_urdf4_assets")
    compiler.set("texturedir", "wheel_leg_urdf4_assets")

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(list(root).index(compiler) + 1, option)
    option.set("timestep", "0.001")
    option.set("gravity", "0 0 -9.81")

    replace_visual_with_env(root, env_root)

    default = root.find("default")
    if default is None:
        default = ET.Element("default")
        root.insert(list(root).index(option) + 1, default)
    default.clear()
    add_child(default, "joint", damping="0.02", armature="0.01")
    add_child(default, "geom", contype="1", conaffinity="1", friction="1.2 0.02 0.001")
    add_child(default, "motor", ctrllimited="true", ctrlrange="-80 80")

    merge_env_assets(root, env_root)
    merge_robot_materials(root)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Missing worldbody in source model.")

    existing_children = list(worldbody)
    for child in existing_children:
        worldbody.remove(child)

    add_env_world(worldbody, env_root)

    worldbody.append(
        ET.Comment(
            "Adjust this base pos to move the whole robot in MuJoCo world coordinates."
        )
    )
    worldbody.append(
        ET.Comment(
            "Current value recenters the exported robot in XY and lifts the whole robot 0.5 m above the touch-down height from the XML default pose."
        )
    )
    base = add_child(worldbody, "body", name="base", pos=BASE_WORLD_POS)
    base.append(
        ET.Comment(
            "Keep base_free for the mujoco_bridge controller. In plain MuJoCo viewer, removing it fixes the robot in space."
        )
    )
    add_child(base, "freejoint", name="base_free")
    add_child(base, "inertial", **BASE_INERTIAL)

    for child in existing_children:
        base.append(child)

    collapse_dummy_bodies(base)

    for body_name, sites in ALIAS_SITES.items():
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise RuntimeError(f"Missing body '{body_name}' while adding alias sites.")
        for site_name, pos in sites:
            add_child(body, "site", name=site_name, pos=pos, size="0.0015", rgba="0.9 0.9 0.2 1")

    apply_robot_materials(root)

    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    actuator = ET.Element("actuator")
    root.append(actuator)
    add_child(actuator, "motor", name="Left_front_joint_act", joint="jIJ")
    add_child(actuator, "motor", name="Left_rear_joint_act", joint="jIO")
    add_child(actuator, "motor", name="Right_rear_joint_act", joint="jAG")
    add_child(actuator, "motor", name="Right_front_joint_act", joint="jAB")
    add_child(actuator, "motor", name="Left_Wheel_act", joint="jwheel_left")
    add_child(actuator, "motor", name="Right_Wheel_act", joint="jwheel_right")

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    tree.write(OUTPUT_MODEL, encoding="utf-8", xml_declaration=True)


def main() -> None:
    ensure_asset_dir()
    build_model()
    print(OUTPUT_MODEL)


if __name__ == "__main__":
    main()
