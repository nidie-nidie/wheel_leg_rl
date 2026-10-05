#!/usr/bin/env python3
r"""Cross-check the offline XML kinematics against the MuJoCo 3.3 runtime.

Run this script with the optional analysis environment created for this report:

    .analysis-venv\Scripts\python tools\validate_offset_kinematics_mujoco.py

It writes a compact JSON report and does not advance dynamics or touch any
controller source file.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "sim" / "models" / "wheel_leg_urdf4_self_mesh_all.xml"
DERIVATION_JSON = ROOT / "output" / "offset_kinematics_results.json"
OUTPUT_JSON = ROOT / "output" / "offset_kinematics_mujoco_validation.json"

ACTIVE = ("jIJ", "jIO")
PASSIVE = ("jJM", "jMK", "jKN", "jOP")
CONNECTS = (
    ("site_mk_io_a1", "site_mk_io_b1"),
    ("site_mk_io_a2", "site_mk_io_b2"),
    ("site_kn_op_a1", "site_kn_op_b1"),
    ("site_kn_op_a2", "site_kn_op_b2"),
)


def object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise KeyError(name)
    return value


def set_hinge(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[joint_id]] = value


def load_geometry_only_model(xml_path: Path) -> mujoco.MjModel:
    """Load the same kinematic XML while ignoring unavailable visual meshes.

    The checked-in asset files are zero-byte placeholders for WSL symlinks, so
    a Windows MuJoCo runtime cannot load them.  Removing only visual mesh geoms
    leaves every body pose, joint, site, equality and proxy geom unchanged.
    """
    root = ET.parse(xml_path).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("type") == "mesh":
                parent.remove(child)
    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            if child.tag in {"mesh", "texture"}:
                asset.remove(child)
            elif child.tag == "material" and "texture" in child.attrib:
                del child.attrib["texture"]
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def main() -> None:
    source = json.loads(DERIVATION_JSON.read_text(encoding="utf-8"))
    model = load_geometry_only_model(XML)
    data = mujoco.MjData(model)
    base_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    hip_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "jIO")
    wheel_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "jwheel_left")

    rows = []
    selected = {0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.42}
    for sample in source["samples"]:
        target = round(float(sample["target_l0_m"]), 2)
        if target not in selected:
            continue
        data.qpos[:] = model.qpos0
        for name, value in zip(ACTIVE, (sample["q_front_rad"], sample["q_rear_rad"])):
            set_hinge(model, data, name, float(value))
        for name, value in zip(PASSIVE, sample["passive_rad"]):
            set_hinge(model, data, name, float(value))
        mujoco.mj_forward(model, data)

        base_rotation = data.xmat[base_id].reshape(3, 3)
        delta_world = data.xpos[wheel_id] - data.xpos[hip_id]
        delta_base = base_rotation.T @ delta_world
        # The exported left-leg mechanism moves in the base Y-Z plane.
        # ``s_w`` is therefore +base Y; it is deliberately not called
        # "forward" because the simulation controller defines its driving
        # x direction with the opposite sign.
        s_w = float(delta_base[1])
        d_w = float(-delta_base[2])
        l0 = math.hypot(s_w, d_w)
        theta_leg = math.atan2(s_w, d_w)

        gaps = []
        for left, right in CONNECTS:
            left_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, left)
            right_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, right)
            gaps.append(float(np.linalg.norm(data.site_xpos[left_id] - data.site_xpos[right_id])))

        rows.append(
            {
                "target_l0_m": target,
                "mujoco_l0_m": l0,
                "offline_l0_m": sample["exact_l0_m"],
                "l0_difference_um": 1e6 * (l0 - float(sample["exact_l0_m"])),
                "mujoco_theta_leg_deg": math.degrees(theta_leg),
                "offline_theta_leg_deg": math.degrees(float(sample["exact_theta_leg_rad"])),
                "theta_difference_microdeg": 1e6
                * math.degrees(
                    math.atan2(
                        math.sin(theta_leg - float(sample["exact_theta_leg_rad"])),
                        math.cos(theta_leg - float(sample["exact_theta_leg_rad"])),
                    )
                ),
                "max_equality_site_gap_um": 1e6 * max(gaps),
            }
        )

    result = {
        "mujoco_version": mujoco.__version__,
        "xml": str(XML.resolve()),
        "method": "Strip unavailable visual meshes only; set active and offline-solved passive qpos; call mj_forward without dynamic settling.",
        "max_abs_l0_difference_um": max(abs(row["l0_difference_um"]) for row in rows),
        "max_abs_theta_difference_microdeg": max(abs(row["theta_difference_microdeg"]) for row in rows),
        "max_equality_site_gap_um": max(row["max_equality_site_gap_um"] for row in rows),
        "samples": rows,
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
