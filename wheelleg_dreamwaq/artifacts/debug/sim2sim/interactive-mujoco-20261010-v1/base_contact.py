"""Create a viewer-only model with the existing base box contacting the floor."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wheelleg_mujoco.contract import sha256_file
from wheelleg_mujoco.model_semantics import compiled_model_semantics, stable_hash


def create_model(source_path, output_path):
    source_path, output_path = Path(source_path).resolve(), Path(output_path).resolve()
    tree = ET.parse(source_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((source_path.parent / compiler.get("meshdir", "")).resolve()))
    contact = root.find("contact")
    assert contact is not None
    original = contact.find("pair[@name='floor_left_wheel']")
    assert original is not None and len(contact.findall("pair")) == 2
    extra = copy.deepcopy(original)
    extra.set("name", "floor_base_debug")
    extra.set("geom2", "base_proxy")
    contact.append(extra)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    before = mujoco.MjModel.from_xml_path(str(source_path))
    after = mujoco.MjModel.from_xml_path(str(output_path))
    old_semantics, new_semantics = map(compiled_model_semantics, (before, after))
    comparison = copy.deepcopy(new_semantics)
    added = [pair for pair in comparison["contact_pairs"] if pair["name"] == "floor_base_debug"]
    assert len(added) == 1 and {added[0]["geom1"], added[0]["geom2"]} == {"floor", "base_proxy"}
    comparison["contact_pairs"] = [pair for pair in comparison["contact_pairs"] if pair["name"] != "floor_base_debug"]
    comparison["dimensions"]["npair"] -= 1
    assert stable_hash(comparison) == stable_hash(old_semantics), "Unexpected dynamics change beyond added base contact"
    for name in ("geom_type", "geom_bodyid", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    identity = {"schema_version": "ViewerBaseGroundContactDebugV1", "formal_source": str(source_path),
                "formal_source_sha256": sha256_file(source_path), "debug_model": str(output_path),
                "debug_model_sha256": sha256_file(output_path), "added_contact_pair": added[0],
                "other_compiled_dynamics_unchanged": True,
                "note": "Approximate existing base_proxy box; viewer experiment, not a formal evaluation model"}
    output_path.with_suffix(".json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return output_path


def base_contact_state(model, data):
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "base_proxy")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    rotation = data.geom_xmat[base_id].reshape(3, 3)
    bottom = float(data.geom_xpos[base_id, 2] - np.dot(np.abs(rotation[2]), model.geom_size[base_id]))
    count, normal_force = 0, 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        if {int(contact.geom1), int(contact.geom2)} == {base_id, floor_id}:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            count += 1
            normal_force += float(force[0])
    return {"base_proxy_min_z_m": bottom, "base_floor_contact_count": count,
            "base_floor_normal_force_n": normal_force}
