from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from debug.sim2sim.root_cause_suite.contracts import sha256_file
from debug.sim2sim.root_cause_suite.variant_builders import (
    IsaacVariantSpec,
    build_common_sphere_mjcf,
    build_robot_variant,
    validate_robot_variant,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
FORMAL = PROJECT_ROOT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"


def test_robot_variants_are_run_local_and_formal_hash_is_unchanged(tmp_path: Path) -> None:
    before = sha256_file(FORMAL)
    result = build_robot_variant(
        FORMAL,
        tmp_path / "variant.xml",
        operations=("no_ground", "gravity_off", "closure_off", "drive_off", "fixed_base"),
    )
    assert sha256_file(FORMAL) == before
    assert result.model_path.is_relative_to(tmp_path.resolve())
    assert result.manifest_path.is_file()
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["source_sha256"] == before
    assert payload["operations"] == [
        "no_ground", "gravity_off", "closure_off", "drive_off", "fixed_base"
    ]
    validate_robot_variant(result.model_path, payload)

    root = ET.parse(result.model_path).getroot()
    assert Path(root.find("compiler").get("meshdir")).is_absolute()
    assert root.find("worldbody/geom[@name='floor']") is None
    assert root.find("worldbody/body/freejoint") is None
    connects = root.findall("equality/connect")
    assert len(connects) == 8
    assert all(item.get("active") == "false" for item in connects)
    joints = root.findall("worldbody//joint")
    assert len(joints) == 26
    assert all(float(item.get("damping", "nan")) == 0.0 for item in joints)


def test_validator_rejects_hidden_armature_mutation(tmp_path: Path) -> None:
    result = build_robot_variant(FORMAL, tmp_path / "variant.xml", operations=("drive_off",))
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    tree = ET.parse(result.model_path)
    tree.getroot().find(".//joint[@name='jIO']").set("armature", "9")
    tree.write(result.model_path, encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError, match="manifest|semantic"):
        validate_robot_variant(result.model_path, payload)


def test_common_sphere_coupon_is_self_contained(tmp_path: Path) -> None:
    result = build_common_sphere_mjcf(
        tmp_path / "sphere.xml",
        radius_m=0.0625,
        mass_kg=0.4,
        friction=(0.0, 0.0, 0.0),
    )
    root = ET.parse(result.model_path).getroot()
    assert root.find("worldbody/body[@name='coupon']/geom[@name='coupon_geom']") is not None
    assert root.find("contact/pair[@name='floor_coupon']") is not None
    validate_robot_variant(result.model_path, json.loads(result.manifest_path.read_text(encoding="utf-8")))


def test_isaac_variant_spec_rejects_multiple_undeclared_factors() -> None:
    spec = IsaacVariantSpec(ground_enabled=False, gravity_enabled=False)
    assert spec.changed_factors() == ("gravity", "ground",)
    with pytest.raises(ValueError):
        IsaacVariantSpec(coupon="mesh")
