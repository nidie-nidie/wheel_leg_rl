from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import run_project_script


@pytest.mark.integration
def test_asset_bundle_v2_reference_and_runtime_overrides(tmp_path: Path) -> None:
    output = tmp_path / "asset-audit.json"
    run_project_script(["scripts/audit_asset.py", "--output", str(output), "--headless"])

    report = json.loads(output.read_text(encoding="utf-8"))
    source_v1 = report["source_v1"]
    source_v2 = report["source_v2"]
    runtime = report["reference_load_v2"]
    assert report["source_v1_files_unchanged"] is True
    assert report["asset_bundle_v2"]["bundle_version"] == "AssetBundleV2"
    assert source_v1["embedded_ground_collision_enabled"] is True
    assert source_v2["embedded_ground_collision_enabled"] is None
    assert source_v2["embedded_ground_absent"] is True
    assert source_v2["root_prims"] == ["/physicsScene", "/wheel_leg_urdf4", "/Render", "/World"]
    assert source_v2["mass"]["total_mass_kg"] == pytest.approx(4.396253988146782)
    assert source_v2["mass"]["dummy_body_count"] == 12
    assert source_v2["mirror_collision_asymmetry"]["jMK_enabled_paths"]
    assert source_v2["mirror_collision_asymmetry"]["jEC_enabled_paths"] == []
    assert runtime["physics_scenes_after_reset"] == ["/physicsScene"]
    assert runtime["robot_namespace_physics_scenes"] == []
    assert runtime["embedded_ground_absent_before_reset"] is True
    assert runtime["embedded_ground_absent_after_reset"] is True
    assert len(runtime["enabled_robot_collision_paths_after_reset"]) == 2
    assert all("jwheel_" in path for path in runtime["enabled_robot_collision_paths_after_reset"])
