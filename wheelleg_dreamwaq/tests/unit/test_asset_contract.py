from __future__ import annotations

from pathlib import Path

import pytest

from wheelleg_dreamwaq.assets.asset_contract import (
    ASSET_BUNDLE_V1,
    ASSET_BUNDLE_V2,
    AssetContractError,
    verify_asset_bundle,
)
from wheelleg_dreamwaq.assets.asset_overrides import is_wheel_collision_path


def test_asset_bundle_v1_matches_the_frozen_source_files() -> None:
    report = verify_asset_bundle()

    assert report.bundle_version == "AssetBundleV1"
    assert report.entry_file == ASSET_BUNDLE_V1.entry_file
    assert report.bundle_hash == ASSET_BUNDLE_V1.bundle_hash
    assert len(report.files) == 5
    assert all(item.exists and item.size_matches and item.sha256_matches for item in report.files)


def test_asset_bundle_v2_matches_the_generated_ground_free_copy() -> None:
    report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)

    assert report.bundle_version == "AssetBundleV2"
    assert report.bundle_hash == "E754AE888F5C5379B3B6152CFA5AD6BBAE20E8C7480CF7AD6960782A5267CB46"
    assert ASSET_BUNDLE_V2.embedded_ground is None
    assert len(report.files) == 5
    assert all(item.exists and item.size_matches and item.sha256_matches for item in report.files)


def test_asset_contract_freezes_stage_structure() -> None:
    assert ASSET_BUNDLE_V1.default_prim == "/wheel_leg_urdf4"
    assert ASSET_BUNDLE_V1.articulation_root == "/wheel_leg_urdf4/base_link"
    assert ASSET_BUNDLE_V1.embedded_ground == "/wheel_leg_urdf4/GroundPlane/CollisionPlane"
    assert ASSET_BUNDLE_V1.up_axis == "Z"
    assert ASSET_BUNDLE_V1.meters_per_unit == 1.0
    assert ASSET_BUNDLE_V1.rigid_body_count == 27
    assert ASSET_BUNDLE_V1.total_mass_kg == pytest.approx(4.396253988146782)
    assert ASSET_BUNDLE_V1.dummy_body_count == 12
    assert ASSET_BUNDLE_V1.dummy_mass_kg == pytest.approx(0.11999999731779099)
    assert ASSET_BUNDLE_V1.tree_joint_count == 26
    assert ASSET_BUNDLE_V1.loop_joint_count == 4


def test_asset_contract_freezes_controlled_and_loop_joint_names() -> None:
    assert ASSET_BUNDLE_V1.controlled_joints == (
        "jIJ",
        "jIO",
        "jAB",
        "jAG",
        "jwheel_left",
        "jwheel_right",
    )
    assert ASSET_BUNDLE_V1.loop_joint_paths == (
        "/wheel_leg_urdf4/jIO/jIO_loop_closure",
        "/wheel_leg_urdf4/jKN/jKN_loop_closure",
        "/wheel_leg_urdf4/jEC/jAG_loop_closure",
        "/wheel_leg_urdf4/jCF/jCF_revolute_joint",
    )


def test_asset_verification_rejects_a_wrong_root(tmp_path: Path) -> None:
    with pytest.raises(AssetContractError, match="missing"):
        verify_asset_bundle(tmp_path)


def test_wheel_only_collision_path_filter_is_name_based() -> None:
    root = "/World/envs/env_0/Robot"
    assert is_wheel_collision_path(f"{root}/jwheel_left/collisions/collision_0/child_0", root)
    assert is_wheel_collision_path(f"{root}/jwheel_right/collisions/collision_0/child_0", root)
    assert not is_wheel_collision_path(f"{root}/jMK/collisions/collision_0/child_0", root)
    assert not is_wheel_collision_path(f"{root}/GroundPlane/CollisionPlane", root)
