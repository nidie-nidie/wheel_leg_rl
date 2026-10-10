from __future__ import annotations

import json

import pytest

from debug.sim2sim.root_cause_suite.integrity import catalog_snapshot
from debug.sim2sim.root_cause_suite.scenario_catalog import (
    FACTOR_PATH_ALLOWLIST_PATH,
    RATIO_PAIR_IDS,
    catalog,
    factor_path_allowlist,
    factor_path_allowlist_artifact,
    sphere_probe_profiles,
    robot_probe_profiles,
    validate_catalog,
)
from debug.sim2sim.root_cause_suite.contracts import canonical_json_bytes


def test_catalog_contains_only_frozen_core_and_ratio_pairs() -> None:
    scenarios = catalog()
    validate_catalog(scenarios)
    assert RATIO_PAIR_IDS == {
        "P30_OPEN_TARGET_TO_DIRECT",
        "P30_CLOSED_TARGET_TO_DIRECT",
        "P40_CLOSURE_ON_TO_OFF",
        "P50_NOMINAL_TO_ZERO_FRICTION",
    }
    names = {scenario.scenario_id for scenario in scenarios}
    assert not any(name.startswith(("P20_D", "P40_C", "P50_B", "P50_D", "P60_A", "P60_E")) for name in names)


def test_catalog_identity_survives_manifest_json_roundtrip() -> None:
    snapshot = catalog_snapshot()
    restored = json.loads(canonical_json_bytes(snapshot).decode("ascii"))
    assert restored == snapshot


def test_catalog_rejects_wildcard_semantic_path() -> None:
    scenario = catalog()[0]
    broken = scenario.with_allowed_paths(("/closure/*",))
    with pytest.raises(ValueError, match="wildcard"):
        validate_catalog([broken])


def test_factor_path_allowlist_artifact_is_canonical_and_exact() -> None:
    payload = factor_path_allowlist_artifact()
    assert (
        FACTOR_PATH_ALLOWLIST_PATH.read_bytes()
        == canonical_json_bytes(payload) + b"\n"
    )
    assert len(
        factor_path_allowlist("mujoco", "P30_OPEN_TARGET_TO_DIRECT")
    ) == 23
    assert len(
        factor_path_allowlist("isaac", "P30_OPEN_TARGET_TO_DIRECT")
    ) == 55
    assert factor_path_allowlist(
        "mujoco", "P50_NOMINAL_TO_ZERO_FRICTION"
    ) == (
        "/contact/pairs/floor_coupon/friction/0",
        "/contact/pairs/floor_coupon/friction/1",
    )


def test_sphere_impact_profiles_freeze_six_initial_conditions() -> None:
    profiles = sphere_probe_profiles("impact", repetitions=3)
    assert len(profiles) == 18
    conditions = {
        (profile["height_m"], profile["vertical_velocity_mps"])
        for profile in profiles
    }
    assert conditions == {
        (0.05, 0.0),
        (0.05, -0.5),
        (0.10, 0.0),
        (0.10, -0.5),
        (0.20, 0.0),
        (0.20, -0.5),
    }
    assert all(profile["horizontal_velocity_mps"] == 0.0 for profile in profiles)


def test_sphere_slide_profiles_freeze_signed_velocity_matrix() -> None:
    profiles = sphere_probe_profiles("slide", repetitions=3)
    assert len(profiles) == 9
    assert {profile["horizontal_velocity_mps"] for profile in profiles} == {
        -1.0,
        0.0,
        1.0,
    }
    assert all(profile["height_m"] == 0.0 for profile in profiles)


def test_sphere_profiles_require_three_repetitions() -> None:
    with pytest.raises(ValueError, match="at least three"):
        sphere_probe_profiles("impact", repetitions=2)
    with pytest.raises(ValueError, match="Unknown"):
        sphere_probe_profiles("unsupported", repetitions=3)


def test_p60_c_profiles_accept_one_frozen_six_channel_action() -> None:
    action = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    profiles = robot_probe_profiles(
        "p60_c", repetitions=3, input_vector=action
    )
    assert len(profiles) == 3
    assert all(profile["actual_input"] == action for profile in profiles)
    with pytest.raises(ValueError, match="six"):
        robot_probe_profiles("p60_c", repetitions=3, input_vector=[0.0])
