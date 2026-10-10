from __future__ import annotations

from copy import deepcopy

import pytest

from debug.sim2sim.root_cause_suite.causal_contract import (
    result_identity_fields,
    robot_configuration_semantics,
    validate_ratio_pair,
    validate_result_identity,
)
from debug.sim2sim.root_cause_suite.contracts import (
    DECLARED_USD_INFINITY_SPECS,
    EvidenceIntegrityError,
    declared_infinity_tag,
    encode_declared_usd_value,
    stable_hash,
)
from debug.sim2sim.root_cause_suite.scenario_catalog import factor_path_allowlist
from debug.sim2sim.root_cause_suite.variant_builders import ROBOT_HINGES


def _p30_resolved_semantics(scenario: str) -> dict:
    direct = scenario.endswith("direct")
    robot_hinges = {
        name: {
            "stiffness": 0.0 if direct else 1.0,
            "damping": 0.0 if direct else 2.0,
            "armature": 0.01,
            "position_limits": [-1.0, 1.0],
            "velocity_limit": 10.0,
            "effort_limit": 20.0,
            "resolved_sources_equal": True,
        }
        for name in ROBOT_HINGES
    }
    return {
        "topology": {
            "body_names": ["base"],
            "joint_names": list(ROBOT_HINGES),
        },
        "environment": {
            "ground_enabled": False,
            "ground_geometry": None,
            "gravity_enabled": False,
            "fixed_base": True,
        },
        "engine_options": {"solver": "frozen", "iterations": 4},
        "static_model": {
            "bodies": {"base": {"mass": 4.0}},
            "joints": {name: {"axis": [0.0, 1.0, 0.0]} for name in ROBOT_HINGES},
        },
        "closure": {"mode": "disabled", "constraints": {}},
        "actuation": {
            "drive_mode": "direct_effort_bypass" if direct else "formal_target_drive",
            "external_controller_enabled": not direct,
            "target_neutralization": direct,
            "robot_hinges": robot_hinges,
            "actuators": {"controlled": {"gear": 1.0}},
        },
        "contact": {"mode": "none", "pairs": {}},
    }


def _p30_transform_semantics(resolved: dict) -> dict:
    actuation = resolved["actuation"]
    return {
        "actuation": {
            "drive_mode": actuation["drive_mode"],
            "external_controller_enabled": actuation[
                "external_controller_enabled"
            ],
            "target_neutralization": actuation["target_neutralization"],
            "robot_hinges": {
                name: {
                    "stiffness": record["stiffness"],
                    "damping": record["damping"],
                }
                for name, record in actuation["robot_hinges"].items()
            },
        }
    }


def _p30_result(
    scenario: str,
    *,
    pre_forward: str = "PRE",
    post_forward: str = "POST",
    reset_returned: str = "RETURNED",
    hidden_gravity_change: bool = False,
    hidden_body_mass_change: bool = False,
    hidden_engine_option_change: bool = False,
    hidden_armature_change: bool = False,
    excitation_stop_s: float = 0.02,
    transform_missing_drive_value: bool = False,
    worker_source: str = "A" * 64,
) -> dict:
    direct = scenario.endswith("direct")
    resolved = _p30_resolved_semantics(scenario)
    if hidden_body_mass_change:
        resolved["static_model"]["bodies"]["base"]["mass"] = 5.0
    if hidden_engine_option_change:
        resolved["engine_options"]["iterations"] = 8
    if hidden_armature_change:
        resolved["actuation"]["robot_hinges"][ROBOT_HINGES[0]][
            "armature"
        ] = 0.02
    semantics = robot_configuration_semantics(
        scenario=scenario,
        engine="isaac",
        physics_dt_s=0.005,
        control_dt_s=0.02,
        source_identity="B" * 64,
        resolved_semantics=resolved,
    )
    if hidden_gravity_change:
        semantics = deepcopy(semantics)
        semantics["environment"]["gravity_enabled"] = True
    transform_semantics = _p30_transform_semantics(resolved)
    if transform_missing_drive_value:
        transform_semantics = deepcopy(transform_semantics)
        transform_semantics["actuation"]["robot_hinges"][ROBOT_HINGES[0]][
            "damping"
        ] = 2.0
    excitation = {
        "input_kind": "direct_effort" if direct else "formal_target_action",
        "input_units": "N*m" if direct else "normalized_action",
        "actual_input_values": [[0.3]] if direct else [[0.3 / (120.0 * 0.35)]],
        "canonical_torque_equivalent": [[0.3]],
        "start_s": 0.0,
        "stop_s": excitation_stop_s,
        "physics_sample_dt_s": 0.005,
        "control_dt_s": 0.02,
    }
    scenario_id = (
        "P30_A_OPEN_DIRECT_EFFORT" if direct else "P30_B_OPEN_FORMAL_TARGET"
    )
    payload = result_identity_fields(
        scenario_id=scenario_id,
        engine="isaac",
        source_model_sha256="B" * 64,
        model_artifact_sha256=("C" if direct else "D") * 64,
        transform_manifest_sha256=("E" if direct else "F") * 64,
        transform_semantics=transform_semantics,
        worker_source_sha256=worker_source,
        configuration_semantics=semantics,
        pre_forward_initial_condition_hash=pre_forward,
        post_forward_state_hash=post_forward,
        reset_returned_policy_hash=reset_returned,
        excitation_hash=stable_hash(excitation),
        comparison_profile_hash="PROFILE",
    )
    payload["excitation_semantics"] = excitation
    return payload


def test_exact_p30_ratio_pair_passes() -> None:
    result = validate_ratio_pair(
        _p30_result("p30_open_target"),
        _p30_result("p30_open_direct"),
        pair_id="P30_OPEN_TARGET_TO_DIRECT",
    )
    assert result["semantic_diff_paths"] == list(
        factor_path_allowlist("isaac", "P30_OPEN_TARGET_TO_DIRECT")
    )
    assert result["excitation_diff_paths"] == [
        "/actual_input_values",
        "/input_kind",
        "/input_units",
    ]


def test_hidden_semantic_change_is_rejected() -> None:
    with pytest.raises(EvidenceIntegrityError, match="semantic diff"):
        validate_ratio_pair(
            _p30_result("p30_open_target"),
            _p30_result("p30_open_direct", hidden_gravity_change=True),
            pair_id="P30_OPEN_TARGET_TO_DIRECT",
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "hidden_body_mass_change",
        "hidden_engine_option_change",
        "hidden_armature_change",
    ),
)
def test_hidden_compiled_semantic_mutations_are_rejected(mutation: str) -> None:
    with pytest.raises(EvidenceIntegrityError, match="semantic diff"):
        validate_ratio_pair(
            _p30_result("p30_open_target"),
            _p30_result("p30_open_direct", **{mutation: True}),
            pair_id="P30_OPEN_TARGET_TO_DIRECT",
        )


def test_transform_manifest_must_match_compiled_diff() -> None:
    with pytest.raises(EvidenceIntegrityError, match="Transform manifest diff"):
        validate_ratio_pair(
            _p30_result("p30_open_target"),
            _p30_result(
                "p30_open_direct", transform_missing_drive_value=True
            ),
            pair_id="P30_OPEN_TARGET_TO_DIRECT",
        )


def test_ratio_pair_must_use_one_worker_source() -> None:
    with pytest.raises(EvidenceIntegrityError, match="worker source"):
        validate_ratio_pair(
            _p30_result("p30_open_target"),
            _p30_result("p30_open_direct", worker_source="9" * 64),
            pair_id="P30_OPEN_TARGET_TO_DIRECT",
        )


def test_p40_configuration_semantics_requires_free_base() -> None:
    resolved = _p30_resolved_semantics("p30_open_direct")
    resolved["environment"]["fixed_base"] = False
    resolved["closure"] = {"mode": "enabled", "constraints": {}}
    semantics = robot_configuration_semantics(
        scenario="p40_on",
        engine="isaac",
        physics_dt_s=0.005,
        control_dt_s=0.02,
        source_identity="B" * 64,
        resolved_semantics=resolved,
    )
    assert semantics["environment"]["fixed_base"] is False
    assert semantics["closure"]["mode"] == "enabled"

    closure_off = deepcopy(resolved)
    closure_off["closure"]["mode"] = "disabled"
    off_semantics = robot_configuration_semantics(
        scenario="p40_off",
        engine="isaac",
        physics_dt_s=0.005,
        control_dt_s=0.02,
        source_identity="B" * 64,
        resolved_semantics=closure_off,
    )
    assert off_semantics["closure"]["mode"] == "disabled"

    broken = deepcopy(resolved)
    broken["environment"]["fixed_base"] = True
    with pytest.raises(EvidenceIntegrityError, match="environment"):
        robot_configuration_semantics(
            scenario="p40_on",
            engine="isaac",
            physics_dt_s=0.005,
            control_dt_s=0.02,
            source_identity="B" * 64,
            resolved_semantics=broken,
        )


def _minimal_identity(configuration_semantics: dict) -> dict:
    return result_identity_fields(
        scenario_id="P50_A_SPHERE_IMPACT",
        engine="isaac",
        source_model_sha256="A" * 64,
        model_artifact_sha256="B" * 64,
        transform_manifest_sha256="C" * 64,
        transform_semantics={"contact": {"friction": 0.7}},
        worker_source_sha256="D" * 64,
        configuration_semantics=configuration_semantics,
        pre_forward_initial_condition_hash="E" * 64,
        post_forward_state_hash="F" * 64,
        reset_returned_policy_hash=None,
        excitation_hash="1" * 64,
        comparison_profile_hash="2" * 64,
    )


def _rebind_minimal_identity(payload: dict) -> None:
    payload["configuration_semantics_hash"] = stable_hash(
        payload["configuration_semantics"]
    )
    payload["configuration_hash"] = stable_hash(
        {
            "engine": payload["engine"],
            "source_model_sha256": payload["source_model_sha256"],
            "model_artifact_sha256": payload["model_artifact_sha256"],
            "transform_manifest_sha256": payload["transform_manifest_sha256"],
            "transform_semantics_hash": payload["transform_semantics_hash"],
            "configuration_semantics_hash": payload[
                "configuration_semantics_hash"
            ],
            "worker_source_sha256": payload["worker_source_sha256"],
        }
    )
    payload["repeatability_key"] = stable_hash(
        {
            "configuration_hash": payload["configuration_hash"],
            "pre_forward_initial_condition_hash": payload[
                "pre_forward_initial_condition_hash"
            ],
            "post_forward_state_hash": payload["post_forward_state_hash"],
            "reset_returned_policy_hash": payload.get(
                "reset_returned_policy_hash"
            ),
            "excitation_hash": payload["excitation_hash"],
        }
    )


def test_result_identity_accepts_exact_declared_usd_infinity_path() -> None:
    tag = encode_declared_usd_value(
        float("inf"),
        prim_path="/physicsScene",
        attribute="physxScene:maxBiasCoefficient",
    )
    payload = _minimal_identity(
        {
            "engine_options": {
                "resolved_physics_scene": {
                    "attributes": {
                        "physxScene:maxBiasCoefficient": {"value": tag}
                    }
                }
            }
        }
    )
    validate_result_identity(payload)


@pytest.mark.parametrize(
    "field", ("mass", "inertia", "stiffness", "damping", "material", "solver")
)
def test_result_identity_rejects_raw_nonfinite_physical_semantics(field: str) -> None:
    with pytest.raises(EvidenceIntegrityError, match="raw NaN/Inf"):
        _minimal_identity({"static_model": {field: float("inf")}})


def test_result_identity_rejects_tag_outside_declared_path() -> None:
    spec = DECLARED_USD_INFINITY_SPECS[
        ("/physicsScene", "physxScene:maxBiasCoefficient")
    ]
    with pytest.raises(EvidenceIntegrityError, match="undeclared path"):
        _minimal_identity(
            {"static_model": {"bodies": {"base": {"mass": declared_infinity_tag(spec)}}}}
        )


def test_result_identity_rejects_self_consistent_malformed_tag() -> None:
    spec = DECLARED_USD_INFINITY_SPECS[
        ("/physicsScene", "physxScene:maxBiasCoefficient")
    ]
    tag = declared_infinity_tag(spec)
    tag["semantics"] = "wrong"
    configuration = {
        "engine_options": {
            "resolved_physics_scene": {
                "attributes": {"physxScene:maxBiasCoefficient": {"value": tag}}
            }
        }
    }
    with pytest.raises(EvidenceIntegrityError, match="invalid infinity tag"):
        _minimal_identity(configuration)


def test_result_identity_rejects_rehashed_missing_kind_and_wrong_version() -> None:
    tag = encode_declared_usd_value(
        float("inf"),
        prim_path="/physicsScene",
        attribute="physxScene:maxBiasCoefficient",
    )
    payload = _minimal_identity(
        {
            "engine_options": {
                "resolved_physics_scene": {
                    "attributes": {
                        "physxScene:maxBiasCoefficient": {"value": tag}
                    }
                }
            }
        }
    )
    malformed = payload["configuration_semantics"]["engine_options"][
        "resolved_physics_scene"
    ]["attributes"]["physxScene:maxBiasCoefficient"]["value"]
    malformed.pop("kind")
    malformed["schema_version"] = "WrongVersion"
    _rebind_minimal_identity(payload)
    with pytest.raises(EvidenceIntegrityError, match="malformed infinity tag"):
        validate_result_identity(payload)


def test_result_identity_rejects_rehashed_all_leaf_shaped_malformed_tags() -> None:
    center = encode_declared_usd_value(
        [-float("inf"), -float("inf"), -float("inf")],
        prim_path="/World/envs/env_0/Coupon",
        attribute="physics:centerOfMass",
    )
    payload = _minimal_identity(
        {
            "static_model": {
                "coupon": {
                    "prim": {
                        "attributes": {
                            "physics:centerOfMass": {"value": center}
                        }
                    }
                }
            }
        }
    )
    malformed = payload["configuration_semantics"]["static_model"]["coupon"][
        "prim"
    ]["attributes"]["physics:centerOfMass"]["value"]
    for tag in malformed:
        tag.pop("kind")
        tag["schema_version"] = "WrongVersion"
    _rebind_minimal_identity(payload)
    with pytest.raises(EvidenceIntegrityError, match="malformed infinity tag"):
        validate_result_identity(payload)


def test_reversed_ratio_pair_is_rejected() -> None:
    with pytest.raises(EvidenceIntegrityError, match="reversed"):
        validate_ratio_pair(
            _p30_result("p30_open_direct"),
            _p30_result("p30_open_target"),
            pair_id="P30_OPEN_TARGET_TO_DIRECT",
        )


@pytest.mark.parametrize(
    ("field", "keyword"),
    (
        ("pre_forward", "pre-forward"),
        ("post_forward", "post_forward_state_hash"),
        ("reset_returned", "reset_returned_policy_hash"),
    ),
)
def test_three_phase_reset_mismatch_is_rejected(field: str, keyword: str) -> None:
    arguments = {field: "DIFFERENT"}
    with pytest.raises(EvidenceIntegrityError, match=keyword):
        validate_ratio_pair(
            _p30_result("p30_open_target"),
            _p30_result("p30_open_direct", **arguments),
            pair_id="P30_OPEN_TARGET_TO_DIRECT",
        )


def test_extra_excitation_change_is_rejected() -> None:
    with pytest.raises(EvidenceIntegrityError, match="excitation diff"):
        validate_ratio_pair(
            _p30_result("p30_open_target"),
            _p30_result("p30_open_direct", excitation_stop_s=0.03),
            pair_id="P30_OPEN_TARGET_TO_DIRECT",
        )
