from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ISAAC_LOOP_CLOSURE_PRIM_PATHS,
    DeclaredInfinitySpec,
    EvidenceIntegrityError,
    canonical_json_bytes,
    declared_usd_infinity_spec,
    sha256_file,
    stable_hash,
    validate_declared_infinity_payload,
)
from .scenario_catalog import (
    ScenarioSpec,
    catalog,
    factor_path_allowlist,
    factor_path_allowlist_artifact,
)


WORKER_SCENARIO_IDS = {
    "p10_a": "P10_A_REST_CLOSURE_ON",
    "p10_b": "P10_B_FREEFALL_CLOSURE_ON",
    "p10_c": "P10_C_REST_CLOSURE_OFF",
    "p30_open_direct": "P30_A_OPEN_DIRECT_EFFORT",
    "p30_open_target": "P30_B_OPEN_FORMAL_TARGET",
    "p30_closed_direct": "P30_C_CLOSED_DIRECT_EFFORT",
    "p30_closed_target": "P30_C_CLOSED_FORMAL_TARGET",
    "p40_on": "P40_B_CLOSURE_ON",
    "p40_off": "P40_B_CLOSURE_OFF",
    "sphere_impact": "P50_A_SPHERE_IMPACT",
    "sphere_slide_nominal": "P50_C_NOMINAL_FRICTION",
    "sphere_slide_zero": "P50_C_ZERO_FRICTION",
    "p60_b": "P60_B_ZERO_ACTION_DRIVE_OFF",
    "p60_c": "P60_C_SHARED_FIRST_ACTION",
}


def scenario_spec(scenario_id: str) -> ScenarioSpec:
    try:
        return next(item for item in catalog() if item.scenario_id == scenario_id)
    except StopIteration as error:
        raise EvidenceIntegrityError(f"Unknown frozen scenario id: {scenario_id}") from error


def robot_scenario_id(scenario: str) -> str:
    if scenario not in WORKER_SCENARIO_IDS:
        raise EvidenceIntegrityError(f"Robot scenario has no frozen identity: {scenario}")
    return WORKER_SCENARIO_IDS[scenario]


def sphere_scenario_id(mode: str, friction: float) -> str:
    if mode == "impact":
        return WORKER_SCENARIO_IDS["sphere_impact"]
    if mode == "slide" and abs(float(friction)) <= 1.0e-12:
        return WORKER_SCENARIO_IDS["sphere_slide_zero"]
    if mode == "slide" and abs(float(friction) - 1.0) <= 1.0e-12:
        return WORKER_SCENARIO_IDS["sphere_slide_nominal"]
    raise EvidenceIntegrityError(
        f"Sphere probe is not a frozen Core scenario: mode={mode}, friction={friction}"
    )


def robot_configuration_semantics(
    *,
    scenario: str,
    engine: str,
    physics_dt_s: float,
    control_dt_s: float,
    source_identity: str,
    resolved_semantics: Mapping[str, Any],
    observer_mode: str = "formal",
) -> dict[str, Any]:
    if engine not in {"isaac", "mujoco"}:
        raise EvidenceIntegrityError(f"Unknown robot semantics engine: {engine}")
    if physics_dt_s <= 0.0 or control_dt_s <= 0.0:
        raise EvidenceIntegrityError("Robot timing semantics must be positive")
    direct = scenario.endswith("direct") or scenario.startswith("p10") or scenario.startswith(
        "p40"
    ) or scenario == "p60_b"
    closure_enabled = scenario not in {
        "p10_c",
        "p30_open_direct",
        "p30_open_target",
        "p40_off",
    }
    ground_enabled = scenario.startswith("p60")
    gravity_enabled = scenario in {"p10_b", "p60_b", "p60_c"}
    fixed_base = scenario.startswith("p30")
    required_sections = {
        "topology",
        "environment",
        "engine_options",
        "static_model",
        "closure",
        "actuation",
        "contact",
    }
    if set(resolved_semantics) != required_sections:
        raise EvidenceIntegrityError(
            "Resolved robot semantics section set is not exact: "
            f"actual={sorted(resolved_semantics)}, expected={sorted(required_sections)}"
        )
    resolved = {name: resolved_semantics[name] for name in sorted(required_sections)}
    expected_environment = {
        "ground_enabled": ground_enabled,
        "gravity_enabled": gravity_enabled,
        "fixed_base": fixed_base,
    }
    environment = resolved["environment"]
    if not isinstance(environment, Mapping) or any(
        environment.get(name) != value for name, value in expected_environment.items()
    ):
        raise EvidenceIntegrityError(
            f"Resolved robot environment disagrees with scenario {scenario}"
        )
    expected_closure = "enabled" if closure_enabled else "disabled"
    closure = resolved["closure"]
    if not isinstance(closure, Mapping) or closure.get("mode") != expected_closure:
        raise EvidenceIntegrityError(
            f"Resolved closure mode disagrees with scenario {scenario}"
        )
    expected_drive = "direct_effort_bypass" if direct else "formal_target_drive"
    actuation = resolved["actuation"]
    if not isinstance(actuation, Mapping) or actuation.get("drive_mode") != expected_drive:
        raise EvidenceIntegrityError(
            f"Resolved drive mode disagrees with scenario {scenario}"
        )
    payload = {
        "schema_version": "RootCauseRobotConfigurationSemanticsV2",
        "engine": engine,
        "source_identity": source_identity,
        "topology": resolved["topology"],
        "environment": dict(environment),
        "timing": {
            "physics_dt_s": float(physics_dt_s),
            "control_dt_s": float(control_dt_s),
            "physics_steps_per_control": int(round(control_dt_s / physics_dt_s)),
        },
        "engine_options": resolved["engine_options"],
        "static_model": resolved["static_model"],
        "closure": dict(closure),
        "actuation": dict(actuation),
        "contact": resolved["contact"],
        "observer_mode": observer_mode,
    }
    if payload["actuation"].get("external_controller_enabled") is not (not direct):
        raise EvidenceIntegrityError("Resolved external-controller state is invalid")
    if payload["actuation"].get("target_neutralization") is not direct:
        raise EvidenceIntegrityError("Resolved target-neutralization state is invalid")
    return payload


def sphere_configuration_semantics(
    *,
    engine: str,
    friction: float,
    physics_dt_s: float,
    source_identity: str,
    resolved_semantics: Mapping[str, Any],
    observer_mode: str = "formal",
) -> dict[str, Any]:
    if engine not in {"isaac", "mujoco"}:
        raise EvidenceIntegrityError(f"Unknown sphere semantics engine: {engine}")
    required_sections = {
        "topology",
        "environment",
        "engine_options",
        "static_model",
        "contact",
    }
    if set(resolved_semantics) != required_sections:
        raise EvidenceIntegrityError(
            "Resolved sphere semantics section set is not exact: "
            f"actual={sorted(resolved_semantics)}, expected={sorted(required_sections)}"
        )
    contact = resolved_semantics["contact"]
    if not isinstance(contact, Mapping):
        raise EvidenceIntegrityError("Resolved sphere contact semantics are invalid")
    pairs = contact.get("pairs")
    if not isinstance(pairs, Mapping) or "floor_coupon" not in pairs:
        raise EvidenceIntegrityError("Resolved sphere pair floor_coupon is missing")
    pair = pairs["floor_coupon"]
    if not isinstance(pair, Mapping):
        raise EvidenceIntegrityError("Resolved sphere pair semantics are invalid")
    if engine == "isaac":
        material = pair.get("material")
        if not isinstance(material, Mapping) or any(
            abs(float(material.get(name, float("nan"))) - float(friction)) > 1.0e-12
            for name in ("static_friction", "dynamic_friction")
        ):
            raise EvidenceIntegrityError("Isaac sphere friction is not resolved exactly")
    else:
        values = pair.get("friction")
        if not isinstance(values, Sequence) or len(values) != 5 or any(
            abs(float(values[index]) - (float(friction) if index < 2 else 0.0))
            > 1.0e-12
            for index in range(5)
        ):
            raise EvidenceIntegrityError("MuJoCo sphere friction is not resolved exactly")
    return {
        "schema_version": "RootCauseSphereConfigurationSemanticsV2",
        "engine": engine,
        "source_identity": source_identity,
        "topology": resolved_semantics["topology"],
        "environment": resolved_semantics["environment"],
        "timing": {"physics_dt_s": float(physics_dt_s)},
        "engine_options": resolved_semantics["engine_options"],
        "static_model": resolved_semantics["static_model"],
        "contact": contact,
        "observer_mode": observer_mode,
    }


def _configuration_infinity_declaration(
    path: tuple[str | int, ...],
) -> DeclaredInfinitySpec | None:
    if path == (
        "engine_options",
        "resolved_physics_scene",
        "attributes",
        "physxScene:maxBiasCoefficient",
        "value",
    ):
        return declared_usd_infinity_spec(
            "/physicsScene", "physxScene:maxBiasCoefficient"
        )
    if (
        len(path) == 7
        and path[:2] == ("closure", "constraints")
        and isinstance(path[2], str)
        and path[2] in ISAAC_LOOP_CLOSURE_PRIM_PATHS
        and path[3:5] == ("resolved_prim", "attributes")
        and path[6] == "value"
    ):
        return declared_usd_infinity_spec(str(path[2]), str(path[5]))
    if (
        len(path) == 6
        and path[:4] == ("static_model", "coupon", "prim", "attributes")
        and path[5] == "value"
    ):
        return declared_usd_infinity_spec(
            "/World/envs/env_0/Coupon", str(path[4])
        )
    if (
        len(path) == 7
        and path[:4] == ("static_model", "coupon", "prim", "attributes")
        and path[4] == "physics:centerOfMass"
        and path[5] == "value"
        and path[6] in (0, 1, 2)
    ):
        return declared_usd_infinity_spec(
            "/World/envs/env_0/Coupon", "physics:centerOfMass"
        )
    return None


def _validate_identity_infinity_contract(
    *,
    engine: str,
    configuration_semantics: Mapping[str, Any],
    transform_semantics: Mapping[str, Any],
) -> None:
    resolver = (
        _configuration_infinity_declaration
        if engine == "isaac"
        else lambda path: None
    )
    validate_declared_infinity_payload(
        configuration_semantics,
        declaration_for_path=resolver,
        label=f"{engine} configuration_semantics",
    )
    validate_declared_infinity_payload(
        transform_semantics,
        declaration_for_path=lambda path: None,
        label=f"{engine} transform_semantics",
    )


def result_identity_fields(
    *,
    scenario_id: str,
    engine: str,
    source_model_sha256: str,
    model_artifact_sha256: str,
    transform_manifest_sha256: str,
    transform_semantics: Mapping[str, Any],
    worker_source_sha256: str,
    configuration_semantics: Mapping[str, Any],
    pre_forward_initial_condition_hash: str,
    post_forward_state_hash: str,
    reset_returned_policy_hash: str | None,
    excitation_hash: str,
    comparison_profile_hash: str | None,
) -> dict[str, Any]:
    _validate_identity_infinity_contract(
        engine=engine,
        configuration_semantics=configuration_semantics,
        transform_semantics=transform_semantics,
    )
    configuration_semantics_hash = stable_hash(configuration_semantics)
    transform_semantics_hash = stable_hash(transform_semantics)
    configuration_hash = stable_hash(
        {
            "engine": engine,
            "source_model_sha256": source_model_sha256,
            "model_artifact_sha256": model_artifact_sha256,
            "transform_manifest_sha256": transform_manifest_sha256,
            "transform_semantics_hash": transform_semantics_hash,
            "configuration_semantics_hash": configuration_semantics_hash,
            "worker_source_sha256": worker_source_sha256,
        }
    )
    repeatability_key = stable_hash(
        {
            "configuration_hash": configuration_hash,
            "pre_forward_initial_condition_hash": pre_forward_initial_condition_hash,
            "post_forward_state_hash": post_forward_state_hash,
            "reset_returned_policy_hash": reset_returned_policy_hash,
            "excitation_hash": excitation_hash,
        }
    )
    return {
        "scenario_id": scenario_id,
        "engine": engine,
        "source_model_sha256": source_model_sha256,
        "model_artifact_sha256": model_artifact_sha256,
        "transform_manifest_sha256": transform_manifest_sha256,
        "transform_semantics": dict(transform_semantics),
        "transform_semantics_hash": transform_semantics_hash,
        "worker_source_sha256": worker_source_sha256,
        "configuration_semantics": dict(configuration_semantics),
        "configuration_semantics_hash": configuration_semantics_hash,
        "configuration_hash": configuration_hash,
        "pre_forward_initial_condition_hash": pre_forward_initial_condition_hash,
        "post_forward_state_hash": post_forward_state_hash,
        "reset_returned_policy_hash": reset_returned_policy_hash,
        "excitation_hash": excitation_hash,
        "comparison_profile_hash": comparison_profile_hash,
        "repeatability_key": repeatability_key,
    }


def _json_pointer_diff(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: set[str] = set()
        for key in sorted(set(left) | set(right), key=str):
            token = str(key).replace("~", "~0").replace("/", "~1")
            child = f"{prefix}/{token}"
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(_json_pointer_diff(left[key], right[key], child))
        return paths
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes, bytearray))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes, bytearray))
    ):
        paths: set[str] = set()
        maximum = max(len(left), len(right))
        for index in range(maximum):
            child = f"{prefix}/{index}"
            if index >= len(left) or index >= len(right):
                paths.add(child)
            else:
                paths.update(_json_pointer_diff(left[index], right[index], child))
        return paths
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        return {prefix or "/"}
    return set()


def load_result(path: Path) -> dict[str, Any]:
    import json

    resolved = Path(path).resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceIntegrityError(f"Worker result is not an object: {resolved}")
    return payload


def validate_result_identity(payload: Mapping[str, Any]) -> None:
    required = (
        "scenario_id",
        "engine",
        "source_model_sha256",
        "model_artifact_sha256",
        "transform_manifest_sha256",
        "transform_semantics",
        "transform_semantics_hash",
        "worker_source_sha256",
        "configuration_semantics",
        "configuration_semantics_hash",
        "configuration_hash",
        "pre_forward_initial_condition_hash",
        "post_forward_state_hash",
        "excitation_hash",
        "repeatability_key",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise EvidenceIntegrityError(f"Result identity fields are missing: {missing}")
    semantics = payload["configuration_semantics"]
    transform_semantics = payload["transform_semantics"]
    if not isinstance(semantics, Mapping) or not isinstance(
        transform_semantics, Mapping
    ):
        raise EvidenceIntegrityError("Result semantic identities must be objects")
    _validate_identity_infinity_contract(
        engine=str(payload["engine"]),
        configuration_semantics=semantics,
        transform_semantics=transform_semantics,
    )
    if stable_hash(semantics) != payload["configuration_semantics_hash"]:
        raise EvidenceIntegrityError("Configuration semantics hash mismatch")
    if stable_hash(transform_semantics) != payload["transform_semantics_hash"]:
        raise EvidenceIntegrityError("Transform semantics hash mismatch")
    expected_configuration = stable_hash(
        {
            "engine": payload["engine"],
            "source_model_sha256": payload["source_model_sha256"],
            "model_artifact_sha256": payload["model_artifact_sha256"],
            "transform_manifest_sha256": payload["transform_manifest_sha256"],
            "transform_semantics_hash": payload["transform_semantics_hash"],
            "configuration_semantics_hash": payload["configuration_semantics_hash"],
            "worker_source_sha256": payload["worker_source_sha256"],
        }
    )
    if expected_configuration != payload["configuration_hash"]:
        raise EvidenceIntegrityError("Configuration identity hash mismatch")
    expected_repeatability = stable_hash(
        {
            "configuration_hash": payload["configuration_hash"],
            "pre_forward_initial_condition_hash": payload[
                "pre_forward_initial_condition_hash"
            ],
            "post_forward_state_hash": payload["post_forward_state_hash"],
            "reset_returned_policy_hash": payload.get("reset_returned_policy_hash"),
            "excitation_hash": payload["excitation_hash"],
        }
    )
    if expected_repeatability != payload["repeatability_key"]:
        raise EvidenceIntegrityError("Repeatability identity hash mismatch")


def validate_ratio_pair(
    baseline: Mapping[str, Any],
    ablation: Mapping[str, Any],
    *,
    pair_id: str,
) -> dict[str, Any]:
    validate_result_identity(baseline)
    validate_result_identity(ablation)
    baseline_spec = scenario_spec(str(baseline["scenario_id"]))
    ablation_spec = scenario_spec(str(ablation["scenario_id"]))
    if baseline_spec.ratio_pair_id != pair_id or ablation_spec.ratio_pair_id != pair_id:
        raise EvidenceIntegrityError(f"Results do not belong to ratio pair {pair_id}")
    if baseline_spec.baseline_scenario_id != baseline["scenario_id"]:
        raise EvidenceIntegrityError(f"Ratio baseline identity is reversed: {pair_id}")
    if baseline_spec.ablation_scenario_id != ablation["scenario_id"]:
        raise EvidenceIntegrityError(f"Ratio ablation identity is reversed: {pair_id}")
    if baseline["engine"] != ablation["engine"]:
        raise EvidenceIntegrityError("Ratio pair mixes engines")
    if baseline["source_model_sha256"] != ablation["source_model_sha256"]:
        raise EvidenceIntegrityError("Ratio pair source model identity differs")
    if baseline["worker_source_sha256"] != ablation["worker_source_sha256"]:
        raise EvidenceIntegrityError("Ratio pair worker source identity differs")
    if (
        baseline["pre_forward_initial_condition_hash"]
        != ablation["pre_forward_initial_condition_hash"]
    ):
        raise EvidenceIntegrityError("Ratio pair pre-forward reset identity differs")
    if pair_id != "P40_CLOSURE_ON_TO_OFF":
        for field in ("post_forward_state_hash", "reset_returned_policy_hash"):
            if baseline.get(field) != ablation.get(field):
                raise EvidenceIntegrityError(f"Ratio pair {field} differs")
    if baseline.get("comparison_profile_hash") != ablation.get(
        "comparison_profile_hash"
    ):
        raise EvidenceIntegrityError("Ratio comparison profile identity differs")
    excitation_differences: set[str] = set()
    if pair_id.startswith("P30_"):
        baseline_excitation = baseline.get("excitation_semantics")
        ablation_excitation = ablation.get("excitation_semantics")
        if not isinstance(baseline_excitation, Mapping) or not isinstance(
            ablation_excitation, Mapping
        ):
            raise EvidenceIntegrityError("P30 ratio excitation semantics are missing")
        if stable_hash(baseline_excitation) != baseline["excitation_hash"] or stable_hash(
            ablation_excitation
        ) != ablation["excitation_hash"]:
            raise EvidenceIntegrityError("P30 ratio excitation hash mismatch")
        raw_excitation_differences = _json_pointer_diff(
            baseline_excitation, ablation_excitation
        )
        excitation_differences = {
            "/actual_input_values"
            if path == "/actual_input_values"
            or path.startswith("/actual_input_values/")
            else path
            for path in raw_excitation_differences
        }
        expected_excitation_differences = {
            "/actual_input_values",
            "/input_kind",
            "/input_units",
        }
        if excitation_differences != expected_excitation_differences:
            raise EvidenceIntegrityError(
                "P30 ratio excitation diff is not exact: "
                f"actual={sorted(excitation_differences)}, "
                f"expected={sorted(expected_excitation_differences)}"
            )
    elif baseline["excitation_hash"] != ablation["excitation_hash"]:
        raise EvidenceIntegrityError("Ratio excitation identity differs")
    differences = _json_pointer_diff(
        baseline["configuration_semantics"], ablation["configuration_semantics"]
    )
    allowed = set(factor_path_allowlist(str(baseline["engine"]), pair_id))
    if differences != allowed:
        raise EvidenceIntegrityError(
            f"Ratio semantic diff is not exact for {pair_id}: "
            f"actual={sorted(differences)}, expected={sorted(allowed)}"
        )
    transform_differences = _json_pointer_diff(
        baseline["transform_semantics"], ablation["transform_semantics"]
    )
    if transform_differences != differences:
        raise EvidenceIntegrityError(
            f"Transform manifest diff is not compiled-semantic exact for {pair_id}: "
            f"transform={sorted(transform_differences)}, compiled={sorted(differences)}"
        )
    if baseline_spec.allowed_ablation_factors != ablation_spec.allowed_ablation_factors:
        raise EvidenceIntegrityError("Ratio pair factor declarations differ")
    return {
        "pair_id": pair_id,
        "engine": baseline["engine"],
        "baseline_scenario_id": baseline["scenario_id"],
        "ablation_scenario_id": ablation["scenario_id"],
        "baseline_configuration_hash": baseline["configuration_hash"],
        "ablation_configuration_hash": ablation["configuration_hash"],
        "baseline_configuration_semantics_hash": baseline[
            "configuration_semantics_hash"
        ],
        "ablation_configuration_semantics_hash": ablation[
            "configuration_semantics_hash"
        ],
        "baseline_pre_forward_initial_condition_hash": baseline[
            "pre_forward_initial_condition_hash"
        ],
        "ablation_pre_forward_initial_condition_hash": ablation[
            "pre_forward_initial_condition_hash"
        ],
        "baseline_post_forward_state_hash": baseline["post_forward_state_hash"],
        "ablation_post_forward_state_hash": ablation["post_forward_state_hash"],
        "baseline_reset_returned_policy_hash": baseline.get(
            "reset_returned_policy_hash"
        ),
        "ablation_reset_returned_policy_hash": ablation.get(
            "reset_returned_policy_hash"
        ),
        "baseline_excitation_hash": baseline["excitation_hash"],
        "ablation_excitation_hash": ablation["excitation_hash"],
        "excitation_diff_paths": sorted(excitation_differences),
        "excitation_diff_paths_sha256": stable_hash(
            sorted(excitation_differences)
        ),
        "comparison_profile_hash": baseline.get("comparison_profile_hash"),
        "semantic_diff_paths": sorted(differences),
        "semantic_diff_paths_sha256": stable_hash(sorted(differences)),
        "transform_diff_paths": sorted(transform_differences),
        "transform_diff_paths_sha256": stable_hash(sorted(transform_differences)),
        "factor_path_allowlist_sha256": stable_hash(
            factor_path_allowlist_artifact()
        ),
        "allowed_ablation_factors": list(baseline_spec.allowed_ablation_factors),
        "expected_improvement_direction": baseline_spec.expected_improvement_direction,
    }


def verify_file_binding(path: Path, expected_sha256: str, *, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise EvidenceIntegrityError(f"{label} hash mismatch: {actual} != {expected_sha256}")
