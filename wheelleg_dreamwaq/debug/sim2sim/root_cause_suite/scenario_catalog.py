from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Iterable, Sequence

from .contracts import StageName, canonical_json_bytes


RATIO_PAIR_IDS = {
    "P30_OPEN_TARGET_TO_DIRECT",
    "P30_CLOSED_TARGET_TO_DIRECT",
    "P40_CLOSURE_ON_TO_OFF",
    "P50_NOMINAL_TO_ZERO_FRICTION",
}

EXTENDED_PREFIXES = ("P20_D", "P40_C", "P50_B", "P50_D", "P60_A", "P60_E")
SCALAR_METRICS = {"rmse", "normalized_rmse", "max_abs", "event_delta"}
CHANNEL_REDUCTIONS = {"named_channel", "l2", "max", "identity", "none"}
ABLATION_FACTORS = {"drive_mode", "input_kind", "closure_mode", "friction"}

ISAAC_CLOSURE_CONSTRAINTS = (
    "/World/envs/env_0/Robot/jIO/jIO_loop_closure",
    "/World/envs/env_0/Robot/jKN/jKN_loop_closure",
    "/World/envs/env_0/Robot/jEC/jAG_loop_closure",
    "/World/envs/env_0/Robot/jCF/jCF_revolute_joint",
)
FACTOR_PATH_ALLOWLIST_PATH = Path(__file__).with_name("factor_path_allowlist.json")


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _generated_factor_path_allowlist(engine: str, pair_id: str) -> tuple[str, ...]:

    if engine not in {"isaac", "mujoco"}:
        raise ValueError(f"Unknown causal engine: {engine}")
    if pair_id.startswith("P30_"):
        from .variant_builders import ROBOT_HINGES

        mujoco_controlled_hinges = {
            "jIJ",
            "jIO",
            "jAB",
            "jAG",
            "jwheel_left",
            "jwheel_right",
        }
        paths = {
            "/actuation/drive_mode",
            "/actuation/external_controller_enabled",
            "/actuation/target_neutralization",
        }
        for name in ROBOT_HINGES:
            base = f"/actuation/robot_hinges/{_pointer_token(name)}"
            if engine == "isaac":
                paths.add(f"{base}/damping")
                paths.add(f"{base}/stiffness")
            elif name not in mujoco_controlled_hinges:
                paths.add(f"{base}/dof_damping")
        return tuple(sorted(paths))
    if pair_id == "P40_CLOSURE_ON_TO_OFF":
        if engine == "isaac":
            names = ISAAC_CLOSURE_CONSTRAINTS
        else:
            from .variant_builders import CONNECT_NAMES

            names = CONNECT_NAMES
        return tuple(
            sorted(
                {
                    "/closure/mode",
                    *(
                        f"/closure/constraints/{_pointer_token(name)}/enabled"
                        for name in names
                    ),
                }
            )
        )
    if pair_id == "P50_NOMINAL_TO_ZERO_FRICTION":
        if engine == "isaac":
            return (
                "/contact/pairs/floor_coupon/material/dynamic_friction",
                "/contact/pairs/floor_coupon/material/static_friction",
            )
        return tuple(
            f"/contact/pairs/floor_coupon/friction/{index}" for index in range(2)
        )
    raise ValueError(f"Unknown causal pair: {pair_id}")


def _generated_factor_path_allowlist_artifact() -> dict[str, object]:
    pairs = {
        pair_id: {
            engine: list(_generated_factor_path_allowlist(engine, pair_id))
            for engine in ("isaac", "mujoco")
        }
        for pair_id in sorted(RATIO_PAIR_IDS)
    }
    return {
        "schema_version": "RootCauseFactorPathAllowlistV1",
        "pairs": pairs,
    }


def factor_path_allowlist_artifact() -> dict[str, object]:
    if not FACTOR_PATH_ALLOWLIST_PATH.is_file():
        raise ValueError(
            f"Frozen factor-path allow-list is missing: {FACTOR_PATH_ALLOWLIST_PATH}"
        )
    payload = json.loads(FACTOR_PATH_ALLOWLIST_PATH.read_text(encoding="utf-8"))
    expected = _generated_factor_path_allowlist_artifact()
    if payload != expected:
        raise ValueError("Frozen factor-path allow-list differs from the code contract")
    if FACTOR_PATH_ALLOWLIST_PATH.read_bytes() != canonical_json_bytes(payload) + b"\n":
        raise ValueError("Frozen factor-path allow-list is not canonical JSON")
    return payload


def factor_path_allowlist(engine: str, pair_id: str) -> tuple[str, ...]:
    """Return the frozen, engine-specific compiled-semantic factor paths."""

    payload = factor_path_allowlist_artifact()
    try:
        paths = payload["pairs"][pair_id][engine]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Unknown causal factor projection: engine={engine}, pair={pair_id}"
        ) from error
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError("Frozen factor-path allow-list entry is invalid")
    return tuple(paths)


def robot_probe_profiles(
    scenario: str,
    *,
    repetitions: int,
    amplitude: float | None = None,
    input_vector: Sequence[float] | None = None,
) -> list[dict[str, object]]:
    """Build the frozen, engine-independent Core robot excitation matrix."""

    if repetitions < 3:
        raise ValueError("Verdict-bearing robot probes require at least three repetitions")
    profiles: list[dict[str, object]] = []
    if scenario.startswith("p30"):
        direct = scenario.endswith("direct")
        for channel in range(6):
            default = (
                0.3
                if direct
                else (0.3 / (120.0 * 0.35) if channel < 4 else 0.3 / (0.6 * 25.0))
            )
            magnitude = default if amplitude is None else float(amplitude)
            for sign in (1, 0, -1):
                values = [0.0] * 6
                values[channel] = sign * magnitude
                torque = [sign * 0.3 if index == channel else 0.0 for index in range(6)]
                for repetition in range(repetitions):
                    profiles.append(
                        {
                            "channel": channel,
                            "sign": sign,
                            "repetition": repetition,
                            "actual_input": values,
                            "canonical_torque_equivalent": torque,
                        }
                    )
        return profiles
    if scenario.startswith("p40"):
        magnitude = 0.3 if amplitude is None else float(amplitude)
        for sign in (1, 0, -1):
            values = [0.0] * 6
            values[0] = sign * magnitude
            values[2] = -sign * magnitude
            for repetition in range(repetitions):
                profiles.append(
                    {
                        "channel": -1,
                        "sign": sign,
                        "repetition": repetition,
                        "actual_input": values,
                        "canonical_torque_equivalent": values,
                    }
                )
        return profiles
    if scenario == "p60_c":
        if input_vector is None:
            magnitude = 0.0 if amplitude is None else float(amplitude)
            values = [magnitude, 0.0, -magnitude, 0.0, 0.0, 0.0]
        else:
            values = [float(value) for value in input_vector]
            if len(values) != 6:
                raise ValueError("P60-C shared first action must contain six values")
            if amplitude is not None:
                raise ValueError("P60-C cannot combine an explicit input vector with amplitude")
    else:
        values = [0.0] * 6
    for repetition in range(repetitions):
        profiles.append(
            {
                "channel": -1,
                "sign": 0 if not any(values) else 1,
                "repetition": repetition,
                "actual_input": values,
                "canonical_torque_equivalent": values,
            }
        )
    return profiles


def sphere_probe_profiles(
    mode: str,
    *,
    repetitions: int,
) -> list[dict[str, object]]:
    """Build the frozen common-sphere initial-condition matrix."""

    if repetitions < 3:
        raise ValueError("Verdict-bearing sphere probes require at least three repetitions")
    profiles: list[dict[str, object]] = []
    if mode == "impact":
        for height_m in (0.05, 0.10, 0.20):
            for vertical_velocity_mps in (0.0, -0.5):
                for repetition in range(repetitions):
                    profiles.append(
                        {
                            "height_m": height_m,
                            "vertical_velocity_mps": vertical_velocity_mps,
                            "horizontal_velocity_mps": 0.0,
                            "sign": 0,
                            "repetition": repetition,
                        }
                    )
        return profiles
    if mode == "slide":
        for sign in (1, 0, -1):
            for repetition in range(repetitions):
                profiles.append(
                    {
                        "height_m": 0.0,
                        "vertical_velocity_mps": 0.0,
                        "horizontal_velocity_mps": float(sign),
                        "sign": sign,
                        "repetition": repetition,
                    }
                )
        return profiles
    raise ValueError(f"Unknown sphere probe mode: {mode}")


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    stage: StageName
    description: str
    primary_signals: tuple[str, ...]
    comparison_window_ms: tuple[int, int] | None
    scalar_metric: str
    channel_reduction: str
    normalization_scale: float | None
    guard_metrics: tuple[str, ...]
    repeatability_family: str | None
    ratio_pair_id: str | None = None
    baseline_scenario_id: str | None = None
    ablation_scenario_id: str | None = None
    expected_improvement_direction: str | None = None
    allowed_ablation_factors: tuple[str, ...] = ()
    allowed_semantic_paths: tuple[str, ...] = ()
    replay_source_spec: str | None = None

    def with_allowed_paths(self, paths: Iterable[str]) -> "ScenarioSpec":
        return replace(self, allowed_semantic_paths=tuple(paths))


def _identity(
    scenario_id: str, stage: StageName, description: str, signal: str = "identity"
) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=scenario_id,
        stage=stage,
        description=description,
        primary_signals=(signal,),
        comparison_window_ms=None,
        scalar_metric="max_abs",
        channel_reduction="identity",
        normalization_scale=None,
        guard_metrics=(),
        repeatability_family=None,
    )


def catalog() -> tuple[ScenarioSpec, ...]:
    def all_engine_paths(pair_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(factor_path_allowlist("isaac", pair_id))
                | set(factor_path_allowlist("mujoco", pair_id))
            )
        )

    drive_paths = all_engine_paths("P30_OPEN_TARGET_TO_DIRECT")
    closure_paths = all_engine_paths("P40_CLOSURE_ON_TO_OFF")
    friction_paths = all_engine_paths("P50_NOMINAL_TO_ZERO_FRICTION")
    scenarios = (
        _identity("G00_IDENTITY", StageName.G00_INTEGRITY, "Frozen identity and scope gate"),
        _identity("G01_REPEATABILITY", StageName.G01_REPEATABILITY, "Within-engine repeatability gate"),
        _identity("G02_ADAPTER", StageName.G02_ADAPTER, "Observation, history, actor, and adapter equivalence"),
        _identity("G03_INSTRUMENTATION", StageName.G03_INSTRUMENTATION, "Observer neutrality gate"),
        ScenarioSpec(
            "P10_A_REST_CLOSURE_ON", StageName.P10_REST,
            "No ground, zero gravity, drive off, closure on", ("system_momentum", "kinetic_energy"),
            (0, 100), "max_abs", "max", 1.0, ("closure_residual",), "p10_rest_closure_on",
        ),
        ScenarioSpec(
            "P10_B_FREEFALL_CLOSURE_ON", StageName.P10_REST,
            "No ground, gravity on, drive off, closure on", ("com_acceleration_z",),
            (0, 100), "rmse", "named_channel", 9.81, ("horizontal_com_drift",),
            "p10_freefall_closure_on",
        ),
        ScenarioSpec(
            "P10_C_REST_CLOSURE_OFF", StageName.P10_REST,
            "No ground, zero gravity, drive off, closure off", ("system_momentum", "kinetic_energy"),
            (0, 100), "max_abs", "max", 1.0, (), "p10_rest_closure_off",
        ),
        _identity(
            "P20_S_STATIC_PROPERTIES", StageName.P20_STATIC_PROPERTIES,
            "Compiled mass, COM, inertia, and momentum golden", "compiled_body_properties",
        ),
        ScenarioSpec(
            "P30_A_OPEN_DIRECT_EFFORT", StageName.P30_ACTUATOR,
            "Fixed base, open chain, direct effort", ("odd_response.velocity",),
            (0, 100), "normalized_rmse", "l2", 1.0,
            ("drive_bypass", "saturation"), "p30_open_direct",
            "P30_OPEN_TARGET_TO_DIRECT", "P30_B_OPEN_FORMAL_TARGET", "P30_A_OPEN_DIRECT_EFFORT",
            "lower", ("drive_mode", "input_kind"), drive_paths,
        ),
        ScenarioSpec(
            "P30_B_OPEN_FORMAL_TARGET", StageName.P30_ACTUATOR,
            "Fixed base, open chain, formal target", ("odd_response.velocity",),
            (0, 100), "normalized_rmse", "l2", 1.0,
            ("saturation",), "p30_open_target",
            "P30_OPEN_TARGET_TO_DIRECT", "P30_B_OPEN_FORMAL_TARGET", "P30_A_OPEN_DIRECT_EFFORT",
            "lower", ("drive_mode", "input_kind"), drive_paths,
        ),
        ScenarioSpec(
            "P30_C_CLOSED_DIRECT_EFFORT", StageName.P30_ACTUATOR,
            "Fixed base, closure on, direct effort", ("odd_response.velocity",),
            (0, 100), "normalized_rmse", "l2", 1.0,
            ("drive_bypass", "saturation"), "p30_closed_direct",
            "P30_CLOSED_TARGET_TO_DIRECT", "P30_C_CLOSED_FORMAL_TARGET", "P30_C_CLOSED_DIRECT_EFFORT",
            "lower", ("drive_mode", "input_kind"), drive_paths,
        ),
        ScenarioSpec(
            "P30_C_CLOSED_FORMAL_TARGET", StageName.P30_ACTUATOR,
            "Fixed base, closure on, formal target", ("odd_response.velocity",),
            (0, 100), "normalized_rmse", "l2", 1.0,
            ("saturation",), "p30_closed_target",
            "P30_CLOSED_TARGET_TO_DIRECT", "P30_C_CLOSED_FORMAL_TARGET", "P30_C_CLOSED_DIRECT_EFFORT",
            "lower", ("drive_mode", "input_kind"), drive_paths,
        ),
        ScenarioSpec(
            "P40_A_RELAXATION", StageName.P40_CLOSURE,
            "Zero-input closure relaxation reuse of P10", ("closure_effect",),
            (0, 100), "normalized_rmse", "l2", 1.0,
            ("no_contact",), "p10_rest_closure_on",
        ),
        ScenarioSpec(
            "P40_B_CLOSURE_ON", StageName.P40_CLOSURE,
            "Symmetric effort pulse with closure on", ("odd_response.velocity",),
            (0, 100), "normalized_rmse", "l2", 1.0,
            ("no_contact", "mechanical_response"), "p40_closure_on_pulse",
            "P40_CLOSURE_ON_TO_OFF", "P40_B_CLOSURE_ON", "P40_B_CLOSURE_OFF",
            "lower", ("closure_mode",), closure_paths,
        ),
        ScenarioSpec(
            "P40_B_CLOSURE_OFF", StageName.P40_CLOSURE,
            "Symmetric effort pulse with closure off", ("odd_response.velocity",),
            (0, 100), "normalized_rmse", "l2", 1.0,
            ("no_contact", "mechanical_response"), "p40_closure_off_pulse",
            "P40_CLOSURE_ON_TO_OFF", "P40_B_CLOSURE_ON", "P40_B_CLOSURE_OFF",
            "lower", ("closure_mode",), closure_paths,
        ),
        ScenarioSpec(
            "P50_A_SPHERE_IMPACT", StageName.P50_CONTACT,
            "Common sphere frictionless vertical impact", ("com_velocity_z", "first_contact_time"),
            (-20, 100), "rmse", "named_channel", 1.0,
            ("coupon_identity", "freefall", "valid_contact"), "p50_sphere_impact",
        ),
        ScenarioSpec(
            "P50_C_NOMINAL_FRICTION", StageName.P50_CONTACT,
            "Common sphere tangential slide with nominal friction", ("com_velocity_x",),
            (0, 400), "normalized_rmse", "named_channel", 1.0,
            ("valid_contact", "normal_gate"), "p50_sphere_slide_nominal_friction",
            "P50_NOMINAL_TO_ZERO_FRICTION", "P50_C_NOMINAL_FRICTION", "P50_C_ZERO_FRICTION",
            "lower", ("friction",), friction_paths,
        ),
        ScenarioSpec(
            "P50_C_ZERO_FRICTION", StageName.P50_CONTACT,
            "Common sphere tangential slide with zero friction", ("com_velocity_x",),
            (0, 400), "normalized_rmse", "named_channel", 1.0,
            ("valid_contact", "normal_gate"), "p50_sphere_slide_zero_friction",
            "P50_NOMINAL_TO_ZERO_FRICTION", "P50_C_NOMINAL_FRICTION", "P50_C_ZERO_FRICTION",
            "lower", ("friction",), friction_paths,
        ),
        ScenarioSpec(
            "P60_B_ZERO_ACTION_DRIVE_OFF", StageName.P60_FULL_ROBOT,
            "Full robot, ground and closure on, drive off", ("base_angular_velocity_y",),
            (0, 100), "normalized_rmse", "named_channel", 1.0,
            ("termination",), "p60_zero_action_drive_off",
        ),
        ScenarioSpec(
            "P60_C_SHARED_FIRST_ACTION", StageName.P60_FULL_ROBOT,
            "Full robot formal drive with a shared first action", ("canonical_joint_velocity",),
            (0, 100), "normalized_rmse", "l2", 1.0,
            ("action_identity", "termination"), "p60_first_action_formal_drive",
        ),
        ScenarioSpec(
            "P60_D_OPEN_LOOP_REPLAY", StageName.P60_FULL_ROBOT,
            "Frozen 499-action DreamWaQ run-01 open-loop replay", ("base_angular_velocity_y",),
            (0, 100), "normalized_rmse", "named_channel", 1.0,
            ("replay_identity", "fresh_reset_equivalence", "termination"), "p60_open_loop_replay",
            replay_source_spec="dreamwaq_run01:nominal_stand:seed=20261007:repetition=0:env=0:horizon=499",
        ),
        ScenarioSpec(
            "C70_CHECKPOINT_SENSITIVITY", StageName.C70_CHECKPOINT,
            "Dual-anchor checkpoint sensitivity on the sealed replay", ("action_rms_gain",),
            (0, 100), "rmse", "l2", 1.0,
            ("replay_identity", "feature_eligibility", "finite_actions"), None,
            replay_source_spec="P60_D_OPEN_LOOP_REPLAY",
        ),
    )
    validate_catalog(scenarios)
    return scenarios


def validate_catalog(scenarios: Iterable[ScenarioSpec]) -> None:
    items = tuple(scenarios)
    if not items:
        raise ValueError("Scenario catalog is empty")
    identifiers: set[str] = set()
    ratio_members: dict[str, set[str]] = {pair: set() for pair in RATIO_PAIR_IDS}
    for scenario in items:
        if scenario.scenario_id in identifiers:
            raise ValueError(f"Duplicate scenario: {scenario.scenario_id}")
        identifiers.add(scenario.scenario_id)
        if scenario.scenario_id.startswith(EXTENDED_PREFIXES):
            raise ValueError(f"Extended-only scenario in Core catalog: {scenario.scenario_id}")
        if not isinstance(scenario.stage, StageName):
            raise ValueError(f"Unknown stage for {scenario.scenario_id}")
        if not scenario.primary_signals or any(not signal for signal in scenario.primary_signals):
            raise ValueError(f"Missing primary signal for {scenario.scenario_id}")
        if scenario.scalar_metric not in SCALAR_METRICS:
            raise ValueError(f"Unknown scalar metric for {scenario.scenario_id}")
        if scenario.channel_reduction not in CHANNEL_REDUCTIONS:
            raise ValueError(f"Unknown channel reduction for {scenario.scenario_id}")
        if any("*" in path for path in scenario.allowed_semantic_paths):
            raise ValueError(f"Semantic path wildcard is forbidden: {scenario.scenario_id}")
        factors = tuple(sorted(scenario.allowed_ablation_factors))
        if factors != scenario.allowed_ablation_factors:
            raise ValueError(f"Ablation factors must be sorted: {scenario.scenario_id}")
        if not set(factors).issubset(ABLATION_FACTORS):
            raise ValueError(f"Unknown ablation factor: {scenario.scenario_id}")
        if scenario.ratio_pair_id is None:
            if any(
                value is not None
                for value in (
                    scenario.baseline_scenario_id,
                    scenario.ablation_scenario_id,
                    scenario.expected_improvement_direction,
                )
            ) or factors or scenario.allowed_semantic_paths:
                raise ValueError(f"Unpaired scenario has ratio metadata: {scenario.scenario_id}")
        else:
            if scenario.ratio_pair_id not in RATIO_PAIR_IDS:
                raise ValueError(f"Unknown ratio pair: {scenario.ratio_pair_id}")
            if scenario.expected_improvement_direction != "lower":
                raise ValueError(f"Ratio direction must be lower: {scenario.scenario_id}")
            if not scenario.baseline_scenario_id or not scenario.ablation_scenario_id:
                raise ValueError(f"Ratio identities are incomplete: {scenario.scenario_id}")
            if not factors or not scenario.allowed_semantic_paths:
                raise ValueError(f"Ratio allow-list is incomplete: {scenario.scenario_id}")
            ratio_members[scenario.ratio_pair_id].add(scenario.scenario_id)
    for pair_id, members in ratio_members.items():
        if len(members) != 2:
            raise ValueError(f"Ratio pair {pair_id} must bind exactly two scenarios")
        sample = next(item for item in items if item.scenario_id in members)
        expected = {sample.baseline_scenario_id, sample.ablation_scenario_id}
        if members != expected:
            raise ValueError(f"Ratio pair {pair_id} members do not match its identities")
