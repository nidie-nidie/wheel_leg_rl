from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

from .contracts import EvidenceIntegrityError, stable_hash


EXPECTED_ACTOR_SLICES = {
    "angular_velocity": [0, 3],
    "projected_gravity": [3, 6],
    "command": [6, 9],
    "leg_position_error": [9, 13],
    "joint_velocity": [13, 19],
    "previous_action": [19, 25],
}
EXPECTED_CANONICAL_JOINT_ORDER = [
    "jIJ",
    "jIO",
    "jAB",
    "jAG",
    "jwheel_left",
    "jwheel_right",
]
EXPECTED_NATIVE_TARGET_SIGNS = np.asarray(
    [1.0, 1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64
)
EXPECTED_CONTROL_DT_S = 0.020

G02_THRESHOLDS = {
    "reset_pre_position_max_abs": 2.0e-6,
    "reset_pre_velocity_max_abs": 1.0e-6,
    "reset_all_hinge_position_max_abs": 1.0e-5,
    "reset_post_position_max_abs": 2.0e-6,
    "reset_post_velocity_max_abs": 2.0e-6,
    "reset_post_vector_max_abs": 2.0e-5,
    "reset_post_orientation_geodesic_rad": 1.0e-5,
    "reset_post_height_abs_m": 2.0e-5,
    "reset_post_closure_abs_m": 5.0e-5,
    "actor_observation_max_abs": 2.0e-5,
    "cenet_actor_max_abs": 2.0e-5,
    "action_target_max_abs": 1.0e-4,
    "same_input_actor_max_abs": 1.0e-6,
    "clock_abs_s": 1.0e-10,
    "pulse_target_max_abs": 1.0e-8,
    "pulse_response_min_abs": 1.0e-12,
    "pulse_response_ratio_max": 20.0,
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceIntegrityError(f"G02 field {label} must be an object")
    return value


def _array(value: Any, *, label: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise EvidenceIntegrityError(f"G02 field {label} is not numeric") from error
    if array.shape != shape:
        raise EvidenceIntegrityError(
            f"G02 field {label} has shape {array.shape}, expected {shape}"
        )
    if not np.isfinite(array).all():
        raise EvidenceIntegrityError(f"G02 field {label} contains NaN or Inf")
    return array


def _scalar(value: Any, *, label: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise EvidenceIntegrityError(f"G02 field {label} is not numeric") from error
    if not math.isfinite(scalar):
        raise EvidenceIntegrityError(f"G02 field {label} contains NaN or Inf")
    return scalar


def _positive_integer(value: Any, *, label: str) -> int:
    scalar = _scalar(value, label=label)
    if not scalar.is_integer() or scalar <= 0.0:
        raise EvidenceIntegrityError(f"G02 field {label} must be a positive integer")
    return int(scalar)


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise EvidenceIntegrityError(
            f"G02 comparison shape mismatch: {left.shape} != {right.shape}"
        )
    return float(np.max(np.abs(left - right), initial=0.0))


def _quaternion_geodesic(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise EvidenceIntegrityError("G02 reset quaternion has zero norm")
    dot = float(np.dot(left / left_norm, right / right_norm))
    return float(2.0 * math.acos(min(1.0, max(-1.0, abs(dot)))))


def _metric(
    metrics: dict[str, Any],
    gates: dict[str, bool],
    name: str,
    value: float,
    threshold: float,
) -> None:
    passed = bool(value <= threshold)
    metrics[name] = {
        "value": value,
        "threshold": threshold,
        "passed": passed,
    }
    gates[name] = passed


def _require_schema(payload: Mapping[str, Any], *, engine: str) -> None:
    expected = (
        "RootCauseIsaacAdapterProbeV1"
        if engine == "isaac"
        else "RootCauseMujocoAdapterProbeV1"
    )
    if payload.get("schema_version") != expected or payload.get("engine") != engine:
        raise EvidenceIntegrityError(
            f"G02 {engine} probe identity is invalid: "
            f"schema={payload.get('schema_version')!r}, engine={payload.get('engine')!r}"
        )


def _history_gate(payload: Mapping[str, Any], *, engine: str) -> tuple[bool, np.ndarray]:
    phases = _mapping(payload.get("reset_phases"), label=f"{engine}.reset_phases")
    returned = _mapping(
        phases.get("returned_policy"), label=f"{engine}.reset_phases.returned_policy"
    )
    current = _array(
        returned.get("actor_obs_current"),
        label=f"{engine}.actor_obs_current",
        shape=(25,),
    ).astype(np.float32)
    history = _array(
        returned.get("policy_input"),
        label=f"{engine}.policy_input",
        shape=(125,),
    ).astype(np.float32)
    expected = np.repeat(current[None, :], 5, axis=0)
    return bool(np.array_equal(history.reshape(5, 25), expected)), current


def _contract_gate(payload: Mapping[str, Any], *, engine: str) -> bool:
    contract = _mapping(payload.get("contract"), label=f"{engine}.contract")
    return bool(
        contract.get("actor_slices") == EXPECTED_ACTOR_SLICES
        and contract.get("canonical_joint_order") == EXPECTED_CANONICAL_JOINT_ORDER
        and contract.get("history_layout") == "frame_major"
        and contract.get("history_length") == 5
        and contract.get("input_dimension") == 125
        and contract.get("runtime_action_clip") == [-1.0, 1.0]
        and contract.get("control_dt_s") == EXPECTED_CONTROL_DT_S
        and contract.get("normalization_schema") == "NormalizationV2"
        and contract.get("control_frame_schema") == "ControlFrameV1"
    )


def _clock_gate(
    payload: Mapping[str, Any],
    *,
    engine: str,
    metrics: dict[str, Any],
    gates: dict[str, bool],
) -> None:
    clock = _mapping(payload.get("clock"), label=f"{engine}.clock")
    physics_dt = _scalar(clock.get("physics_dt_s"), label=f"{engine}.physics_dt_s")
    steps = _positive_integer(
        clock.get("physics_steps_per_action"),
        label=f"{engine}.physics_steps_per_action",
    )
    control_dt = _scalar(clock.get("control_dt_s"), label=f"{engine}.control_dt_s")
    before = _scalar(clock.get("time_before_s"), label=f"{engine}.time_before_s")
    after = _scalar(clock.get("time_after_s"), label=f"{engine}.time_after_s")
    _metric(
        metrics,
        gates,
        f"{engine}_clock_declared_abs_s",
        abs(control_dt - EXPECTED_CONTROL_DT_S),
        G02_THRESHOLDS["clock_abs_s"],
    )
    _metric(
        metrics,
        gates,
        f"{engine}_clock_substep_product_abs_s",
        abs(physics_dt * steps - EXPECTED_CONTROL_DT_S),
        G02_THRESHOLDS["clock_abs_s"],
    )
    _metric(
        metrics,
        gates,
        f"{engine}_clock_elapsed_abs_s",
        abs((after - before) - EXPECTED_CONTROL_DT_S),
        G02_THRESHOLDS["clock_abs_s"],
    )
    gates[f"{engine}_target_refresh_first_substep"] = bool(
        clock.get("target_refresh_first_substep") is True
    )
    gates[f"{engine}_targets_constant_within_action"] = bool(
        clock.get("targets_constant_within_action") is True
    )


def _pulse_gate(
    isaac: Mapping[str, Any],
    mujoco: Mapping[str, Any],
    *,
    metrics: dict[str, Any],
    gates: dict[str, bool],
) -> None:
    isaac_amplitude = _scalar(isaac.get("pulse_amplitude"), label="isaac.pulse_amplitude")
    mujoco_amplitude = _scalar(
        mujoco.get("pulse_amplitude"), label="mujoco.pulse_amplitude"
    )
    if isaac_amplitude <= 0.0 or mujoco_amplitude <= 0.0:
        raise EvidenceIntegrityError("G02 pulse amplitude must be positive")
    gates["pulse_amplitude_identity"] = bool(isaac_amplitude == mujoco_amplitude)
    isaac_records = _mapping(isaac.get("pulse_records"), label="isaac.pulse_records")
    mujoco_records = _mapping(mujoco.get("pulse_records"), label="mujoco.pulse_records")
    expected_keys = {str(index) for index in range(6)}
    if set(isaac_records) != expected_keys or set(mujoco_records) != expected_keys:
        raise EvidenceIntegrityError("G02 pulse record set must contain exactly six channels")

    target_cross_max = 0.0
    target_expected_max = 0.0
    response_ratio_max = 1.0
    response_min_abs = math.inf
    target_direction_passed = True
    feedback_direction_passed = True
    reported_target_sign_consistency = True
    reported_velocity_sign_consistency = True
    reported_sign_mismatches: list[str] = []
    native_polarity_passed = True
    pulse_clock_passed = True
    expected_scales = np.asarray([0.35, 0.35, 0.35, 0.35, 25.0, 25.0])
    for channel in range(6):
        records = {
            "isaac": _mapping(
                isaac_records[str(channel)], label=f"isaac.pulse_records.{channel}"
            ),
            "mujoco": _mapping(
                mujoco_records[str(channel)], label=f"mujoco.pulse_records.{channel}"
            ),
        }
        odd_targets: dict[str, np.ndarray] = {}
        magnitudes: dict[str, float] = {}
        for engine, record in records.items():
            odd_target = _array(
                record.get("odd_target_response"),
                label=f"{engine}.pulse.{channel}.odd_target_response",
                shape=(6,),
            )
            odd_velocity = _array(
                record.get("odd_velocity_response"),
                label=f"{engine}.pulse.{channel}.odd_velocity_response",
                shape=(6,),
            )
            plus = _mapping(record.get("plus"), label=f"{engine}.pulse.{channel}.plus")
            minus = _mapping(record.get("minus"), label=f"{engine}.pulse.{channel}.minus")
            plus_native = _array(
                plus.get("target_engine_native"),
                label=f"{engine}.pulse.{channel}.plus.native_target",
                shape=(6,),
            )
            minus_native = _array(
                minus.get("target_engine_native"),
                label=f"{engine}.pulse.{channel}.minus.native_target",
                shape=(6,),
            )
            odd_native = 0.5 * (plus_native - minus_native)
            derived_target_sign = int(np.sign(odd_target[channel]))
            derived_velocity_sign = int(np.sign(odd_velocity[channel]))
            reported_target_sign = record.get("driven_channel_target_sign")
            reported_velocity_sign = record.get("driven_channel_velocity_sign")
            if reported_target_sign != derived_target_sign:
                reported_sign_mismatches.append(
                    f"{engine} channel {channel} target sign "
                    f"reported={reported_target_sign!r} derived={derived_target_sign}"
                )
            if reported_velocity_sign != derived_velocity_sign:
                reported_sign_mismatches.append(
                    f"{engine} channel {channel} velocity sign "
                    f"reported={reported_velocity_sign!r} derived={derived_velocity_sign}"
                )
            reported_target_sign_consistency = bool(
                reported_target_sign_consistency
                and reported_target_sign == derived_target_sign
            )
            reported_velocity_sign_consistency = bool(
                reported_velocity_sign_consistency
                and reported_velocity_sign == derived_velocity_sign
            )
            target_direction_passed = bool(
                target_direction_passed and derived_target_sign == 1
            )
            feedback_direction_passed = bool(
                feedback_direction_passed and derived_velocity_sign == 1
            )
            native_polarity_passed = native_polarity_passed and bool(
                int(np.sign(odd_native[channel]))
                == int(EXPECTED_NATIVE_TARGET_SIGNS[channel])
            )
            expected = expected_scales[channel] * isaac_amplitude
            target_expected_max = max(
                target_expected_max, abs(float(odd_target[channel]) - expected)
            )
            target_expected_max = max(
                target_expected_max,
                float(np.max(np.abs(np.delete(odd_target, channel)), initial=0.0)),
            )
            magnitude = abs(float(odd_velocity[channel]))
            response_min_abs = min(response_min_abs, magnitude)
            magnitudes[engine] = magnitude
            odd_targets[engine] = odd_target
            for branch_name, branch in (("plus", plus), ("minus", minus)):
                pulse_clock_passed = pulse_clock_passed and bool(
                    abs(
                        _scalar(
                            branch.get("control_time_delta_s"),
                            label=(
                                f"{engine}.pulse.{channel}.{branch_name}."
                                "control_time_delta_s"
                            ),
                        )
                        - EXPECTED_CONTROL_DT_S
                    )
                    <= G02_THRESHOLDS["clock_abs_s"]
                )
        target_cross_max = max(
            target_cross_max, _max_abs(odd_targets["isaac"], odd_targets["mujoco"])
        )
        smaller = min(magnitudes.values())
        larger = max(magnitudes.values())
        if smaller <= 0.0:
            response_ratio_max = math.inf
        else:
            response_ratio_max = max(response_ratio_max, larger / smaller)

    if reported_sign_mismatches:
        raise EvidenceIntegrityError(
            "G02 reported pulse sign contradicts raw-vector derivation: "
            + "; ".join(reported_sign_mismatches)
        )

    _metric(
        metrics,
        gates,
        "pulse_target_cross_engine_max_abs",
        target_cross_max,
        G02_THRESHOLDS["pulse_target_max_abs"],
    )
    _metric(
        metrics,
        gates,
        "pulse_target_expected_max_abs",
        target_expected_max,
        G02_THRESHOLDS["pulse_target_max_abs"],
    )
    gates["pulse_target_direction"] = target_direction_passed
    gates["pulse_feedback_direction"] = feedback_direction_passed
    gates["pulse_native_polarity"] = native_polarity_passed
    gates["pulse_clock"] = pulse_clock_passed
    gates["pulse_response_nonzero"] = bool(
        response_min_abs >= G02_THRESHOLDS["pulse_response_min_abs"]
    )
    gates["pulse_response_magnitude_ratio"] = bool(
        response_ratio_max <= G02_THRESHOLDS["pulse_response_ratio_max"]
    )
    metrics["pulse_response_min_abs"] = response_min_abs
    metrics["pulse_response_ratio_max"] = response_ratio_max
    metrics["pulse_reported_target_sign_consistency"] = (
        reported_target_sign_consistency
    )
    metrics["pulse_reported_velocity_sign_consistency"] = (
        reported_velocity_sign_consistency
    )


def _gate_group(name: str) -> str:
    if (
        name == "actor_observation_max_abs"
        or name.startswith("reset_post_forward_")
        or name.startswith("actor_chain_")
    ):
        return "post_forward_projection"
    if name in {
        "pulse_feedback_direction",
        "pulse_response_nonzero",
        "pulse_response_magnitude_ratio",
    }:
        return "plant_response"
    return "digital_chain"


def _gate_groups(gates: Mapping[str, bool]) -> dict[str, dict[str, Any]]:
    grouped = {
        "digital_chain": {},
        "post_forward_projection": {},
        "plant_response": {},
    }
    for name, passed in sorted(gates.items()):
        grouped[_gate_group(name)][name] = bool(passed)
    return {
        name: {
            "passed": all(values.values()),
            "failures": sorted(
                gate_name for gate_name, passed in values.items() if not passed
            ),
            "gates": values,
        }
        for name, values in grouped.items()
    }


def evaluate_adapter_gate(
    isaac_payload: Mapping[str, Any],
    mujoco_payload: Mapping[str, Any],
    *,
    same_input_actor_checks: Mapping[str, Any],
) -> dict[str, Any]:
    isaac = _mapping(isaac_payload, label="isaac")
    mujoco = _mapping(mujoco_payload, label="mujoco")
    _require_schema(isaac, engine="isaac")
    _require_schema(mujoco, engine="mujoco")
    gates: dict[str, bool] = {}
    metrics: dict[str, Any] = {}

    gates["isaac_contract"] = _contract_gate(isaac, engine="isaac")
    gates["mujoco_contract"] = _contract_gate(mujoco, engine="mujoco")
    gates["contract_identity"] = bool(isaac.get("contract") == mujoco.get("contract"))

    isaac_history_ok, isaac_current = _history_gate(isaac, engine="isaac")
    mujoco_history_ok, mujoco_current = _history_gate(mujoco, engine="mujoco")
    gates["isaac_history_five_current_frames"] = isaac_history_ok
    gates["mujoco_history_five_current_frames"] = mujoco_history_ok
    _metric(
        metrics,
        gates,
        "actor_observation_max_abs",
        _max_abs(isaac_current.astype(np.float64), mujoco_current.astype(np.float64)),
        G02_THRESHOLDS["actor_observation_max_abs"],
    )

    isaac_phases = _mapping(isaac.get("reset_phases"), label="isaac.reset_phases")
    mujoco_phases = _mapping(mujoco.get("reset_phases"), label="mujoco.reset_phases")
    for phase, position_threshold, velocity_threshold in (
        (
            "pre_forward",
            G02_THRESHOLDS["reset_pre_position_max_abs"],
            G02_THRESHOLDS["reset_pre_velocity_max_abs"],
        ),
        (
            "post_forward",
            G02_THRESHOLDS["reset_post_position_max_abs"],
            G02_THRESHOLDS["reset_post_velocity_max_abs"],
        ),
    ):
        isaac_state = _mapping(isaac_phases.get(phase), label=f"isaac.{phase}")
        mujoco_state = _mapping(mujoco_phases.get(phase), label=f"mujoco.{phase}")
        for field, shape, threshold in (
            ("controlled_position_canonical", (6,), position_threshold),
            ("controlled_velocity_canonical", (6,), velocity_threshold),
            (
                "all_hinge_position_named",
                (26,),
                G02_THRESHOLDS["reset_all_hinge_position_max_abs"],
            ),
            ("all_hinge_velocity_named", (26,), velocity_threshold),
        ):
            _metric(
                metrics,
                gates,
                f"reset_{phase}_{field}_max_abs",
                _max_abs(
                    _array(
                        isaac_state.get(field), label=f"isaac.{phase}.{field}", shape=shape
                    ),
                    _array(
                        mujoco_state.get(field), label=f"mujoco.{phase}.{field}", shape=shape
                    ),
                ),
                threshold,
            )

    isaac_post = _mapping(isaac_phases.get("post_forward"), label="isaac.post_forward")
    mujoco_post = _mapping(mujoco_phases.get("post_forward"), label="mujoco.post_forward")
    for field in (
        "base_linear_velocity_control",
        "base_angular_velocity_control",
        "projected_gravity",
    ):
        _metric(
            metrics,
            gates,
            f"reset_post_forward_{field}_max_abs",
            _max_abs(
                _array(isaac_post.get(field), label=f"isaac.post.{field}", shape=(3,)),
                _array(mujoco_post.get(field), label=f"mujoco.post.{field}", shape=(3,)),
            ),
            G02_THRESHOLDS["reset_post_vector_max_abs"],
        )
    _metric(
        metrics,
        gates,
        "reset_post_forward_orientation_geodesic_rad",
        _quaternion_geodesic(
            _array(
                isaac_post.get("base_orientation_control_wxyz"),
                label="isaac.post.orientation",
                shape=(4,),
            ),
            _array(
                mujoco_post.get("base_orientation_control_wxyz"),
                label="mujoco.post.orientation",
                shape=(4,),
            ),
        ),
        G02_THRESHOLDS["reset_post_orientation_geodesic_rad"],
    )
    _metric(
        metrics,
        gates,
        "reset_post_forward_base_height_abs_m",
        abs(
            _scalar(isaac_post.get("base_height"), label="isaac.post.base_height")
            - _scalar(mujoco_post.get("base_height"), label="mujoco.post.base_height")
        ),
        G02_THRESHOLDS["reset_post_height_abs_m"],
    )
    _metric(
        metrics,
        gates,
        "reset_post_forward_loop_closure_abs_m",
        abs(
            _scalar(
                isaac_post.get("loop_closure_error"), label="isaac.post.loop_closure_error"
            )
            - _scalar(
                mujoco_post.get("loop_closure_error"), label="mujoco.post.loop_closure_error"
            )
        ),
        G02_THRESHOLDS["reset_post_closure_abs_m"],
    )

    isaac_actor = _mapping(isaac.get("actor_chain"), label="isaac.actor_chain")
    mujoco_actor = _mapping(mujoco.get("actor_chain"), label="mujoco.actor_chain")
    for field, shape, threshold in (
        ("estimated_velocity", (3,), G02_THRESHOLDS["cenet_actor_max_abs"]),
        ("context_mu", (16,), G02_THRESHOLDS["cenet_actor_max_abs"]),
        ("context_logvar", (16,), G02_THRESHOLDS["cenet_actor_max_abs"]),
        ("raw_action", (6,), G02_THRESHOLDS["cenet_actor_max_abs"]),
        ("clipped_action", (6,), G02_THRESHOLDS["cenet_actor_max_abs"]),
        ("target_canonical", (6,), G02_THRESHOLDS["action_target_max_abs"]),
        ("target_engine_native", (6,), G02_THRESHOLDS["action_target_max_abs"]),
    ):
        _metric(
            metrics,
            gates,
            f"actor_chain_{field}_max_abs",
            _max_abs(
                _array(isaac_actor.get(field), label=f"isaac.actor.{field}", shape=shape),
                _array(mujoco_actor.get(field), label=f"mujoco.actor.{field}", shape=shape),
            ),
            threshold,
        )

    for engine, actor in (("isaac", isaac_actor), ("mujoco", mujoco_actor)):
        raw = _array(actor.get("raw_action"), label=f"{engine}.raw_action", shape=(6,))
        clipped = _array(
            actor.get("clipped_action"), label=f"{engine}.clipped_action", shape=(6,)
        )
        previous_before = _array(
            actor.get("previous_action_before"),
            label=f"{engine}.previous_action_before",
            shape=(6,),
        )
        previous_after = _array(
            actor.get("previous_action_after"),
            label=f"{engine}.previous_action_after",
            shape=(6,),
        )
        next_current = _array(
            actor.get("next_actor_obs_current"),
            label=f"{engine}.next_actor_obs_current",
            shape=(25,),
        )
        gates[f"{engine}_action_clipping"] = bool(
            _max_abs(np.clip(raw, -1.0, 1.0), clipped) <= 1.0e-7
        )
        gates[f"{engine}_previous_action_before_zero"] = bool(
            _max_abs(previous_before, np.zeros(6)) <= 1.0e-7
        )
        gates[f"{engine}_previous_action_after_matches_clipped"] = bool(
            _max_abs(previous_after, clipped) <= 1.0e-7
            and _max_abs(next_current[19:25], clipped) <= 1.0e-7
        )

    same_checks = _mapping(same_input_actor_checks, label="same_input_actor_checks")
    if set(same_checks) != {"dreamwaq_run01", "phase1r_run03"}:
        raise EvidenceIntegrityError("G02 same-input actor check set is incomplete")
    same_input_max = 0.0
    for name, record_value in same_checks.items():
        record = _mapping(record_value, label=f"same_input_actor_checks.{name}")
        same_input_max = max(
            same_input_max,
            _scalar(record.get("maximum_abs_error"), label=f"{name}.maximum_abs_error"),
        )
        if name == "dreamwaq_run01":
            gates["dreamwaq_golden_vector"] = bool(
                _scalar(
                    record.get("golden_vector_max_abs"),
                    label="dreamwaq_run01.golden_vector_max_abs",
                )
                <= G02_THRESHOLDS["same_input_actor_max_abs"]
            )
    _metric(
        metrics,
        gates,
        "same_input_actor_max_abs",
        same_input_max,
        G02_THRESHOLDS["same_input_actor_max_abs"],
    )

    _clock_gate(isaac, engine="isaac", metrics=metrics, gates=gates)
    _clock_gate(mujoco, engine="mujoco", metrics=metrics, gates=gates)
    _pulse_gate(isaac, mujoco, metrics=metrics, gates=gates)

    groups = _gate_groups(gates)
    digital_failures = groups["digital_chain"]["failures"]
    projection_failures = groups["post_forward_projection"]["failures"]
    plant_failures = groups["plant_response"]["failures"]
    deferred_failures = sorted([*projection_failures, *plant_failures])
    all_failures = sorted([*digital_failures, *deferred_failures])
    pre_failures = [
        name for name in digital_failures if name.startswith("reset_pre_forward")
    ]
    if pre_failures:
        classification = "SIM2SIM_ADAPTER_BUG/reset_serialization"
    elif digital_failures:
        classification = "SIM2SIM_ADAPTER_BUG/adapter_chain"
    elif projection_failures and plant_failures:
        classification = (
            "adapter_chain_equivalent/post_forward_and_plant_deferred"
        )
    elif projection_failures:
        classification = (
            "adapter_chain_equivalent/post_forward_projection_deferred"
        )
    elif plant_failures:
        classification = "adapter_chain_equivalent/plant_response_deferred"
    else:
        classification = "adapter_chain_equivalent"
    digital_chain_passed = bool(groups["digital_chain"]["passed"])
    result = {
        "schema_version": "RootCauseAdapterGateV1",
        "passed": digital_chain_passed,
        "evidence_valid": True,
        "digital_chain_passed": digital_chain_passed,
        "post_forward_projection_equivalent": bool(
            groups["post_forward_projection"]["passed"]
        ),
        "plant_response_equivalent": bool(groups["plant_response"]["passed"]),
        "terminal_adapter_bug": not digital_chain_passed,
        "classification": classification,
        "gates": gates,
        "gate_groups": groups,
        "failures": digital_failures,
        "deferred_failures": deferred_failures,
        "all_failures": all_failures,
        "metrics": metrics,
        "thresholds": dict(G02_THRESHOLDS),
        "isaac_evidence_hash": stable_hash(isaac),
        "mujoco_evidence_hash": stable_hash(mujoco),
        "same_input_actor_checks_hash": stable_hash(same_checks),
    }
    result["identity_hash"] = stable_hash(result)
    return result
