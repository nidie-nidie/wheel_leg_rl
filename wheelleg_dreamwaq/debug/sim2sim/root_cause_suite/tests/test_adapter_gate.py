from __future__ import annotations

from copy import deepcopy

import pytest

from debug.sim2sim.root_cause_suite.adapter_gate import (
    EXPECTED_ACTOR_SLICES,
    EXPECTED_CANONICAL_JOINT_ORDER,
    evaluate_adapter_gate,
)
from debug.sim2sim.root_cause_suite.contracts import EvidenceIntegrityError


def _contract() -> dict:
    return {
        "actor_slices": EXPECTED_ACTOR_SLICES,
        "canonical_joint_order": EXPECTED_CANONICAL_JOINT_ORDER,
        "history_layout": "frame_major",
        "history_length": 5,
        "input_dimension": 125,
        "runtime_action_clip": [-1.0, 1.0],
        "control_dt_s": 0.02,
        "normalization_schema": "NormalizationV2",
        "control_frame_schema": "ControlFrameV1",
    }


def _pulse_records(amplitude: float) -> dict[str, dict]:
    records = {}
    scales = [0.35, 0.35, 0.35, 0.35, 25.0, 25.0]
    native_signs = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
    for channel in range(6):
        odd_target = [0.0] * 6
        odd_target[channel] = scales[channel] * amplitude
        plus_native = [0.0] * 6
        minus_native = [0.0] * 6
        plus_native[channel] = native_signs[channel] * odd_target[channel]
        minus_native[channel] = -native_signs[channel] * odd_target[channel]
        odd_velocity = [0.0] * 6
        odd_velocity[channel] = 1.0e-4
        records[str(channel)] = {
            "plus": {
                "target_engine_native": plus_native,
                "control_time_delta_s": 0.02,
            },
            "minus": {
                "target_engine_native": minus_native,
                "control_time_delta_s": 0.02,
            },
            "zero": {},
            "odd_velocity_response": odd_velocity,
            "odd_target_response": odd_target,
            "driven_channel_velocity_sign": 1,
            "driven_channel_target_sign": 1,
        }
    return records


def _probe(engine: str) -> dict:
    current = [0.0] * 25
    current[5] = -1.0
    post = {
        "controlled_position_canonical": [0.0] * 6,
        "controlled_velocity_canonical": [0.0] * 6,
        "all_hinge_position_named": [0.0] * 26,
        "all_hinge_velocity_named": [0.0] * 26,
        "base_orientation_control_wxyz": [1.0, 0.0, 0.0, 0.0],
        "base_linear_velocity_control": [0.0] * 3,
        "base_angular_velocity_control": [0.0] * 3,
        "projected_gravity": [0.0, 0.0, -1.0],
        "base_height": 0.2,
        "loop_closure_error": 0.0,
    }
    pre = {
        key: value
        for key, value in post.items()
        if key
        in {
            "controlled_position_canonical",
            "controlled_velocity_canonical",
            "all_hinge_position_named",
            "all_hinge_velocity_named",
        }
    }
    physics_dt = 0.005 if engine == "isaac" else 0.001
    steps = 4 if engine == "isaac" else 20
    return {
        "schema_version": (
            "RootCauseIsaacAdapterProbeV1"
            if engine == "isaac"
            else "RootCauseMujocoAdapterProbeV1"
        ),
        "engine": engine,
        "contract": _contract(),
        "reset_phases": {
            "pre_forward": pre,
            "post_forward": post,
            "returned_policy": {
                "actor_obs_current": current,
                "policy_input": current * 5,
                "previous_action": [0.0] * 6,
            },
        },
        "actor_chain": {
            "estimated_velocity": [0.0] * 3,
            "context_mu": [0.0] * 16,
            "context_logvar": [0.0] * 16,
            "raw_action": [0.0] * 6,
            "clipped_action": [0.0] * 6,
            "target_canonical": [0.0] * 6,
            "target_engine_native": [0.0] * 6,
            "previous_action_before": [0.0] * 6,
            "previous_action_after": [0.0] * 6,
            "next_actor_obs_current": current,
        },
        "clock": {
            "physics_dt_s": physics_dt,
            "physics_steps_per_action": steps,
            "control_dt_s": 0.02,
            "time_before_s": 1.0,
            "time_after_s": 1.02,
            "target_refresh_first_substep": True,
            "targets_constant_within_action": True,
        },
        "pulse_amplitude": 1.0e-3,
        "pulse_records": _pulse_records(1.0e-3),
    }


def _same_input_checks() -> dict:
    return {
        "dreamwaq_run01": {
            "maximum_abs_error": 0.0,
            "golden_vector_max_abs": 0.0,
        },
        "phase1r_run03": {"maximum_abs_error": 0.0},
    }


def test_valid_dual_engine_adapter_gate_passes() -> None:
    result = evaluate_adapter_gate(
        _probe("isaac"),
        _probe("mujoco"),
        same_input_actor_checks=_same_input_checks(),
    )
    assert result["passed"] is True
    assert result["failures"] == []
    assert result["digital_chain_passed"] is True
    assert result["post_forward_projection_equivalent"] is True
    assert result["plant_response_equivalent"] is True
    assert result["terminal_adapter_bug"] is False


def test_missing_dual_engine_evidence_fails_closed() -> None:
    mujoco = _probe("mujoco")
    del mujoco["pulse_records"]
    with pytest.raises(EvidenceIntegrityError, match="pulse_records"):
        evaluate_adapter_gate(
            _probe("isaac"),
            mujoco,
            same_input_actor_checks=_same_input_checks(),
        )


def test_history_not_five_current_frames_is_rejected() -> None:
    isaac = _probe("isaac")
    isaac["reset_phases"]["returned_policy"]["policy_input"][0] = 1.0
    result = evaluate_adapter_gate(
        isaac,
        _probe("mujoco"),
        same_input_actor_checks=_same_input_checks(),
    )
    assert result["passed"] is False
    assert "isaac_history_five_current_frames" in result["failures"]
    assert result["terminal_adapter_bug"] is True


def test_previous_action_one_tick_lag_is_rejected() -> None:
    isaac = _probe("isaac")
    isaac["actor_chain"]["previous_action_after"][0] = 0.25
    result = evaluate_adapter_gate(
        isaac,
        _probe("mujoco"),
        same_input_actor_checks=_same_input_checks(),
    )
    assert result["passed"] is False
    assert "isaac_previous_action_after_matches_clipped" in result["failures"]


def test_reported_velocity_sign_contradiction_fails_closed() -> None:
    mujoco = _probe("mujoco")
    mujoco["pulse_records"]["5"]["driven_channel_velocity_sign"] = -1
    with pytest.raises(EvidenceIntegrityError, match="velocity sign"):
        evaluate_adapter_gate(
            _probe("isaac"),
            mujoco,
            same_input_actor_checks=_same_input_checks(),
        )


def test_reported_target_sign_contradiction_fails_closed() -> None:
    isaac = _probe("isaac")
    isaac["pulse_records"]["2"]["driven_channel_target_sign"] = -1
    with pytest.raises(EvidenceIntegrityError, match="target sign"):
        evaluate_adapter_gate(
            isaac,
            _probe("mujoco"),
            same_input_actor_checks=_same_input_checks(),
        )


def test_raw_feedback_polarity_cannot_be_hidden_by_reported_sign() -> None:
    mujoco = _probe("mujoco")
    mujoco["pulse_records"]["5"]["odd_velocity_response"][5] = -1.0e-4
    assert mujoco["pulse_records"]["5"]["driven_channel_velocity_sign"] == 1

    with pytest.raises(EvidenceIntegrityError, match="velocity sign"):
        evaluate_adapter_gate(
            _probe("isaac"),
            mujoco,
            same_input_actor_checks=_same_input_checks(),
        )


def test_cross_engine_pulse_response_scale_mismatch_is_deferred_to_plant() -> None:
    mujoco = deepcopy(_probe("mujoco"))
    mujoco["pulse_records"]["0"]["odd_velocity_response"][0] = 2.1e-3
    result = evaluate_adapter_gate(
        _probe("isaac"),
        mujoco,
        same_input_actor_checks=_same_input_checks(),
    )
    assert result["passed"] is True
    assert result["digital_chain_passed"] is True
    assert result["plant_response_equivalent"] is False
    assert "pulse_response_magnitude_ratio" in result["deferred_failures"]
    assert result["classification"] == "adapter_chain_equivalent/plant_response_deferred"


def test_post_forward_projection_difference_is_deferred_to_p40() -> None:
    mujoco = deepcopy(_probe("mujoco"))
    mujoco["reset_phases"]["pre_forward"]["all_hinge_position_named"] = list(
        mujoco["reset_phases"]["pre_forward"]["all_hinge_position_named"]
    )
    mujoco["reset_phases"]["post_forward"][
        "all_hinge_position_named"
    ][10] = 2.0e-5

    result = evaluate_adapter_gate(
        _probe("isaac"),
        mujoco,
        same_input_actor_checks=_same_input_checks(),
    )

    assert result["passed"] is True
    assert result["digital_chain_passed"] is True
    assert result["post_forward_projection_equivalent"] is False
    assert "reset_post_forward_all_hinge_position_named_max_abs" in result[
        "deferred_failures"
    ]
    assert result["classification"] == (
        "adapter_chain_equivalent/post_forward_projection_deferred"
    )


def test_missing_physics_step_count_fails_closed() -> None:
    isaac = _probe("isaac")
    del isaac["clock"]["physics_steps_per_action"]
    with pytest.raises(EvidenceIntegrityError, match="physics_steps_per_action"):
        evaluate_adapter_gate(
            isaac,
            _probe("mujoco"),
            same_input_actor_checks=_same_input_checks(),
        )


def test_policy_clock_shift_is_rejected() -> None:
    isaac = deepcopy(_probe("isaac"))
    isaac["clock"]["time_after_s"] = 1.021
    result = evaluate_adapter_gate(
        isaac,
        _probe("mujoco"),
        same_input_actor_checks=_same_input_checks(),
    )
    assert result["passed"] is False
    assert "isaac_clock_elapsed_abs_s" in result["failures"]


def test_passive_hinge_audit_has_separate_bounded_tolerance() -> None:
    mujoco = _probe("mujoco")
    mujoco["reset_phases"]["pre_forward"]["all_hinge_position_named"][10] = 9.0e-6
    mujoco["reset_phases"]["post_forward"]["all_hinge_position_named"][10] = 9.0e-6
    within = evaluate_adapter_gate(
        _probe("isaac"),
        mujoco,
        same_input_actor_checks=_same_input_checks(),
    )
    assert within["passed"] is True

    mujoco["reset_phases"]["pre_forward"]["all_hinge_position_named"][10] = 2.0e-5
    mujoco["reset_phases"]["post_forward"]["all_hinge_position_named"][10] = 2.0e-5
    outside = evaluate_adapter_gate(
        _probe("isaac"),
        mujoco,
        same_input_actor_checks=_same_input_checks(),
    )
    assert outside["passed"] is False
    assert "reset_pre_forward_all_hinge_position_named_max_abs" in outside["failures"]
    assert "reset_post_forward_all_hinge_position_named_max_abs" in outside[
        "deferred_failures"
    ]
