from __future__ import annotations

import numpy as np

from debug.sim2sim.compare_traces import (
    compare_control_traces,
    compare_metadata_identity,
    compare_reset_snapshots,
    first_threshold_crossing,
    quaternion_geodesic_error,
    summarize_signal,
    wrapped_angle_difference,
)


def test_quaternion_sign_has_zero_geodesic_error() -> None:
    first = np.array([[1.0, 0.0, 0.0, 0.0]])
    second = -first
    np.testing.assert_allclose(quaternion_geodesic_error(first, second), 0.0, atol=1.0e-12)


def test_phi0_uses_wrapped_shortest_angle() -> None:
    epsilon = 1.0e-4
    error = wrapped_angle_difference(
        np.array([np.pi - epsilon]),
        np.array([-np.pi + epsilon]),
    )
    np.testing.assert_allclose(np.abs(error), 2.0 * epsilon, rtol=0.0, atol=1.0e-12)


def test_first_threshold_crossing_and_persistence() -> None:
    error = np.array([0.0, 0.2, 1.1, 1.2, 1.3, 0.0])
    numeric, material, persistent = first_threshold_crossing(error, numeric_limit=0.1, material_limit=1.0)
    assert numeric == 1
    assert material == 2
    assert persistent == 2


def test_signal_summary_reports_max_and_rms() -> None:
    lhs = np.zeros((3, 2))
    rhs = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    summary = summarize_signal(lhs, rhs, numeric_limit=0.1, material_limit=2.0)
    assert summary["max_abs_error"] == 4.0
    np.testing.assert_allclose(summary["rms_error"], np.sqrt(25.0 / 6.0))
    assert summary["first_material_tick"] == 1


def _control_trace(rows: int = 5) -> dict[str, np.ndarray]:
    trace: dict[str, np.ndarray] = {
        name: np.zeros((rows, width), dtype=np.float64)
        for name, width in (
            ("actor_obs_policy_pre_step", 25),
            ("previous_action_before_inference", 6),
            ("actor_output_raw", 6),
            ("action_clipped", 6),
            ("target_command_canonical", 6),
            ("active_joint_position_canonical_post_step_pre_reset", 6),
            ("active_joint_velocity_canonical_post_step_pre_reset", 6),
            ("all_hinge_position_named_post_step_pre_reset", 26),
            ("all_hinge_velocity_named_post_step_pre_reset", 26),
            ("base_com_position_diag_post_step_pre_reset", 3),
            ("base_linear_velocity_control_post_step_pre_reset", 3),
            ("base_angular_velocity_control_post_step_pre_reset", 3),
            ("projected_gravity_post_step_pre_reset", 3),
            ("virtual_leg_phi0_post_step_pre_reset", 2),
            ("virtual_leg_length_post_step_pre_reset", 2),
            ("loop_closure_error_post_step_pre_reset", 2),
        )
    }
    trace["base_orientation_control_wxyz_post_step_pre_reset"] = np.tile(
        np.array((1.0, 0.0, 0.0, 0.0)), (rows, 1)
    )
    trace["base_height_post_step_pre_reset"] = np.zeros(rows)
    trace["control_tick"] = np.arange(rows, dtype=np.int64)
    trace["common_diagnostic_flags_int8"] = np.zeros((rows, 8), dtype=np.int8)
    return trace


def test_comparator_locates_passive_hinge_before_active_state() -> None:
    isaac = _control_trace()
    mujoco = _control_trace()
    mujoco["all_hinge_position_named_post_step_pre_reset"][2:, 9] = 0.01
    summary = compare_control_traces(isaac, mujoco)
    assert summary["first_material_divergence"] == {
        "control_tick": 2,
        "signal": "all_hinge_position_named_post_step_pre_reset",
    }
    assert summary["first_persistent_divergence"] == {
        "control_tick": 2,
        "signal": "all_hinge_position_named_post_step_pre_reset",
    }


def test_control_phi0_wrap_does_not_report_material_difference() -> None:
    isaac = _control_trace()
    mujoco = _control_trace()
    isaac["virtual_leg_phi0_post_step_pre_reset"][:] = np.pi - 1.0e-4
    mujoco["virtual_leg_phi0_post_step_pre_reset"][:] = -np.pi + 1.0e-4
    summary = compare_control_traces(isaac, mujoco)
    metrics = summary["signals"]["virtual_leg_phi0_post_step_pre_reset"]
    assert metrics["max_abs_error"] < 1.0e-3
    assert metrics["first_material_tick"] is None


def test_exact_event_difference_is_reported() -> None:
    isaac = _control_trace()
    mujoco = _control_trace()
    mujoco["common_diagnostic_flags_int8"][3, 0] = 1
    summary = compare_control_traces(isaac, mujoco)
    assert summary["signals"]["common_diagnostic_flags_int8"]["first_material_tick"] == 3
    assert summary["signals"]["common_diagnostic_flags_int8"]["first_persistent_tick"] is None


def test_exact_event_persistence_requires_three_ticks() -> None:
    isaac = _control_trace()
    mujoco = _control_trace()
    mujoco["common_diagnostic_flags_int8"][1:4, 0] = 1
    summary = compare_control_traces(isaac, mujoco)
    assert summary["signals"]["common_diagnostic_flags_int8"]["first_persistent_tick"] == 1


def test_metadata_and_reset_gates_report_failures() -> None:
    metadata = {
        "schema_version": "Sim2SimDebugTraceV1",
        "scenario_hash": "same",
        "frozen_files": {"actor": "same"},
        "scenario_variant": "zero_action",
        "action_sequence_identity": {"content_sha256": "same"},
        "command": [0.0, 0.0, 0.2],
        "random_seed": 0,
        "requested_control_ticks": 5,
        "inference_dtype": "float32",
        "actor_sha256": "same",
        "policy_manifest_sha256": "same",
        "model_manifest_sha256": "same",
        "model_xml_sha256": "same",
        "control_dt_s": 0.02,
        "canonical_joint_order": list(range(6)),
        "canonical_from_engine_native": [1, 1, 1, 1, 1, -1],
        "all_hinge_order": list(range(26)),
        "r_diag_from_engine_world": np.eye(3).tolist(),
    }
    changed = dict(metadata, actor_sha256="different")
    assert compare_metadata_identity(metadata, changed)["failures"] == ["actor_sha256"]

    reset = {
        "reset_forwarded_post_forward": {
            "active_joint_position_canonical": [0.0] * 6,
        },
        "reset_returned_to_policy": {
            "actor_obs_policy_returned": [0.0] * 25,
            "previous_action": [0.0] * 6,
        },
    }
    changed_reset = {
        **reset,
        "reset_forwarded_post_forward": {
            "active_joint_position_canonical": [0.01] + [0.0] * 5,
        },
    }
    report = compare_reset_snapshots(reset, changed_reset)
    assert report["failures"] == [
        "reset_forwarded_post_forward.active_joint_position_canonical"
    ]
