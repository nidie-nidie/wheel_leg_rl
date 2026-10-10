from __future__ import annotations

from pathlib import Path

import numpy as np

from debug.sim2sim.collect_mujoco_trace import collect_mujoco_trace
from debug.sim2sim.compare_traces import load_verified_engine_run
from debug.sim2sim.pitch_torque_pulse import BasePitchTorquePulse
from debug.sim2sim.trace_schema import load_npz, validate_control_trace, validate_substep_trace


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_one_tick_headless_collection(tmp_path) -> None:
    output = tmp_path / "mujoco"
    result = collect_mujoco_trace(
        project_root=PROJECT_ROOT,
        output_directory=output,
        control_ticks=1,
        mode="closed_loop",
    )
    assert result["completed_control_ticks"] == 1
    assert result["mujoco_solver"] == "mjSOL_NEWTON"
    assert result["mujoco_integrator"] == "mjINT_EULER"
    assert result["mujoco_iterations"] == 100
    control = load_npz(output / "control_trace.npz")
    substeps = load_npz(output / "substep_trace.npz")
    validate_control_trace(control)
    validate_substep_trace(substeps, physics_steps_per_action=20, continuity_atol=1.0e-12)
    assert control["actor_obs_policy_pre_step"].shape == (1, 25)
    assert control["actor_output_raw"].shape == (1, 6)
    assert substeps["active_joint_position_canonical_pre_step"].shape == (20, 6)
    np.testing.assert_allclose(
        control["target_command_engine_native"][:, 5],
        -control["target_command_canonical"][:, 5],
        rtol=0.0,
        atol=0.0,
    )
    assert (output / "reset_snapshot.json").is_file()
    assert (output / "metadata.json").is_file()
    assert (output / "file_hashes.json").is_file()
    verified = load_verified_engine_run(output)
    assert verified["metadata"]["action_sequence_identity"]["source"] == "policy_closed_loop"


def test_zero_action_collection_embeds_verified_sequence(tmp_path) -> None:
    output = tmp_path / "mujoco-zero"
    result = collect_mujoco_trace(
        project_root=PROJECT_ROOT,
        output_directory=output,
        control_ticks=1,
        mode="zero_action",
    )
    verified = load_verified_engine_run(output)
    assert result["action_sequence_identity"]["source"] == "generated_zero"
    assert (output / "input_action_sequence.npz").is_file()
    np.testing.assert_array_equal(
        verified["control"]["action_clipped"],
        np.zeros((1, 6), dtype=np.float32),
    )


def test_pitch_torque_pulse_is_applied_in_control_and_world_frames(tmp_path) -> None:
    output = tmp_path / "mujoco-pitch-torque"
    pulse = BasePitchTorquePulse(magnitude_nm=4.0, start_tick=0, end_tick=1)

    result = collect_mujoco_trace(
        project_root=PROJECT_ROOT,
        output_directory=output,
        control_ticks=1,
        mode="zero_action",
        pitch_torque_pulse=pulse,
    )

    substeps = load_npz(output / "substep_trace.npz")
    expected_control = np.tile(np.asarray((0.0, 4.0, 0.0)), (20, 1))
    np.testing.assert_allclose(substeps["external_base_torque_control"], expected_control)
    world_torque = substeps["external_base_torque_engine_world"]
    np.testing.assert_allclose(world_torque[0], np.asarray((4.0, 0.0, 0.0)), atol=1.0e-12)
    np.testing.assert_allclose(np.linalg.norm(world_torque, axis=1), 4.0, atol=1.0e-12)
    assert result["base_pitch_torque_pulse"] == pulse.payload
