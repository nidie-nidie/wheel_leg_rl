from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import debug.sim2sim.collect_mujoco_isolation_trace as isolation_collector
from debug.sim2sim.collect_mujoco_isolation_trace import collect_isolation_trace
from debug.sim2sim.isolation_scenarios import IsolationScenario
from debug.sim2sim.trace_schema import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_fixed_base_collection_keeps_base_motion_zero(tmp_path: Path) -> None:
    scenario = IsolationScenario.fixed_base_pulse(
        control_ticks=2,
        channel=0,
        amplitude=0.1,
        pulse_start_tick=0,
        pulse_end_tick=2,
    )
    output = tmp_path / "fixed"

    metadata = collect_isolation_trace(PROJECT_ROOT, output, scenario)
    trace = load_npz(output / "isolation_trace.npz")

    assert metadata["schema_version"] == "MujocoIsolationTraceV1"
    assert metadata["completed_control_ticks"] == 2
    assert not metadata["model_has_free_joint"]
    np.testing.assert_array_equal(trace["base_linear_velocity_control"], 0.0)
    np.testing.assert_array_equal(trace["base_angular_velocity_control"], 0.0)
    np.testing.assert_allclose(
        trace["target_engine_native"],
        trace["target_canonical"] * np.asarray((1, 1, 1, 1, 1, -1)),
        rtol=0.0,
        atol=0.0,
    )
    assert trace["loop_closure_error_m"].shape == (2, 8)
    assert (output / "metadata.json").is_file()
    assert (output / "file_hashes.json").is_file()


def test_suspended_collection_has_no_wheel_contact(tmp_path: Path) -> None:
    scenario = IsolationScenario.suspended_pulse(
        control_ticks=2,
        channel=4,
        amplitude=0.1,
        pulse_start_tick=0,
        pulse_end_tick=2,
    )
    output = tmp_path / "suspended"

    metadata = collect_isolation_trace(PROJECT_ROOT, output, scenario)
    trace = load_npz(output / "isolation_trace.npz")

    assert metadata["model_has_free_joint"]
    np.testing.assert_array_equal(trace["wheel_contact_active"], 0)
    np.testing.assert_array_equal(trace["wheel_normal_force_n_mean"], 0.0)


def test_grounded_pitch_collection_applies_control_frame_pitch_and_wheel_target(tmp_path: Path) -> None:
    scenario = IsolationScenario.grounded_pitch(
        control_ticks=2,
        initial_pitch_deg=5.0,
        wheel_common_action=0.2,
    )
    output = tmp_path / "grounded"

    metadata = collect_isolation_trace(PROJECT_ROOT, output, scenario)
    trace = load_npz(output / "isolation_trace.npz")

    assert abs(metadata["initial_state"]["pitch_deg"] - 5.0) < 1.0e-5
    np.testing.assert_array_equal(
        trace["action_canonical"][:, 4:6],
        np.full((2, 2), 0.2, dtype=np.float32),
    )
    np.testing.assert_allclose(trace["target_canonical"][:, 4:6], 5.0, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(
        trace["target_engine_native"][:, 4:6],
        np.tile(np.asarray((5.0, -5.0)), (2, 1)),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert trace["commanded_torque_canonical_peak_abs"].shape == (2, 6)
    assert trace["effort_limit_fraction"].shape == (2, 6)
    assert trace["velocity_limit_fraction"].shape == (2, 6)
    assert np.all(
        trace["commanded_torque_canonical_peak_abs"]
        >= np.abs(trace["commanded_torque_canonical_mean"])
    )
    assert np.max(trace["wheel_contact_active"]) == 1


def test_collection_rejects_scenario_model_boundary_mismatch(tmp_path: Path) -> None:
    scenario = IsolationScenario(
        kind="grounded_pitch",
        control_ticks=2,
        model_relative="debug/sim2sim/models/wheel_leg_urdf4_suspended_debug.xml",
        has_ground_contact=True,
        initial_pitch_deg=5.0,
        wheel_common_action=0.2,
    )
    output = tmp_path / "bad-model"

    with pytest.raises(ValueError, match="unexpected gravity|ground contact"):
        collect_isolation_trace(PROJECT_ROOT, output, scenario)

    assert not output.exists()


def test_collection_removes_temporary_output_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = IsolationScenario.fixed_base_pulse(
        control_ticks=2,
        channel=0,
        pulse_start_tick=0,
        pulse_end_tick=2,
    )
    output = tmp_path / "broken"

    def fail_after_partial_write(project_root: Path, temporary: Path, value: IsolationScenario):
        del project_root, value
        temporary.mkdir(parents=True)
        (temporary / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected failure")

    monkeypatch.setattr(
        isolation_collector,
        "_collect_isolation_trace_into_directory",
        fail_after_partial_write,
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        collect_isolation_trace(PROJECT_ROOT, output, scenario)

    assert not output.exists()
    assert list(tmp_path.glob(".broken.tmp-*")) == []
