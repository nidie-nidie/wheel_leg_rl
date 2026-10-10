from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from debug.sim2sim.collect_mujoco_isolation_trace import _scenario_from_args
from debug.sim2sim.isolation_scenarios import IsolationScenario


def test_grounded_pitch_uses_constant_canonical_wheel_action() -> None:
    scenario = IsolationScenario.grounded_pitch(
        control_ticks=8,
        initial_pitch_deg=5.0,
        wheel_common_action=0.2,
    )

    sequence = scenario.action_sequence()

    assert scenario.model_relative == "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
    assert scenario.has_ground_contact
    assert sequence.dtype == np.float32
    assert sequence.shape == (8, 6)
    np.testing.assert_array_equal(sequence[:, :4], 0.0)
    np.testing.assert_array_equal(sequence[:, 4:6], np.full((8, 2), 0.2, dtype=np.float32))


@pytest.mark.parametrize("kind", ("fixed_base", "suspended"))
def test_isolation_pulse_only_drives_selected_channel(kind: str) -> None:
    factory = {
        "fixed_base": IsolationScenario.fixed_base_pulse,
        "suspended": IsolationScenario.suspended_pulse,
    }[kind]
    scenario = factory(
        control_ticks=10,
        channel=4,
        amplitude=0.1,
        pulse_start_tick=2,
        pulse_end_tick=6,
    )

    sequence = scenario.action_sequence()

    np.testing.assert_array_equal(sequence[:2], 0.0)
    np.testing.assert_array_equal(sequence[2:6, 4], np.full(4, 0.1, dtype=np.float32))
    np.testing.assert_array_equal(sequence[2:6, :4], 0.0)
    np.testing.assert_array_equal(sequence[2:6, 5], 0.0)
    np.testing.assert_array_equal(sequence[6:], 0.0)


def test_scenario_rejects_invalid_experiment_parameters() -> None:
    with pytest.raises(ValueError, match="control_ticks"):
        IsolationScenario.grounded_pitch(control_ticks=0, initial_pitch_deg=5.0, wheel_common_action=0.2)
    with pytest.raises(ValueError, match="initial_pitch_deg"):
        IsolationScenario.grounded_pitch(control_ticks=10, initial_pitch_deg=0.0, wheel_common_action=0.2)
    with pytest.raises(ValueError, match="wheel_common_action"):
        IsolationScenario.grounded_pitch(control_ticks=10, initial_pitch_deg=5.0, wheel_common_action=1.1)
    with pytest.raises(ValueError, match="channel"):
        IsolationScenario.fixed_base_pulse(control_ticks=10, channel=6)
    with pytest.raises(ValueError, match="pulse"):
        IsolationScenario.suspended_pulse(
            control_ticks=10,
            channel=4,
            pulse_start_tick=5,
            pulse_end_tick=5,
        )


@pytest.mark.parametrize(
    ("mode", "expected"),
    (("common", (0.1, 0.1)), ("differential", (0.1, -0.1))),
)
def test_suspended_wheel_pulse_supports_common_and_differential_modes(
    mode: str,
    expected: tuple[float, float],
) -> None:
    scenario = IsolationScenario.suspended_wheel_pulse(
        control_ticks=6,
        mode=mode,
        amplitude=0.1,
        pulse_start_tick=1,
        pulse_end_tick=5,
    )

    sequence = scenario.action_sequence()

    np.testing.assert_array_equal(sequence[0], 0.0)
    np.testing.assert_array_equal(
        sequence[1:5, 4:6],
        np.tile(np.asarray(expected, dtype=np.float32), (4, 1)),
    )
    np.testing.assert_array_equal(sequence[5], 0.0)


def _args(**overrides):
    values = {
        "scenario": "fixed_base",
        "control_ticks": 50,
        "channel": None,
        "wheel_pulse_mode": None,
        "amplitude": None,
        "pulse_start_tick": None,
        "pulse_end_tick": None,
        "initial_pitch_deg": None,
        "wheel_action": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cli_scenario_defaults_are_applied_only_to_relevant_mode() -> None:
    fixed = _scenario_from_args(_args())
    grounded = _scenario_from_args(_args(scenario="grounded_pitch"))

    assert fixed.pulse_channel == 4
    assert fixed.pulse_amplitude == 0.1
    assert grounded.initial_pitch_deg == 5.0
    assert grounded.wheel_common_action == 0.2


def test_cli_scenario_rejects_inapplicable_arguments() -> None:
    with pytest.raises(ValueError, match="pulse arguments"):
        _scenario_from_args(_args(scenario="grounded_pitch", channel=4))
    with pytest.raises(ValueError, match="grounded arguments"):
        _scenario_from_args(_args(scenario="fixed_base", wheel_action=0.2))
    with pytest.raises(ValueError, match="either"):
        _scenario_from_args(_args(scenario="suspended", channel=4, wheel_pulse_mode="common"))
