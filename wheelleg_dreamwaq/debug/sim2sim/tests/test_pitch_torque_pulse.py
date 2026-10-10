from __future__ import annotations

import numpy as np
import pytest

from debug.sim2sim.pitch_torque_pulse import BasePitchTorquePulse


def test_pitch_torque_pulse_is_end_exclusive_and_uses_control_pitch_axis() -> None:
    pulse = BasePitchTorquePulse(magnitude_nm=4.0, start_tick=2, end_tick=5)

    np.testing.assert_array_equal(pulse.control_vector(1), np.zeros(3))
    np.testing.assert_array_equal(pulse.control_vector(2), np.asarray((0.0, 4.0, 0.0)))
    np.testing.assert_array_equal(pulse.control_vector(4), np.asarray((0.0, 4.0, 0.0)))
    np.testing.assert_array_equal(pulse.control_vector(5), np.zeros(3))
    pulse.validate_control_ticks(6)


def test_disabled_pitch_torque_pulse_has_empty_interval() -> None:
    pulse = BasePitchTorquePulse.disabled()

    assert pulse.payload == {"magnitude_nm": 0.0, "start_tick": 0, "end_tick": 0}
    np.testing.assert_array_equal(pulse.control_vector(0), np.zeros(3))
    pulse.validate_control_ticks(1)


@pytest.mark.parametrize(
    ("magnitude_nm", "start_tick", "end_tick"),
    (
        (float("nan"), 0, 1),
        (4.0, -1, 1),
        (4.0, 2, 2),
        (0.0, 0, 1),
    ),
)
def test_invalid_pitch_torque_pulses_are_rejected(
    magnitude_nm: float, start_tick: int, end_tick: int
) -> None:
    with pytest.raises(ValueError):
        BasePitchTorquePulse(
            magnitude_nm=magnitude_nm,
            start_tick=start_tick,
            end_tick=end_tick,
        )


def test_pitch_torque_pulse_must_fit_collection_window() -> None:
    pulse = BasePitchTorquePulse(magnitude_nm=-4.0, start_tick=5, end_tick=10)

    with pytest.raises(ValueError, match="collection window"):
        pulse.validate_control_ticks(9)
