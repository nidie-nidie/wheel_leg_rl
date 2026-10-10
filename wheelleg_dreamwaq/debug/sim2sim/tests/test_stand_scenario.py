from __future__ import annotations

import numpy as np
import pytest

from debug.sim2sim.stand_scenario import StandScenarioV1, build_action_sequences


def test_scenario_is_frozen_and_deterministic() -> None:
    scenario = StandScenarioV1()
    assert scenario.command == (0.0, 0.0, 0.20)
    assert scenario.control_ticks == 500
    assert scenario.payload_hash == StandScenarioV1().payload_hash
    with pytest.raises(Exception):
        scenario.control_ticks = 1


def test_action_sequences_have_fixed_shapes_and_pulses() -> None:
    sequences = build_action_sequences(StandScenarioV1())
    assert sequences["zero_action"].shape == (100, 6)
    assert "closed_loop_placeholder" not in sequences
    for channel in range(6):
        pulse = sequences[f"channel_pulse_{channel}"]
        assert pulse.shape == (100, 6)
        np.testing.assert_array_equal(pulse[:25], 0.0)
        np.testing.assert_array_equal(pulse[25:50, channel], 0.1)
        np.testing.assert_array_equal(pulse[50:75], 0.0)
        np.testing.assert_array_equal(pulse[75:100, channel], -0.1)
