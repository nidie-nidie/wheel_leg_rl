from __future__ import annotations

import math

import numpy as np
import pytest

from debug.sim2sim.analyze_pitch_torque_response import (
    centered_response,
    pitch_from_wxyz,
)


def test_pitch_from_wxyz_recovers_control_pitch() -> None:
    angle = math.radians(12.0)
    quaternion = np.asarray((math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0))

    assert math.degrees(float(pitch_from_wxyz(quaternion))) == pytest.approx(12.0)


def test_centered_response_separates_odd_response_and_even_bias() -> None:
    zero = np.asarray((1.0, 2.0))
    positive = np.asarray((4.0, 8.0))
    negative = np.asarray((-2.0, 0.0))

    odd, even_bias = centered_response(zero, positive, negative)

    np.testing.assert_array_equal(odd, np.asarray((3.0, 4.0)))
    np.testing.assert_array_equal(even_bias, np.asarray((0.0, 2.0)))
