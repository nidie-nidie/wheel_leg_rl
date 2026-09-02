from __future__ import annotations

import math
import unittest

from serial_leg_rl.kinematics.offset_leg import OffsetLegKinematics, wrap_to_pi


class OffsetLegKinematicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kin = OffsetLegKinematics()

    def test_zero_pose_is_nearly_vertical(self) -> None:
        state = self.kin.forward(0.0, 0.0)
        self.assertAlmostEqual(state.leg_length, 0.120493, places=5)
        self.assertAlmostEqual(math.degrees(state.phi0), 90.015, places=2)
        self.assertAlmostEqual(math.degrees(state.theta_leg), -0.015, places=2)
        self.assertLess(state.m[1], state.l[1])
        self.assertAlmostEqual(math.dist(state.m, state.l), self.kin.geometry.l_lm, places=9)
        self.assertAlmostEqual(math.dist(state.m, state.j), self.kin.geometry.l_jm, places=9)

    def test_analytic_jacobian_matches_central_difference(self) -> None:
        samples = [
            (0.0, 0.0),
            (0.15, -0.12),
            (-0.20, 0.18),
            (0.30, 0.10),
        ]
        eps = 1.0e-6
        for q_front, q_rear in samples:
            with self.subTest(q=(q_front, q_rear)):
                state = self.kin.forward(q_front, q_rear)
                numeric = []
                for col in range(2):
                    dq = [0.0, 0.0]
                    dq[col] = eps
                    plus = self.kin.forward(q_front + dq[0], q_rear + dq[1])
                    minus = self.kin.forward(q_front - dq[0], q_rear - dq[1])
                    d_l = (plus.leg_length - minus.leg_length) / (2.0 * eps)
                    d_phi = wrap_to_pi(plus.phi0 - minus.phi0) / (2.0 * eps)
                    numeric.append((d_l, d_phi))

                self.assertAlmostEqual(state.jacobian[0][0], numeric[0][0], places=7)
                self.assertAlmostEqual(state.jacobian[1][0], numeric[0][1], places=7)
                self.assertAlmostEqual(state.jacobian[0][1], numeric[1][0], places=7)
                self.assertAlmostEqual(state.jacobian[1][1], numeric[1][1], places=7)

    def test_fk_ik_roundtrip(self) -> None:
        samples = [
            (0.0, 0.0),
            (0.10, -0.08),
            (-0.16, 0.12),
        ]
        for q_front, q_rear in samples:
            with self.subTest(q=(q_front, q_rear)):
                state = self.kin.forward(q_front, q_rear)
                solved = self.kin.inverse(state.leg_length, state.phi0, seed=(q_front + 0.02, q_rear - 0.02))
                self.assertAlmostEqual(wrap_to_pi(solved[0] - q_front), 0.0, places=7)
                self.assertAlmostEqual(wrap_to_pi(solved[1] - q_rear), 0.0, places=7)

    def test_virtual_velocity_uses_jacobian(self) -> None:
        q_front, q_rear = 0.12, -0.07
        dq_front, dq_rear = 0.5, -0.3
        state = self.kin.forward(q_front, q_rear)
        d_l0, d_phi0 = self.kin.virtual_velocity(q_front, q_rear, dq_front, dq_rear)
        self.assertAlmostEqual(d_l0, state.jacobian[0][0] * dq_front + state.jacobian[0][1] * dq_rear)
        self.assertAlmostEqual(d_phi0, state.jacobian[1][0] * dq_front + state.jacobian[1][1] * dq_rear)


if __name__ == "__main__":
    unittest.main()
