import unittest

import torch

from recovery.standing_reference import StandingReference
from independent_real_data import REFERENCE_DIR, REFERENCE_SHA256, real_reference


class StandingReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference = real_reference()

    def test_real_reference_identity_and_paired_samples(self):
        reference = self.reference
        self.assertEqual(reference.q.shape, (560, 4))
        self.assertEqual(reference.identity["bank_sha256"], REFERENCE_SHA256)
        match = reference.nearest(reference.q, reference.height)
        torch.testing.assert_close(match.q, reference.q, rtol=0, atol=0)
        torch.testing.assert_close(match.height, reference.height, rtol=0, atol=0)
        self.assertEqual(float(match.distance.max()), 0.)

    def test_pair_selection_uses_identical_index_for_q_and_height(self):
        q = self.reference.q[::11] + .003
        height = self.reference.height[::11] + .002
        match = self.reference.nearest(q, height)
        expected = (((q[:, None] - self.reference.q[None]) / .1).square().mean(dim=-1)
                    + ((height[:, None] - self.reference.height[None]) / .02).square())
        torch.testing.assert_close(match.index, expected.argmin(dim=1))
        torch.testing.assert_close(match.q, self.reference.q[match.index])
        torch.testing.assert_close(match.height, self.reference.height[match.index])

    def test_queries_exclude_wheel_phase_and_yaw(self):
        data = self.reference.arrays
        first = self.reference.nearest(torch.tensor(data["q_control_rad"][:, :4], dtype=torch.float32),
                                       torch.tensor(data["height_m"], dtype=torch.float32))
        q_with_phase = torch.tensor(data["q_control_rad"], dtype=torch.float32)
        q_with_phase[:, 4:] += 200.
        second = self.reference.nearest(q_with_phase[:, :4], self.reference.height)
        torch.testing.assert_close(first.index, second.index, rtol=0, atol=0)

    def test_invalid_hash_is_rejected(self):
        with self.assertRaises(ValueError):
            StandingReference.load(REFERENCE_DIR / "manifest.json", REFERENCE_DIR / "recovery_contact_paired_reference.npz", "0" * 64)

    def test_nonfinite_query_is_rejected(self):
        height = self.reference.height[:1].clone()
        height[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.reference.nearest(self.reference.q[:1], height)
