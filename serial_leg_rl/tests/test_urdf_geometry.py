from __future__ import annotations

import unittest

from serial_leg_rl.assets.urdf_geometry import compare_geometry, extract_left_geometry, extract_right_geometry
from serial_leg_rl.kinematics.offset_leg import OffsetLegGeometry


class UrdfGeometryTest(unittest.TestCase):
    def test_left_geometry_matches_current_reference(self) -> None:
        reference = OffsetLegGeometry()
        extracted = extract_left_geometry().offset_geometry
        diffs = compare_geometry(reference, extracted)
        self.assertLess(diffs["v_ij_s"], 1.0e-12)
        self.assertLess(diffs["v_ij_z"], 1.0e-12)
        self.assertLess(diffs["v_il_s"], 1.0e-12)
        self.assertLess(diffs["v_il_z"], 1.0e-12)
        self.assertLess(diffs["v_ip_s_raw"], 1.0e-12)
        self.assertLess(diffs["v_ip_z_raw"], 1.0e-12)
        self.assertLess(diffs["l_jm"], 1.0e-6)
        self.assertLess(diffs["l_lm"], 1.0e-6)
        self.assertLess(diffs["l_pw"], 1.0e-6)

    def test_right_geometry_is_left_mirror_in_leg_plane(self) -> None:
        left = extract_left_geometry().offset_geometry
        right = extract_right_geometry().offset_geometry
        diffs = compare_geometry(left, right)
        self.assertLess(diffs["v_ij_s"], 1.0e-6)
        self.assertLess(diffs["v_ij_z"], 1.0e-6)
        self.assertLess(diffs["v_il_s"], 1.0e-6)
        self.assertLess(diffs["v_il_z"], 1.0e-6)
        self.assertLess(diffs["l_jm"], 1.0e-6)
        self.assertLess(diffs["l_lm"], 1.0e-6)
        self.assertLess(diffs["l_pw"], 1.0e-6)


if __name__ == "__main__":
    unittest.main()

