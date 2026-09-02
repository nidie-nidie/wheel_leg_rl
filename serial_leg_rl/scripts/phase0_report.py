#!/usr/bin/env python3
"""Print a compact Phase 0 validation report."""

from __future__ import annotations

import math

from serial_leg_rl.assets.urdf_geometry import compare_geometry, extract_left_geometry, extract_right_geometry
from serial_leg_rl.kinematics.offset_leg import OffsetLegGeometry, OffsetLegKinematics


def main() -> None:
    reference = OffsetLegGeometry()
    kin = OffsetLegKinematics(reference)
    zero = kin.forward(0.0, 0.0)

    print("Phase 0 validation report")
    print("=========================")
    print("zero pose:")
    print(f"  L0        : {zero.leg_length:.6f} m")
    print(f"  phi0      : {math.degrees(zero.phi0):.3f} deg")
    print(f"  theta_leg : {math.degrees(zero.theta_leg):.3f} deg")
    print(f"  JH        : {zero.jacobian}")
    print(f"  delta_m   : {zero.delta_m:.9e} m^2")
    print()

    for extracted in (extract_left_geometry(), extract_right_geometry()):
        geometry = extracted.offset_geometry
        diffs = compare_geometry(reference, geometry)
        max_diff = max(diffs.values())
        print(f"{extracted.side} geometry:")
        print(f"  l_jm      : {geometry.l_jm:.9f} m")
        print(f"  l_lm      : {geometry.l_lm:.9f} m")
        print(f"  l_pw      : {geometry.l_pw:.9f} m")
        print(f"  k_a       : {geometry.k_a:.9f}")
        print(f"  k_b       : {geometry.k_b:.9f}")
        print(f"  max diff to reference: {max_diff:.9e}")
        print()


if __name__ == "__main__":
    main()

