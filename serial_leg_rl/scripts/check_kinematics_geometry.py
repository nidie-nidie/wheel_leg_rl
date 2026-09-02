#!/usr/bin/env python3
"""Compare kinematics constants against the source URDF."""

from __future__ import annotations

from pathlib import Path
import argparse

from serial_leg_rl.assets.urdf_geometry import (
    DEFAULT_SOURCE_URDF,
    compare_geometry,
    extract_left_geometry,
    extract_right_geometry,
)
from serial_leg_rl.kinematics.offset_leg import OffsetLegGeometry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=DEFAULT_SOURCE_URDF)
    args = parser.parse_args()

    reference = OffsetLegGeometry()
    for extracted in (extract_left_geometry(args.urdf), extract_right_geometry(args.urdf)):
        geometry = extracted.offset_geometry
        print(f"[{extracted.side}]")
        print(f"v_ij = {geometry.v_ij}")
        print(f"v_il = {geometry.v_il}")
        print(f"v_ip(raw) = {geometry.v_ip}")
        print(f"l_jm = {geometry.l_jm:.9f}")
        print(f"l_lm = {geometry.l_lm:.9f}")
        print(f"l_pw = {geometry.l_pw:.9f}")
        print(f"k_a = {geometry.k_a:.9f}")
        print(f"k_b = {geometry.k_b:.9f}")
        print("abs diff to current left-reference constants:")
        for name, value in compare_geometry(reference, geometry).items():
            print(f"  {name}: {value:.9e}")
        print()


if __name__ == "__main__":
    main()

