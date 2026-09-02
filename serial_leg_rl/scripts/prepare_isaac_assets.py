#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "serial_leg_rl" / "source" / "serial_leg_rl"
sys.path.insert(0, SOURCE_DIR.as_posix())

from serial_leg_rl.assets.sim_urdf import prepare_sim_urdf
from serial_leg_rl.assets.local_terrain import prepare_local_flat_terrain_usd


if __name__ == "__main__":
    print(prepare_sim_urdf())
    print(prepare_local_flat_terrain_usd())
