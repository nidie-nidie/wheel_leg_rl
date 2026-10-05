from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "Config" / "pace_motor_manifest.h"


def load_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("X(PACE_MOTOR_"):
            if stripped.endswith(chr(92)):
                stripped = stripped[:-1].rstrip()
            rows.append([part.strip() for part in stripped[2:].split(",")])
    return rows


class MotorManifestTest(unittest.TestCase):
    def test_order_and_ids(self) -> None:
        rows = load_rows()
        self.assertEqual(6, len(rows))
        self.assertEqual(
            ['"L_front"', '"L_rear"', '"R_rear"', '"R_front"', '"L_wheel"', '"R_wheel"'],
            [row[1] for row in rows],
        )
        device_ids = [int(row[4][:-1]) for row in rows]
        feedback_ids = [int(row[6][:-1], 16) for row in rows]
        self.assertEqual([1, 2, 3, 6, 4, 5], device_ids)
        self.assertEqual(6, len(set(device_ids)))
        self.assertEqual(6, len(set(feedback_ids)))

    def test_joint_mapping_and_right_leg_order(self) -> None:
        rows = load_rows()
        self.assertEqual(
            ['"jIJ"', '"jIO"', '"jAG"', '"jAB"', '"jwheel_left"', '"jwheel_right"'],
            [row[7] for row in rows],
        )
        self.assertEqual('"R_rear"', rows[2][1])
        self.assertEqual('"R_front"', rows[3][1])

    def test_lk_bus_identifier_offset(self) -> None:
        for row in load_rows()[4:]:
            device_id = int(row[4][:-1])
            command_id = int(row[5][:-1], 16)
            feedback_id = int(row[6][:-1], 16)
            self.assertEqual(device_id + 0x140, command_id)
            self.assertEqual(command_id, feedback_id)


if __name__ == "__main__":
    unittest.main()
