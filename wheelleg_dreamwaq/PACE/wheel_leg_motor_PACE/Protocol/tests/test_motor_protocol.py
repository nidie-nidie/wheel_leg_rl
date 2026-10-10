from __future__ import annotations

import math
import struct
import unittest


def float_to_uint(value: float, minimum: float, maximum: float, bits: int) -> int:
    value = min(max(value, minimum), maximum)
    return int((value - minimum) * ((1 << bits) - 1) / (maximum - minimum))


def uint_to_float(value: int, minimum: float, maximum: float, bits: int) -> float:
    return value * (maximum - minimum) / ((1 << bits) - 1) + minimum


def pack_dm(q: float, dq: float, kp: float, kd: float, tau: float) -> tuple[bytes, tuple[int, ...]]:
    raw = (
        float_to_uint(q, -12.5, 12.5, 16),
        float_to_uint(dq, -45.0, 45.0, 12),
        float_to_uint(kp, 0.0, 500.0, 12),
        float_to_uint(kd, 0.0, 5.0, 12),
        float_to_uint(tau, -54.0, 54.0, 12),
    )
    q_raw, dq_raw, kp_raw, kd_raw, tau_raw = raw
    payload = bytes([
        q_raw >> 8,
        q_raw & 0xFF,
        dq_raw >> 4,
        ((dq_raw & 0xF) << 4) | (kp_raw >> 8),
        kp_raw & 0xFF,
        kd_raw >> 4,
        ((kd_raw & 0xF) << 4) | (tau_raw >> 8),
        tau_raw & 0xFF,
    ])
    return payload, raw


class MotorProtocolTest(unittest.TestCase):
    def test_dm_baseline_command_vector(self) -> None:
        payload, raw = pack_dm(0.0, 0.0, 20.0, 0.6, 0.0)
        self.assertEqual((32767, 2047, 163, 491, 2047), raw)
        self.assertEqual(bytes.fromhex("7fff7ff0a31eb7ff"), payload)

    def test_dm_feedback_decoder_vector(self) -> None:
        q_raw, dq_raw, tau_raw = 0x9234, 0xABC, 0x567
        payload = bytes([
            0x30, q_raw >> 8, q_raw & 0xFF, dq_raw >> 4,
            ((dq_raw & 0xF) << 4) | (tau_raw >> 8), tau_raw & 0xFF, 42, 43,
        ])
        decoded_q = (payload[1] << 8) | payload[2]
        decoded_dq = (payload[3] << 4) | (payload[4] >> 4)
        decoded_tau = ((payload[4] & 0xF) << 8) | payload[5]
        self.assertEqual((q_raw, dq_raw, tau_raw), (decoded_q, decoded_dq, decoded_tau))
        self.assertAlmostEqual(uint_to_float(q_raw, -12.5, 12.5, 16), 1.77787, places=4)

    def test_lk_torque_and_velocity_vectors(self) -> None:
        iq = int((1.0 / 0.32) * 124.1212121212121)
        torque_payload = bytes([0xA1, 0, 0, 0]) + struct.pack("<h", iq) + bytes([0, 0])
        self.assertEqual(8, len(torque_payload))
        self.assertEqual(iq, struct.unpack_from("<h", torque_payload, 4)[0])
        speed = int(2.0 * 57.2957795131 * 100)
        velocity_payload = bytes([0xA2, 0, 0, 0]) + struct.pack("<i", speed)
        self.assertEqual(speed, struct.unpack_from("<i", velocity_payload, 4)[0])

    def test_lk_feedback_units(self) -> None:
        payload = bytes([0xA1, 35]) + struct.pack("<h", -512) + struct.pack("<h", 180) + struct.pack("<H", 32768)
        iq, speed, encoder = struct.unpack_from("<hhH", payload, 2)
        self.assertEqual((-512, 180, 32768), (iq, speed, encoder))
        self.assertAlmostEqual(-512 * 0.008056640625, -4.125)
        self.assertAlmostEqual(speed * 0.0174532925, math.pi, places=5)


if __name__ == "__main__":
    unittest.main()
