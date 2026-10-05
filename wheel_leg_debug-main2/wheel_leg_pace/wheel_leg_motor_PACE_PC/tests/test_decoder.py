from __future__ import annotations

import pathlib
import struct
import sys
import tempfile
import unittest
import zlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace_raw.decoder import decode_bytes
from pace_raw.manifest import load_firmware_manifest
from pace_raw.normalize import normalize_session


MAGIC = b"PA"
MANIFEST_HASH = 0x4D4F5431


def common(frame_type: int, length: int, sequence: int) -> bytearray:
    frame = bytearray(length)
    struct.pack_into("<2sBBHI", frame, 0, MAGIC, 1, frame_type, length, sequence)
    return frame


def finish(frame: bytearray) -> bytes:
    struct.pack_into("<I", frame, len(frame) - 4, zlib.crc32(frame[:-4]) & 0xFFFFFFFF)
    return bytes(frame)


def session_header(sequence: int = 0) -> bytes:
    frame = common(1, 58, sequence)
    frame[10:13] = bytes([1, 1, 1])
    struct.pack_into("<IHHIIH", frame, 14, 123, 500, 10, 0x12345678, 0x50414331, 126)
    struct.pack_into("<IIIIII", frame, 34, 921600, 1_000_000, MANIFEST_HASH, 0x20260928, 0x574C5031, 0)
    return finish(frame)


def stage_config(sequence: int = 1, config_seq: int = 1) -> bytes:
    frame = common(3, 76, sequence)
    frame[10] = config_seq
    frame[11] = 2
    frame[12:18] = bytes([1, 1, 1, 1, 5, 5])
    struct.pack_into("<6H", frame, 18, 500, 500, 500, 500, 100, 100)
    struct.pack_into("<4H", frame, 30, 163, 163, 163, 163)
    struct.pack_into("<4H", frame, 38, 491, 491, 491, 491)
    frame[46] = 1
    struct.pack_into("<fffIH", frame, 48, 0.15, 0.2, 5.0, 12000, 1)
    struct.pack_into("<I", frame, 68, 0x50414331)
    return finish(frame)


def sample(sequence: int, sample_time: int, config_seq: int = 1) -> bytes:
    frame = common(2, 126, sequence)
    struct.pack_into(
        "<IBBBBBBBBHH",
        frame,
        10,
        sample_time,
        2,
        config_seq,
        0x0F,
        0x3F,
        0x3F,
        0x3F,
        0,
        0,
        0,
        0,
    )
    for index in range(4):
        struct.pack_into(
            "<8H",
            frame,
            26 + index * 16,
            1000 + index,
            2000 + index,
            3000 + index,
            4000 + index,
            500 + index,
            600 + index,
            100 + index,
            200 + index,
        )
    for index in range(2):
        struct.pack_into(
            "<iHhhhHH",
            frame,
            90 + index * 16,
            -12345 - index,
            32000 + index,
            -2 - index,
            -100 - index,
            200 + index,
            300 + index,
            400 + index,
        )
    return finish(frame)


def footer(sequence: int, total_frames: int) -> bytes:
    frame = common(6, 34, sequence)
    struct.pack_into("<IIIBBHI", frame, 10, 5000, total_frames, 0, 0, 1, 0, 0x50414331)
    return finish(frame)


def status(sequence: int, enqueue_fail: int = 0, tx_event_lost: int = 0,
           can_error: int = 0, can_state: int = 0) -> bytes:
    frame = common(4, 64, sequence)
    struct.pack_into("<I", frame, 10, 4000)
    struct.pack_into("<6H", frame, 27, 1, 2, 3, 4, 5, 6)
    struct.pack_into("<6H", frame, 39, 6, 5, 4, 3, 2, 1)
    struct.pack_into(
        "<HHHBBB",
        frame,
        51,
        enqueue_fail,
        tx_event_lost,
        can_error,
        7,
        can_state,
        0,
    )
    return finish(frame)


class DecoderTest(unittest.TestCase):
    def test_exact_raw_values_context_and_time_wrap(self):
        stream = b"".join(
            [
                session_header(),
                stage_config(),
                sample(2, 0xFFFFFF00),
                sample(3, 0x000006D0),
                footer(4, 5),
            ]
        )
        decoded = decode_bytes(stream)
        self.assertEqual([], decoded.issues)
        self.assertEqual(2, len(decoded.samples))
        first, second = decoded.samples
        self.assertEqual(1000, first.dm[0].cmd_q_raw)
        self.assertEqual(600, first.dm[0].fb_tau_raw)
        self.assertEqual(-12345, first.lk[0].cmd_primary_raw)
        self.assertEqual(32000, first.lk[0].fb_encoder_raw)
        self.assertEqual(2000, second.extended_time_us - first.extended_time_us)
        self.assertEqual(1, first.stage_config.config_seq)
        self.assertEqual("PROVISIONAL_FIT", first.role)
        self.assertEqual(first.extended_time_us - 100, first.dm[0].tx_time_us)

    def test_missing_stage_config_excludes_sample(self):
        stream = session_header() + sample(1, 2000, config_seq=9) + footer(2, 3)
        decoded = decode_bytes(stream)
        self.assertIn("missing_stage_config", [issue.code for issue in decoded.issues])
        self.assertFalse(decoded.samples[0].fit_eligible)
        self.assertEqual([], decoded.fit_samples)

    def test_malformed_length_and_crc_are_reported(self):
        malformed = common(2, 125, 2)
        malformed = finish(malformed)
        corrupted = bytearray(sample(3, 2000))
        corrupted[40] ^= 0x55
        valid = sample(4, 4000)
        stream = session_header() + stage_config() + malformed + bytes(corrupted) + valid + footer(5, 6)
        decoded = decode_bytes(stream)
        codes = [issue.code for issue in decoded.issues]
        self.assertIn("malformed_frame", codes)
        self.assertIn("crc_error", codes)
        self.assertIn("sequence_gap", codes)
        self.assertEqual(1, len(decoded.samples))

    def test_manifest_and_normalization_preserve_raw_integers(self):
        manifest = load_firmware_manifest()
        decoded = decode_bytes(session_header() + stage_config() + sample(2, 2000) + footer(3, 4))
        dataset = normalize_session(decoded, manifest)
        row = dataset.rows[0]
        self.assertEqual(1000, row["m0_cmd_q_raw"])
        self.assertEqual(-12345, row["m4_cmd_primary_raw"])
        self.assertEqual("L_front", row["m0_name"])
        self.assertAlmostEqual(20.0, row["m0_kp"], delta=0.2)
        self.assertEqual(6, len(manifest.motors))
        self.assertEqual(4, len(dataset.leg_dm_view))
        self.assertEqual(2, len(dataset.wheel_lk_view))

    def test_timing_gap_excludes_the_affected_sample(self):
        stream = b"".join(
            [
                session_header(),
                stage_config(),
                sample(2, 2000),
                sample(3, 6000),
                footer(4, 5),
            ]
        )
        decoded = decode_bytes(stream)
        self.assertIn("timestamp_discontinuity", [issue.code for issue in decoded.issues])
        self.assertTrue(decoded.samples[0].fit_eligible)
        self.assertFalse(decoded.samples[1].fit_eligible)

    def test_status_faults_and_counts_are_reported(self):
        stream = b"".join(
            [
                session_header(),
                stage_config(),
                sample(2, 2000),
                status(3, enqueue_fail=1, tx_event_lost=2, can_error=3, can_state=2),
                footer(4, 5),
            ]
        )
        decoded = decode_bytes(stream)
        codes = {issue.code for issue in decoded.issues}
        self.assertTrue(
            {"can_enqueue_failure", "tx_event_loss", "can_error", "can_state_fault"}
            <= codes
        )
        summary = decoded.summary()
        self.assertEqual([1, 2, 3, 4, 5, 6], summary["communication"]["tx_count_delta_total"])
        self.assertEqual(1, summary["communication"]["enqueue_fail_delta_total"])
        self.assertEqual(2, summary["communication"]["tx_event_lost_delta_total"])
        self.assertEqual(3, summary["communication"]["can_error_delta_total"])

    def test_config_hash_mismatch_is_reported(self):
        bad_stage = bytearray(stage_config())
        struct.pack_into("<I", bad_stage, 68, 0xDEADBEEF)
        bad_stage = finish(bad_stage)
        stream = session_header() + bad_stage + sample(2, 2000) + footer(3, 4)
        decoded = decode_bytes(stream)
        self.assertIn("config_hash_mismatch", [issue.code for issue in decoded.issues])

    def test_csv_and_npz_exports(self):
        decoded = decode_bytes(session_header() + stage_config() + sample(2, 2000) + footer(3, 4))
        dataset = normalize_session(decoded)
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            dataset.export_csv(directory / "session.csv")
            dataset.export_npz(directory / "session.npz")
            self.assertGreater((directory / "session.csv").stat().st_size, 100)
            self.assertGreater((directory / "session.npz").stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
