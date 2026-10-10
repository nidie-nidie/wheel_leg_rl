from __future__ import annotations

import pathlib
import struct
import unittest
import zlib


ROOT = pathlib.Path(__file__).resolve().parents[2]
FRAME_HEADER = ROOT / "Protocol" / "Inc" / "pace_frame.h"

MAGIC = bytes([0x50, 0x41])
VERSION = 1
SAMPLE_SIZE = 126
STATUS_BASE_SIZE = 64
STATUS_MAX_SIZE = 128


def finish(frame: bytearray) -> bytes:
    struct.pack_into("<I", frame, len(frame) - 4, zlib.crc32(frame[:-4]) & 0xFFFFFFFF)
    return bytes(frame)


def common(frame_type: int, length: int, sequence: int) -> bytearray:
    frame = bytearray(length)
    struct.pack_into("<2sBBHI", frame, 0, MAGIC, VERSION, frame_type, length, sequence)
    return frame


def sample_fixture(sequence: int = 7) -> bytes:
    frame = common(0x02, SAMPLE_SIZE, sequence)
    struct.pack_into("<IBBBBBBBBHH", frame, 10, 123456, 4, 9, 0x0F, 0x3F,
                     0x3F, 0x3F, 0x05, 0, 0x1234, 0x5678)
    for index in range(4):
        values = tuple(0x1000 * (index + 1) + item for item in range(8))
        struct.pack_into("<8H", frame, 26 + index * 16, *values)
    for index in range(2):
        struct.pack_into("<iHhhhHH", frame, 90 + index * 16,
                         -100000 - index, 200 + index, -3 - index,
                         -400 - index, 500 + index, 600 + index, 700 + index)
    return finish(frame)


def status_fixture(extension: bytes, sequence: int = 8) -> bytes:
    length = STATUS_BASE_SIZE + len(extension)
    frame = common(0x04, length, sequence)
    struct.pack_into("<I", frame, 10, 222222)
    frame[14:17] = bytes([0x21, 0x43, 0x65])
    frame[17:27] = bytes(range(10, 20))
    for index in range(6):
        struct.pack_into("<H", frame, 27 + index * 2, 100 + index)
        struct.pack_into("<H", frame, 39 + index * 2, 200 + index)
    struct.pack_into("<HHHBBB", frame, 51, 3, 4, 5, 6, 7, len(extension))
    frame[60:60 + len(extension)] = extension
    return finish(frame)


def validate(frame: bytes) -> bool:
    if len(frame) < 14 or frame[:2] != MAGIC or frame[2] != VERSION:
        return False
    length = struct.unpack_from("<H", frame, 4)[0]
    if length != len(frame):
        return False
    frame_type = frame[3]
    if frame_type == 0x02 and length != SAMPLE_SIZE:
        return False
    if frame_type == 0x04:
        if not STATUS_BASE_SIZE <= length <= STATUS_MAX_SIZE:
            return False
        if STATUS_BASE_SIZE + frame[59] != length:
            return False
    expected = struct.unpack_from("<I", frame, length - 4)[0]
    return expected == (zlib.crc32(frame[:-4]) & 0xFFFFFFFF)


class PaceFrameTest(unittest.TestCase):
    def test_constants_match_c_header(self) -> None:
        text = FRAME_HEADER.read_text(encoding="utf-8")
        self.assertIn("#define PACE_SAMPLE_FRAME_SIZE 126U", text)
        self.assertIn("#define PACE_STATUS_BASE_FRAME_SIZE 64U", text)
        self.assertIn("#define PACE_STATUS_MAX_FRAME_SIZE 128U", text)

    def test_sample_exact_offsets_and_crc(self) -> None:
        frame = sample_fixture()
        self.assertEqual(SAMPLE_SIZE, len(frame))
        self.assertTrue(validate(frame))
        self.assertEqual((123456, 4, 9), struct.unpack_from("<IBB", frame, 10))
        self.assertEqual(tuple(0x1000 + item for item in range(8)),
                         struct.unpack_from("<8H", frame, 26))
        self.assertEqual(-100000, struct.unpack_from("<i", frame, 90)[0])
        self.assertEqual(700, struct.unpack_from("<H", frame, 104)[0])

    def test_status_minimum_and_maximum(self) -> None:
        base = status_fixture(bytes())
        maximum = status_fixture(bytes(range(64)))
        self.assertEqual(64, len(base))
        self.assertEqual(128, len(maximum))
        self.assertTrue(validate(base))
        self.assertTrue(validate(maximum))
        self.assertEqual(64, maximum[59])

    def test_corruption_and_bad_lengths_are_rejected(self) -> None:
        corrupted = bytearray(sample_fixture())
        corrupted[40] ^= 0x01
        self.assertFalse(validate(bytes(corrupted)))
        malformed = bytearray(sample_fixture())
        struct.pack_into("<H", malformed, 4, 125)
        self.assertFalse(validate(bytes(malformed)))
        bad_status = bytearray(status_fixture(bytes([1, 0])))
        bad_status[59] = 3
        self.assertFalse(validate(bytes(bad_status)))

    def test_sequence_gap_is_reportable(self) -> None:
        frames = [sample_fixture(10), sample_fixture(11), sample_fixture(13)]
        sequences = [struct.unpack_from("<I", frame, 6)[0] for frame in frames]
        gaps = [(left, right) for left, right in zip(sequences, sequences[1:])
                if right != ((left + 1) & 0xFFFFFFFF)]
        self.assertEqual([(11, 13)], gaps)


if __name__ == "__main__":
    unittest.main()
