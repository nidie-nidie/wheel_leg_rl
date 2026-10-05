import pathlib
import struct
import sys
import unittest
import zlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace_raw.commands import CONFIGURE, START, configure, start


class CommandTest(unittest.TestCase):
    def test_configure_layout_and_crc(self):
        frame = configure(7)
        self.assertEqual(b"PC", frame[:2])
        self.assertEqual(CONFIGURE, frame[3])
        self.assertEqual(15, len(frame))
        self.assertEqual(7, struct.unpack_from("<I", frame, 6)[0])
        self.assertEqual(1, frame[10])
        self.assertEqual(zlib.crc32(frame[:-4]) & 0xFFFFFFFF, struct.unpack_from("<I", frame, 11)[0])

    def test_start_has_no_payload(self):
        frame = start(8)
        self.assertEqual(START, frame[3])
        self.assertEqual(14, len(frame))


if __name__ == "__main__":
    unittest.main()
