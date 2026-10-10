import pathlib
import tempfile
import unittest

from tools.capture_session import next_available_output


class CaptureOutputNamingTest(unittest.TestCase):
    def test_unused_name_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            requested = pathlib.Path(directory) / "commissioning_001.raw"
            self.assertEqual(next_available_output(requested), requested)

    def test_existing_numbered_names_increment_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            first = root / "commissioning_001.raw"
            second = root / "commissioning_002.raw"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            self.assertEqual(
                next_available_output(first),
                root / "commissioning_003.raw",
            )
            self.assertEqual(first.read_bytes(), b"first")
            self.assertEqual(second.read_bytes(), b"second")

    def test_existing_unnumbered_name_gets_001_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            requested = root / "commissioning.raw"
            requested.write_bytes(b"existing")
            self.assertEqual(
                next_available_output(requested),
                root / "commissioning_001.raw",
            )


if __name__ == "__main__":
    unittest.main()
