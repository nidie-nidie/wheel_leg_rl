"""One incremental frame scanner for capture and offline decoding."""
from __future__ import annotations

import dataclasses
import struct

from .schema import MAGIC, FrameDecodeError, decode_frame, expected_frame_size


@dataclasses.dataclass(frozen=True)
class ScanIssue:
    offset: int
    code: str
    message: str


class FrameScanner:
    def __init__(self):
        self.buffer = bytearray()
        self.offset = 0
        self.issues = []

    def _drop(self, count, code, message):
        self.issues.append(ScanIssue(self.offset, code, message))
        del self.buffer[:count]
        self.offset += count

    def _later_valid_frame(self):
        at = self.buffer.find(MAGIC, 1)
        while at >= 0:
            if len(self.buffer) - at >= 6:
                length = struct.unpack_from("<H", self.buffer, at + 4)[0]
                if 14 <= length <= 128 and at + length <= len(self.buffer):
                    try:
                        decode_frame(bytes(self.buffer[at:at + length]))
                    except FrameDecodeError:
                        pass
                    else:
                        return at
            at = self.buffer.find(MAGIC, at + 1)
        return -1

    def feed(self, data, *, final=False):
        self.buffer.extend(data)
        output = []
        while self.buffer:
            at = self.buffer.find(MAGIC)
            if at < 0:
                count = len(self.buffer) if final or self.buffer[-1:] != MAGIC[:1] else len(self.buffer) - 1
                if count:
                    self._drop(count, "trailing_bytes" if final else "resync", "non-frame bytes")
                break
            if at:
                self._drop(at, "resync", f"skipped {at} bytes before magic")
            if len(self.buffer) < 6:
                if final:
                    self._drop(len(self.buffer), "truncated_header", "stream ended before frame length")
                break
            version, kind = self.buffer[2:4]
            length = struct.unpack_from("<H", self.buffer, 4)[0]
            expected = expected_frame_size(version, kind)
            if expected is None or not 14 <= length <= 128 or (expected != -1 and expected != length):
                self._drop(1, "invalid_length", f"unsupported version/type/length {version}/{kind}/{length}")
                continue
            if len(self.buffer) < length:
                # Payload can contain a complete CRC-valid frame by chance.
                # Until the outer candidate is complete, fragmentation must not
                # change interpretation. Only EOF permits forensic recovery.
                later = self._later_valid_frame() if final else -1
                if later >= 0:
                    self._drop(later, "resync", "incomplete candidate obscured a later valid frame")
                    continue
                if final:
                    self._drop(len(self.buffer), "truncated_frame", f"need {length} bytes, stream ended early")
                break
            raw = bytes(self.buffer[:length])
            try:
                frame = decode_frame(raw)
            except FrameDecodeError as error:
                self._drop(1, "crc_error" if "CRC" in str(error) else "malformed_frame", str(error))
                continue
            output.append((self.offset, frame))
            del self.buffer[:length]
            self.offset += length
        return output

    def finish(self):
        return self.feed(b"", final=True)
