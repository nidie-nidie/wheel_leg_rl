from __future__ import annotations

import argparse
import pathlib
import struct
import sys
import time

import serial


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace_raw.commands import (
    COMMISSIONING_SESSION,
    FINAL_IDENTIFICATION_SESSION,
    abort,
    configure,
    start,
    stop,
)
from pace_raw.schema import FOOTER, MAGIC, FrameDecodeError, decode_frame


def next_available_output(path: pathlib.Path) -> pathlib.Path:
    """Return a numbered path without ever selecting an existing file."""
    path = pathlib.Path(path)
    if not path.exists():
        return path

    stem = path.stem
    base, separator, number_text = stem.rpartition("_")
    if separator and number_text.isdigit():
        number = int(number_text) + 1
        width = len(number_text)
    else:
        base = stem
        number = 1
        width = 3

    while True:
        candidate = path.with_name(
            f"{base}_{number:0{width}d}{path.suffix}"
        )
        if not candidate.exists():
            return candidate
        number += 1


class FooterDetector:
    def __init__(self):
        self.buffer = bytearray()
        self.footer_seen = False

    def feed(self, data: bytes) -> None:
        self.buffer.extend(data)
        while True:
            magic = self.buffer.find(MAGIC)
            if magic < 0:
                if len(self.buffer) > 1:
                    del self.buffer[:-1]
                return
            if magic > 0:
                del self.buffer[:magic]
            if len(self.buffer) < 6:
                return
            length = struct.unpack_from("<H", self.buffer, 4)[0]
            if length < 14 or length > 128:
                del self.buffer[0]
                continue
            if len(self.buffer) < length:
                return
            frame = bytes(self.buffer[:length])
            del self.buffer[:length]
            try:
                decoded = decode_frame(frame)
            except FrameDecodeError:
                continue
            if decoded.header.frame_type == FOOTER:
                self.footer_seen = True


def capture(
    port: str,
    output: pathlib.Path,
    stop_after: float,
    idle_timeout: float,
    startup_timeout: float,
    session_type: int = COMMISSIONING_SESSION,
    passive: bool = False,
) -> None:
    output = next_available_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving raw capture to: {output.resolve()}", flush=True)
    detector = FooterDetector()
    sequence = 0
    started_at = time.monotonic()
    last_data_at = started_at
    received_data = False
    stop_sent = False

    with serial.Serial(port=port, baudrate=921600, timeout=0.05) as device, output.open("xb") as stream:
        device.reset_input_buffer()
        if not passive:
            device.write(configure(sequence, session_type=session_type))
            sequence += 1
            device.flush()
            time.sleep(0.05)
            device.write(start(sequence))
            sequence += 1
            device.flush()

        try:
            while not detector.footer_seen:
                chunk = device.read(4096)
                now = time.monotonic()
                if chunk:
                    stream.write(chunk)
                    stream.flush()
                    detector.feed(chunk)
                    last_data_at = now
                    received_data = True
                if stop_after > 0.0 and not stop_sent and now - started_at >= stop_after:
                    if not passive:
                        device.write(stop(sequence))
                        sequence += 1
                        device.flush()
                    stop_sent = True
                active_timeout = (startup_timeout if passive and not received_data
                                  else idle_timeout)
                if now - last_data_at >= active_timeout:
                    if not passive:
                        device.write(abort(sequence))
                        device.flush()
                    raise TimeoutError(
                        f"no UART data received for {active_timeout:.1f} s"
                    )
        except KeyboardInterrupt:
            if passive:
                return
            device.write(stop(sequence))
            sequence += 1
            device.flush()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not detector.footer_seen:
                chunk = device.read(4096)
                if chunk:
                    stream.write(chunk)
                    stream.flush()
                    detector.feed(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one wheel_leg_motor_PACE commissioning session")
    parser.add_argument("--port", required=True, help="Windows COM port, for example COM7")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--stop-after", type=float, default=0.0, help="send STOP after N seconds; 0 waits for normal completion")
    parser.add_argument("--idle-timeout", type=float, default=5.0)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=15.0,
        help="maximum wait for the first byte in passive mode",
    )
    parser.add_argument(
        "--passive",
        action="store_true",
        help="receive only; do not send CONFIGURE, START, STOP, or ABORT",
    )
    parser.add_argument(
        "--session-type",
        choices=("commissioning", "final"),
        default="commissioning",
        help="final is accepted by firmware only after the final actuator contract is frozen",
    )
    args = parser.parse_args()
    if args.passive and args.stop_after > 0.0:
        parser.error("--stop-after cannot stop an autonomous one-way session")
    session_type = (
        COMMISSIONING_SESSION
        if args.session_type == "commissioning"
        else FINAL_IDENTIFICATION_SESSION
    )
    capture(
        args.port,
        args.output,
        args.stop_after,
        args.idle_timeout,
        args.startup_timeout,
        session_type=session_type,
        passive=args.passive,
    )


if __name__ == "__main__":
    main()
