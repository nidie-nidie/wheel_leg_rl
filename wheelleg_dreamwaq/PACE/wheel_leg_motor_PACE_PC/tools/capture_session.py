from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace_raw.commands import COMMISSIONING_SESSION, FINAL_IDENTIFICATION_SESSION, abort, configure, start, stop
from pace_raw.decoder import decode_bytes
from pace_raw.manifest import load_firmware_manifest
from pace_raw.quality import quality_report
from pace_raw.scanner import FrameScanner
from pace_raw.schema import FooterFrame, SessionHeaderFrame
from pace_raw.session_metadata import build_session_metadata, manifest_snapshot, write_session_metadata, sidecar_path, quality_path


def next_available_output(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    def available(candidate):
        return not candidate.exists() and not sidecar_path(candidate).exists() and not quality_path(candidate).exists()
    if available(path):
        return path
    stem = path.stem
    base, separator, number_text = stem.rpartition("_")
    if separator and number_text.isdigit():
        number, width = int(number_text) + 1, len(number_text)
    else:
        base, number, width = stem, 1, 3
    while True:
        candidate = path.with_name(f"{base}_{number:0{width}d}{path.suffix}")
        if available(candidate):
            return candidate
        number += 1


class FooterDetector:
    def __init__(self):
        self.scanner = FrameScanner()
        self.footer_seen = False
        self.header_seen = False

    def _accept(self, frames):
        for _, frame in frames:
            if isinstance(frame, SessionHeaderFrame):
                self.header_seen = True
            if isinstance(frame, FooterFrame) and self.header_seen:
                self.footer_seen = True

    def feed(self, data):
        self._accept(self.scanner.feed(data))

    def finish(self):
        self._accept(self.scanner.finish())


def _source_hashes(manifest):
    root = manifest.source_path.parents[1]
    result = {}
    for folder in ("Config", "Protocol", "Motor", "PACE", "Board"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in (".c", ".h"):
                result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def capture(port, output, stop_after=0.0, idle_timeout=5.0, startup_timeout=15.0,
            session_type=COMMISSIONING_SESSION, passive=True, metadata_plan=None,
            serial_factory=None, clock=None, sleeper=None):
    if passive and stop_after > 0:
        raise ValueError("passive one-way capture cannot stop the firmware")
    if min(idle_timeout, startup_timeout) <= 0:
        raise ValueError("capture timeouts must be positive")
    if serial_factory is None:
        import serial
        serial_factory = serial.Serial
    clock, sleeper = clock or time.monotonic, sleeper or time.sleep
    output = next_available_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    firmware = load_firmware_manifest()
    snapshot = manifest_snapshot(firmware)
    sources = _source_hashes(firmware)
    plan = dict(metadata_plan or {})
    print(f"Saving raw capture to: {output.resolve()}", flush=True)
    detector = FooterDetector()
    sequence, stop_sent, received = 0, False, False
    capture_state = "error"
    try:
        with output.open("xb") as stream, serial_factory(port=port, baudrate=921600, timeout=0.05) as device:
            # Never clear pre-existing bytes: an autonomous header may already be buffered.
            started_at = last_data_at = clock()
            if not passive:
                device.write(configure(sequence, session_type=session_type))
                sequence += 1
                device.flush()
                sleeper(0.05)
                device.write(start(sequence))
                sequence += 1
                device.flush()
            try:
                while not detector.footer_seen:
                    chunk = device.read(4096)
                    now = clock()
                    if chunk:
                        stream.write(chunk)
                        stream.flush()
                        detector.feed(chunk)
                        received, last_data_at = True, now
                    if stop_after > 0 and not stop_sent and now - started_at >= stop_after:
                        device.write(stop(sequence))
                        sequence += 1
                        device.flush()
                        stop_sent = True
                    limit = idle_timeout if received else max(startup_timeout, 15.0)
                    if not detector.footer_seen and now - last_data_at >= limit:
                        detector.finish()
                        if detector.footer_seen:
                            break
                        capture_state = "timeout"
                        if not passive:
                            device.write(abort(sequence))
                            device.flush()
                        raise TimeoutError(f"no UART data for {limit:.1f} s; incomplete raw retained")
                capture_state = "complete"
            except KeyboardInterrupt:
                capture_state = "interrupted"
                if not passive:
                    device.write(stop(sequence))
                    device.flush()
                    deadline = clock() + 3.0
                    while clock() < deadline and not detector.footer_seen:
                        chunk = device.read(4096)
                        if chunk:
                            stream.write(chunk)
                            stream.flush()
                            detector.feed(chunk)
                    if detector.footer_seen:
                        capture_state = "complete"
                if capture_state != "complete":
                    raise RuntimeError("capture interrupted without a footer; incomplete raw retained")
    finally:
        if output.exists():
            decoded = decode_bytes(output.read_bytes())
            metadata = build_session_metadata(decoded, snapshot, capture_state=capture_state, plan=plan, firmware_sources=sources)
            decoded.metadata = metadata
            metadata["quality_report"] = quality_report(decoded)
            metadata.pop("metadata_sha256")
            from pace_raw.session_metadata import content_hash
            metadata["metadata_sha256"] = content_hash(metadata)
            write_session_metadata(output, metadata)
            with quality_path(output).open("x", encoding="utf-8") as report_stream:
                json.dump(metadata["quality_report"], report_stream, indent=2, sort_keys=True, allow_nan=False)
                report_stream.write("\n")
    if decoded.footer is None or not metadata["quality_report"]["fit_quality_passed"]:
        print("Capture retained for diagnosis; quality gate rejected fitting.", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="Capture one autonomous PACE V2 session and immutable source sidecar")
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--active", action="store_true", help="explicitly send commands; requires bidirectional command firmware")
    mode.add_argument("--passive", action="store_true", help="receive only (default)")
    parser.add_argument("--stop-after", type=float, default=0.0)
    parser.add_argument("--idle-timeout", type=float, default=5.0)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--session-type", choices=("commissioning", "final"), default="commissioning")
    parser.add_argument("--metadata-plan", type=pathlib.Path, help="expected recipe/fixture/provenance; never overrides wire fields")
    args = parser.parse_args()
    if not args.active and args.stop_after > 0:
        parser.error("--stop-after requires --active")
    plan = json.loads(args.metadata_plan.read_text(encoding="utf-8")) if args.metadata_plan else None
    output = capture(args.port, args.output, args.stop_after, args.idle_timeout, args.startup_timeout,
            FINAL_IDENTIFICATION_SESSION if args.session_type == "final" else COMMISSIONING_SESSION,
            passive=not args.active, metadata_plan=plan)
    if not json.loads(quality_path(output).read_text(encoding="utf-8"))["fit_quality_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
