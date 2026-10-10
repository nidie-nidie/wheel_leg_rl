from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import struct
from typing import Dict, List, Optional

from .schema import (
    MAGIC,
    STATUS_MAX_SIZE,
    FooterFrame,
    Frame,
    FrameDecodeError,
    EventFrame,
    SampleFrame,
    SessionHeaderFrame,
    StageConfigFrame,
    StatusFrame,
    decode_frame,
)


@dataclasses.dataclass(frozen=True)
class DecodeIssue:
    offset: int
    code: str
    message: str
    sequence: Optional[int] = None


@dataclasses.dataclass
class DecodedSession:
    raw_stream: bytes
    frames: List[Frame]
    samples: List[SampleFrame]
    statuses: List[StatusFrame]
    events: List[EventFrame]
    stage_configs: Dict[int, StageConfigFrame]
    header: Optional[SessionHeaderFrame]
    footer: Optional[FooterFrame]
    issues: List[DecodeIssue]

    @property
    def fit_samples(self) -> List[SampleFrame]:
        return [sample for sample in self.samples if sample.fit_eligible]

    def summary(self) -> Dict[str, object]:
        issue_counts: Dict[str, int] = {}
        for issue in self.issues:
            issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1
        tx_count = [0] * 6
        rx_count = [0] * 6
        for status in self.statuses:
            for index in range(6):
                tx_count[index] += status.tx_count_delta[index]
                rx_count[index] += status.rx_count_delta[index]
        return {
            "frames": len(self.frames),
            "samples": len(self.samples),
            "fit_samples": len(self.fit_samples),
            "statuses": len(self.statuses),
            "events": len(self.events),
            "stage_configs": sorted(self.stage_configs),
            "session_type": self.header.session_type if self.header else None,
            "footer_present": self.footer is not None,
            "communication": {
                "tx_count_delta_total": tx_count,
                "rx_count_delta_total": rx_count,
                "enqueue_fail_delta_total": sum(
                    status.enqueue_fail_delta for status in self.statuses
                ),
                "tx_event_lost_delta_total": sum(
                    status.tx_event_lost_delta for status in self.statuses
                ),
                "can_error_delta_total": sum(
                    status.can_error_delta for status in self.statuses
                ),
                "tx_fifo_high_water": max(
                    (status.tx_fifo_high_water for status in self.statuses),
                    default=0,
                ),
            },
            "issues": issue_counts,
        }


def _role_for_sample(session_type: Optional[int], stage_id: int) -> str:
    if session_type == 2:
        if stage_id in (2, 4):
            return "FIT"
        if stage_id in (3, 5):
            return "VALIDATION"
        return "DIAGNOSTIC"
    if stage_id in (2, 4, 5):
        return "PROVISIONAL_FIT"
    if stage_id == 3:
        return "VALIDATION"
    return "DIAGNOSTIC"


def _reconstruct_age(sample: SampleFrame, block, motor_index: int, tx: bool) -> None:
    valid_mask = sample.tx_valid_mask if tx else sample.rx_valid_mask
    age = block.tx_age_us if tx else block.rx_age_us
    valid = (valid_mask & (1 << motor_index)) != 0
    saturated = valid and age == 0xFFFF
    timestamp = None
    if valid and not saturated and sample.extended_time_us is not None:
        timestamp = sample.extended_time_us - age
    if tx:
        block.tx_time_us = timestamp
        block.tx_age_saturated = saturated
    else:
        block.rx_time_us = timestamp
        block.rx_age_saturated = saturated


def _attach_sample_context(
    sample: SampleFrame,
    offset: int,
    header: Optional[SessionHeaderFrame],
    stage_configs: Dict[int, StageConfigFrame],
    issues: List[DecodeIssue],
    last_raw_time: Optional[int],
    last_extended_time: Optional[int],
    wraps: int,
):
    raw_time = sample.sample_time_us
    if last_raw_time is not None and raw_time < last_raw_time:
        if (last_raw_time - raw_time) > 0x80000000:
            wraps += 1
        else:
            sample.fit_eligible = False
            issues.append(DecodeIssue(offset, "timestamp_backwards", "sample time moved backwards", sample.header.sequence))
    sample.extended_time_us = raw_time + wraps * (1 << 32)
    if last_extended_time is not None:
        delta = sample.extended_time_us - last_extended_time
        expected = 1_000_000 // (header.sample_rate_hz if header else 500)
        if delta <= 0:
            sample.fit_eligible = False
        elif abs(delta - expected) > 1:
            sample.fit_eligible = False
            issues.append(
                DecodeIssue(
                    offset,
                    "timestamp_discontinuity",
                    f"sample interval is {delta} us, expected {expected} us",
                    sample.header.sequence,
                )
            )

    stage = stage_configs.get(sample.config_seq)
    if stage is None:
        sample.fit_eligible = False
        issues.append(
            DecodeIssue(offset, "missing_stage_config", f"config_seq {sample.config_seq} is unknown", sample.header.sequence)
        )
    elif stage.stage_id != sample.stage_id:
        sample.fit_eligible = False
        issues.append(
            DecodeIssue(offset, "stage_mismatch", "sample stage_id disagrees with its stage config", sample.header.sequence)
        )
    else:
        sample.stage_config = stage
    sample.role = _role_for_sample(header.session_type if header else None, sample.stage_id)
    if sample.safety_flags or sample.can_error_flags or sample.saturation_mask:
        sample.fit_eligible = False

    for index, block in enumerate(sample.dm):
        _reconstruct_age(sample, block, index, True)
        _reconstruct_age(sample, block, index, False)
    for index, block in enumerate(sample.lk, start=4):
        _reconstruct_age(sample, block, index, True)
        _reconstruct_age(sample, block, index, False)
    return raw_time, sample.extended_time_us, wraps


def decode_bytes(data: bytes) -> DecodedSession:
    frames: List[Frame] = []
    samples: List[SampleFrame] = []
    statuses: List[StatusFrame] = []
    events: List[EventFrame] = []
    stage_configs: Dict[int, StageConfigFrame] = {}
    issues: List[DecodeIssue] = []
    session_header: Optional[SessionHeaderFrame] = None
    footer: Optional[FooterFrame] = None
    expected_sequence: Optional[int] = None
    last_raw_time: Optional[int] = None
    last_extended_time: Optional[int] = None
    wraps = 0
    offset = 0

    while offset < len(data):
        magic_offset = data.find(MAGIC, offset)
        if magic_offset < 0:
            issues.append(DecodeIssue(offset, "trailing_bytes", f"{len(data) - offset} non-frame bytes at stream end"))
            break
        if magic_offset > offset:
            issues.append(DecodeIssue(offset, "resync", f"skipped {magic_offset - offset} bytes before magic"))
            offset = magic_offset
        if len(data) - offset < 6:
            issues.append(DecodeIssue(offset, "truncated_header", "stream ended before frame length"))
            break
        frame_length = struct.unpack_from("<H", data, offset + 4)[0]
        if frame_length < 14 or frame_length > STATUS_MAX_SIZE:
            issues.append(DecodeIssue(offset, "invalid_length", f"implausible frame length {frame_length}"))
            offset += 1
            continue
        if offset + frame_length > len(data):
            issues.append(DecodeIssue(offset, "truncated_frame", f"need {frame_length} bytes, stream ended early"))
            break
        raw_frame = data[offset : offset + frame_length]
        sequence_hint = struct.unpack_from("<I", raw_frame, 6)[0] if frame_length >= 10 else None
        try:
            frame = decode_frame(raw_frame)
        except FrameDecodeError as error:
            code = "crc_error" if "CRC" in str(error) else "malformed_frame"
            issues.append(DecodeIssue(offset, code, str(error), sequence_hint))
            offset += frame_length
            continue

        sequence = frame.header.sequence
        if expected_sequence is None:
            if sequence != 0:
                issues.append(DecodeIssue(offset, "sequence_start", f"stream starts at sequence {sequence}", sequence))
        elif sequence != expected_sequence:
            code = "sequence_gap" if sequence > expected_sequence else "sequence_reorder"
            issues.append(
                DecodeIssue(offset, code, f"expected sequence {expected_sequence}, received {sequence}", sequence)
            )
        expected_sequence = (sequence + 1) & 0xFFFFFFFF
        frames.append(frame)

        if isinstance(frame, SessionHeaderFrame):
            if session_header is not None:
                issues.append(DecodeIssue(offset, "duplicate_header", "multiple session headers", sequence))
            else:
                session_header = frame
                if frame.wire_endianness != 1:
                    issues.append(DecodeIssue(offset, "header_endianness", "session is not little-endian", sequence))
                # Protocol v1 recordings exist at the original wired 500 Hz
                # rate and at the wireless-safe 250 Hz export rate.
                if frame.sample_rate_hz not in (250, 500) or frame.status_rate_hz != 10:
                    issues.append(DecodeIssue(offset, "header_rate", "session rates do not match protocol v1", sequence))
                if frame.experiment_config_hash == 0:
                    issues.append(DecodeIssue(offset, "zero_config_hash", "session config hash is zero", sequence))
                if frame.sample_frame_size_bytes != 126:
                    issues.append(DecodeIssue(offset, "header_sample_size", "header does not declare 126-byte samples", sequence))
        elif isinstance(frame, StageConfigFrame):
            if frame.config_seq in stage_configs:
                issues.append(DecodeIssue(offset, "duplicate_config", f"config_seq {frame.config_seq} repeated", sequence))
            if session_header is None:
                issues.append(DecodeIssue(offset, "stage_before_header", "stage config precedes session header", sequence))
            elif frame.experiment_config_hash != session_header.experiment_config_hash:
                issues.append(DecodeIssue(offset, "config_hash_mismatch", "stage config hash differs from session header", sequence))
            stage_configs[frame.config_seq] = frame
        elif isinstance(frame, SampleFrame):
            if session_header is None:
                frame.fit_eligible = False
                issues.append(DecodeIssue(offset, "missing_session_header", "sample precedes session header", sequence))
            last_raw_time, last_extended_time, wraps = _attach_sample_context(
                frame,
                offset,
                session_header,
                stage_configs,
                issues,
                last_raw_time,
                last_extended_time,
                wraps,
            )
            samples.append(frame)
        elif isinstance(frame, StatusFrame):
            statuses.append(frame)
            if frame.enqueue_fail_delta:
                issues.append(DecodeIssue(offset, "can_enqueue_failure", "status reports a CAN enqueue failure", sequence))
            if frame.tx_event_lost_delta:
                issues.append(DecodeIssue(offset, "tx_event_loss", "status reports a lost Tx Event", sequence))
            if frame.can_error_delta:
                issues.append(DecodeIssue(offset, "can_error", "status reports a CAN/HAL error", sequence))
            if frame.can_state & 0x06:
                issues.append(DecodeIssue(offset, "can_state_fault", "status reports CAN passive or bus-off", sequence))
        elif isinstance(frame, EventFrame):
            events.append(frame)
        elif isinstance(frame, FooterFrame):
            if footer is not None:
                issues.append(DecodeIssue(offset, "duplicate_footer", "multiple session footers", sequence))
            footer = frame
            if frame.total_frames != sequence + 1:
                issues.append(DecodeIssue(offset, "footer_frame_count", "footer total_frames disagrees with its sequence", sequence))
            if session_header and frame.experiment_config_hash != session_header.experiment_config_hash:
                issues.append(DecodeIssue(offset, "config_hash_mismatch", "footer config hash differs from session header", sequence))
            if frame.overflow or frame.dropped_frames or not frame.statistics_complete:
                issues.append(DecodeIssue(offset, "invalid_session_footer", "footer marks the session incomplete", sequence))
        offset += frame_length

    if frames and session_header is None:
        issues.append(DecodeIssue(0, "missing_session_header", "stream has no session header"))
    if frames and footer is None:
        issues.append(DecodeIssue(len(data), "missing_footer", "stream has no session footer"))
    return DecodedSession(
        data,
        frames,
        samples,
        statuses,
        events,
        stage_configs,
        session_header,
        footer,
        issues,
    )


def decode_file(path: pathlib.Path) -> DecodedSession:
    path = pathlib.Path(path)
    return decode_bytes(path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode a wheel_leg_motor_PACE raw binary stream")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--summary-json", type=pathlib.Path)
    args = parser.parse_args()
    decoded = decode_file(args.input)
    summary = decoded.summary()
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.summary_json:
        args.summary_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
