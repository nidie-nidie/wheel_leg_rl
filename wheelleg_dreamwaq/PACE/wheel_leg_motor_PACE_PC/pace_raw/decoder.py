from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import math
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

from .scanner import FrameScanner


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
    metadata: Optional[dict] = None
    source_path: Optional[pathlib.Path] = None

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
            "protocol_version": self.header.header.version if self.header else None,
            "segments": [stage.segment_id for stage in self.stage_configs.values()],
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
        rate = header.sample_rate_hz if header else 500
        expected = 1_000_000 / rate if rate > 0 else 0
        tolerance = max(1.0, expected * 0.25) if header and header.header.version == 2 else 1.0
        if delta <= 0:
            sample.fit_eligible = False
        elif expected <= 0 or abs(delta - expected) > tolerance:
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
        if stage.header.version == 2 and sample.active_mask != stage.group_mask:
            sample.fit_eligible = False
            issues.append(DecodeIssue(offset, "active_group_mismatch", "sample active mask differs from its stage group", sample.header.sequence))
        if stage.header.version == 2 and stage.config_seq != next(reversed(stage_configs)):
            sample.fit_eligible = False
            issues.append(DecodeIssue(offset, "stale_stage_config", "sample references an earlier segment after a new stage", sample.header.sequence))
    sample.role = (
        {0: "DIAGNOSTIC", 1: "FIT", 2: "VALIDATION", 3: "STATIC", 4: "STOP"}.get(stage.experiment_role, "DIAGNOSTIC")
        if stage and stage.header.version == 2 else
        _role_for_sample(header.session_type if header else None, sample.stage_id)
    )
    if any(mask & ~0x3F for mask in (sample.active_mask, sample.online_mask, sample.tx_valid_mask, sample.rx_valid_mask, sample.saturation_mask)):
        issues.append(DecodeIssue(offset, "invalid_sample_mask", "sample motor mask has undefined bits", sample.header.sequence))
        sample.fit_eligible = False
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

    scanner = FrameScanner()
    scanned = scanner.feed(data) + scanner.finish()
    issues.extend(DecodeIssue(issue.offset, issue.code, issue.message) for issue in scanner.issues)
    closed_offset = None
    segment_ids = set()
    for offset, frame in scanned:
        frame_length = frame.header.frame_length
        if closed_offset is not None:
            issues.append(DecodeIssue(offset, "post_footer_data", "frame follows closed session", frame.header.sequence))
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
        if session_header is not None and frame.header.version != session_header.header.version:
            issues.append(DecodeIssue(offset, "mixed_protocol_version", "frame version differs from session header", sequence))

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
                    issues.append(DecodeIssue(offset, "header_rate", "unsupported sample/status rates", sequence))
                if frame.header.version == 2 and (frame.timestamp_timebase != 1 or frame.control_rate_hz != 500):
                    issues.append(DecodeIssue(offset, "header_timebase", "V2 requires TIM5 service time and 500 Hz control", sequence))
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
            if any(value > 4095 for value in (*frame.dm_kp_raw, *frame.dm_kd_raw)):
                issues.append(DecodeIssue(offset, "invalid_controller_gain", "controller gain raw value exceeds 12-bit range", sequence))
            if any(mode != 1 for mode in frame.command_mode[:4]) or any(mode not in (5, 6) for mode in frame.command_mode[4:]):
                issues.append(DecodeIssue(offset, "invalid_command_mode", "unsupported motor command mode", sequence))
            if any(rate <= 0 or rate > 500 or 500 % rate for rate in frame.command_rate_hz):
                issues.append(DecodeIssue(offset, "invalid_command_rate", "invalid CAN command rate", sequence))
            values = (frame.amplitude, frame.frequency_start_hz, frame.frequency_end_hz)
            if not all(math.isfinite(value) for value in values) or frame.duration_ms <= 0:
                issues.append(DecodeIssue(offset, "invalid_stage", "invalid trajectory values/duration", sequence))
            if frame.header.version == 2:
                if frame.config_seq == 0:
                    issues.append(DecodeIssue(offset, "invalid_config_seq", "V2 config_seq must be nonzero", sequence))
                if frame.segment_id == 0 or frame.segment_id in segment_ids:
                    issues.append(DecodeIssue(offset, "duplicate_segment", "segment ID must be nonzero and unique", sequence))
                segment_ids.add(frame.segment_id)
                if frame.experiment_role not in range(5) or frame.group_mask & ~0x3F:
                    issues.append(DecodeIssue(offset, "invalid_stage_role", "invalid role/group mask", sequence))
                allowed_stages = {1: (2,4,5), 2: (3,4,5), 3: (1,), 4: (6,)}
                if frame.stage_id not in range(1,7) or (frame.experiment_role in allowed_stages and frame.stage_id not in allowed_stages[frame.experiment_role]):
                    issues.append(DecodeIssue(offset, "invalid_stage_role", "stage category disagrees with experiment role", sequence))
                if frame.action_update_rate_hz != 50 or frame.command_hold_method != 1 or frame.center_source not in (1, 2):
                    issues.append(DecodeIssue(offset, "invalid_control_contract", "V2 requires declared 50 Hz ZOH and center source", sequence))
                if not all(math.isfinite(value) for value in (*frame.dm_center_rad, *frame.excitation_multiplier, frame.phase_rad)):
                    issues.append(DecodeIssue(offset, "invalid_stage", "nonfinite V2 center/excitation", sequence))
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
            closed_offset = offset + frame_length
            if frame.total_frames != sequence + 1:
                issues.append(DecodeIssue(offset, "footer_frame_count", "footer total_frames disagrees with its sequence", sequence))
            if session_header and frame.experiment_config_hash != session_header.experiment_config_hash:
                issues.append(DecodeIssue(offset, "config_hash_mismatch", "footer config hash differs from session header", sequence))
            if frame.overflow or frame.dropped_frames or not frame.statistics_complete:
                issues.append(DecodeIssue(offset, "invalid_session_footer", "footer marks the session incomplete", sequence))
        offset += frame_length

    if closed_offset is not None and closed_offset != len(data):
        if not any(issue.code == "post_footer_data" for issue in issues):
            issues.append(DecodeIssue(closed_offset, "post_footer_data", "bytes follow closed session"))
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
    decoded = decode_bytes(path.read_bytes())
    decoded.source_path = path
    from .session_metadata import load_session_metadata
    decoded.metadata = load_session_metadata(path)
    return decoded


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
