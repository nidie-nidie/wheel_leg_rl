from __future__ import annotations

import dataclasses
import struct
import zlib
from typing import List, Optional, Tuple, Union


MAGIC = b"PA"
PROTOCOL_VERSION = 1
SESSION_HEADER = 0x01
SAMPLE = 0x02
STAGE_CONFIG = 0x03
STATUS = 0x04
EVENT = 0x05
FOOTER = 0x06

SESSION_HEADER_SIZE = 58
SAMPLE_SIZE = 126
STAGE_CONFIG_SIZE = 76
STATUS_MIN_SIZE = 64
STATUS_MAX_SIZE = 128
EVENT_SIZE = 30
FOOTER_SIZE = 34

FRAME_SIZES = {
    SESSION_HEADER: SESSION_HEADER_SIZE,
    SAMPLE: SAMPLE_SIZE,
    STAGE_CONFIG: STAGE_CONFIG_SIZE,
    EVENT: EVENT_SIZE,
    FOOTER: FOOTER_SIZE,
}


class FrameDecodeError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class CommonHeader:
    frame_type: int
    frame_length: int
    sequence: int


@dataclasses.dataclass
class SessionHeaderFrame:
    header: CommonHeader
    session_type: int
    motor_order_version: int
    wire_endianness: int
    session_id: int
    sample_rate_hz: int
    status_rate_hz: int
    start_timestamp_us: int
    experiment_config_hash: int
    sample_frame_size_bytes: int
    uart_baud: int
    can_nominal_bitrate: int
    scale_manifest_hash: int
    firmware_build_id: int
    robot_variant: int
    raw: bytes


@dataclasses.dataclass
class DmRawBlock:
    cmd_q_raw: int
    cmd_dq_raw: int
    cmd_tau_raw: int
    fb_q_raw: int
    fb_dq_raw: int
    fb_tau_raw: int
    tx_age_us: int
    rx_age_us: int
    tx_time_us: Optional[int] = None
    rx_time_us: Optional[int] = None
    tx_age_saturated: bool = False
    rx_age_saturated: bool = False


@dataclasses.dataclass
class LkRawBlock:
    cmd_primary_raw: int
    fb_encoder_raw: int
    fb_turn_count: int
    fb_speed_raw: int
    fb_iq_raw: int
    tx_age_us: int
    rx_age_us: int
    tx_time_us: Optional[int] = None
    rx_time_us: Optional[int] = None
    tx_age_saturated: bool = False
    rx_age_saturated: bool = False


@dataclasses.dataclass
class SampleFrame:
    header: CommonHeader
    sample_time_us: int
    stage_id: int
    config_seq: int
    active_mask: int
    online_mask: int
    tx_valid_mask: int
    rx_valid_mask: int
    saturation_mask: int
    safety_flags: int
    can_error_flags: int
    dm: List[DmRawBlock]
    lk: List[LkRawBlock]
    raw: bytes
    extended_time_us: Optional[int] = None
    stage_config: Optional["StageConfigFrame"] = None
    role: str = "DIAGNOSTIC"
    fit_eligible: bool = True


@dataclasses.dataclass
class StageConfigFrame:
    header: CommonHeader
    config_seq: int
    stage_id: int
    command_mode: Tuple[int, ...]
    command_rate_hz: Tuple[int, ...]
    dm_kp_raw: Tuple[int, ...]
    dm_kd_raw: Tuple[int, ...]
    excitation_type: int
    amplitude: float
    frequency_start_hz: float
    frequency_end_hz: float
    duration_ms: int
    safety_limit_set_id: int
    experiment_config_hash: int
    raw: bytes


@dataclasses.dataclass
class StatusFrame:
    header: CommonHeader
    status_time_us: int
    motor_state_nibbles: Tuple[int, ...]
    dm_temp_mos: Tuple[int, ...]
    dm_temp_rotor: Tuple[int, ...]
    lk_temp: Tuple[int, ...]
    tx_count_delta: Tuple[int, ...]
    rx_count_delta: Tuple[int, ...]
    enqueue_fail_delta: int
    tx_event_lost_delta: int
    can_error_delta: int
    tx_fifo_high_water: int
    can_state: int
    extension_tlv: bytes
    raw: bytes


@dataclasses.dataclass
class EventFrame:
    header: CommonHeader
    event_time_us: int
    event_code: int
    severity: int
    motor_index: int
    argument0: int
    argument1: int
    raw: bytes


@dataclasses.dataclass
class FooterFrame:
    header: CommonHeader
    stop_time_us: int
    total_frames: int
    dropped_frames: int
    overflow: bool
    statistics_complete: bool
    stop_reason: int
    experiment_config_hash: int
    raw: bytes


Frame = Union[
    SessionHeaderFrame,
    SampleFrame,
    StageConfigFrame,
    StatusFrame,
    EventFrame,
    FooterFrame,
]


def crc32(frame_without_crc: bytes) -> int:
    return zlib.crc32(frame_without_crc) & 0xFFFFFFFF


def _decode_common(frame: bytes) -> CommonHeader:
    if len(frame) < 14:
        raise FrameDecodeError("frame shorter than common header plus CRC")
    magic, version, frame_type, frame_length, sequence = struct.unpack_from("<2sBBHI", frame, 0)
    if magic != MAGIC:
        raise FrameDecodeError("bad magic")
    if version != PROTOCOL_VERSION:
        raise FrameDecodeError("unsupported protocol version")
    if frame_length != len(frame):
        raise FrameDecodeError("declared frame length does not match available bytes")
    if frame_type in FRAME_SIZES and frame_length != FRAME_SIZES[frame_type]:
        raise FrameDecodeError("fixed frame type has an invalid length")
    if frame_type == STATUS:
        if not STATUS_MIN_SIZE <= frame_length <= STATUS_MAX_SIZE:
            raise FrameDecodeError("status length is outside 64..128 bytes")
        if frame_length != STATUS_MIN_SIZE + frame[59]:
            raise FrameDecodeError("status extension length is inconsistent")
    elif frame_type not in FRAME_SIZES:
        raise FrameDecodeError("unknown frame type")
    expected_crc = struct.unpack_from("<I", frame, frame_length - 4)[0]
    if expected_crc != crc32(frame[:-4]):
        raise FrameDecodeError("CRC mismatch")
    return CommonHeader(frame_type, frame_length, sequence)


def _decode_session(frame: bytes, header: CommonHeader) -> SessionHeaderFrame:
    return SessionHeaderFrame(
        header=header,
        session_type=frame[10],
        motor_order_version=frame[11],
        wire_endianness=frame[12],
        session_id=struct.unpack_from("<I", frame, 14)[0],
        sample_rate_hz=struct.unpack_from("<H", frame, 18)[0],
        status_rate_hz=struct.unpack_from("<H", frame, 20)[0],
        start_timestamp_us=struct.unpack_from("<I", frame, 22)[0],
        experiment_config_hash=struct.unpack_from("<I", frame, 26)[0],
        sample_frame_size_bytes=struct.unpack_from("<H", frame, 30)[0],
        uart_baud=struct.unpack_from("<I", frame, 34)[0],
        can_nominal_bitrate=struct.unpack_from("<I", frame, 38)[0],
        scale_manifest_hash=struct.unpack_from("<I", frame, 42)[0],
        firmware_build_id=struct.unpack_from("<I", frame, 46)[0],
        robot_variant=struct.unpack_from("<I", frame, 50)[0],
        raw=frame,
    )


def _decode_sample(frame: bytes, header: CommonHeader) -> SampleFrame:
    fields = struct.unpack_from("<IBBBBBBBBHH", frame, 10)
    dm = [DmRawBlock(*struct.unpack_from("<8H", frame, 26 + index * 16)) for index in range(4)]
    lk = [LkRawBlock(*struct.unpack_from("<iHhhhHH", frame, 90 + index * 16)) for index in range(2)]
    return SampleFrame(
        header=header,
        sample_time_us=fields[0],
        stage_id=fields[1],
        config_seq=fields[2],
        active_mask=fields[3],
        online_mask=fields[4],
        tx_valid_mask=fields[5],
        rx_valid_mask=fields[6],
        saturation_mask=fields[7],
        safety_flags=fields[9],
        can_error_flags=fields[10],
        dm=dm,
        lk=lk,
        raw=frame,
    )


def _decode_stage(frame: bytes, header: CommonHeader) -> StageConfigFrame:
    return StageConfigFrame(
        header=header,
        config_seq=frame[10],
        stage_id=frame[11],
        command_mode=tuple(frame[12:18]),
        command_rate_hz=struct.unpack_from("<6H", frame, 18),
        dm_kp_raw=struct.unpack_from("<4H", frame, 30),
        dm_kd_raw=struct.unpack_from("<4H", frame, 38),
        excitation_type=frame[46],
        amplitude=struct.unpack_from("<f", frame, 48)[0],
        frequency_start_hz=struct.unpack_from("<f", frame, 52)[0],
        frequency_end_hz=struct.unpack_from("<f", frame, 56)[0],
        duration_ms=struct.unpack_from("<I", frame, 60)[0],
        safety_limit_set_id=struct.unpack_from("<H", frame, 64)[0],
        experiment_config_hash=struct.unpack_from("<I", frame, 68)[0],
        raw=frame,
    )


def _decode_status(frame: bytes, header: CommonHeader) -> StatusFrame:
    extension_length = frame[59]
    return StatusFrame(
        header=header,
        status_time_us=struct.unpack_from("<I", frame, 10)[0],
        motor_state_nibbles=tuple(frame[14:17]),
        dm_temp_mos=struct.unpack_from("<4b", frame, 17),
        dm_temp_rotor=struct.unpack_from("<4b", frame, 21),
        lk_temp=struct.unpack_from("<2b", frame, 25),
        tx_count_delta=struct.unpack_from("<6H", frame, 27),
        rx_count_delta=struct.unpack_from("<6H", frame, 39),
        enqueue_fail_delta=struct.unpack_from("<H", frame, 51)[0],
        tx_event_lost_delta=struct.unpack_from("<H", frame, 53)[0],
        can_error_delta=struct.unpack_from("<H", frame, 55)[0],
        tx_fifo_high_water=frame[57],
        can_state=frame[58],
        extension_tlv=frame[60 : 60 + extension_length],
        raw=frame,
    )


def decode_frame(frame: bytes) -> Frame:
    header = _decode_common(frame)
    if header.frame_type == SESSION_HEADER:
        return _decode_session(frame, header)
    if header.frame_type == SAMPLE:
        return _decode_sample(frame, header)
    if header.frame_type == STAGE_CONFIG:
        return _decode_stage(frame, header)
    if header.frame_type == STATUS:
        return _decode_status(frame, header)
    if header.frame_type == EVENT:
        values = struct.unpack_from("<IHBBII", frame, 10)
        return EventFrame(header, *values, raw=frame)
    if header.frame_type == FOOTER:
        stop_time, total, dropped, overflow, complete, reason, config_hash = struct.unpack_from(
            "<IIIBBHI", frame, 10
        )
        return FooterFrame(
            header, stop_time, total, dropped, bool(overflow), bool(complete),
            reason, config_hash, frame
        )
    raise FrameDecodeError("unreachable frame type")


def frame_length_from_prefix(prefix: bytes) -> int:
    if len(prefix) < 6:
        raise FrameDecodeError("not enough bytes for length")
    return struct.unpack_from("<H", prefix, 4)[0]
