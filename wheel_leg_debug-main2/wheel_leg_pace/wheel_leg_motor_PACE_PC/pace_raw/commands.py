from __future__ import annotations

import struct
import zlib


MAGIC = b"PC"
VERSION = 1
CONFIGURE = 0x01
START = 0x02
STOP = 0x03
STATUS = 0x04
ABORT = 0x05
COMMISSIONING_SESSION = 1
FINAL_IDENTIFICATION_SESSION = 2


def encode_command(command_id: int, sequence: int, payload: bytes = b"") -> bytes:
    if command_id not in (CONFIGURE, START, STOP, STATUS, ABORT):
        raise ValueError(f"unknown host command {command_id}")
    payload = bytes(payload)
    length = 14 + len(payload)
    if length > 128:
        raise ValueError("host command exceeds 128-byte protocol limit")
    frame = bytearray(length)
    struct.pack_into("<2sBBHI", frame, 0, MAGIC, VERSION, command_id, length, sequence & 0xFFFFFFFF)
    frame[10 : 10 + len(payload)] = payload
    struct.pack_into("<I", frame, length - 4, zlib.crc32(frame[:-4]) & 0xFFFFFFFF)
    return bytes(frame)


def configure(sequence: int, session_type: int = COMMISSIONING_SESSION) -> bytes:
    if session_type not in (COMMISSIONING_SESSION, FINAL_IDENTIFICATION_SESSION):
        raise ValueError("invalid session type")
    return encode_command(CONFIGURE, sequence, bytes([session_type]))


def start(sequence: int) -> bytes:
    return encode_command(START, sequence)


def stop(sequence: int) -> bytes:
    return encode_command(STOP, sequence)


def status(sequence: int) -> bytes:
    return encode_command(STATUS, sequence)


def abort(sequence: int) -> bytes:
    return encode_command(ABORT, sequence)
