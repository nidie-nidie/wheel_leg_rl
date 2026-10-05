#!/usr/bin/env python3
"""Common protocol, transport and Isaac Sim helpers for wheel-leg alignment tools."""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import select
import socket
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Iterable


# Keep the model beside these tools so the directory can be moved or shared as
# a self-contained alignment package.  Callers can still override it with
# ``--usd-path``.
DEFAULT_USD_PATH = Path(__file__).resolve().with_name(
    "wheel_leg_urdf4_cod_real_closed_chain_floating_base_artic.usd"
)

SOF = 0xAA55
VERSION = 1
MSG_COMMAND = 1
MSG_STATE = 2
HEADER_FMT = ">HBBHIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CRC_SIZE = 4

LEG_JOINTS = ("jIO", "jAG", "jIJ", "jAB")
WHEEL_JOINTS = ("jwheel_left", "jwheel_right")
MOTOR_JOINTS = LEG_JOINTS + WHEEL_JOINTS

CMD_PAYLOAD_FMT = ">BBH4f4f2f4f4f2f2f6f6f"
STATE_PAYLOAD_FMT = ">BBHI3f4f4f4f2f6f6ff6f"
CMD_PAYLOAD_SIZE = struct.calcsize(CMD_PAYLOAD_FMT)
STATE_PAYLOAD_SIZE = struct.calcsize(STATE_PAYLOAD_FMT)
CMD_FRAME_SIZE = HEADER_SIZE + CMD_PAYLOAD_SIZE + CRC_SIZE
STATE_FRAME_SIZE = HEADER_SIZE + STATE_PAYLOAD_SIZE + CRC_SIZE
REVOLUTE_TARGET_LIMIT_RAD = 2.0 * math.pi - 1e-6


def _add_local_venv_site_packages() -> None:
    """Expose pure-Python helper packages when running under Isaac Sim Python."""

    tool_dir = Path(__file__).resolve().parent
    candidates = [tool_dir / ".venv" / "Lib" / "site-packages"]
    candidates.extend((tool_dir / ".venv" / "lib").glob("python*/site-packages"))

    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def wrap_angle_rad(value: float) -> float:
    """Wrap a joint angle to [-pi, pi] for revolute-joint visualization."""

    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def clamp_revolute_target_rad(value: float) -> float | None:
    """Return a finite PhysX-safe revolute target, or None for invalid data."""

    value = float(value)
    if not math.isfinite(value):
        return None
    return max(-REVOLUTE_TARGET_LIMIT_RAD, min(REVOLUTE_TARGET_LIMIT_RAD, value))


@dataclasses.dataclass
class FrameHeader:
    """Frame header shared by command and state packets."""

    msg_type: int
    payload_len: int
    seq: int
    t_us: int


@dataclasses.dataclass
class StatePayload:
    """Lower computer feedback payload, big-endian, 152 bytes."""

    status: int
    fault_code: int
    cmd_seq_echo: int
    imu_gyro_rad_s: tuple[float, float, float]
    imu_quat_wxyz: tuple[float, float, float, float]
    active_leg_q_rad: tuple[float, float, float, float]
    active_leg_dq_rad_s: tuple[float, float, float, float]
    wheel_dq_rad_s: tuple[float, float]
    joint_tau_nm: tuple[float, float, float, float, float, float]
    motor_current_a: tuple[float, float, float, float, float, float]
    bus_voltage_v: float
    temperature_c: tuple[float, float, float, float, float, float]


@dataclasses.dataclass
class CmdPayload:
    """Upper computer command payload, big-endian, 140 bytes."""

    mode: int = 0
    ttl_ms: int = 50
    leg_q_des_rad: tuple[float, float, float, float] = (0.5235987756, 0.5235987756, -0.5235987756, -0.5235987756)
    leg_dq_des_rad_s: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    wheel_dq_des_rad_s: tuple[float, float] = (0.0, 0.0)
    leg_kp: tuple[float, float, float, float] = (35.0, 35.0, 35.0, 35.0)
    leg_kd: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    wheel_kp: tuple[float, float] = (1.0, 1.0)
    wheel_kd: tuple[float, float] = (0.0, 0.0)
    tau_limit_nm: tuple[float, float, float, float, float, float] = (45.0, 45.0, 45.0, 45.0, 15.0, 15.0)
    dq_limit_rad_s: tuple[float, float, float, float, float, float] = (25.0, 25.0, 25.0, 25.0, 20.0, 20.0)


def add_transport_args(parser: argparse.ArgumentParser, *, send: bool) -> None:
    """Add serial/UDP options used by all alignment tools."""

    parser.add_argument("--transport", choices=("serial", "udp", "none"), default="none")
    parser.add_argument("--serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--udp-bind-host", default="0.0.0.0")
    parser.add_argument("--udp-bind-port", type=int, default=15002)
    if send:
        parser.add_argument("--udp-remote-host", default="127.0.0.1")
        parser.add_argument("--udp-remote-port", type=int, default=15001)
    else:
        parser.add_argument("--udp-remote-host", default="")
        parser.add_argument("--udp-remote-port", type=int, default=0)


class Endpoint:
    """Small non-blocking byte transport wrapper."""

    def read(self) -> bytes:
        return b""

    def write(self, data: bytes) -> None:
        del data

    def close(self) -> None:
        pass


class SerialEndpoint(Endpoint):
    """Serial endpoint.

    Prefer pyserial when it exists. Isaac Sim's bundled Python often does not
    include it, so this falls back to a small Linux termios implementation.
    """

    def __init__(self, port: str, baudrate: int):
        self._serial = None
        self._fd: int | None = None
        try:
            _add_local_venv_site_packages()
            import serial

            self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=0, write_timeout=0)
        except ModuleNotFoundError as exc:
            if os.name == "nt":
                raise RuntimeError(
                    "pyserial is required for serial transport on Windows. "
                    "Run the tools from a directory with .venv installed, or install pyserial into Isaac Sim Python."
                ) from exc
            self._fd = self._open_posix_serial(port, baudrate)

    def read(self) -> bytes:
        if self._serial is not None:
            n = self._serial.in_waiting
            return self._serial.read(n if n > 0 else 1)
        if self._fd is None:
            return b""
        ready, _, _ = select.select([self._fd], [], [], 0)
        if not ready:
            return b""
        try:
            return os.read(self._fd, 4096)
        except BlockingIOError:
            return b""

    def write(self, data: bytes) -> None:
        if self._serial is not None:
            self._serial.write(data)
        elif self._fd is not None:
            os.write(self._fd, data)

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    @staticmethod
    def _open_posix_serial(port: str, baudrate: int) -> int:
        import termios

        baud_attr = {
            9600: termios.B9600,
            19200: termios.B19200,
            38400: termios.B38400,
            57600: termios.B57600,
            115200: termios.B115200,
            230400: getattr(termios, "B230400", termios.B115200),
            460800: getattr(termios, "B460800", termios.B115200),
            500000: getattr(termios, "B500000", termios.B115200),
            576000: getattr(termios, "B576000", termios.B115200),
            921600: getattr(termios, "B921600", termios.B115200),
            1000000: getattr(termios, "B1000000", termios.B115200),
        }.get(baudrate)
        if baud_attr is None:
            raise ValueError(f"Unsupported baudrate without pyserial: {baudrate}")

        fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CLOCAL | termios.CREAD | termios.CS8
        attrs[3] = 0
        attrs[4] = baud_attr
        attrs[5] = baud_attr
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIOFLUSH)
        return fd


class UdpEndpoint(Endpoint):
    """UDP endpoint. Uses last sender as remote when no remote is configured."""

    def __init__(self, bind_host: str, bind_port: int, remote_host: str = "", remote_port: int = 0):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((bind_host, bind_port))
        self._sock.setblocking(False)
        self._remote = (remote_host, remote_port) if remote_host and remote_port else None
        self._last_peer: tuple[str, int] | None = None

    def read(self) -> bytes:
        chunks = []
        while True:
            try:
                data, peer = self._sock.recvfrom(4096)
            except BlockingIOError:
                break
            self._last_peer = peer
            chunks.append(data)
        return b"".join(chunks)

    def write(self, data: bytes) -> None:
        peer = self._remote or self._last_peer
        if peer is not None:
            self._sock.sendto(data, peer)

    def close(self) -> None:
        self._sock.close()


def make_endpoint(args: argparse.Namespace) -> Endpoint:
    if args.transport == "serial":
        return SerialEndpoint(args.serial_port, args.baudrate)
    if args.transport == "udp":
        return UdpEndpoint(args.udp_bind_host, args.udp_bind_port, args.udp_remote_host, args.udp_remote_port)
    return Endpoint()


class FrameReader:
    """Incremental frame parser for AA55 big-endian packets."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[tuple[FrameHeader, bytes]]:
        self._buf.extend(data)
        frames: list[tuple[FrameHeader, bytes]] = []
        while True:
            sof_index = self._buf.find(b"\xaa\x55")
            if sof_index < 0:
                self._buf.clear()
                return frames
            if sof_index:
                del self._buf[:sof_index]
            if len(self._buf) < HEADER_SIZE:
                return frames
            sof, ver, msg_type, payload_len, seq, t_us = struct.unpack(HEADER_FMT, self._buf[:HEADER_SIZE])
            if sof != SOF or ver != VERSION or payload_len > 2048:
                del self._buf[0]
                continue
            frame_len = HEADER_SIZE + payload_len + CRC_SIZE
            if len(self._buf) < frame_len:
                return frames
            raw_no_crc = bytes(self._buf[: HEADER_SIZE + payload_len])
            got_crc = struct.unpack(">I", self._buf[HEADER_SIZE + payload_len : frame_len])[0]
            del self._buf[:frame_len]
            if (zlib.crc32(raw_no_crc) & 0xFFFFFFFF) != got_crc:
                continue
            header = FrameHeader(msg_type=msg_type, payload_len=payload_len, seq=seq, t_us=t_us)
            frames.append((header, raw_no_crc[HEADER_SIZE:]))


def pack_frame(msg_type: int, payload: bytes, seq: int) -> bytes:
    header = struct.pack(HEADER_FMT, SOF, VERSION, msg_type, len(payload), seq, int(time.time() * 1_000_000))
    raw = header + payload
    return raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)


def unpack_state(payload: bytes) -> StatePayload:
    if len(payload) != STATE_PAYLOAD_SIZE:
        raise ValueError(f"State payload must be {STATE_PAYLOAD_SIZE} bytes, got {len(payload)}")
    values = list(struct.unpack(STATE_PAYLOAD_FMT, payload))
    i = 0
    status = int(values[i]); i += 1
    i += 1  # reserved
    fault_code = int(values[i]); i += 1
    cmd_seq_echo = int(values[i]); i += 1
    imu_gyro = tuple(float(x) for x in values[i : i + 3]); i += 3
    imu_quat = tuple(float(x) for x in values[i : i + 4]); i += 4
    leg_q = tuple(float(x) for x in values[i : i + 4]); i += 4
    leg_dq = tuple(float(x) for x in values[i : i + 4]); i += 4
    wheel_dq = tuple(float(x) for x in values[i : i + 2]); i += 2
    tau = tuple(float(x) for x in values[i : i + 6]); i += 6
    current = tuple(float(x) for x in values[i : i + 6]); i += 6
    bus_voltage = float(values[i]); i += 1
    temp = tuple(float(x) for x in values[i : i + 6])
    return StatePayload(status, fault_code, cmd_seq_echo, imu_gyro, imu_quat, leg_q, leg_dq, wheel_dq, tau, current, bus_voltage, temp)


def pack_command(payload: CmdPayload) -> bytes:
    return struct.pack(
        CMD_PAYLOAD_FMT,
        int(payload.mode),
        0,
        int(payload.ttl_ms),
        *payload.leg_q_des_rad,
        *payload.leg_dq_des_rad_s,
        *payload.wheel_dq_des_rad_s,
        *payload.leg_kp,
        *payload.leg_kd,
        *payload.wheel_kp,
        *payload.wheel_kd,
        *payload.tau_limit_nm,
        *payload.dq_limit_rad_s,
    )


def quat_wxyz_to_rpy_rad(q: Iterable[float]) -> tuple[float, float, float]:
    """Convert scalar-first quaternion [w, x, y, z] to roll/pitch/yaw in rad."""

    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return 0.0, 0.0, 0.0
    w, x, y, z = w / n, x / n, y / n, z / n
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def open_stage_blocking(simulation_app, usd_path: Path):
    """Open a USD stage and wait until assets finish loading."""

    import omni.usd
    from isaacsim.core.utils.stage import is_stage_loading

    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(usd_path)):
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    for _ in range(2):
        simulation_app.update()
    while is_stage_loading():
        simulation_app.update()
    for _ in range(60):
        simulation_app.update()
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError(f"USD stage is not available after loading: {usd_path}")
    return stage


def find_articulation_root_path(stage) -> str:
    """Return the first articulation root path, falling back to the default prim."""

    default_prim = stage.GetDefaultPrim()
    if default_prim and "PhysicsArticulationRootAPI" in set(default_prim.GetAppliedSchemas()):
        return default_prim.GetPath().pathString
    for prim in stage.Traverse():
        if "PhysicsArticulationRootAPI" in set(prim.GetAppliedSchemas()):
            return prim.GetPath().pathString
    if default_prim:
        return default_prim.GetPath().pathString
    raise RuntimeError("Cannot find articulation root or default prim in USD stage.")


def find_named_prim_path(stage, name: str) -> str:
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim.GetPath().pathString
    raise RuntimeError(f"Cannot find prim named {name!r}.")


def make_world_and_robot(articulation_root_path: str):
    """Create Isaac Sim World and wrap the loaded articulation."""

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation

    World.clear_instance()
    world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
    robot = world.scene.add(SingleArticulation(prim_path=articulation_root_path, name="wheel_leg"))
    world.reset()
    return world, robot


def set_robot_joint_state(robot, joint_pos: dict[str, float], joint_vel: dict[str, float] | None = None) -> None:
    """Write selected joint positions/velocities by name."""

    q = robot.get_joint_positions()
    dq = robot.get_joint_velocities()
    for name, value in joint_pos.items():
        if name in robot.dof_names:
            safe_value = clamp_revolute_target_rad(value)
            if safe_value is not None:
                q[robot.get_dof_index(name)] = safe_value
    if joint_vel:
        for name, value in joint_vel.items():
            if name in robot.dof_names:
                value = float(value)
                if math.isfinite(value):
                    dq[robot.get_dof_index(name)] = value
    robot.set_joint_positions(q)
    robot.set_joint_velocities(dq)


def set_prim_local_quat_wxyz(stage, prim_path: str, quat_wxyz: Iterable[float]) -> None:
    """Set local orientation op on a prim. The quaternion is [w, x, y, z]."""

    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Prim not found: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    orient_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op
            break
    if orient_op is None:
        orient_op = xform.AddOrientOp()
    w, x, y, z = [float(v) for v in quat_wxyz]
    orient_op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
