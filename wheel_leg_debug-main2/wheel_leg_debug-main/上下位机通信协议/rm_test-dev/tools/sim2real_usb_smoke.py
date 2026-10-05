import argparse
import binascii
import struct
import time


SOF = 0xAA55
VER = 1
MSG_COMMAND = 1
MSG_STATE = 2
HEADER_LEN = 18
CRC_LEN = 4
CMD_PAYLOAD_LEN = 140
STATE_PAYLOAD_LEN = 152
CMD_FRAME_LEN = HEADER_LEN + CMD_PAYLOAD_LEN + CRC_LEN
STATE_FRAME_LEN = HEADER_LEN + STATE_PAYLOAD_LEN + CRC_LEN


def crc32(data: bytes) -> int:
    return binascii.crc32(data) & 0xFFFFFFFF


def build_command(seq: int, mode: int, ttl_ms: int) -> bytes:
    payload = bytearray()
    payload += struct.pack(">BBH", mode, 0, ttl_ms)
    payload += struct.pack(">4f", 0.0, 0.0, 0.0, 0.0)  # leg_q_des_rad
    payload += struct.pack(">4f", 0.0, 0.0, 0.0, 0.0)  # leg_dq_des_rad_s
    payload += struct.pack(">2f", 0.0, 0.0)  # wheel_dq_des_rad_s
    payload += struct.pack(">4f", 0.0, 0.0, 0.0, 0.0)  # leg_kp
    payload += struct.pack(">4f", 0.0, 0.0, 0.0, 0.0)  # leg_kd
    payload += struct.pack(">2f", 0.0, 0.0)  # wheel_kp
    payload += struct.pack(">2f", 0.0, 0.0)  # wheel_kd
    payload += struct.pack(">6f", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # tau_limit_nm
    payload += struct.pack(">6f", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # dq_limit_rad_s
    if len(payload) != CMD_PAYLOAD_LEN:
        raise RuntimeError(f"bad command payload length: {len(payload)}")

    t_us = time.monotonic_ns() // 1000
    frame = bytearray(struct.pack(">HBBHIQ", SOF, VER, MSG_COMMAND, CMD_PAYLOAD_LEN, seq, t_us))
    frame += payload
    frame += struct.pack(">I", crc32(frame))
    if len(frame) != CMD_FRAME_LEN:
        raise RuntimeError(f"bad command frame length: {len(frame)}")
    return bytes(frame)


def pop_state_frame(rx: bytearray):
    while len(rx) >= HEADER_LEN:
        if rx[0] != 0xAA or rx[1] != 0x55:
            del rx[0]
            continue

        sof, ver, msg_type, payload_len, seq, t_us = struct.unpack(">HBBHIQ", rx[:HEADER_LEN])
        frame_len = HEADER_LEN + payload_len + CRC_LEN
        if payload_len > STATE_PAYLOAD_LEN:
            del rx[0]
            continue
        if len(rx) < frame_len:
            return None

        frame = bytes(rx[:frame_len])
        del rx[:frame_len]
        rx_crc = struct.unpack(">I", frame[-CRC_LEN:])[0]
        calc_crc = crc32(frame[:-CRC_LEN])
        if sof != SOF or ver != VER or msg_type != MSG_STATE or payload_len != STATE_PAYLOAD_LEN:
            continue
        if rx_crc != calc_crc:
            print(f"bad state crc rx=0x{rx_crc:08X} calc=0x{calc_crc:08X}")
            continue
        return seq, t_us, frame[HEADER_LEN:-CRC_LEN]

    return None


def parse_state_payload(payload: bytes):
    status, reserved, fault_code, cmd_seq_echo = struct.unpack(">BBHI", payload[:8])
    off = 8
    gyro = struct.unpack(">3f", payload[off:off + 12])
    off += 12
    quat = struct.unpack(">4f", payload[off:off + 16])
    off += 16
    leg_q = struct.unpack(">4f", payload[off:off + 16])
    off += 16
    leg_dq = struct.unpack(">4f", payload[off:off + 16])
    off += 16
    wheel_dq = struct.unpack(">2f", payload[off:off + 8])
    off += 8
    off += 24  # joint_tau_nm
    off += 24  # motor_current_a
    bus_voltage = struct.unpack(">f", payload[off:off + 4])[0]
    return {
        "status": status,
        "fault_code": fault_code,
        "cmd_seq_echo": cmd_seq_echo,
        "gyro": gyro,
        "quat": quat,
        "leg_q": leg_q,
        "leg_dq": leg_dq,
        "wheel_dq": wheel_dq,
        "bus_voltage": bus_voltage,
    }


def main():
    parser = argparse.ArgumentParser(description="Sim2Real USB CDC smoke test")
    parser.add_argument("--port", required=True, help="Windows COM port, for example COM7")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--mode", type=int, default=1, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--ttl-ms", type=int, default=50)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required: python -m pip install pyserial") from exc

    period = 1.0 / args.rate_hz
    rx = bytearray()
    seq = 0
    next_tx = time.monotonic()
    stop_time = next_tx + args.duration

    with serial.Serial(args.port, args.baud, timeout=0.02) as ser:
        while time.monotonic() < stop_time:
            now = time.monotonic()
            if now >= next_tx:
                ser.write(build_command(seq, args.mode, args.ttl_ms))
                seq += 1
                next_tx += period

            data = ser.read(512)
            if data:
                rx.extend(data)

            state = pop_state_frame(rx)
            if state is not None:
                state_seq, state_t_us, payload = state
                parsed = parse_state_payload(payload)
                print(
                    f"state_seq={state_seq} echo={parsed['cmd_seq_echo']} "
                    f"status={parsed['status']} fault=0x{parsed['fault_code']:04X} "
                    f"gyro=({parsed['gyro'][0]:.3f},{parsed['gyro'][1]:.3f},{parsed['gyro'][2]:.3f}) "
                    f"quat=({parsed['quat'][0]:.4f},{parsed['quat'][1]:.4f},{parsed['quat'][2]:.4f},{parsed['quat'][3]:.4f}) "
                    f"vbus={parsed['bus_voltage']:.2f}"
                )


if __name__ == "__main__":
    main()
