#!/usr/bin/env python3
"""Standalone Qt serial UI for sim2real command/state smoke testing.

Run with:
  python3 测试对齐/04_qt_serial_comm_ui.py --serial-port /dev/ttyUSB0 --baudrate 921600
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import time
from pathlib import Path
from typing import Iterable


def _configure_qt_plugin_path() -> None:
    """Help PyQt5 find qwindows.dll when this folder lives in a non-ASCII path."""
    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        return

    platforms_dir = (
        Path(__file__).resolve().parent
        / ".venv"
        / "Lib"
        / "site-packages"
        / "PyQt5"
        / "Qt5"
        / "plugins"
        / "platforms"
    )
    if (platforms_dir / "qwindows.dll").exists():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)


_configure_qt_plugin_path()

from PyQt5 import QtCore, QtWidgets

from align_common import (
    CMD_FRAME_SIZE,
    LEG_JOINTS,
    MOTOR_JOINTS,
    MSG_COMMAND,
    MSG_STATE,
    STATE_FRAME_SIZE,
    WHEEL_JOINTS,
    CmdPayload,
    FrameReader,
    SerialEndpoint,
    pack_command,
    pack_frame,
    quat_wxyz_to_rpy_rad,
    unpack_state,
)


STATUS_NAMES = {
    0: "IDLE",
    1: "READY",
    2: "RUN",
    3: "DAMPING",
    4: "ESTOP",
    5: "FAULT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Qt serial command/state debugger.")
    parser.add_argument("--serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=921600)
    return parser.parse_args()


def available_ports() -> list[str]:
    patterns = ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyTHS*", "/dev/ttyS*")
    ports: list[str] = []
    try:
        from serial.tools import list_ports

        ports.extend(port.device for port in list_ports.comports())
    except Exception:
        pass
    for pattern in patterns:
        ports.extend(glob.glob(pattern))
    return sorted(set(ports))


def csv_floats(text: str, count: int, name: str) -> tuple[float, ...]:
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != count:
        raise ValueError(f"{name} needs {count} values, got {len(parts)}")
    return tuple(float(p) for p in parts)


def fmt_values(values: Iterable[float], precision: int = 3) -> str:
    return ", ".join(f"{float(v):+.{precision}f}" for v in values)


class FloatListEdit(QtWidgets.QLineEdit):
    def __init__(self, values: Iterable[float], count: int, parent=None):
        super().__init__(parent)
        self.count = count
        self.setText(", ".join(f"{float(v):g}" for v in values))

    def values(self, name: str) -> tuple[float, ...]:
        return csv_floats(self.text(), self.count, name)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.setWindowTitle("Sim2Real Serial Debugger")
        self.resize(1360, 760)

        self.endpoint: SerialEndpoint | None = None
        self.reader = FrameReader()
        self.tx_seq = 0
        self.rx_bytes = 0
        self.rx_frames = 0
        self.last_state_t = 0.0

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.setInterval(10)
        self.poll_timer.timeout.connect(self.poll_serial)

        self.send_timer = QtCore.QTimer(self)
        self.send_timer.timeout.connect(self.send_command)

        self.default_command = CmdPayload()
        self._build_ui(args)
        self.refresh_ports()
        self.port_combo.setEditText(args.serial_port)
        self.baud_spin.setValue(args.baudrate)
        self._set_connected(False)

    def _build_ui(self, args: argparse.Namespace) -> None:
        del args
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)

        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setEditable(True)
        self.baud_spin = QtWidgets.QSpinBox()
        self.baud_spin.setRange(9600, 4000000)
        self.baud_spin.setSingleStep(115200)
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.connect_button = QtWidgets.QPushButton("Open")
        self.status_label = QtWidgets.QLabel("closed")
        top.addWidget(QtWidgets.QLabel("Port"))
        top.addWidget(self.port_combo, 2)
        top.addWidget(QtWidgets.QLabel("Baud"))
        top.addWidget(self.baud_spin)
        top.addWidget(self.refresh_button)
        top.addWidget(self.connect_button)
        top.addWidget(self.status_label, 2)

        self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button.clicked.connect(self.toggle_connection)

        body = QtWidgets.QHBoxLayout()
        root.addLayout(body, 1)

        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()
        body.addLayout(left, 1)
        body.addLayout(right, 1)

        left.addWidget(self._make_receive_group())
        left.addWidget(self._make_raw_group())
        right.addWidget(self._make_command_group())

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        root.addWidget(self.log, 1)
        self.log_line("Ready. Frame sizes: command=%d B, state=%d B" % (CMD_FRAME_SIZE, STATE_FRAME_SIZE))

    def _make_receive_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Receive state")
        layout = QtWidgets.QVBoxLayout(box)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)
        self.rx_stat = QtWidgets.QLabel("rx bytes=0 frames=0")
        self.rx_state = QtWidgets.QLabel("state: no packet")
        self.rx_imu = QtWidgets.QLabel("imu: -")
        self.rx_quat = QtWidgets.QLabel("quat: -")
        self.rx_leg_q = QtWidgets.QLabel("leg q: -")
        self.rx_leg_dq = QtWidgets.QLabel("leg dq: -")
        self.rx_wheel = QtWidgets.QLabel("wheel dq: -")
        self.rx_tau = QtWidgets.QLabel("tau: -")
        self.rx_current = QtWidgets.QLabel("current: -")
        self.rx_temp = QtWidgets.QLabel("temp: -")
        labels = [
            self.rx_stat,
            self.rx_state,
            self.rx_imu,
            self.rx_quat,
            self.rx_leg_q,
            self.rx_leg_dq,
            self.rx_wheel,
            self.rx_tau,
            self.rx_current,
            self.rx_temp,
        ]
        for row, label in enumerate(labels):
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            grid.addWidget(label, row, 0)
        return box

    def _make_command_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Send command frame")
        layout = QtWidgets.QVBoxLayout(box)

        row = QtWidgets.QHBoxLayout()
        layout.addLayout(row)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["IDLE", "READY", "RUN", "DAMPING", "ESTOP"])
        self.ttl_spin = QtWidgets.QSpinBox()
        self.ttl_spin.setRange(1, 5000)
        self.ttl_spin.setValue(self.default_command.ttl_ms)
        self.send_hz = QtWidgets.QDoubleSpinBox()
        self.send_hz.setRange(0.1, 500.0)
        self.send_hz.setValue(50.0)
        self.send_hz.setDecimals(1)
        self.periodic_check = QtWidgets.QCheckBox("Periodic")
        row.addWidget(QtWidgets.QLabel("Mode"))
        row.addWidget(self.mode_combo)
        row.addWidget(QtWidgets.QLabel("TTL ms"))
        row.addWidget(self.ttl_spin)
        row.addWidget(QtWidgets.QLabel("Hz"))
        row.addWidget(self.send_hz)
        row.addWidget(self.periodic_check)

        self.cmd_edits: dict[str, FloatListEdit] = {}
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        def add_edit(key: str, title: str, unit: str, values: Iterable[float], count: int) -> None:
            edit = FloatListEdit(values, count)
            edit.setMinimumWidth(430)
            self.cmd_edits[key] = edit
            value_row = QtWidgets.QHBoxLayout()
            value_row.addWidget(edit, 1)
            unit_label = QtWidgets.QLabel(unit)
            unit_label.setMinimumWidth(95)
            value_row.addWidget(unit_label)
            form.addRow(title, value_row)

        add_edit("leg_q", f"leg q_des {LEG_JOINTS}", "rad", self.default_command.leg_q_des_rad, 4)
        add_edit("leg_dq", f"leg dq_des {LEG_JOINTS}", "rad/s", self.default_command.leg_dq_des_rad_s, 4)
        add_edit("wheel_dq", f"wheel dq_des {WHEEL_JOINTS}", "rad/s", self.default_command.wheel_dq_des_rad_s, 2)
        add_edit("leg_kp", f"leg kp {LEG_JOINTS}", "N*m/rad", self.default_command.leg_kp, 4)
        add_edit("leg_kd", f"leg kd {LEG_JOINTS}", "N*m*s/rad", self.default_command.leg_kd, 4)
        add_edit("wheel_kp", f"wheel kp {WHEEL_JOINTS}", "N*m/(rad/s)", self.default_command.wheel_kp, 2)
        add_edit("wheel_kd", f"wheel kd {WHEEL_JOINTS}", "N*m*s/rad", self.default_command.wheel_kd, 2)
        add_edit("tau_limit", f"tau limit {MOTOR_JOINTS}", "N*m", self.default_command.tau_limit_nm, 6)
        add_edit("dq_limit", f"dq limit {MOTOR_JOINTS}", "rad/s", self.default_command.dq_limit_rad_s, 6)

        buttons = QtWidgets.QHBoxLayout()
        layout.addLayout(buttons)
        self.send_button = QtWidgets.QPushButton("Send once")
        self.idle_button = QtWidgets.QPushButton("Send IDLE")
        self.estop_button = QtWidgets.QPushButton("Send ESTOP")
        self.reset_button = QtWidgets.QPushButton("Reset defaults")
        buttons.addWidget(self.send_button)
        buttons.addWidget(self.idle_button)
        buttons.addWidget(self.estop_button)
        buttons.addWidget(self.reset_button)

        self.send_button.clicked.connect(self.send_command)
        self.idle_button.clicked.connect(lambda: self.send_mode_once(0))
        self.estop_button.clicked.connect(lambda: self.send_mode_once(4))
        self.reset_button.clicked.connect(self.reset_defaults)
        self.periodic_check.toggled.connect(self.set_periodic_send)
        self.send_hz.valueChanged.connect(self.update_send_timer_interval)
        return box

    def _make_raw_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Raw USB write")
        layout = QtWidgets.QVBoxLayout(box)
        self.raw_hex = QtWidgets.QPlainTextEdit()
        self.raw_hex.setPlaceholderText("Hex bytes, for example: AA 55 01 01 ...")
        self.raw_hex.setFixedHeight(90)
        self.send_raw_button = QtWidgets.QPushButton("Send raw hex")
        layout.addWidget(self.raw_hex)
        layout.addWidget(self.send_raw_button)
        self.send_raw_button.clicked.connect(self.send_raw_hex)
        return box

    def refresh_ports(self) -> None:
        current = self.port_combo.currentText() if hasattr(self, "port_combo") else ""
        self.port_combo.clear()
        ports = available_ports()
        self.port_combo.addItems(ports)
        if current:
            self.port_combo.setEditText(current)

    def toggle_connection(self) -> None:
        if self.endpoint is None:
            self.open_serial()
        else:
            self.close_serial()

    def open_serial(self) -> None:
        port = self.port_combo.currentText().strip()
        baud = int(self.baud_spin.value())
        if not port:
            self.log_line("No serial port selected")
            return
        try:
            self.endpoint = SerialEndpoint(port, baud)
        except Exception as exc:
            self.endpoint = None
            self.log_line(f"Open failed: {exc}")
            self.status_label.setText(f"open failed: {exc}")
            return
        self.reader = FrameReader()
        self.rx_bytes = 0
        self.rx_frames = 0
        self.poll_timer.start()
        self._set_connected(True)
        self.log_line(f"Opened {port} @ {baud}")

    def close_serial(self) -> None:
        self.poll_timer.stop()
        self.send_timer.stop()
        self.periodic_check.setChecked(False)
        if self.endpoint is not None:
            self.endpoint.close()
            self.endpoint = None
        self._set_connected(False)
        self.log_line("Closed")

    def _set_connected(self, connected: bool) -> None:
        self.connect_button.setText("Close" if connected else "Open")
        self.status_label.setText("open" if connected else "closed")
        self.port_combo.setEnabled(not connected)
        self.baud_spin.setEnabled(not connected)

    def poll_serial(self) -> None:
        if self.endpoint is None:
            return
        try:
            data = self.endpoint.read()
        except Exception as exc:
            self.log_line(f"Read failed: {exc}")
            self.close_serial()
            return
        if not data:
            self.update_rx_stat()
            return
        self.rx_bytes += len(data)
        try:
            frames = self.reader.feed(data)
        except Exception as exc:
            self.log_line(f"Parse failed: {exc}")
            return
        for header, payload in frames:
            if header.msg_type != MSG_STATE:
                self.log_line(f"RX frame type={header.msg_type} len={header.payload_len} seq={header.seq}")
                continue
            try:
                state = unpack_state(payload)
            except Exception as exc:
                self.log_line(f"Bad state payload: {exc}")
                continue
            self.rx_frames += 1
            self.last_state_t = time.time()
            self.show_state(header.seq, state)
        self.update_rx_stat()

    def update_rx_stat(self) -> None:
        age = "-"
        if self.last_state_t > 0.0:
            age = f"{(time.time() - self.last_state_t) * 1000.0:.0f} ms"
        self.rx_stat.setText(f"rx bytes={self.rx_bytes} state frames={self.rx_frames} last age={age}")

    def show_state(self, seq: int, state) -> None:
        status = STATUS_NAMES.get(state.status, str(state.status))
        rpy = quat_wxyz_to_rpy_rad(state.imu_quat_wxyz)
        self.rx_state.setText(
            f"state seq={seq} status={status} fault={state.fault_code} cmd_echo={state.cmd_seq_echo} "
            f"bus={state.bus_voltage_v:.2f} V"
        )
        self.rx_imu.setText(f"gyro rad/s: {fmt_values(state.imu_gyro_rad_s)}")
        self.rx_quat.setText(
            f"quat wxyz: {fmt_values(state.imu_quat_wxyz)}  rpy rad: {fmt_values(rpy)}"
        )
        self.rx_leg_q.setText(f"leg q rad: {fmt_values(state.active_leg_q_rad)}")
        self.rx_leg_dq.setText(f"leg dq rad/s: {fmt_values(state.active_leg_dq_rad_s)}")
        self.rx_wheel.setText(f"wheel dq rad/s: {fmt_values(state.wheel_dq_rad_s)}")
        self.rx_tau.setText(f"tau Nm: {fmt_values(state.joint_tau_nm)}")
        self.rx_current.setText(f"current A: {fmt_values(state.motor_current_a)}")
        self.rx_temp.setText(f"temp C: {fmt_values(state.temperature_c, 1)}")

    def current_command(self) -> CmdPayload:
        return dataclasses.replace(
            self.default_command,
            mode=int(self.mode_combo.currentIndex()),
            ttl_ms=int(self.ttl_spin.value()),
            leg_q_des_rad=self.cmd_edits["leg_q"].values("leg q_des"),
            leg_dq_des_rad_s=self.cmd_edits["leg_dq"].values("leg dq_des"),
            wheel_dq_des_rad_s=self.cmd_edits["wheel_dq"].values("wheel dq_des"),
            leg_kp=self.cmd_edits["leg_kp"].values("leg kp"),
            leg_kd=self.cmd_edits["leg_kd"].values("leg kd"),
            wheel_kp=self.cmd_edits["wheel_kp"].values("wheel kp"),
            wheel_kd=self.cmd_edits["wheel_kd"].values("wheel kd"),
            tau_limit_nm=self.cmd_edits["tau_limit"].values("tau limit"),
            dq_limit_rad_s=self.cmd_edits["dq_limit"].values("dq limit"),
        )

    def send_command(self) -> None:
        if self.endpoint is None:
            self.log_line("Not open")
            return
        try:
            payload = pack_command(self.current_command())
            frame = pack_frame(MSG_COMMAND, payload, self.tx_seq)
            self.endpoint.write(frame)
        except Exception as exc:
            self.log_line(f"Send command failed: {exc}")
            return
        self.log_line(f"TX command seq={self.tx_seq} mode={self.mode_combo.currentText()} bytes={len(frame)}")
        self.tx_seq = (self.tx_seq + 1) & 0xFFFFFFFF

    def send_mode_once(self, mode: int) -> None:
        old_mode = self.mode_combo.currentIndex()
        self.mode_combo.setCurrentIndex(mode)
        self.send_command()
        self.mode_combo.setCurrentIndex(old_mode)

    def send_raw_hex(self) -> None:
        if self.endpoint is None:
            self.log_line("Not open")
            return
        text = self.raw_hex.toPlainText()
        cleaned = text.replace(",", " ").replace("0x", " ").replace("\\x", " ")
        try:
            data = bytes(int(part, 16) for part in cleaned.split())
        except ValueError as exc:
            self.log_line(f"Bad hex: {exc}")
            return
        if not data:
            self.log_line("No raw bytes to send")
            return
        try:
            self.endpoint.write(data)
        except Exception as exc:
            self.log_line(f"Raw send failed: {exc}")
            return
        self.log_line(f"TX raw bytes={len(data)}")

    def reset_defaults(self) -> None:
        cmd = self.default_command
        self.mode_combo.setCurrentIndex(cmd.mode)
        self.ttl_spin.setValue(cmd.ttl_ms)
        values = {
            "leg_q": cmd.leg_q_des_rad,
            "leg_dq": cmd.leg_dq_des_rad_s,
            "wheel_dq": cmd.wheel_dq_des_rad_s,
            "leg_kp": cmd.leg_kp,
            "leg_kd": cmd.leg_kd,
            "wheel_kp": cmd.wheel_kp,
            "wheel_kd": cmd.wheel_kd,
            "tau_limit": cmd.tau_limit_nm,
            "dq_limit": cmd.dq_limit_rad_s,
        }
        for key, vals in values.items():
            self.cmd_edits[key].setText(", ".join(f"{float(v):g}" for v in vals))

    def set_periodic_send(self, enabled: bool) -> None:
        if enabled:
            if self.endpoint is None:
                self.periodic_check.blockSignals(True)
                self.periodic_check.setChecked(False)
                self.periodic_check.blockSignals(False)
                self.log_line("Open serial before enabling periodic send")
                return
            self.update_send_timer_interval()
            self.send_timer.start()
            self.log_line(f"Periodic send enabled at {self.send_hz.value():.1f} Hz")
        else:
            self.send_timer.stop()
            self.log_line("Periodic send disabled")

    def update_send_timer_interval(self) -> None:
        interval_ms = max(1, int(round(1000.0 / max(float(self.send_hz.value()), 1e-6))))
        self.send_timer.setInterval(interval_ms)

    def log_line(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {text}")

    def closeEvent(self, event) -> None:
        self.close_serial()
        event.accept()


def main() -> None:
    args = parse_args()
    app = QtWidgets.QApplication([])
    window = MainWindow(args)
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
