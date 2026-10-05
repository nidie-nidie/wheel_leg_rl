#!/usr/bin/env python3
"""Bidirectional motor command UI for sim2real alignment.

Run with:
  ./IsaacLab-2.3.2/_isaac_sim/python.sh 测试对齐/02_motor_control_ui.py --show --transport udp --udp-bind-port 15002 --udp-remote-host 127.0.0.1 --udp-remote-port 15001
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

from align_common import (
    CMD_FRAME_SIZE,
    DEFAULT_USD_PATH,
    FrameReader,
    LEG_JOINTS,
    MSG_COMMAND,
    MSG_STATE,
    WHEEL_JOINTS,
    CmdPayload,
    add_transport_args,
    find_articulation_root_path,
    make_endpoint,
    make_world_and_robot,
    open_stage_blocking,
    pack_command,
    pack_frame,
    set_robot_joint_state,
    unpack_state,
    wrap_angle_rad,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motor PD/target command UI with lower-computer feedback.")
    parser.add_argument("--usd-path", default=str(DEFAULT_USD_PATH))
    parser.add_argument("--show", action="store_true", help="Show Isaac Sim window. UI requires this.")
    parser.add_argument("--send-hz", type=float, default=50.0)
    parser.add_argument("--ttl-ms", type=int, default=50)
    add_transport_args(parser, send=True)
    return parser.parse_args()


class MotorPanel:
    """Lightweight Omni UI panel backed by CmdPayload."""

    def __init__(self, command: CmdPayload):
        import omni.ui as ui

        self.ui = ui
        self.command = command
        self.mode = ui.SimpleIntModel(command.mode)
        self.ttl_ms = ui.SimpleIntModel(command.ttl_ms)
        self.feedback_label = ui.SimpleStringModel("state: no packet")
        self.seq_label = ui.SimpleStringModel("tx seq: 0")
        self.models: dict[str, list] = {
            "leg_q": [ui.SimpleFloatModel(v) for v in command.leg_q_des_rad],
            "leg_dq": [ui.SimpleFloatModel(v) for v in command.leg_dq_des_rad_s],
            "wheel_dq": [ui.SimpleFloatModel(v) for v in command.wheel_dq_des_rad_s],
            "leg_kp": [ui.SimpleFloatModel(v) for v in command.leg_kp],
            "leg_kd": [ui.SimpleFloatModel(v) for v in command.leg_kd],
            "wheel_kp": [ui.SimpleFloatModel(v) for v in command.wheel_kp],
            "wheel_kd": [ui.SimpleFloatModel(v) for v in command.wheel_kd],
        }
        self._build()

    def _drag(self, label: str, model, step: float, width: int = 95) -> None:
        ui = self.ui
        with ui.HStack(height=24):
            ui.Label(label, width=105)
            ui.FloatDrag(model=model, step=step, width=width)

    def _joint_rows(self, title: str, names: tuple[str, ...], model_key: str, unit: str, step: float) -> None:
        ui = self.ui
        with ui.CollapsableFrame(title, collapsed=False):
            with ui.VStack(spacing=3):
                for name, model in zip(names, self.models[model_key]):
                    self._drag(f"{name} {unit}", model, step)

    def _set_mode(self, value: int) -> None:
        self.mode.set_value(value)

    def _build(self) -> None:
        ui = self.ui
        self.window = ui.Window("Wheel-Leg Motor Alignment", width=430, height=820)
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6):
                    ui.Label("Command mode")
                    with ui.HStack(height=28):
                        ui.Button("IDLE", clicked_fn=lambda: self._set_mode(0))
                        ui.Button("READY", clicked_fn=lambda: self._set_mode(1))
                        ui.Button("RUN", clicked_fn=lambda: self._set_mode(2))
                        ui.Button("DAMP", clicked_fn=lambda: self._set_mode(3))
                        ui.Button("ESTOP", clicked_fn=lambda: self._set_mode(4))
                    with ui.HStack(height=24):
                        ui.Label("ttl ms", width=105)
                        ui.IntDrag(model=self.ttl_ms, min=1, max=1000, width=95)
                    self._joint_rows("Leg q_des rad [jIO, jAG, jIJ, jAB]", LEG_JOINTS, "leg_q", "rad", 0.01)
                    self._joint_rows("Leg dq_des rad/s", LEG_JOINTS, "leg_dq", "rad/s", 0.05)
                    self._joint_rows("Wheel dq_des rad/s [left, right]", WHEEL_JOINTS, "wheel_dq", "rad/s", 0.1)
                    self._joint_rows("Leg Kp N*m/rad", LEG_JOINTS, "leg_kp", "Kp", 0.5)
                    self._joint_rows("Leg Kd N*m*s/rad", LEG_JOINTS, "leg_kd", "Kd", 0.05)
                    self._joint_rows("Wheel speed Kp", WHEEL_JOINTS, "wheel_kp", "Kp", 0.05)
                    self._joint_rows("Wheel damping Kd", WHEEL_JOINTS, "wheel_kd", "Kd", 0.05)
                    ui.Line()
                    ui.Label(self.feedback_label)
                    ui.Label(self.seq_label)

    def snapshot(self) -> CmdPayload:
        """Read UI values into a fresh command payload."""

        def vals(key: str) -> tuple[float, ...]:
            return tuple(float(m.get_value_as_float()) for m in self.models[key])

        return dataclasses.replace(
            self.command,
            mode=int(self.mode.get_value_as_int()),
            ttl_ms=max(1, int(self.ttl_ms.get_value_as_int())),
            leg_q_des_rad=vals("leg_q"),
            leg_dq_des_rad_s=vals("leg_dq"),
            wheel_dq_des_rad_s=vals("wheel_dq"),
            leg_kp=vals("leg_kp"),
            leg_kd=vals("leg_kd"),
            wheel_kp=vals("wheel_kp"),
            wheel_kd=vals("wheel_kd"),
        )

    def set_feedback_text(self, text: str) -> None:
        self.feedback_label.set_value(text)

    def set_seq(self, seq: int) -> None:
        self.seq_label.set_value(f"tx seq: {seq}")


def main() -> None:
    args = parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not args.show,
            "hide_ui": not args.show,
            "renderer": "RaytracedLighting",
            "width": 1400,
            "height": 900,
            "multi_gpu": False,
        }
    )

    stage = open_stage_blocking(simulation_app, Path(args.usd_path).resolve())
    root_path = find_articulation_root_path(stage)
    world, robot = make_world_and_robot(root_path)
    endpoint = make_endpoint(args)
    reader = FrameReader()

    command = CmdPayload(ttl_ms=args.ttl_ms)
    panel = MotorPanel(command)
    tx_seq = 0
    next_send_t = 0.0
    wheel_q = {name: 0.0 for name in WHEEL_JOINTS}
    last_t = time.time()

    print(f"USD: {args.usd_path}")
    print(f"Articulation root: {root_path}")
    print(f"Command frame size: {CMD_FRAME_SIZE} bytes, endian: big")

    try:
        while simulation_app.is_running():
            now = time.time()
            dt = max(0.0, now - last_t)
            last_t = now

            for header, payload in reader.feed(endpoint.read()):
                if header.msg_type != MSG_STATE:
                    continue
                state = unpack_state(payload)
                for name, dq in zip(WHEEL_JOINTS, state.wheel_dq_rad_s):
                    wheel_q[name] = wrap_angle_rad(wheel_q[name] + float(dq) * dt)
                joint_pos = {name: value for name, value in zip(LEG_JOINTS, state.active_leg_q_rad)}
                joint_pos.update(wheel_q)
                joint_vel = {name: value for name, value in zip(LEG_JOINTS, state.active_leg_dq_rad_s)}
                joint_vel.update({name: value for name, value in zip(WHEEL_JOINTS, state.wheel_dq_rad_s)})
                set_robot_joint_state(robot, joint_pos, joint_vel)
                panel.set_feedback_text(
                    "state: "
                    f"status={state.status} fault={state.fault_code} "
                    f"q=[{', '.join(f'{v:+.2f}' for v in state.active_leg_q_rad)}] "
                    f"wheel=[{', '.join(f'{v:+.2f}' for v in state.wheel_dq_rad_s)}]"
                )

            if now >= next_send_t:
                payload = pack_command(panel.snapshot())
                endpoint.write(pack_frame(MSG_COMMAND, payload, tx_seq))
                panel.set_seq(tx_seq)
                tx_seq = (tx_seq + 1) & 0xFFFFFFFF
                next_send_t = now + (1.0 / max(args.send_hz, 1e-6))

            world.step(render=True)
    finally:
        endpoint.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
