#!/usr/bin/env python3
"""Receive IMU state packets and drive base_link pose for attitude alignment."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from align_common import (
    DEFAULT_USD_PATH,
    FrameReader,
    MSG_STATE,
    STATE_FRAME_SIZE,
    add_transport_args,
    find_named_prim_path,
    make_endpoint,
    open_stage_blocking,
    quat_wxyz_to_rpy_rad,
    set_prim_local_quat_wxyz,
    unpack_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMU quaternion -> base_link visual attitude alignment.")
    parser.add_argument("--usd-path", default=str(DEFAULT_USD_PATH))
    parser.add_argument("--show", action="store_true", help="Show Isaac Sim window. UI requires this.")
    parser.add_argument("--base-prim-name", default="base_link")
    parser.add_argument("--rate-hz", type=float, default=60.0)
    add_transport_args(parser, send=False)
    return parser.parse_args()


class ImuPanel:
    """Small UI panel displaying quaternion, Euler angles and gyro."""

    def __init__(self):
        import omni.ui as ui

        self.ui = ui
        self.labels = {
            "state": ui.SimpleStringModel("state: no packet"),
            "quat": ui.SimpleStringModel("quat wxyz: -"),
            "rpy": ui.SimpleStringModel("rpy deg: -"),
            "gyro": ui.SimpleStringModel("gyro rad/s: -"),
        }
        self.window = ui.Window("Wheel-Leg IMU Alignment", width=420, height=190)
        with self.window.frame:
            with ui.VStack(spacing=6):
                for model in self.labels.values():
                    ui.Label(model)

    def update(self, status: int, fault: int, quat, rpy_rad, gyro) -> None:
        rpy_deg = [math.degrees(v) for v in rpy_rad]
        self.labels["state"].set_value(f"state: status={status} fault={fault}")
        self.labels["quat"].set_value("quat wxyz: " + ", ".join(f"{v:+.4f}" for v in quat))
        self.labels["rpy"].set_value("rpy deg: " + ", ".join(f"{v:+.2f}" for v in rpy_deg))
        self.labels["gyro"].set_value("gyro rad/s: " + ", ".join(f"{v:+.3f}" for v in gyro))


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
    base_path = find_named_prim_path(stage, args.base_prim_name)
    endpoint = make_endpoint(args)
    reader = FrameReader()
    panel = ImuPanel()

    print(f"USD: {args.usd_path}")
    print(f"Driving prim: {base_path}")
    print(f"State frame size: {STATE_FRAME_SIZE} bytes, IMU quaternion order: [w, x, y, z]")

    try:
        while simulation_app.is_running():
            loop_t = time.time()
            for header, payload in reader.feed(endpoint.read()):
                if header.msg_type != MSG_STATE:
                    continue
                state = unpack_state(payload)
                rpy = quat_wxyz_to_rpy_rad(state.imu_quat_wxyz)
                set_prim_local_quat_wxyz(stage, base_path, state.imu_quat_wxyz)
                panel.update(state.status, state.fault_code, state.imu_quat_wxyz, rpy, state.imu_gyro_rad_s)

            simulation_app.update()
            if args.rate_hz > 0:
                sleep_t = (1.0 / args.rate_hz) - (time.time() - loop_t)
                if sleep_t > 0:
                    time.sleep(sleep_t)
    finally:
        endpoint.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

