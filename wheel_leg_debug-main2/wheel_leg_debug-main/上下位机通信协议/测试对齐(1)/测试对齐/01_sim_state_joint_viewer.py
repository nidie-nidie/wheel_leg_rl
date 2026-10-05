#!/usr/bin/env python3
"""Receive lower-computer state packets and visualize joint mapping in Isaac Sim.

Run with:
  ./IsaacLab-2.3.2/_isaac_sim/python.sh 测试对齐/01_sim_state_joint_viewer.py --show --transport serial --serial-port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from align_common import (
    DEFAULT_USD_PATH,
    FrameReader,
    LEG_JOINTS,
    MSG_STATE,
    STATE_FRAME_SIZE,
    WHEEL_JOINTS,
    add_transport_args,
    find_articulation_root_path,
    make_endpoint,
    make_world_and_robot,
    open_stage_blocking,
    set_robot_joint_state,
    unpack_state,
    wrap_angle_rad,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="State packet -> Isaac Sim joint visualization.")
    parser.add_argument("--usd-path", default=str(DEFAULT_USD_PATH))
    parser.add_argument("--show", action="store_true", help="Show Isaac Sim window.")
    parser.add_argument("--rate-hz", type=float, default=60.0, help="Render/update loop rate.")
    parser.add_argument("--print-every", type=int, default=50, help="Print one line every N received state frames.")
    parser.add_argument("--wheel-visual", choices=("integrate", "zero"), default="integrate")
    add_transport_args(parser, send=False)
    return parser.parse_args()


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
    wheel_q = {name: 0.0 for name in WHEEL_JOINTS}
    last_step_t = time.time()
    rx_count = 0

    print(f"USD: {args.usd_path}")
    print(f"Articulation root: {root_path}")
    print(f"State frame size: {STATE_FRAME_SIZE} bytes, endian: big")
    print(f"Leg order: {LEG_JOINTS}; wheel order: {WHEEL_JOINTS}")

    try:
        while simulation_app.is_running():
            now = time.time()
            dt = max(0.0, now - last_step_t)
            last_step_t = now

            for header, payload in reader.feed(endpoint.read()):
                if header.msg_type != MSG_STATE:
                    continue
                state = unpack_state(payload)
                rx_count += 1

                if args.wheel_visual == "integrate":
                    for name, dq in zip(WHEEL_JOINTS, state.wheel_dq_rad_s):
                        wheel_q[name] = wrap_angle_rad(wheel_q[name] + float(dq) * dt)
                else:
                    wheel_q = {name: 0.0 for name in WHEEL_JOINTS}

                joint_pos = {name: value for name, value in zip(LEG_JOINTS, state.active_leg_q_rad)}
                joint_pos.update(wheel_q)
                joint_vel = {name: value for name, value in zip(LEG_JOINTS, state.active_leg_dq_rad_s)}
                joint_vel.update({name: value for name, value in zip(WHEEL_JOINTS, state.wheel_dq_rad_s)})
                set_robot_joint_state(robot, joint_pos, joint_vel)

                if args.print_every > 0 and rx_count % args.print_every == 0:
                    leg_text = ", ".join(f"{n}={v:+.3f}rad" for n, v in zip(LEG_JOINTS, state.active_leg_q_rad))
                    wheel_text = ", ".join(f"{n}={v:+.3f}rad/s" for n, v in zip(WHEEL_JOINTS, state.wheel_dq_rad_s))
                    print(f"[state {rx_count}] status={state.status} fault={state.fault_code} {leg_text} | {wheel_text}", flush=True)

            world.step(render=True)
            if args.rate_hz > 0:
                sleep_t = (1.0 / args.rate_hz) - (time.time() - now)
                if sleep_t > 0:
                    time.sleep(sleep_t)
    finally:
        endpoint.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
