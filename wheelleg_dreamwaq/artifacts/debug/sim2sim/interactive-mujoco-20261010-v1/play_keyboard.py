"""Interactive command playback using the unchanged production MuJoCo runtime."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import queue
import time

from glfw import KEY_ESCAPE
import mujoco
import numpy as np
import torch

from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.observation import build_actor_observation, collect_kinematic_state
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from base_contact import base_contact_state, create_model
from keyboard_viewer import KeyboardViewer

PROJECT = Path(__file__).resolve().parents[4]
DEFAULT_EXPORT = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2/exports/run-04"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--duration", type=float, default=0, help="Optional wall-clock duration; 0 keeps the viewer open.")
    parser.add_argument("--smoke", action="store_true", help="Check keyboard commands and run 20 ticks without opening a window.")
    parser.add_argument("--base-contact", action="store_true", help="Viewer experiment: enable the existing base box against the floor.")
    parser.add_argument("--max-speed", type=float, default=10.0, help="Keyboard speed target limit in m/s; default 10.0 (training limit is 1.5).")
    parser.add_argument("--session", type=Path, default=None, help="Directory for live status and keyboard trajectory.")
    args = parser.parse_args()
    if not np.isfinite(args.max_speed) or args.max_speed <= 0:
        parser.error("--max-speed must be finite and positive")
    model_path = PROJECT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
    contract, manifest, model_manifest = load_policy_contract(
        args.export / "policy_manifest.json", args.export / "actor.ts", PROJECT / "sim2sim/mujoco/model_manifest.json")
    assert sha256_file(model_path) == model_manifest["model_xml"]["sha256"]
    policy = torch.jit.load(str(args.export / "actor.ts"), map_location="cpu").eval()
    torch.set_num_threads(1)
    if args.base_contact:
        model_path = create_model(model_path, Path(__file__).resolve().parent / "models/base_contact_debug.xml")
    runtime = WheelLegMujocoRuntime(model_path, contract, None if args.base_contact else model_manifest)
    command = np.asarray((0, 0, .20), dtype=np.float64)
    ranges = manifest["command_sampling"]
    lower = np.asarray([ranges[key][0] for key in ("vx", "yaw_rate", "base_height")])
    upper = np.asarray([ranges[key][1] for key in ("vx", "yaw_rate", "base_height")])
    lower[0], upper[0] = -args.max_speed, args.max_speed
    vx_scale = contract.normalization["vx_max_abs"]

    def reset_pose():
        runtime.reset(command)
        if runtime._history is not None:
            runtime._history[:, 6] = command[0] / vx_scale

    reset_pose()
    pending = queue.SimpleQueue()
    paused = False
    quit_requested = False
    fall_reason = ""

    def handle_keys():
        nonlocal paused, quit_requested, fall_reason
        while not pending.empty():
            key = pending.get()
            if key == KEY_ESCAPE:
                quit_requested = True
            elif key == ord("P"):
                paused = not paused
            elif key == ord("R"):
                reset_pose()
                paused = False
                fall_reason = ""
            elif key == ord("X"):
                command[:2] = 0
            elif key in (ord("W"), ord("S")):
                command[0] += .1 if key == ord("W") else -.1
            elif key in (ord("A"), ord("D")):
                command[1] += .1 if key == ord("A") else -.1
            elif key in (ord("Q"), ord("E")):
                command[2] += .01 if key == ord("Q") else -.01
            command[:] = np.clip(command, lower, upper)

    def tick():
        state = collect_kinematic_state(runtime.model, runtime.data, runtime.model_map, contract)
        # Change the command in the current frame at this action boundary. Older frames remain historical.
        current = build_actor_observation(state, command, runtime.previous_action, contract)
        # This viewer experiment permits speed targets outside the training range.
        # Keep the trained normalization scale, rather than silently clamping to +/-1.
        current[6] = command[0] / vx_scale
        if contract.policy_input_dimension == 125:
            assert runtime._history is not None
            runtime._history[-1] = current
            observation = runtime._history.reshape(-1).astype(np.float32, copy=True)
        else:
            observation = current
        with torch.inference_mode():
            tensor = torch.from_numpy(observation).unsqueeze(0)
            action = policy(tensor).numpy()[0]
            estimated_vx = (float(policy.encoder(tensor)[0, 0]) / contract.normalization["root_linear_velocity_scale"]
                            if contract.policy_input_dimension == 125 else float("nan"))
        result = runtime.step(action, command)
        if runtime._history is not None:
            runtime._history[-1, 6] = command[0] / vx_scale
        return result.metrics, estimated_vx

    if args.smoke:
        pending.put(ord("W"))
        pending.put(ord("A"))
        pending.put(ord("Q"))
        handle_keys()
        np.testing.assert_allclose(command, (.1, .1, .21))
        tick()
        np.testing.assert_allclose(runtime._history[-2, 6:9], (.1 / 1.5, .1, .25), atol=1e-6)
        for key in "SDE":
            pending.put(ord(key))
        handle_keys()
        np.testing.assert_allclose(command, (0, 0, .20), atol=1e-12)
        for _ in range(int(np.ceil(2 * args.max_speed / .1)) + 2):
            pending.put(ord("W"))
        handle_keys()
        assert command[0] == upper[0]
        reset_pose()
        tick()
        np.testing.assert_allclose(runtime._history[:, 6], args.max_speed / vx_scale)
        for _ in range(int(np.ceil(2 * args.max_speed / .1)) + 2):
            pending.put(ord("S"))
        handle_keys()
        assert command[0] == lower[0]
        reset_pose()
        tick()
        np.testing.assert_allclose(runtime._history[:, 6], -args.max_speed / vx_scale)
        pending.put(ord("X"))
        pending.put(ord("R"))
        handle_keys()
        assert runtime.data.time == 0 and np.all(runtime.previous_action == 0)
        np.testing.assert_array_equal(command[:2], 0)
        command[2] = .20
        reset_pose()
        for _ in range(20):
            metrics, _ = tick()
        pending.put(ord("P"))
        handle_keys()
        assert paused
        pending.put(ord("P"))
        handle_keys()
        assert not paused
        pending.put(KEY_ESCAPE)
        handle_keys()
        assert quit_requested
        print("KEYBOARD_SPEED_LIMIT_VERIFIED", json.dumps({
            "command_limit_mps": args.max_speed, "trained_vx_scale": vx_scale,
            "normalized_limit": args.max_speed / vx_scale,
            "positive_and_negative_reset_history_and_policy_tick": "passed",
        }), flush=True)
        print("KEYBOARD_SMOKE_OK", json.dumps(metrics), flush=True)
        return

    help_text = (f"Speed target range: {-args.max_speed:+.1f} to {args.max_speed:+.1f} m/s (trained: {ranges['vx'][0]:+.1f} to {ranges['vx'][1]:+.1f})\n"
                 "W/S: speed +/- 0.1 | A/D: yaw +/- 0.1 | Q/E: height +/- 0.01\nX: zero speed/yaw | R: reset pose | P: pause/resume | Esc: close")
    mode = "BASE CONTACT DEBUG" if args.base_contact else "FORMAL WHEEL CONTACT"
    print("RUN04_MUJOCO_KEYBOARD", mode, help_text, flush=True)
    started = time.monotonic()
    # Passive viewer sync rewrites xfrc_applied. Render a separate model/data pair so
    # graphics and mouse interaction cannot clear runtime angular-limit torques.
    display_model = mujoco.MjModel.from_xml_path(str(model_path))
    display_data = mujoco.MjData(display_model)
    base_id = mujoco.mj_name2id(display_model, mujoco.mjtObj.mjOBJ_GEOM, "base_proxy")
    display_model.geom_group[base_id] = 2
    mujoco.mj_copyData(display_data, display_model, runtime.data)
    session = args.session or Path(__file__).resolve().parent / ("session-" + time.strftime("%Y%m%d-%H%M%S"))
    session.mkdir(parents=True, exist_ok=True)
    log_stream = (session / "trajectory.csv").open("w", encoding="utf-8", newline="")
    csv_writer = None
    with log_stream, KeyboardViewer(display_model, display_data, key_callback=pending.put,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.distance = 1.1
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            viewer.cam.lookat[:] = display_data.xipos[runtime.model_map.base_body_id]
            viewer.opt.geomgroup[2] = 1
            viewer.opt.geomgroup[3] = 0
        print("MUJOCO_VIEWER_READY", mode, str(session), flush=True)
        metrics = runtime.last_reset_metrics
        estimated_vx = float("nan")
        while viewer.is_running() and not quit_requested:
            if args.duration > 0 and time.monotonic() - started >= args.duration:
                break
            tick_start = time.monotonic()
            handle_keys()
            if quit_requested:
                break
            if not paused:
                metrics, estimated_vx = tick()
                collision = base_contact_state(runtime.model, runtime.data)
                if not .10 <= metrics["base_height_m"] <= .40:
                    fall_reason = "base_height"
                elif metrics["tilt_rad"] > .80:
                    fall_reason = "tilt"
                elif metrics["max_loop_closure_error_m"] > .005:
                    fall_reason = "loop_closure"
                if fall_reason:
                    paused = True
                    print("VIEWER_FALL_PAUSED", runtime.data.time, fall_reason, json.dumps(metrics), flush=True)
                row = {"time_s": runtime.data.time, "command_vx_mps": float(command[0]),
                       "command_yaw_rad_s": float(command[1]), "command_height_m": float(command[2]),
                       "estimated_vx_before_action_mps": estimated_vx, **metrics, **collision,
                       "failure_reason": fall_reason}
                if csv_writer is None:
                    csv_writer = csv.DictWriter(log_stream, fieldnames=list(row))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                log_stream.flush()
            else:
                collision = base_contact_state(runtime.model, runtime.data)
            with viewer.lock():
                mujoco.mj_copyData(display_data, display_model, runtime.data)
                viewer.cam.lookat[:] = display_data.xipos[runtime.model_map.base_body_id]
            info = ("Run-04 | " + mode + " | " + ("PAUSED" if paused else "RUNNING") +
                    f" | sim t={runtime.data.time:.1f}s\n"
                    f"Command: vx={command[0]:+.2f} m/s  yaw={command[1]:+.2f} rad/s  height={command[2]:.2f} m\n"
                    f"Actual vx={metrics['vx_mps']:+.3f} m/s  CENet vx (before action)={estimated_vx:+.3f} m/s\n"
                    f"COM height={metrics['base_height_m']:.3f} m  tilt={np.rad2deg(metrics['tilt_rad']):.1f} deg\n"
                    f"Base box bottom={collision['base_proxy_min_z_m']:.3f} m  contacts={collision['base_floor_contact_count']}\n"
                    + (f"FALL: {fall_reason}. Press R to reset.\n" if fall_reason else "") + "\n" + help_text)
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, info, ""))
            viewer.sync(state_only=True)
            (session / "status.json").write_text(json.dumps({
                "mode": mode, "paused": paused, "time_s": runtime.data.time, "command": command.tolist(),
                "failure_reason": fall_reason, "metrics": metrics, "collision": collision,
                "display_uses_independent_state": True,
                "keyboard_renderer": "custom_glfw_no_native_shortcuts",
                "command_limits": {"lower": lower.tolist(), "upper": upper.tolist()},
                "trained_vx_normalization_scale": vx_scale,
            }, indent=2), encoding="utf-8")
            time.sleep(max(0, contract.control_dt_s - (time.monotonic() - tick_start)))
    print("MUJOCO_VIEWER_CLOSED", flush=True)


if __name__ == "__main__":
    main()
