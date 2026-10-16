from __future__ import annotations

import argparse
import csv
import json
import queue
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np
import torch

from debug.no_action_clip.mujoco_adapter import RawActionRuntime, load_runtime


DEFAULT_EXPORT = Path(__file__).resolve().parent / "runs/fresh1000-fk-diagnostic-20261016/export"


class KeyboardViewer:
    def __init__(self, model, data, key_callback, experiment_id):
        if not glfw.init():
            raise RuntimeError("GLFW 初始化失败")
        self.window = glfw.create_window(1280, 800, f"WheelLeg MuJoCo | {experiment_id}", None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("MuJoCo 可视化窗口创建失败")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.model = model
        self.data = data
        self.key_callback = key_callback
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.last_cursor = glfw.get_cursor_pos(self.window)
        self.text = ""
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    def _on_key(self, window, key, scancode, action, mods):
        del window, scancode, mods
        if action == glfw.PRESS:
            self.key_callback(key)

    def _on_cursor(self, window, x, y):
        old_x, old_y = self.last_cursor
        self.last_cursor = (x, y)
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not (left or right):
            return
        height = max(1, glfw.get_window_size(window)[1])
        mouse_action = mujoco.mjtMouse.mjMOUSE_ROTATE_V if left else mujoco.mjtMouse.mjMOUSE_MOVE_V
        mujoco.mjv_moveCamera(self.model, mouse_action, (x - old_x) / height, (y - old_y) / height, self.camera)

    def _on_scroll(self, window, dx, dy):
        del window, dx
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -.05 * dy, self.camera)

    def is_running(self):
        return not glfw.window_should_close(self.window)

    def render(self, text):
        glfw.poll_events()
        self.text = text
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            self.text,
            "",
            self.context,
        )
        glfw.swap_buffers(self.window)

    def close(self):
        if self.window is not None:
            self.context.free()
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _command_limits(max_speed: float) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray((-max_speed, -1.0, 0.16)), np.asarray((max_speed, 1.0, 0.24))


def main() -> None:
    parser = argparse.ArgumentParser(description="DebugNoActionClipV2 MuJoCo 可视化与键盘控制")
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--max-speed", type=float, default=1.5)
    parser.add_argument("--session", type=Path, default=None)
    args = parser.parse_args()
    if not np.isfinite(args.max_speed) or args.max_speed <= 0:
        parser.error("--max-speed 必须为正数")

    export_dir = args.export.resolve()
    runtime = load_runtime(export_dir)
    metadata = json.loads((export_dir / "experiment_export.json").read_text(encoding="utf-8"))
    experiment_id = metadata["experiment_id"]
    policy = torch.jit.load(str(export_dir / "actor.ts"), map_location="cpu").eval()
    torch.set_num_threads(1)
    command = np.asarray((0.0, 0.0, 0.20), dtype=np.float64)
    lower, upper = _command_limits(args.max_speed)
    session = args.session or export_dir.parent / ("viewer-" + time.strftime("%Y%m%d-%H%M%S"))
    session.mkdir(parents=True, exist_ok=True)
    status_path = session / "status.json"
    trajectory_path = session / "trajectory.csv"
    pending = queue.SimpleQueue()
    paused = False
    quit_requested = False
    failure_reason = ""

    def reset_pose():
        nonlocal failure_reason
        runtime.reset(command)
        failure_reason = ""

    def handle_keys():
        nonlocal paused, quit_requested
        while not pending.empty():
            key = pending.get()
            if key == glfw.KEY_ESCAPE:
                quit_requested = True
            elif key == ord("P"):
                paused = not paused
            elif key == ord("R"):
                reset_pose()
                paused = False
            elif key == ord("X"):
                command[:2] = 0.0
            elif key == ord("W"):
                command[0] += 0.1
            elif key == ord("S"):
                command[0] -= 0.1
            elif key == ord("A"):
                command[1] += 0.1
            elif key == ord("D"):
                command[1] -= 0.1
            elif key == ord("Q"):
                command[2] += 0.01
            elif key == ord("E"):
                command[2] -= 0.01
            command[:] = np.clip(command, lower, upper)

    reset_pose()
    display_model = mujoco.MjModel.from_xml_path(str(runtime.model_path))
    display_data = mujoco.MjData(display_model)
    mujoco.mj_copyData(display_data, display_model, runtime.data)
    rows = []
    csv_stream = trajectory_path.open("w", encoding="utf-8", newline="")
    csv_writer = None
    started = time.monotonic()
    with csv_stream, KeyboardViewer(display_model, display_data, pending.put, experiment_id) as viewer:
        viewer.camera.distance = 1.1
        viewer.camera.azimuth = 135
        viewer.camera.elevation = -15
        viewer.camera.lookat[:] = display_data.xipos[runtime.model_map.base_body_id]
        print("MUJOCO_VIEWER_READY", str(session), flush=True)
        while viewer.is_running() and not quit_requested:
            tick_start = time.monotonic()
            handle_keys()
            if quit_requested:
                break
            if not paused:
                with torch.inference_mode():
                    observation = runtime._history.reshape(-1).astype(np.float32, copy=True)
                    action = policy(torch.from_numpy(observation).unsqueeze(0)).detach().cpu().numpy().reshape(-1)
                result = runtime.step(action, command)
                metrics = result.metrics
                outside_fraction = metrics["action_outside_former_unit_interval_fraction"]
                if not 0.10 <= metrics["base_height_m"] <= 0.40:
                    failure_reason = "base_height"
                elif metrics["tilt_rad"] > 0.80:
                    failure_reason = "tilt"
                row = {
                    "time_s": float(runtime.data.time),
                    "command_vx_mps": float(command[0]),
                    "command_yaw_rate_rad_s": float(command[1]),
                    "command_base_height_m": float(command[2]),
                    "raw_action_outside_former_unit_interval_fraction": outside_fraction,
                    "failure_reason": failure_reason,
                    **metrics,
                }
                rows.append(row)
                if csv_writer is None:
                    csv_writer = csv.DictWriter(csv_stream, fieldnames=list(row))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                csv_stream.flush()
                if failure_reason:
                    paused = True
            mujoco.mj_copyData(display_data, display_model, runtime.data)
            metrics = runtime.last_reset_metrics if not rows else rows[-1]
            info = (
                f"{experiment_id} | W/S: vx  A/D: yaw  Q/E: height\n"
                "X: zero vx/yaw  R: reset  P: pause  Esc: close\n"
                f"Command: vx={command[0]:+.2f} m/s  yaw={command[1]:+.2f} rad/s  height={command[2]:.2f} m\n"
                f"Actual vx={metrics['vx_mps']:+.3f} m/s  height={metrics['base_height_m']:.3f} m\n"
                f"tilt={np.rad2deg(metrics['tilt_rad']):.2f} deg  phi0 delta={metrics['phi0_delta_abs_rad']:.3f} rad\n"
                f"raw |a|>=1 fraction={metrics.get('action_outside_former_unit_interval_fraction', 0.0):.3f}  "
                f"torque saturation={metrics.get('effort_saturation_fraction', 0.0):.3f}\n"
                f"status={'PAUSED' if paused else 'RUNNING'}  sim t={runtime.data.time:.2f}s"
                + (f"\nFALL: {failure_reason}. Press R to reset." if failure_reason else "")
            )
            viewer.render(info)
            status_path.write_text(
                json.dumps(
                    {
                        "experiment_id": experiment_id,
                        "export": str(export_dir),
                        "session": str(session),
                        "paused": paused,
                        "time_s": float(runtime.data.time),
                        "command": command.tolist(),
                        "failure_reason": failure_reason,
                        "rows": len(rows),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            time.sleep(max(0.0, runtime.contract.control_dt_s - (time.monotonic() - tick_start)))
    print("MUJOCO_VIEWER_CLOSED", flush=True)


if __name__ == "__main__":
    main()
