"""Exercise the installed keyboard handler and check rendering stays read-only."""
import json
from pathlib import Path
import queue

import glfw
import mujoco
import numpy as np

from keyboard_viewer import KeyboardViewer

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
model = mujoco.MjModel.from_xml_path(str(PROJECT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
data.xfrc_applied[1, 3:6] = (1.1, -2.2, 3.3)
fields = {name: getattr(data, name).copy() for name in ("qpos", "qvel", "xfrc_applied")}
keys = queue.SimpleQueue()
with KeyboardViewer(model, data, key_callback=keys.put) as viewer:
    viewer.cam.distance = 1.1
    viewer.cam.lookat[:] = data.xipos[1]
    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Keyboard regression check: W/S A/D Q/E X R P", ""))
    viewer.sync()
    visual_flags = viewer.scene.flags.copy()
    options = viewer.opt.flags.copy()
    for key in [*(ord(char) for char in "WSADQEXRP"), glfw.KEY_ESCAPE]:
        viewer.on_key(viewer.window, key, 0, glfw.PRESS, 0)
        assert keys.get() == key
        viewer.on_key(viewer.window, key, 0, glfw.RELEASE, 0)
        assert keys.empty()
        viewer.sync()
        np.testing.assert_array_equal(viewer.scene.flags, visual_flags)
        np.testing.assert_array_equal(viewer.opt.flags, options)
        for name, before in fields.items():
            np.testing.assert_array_equal(getattr(data, name), before)
report = {"passed": True, "press_keys_verified": "WSADQEXRP + Escape", "release_ignored": True,
          "native_display_shortcuts_installed": False, "visual_flags_changed": False,
          "physics_state_and_external_force_changed": False,
          "validation": "Calls the same handler registered with GLFW; renders the real OpenGL window"}
(ROOT / "keyboard-viewer-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("KEYBOARD_VIEWER_VERIFIED", json.dumps(report), flush=True)
