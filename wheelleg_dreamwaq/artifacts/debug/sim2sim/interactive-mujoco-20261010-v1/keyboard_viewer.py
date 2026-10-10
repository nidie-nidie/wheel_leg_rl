"""Small MuJoCo renderer whose keyboard callback owns every key event."""
from __future__ import annotations

import contextlib

import glfw
import mujoco


class KeyboardViewer:
    def __init__(self, model, data, *, key_callback, show_left_ui=False, show_right_ui=False):
        del show_left_ui, show_right_ui
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        self.window = glfw.create_window(1280, 800, "WheelLeg MuJoCo | Run-04 keyboard control", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Unable to create MuJoCo keyboard window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.model, self.data = model, data
        self.key_callback = key_callback
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.texts = []
        self.last_cursor = glfw.get_cursor_pos(self.window)
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_cursor_pos_callback(self.window, self.on_cursor)
        glfw.set_scroll_callback(self.window, self.on_scroll)

    def on_key(self, window, key, scancode, action, mods):
        del window, scancode, mods
        if action == glfw.PRESS:
            # No MuJoCo Simulate UI callback is installed, so W/D cannot also
            # toggle wireframe, collision rendering or other native shortcuts.
            self.key_callback(key)

    def on_cursor(self, window, x, y):
        old_x, old_y = self.last_cursor
        self.last_cursor = (x, y)
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not (left or right):
            return
        height = max(1, glfw.get_window_size(window)[1])
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_V if left else mujoco.mjtMouse.mjMOUSE_MOVE_V
        mujoco.mjv_moveCamera(self.model, action, (x - old_x) / height, (y - old_y) / height, self.cam)

    def on_scroll(self, window, dx, dy):
        del window, dx
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -.05 * dy, self.cam)

    def lock(self):
        return contextlib.nullcontext()

    def is_running(self):
        return self.window is not None and not glfw.window_should_close(self.window)

    def set_texts(self, texts):
        self.texts = [texts] if isinstance(texts, tuple) else texts

    def sync(self, state_only=True):
        del state_only
        glfw.poll_events()
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        for _, gridpos, left, right in self.texts:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, gridpos, viewport, left or "", right or "", self.context)
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
