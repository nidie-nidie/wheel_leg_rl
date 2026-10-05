#!/usr/bin/env bash
set -u
python3 -c 'import sys; print(sys.version); import mujoco; print("mujoco", mujoco.__version__)' || true
ls -lh /home/shun/MuJoCoBin/rm_control/mujoco_control_extract/sim/models/wheel_leg_urdf4_assets | head
ls -lh /home/shun/MuJoCoBin/rm_control/mujoco_control_extract/build/mujoco_bridge
ldd /home/shun/MuJoCoBin/rm_control/mujoco_control_extract/build/mujoco_bridge | grep -E 'mujoco|glfw|GL'
find /home/shun -maxdepth 5 -type f \( -name simulate -o -name 'libmujoco.so*' \) 2>/dev/null | head -30
for tool in import xwd grim gnome-screenshot scrot; do command -v "$tool" || true; done
printf 'DISPLAY=%s WAYLAND_DISPLAY=%s\n' "${DISPLAY:-}" "${WAYLAND_DISPLAY:-}"
