#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/c/Users/shun/Desktop/wheel_leg_debug-main2
MUJOCO=/home/shun/MuJoCoBin/mujoco-3.3.0
OUT="$ROOT/tmp/slides/robot_motion_interview_v2/assets"

gcc "$ROOT/tmp/render_mujoco_cover.c" \
  -I"$MUJOCO/include" -L"$MUJOCO/lib" \
  -Wl,-rpath,"$MUJOCO/lib" \
  -lmujoco -lglfw -lGL -ldl -lpthread -lm \
  -o "$ROOT/tmp/render_mujoco_cover"

"$ROOT/tmp/render_mujoco_cover" \
  "$OUT/mujoco_suspended.xml" \
  "$OUT/mujoco_suspended.ppm"
