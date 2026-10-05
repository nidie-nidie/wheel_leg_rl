#!/usr/bin/env bash
set -euo pipefail

SIMULATE=/home/shun/MuJoCoBin/mujoco-3.3.0/bin/simulate
MODEL=/mnt/c/Users/shun/Desktop/wheel_leg_debug-main2/tmp/slides/robot_motion_interview_v2/assets/mujoco_suspended.xml
OUT=/mnt/c/Users/shun/Desktop/wheel_leg_debug-main2/tmp/slides/robot_motion_interview_v2/assets/mujoco_suspended.xwd

"$SIMULATE" "$MODEL" >/tmp/codex_mujoco_simulate.log 2>&1 &
pid=$!
cleanup() {
  kill "$pid" 2>/dev/null || true
}
trap cleanup EXIT
sleep 6
window_id="$(xwininfo -root -tree | awk '/MuJoCo|simulate/ {print $1; exit}')"
if [[ -z "$window_id" ]]; then
  echo "MuJoCo window not found" >&2
  exit 2
fi
echo "Capturing MuJoCo window $window_id"
xwd -id "$window_id" -silent -out "$OUT"
