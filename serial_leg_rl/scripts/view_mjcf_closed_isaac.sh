#!/usr/bin/env bash
set -euo pipefail

cd /home/shun/桌面/wheel_leg_rl
source wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab/.venv/bin/activate

TERM=xterm ./wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab/isaaclab.sh \
  -p serial_leg_rl/scripts/view_mjcf_closed_in_isaac.py \
  --num_steps 3000 \
  --real_time \
  --device cuda:0 \
  --kit_args "--/ngx/enabled=false --/rtx/post/dlss/enabled=false --/rtx/post/dlss/execMode=0 --/rtx-transient/dlssg/enabled=false --/rtx-transient/dldenoiser/enabled=false"
