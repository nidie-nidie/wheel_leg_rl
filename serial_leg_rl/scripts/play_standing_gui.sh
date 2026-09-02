#!/usr/bin/env bash
set -euo pipefail

cd /home/shun/桌面/wheel_leg_rl
source wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab/.venv/bin/activate

LOG_ROOT="logs/rsl_rl/serial_leg_standing_direct"
mapfile -t MODEL_LINES < <(find "${LOG_ROOT}" -mindepth 2 -maxdepth 2 -name 'model_*.pt' -printf '%T@ %p\n' | sort -nr)
if ((${#MODEL_LINES[@]} == 0)); then
  echo "[ERROR] No model_*.pt checkpoint found under ${LOG_ROOT}" >&2
  exit 1
fi
LATEST_MODEL="${MODEL_LINES[0]#* }"

LOAD_RUN="${LOAD_RUN:-$(basename "$(dirname "${LATEST_MODEL}")")}"
CHECKPOINT="${CHECKPOINT:-$(basename "${LATEST_MODEL}")}"
echo "[INFO] Playing checkpoint: ${LOAD_RUN}/${CHECKPOINT}"

TERM=xterm ./wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab/isaaclab.sh \
  -p serial_leg_rl/scripts/play_standing_rsl_rl.py \
  --task SerialLeg-Standing-Direct-v0 \
  --num_envs 1 \
  --load_run "${LOAD_RUN}" \
  --checkpoint "${CHECKPOINT}" \
  --num_steps 3000 \
  --device cuda:0 \
  --kit_args "--/ngx/enabled=false --/rtx/post/dlss/enabled=false --/rtx/post/dlss/execMode=0 --/rtx-transient/dlssg/enabled=false --/rtx-transient/dldenoiser/enabled=false" \
  "$@"
