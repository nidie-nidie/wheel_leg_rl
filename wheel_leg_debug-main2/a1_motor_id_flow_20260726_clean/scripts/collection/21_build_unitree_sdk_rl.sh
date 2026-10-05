#!/usr/bin/env bash
set -euo pipefail

BUILD_ARGS=()
if [[ $# -eq 0 ]]; then
  :
elif [[ $# -eq 2 && "$1" == "--target" && \
  ( "$2" == "a1_pace_collect_real" || "$2" == "a1_joint_chirp_collect_real" ) ]]; then
  BUILD_ARGS=(--target "$2")
else
  echo "usage: $0 [--target a1_pace_collect_real|a1_joint_chirp_collect_real]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A1_BASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RL_DIR="${A1_BASE_ROOT}/real/unitree_sdk_rl"
BUILD_DIR="${RL_DIR}/build"

if [[ -z "${UNITREE_LEGGED_SDK_ROOT:-}" ]]; then
  if [[ -d /home/unitree/unitree_legged_sdk ]]; then
    UNITREE_LEGGED_SDK_ROOT="/home/unitree/unitree_legged_sdk"
  else
    UNITREE_LEGGED_SDK_ROOT="/home/changba01/A1_Base-main/external/unitree_legged_sdk_v3.3.1"
  fi
fi
if [[ -z "${UNITREE_LCM_ROOT:-}" ]]; then
  if [[ -d /home/unitree/lcm-1.4.0 ]]; then
    UNITREE_LCM_ROOT="/home/unitree/lcm-1.4.0"
  else
    UNITREE_LCM_ROOT="/home/changba01/A1_Base-main/external/lcm-local"
  fi
fi
GOGO_A1_POLICY_MODEL_DIR="${GOGO_A1_POLICY_MODEL_DIR:-${A1_BASE_ROOT}/src_manifest/gogo_a1_policy_model_8346}"

echo "== Build Unitree SDK real A1 RL runtime =="
echo "Runtime: ${RL_DIR}"
echo "Unitree SDK: ${UNITREE_LEGGED_SDK_ROOT}"
echo "Unitree LCM: ${UNITREE_LCM_ROOT}"
echo "Policy model: ${GOGO_A1_POLICY_MODEL_DIR}"

mkdir -p "${BUILD_DIR}"
(
  cd "${BUILD_DIR}"
  cmake "${RL_DIR}" \
    -DUNITREE_LEGGED_SDK_ROOT="${UNITREE_LEGGED_SDK_ROOT}" \
    -DUNITREE_LCM_ROOT="${UNITREE_LCM_ROOT}" \
    -DGOGO_A1_POLICY_MODEL_DIR="${GOGO_A1_POLICY_MODEL_DIR}"
  cmake --build . "${BUILD_ARGS[@]}"
)

echo
if [[ ${#BUILD_ARGS[@]} -eq 0 ]]; then
  echo "Built: ${BUILD_DIR}/a1_rl_policy_real"
  echo "Built: ${BUILD_DIR}/a1_pace_observe"
fi
echo "Built: ${BUILD_DIR}/a1_pace_collect_real"
echo "Built: ${BUILD_DIR}/a1_joint_chirp_collect_real"
