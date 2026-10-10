"""Self-contained MuJoCo runtime for WheelLeg Phase 1 PPO policies."""

from .runner import WheelLegMujocoRuntime
from .versions import MUJOCO_MODEL_VERSION

__all__ = ["MUJOCO_MODEL_VERSION", "WheelLegMujocoRuntime"]
