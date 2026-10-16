from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from wheelleg_dreamwaq.schemas.action import ACTION_DIM
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import (
    RewardWeights, SplitSmoothRewardWeights,
)

from debug.no_action_clip.contract import EXPERIMENT_ID


BASE_REWARD_PROFILE = "stop_reverse_v1"
SPLIT_STRAIGHT_PROFILE = "no_clip_split_straight_v1"
SPLIT_STRAIGHT_EXPERIMENT_ID = "DebugNoActionClipSplitStraightV1"
REWARD_PROFILES = (BASE_REWARD_PROFILE, SPLIT_STRAIGHT_PROFILE)
CONTROL_DT = 0.02


@dataclass(frozen=True, kw_only=True)
class SplitStraightRewardWeights(SplitSmoothRewardWeights):
    wheel_differential: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if (isinstance(self.wheel_differential, bool)
                or not math.isfinite(self.wheel_differential)
                or self.wheel_differential >= 0.0):
            raise ValueError("wheel_differential 必须为有限负数")


def resolve_reward_weights(profile: str, coefficient: float | None) -> RewardWeights:
    if profile == BASE_REWARD_PROFILE:
        if coefficient is not None:
            raise ValueError("旧 profile 不接受 --wheel-differential-weight")
        return RewardWeights(tracking_vx=2.0, tracking_vx_enhance=2.0)
    if profile != SPLIT_STRAIGHT_PROFILE:
        raise ValueError(f"未知 debug 奖励 profile：{profile}")
    if (coefficient is None or isinstance(coefficient, bool)
            or not math.isfinite(coefficient) or coefficient <= 0.0):
        raise ValueError("新 profile 必须显式指定有限正数 --wheel-differential-weight")
    return SplitStraightRewardWeights(
        tracking_vx=2.0, tracking_vx_enhance=2.0, wheel_differential=-float(coefficient),
    )


def reward_manifest_fields(weights: RewardWeights) -> dict:
    if isinstance(weights, SplitStraightRewardWeights):
        return {
            "experiment_id": SPLIT_STRAIGHT_EXPERIMENT_ID,
            "task_profile": SPLIT_STRAIGHT_PROFILE,
            "reward_weights": asdict(weights),
            "reward_contract": {
                "schema_version": "DebugSplitStraightRewardV1",
                "control_dt_s": CONTROL_DT,
                "smooth": {
                    "formula": "sum((a_t-2*a_previous+a_previous_previous)^2)",
                    "leg_indices": [0, 1, 2, 3], "wheel_indices": [4, 5],
                    "leg_weight": weights.action_smooth_leg,
                    "wheel_weight": weights.action_smooth_wheel,
                    "composite_weight": 0.0,
                },
                "wheel_differential": {
                    "formula": "(raw_action_left-raw_action_right)^2",
                    "action_frame": "canonical_same_forward_direction",
                    "command_vx_not_equal": 0.0,
                    "command_yaw_rate_equal": 0.0,
                    "command_timing": "before_command_practice_advance",
                    "weight": weights.wheel_differential,
                },
            },
        }
    if type(weights) is not RewardWeights:
        raise ValueError("不支持的 debug 奖励权重类型")
    return {"experiment_id": EXPERIMENT_ID, "task_profile": BASE_REWARD_PROFILE,
            "reward_weights": asdict(weights)}


def validate_reward_manifest(manifest: dict) -> None:
    profile = manifest.get("task_profile")
    saved_weights = manifest.get("reward_weights", {})
    coefficient = None
    if profile == SPLIT_STRAIGHT_PROFILE:
        saved_coefficient = saved_weights.get("wheel_differential")
        if (not isinstance(saved_coefficient, (int, float))
                or isinstance(saved_coefficient, bool)):
            raise ValueError("保存的差速奖励缺少有效权重")
        coefficient = -saved_coefficient
    expected = reward_manifest_fields(resolve_reward_weights(profile, coefficient))
    for name in ("experiment_id", "task_profile", "reward_weights", "reward_contract"):
        if manifest.get(name) != expected.get(name):
            raise ValueError(f"保存的 debug 奖励身份或配置不符：{name}")


def straight_wheel_cost(actions: torch.Tensor, commands: torch.Tensor):
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or commands.shape != (actions.shape[0], 3):
        raise ValueError("差速成本要求动作 [N,6] 和命令 [N,3]")
    if actions.device != commands.device:
        raise ValueError("动作和命令必须在同一设备")
    if not torch.isfinite(actions).all() or not torch.isfinite(commands).all():
        raise FloatingPointError("差速成本输入包含 NaN 或 Inf")
    enabled = (commands[:, 0] != 0.0) & (commands[:, 1] == 0.0)
    difference_squared = (actions[:, 4] - actions[:, 5]).square()
    if not torch.isfinite(difference_squared).all():
        raise FloatingPointError("差速成本平方溢出")
    cost = torch.where(enabled, difference_squared, torch.zeros_like(difference_squared))
    return cost, enabled


def wheel_differential_reward(actions, commands, weights, *, control_dt: float):
    if not isinstance(weights, SplitStraightRewardWeights):
        raise ValueError("差速奖励要求 SplitStraightRewardWeights")
    weights.__post_init__()
    if not math.isfinite(control_dt) or not math.isclose(control_dt, CONTROL_DT, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("差速奖励要求 control_dt=0.02 秒")
    cost, enabled = straight_wheel_cost(actions, commands)
    weighted = weights.wheel_differential * control_dt * cost
    if not torch.isfinite(weighted).all():
        raise FloatingPointError("加权差速奖励包含 NaN 或 Inf")
    return weighted, cost, enabled
