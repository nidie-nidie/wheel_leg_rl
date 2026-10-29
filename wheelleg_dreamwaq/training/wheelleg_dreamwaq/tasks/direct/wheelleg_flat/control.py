from __future__ import annotations

from dataclasses import dataclass

import torch

from wheelleg_dreamwaq.schemas.action import clip_action, wheel_targets_control_to_usd


@dataclass(frozen=True)
class ControlLimits:
    leg_action_scale: float = 0.35
    wheel_action_scale: float = 25.0
    leg_target_lower: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0)
    leg_target_upper: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


def compute_action_targets(
    actions: torch.Tensor,
    q_nominal: torch.Tensor,
    limits: ControlLimits,
    action_clip: float | None = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if actions.shape[-1] != 6 or not torch.isfinite(actions).all():
        raise ValueError("动作必须为有限六维向量")
    if action_clip not in (None, 1.0):
        raise ValueError("未支持的动作裁剪边界")
    clipped = actions if action_clip is None else clip_action(actions)
    nominal = q_nominal.to(dtype=actions.dtype, device=actions.device)
    lower = actions.new_tensor(limits.leg_target_lower)
    upper = actions.new_tensor(limits.leg_target_upper)
    leg_targets = torch.clamp(nominal + limits.leg_action_scale * clipped[..., :4], lower, upper)
    wheel_targets_control = limits.wheel_action_scale * clipped[..., 4:6]
    wheel_targets_usd = wheel_targets_control_to_usd(wheel_targets_control)
    if not torch.isfinite(leg_targets).all() or not torch.isfinite(wheel_targets_usd).all():
        raise FloatingPointError("电机目标包含非有限数值")
    return clipped, leg_targets, wheel_targets_usd
