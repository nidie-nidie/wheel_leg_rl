from __future__ import annotations

import torch

from .contracts import RecoveryContractV2
from .validation import all_finite


def compute_reward(projected_gravity: torch.Tensor, joint_position: torch.Tensor,
                   joint_velocity: torch.Tensor, angular_velocity: torch.Tensor,
                   height: torch.Tensor, torque: torch.Tensor, effort_limits: torch.Tensor,
                   action: torch.Tensor, previous_action: torch.Tensor,
                   previous_previous_action: torch.Tensor, previous_potential: torch.Tensor,
                   success: torch.Tensor, failure: torch.Tensor, has_second_difference: torch.Tensor,
                   q_nominal: torch.Tensor | tuple[float, ...] | None = None,
                   contract: RecoveryContractV2 | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = contract or RecoveryContractV2()
    batch = action.shape[:-1]
    scalar = lambda value: torch.as_tensor(value, device=action.device).reshape(batch)
    success, failure = scalar(success).bool(), scalar(failure).bool()
    if (success & failure).any():
        raise ValueError("奖励成功与失败标志不能同时成立")
    values = (projected_gravity, joint_position, joint_velocity, angular_velocity, height, torque,
              effort_limits, action, previous_action, previous_previous_action, previous_potential)
    if not (all_finite(*values) & (effort_limits > 0).all()):
        raise FloatingPointError("非有限状态或无效力矩能力不能进入PPO")
    if not torch.allclose(torch.linalg.vector_norm(projected_gravity, dim=-1), torch.ones(batch,
                          device=action.device, dtype=action.dtype), atol=1e-4):
        raise ValueError("奖励使用真实单位重力方向")
    nominal = torch.as_tensor(cfg.q_nominal if q_nominal is None else q_nominal,
                              device=action.device, dtype=action.dtype)
    c = ((1 - projected_gravity[..., 2]) / 2).clamp(0, 1)
    gate = (-projected_gravity[..., 2]).clamp(-1, 1).acos() < cfg.reward_gate_tilt_rad
    terminal = success | failure
    next_potential = torch.where(terminal, torch.zeros_like(c), c)
    difference = action - 2 * previous_action + previous_previous_action
    valid_difference = scalar(has_second_difference).to(action.dtype)
    dt = cfg.control_dt_s
    terms = {
        "upright": dt * cfg.upright_weight * c,
        "progress": cfg.potential_weight * (cfg.gamma * next_potential - scalar(previous_potential)),
        "height": dt * cfg.height_weight * gate * torch.exp(-((scalar(height) - cfg.height_target_m) / cfg.height_sigma_m) ** 2),
        "stable": dt * cfg.stable_weight * gate * torch.exp(
            -((joint_position[..., :4] - nominal) / cfg.pose_sigma_rad).square().mean(-1)
            -(angular_velocity / cfg.angular_sigma_rad_s).square().sum(-1)
            -(joint_velocity[..., 4:] / cfg.wheel_sigma_rad_s).square().mean(-1)),
        "torque": -dt * cfg.torque_weight * (torque / effort_limits).square().mean(-1),
        "leg_smooth": -dt * cfg.leg_smooth_weight * difference[..., :4].square().mean(-1) * valid_difference,
        "wheel_smooth": -dt * cfg.wheel_smooth_weight * difference[..., 4:].square().mean(-1) * valid_difference,
        "time": torch.full_like(c, -dt * cfg.time_weight),
        "success": success.to(action.dtype) * cfg.success_reward,
        "failure": failure.to(action.dtype) * cfg.failure_reward,
    }
    reward = torch.stack(tuple(terms.values())).sum(dim=0)
    if not torch.isfinite(reward).all():
        raise FloatingPointError("非有限奖励不能进入PPO")
    terms["potential"] = c
    return reward, terms
