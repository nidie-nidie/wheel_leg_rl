from __future__ import annotations

import torch

from .contracts import RecoveryContractV2, WALKER_LEG_TARGET_LOWER, WALKER_LEG_TARGET_UPPER


def _finite(values: torch.Tensor, dimension: int, name: str) -> None:
    if values.shape[-1] != dimension or not torch.isfinite(values).all():
        raise ValueError(f"{name}必须为有限{dimension}维张量")


def decode_action(actions: torch.Tensor, q_reference: torch.Tensor,
                  contract: RecoveryContractV2 | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    contract = contract or RecoveryContractV2()
    _finite(actions, 6, "动作")
    reference = torch.as_tensor(q_reference, dtype=actions.dtype, device=actions.device)
    _finite(reference, 4, "执行器参考")
    legs = reference + contract.leg_action_scale * actions[..., :4]
    wheels = contract.wheel_action_scale * actions[..., 4:]
    _finite(legs, 4, "腿部目标")
    _finite(wheels, 2, "轮速目标")
    return legs, wheels


def decode_walker_action(actions: torch.Tensor, q_reference: torch.Tensor,
                         contract: RecoveryContractV2 | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    legs, wheels = decode_action(actions, q_reference, contract)
    return torch.clamp(legs, WALKER_LEG_TARGET_LOWER, WALKER_LEG_TARGET_UPPER), wheels


def estimate_pd_torque(leg_targets: torch.Tensor, wheel_targets: torch.Tensor,
                       joint_position: torch.Tensor, joint_velocity: torch.Tensor,
                       kp: torch.Tensor, kd: torch.Tensor,
                       effort_limits: torch.Tensor) -> torch.Tensor:
    for value, size, name in ((leg_targets, 4, "腿部目标"), (wheel_targets, 2, "轮速目标"),
                              (joint_position, 6, "关节角度"), (joint_velocity, 6, "关节速度"),
                              (kp, 6, "刚度"), (kd, 6, "阻尼"), (effort_limits, 6, "力矩能力")):
        _finite(value, size, name)
    if (effort_limits <= 0).any() or (kp < 0).any() or (kd < 0).any():
        raise ValueError("执行器参数必须非负且力矩能力为正")
    leg_torque = kp[..., :4] * (leg_targets - joint_position[..., :4]) - kd[..., :4] * joint_velocity[..., :4]
    wheel_torque = kd[..., 4:] * (wheel_targets - joint_velocity[..., 4:])
    torque = torch.cat((leg_torque, wheel_torque), dim=-1)
    return torch.maximum(torch.minimum(torque, effort_limits), -effort_limits)


def check_action_compatibility(actual_action: torch.Tensor, candidate_action: torch.Tensor,
                               joint_position: torch.Tensor, joint_velocity: torch.Tensor,
                               q_reference: torch.Tensor, kp: torch.Tensor, kd: torch.Tensor,
                               effort_limits: torch.Tensor, target_tolerance: torch.Tensor,
                               torque_tolerance: torch.Tensor,
                               contract: RecoveryContractV2 | None = None, *,
                               actual_is_walker: bool = False) -> dict[str, torch.Tensor]:
    _finite(target_tolerance, 6, "目标差门限")
    _finite(torque_tolerance, 6, "力矩差门限")
    if (target_tolerance <= 0).any() or (torque_tolerance <= 0).any():
        raise ValueError("交接兼容门限必须来自有效的正数校准结果")
    actual_decoder = decode_walker_action if actual_is_walker else decode_action
    actual_leg, actual_wheel = actual_decoder(actual_action, q_reference, contract)
    candidate_leg, candidate_wheel = decode_walker_action(candidate_action, q_reference, contract)
    actual_torque = estimate_pd_torque(actual_leg, actual_wheel, joint_position, joint_velocity, kp, kd, effort_limits)
    candidate_torque = estimate_pd_torque(candidate_leg, candidate_wheel, joint_position, joint_velocity, kp, kd, effort_limits)
    target_difference = torch.abs(torch.cat((actual_leg - candidate_leg, actual_wheel - candidate_wheel), dim=-1))
    torque_difference = torch.abs(actual_torque - candidate_torque)
    target_blocked = target_difference >= target_tolerance
    torque_blocked = torque_difference >= torque_tolerance
    return {"compatible": ~(target_blocked | torque_blocked).any(dim=-1),
            "target_difference": target_difference, "torque_difference": torque_difference,
            "target_blocked": target_blocked, "torque_blocked": torque_blocked}
