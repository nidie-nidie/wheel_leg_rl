from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WheelLegState:
    root_link_pos_w: torch.Tensor
    root_link_quat_w: torch.Tensor
    root_com_pos_w: torch.Tensor
    root_com_linear_velocity: torch.Tensor
    root_angular_velocity: torch.Tensor
    projected_gravity: torch.Tensor
    base_height: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    joint_acceleration: torch.Tensor
    applied_torque: torch.Tensor
    last_applied_action: torch.Tensor
    previous_applied_action: torch.Tensor
    previous_previous_applied_action: torch.Tensor
    command: torch.Tensor
    virtual_leg_length_true: torch.Tensor
    virtual_leg_phi0_true: torch.Tensor
    virtual_leg_length_fk: torch.Tensor
    virtual_leg_phi0_fk: torch.Tensor
    virtual_leg_fk_valid: torch.Tensor
    virtual_leg_wheel_error: torch.Tensor
    loop_closure_position_error: torch.Tensor

    def assert_finite(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise RuntimeError(f"WheelLegState field contains NaN or Inf: {name}")
