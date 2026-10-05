from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass(frozen=True)
class NormalizationV1:
    schema_version: ClassVar[str] = "NormalizationV1"
    angular_velocity_scale: float = 0.25
    angular_velocity_clip: float = 5.0
    projected_gravity_clip: float = 1.0
    vx_max_abs: float = 1.0
    yaw_rate_max_abs: float = 1.5
    nominal_base_height: float = 0.20
    height_command_span: float = 0.04
    leg_position_scale: float = 1.0
    leg_position_clip: float = 1.5
    joint_velocity_scale: float = 0.05
    joint_velocity_clip: float = 5.0
    root_linear_velocity_scale: float = 2.0
    root_linear_velocity_clip: float = 6.0
    joint_acceleration_scale: float = 0.002
    joint_acceleration_clip: float = 10.0
    applied_torque_scale: float = 0.05
    applied_torque_clip: float = 5.0

    @staticmethod
    def _scaled_clip(values: torch.Tensor, scale: float, clip: float) -> torch.Tensor:
        return torch.clamp(values * scale, -clip, clip)

    def normalize_angular_velocity(self, values: torch.Tensor) -> torch.Tensor:
        return self._scaled_clip(values, self.angular_velocity_scale, self.angular_velocity_clip)

    def normalize_projected_gravity(self, values: torch.Tensor) -> torch.Tensor:
        return torch.clamp(values, -self.projected_gravity_clip, self.projected_gravity_clip)

    def normalize_command(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != 3:
            raise ValueError(f"Expected CommandV1 dimension 3, got {values.shape[-1]}")
        result = torch.empty_like(values)
        result[..., 0] = values[..., 0] / self.vx_max_abs
        result[..., 1] = values[..., 1] / self.yaw_rate_max_abs
        result[..., 2] = (values[..., 2] - self.nominal_base_height) / self.height_command_span
        return torch.clamp(result, -1.0, 1.0)

    def normalize_leg_position_error(self, values: torch.Tensor) -> torch.Tensor:
        return self._scaled_clip(values, self.leg_position_scale, self.leg_position_clip)

    def normalize_joint_velocity(self, values: torch.Tensor) -> torch.Tensor:
        return self._scaled_clip(values, self.joint_velocity_scale, self.joint_velocity_clip)

    @staticmethod
    def normalize_previous_action(values: torch.Tensor) -> torch.Tensor:
        return torch.clamp(values, -1.0, 1.0)

    def normalize_root_linear_velocity(self, values: torch.Tensor) -> torch.Tensor:
        return self._scaled_clip(values, self.root_linear_velocity_scale, self.root_linear_velocity_clip)

    def normalize_base_height(self, values: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (values - self.nominal_base_height) / self.height_command_span,
            -5.0,
            5.0,
        )

    def normalize_joint_acceleration(self, values: torch.Tensor) -> torch.Tensor:
        return self._scaled_clip(values, self.joint_acceleration_scale, self.joint_acceleration_clip)

    def normalize_applied_torque(self, values: torch.Tensor) -> torch.Tensor:
        return self._scaled_clip(values, self.applied_torque_scale, self.applied_torque_clip)


@dataclass(frozen=True)
class NormalizationV2(NormalizationV1):
    schema_version: ClassVar[str] = "NormalizationV2"
    vx_max_abs: float = 1.5
    yaw_rate_max_abs: float = 1.0


NORMALIZATION_SCHEMA_VERSION = NormalizationV2.schema_version
