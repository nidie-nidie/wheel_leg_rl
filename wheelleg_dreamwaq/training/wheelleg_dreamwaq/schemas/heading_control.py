from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from .frames import R_CONTROL_FROM_USD, quat_rotate_wxyz


HEADING_HOLD_VERSION = "HeadingHoldV1"
RAW_BASELINE_PROFILE = "stop_reverse_raw_split_straight_yaw2_headingp9_v1"
RAW_BASELINE_NO_HEADING_PROFILE = "stop_reverse_raw_split_straight_yaw2_headingoff_v1"
RAW_BASELINE_PROFILES = (RAW_BASELINE_PROFILE, RAW_BASELINE_NO_HEADING_PROFILE)
RUNTIME_GEOMETRY = {
    "mode": "actual_geometry_fk_diagnostic_v1", "minimum_leg_length_m": 0.05,
    "maximum_loop_error_m": 0.005, "fk_non_finite": "fail",
    "reset_geometry": "original_strict_guards",
}


@dataclass(frozen=True)
class HeadingHoldConfig:
    enabled: bool = True
    kp: float = 9.0
    kd: float = 0.0
    max_yaw_rate: float = 1.0
    control_dt_s: float = 0.02

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("heading enabled 必须为 bool")
        if (not all(math.isfinite(v) for v in (self.kp, self.kd, self.max_yaw_rate, self.control_dt_s))
                or min(self.kp, self.kd) < 0 or self.max_yaw_rate <= 0 or self.control_dt_s <= 0):
            raise ValueError("航向控制参数无效")


def heading_contract(config: HeadingHoldConfig) -> dict:
    config.__post_init__()
    return {
        "version": HEADING_HOLD_VERSION, **asdict(config),
        "orientation": "R_control_from_model@R_world_from_body@R_control_from_model.T",
        "error": "atan2(sin(reference-yaw),cos(reference-yaw))",
        "reference_capture": "reset_toggle_and_first_zero_after_nonzero_source_yaw",
        "command_timing": "prepare_before_policy_reward_before_source_advance",
        "wheel_differential_gate": "effective_vx_ne_0_and_effective_yaw_eq_0",
    }


def heading_profile_name(saved: dict) -> str:
    if not isinstance(saved, dict) or type(saved.get("enabled")) is not bool:
        raise ValueError("保存的航向 baseline 开关必须为 bool")
    for name in RAW_BASELINE_PROFILES:
        if saved == heading_contract(HeadingHoldConfig(enabled=name == RAW_BASELINE_PROFILE)):
            return name
    raise ValueError("保存的航向 baseline 配置不符合具名 profile")


def yaw_from_rotation(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3) or not torch.isfinite(rotation).all():
        raise ValueError("航向姿态必须为有限旋转矩阵")
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    if (not torch.allclose(rotation @ rotation.transpose(-2, -1), identity.expand_as(rotation), atol=1e-4, rtol=0)
            or not torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]), atol=1e-4, rtol=0)):
        raise ValueError("航向姿态矩阵不满足旋转定义")
    control = R_CONTROL_FROM_USD.to(device=rotation.device, dtype=rotation.dtype)
    rotated = control @ rotation @ control.T
    return torch.atan2(rotated[..., 1, 0], rotated[..., 0, 0])


def yaw_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    if (quaternion.shape[-1] != 4 or not torch.isfinite(quaternion).all()
            or (quaternion.norm(dim=-1) < 1e-8).any()):
        raise ValueError("航向 quaternion 无效")
    basis = torch.eye(3, dtype=quaternion.dtype, device=quaternion.device)
    columns = [quat_rotate_wxyz(quaternion, basis[i].expand_as(quaternion[..., :3])) for i in range(3)]
    return yaw_from_rotation(torch.stack(columns, dim=-1))


class HeadingHold:
    def __init__(self, config: HeadingHoldConfig, count: int, device="cpu", dtype=torch.float32):
        config.__post_init__()
        if count <= 0:
            raise ValueError("航向环境数量必须为正")
        self.config = config
        self.enabled = config.enabled
        self.reference = torch.zeros(count, device=device, dtype=dtype)
        self.was_turning = torch.zeros(count, device=device, dtype=torch.bool)
        self.initialized = torch.zeros(count, device=device, dtype=torch.bool)
        self.error = torch.zeros_like(self.reference)
        self.saturated = torch.zeros_like(self.was_turning)

    def reset(self, yaw: torch.Tensor, indices: torch.Tensor | None = None) -> None:
        indices = torch.arange(len(self.reference), device=self.reference.device) if indices is None else indices
        if yaw.shape != self.reference[indices].shape or not torch.isfinite(yaw).all():
            raise ValueError("reset 航向形状或数值无效")
        yaw = yaw.to(self.reference)
        self.reference[indices] = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        self.was_turning[indices] = False
        self.initialized[indices] = True
        self.error[indices] = 0
        self.saturated[indices] = False

    def toggle(self, enabled: bool, yaw: torch.Tensor) -> None:
        if type(enabled) is not bool:
            raise ValueError("航向开关必须为 bool")
        self.reset(yaw)
        self.enabled = enabled

    def update(self, yaw: torch.Tensor, yaw_rate: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if (yaw.shape != self.reference.shape or yaw_rate.shape != yaw.shape
                or source.shape != (len(yaw), 3) or not self.initialized.all()
                or not all(torch.isfinite(v).all() for v in (yaw, yaw_rate, source))):
            raise ValueError("航向输入未初始化、形状错误或包含非有限数值")
        if (source[:, 1].abs() > self.config.max_yaw_rate).any():
            raise ValueError("源偏航命令超出范围")
        turning = source[:, 1] != 0
        capture = turning | self.was_turning | (not self.enabled)
        self.reference.copy_(torch.where(capture, yaw, self.reference))
        self.error.copy_(torch.atan2(torch.sin(self.reference - yaw), torch.cos(self.reference - yaw)))
        correction = self.config.kp * self.error - self.config.kd * yaw_rate
        if not torch.isfinite(correction).all():
            raise FloatingPointError("航向控制输出溢出")
        holding = (~turning) & self.enabled
        self.saturated.copy_(holding & (correction.abs() > self.config.max_yaw_rate))
        result = source.clone()
        result[:, 1] = torch.where(holding, correction.clamp(-self.config.max_yaw_rate,
                                                          self.config.max_yaw_rate), source[:, 1])
        self.was_turning.copy_(turning & self.enabled)
        return result
