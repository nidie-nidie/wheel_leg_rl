from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import RecoveryContractV2
from .validation import all_finite


@dataclass(frozen=True)
class ActorNoise:
    angular_velocity: torch.Tensor
    projected_gravity: torch.Tensor
    leg_position_error: torch.Tensor
    joint_velocity: torch.Tensor


def normalize_physical_frame(frame: torch.Tensor, contract: RecoveryContractV2 | None = None) -> torch.Tensor:
    cfg = contract or RecoveryContractV2()
    if frame.shape[-1] != 22 or not torch.isfinite(frame).all():
        raise ValueError("起身物理观测必须为有限22维张量")
    return torch.cat((torch.clamp(frame[..., :3] * cfg.angular_velocity_scale, -cfg.angular_velocity_clip, cfg.angular_velocity_clip),
                      torch.clamp(frame[..., 3:6], -cfg.projected_gravity_clip, cfg.projected_gravity_clip),
                      torch.clamp(frame[..., 6:10] * cfg.leg_position_scale, -cfg.leg_position_clip, cfg.leg_position_clip),
                      torch.clamp(frame[..., 10:16] * cfg.joint_velocity_scale, -cfg.joint_velocity_clip, cfg.joint_velocity_clip),
                      frame[..., 16:22]), dim=-1)


def build_actor_frame(angular_velocity: torch.Tensor, projected_gravity: torch.Tensor,
                      joint_position: torch.Tensor, joint_velocity: torch.Tensor,
                      previous_action: torch.Tensor, q_reference: torch.Tensor,
                      actor_noise: ActorNoise | None = None) -> torch.Tensor:
    batch = angular_velocity.shape[:-1]
    for value, size in ((angular_velocity, 3), (projected_gravity, 3), (joint_position, 6),
                        (joint_velocity, 6), (previous_action, 6)):
        if value.shape != (*batch, size):
            raise ValueError("起身观测的形状或有限数值检查失败")
    reference = torch.as_tensor(q_reference, dtype=joint_position.dtype, device=joint_position.device)
    if reference.shape[-1] != 4 or not all_finite(angular_velocity, projected_gravity, joint_position,
                                               joint_velocity, previous_action, reference):
        raise ValueError("起身角度参考无效")
    gravity = projected_gravity
    if not torch.allclose(torch.linalg.vector_norm(gravity, dim=-1), torch.ones(batch, device=gravity.device,
                          dtype=gravity.dtype), atol=1e-4):
        raise ValueError("真实重力观测必须为单位向量")
    gyro, leg_error, velocity = angular_velocity, joint_position[..., :4] - reference, joint_velocity
    if actor_noise is not None:
        pairs = ((gyro, actor_noise.angular_velocity), (gravity, actor_noise.projected_gravity),
                 (leg_error, actor_noise.leg_position_error), (velocity, actor_noise.joint_velocity))
        if any(clean.shape != noise.shape for clean, noise in pairs) or not all_finite(*(noise for _, noise in pairs)):
            raise ValueError("起身观测增强形状错误或数值非有限")
        gyro = gyro + actor_noise.angular_velocity
        gravity = gravity + actor_noise.projected_gravity
        norm = torch.linalg.vector_norm(gravity, dim=-1, keepdim=True)
        if (norm < 1e-6).any():
            raise ValueError("增强重力方向不可归一化")
        gravity = gravity / norm
        leg_error = leg_error + actor_noise.leg_position_error
        velocity = velocity + actor_noise.joint_velocity
    return normalize_physical_frame(torch.cat((gyro, gravity, leg_error, velocity, previous_action), dim=-1))


def build_critic_frame(clean_frame: torch.Tensor, linear_velocity: torch.Tensor,
                       height: torch.Tensor, torque: torch.Tensor, confirm_fraction: torch.Tensor,
                       remaining_fraction: torch.Tensor) -> torch.Tensor:
    cfg = RecoveryContractV2()
    batch = clean_frame.shape[:-1]
    if clean_frame.shape[-1] != 22 or linear_velocity.shape != (*batch, 3) or torque.shape != (*batch, 6):
        raise ValueError("Critic观测形状不一致")
    scalars = [torch.as_tensor(v, device=clean_frame.device, dtype=clean_frame.dtype).reshape(*batch, 1)
               for v in (height, confirm_fraction, remaining_fraction)]
    result = torch.cat((clean_frame, torch.clamp(linear_velocity * cfg.root_linear_velocity_scale,
                        -cfg.root_linear_velocity_clip, cfg.root_linear_velocity_clip),
                        torch.clamp((scalars[0] - cfg.nominal_base_height) / cfg.height_command_span,
                                    -cfg.base_height_clip, cfg.base_height_clip),
                        torch.clamp(torque * cfg.applied_torque_scale, -cfg.applied_torque_clip, cfg.applied_torque_clip),
                        scalars[1].clamp(0, 1), scalars[2].clamp(0, 1)), dim=-1)
    if result.shape[-1] != 34 or not torch.isfinite(result).all():
        raise ValueError("Critic必须为有限34维观测")
    return result


class HistoryBuffer:
    def __init__(self, num_envs: int, frame_dim: int = 22, history_length: int = 5,
                 device: str | torch.device = "cpu") -> None:
        if num_envs <= 0 or frame_dim <= 0 or history_length != 5:
            raise ValueError("历史配置无效")
        self._frames = torch.zeros(num_envs, history_length, frame_dim, device=device)
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._last_frame_id: int | None = None

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._frames.device)
        expected = (len(ids), *self._frames.shape[1:])
        if frames.shape != expected or not torch.isfinite(frames).all() or len(ids) != len(torch.unique(ids)):
            raise ValueError("局部历史reset数据无效")
        self._frames[ids] = frames.to(self._frames)
        self._initialized[ids] = True

    def append(self, frame: torch.Tensor, frame_id: int) -> None:
        if not self._initialized.all() or frame.shape != self._frames[:, -1].shape or not torch.isfinite(frame).all():
            raise ValueError("历史尚未初始化或当前帧无效")
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError("每个真实控制步只允许追加一次历史")
        self._frames[:, :-1] = self._frames[:, 1:].clone()
        self._frames[:, -1] = frame
        self._last_frame_id = frame_id

    def frames(self) -> torch.Tensor:
        return self._frames.clone()

    def flattened(self) -> torch.Tensor:
        return self._frames.flatten(start_dim=1).clone()
