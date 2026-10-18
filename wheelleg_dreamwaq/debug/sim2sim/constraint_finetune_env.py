from __future__ import annotations

from typing import Any

import torch

from .constraint_finetune_math import excess_over_budget, validate_constraint_tensor


DEFAULT_WINDOW_STEPS = 50
DEFAULT_BUDGET = 0.181363117489024
DEFAULT_PENALTY_COEF = 1.0


class QddotWindowTracker:
    def __init__(self, num_envs: int, window_steps: int = DEFAULT_WINDOW_STEPS, device: str | torch.device = "cpu"):
        if num_envs <= 0 or window_steps <= 0:
            raise ValueError("num_envs and window_steps must be positive")
        self.num_envs = int(num_envs)
        self.window_steps = int(window_steps)
        self.device = torch.device(device)
        self.buffer = torch.zeros((self.num_envs, self.window_steps, 4), device=self.device)
        self.write_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.sum_squared = torch.zeros(self.num_envs, device=self.device)
        self.rms = torch.zeros(self.num_envs, device=self.device)
        self.excess = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if ids.numel() == 0:
            return
        self.buffer[ids] = 0.0
        self.write_index[ids] = 0
        self.count[ids] = 0
        self.sum_squared[ids] = 0.0
        self.rms[ids] = 0.0
        self.excess[ids] = 0.0

    def update(self, joint_acceleration: torch.Tensor, budget: float = DEFAULT_BUDGET) -> tuple[torch.Tensor, torch.Tensor]:
        if joint_acceleration.ndim != 2 or joint_acceleration.shape != (self.num_envs, 6):
            raise ValueError(f"joint_acceleration must have shape [{self.num_envs},6]")
        validate_constraint_tensor(joint_acceleration, "joint acceleration")
        values = joint_acceleration[:, :4]
        rows = torch.arange(self.num_envs, device=self.device)
        slots = self.write_index
        old = self.buffer[rows, slots]
        full = self.count == self.window_steps
        self.sum_squared = self.sum_squared - torch.where(full, old.square().sum(dim=-1), torch.zeros_like(self.sum_squared))
        self.buffer[rows, slots] = values
        self.sum_squared = self.sum_squared + values.square().sum(dim=-1)
        self.count = torch.minimum(self.count + 1, torch.full_like(self.count, self.window_steps))
        self.write_index = (self.write_index + 1) % self.window_steps
        denominator = self.count.to(dtype=values.dtype).clamp_min(1.0) * values.shape[-1]
        self.rms = (self.sum_squared / denominator).clamp_min(0.0).sqrt()
        self.excess = excess_over_budget(self.rms, budget)
        validate_constraint_tensor(self.rms, "qddot window RMS")
        validate_constraint_tensor(self.excess, "qddot excess")
        return self.rms, self.excess


def make_constrained_wheelleg_env_class():
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv

    class ConstrainedWheelLegEnv(WheelLegFlatEnv):
        def __init__(self, cfg, render_mode: str | None = None, *, qddot_budget: float = DEFAULT_BUDGET,
                     qddot_penalty_coef: float = DEFAULT_PENALTY_COEF, **kwargs: Any):
            self._debug_qddot_budget = float(qddot_budget)
            self._debug_qddot_penalty_coef = float(qddot_penalty_coef)
            super().__init__(cfg, render_mode, **kwargs)
            self._debug_qddot_tracker = QddotWindowTracker(
                self.num_envs, DEFAULT_WINDOW_STEPS, self.device
            )
            self._debug_qddot_reward_before_constraint = torch.zeros(self.num_envs, device=self.device)
            self._debug_qddot_penalty = torch.zeros(self.num_envs, device=self.device)

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            if hasattr(self, "_debug_qddot_tracker"):
                self._debug_qddot_tracker.reset(env_ids)

        def _get_rewards(self) -> torch.Tensor:
            reward_before = super()._get_rewards()
            state = self._current_state()
            rms, excess = self._debug_qddot_tracker.update(
                state.joint_acceleration, budget=self._debug_qddot_budget
            )
            penalty = -self._debug_qddot_penalty_coef * excess
            self._debug_qddot_reward_before_constraint = reward_before.detach().clone()
            self._debug_qddot_penalty = penalty.detach().clone()
            return reward_before + penalty

        @property
        def debug_constraint_metrics(self) -> dict[str, torch.Tensor]:
            tracker = self._debug_qddot_tracker
            return {
                "qddot_window_rms": tracker.rms.detach(),
                "qddot_excess": tracker.excess.detach(),
                "qddot_penalty": self._debug_qddot_penalty.detach(),
                "reward_before_constraint": self._debug_qddot_reward_before_constraint.detach(),
            }

    return ConstrainedWheelLegEnv

