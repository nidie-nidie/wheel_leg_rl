from __future__ import annotations

from dataclasses import asdict
import math

import torch

from .contracts import GuidanceConfigV1


def upward_force(height: torch.Tensor, alpha: torch.Tensor, maximum_force: torch.Tensor,
                 *, target_height: float, mu: float) -> torch.Tensor:
    deficit = (target_height - height).clamp_min(0)
    return -torch.expm1(-mu * deficit) * alpha * maximum_force


def raw_height_diagnostic(height: torch.Tensor, alpha: torch.Tensor, *, target_height: float,
                          mu: float) -> dict[str, torch.Tensor]:
    exponent = mu * (height.to(torch.float64) - target_height)
    safe = exponent <= math.log(torch.finfo(torch.float64).max) - 1.0
    raw = -torch.expm1(torch.where(safe, exponent, torch.zeros_like(exponent))) * alpha.to(torch.float64)
    return {"raw_coefficient": raw, "raw_overflow": ~safe, "exponent": exponent,
            "negative_coefficient": (height > target_height) & (alpha > 0)}


class RecoveryForceGuidance:
    def __init__(self, config: dict, masses_kg: torch.Tensor, *, gravity_m_s2: float = 9.81):
        self.config = GuidanceConfigV1.from_dict(config)
        if (masses_kg.ndim != 1 or not torch.isfinite(masses_kg).all()
                or (masses_kg <= 0).any() or not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0):
            raise ValueError("辅助力要求有效整机质量和重力加速度")
        self.mass_kg = masses_kg.detach().clone()
        self.max_force_n = self.mass_kg * gravity_m_s2 * self.config.max_force_body_weights
        self.global_control_steps = 0
        self.pending = False
        self._substeps = 0
        self._alpha = torch.ones_like(self.mass_kg)
        self._alpha_t = self._alpha.clone()
        self._sum_force = torch.zeros_like(self.mass_kg)
        self._peak_force = self._sum_force.clone()
        self.force_world = self.mass_kg.new_zeros(len(self.mass_kg), 1, 3)
        self.zero_torque_world = torch.zeros_like(self.force_world)
        self.last_transition: dict | None = None
        self._negative_raw_count = self.mass_kg.new_zeros((), dtype=torch.int64)
        self._raw_overflow_count = self._negative_raw_count.clone()
        self.episode_discounted_cost = torch.zeros_like(self.mass_kg)
        self.episode_steps = self.mass_kg.new_zeros(len(self.mass_kg), dtype=torch.int64)
        self.last_episode_cost = torch.zeros_like(self.mass_kg)
        self.completed_episode_count = self.mass_kg.new_zeros((), dtype=torch.int64)
        self.completed_episode_cost_sum = self.mass_kg.new_zeros(())
        self._last_raw_diagnostic = None
        self._refresh_alpha()

    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha

    def _refresh_alpha(self) -> None:
        alpha = max(1.0 - self.global_control_steps / self.config.end_control_steps, 0.0)
        self._alpha.fill_(alpha if self.config.force_enabled else 0.0)

    def begin_transition(self) -> None:
        if self.pending:
            raise RuntimeError("上一次辅助转移尚未完成")
        self.pending = True
        self._substeps = 0
        self._alpha_t.copy_(self._alpha)
        self._sum_force.zero_()
        self._peak_force.zero_()

    def substep(self, height: torch.Tensor) -> torch.Tensor:
        if not self.pending or self._substeps >= 4 or height.shape != self.mass_kg.shape:
            raise RuntimeError("辅助物理子步顺序或高度形状无效")
        force = upward_force(height, self._alpha_t, self.max_force_n,
                             target_height=self.config.h_target, mu=self.config.mu)
        torch._assert_async(torch.isfinite(force).all(), "辅助力包含非有限数值")
        self.force_world.zero_()
        self.force_world[:, 0, 2].copy_(force)
        self._sum_force.add_(force)
        self._peak_force.copy_(torch.maximum(self._peak_force, force))
        self._negative_raw_count.add_(((height > self.config.h_target) & (self._alpha_t > 0)).sum())
        if self.global_control_steps % self.config.diagnostic_interval_steps == 0:
            diagnostic = raw_height_diagnostic(height, self._alpha_t,
                target_height=self.config.h_target, mu=self.config.mu)
            self._raw_overflow_count.add_(diagnostic["raw_overflow"].sum())
            if self._substeps == 0:
                self._last_raw_diagnostic = {
                    "control_step": self.global_control_steps,
                    "height_m": height[:8].clone(),
                    **{key: value[:8].clone() for key, value in diagnostic.items()},
                }
        self._substeps += 1
        return self.force_world

    def finish_transition(self) -> dict:
        if not self.pending or self._substeps != 4:
            raise RuntimeError("辅助转移要求恰好四个完整物理子步")
        cost = self._sum_force / 4.0
        self.episode_discounted_cost.add_(self.config.gamma ** self.episode_steps * cost)
        self.episode_steps.add_(1)
        self.global_control_steps += 1
        self._refresh_alpha()
        self.pending = False
        self.last_transition = {
            "schema_version": "RecoveryGuidanceTransitionV1", "cost_force_n": cost.clone(),
            "peak_force_n": self._peak_force.clone(), "alpha_t": self._alpha_t.clone(),
            "alpha_next": self._alpha.clone(), "global_control_steps": self.global_control_steps,
            "max_force_n": self.max_force_n.clone(),
        }
        return self.last_transition

    def reset(self, env_ids: torch.Tensor, *, completed: torch.Tensor | None = None) -> None:
        if self.pending:
            raise RuntimeError("辅助转移冻结之前不允许重置环境")
        self.last_episode_cost[env_ids] = self.episode_discounted_cost[env_ids]
        completed = self.episode_steps[env_ids] > 0 if completed is None else completed.bool() & (self.episode_steps[env_ids] > 0)
        self.completed_episode_count.add_(completed.sum())
        self.completed_episode_cost_sum.add_(torch.where(completed, self.episode_discounted_cost[env_ids], 0).sum())
        self.episode_discounted_cost[env_ids] = 0
        self.episode_steps[env_ids] = 0
        self.force_world[env_ids] = 0

    def restore_progress(self, global_control_steps: int) -> None:
        if (self.pending or isinstance(global_control_steps, bool) or not isinstance(global_control_steps, int)
                or global_control_steps < 0):
            raise ValueError("辅助恢复进度必须为非负完整控制步")
        self.global_control_steps = global_control_steps
        self._refresh_alpha()
        self.last_transition = None
        self.force_world.zero_()

    def identity(self, *, body_name: str, body_index: int) -> dict:
        return {"schema_version": "RecoveryGuidanceExecutionV1", "config": asdict(self.config),
                "body_name": body_name, "body_index": body_index, "height_reference": "base_link_COM_above_flat_ground",
                "force_frame": "world", "application_point": "base_link_COM",
                "application_api": "instantaneous_wrench_composer.set_forces_and_torques_positions_None",
                "mass_kg": self.mass_kg.cpu().tolist(), "max_force_n": self.max_force_n.cpu().tolist()}

    def statistics(self) -> dict:
        episode_count = int(self.completed_episode_count)
        diagnostic = None
        if self._last_raw_diagnostic is not None:
            saved = self._last_raw_diagnostic
            heights = saved["height_m"].cpu().tolist()
            raw = saved["raw_coefficient"].cpu().tolist()
            overflow = saved["raw_overflow"].cpu().tolist()
            exponents = saved["exponent"].cpu().tolist()
            negative = saved["negative_coefficient"].cpu().tolist()
            diagnostic = {"control_step": saved["control_step"], "samples": [
                {"env_id": index, "height_m": height, "raw_coefficient": None if overflow[index] else raw[index],
                 "raw_overflow": overflow[index], "exponent": exponents[index],
                 "negative_coefficient": negative[index]}
                for index, height in enumerate(heights)]}
        return {"global_control_steps": self.global_control_steps, "alpha": float(self.alpha[0]),
                "negative_raw_substep_count": int(self._negative_raw_count),
                "raw_overflow_substep_count": int(self._raw_overflow_count),
                "completed_episode_count": episode_count,
                "mean_completed_episode_discounted_cost_n": (None if not episode_count else
                    float(self.completed_episode_cost_sum) / episode_count),
                "raw_formula_diagnostic": diagnostic,
                "episode_cost_scope": "descriptive_mixed_policy_and_changing_alpha"}
