"""Training-only leg target reversal cost; no controller or observation changes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class LegReversalConfig:
    enabled: bool = False
    amplitude_rad: float = 0.01
    window_seconds: float = 1.0
    budget_per_second: float = 12.0
    max_penalty_weight: float = 0.02
    rollout_steps: int = 24

    def validate(self, control_dt: float = 0.02) -> int:
        if type(self.enabled) is not bool:
            raise ValueError("reversal enabled must be bool")
        for name in ("amplitude_rad", "window_seconds", "budget_per_second", "max_penalty_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"reversal {name} must be finite and positive")
        if type(self.rollout_steps) is not int or self.rollout_steps <= 0:
            raise ValueError("reversal rollout_steps must be a positive integer")
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("reversal control_dt must be positive")
        steps = round(self.window_seconds / control_dt)
        if steps < 1 or not math.isclose(steps * control_dt, self.window_seconds, abs_tol=1e-9):
            raise ValueError("reversal window must contain an integer number of control steps")
        return steps


def reversal_contract_payload(cfg: LegReversalConfig, control_dt: float = 0.02) -> dict[str, Any]:
    steps = cfg.validate(control_dt)
    return {
        "schema_version": "LegTargetReversalV1",
        **asdict(cfg),
        "signal": "final_leg_position_targets_after_clip_scale_and_limits",
        "event_rule": "opposite_excursion_from_running_extremum_greater_equal_amplitude",
        "first_direction_is_event": False,
        "window_steps": steps,
        "control_dt": control_dt,
        "warmup": "exclude_until_full_window",
        "cost": "sum_j(max(rate_j/budget_per_second-1,0))",
        "gain_update": "clip(max_j(mean_valid_excess_j)^2,0,1)",
        "zero_excess_included": True,
        "reward": "-control_dt*max_penalty_weight*gain_used*cost",
        "resume_local_window": "clear_for_new_environment_and_warm_up",
    }


class LegTargetReversalCounter:
    """Counts hysteretic target excursions independently for each environment/joint."""

    def __init__(self, num_envs: int, cfg: LegReversalConfig, device: str | torch.device, control_dt: float = 0.02):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.cfg = cfg
        self.window_steps = cfg.validate(control_dt)
        self.extreme = torch.zeros(num_envs, 4, device=device)
        self.direction = torch.zeros(num_envs, 4, dtype=torch.int8, device=device)
        self.events = torch.zeros(self.window_steps, num_envs, 4, dtype=torch.bool, device=device)
        self.count = torch.zeros(num_envs, 4, dtype=torch.long, device=device)
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.rate = torch.zeros(num_envs, 4, device=device)
        self.cursor = 0

    @property
    def eligible(self) -> torch.Tensor:
        return self.age >= self.window_steps

    def reset(self, env_ids: torch.Tensor, targets: torch.Tensor) -> None:
        if targets.shape != (env_ids.numel(), 4) or not torch.isfinite(targets).all():
            raise ValueError("reset targets must be finite [selected_envs,4]")
        self.extreme[env_ids] = targets
        self.direction[env_ids] = 0
        self.events[:, env_ids] = False
        self.count[env_ids] = 0
        self.age[env_ids] = 0
        self.rate[env_ids] = 0

    def update(self, targets: torch.Tensor) -> torch.Tensor:
        if targets.shape != self.extreme.shape or not torch.isfinite(targets).all():
            raise ValueError("leg targets must be finite [num_envs,4]")
        targets = targets.detach()
        direction = self.direction
        extreme = torch.where(direction > 0, torch.maximum(self.extreme, targets),
                              torch.where(direction < 0, torch.minimum(self.extreme, targets), self.extreme))
        epsilon = self.cfg.amplitude_rad
        reverse = ((direction > 0) & (targets <= extreme - epsilon)) | (
            (direction < 0) & (targets >= extreme + epsilon))
        establish_up = (direction == 0) & (targets >= extreme + epsilon)
        establish_down = (direction == 0) & (targets <= extreme - epsilon)
        next_direction = torch.where(reverse, -direction, torch.where(
            establish_up, torch.ones_like(direction),
            torch.where(establish_down, -torch.ones_like(direction), direction)))
        self.extreme.copy_(torch.where(reverse | establish_up | establish_down, targets, extreme))
        self.direction.copy_(next_direction)
        self.count -= self.events[self.cursor].long()
        self.events[self.cursor].copy_(reverse)
        self.count += reverse.long()
        self.cursor = (self.cursor + 1) % self.window_steps
        self.age.add_(1).clamp_(max=self.window_steps)
        self.rate.copy_(self.count / self.cfg.window_seconds)
        return reverse


class LegReversalReward:
    """Bounded A1-inspired ROGER-style gain, held fixed for each rollout."""

    def __init__(self, num_envs: int, cfg: LegReversalConfig, device: str | torch.device, control_dt: float = 0.02):
        self.cfg = cfg
        self.control_dt = control_dt
        self.counter = LegTargetReversalCounter(num_envs, cfg, device, control_dt)
        self.cached_reward = torch.zeros(num_envs, device=device)
        self.excess = torch.zeros(num_envs, 4, device=device)
        self.gain = 0.0
        self.gain_used = 0.0
        self.completed_rollouts = 0
        self.step_in_rollout = 0
        self._sum_rate = torch.zeros(4, dtype=torch.float64, device=device)
        self._sum_excess = torch.zeros_like(self._sum_rate)
        self._valid_count = torch.zeros((), dtype=torch.long, device=device)
        self._reward_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.latest = {
            "gain_used": 0.0, "gain_next": 0.0, "valid_windows": 0,
            "mean_rate": [0.0] * 4, "mean_excess": [0.0] * 4,
            "weighted_reward_per_step_mean": 0.0,
        }

    def reset(self, env_ids: torch.Tensor, targets: torch.Tensor) -> None:
        self.counter.reset(env_ids, targets)
        self.cached_reward[env_ids] = 0
        self.excess[env_ids] = 0

    def step(self, targets: torch.Tensor) -> torch.Tensor:
        self.counter.update(targets)
        eligible = self.counter.eligible
        self.excess.copy_(torch.where(
            eligible[:, None],
            (self.counter.rate / self.cfg.budget_per_second - 1.0).clamp_min(0.0),
            torch.zeros_like(self.excess)))
        self.gain_used = self.gain
        self.cached_reward.copy_(-self.control_dt * self.cfg.max_penalty_weight *
                                 self.gain_used * self.excess.sum(dim=-1))
        self._sum_rate += torch.where(eligible[:, None], self.counter.rate, 0.0).double().sum(dim=0)
        self._sum_excess += self.excess.double().sum(dim=0)
        self._valid_count += eligible.sum()
        self._reward_sum += self.cached_reward.double().mean()
        self.step_in_rollout += 1
        if self.step_in_rollout == self.cfg.rollout_steps:
            count = int(self._valid_count.item())
            rates = (self._sum_rate / max(count, 1)).tolist()
            excess = (self._sum_excess / max(count, 1)).tolist()
            next_gain = min(max(excess) ** 2, 1.0) if count else 0.0
            self.latest = {
                "gain_used": self.gain_used, "gain_next": next_gain, "valid_windows": count,
                "mean_rate": rates, "mean_excess": excess,
                "weighted_reward_per_step_mean": float(self._reward_sum.item()) / self.cfg.rollout_steps,
            }
            self.gain = next_gain
            self.completed_rollouts += 1
            self.step_in_rollout = 0
            self._sum_rate.zero_()
            self._sum_excess.zero_()
            self._valid_count.zero_()
            self._reward_sum.zero_()
        return self.cached_reward

    def metrics(self) -> dict[str, torch.Tensor | float]:
        result: dict[str, torch.Tensor | float] = {
            "Reversal/gain_used": self.gain_used,
            "Reversal/gain_next": self.gain,
            "Reversal/eligible_fraction": self.counter.eligible.float().mean(),
        }
        for joint in range(4):
            result[f"Reversal/target_rate_{joint}_per_s"] = self.counter.rate[:, joint].mean()
            result[f"Reversal/excess_{joint}"] = self.excess[:, joint].mean()
        return result

    def state_dict(self) -> dict[str, Any]:
        if self.step_in_rollout:
            raise RuntimeError("reversal gain checkpoint requires a rollout boundary")
        return {
            "schema_version": "LegReversalGainStateV1",
            "contract": reversal_contract_payload(self.cfg, self.control_dt),
            "completed_rollouts": self.completed_rollouts,
            "gain": self.gain,
            "latest": dict(self.latest),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        validate_gain_state(payload, reversal_contract_payload(self.cfg, self.control_dt))
        if self.step_in_rollout:
            raise RuntimeError("cannot restore reversal gain during a rollout")
        self.gain = float(payload["gain"])
        self.gain_used = self.gain
        self.completed_rollouts = payload["completed_rollouts"]
        self.latest = dict(payload["latest"])


def validate_gain_state(payload: Any, expected_contract: dict[str, Any], completed_rollouts: int | None = None) -> None:
    fields = {"schema_version", "contract", "completed_rollouts", "gain", "latest"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("reversal gain state fields are invalid")
    if payload["schema_version"] != "LegReversalGainStateV1" or payload["contract"] != expected_contract:
        raise ValueError("reversal gain state contract mismatch")
    count = payload["completed_rollouts"]
    if type(count) is not int or count < 0 or (completed_rollouts is not None and count != completed_rollouts):
        raise ValueError("reversal completed_rollouts mismatch")
    gain = payload["gain"]
    if type(gain) not in (int, float) or not math.isfinite(gain) or not 0 <= gain <= 1:
        raise ValueError("reversal gain must be finite in [0,1]")
    latest = payload["latest"]
    fields = {"gain_used", "gain_next", "valid_windows", "mean_rate", "mean_excess", "weighted_reward_per_step_mean"}
    if not isinstance(latest, dict) or set(latest) != fields:
        raise ValueError("reversal latest summary fields invalid")
    if type(latest["valid_windows"]) is not int or latest["valid_windows"] < 0:
        raise ValueError("reversal valid_windows invalid")
    for name in ("mean_rate", "mean_excess"):
        values = latest[name]
        if not isinstance(values, list) or len(values) != 4 or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"reversal {name} invalid")
    for name in ("gain_used", "gain_next"):
        v = latest[name]
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f"reversal {name} invalid")
    v = latest["weighted_reward_per_step_mean"]
    if type(v) not in (int, float) or not math.isfinite(v) or v > 0:
        raise ValueError("reversal weighted reward invalid")
    if latest["gain_next"] != gain:
        raise ValueError("reversal next gain differs from stored gain")

    expected_gain = min(max(latest["mean_excess"]) ** 2, 1.0) if latest["valid_windows"] else 0.0
    if not math.isclose(float(gain), expected_gain, rel_tol=0, abs_tol=1e-12):
        raise ValueError("reversal gain differs from rollout summary rule")
    if not latest["valid_windows"] and (any(latest["mean_rate"]) or any(latest["mean_excess"]) or v):
        raise ValueError("reversal empty-window summary must be zero")
