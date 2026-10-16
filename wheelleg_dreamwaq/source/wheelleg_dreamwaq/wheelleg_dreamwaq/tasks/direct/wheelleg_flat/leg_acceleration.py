"""Soft budget for actual four-leg-joint angular acceleration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch

LEG_ACCELERATION_JOINTS = ("jIJ", "jIO", "jAB", "jAG")
DEFAULT_REFERENCE_RMS_RAD_S2 = (
    40.1703144886,
    38.3671852533,
    7.0362190023,
    4.9735461617,
)


@dataclass(frozen=True)
class LegAccelerationConfig:
    enabled: bool = False
    reference_rms_rad_s2: tuple[float, float, float, float] = DEFAULT_REFERENCE_RMS_RAD_S2
    budget_fraction: float = 0.80
    max_penalty_weight: float = 0.02
    rollout_steps: int = 24

    def validate(self, control_dt: float = 0.02) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("acceleration enabled must be bool")
        if len(self.reference_rms_rad_s2) != 4:
            raise ValueError("acceleration reference must contain four joints")
        for value in self.reference_rms_rad_s2:
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("acceleration reference values must be finite and positive")
        for name in ("budget_fraction", "max_penalty_weight", "control_dt"):
            value = control_dt if name == "control_dt" else getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"acceleration {name} must be finite and positive")
        if self.budget_fraction > 2.0:
            raise ValueError("acceleration budget fraction is unreasonably large")
        if type(self.rollout_steps) is not int or self.rollout_steps <= 0:
            raise ValueError("acceleration rollout_steps must be a positive integer")

    def budgets(self) -> tuple[float, float, float, float]:
        self.validate()
        return tuple(value * self.budget_fraction for value in self.reference_rms_rad_s2)


def acceleration_contract_payload(
    cfg: LegAccelerationConfig, control_dt: float = 0.02
) -> dict[str, Any]:
    cfg.validate(control_dt)
    return {
        "schema_version": "LegActualAccelerationBudgetV1",
        **asdict(cfg),
        "reference_rms_rad_s2": list(cfg.reference_rms_rad_s2),
        "joint_names": list(LEG_ACCELERATION_JOINTS),
        "reference_units": "rad/s^2 RMS p90",
        "reference_source": "MuJoCo leg-motion-metric-analysis-20261012-v1",
        "budget_units": "rad/s^2 RMS reference scaled per joint",
        "budget_rule": "budget_j=budget_fraction*reference_rms_j",
        "training_signal": "instantaneous max(a_j^2-budget_j^2,0)/budget_j^2",
        "acceptance_signal": "one_second_window_RMS_p90_in_MuJoCo",
        "control_dt": control_dt,
        "reward": "-control_dt*max_penalty_weight*gain_used*sum(excess_j)",
        "gain_update": "clip(max_j(mean_excess_j)^2,0,1) at rollout boundary",
        "zero_excess_included": True,
        "resume_local_state": "clear_on_environment_reset",
    }


class LegAccelerationReward:
    """Bounded ROGER-style gain for actual leg acceleration excess."""

    def __init__(
        self,
        num_envs: int,
        cfg: LegAccelerationConfig,
        device: str | torch.device,
        control_dt: float = 0.02,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        cfg.validate(control_dt)
        self.cfg = cfg
        self.control_dt = control_dt
        self.budget = torch.tensor(cfg.budgets(), dtype=torch.float32, device=device)
        self.budget_sq = self.budget.square()
        self.cached_reward = torch.zeros(num_envs, device=device)
        self.excess = torch.zeros(num_envs, 4, device=device)
        self.gain = 0.0
        self.gain_used = 0.0
        self.completed_rollouts = 0
        self.step_in_rollout = 0
        self._sum_excess = torch.zeros(4, dtype=torch.float64, device=device)
        self._valid_count = 0
        self._reward_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.latest = {
            "gain_used": 0.0,
            "gain_next": 0.0,
            "valid_windows": 0,
            "mean_excess": [0.0] * 4,
            "weighted_reward_per_step_mean": 0.0,
        }

    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.cached_reward.device)
        if env_ids.ndim != 1:
            raise ValueError("acceleration reset env_ids must be one-dimensional")
        self.cached_reward[env_ids] = 0.0
        self.excess[env_ids] = 0.0

    def step(self, joint_acceleration: torch.Tensor) -> torch.Tensor:
        if (joint_acceleration.ndim != 2 or joint_acceleration.shape[0] != len(self.cached_reward)
                or joint_acceleration.shape[1] < 4):
            raise ValueError("joint acceleration must have shape [num_envs,>=4]")
        values = joint_acceleration[:, :4]
        if not torch.isfinite(values).all():
            raise ValueError("joint acceleration must be finite")
        self.gain_used = self.gain
        self.excess.copy_((values.square() / self.budget_sq - 1.0).clamp_min(0.0))
        self.cached_reward.copy_(
            -self.control_dt
            * self.cfg.max_penalty_weight
            * self.gain_used
            * self.excess.sum(dim=-1)
        )
        self._sum_excess += self.excess.detach().double().sum(dim=0)
        self._valid_count += values.shape[0]
        self._reward_sum += self.cached_reward.detach().double().mean()
        self.step_in_rollout += 1
        if self.step_in_rollout == self.cfg.rollout_steps:
            count = max(self._valid_count, 1)
            mean_excess = (self._sum_excess / count).tolist()
            next_gain = min(max(mean_excess) ** 2, 1.0) if self._valid_count else 0.0
            self.latest = {
                "gain_used": self.gain_used,
                "gain_next": next_gain,
                "valid_windows": self._valid_count,
                "mean_excess": mean_excess,
                "weighted_reward_per_step_mean": float(self._reward_sum.item()) / self.cfg.rollout_steps,
            }
            self.gain = next_gain
            self.completed_rollouts += 1
            self.step_in_rollout = 0
            self._sum_excess.zero_()
            self._valid_count = 0
            self._reward_sum.zero_()
        return self.cached_reward

    def metrics(self) -> dict[str, torch.Tensor | float]:
        result: dict[str, torch.Tensor | float] = {
            "Acceleration/gain_used": self.gain_used,
            "Acceleration/gain_next": self.gain,
            "Acceleration/excess_mean": self.excess.mean(),
        }
        for index, joint in enumerate(LEG_ACCELERATION_JOINTS):
            result[f"Acceleration/excess_{joint}"] = self.excess[:, index].mean()
        return result

    def state_dict(self) -> dict[str, Any]:
        if self.step_in_rollout:
            raise RuntimeError("acceleration gain checkpoint requires a rollout boundary")
        return {
            "schema_version": "LegAccelerationGainStateV1",
            "contract": acceleration_contract_payload(self.cfg, self.control_dt),
            "completed_rollouts": self.completed_rollouts,
            "gain": self.gain,
            "latest": dict(self.latest),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        validate_acceleration_gain_state(
            payload, acceleration_contract_payload(self.cfg, self.control_dt)
        )
        if self.step_in_rollout:
            raise RuntimeError("cannot restore acceleration gain during a rollout")
        self.gain = float(payload["gain"])
        self.gain_used = self.gain
        self.completed_rollouts = int(payload["completed_rollouts"])
        self.latest = dict(payload["latest"])


def validate_acceleration_gain_state(
    payload: Any,
    expected_contract: dict[str, Any],
    completed_rollouts: int | None = None,
) -> None:
    fields = {"schema_version", "contract", "completed_rollouts", "gain", "latest"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("acceleration gain state fields are invalid")
    if payload["schema_version"] != "LegAccelerationGainStateV1" or payload["contract"] != expected_contract:
        raise ValueError("acceleration gain state contract mismatch")
    count = payload["completed_rollouts"]
    if type(count) is not int or count < 0 or (completed_rollouts is not None and count != completed_rollouts):
        raise ValueError("acceleration completed_rollouts mismatch")
    gain = payload["gain"]
    if type(gain) not in (int, float) or not math.isfinite(gain) or not 0 <= gain <= 1:
        raise ValueError("acceleration gain must be finite in [0,1]")
    latest = payload["latest"]
    required = {"gain_used", "gain_next", "valid_windows", "mean_excess", "weighted_reward_per_step_mean"}
    if not isinstance(latest, dict) or set(latest) != required:
        raise ValueError("acceleration latest summary fields invalid")
    if type(latest["valid_windows"]) is not int or latest["valid_windows"] < 0:
        raise ValueError("acceleration valid_windows invalid")
    values = latest["mean_excess"]
    if not isinstance(values, list) or len(values) != 4 or any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values
    ):
        raise ValueError("acceleration mean_excess invalid")
    for name in ("gain_used", "gain_next"):
        value = latest[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"acceleration {name} invalid")
    reward = latest["weighted_reward_per_step_mean"]
    if type(reward) not in (int, float) or not math.isfinite(reward) or reward > 0:
        raise ValueError("acceleration weighted reward invalid")
    if latest["gain_next"] != gain:
        raise ValueError("acceleration next gain differs from stored gain")
    expected_gain = min(max(values) ** 2, 1.0) if latest["valid_windows"] else 0.0
    if not math.isclose(float(gain), expected_gain, rel_tol=0, abs_tol=1e-12):
        raise ValueError("acceleration gain differs from rollout summary rule")

