from __future__ import annotations

import torch
from rsl_rl.storage import RolloutStorage


class GuidanceRolloutStorage(RolloutStorage):
    def __init__(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape, device="cpu"):
        if training_type != "rl":
            raise ValueError("辅助引导仅支持普通强化学习 rollout")
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        shape = (num_transitions_per_env, num_envs, 1)
        for name in ("cost_force_n", "cost_values", "cost_returns", "cost_advantages_scaled",
                     "cost_advantages_physical", "alpha_t", "alpha_next", "reward_advantages",
                     "mixed_advantages", "penalty_advantages"):
            setattr(self, name, torch.zeros(shape, device=device))

    def add_guidance_transition(self, transition, *, cost_force_n, cost_values, alpha_t, alpha_next):
        index = self.step
        if index >= self.num_transitions_per_env:
            raise OverflowError("辅助 rollout 已满")
        super().add_transitions(transition)
        for name, value in (("cost_force_n", cost_force_n), ("cost_values", cost_values),
                            ("alpha_t", alpha_t), ("alpha_next", alpha_next)):
            getattr(self, name)[index].copy_(value.reshape(self.num_envs, 1))

    @torch.no_grad()
    def compute_guidance_returns(self, last_values, last_cost_values, gamma, lam, cost_scale_n, beta_force):
        if self.step != self.num_transitions_per_env:
            raise ValueError("辅助优势计算要求完整 rollout")
        if not 0 < gamma < 1 or not 0 <= lam <= 1 or cost_scale_n <= 0 or beta_force < 0:
            raise ValueError("辅助优势参数超出有效范围")
        super().compute_returns(last_values, gamma, lam, normalize_advantage=False)
        self.reward_advantages.copy_(self.advantages)
        advantage = torch.zeros_like(last_cost_values)
        for index in reversed(range(self.num_transitions_per_env)):
            next_value = last_cost_values if index == self.num_transitions_per_env - 1 else self.cost_values[index + 1]
            next_value = torch.where(self.alpha_next[index] > 0, next_value, torch.zeros_like(next_value))
            alive = 1.0 - self.dones[index].float()
            delta = self.cost_force_n[index] / cost_scale_n + gamma * alive * next_value - self.cost_values[index]
            advantage = delta + gamma * lam * alive * advantage
            advantage = torch.where(self.alpha_t[index] > 0, advantage, torch.zeros_like(advantage))
            self.cost_advantages_scaled[index].copy_(advantage)
            self.cost_returns[index].copy_(advantage + self.cost_values[index])
        self.cost_advantages_physical.copy_(self.cost_advantages_scaled * cost_scale_n)
        self.penalty_advantages.copy_(beta_force / (1.0 - gamma) * self.cost_advantages_physical)
        self.mixed_advantages.copy_(self.reward_advantages - self.penalty_advantages)
        self.advantages.copy_(
            (self.mixed_advantages - self.mixed_advantages.mean())
            / (self.mixed_advantages.std(unbiased=False) + 1e-8)
        )

    def guidance_mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
        if self.step != self.num_transitions_per_env:
            raise ValueError("辅助 mini batch 要求完整 rollout")
        batch_size = self.num_envs * self.num_transitions_per_env
        if num_mini_batches <= 0 or num_epochs <= 0 or batch_size % num_mini_batches:
            raise ValueError("辅助 rollout 必须能完整划分 mini batch")
        mini_batch_size = batch_size // num_mini_batches
        observations = self.observations.flatten(0, 1)
        fields = {name: getattr(self, name).flatten(0, 1) for name in (
            "actions", "values", "returns", "actions_log_prob", "advantages", "mu", "sigma",
            "cost_values", "cost_returns", "cost_advantages_physical", "reward_advantages",
            "cost_force_n", "alpha_t", "alpha_next",
        )}
        # 奖励、策略和成本使用同一次索引，不分别打乱顺序。
        indices = torch.randperm(batch_size, device=self.device)
        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                selected = indices[index * mini_batch_size:(index + 1) * mini_batch_size]
                yield observations[selected], {name: value[selected] for name, value in fields.items()}
