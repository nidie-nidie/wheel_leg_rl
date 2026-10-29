from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class BoundedPolicySample(NamedTuple):
    latent: torch.Tensor
    action: torch.Tensor
    normal_log_prob: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


def tanh_log_jacobian(latent: torch.Tensor) -> torch.Tensor:
    return 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))


def latent_normal_kl(old_mean, old_std, mean, std):
    return (torch.log(std / old_std) + (old_std.square() + (old_mean - mean).square())
            / (2.0 * std.square()) - 0.5).sum(dim=-1)


def _mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    network = nn.Sequential(nn.Linear(input_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, output_dim))
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
    return network


class DeterministicStandingActor(nn.Module):
    def __init__(self, actor: nn.Sequential):
        super().__init__()
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(observation))


class BoundedActorCritic(nn.Module):
    def __init__(self, reference_q_median):
        super().__init__()
        reference = torch.as_tensor(reference_q_median, dtype=torch.float32).reshape(-1)
        if reference.shape != (4,) or not torch.isfinite(reference).all() or (reference.abs() >= 1).any():
            raise ValueError("参考腿角度必须包含四个有限值，并严格处于动作范围内")
        self.actor = _mlp(110, 6)
        self.critic = _mlp(41, 1)
        nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad():
            self.actor[-1].bias[:4].copy_(torch.atanh(reference))
            self.actor[-1].bias[4:].zero_()
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        self.register_buffer("std_min", torch.tensor([0.05] * 4 + [0.03] * 2))
        self.register_buffer("std_max", torch.tensor([0.50] * 4 + [0.25] * 2))
        initial = torch.tensor([0.15] * 4 + [0.10] * 2)
        fraction = (initial - self.std_min) / (self.std_max - self.std_min)
        self.raw_std = nn.Parameter(torch.logit(fraction))

    @property
    def std(self):
        return self.std_min + (self.std_max - self.std_min) * torch.sigmoid(self.raw_std)

    def distribution(self, actor_observation):
        if actor_observation.shape[-1] != 110 or not torch.isfinite(actor_observation).all():
            raise ValueError("Actor 观测必须为有限的 110 维输入")
        return Normal(self.actor(actor_observation), self.std)

    def value(self, critic_observation):
        if critic_observation.shape[-1] != 41 or not torch.isfinite(critic_observation).all():
            raise ValueError("Critic 观测必须为有限的 41 维输入")
        return self.critic(critic_observation).squeeze(-1)

    def sample(self, actor_observation, critic_observation):
        distribution = self.distribution(actor_observation)
        latent = distribution.sample()
        normal = distribution.log_prob(latent).sum(-1)
        log_prob = normal - tanh_log_jacobian(latent).sum(-1)
        return BoundedPolicySample(latent, torch.tanh(latent), normal, log_prob,
                                   self.value(critic_observation), distribution.mean,
                                   distribution.stddev)

    def evaluate(self, actor_observation, critic_observation, latent):
        distribution = self.distribution(actor_observation)
        if latent.shape != distribution.mean.shape or not torch.isfinite(latent).all():
            raise ValueError("PPO 必须使用 storage 保存的有限潜在动作")
        normal = distribution.log_prob(latent).sum(-1)
        entropy_latent = distribution.rsample()
        entropy = -(distribution.log_prob(entropy_latent) - tanh_log_jacobian(entropy_latent)).sum(-1)
        return {"normal_log_prob": normal, "log_prob": normal - tanh_log_jacobian(latent).sum(-1),
                "entropy": entropy, "value": self.value(critic_observation),
                "mean": distribution.mean, "std": distribution.stddev}

    def deterministic(self, actor_observation):
        return torch.tanh(self.distribution(actor_observation).mean)
