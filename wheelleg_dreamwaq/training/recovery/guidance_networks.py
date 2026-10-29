from __future__ import annotations

import torch
from torch import nn


class GuidanceCostCritic(nn.Module):
    def __init__(self, observation_dim: int = 35):
        super().__init__()
        if observation_dim != 35:
            raise ValueError("辅助成本 critic 输入必须为 35 维")
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, critic_observation: torch.Tensor) -> torch.Tensor:
        if critic_observation.shape[-1] != 35:
            raise ValueError("辅助成本 critic 输入必须为 35 维")
        estimate = self.network(critic_observation)
        # 撤除后的成本价值精确为零，训练进度边界以前的样本继续保留。
        return torch.where(critic_observation[..., -1:] > 0, estimate, torch.zeros_like(estimate))
