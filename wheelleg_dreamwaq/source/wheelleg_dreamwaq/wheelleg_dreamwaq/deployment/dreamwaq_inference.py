from __future__ import annotations

import copy

import torch
import torch.nn as nn

from wheelleg_dreamwaq.algorithms.dreamwaq.actor_critic import DreamWaQActorCritic
from wheelleg_dreamwaq.schemas.observation import ACTOR_OBS_DIM, HISTORY_OBS_DIM


class DreamWaQInferenceActorV1(nn.Module):
    """Stateless canonical ``history[N,125] -> action_mean[N,6]`` export graph."""

    def __init__(self, encoder: nn.Module, actor: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.history_dim = HISTORY_OBS_DIM
        self.current_obs_dim = ACTOR_OBS_DIM

    @classmethod
    def from_policy(cls, policy: DreamWaQActorCritic) -> "DreamWaQInferenceActorV1":
        return cls(copy.deepcopy(policy.cenet.encoder), copy.deepcopy(policy.actor))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        torch._assert(history.dim() == 2, "history must be rank 2")
        torch._assert(history.size(1) == self.history_dim, "history dimension must be 125")
        torch._assert(history.dtype == torch.float32, "history must be float32")
        encoded = self.encoder(history)
        estimated_velocity = encoded[:, 0:3]
        context_mu = encoded[:, 3:19]
        current_observation = history[:, -self.current_obs_dim:]
        raw_mean = self.actor(torch.cat((current_observation, estimated_velocity, context_mu), dim=-1))
        torch._assert(torch.isfinite(raw_mean).all(), "raw action mean must be finite")
        return raw_mean.clamp(-20.0, 20.0)
