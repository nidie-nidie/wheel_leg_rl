from __future__ import annotations

from typing import Any, NamedTuple, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from wheelleg_dreamwaq.schemas.observation import (
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    HISTORY_OBS_DIM,
    PROPRIO_RECONSTRUCTION_TARGET_DIM,
    VELOCITY_TARGET_DIM,
    extract_velocity_target,
)

from .cenet import CENet, CENetEncoding, CONTEXT_DIM


EXPECTED_OBS_GROUPS = {"policy": ["policy"], "critic": ["critic"]}
EXPECTED_OBSERVATION_KEYS = {"policy", "policy_history", "critic"}


class DreamWaQBatchOutput(NamedTuple):
    action_mean: torch.Tensor
    action_sigma: torch.Tensor
    action_log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    estimated_velocity: torch.Tensor
    context_mu: torch.Tensor
    context_logvar: torch.Tensor
    predicted_next_proprio: torch.Tensor


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


class DreamWaQActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 128, 64),
        critic_hidden_dims: tuple[int, ...] | list[int] = (256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        current_obs_group: str = "policy",
        history_obs_group: str = "policy_history",
        history_length: int = 5,
        current_obs_dim: int = ACTOR_OBS_DIM,
        history_obs_dim: int = HISTORY_OBS_DIM,
        critic_obs_dim: int = CRITIC_OBS_DIM,
        velocity_dim: int = VELOCITY_TARGET_DIM,
        context_dim: int = CONTEXT_DIM,
        reconstruction_target_dim: int = PROPRIO_RECONSTRUCTION_TARGET_DIM,
        cenet_encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        cenet_decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 128),
        actor_context_mode: str = "deterministic_context_mu",
        action_mean_clip: float = 20.0,
        action_std_min: float = 1.0e-4,
        action_std_max: float = 2.0,
        **unknown: Any,
    ) -> None:
        super().__init__()
        if unknown:
            raise TypeError(f"Unexpected DreamWaQActorCritic arguments: {sorted(unknown)}")
        if obs_groups != EXPECTED_OBS_GROUPS:
            raise ValueError(f"obs_groups must equal {EXPECTED_OBS_GROUPS}, got {obs_groups}")
        if set(obs.keys()) != EXPECTED_OBSERVATION_KEYS:
            raise ValueError(f"observation keys must equal {sorted(EXPECTED_OBSERVATION_KEYS)}, got {sorted(obs.keys())}")
        expected_shapes = {
            current_obs_group: current_obs_dim,
            history_obs_group: history_obs_dim,
            "critic": critic_obs_dim,
        }
        for key, width in expected_shapes.items():
            if key not in obs or obs[key].ndim != 2 or obs[key].shape[-1] != width:
                actual = None if key not in obs else tuple(obs[key].shape)
                raise ValueError(f"observation {key!r} must have shape [N,{width}], got {actual}")
        fixed_values = {
            "num_actions": (num_actions, 6),
            "history_length": (history_length, 5),
            "current_obs_dim": (current_obs_dim, ACTOR_OBS_DIM),
            "history_obs_dim": (history_obs_dim, HISTORY_OBS_DIM),
            "critic_obs_dim": (critic_obs_dim, CRITIC_OBS_DIM),
            "velocity_dim": (velocity_dim, VELOCITY_TARGET_DIM),
            "context_dim": (context_dim, CONTEXT_DIM),
            "reconstruction_target_dim": (
                reconstruction_target_dim,
                PROPRIO_RECONSTRUCTION_TARGET_DIM,
            ),
        }
        for name, (actual, expected) in fixed_values.items():
            if actual != expected:
                raise ValueError(f"{name} must be {expected}, got {actual}")
        if tuple(actor_hidden_dims) != (256, 128, 64) or tuple(critic_hidden_dims) != (256, 128, 64):
            raise ValueError("Actor and critic hidden dimensions must both be [256,128,64]")
        if activation.lower() != "elu":
            raise ValueError("DreamWaQ activation must be ELU")
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("DreamWaQ empirical Actor/Critic normalization must be disabled")
        if noise_std_type != "scalar" or state_dependent_std:
            raise ValueError("DreamWaQ requires scalar state-independent action standard deviation")
        if actor_context_mode != "deterministic_context_mu":
            raise ValueError("Actor must use deterministic context_mu")
        if not (action_mean_clip == 20.0 and action_std_min == 1.0e-4 and action_std_max == 2.0):
            raise ValueError("DreamWaQ action distribution bounds differ from the frozen contract")
        if init_noise_std != 1.0:
            raise ValueError("DreamWaQ init_noise_std must be 1.0")

        self.obs_groups = {key: list(value) for key, value in obs_groups.items()}
        self.current_obs_group = current_obs_group
        self.history_obs_group = history_obs_group
        self.current_obs_dim = current_obs_dim
        self.history_obs_dim = history_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.num_actions = num_actions
        self.action_mean_clip = action_mean_clip
        self.action_std_min = action_std_min
        self.action_std_max = action_std_max
        self.actor_obs_normalization = False
        self.critic_obs_normalization = False
        self.state_dependent_std = False
        self.noise_std_type = noise_std_type

        self.cenet = CENet(
            history_dim=history_obs_dim,
            velocity_dim=velocity_dim,
            context_dim=context_dim,
            reconstruction_target_dim=reconstruction_target_dim,
            encoder_hidden_dims=cenet_encoder_hidden_dims,
            decoder_hidden_dims=cenet_decoder_hidden_dims,
        )
        self.actor = _mlp(current_obs_dim + velocity_dim + context_dim, tuple(actor_hidden_dims), num_actions)
        self.critic = _mlp(critic_obs_dim, tuple(critic_hidden_dims), 1)
        self.raw_std = nn.Parameter(torch.full((num_actions,), float(init_noise_std)))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def get_hidden_states(self) -> tuple[None, None]:
        return None, None

    def _encoding(self, obs: TensorDict) -> CENetEncoding:
        return self.cenet.encode(obs[self.history_obs_group])

    def _effective_mean(self, current_obs: torch.Tensor, encoding: CENetEncoding) -> torch.Tensor:
        actor_input = torch.cat((current_obs, encoding.estimated_velocity, encoding.context_mu), dim=-1)
        raw_mean = self.actor(actor_input)
        _require_finite("raw action mean", raw_mean)
        effective_mean = raw_mean.clamp(-self.action_mean_clip, self.action_mean_clip)
        _require_finite("effective action mean", effective_mean)
        return effective_mean

    def _effective_std(self, mean: torch.Tensor) -> torch.Tensor:
        _require_finite("raw action standard deviation", self.raw_std)
        effective = self.raw_std.abs().clamp(self.action_std_min, self.action_std_max).expand_as(mean)
        _require_finite("effective action standard deviation", effective)
        if not torch.all(effective > 0.0):
            raise FloatingPointError("effective action standard deviation must be positive")
        return effective

    def _set_distribution(self, mean: torch.Tensor) -> None:
        self.distribution = Normal(mean, self._effective_std(mean))

    def act_with_context(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = self._encoding(obs)
        mean = self._effective_mean(obs[self.current_obs_group], encoding)
        self._set_distribution(mean)
        return self.distribution.sample(), encoding.context_mu

    def act(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        del kwargs
        action, _ = self.act_with_context(obs)
        return action

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        encoding = self._encoding(obs)
        return self._effective_mean(obs[self.current_obs_group], encoding)

    def act_inference_with_estimator(
        self,
        obs: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoding = self._encoding(obs)
        action_mean = self._effective_mean(obs[self.current_obs_group], encoding)
        return action_mean, encoding.estimated_velocity, encoding.context_mu

    def estimator_inference(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = self._encoding(obs)
        return encoding.estimated_velocity, encoding.context_mu

    def evaluate(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        del kwargs
        value = self.critic(obs["critic"])
        _require_finite("value", value)
        return value

    def batch_forward(self, obs: TensorDict, actions: torch.Tensor) -> DreamWaQBatchOutput:
        encoding = self._encoding(obs)
        mean = self._effective_mean(obs[self.current_obs_group], encoding)
        sigma = self._effective_std(mean)
        distribution = Normal(mean, sigma)
        action_log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(obs["critic"])
        context_z = self.cenet.sample_context(encoding.context_mu, encoding.context_logvar)
        predicted = self.cenet.decode(context_z, encoding.estimated_velocity, actions.clamp(-1.0, 1.0))
        output = DreamWaQBatchOutput(
            mean,
            sigma,
            action_log_prob,
            entropy,
            value,
            encoding.estimated_velocity,
            encoding.context_mu,
            encoding.context_logvar,
            predicted,
        )
        for name, tensor in zip(DreamWaQBatchOutput._fields, output, strict=True):
            _require_finite(name, tensor)
        return output

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            return self.raw_std.abs().clamp(self.action_std_min, self.action_std_max)
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        result = self.distribution.log_prob(actions).sum(dim=-1)
        _require_finite("action log probability", result)
        return result

    def get_velocity_target(self, obs: TensorDict) -> torch.Tensor:
        return extract_velocity_target(obs["critic"])

    def update_normalization(self, obs: TensorDict) -> None:
        del obs
        if self.actor_obs_normalization or self.critic_obs_normalization:
            raise RuntimeError("DreamWaQ normalization flags changed after construction")

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
