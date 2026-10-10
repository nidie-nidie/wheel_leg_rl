from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheelleg_dreamwaq.schemas.observation import (
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    HISTORY_OBS_DIM,
    append_frame_major_history,
    flatten_frame_major_history,
    initialize_frame_major_history,
)


class DreamWaQHistoryVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL wrapper that appends an immutable frame-major five-frame policy history."""

    def __init__(self, env, clip_actions: float | None = None) -> None:
        super().__init__(env, clip_actions=clip_actions)
        first_observation = super().get_observations()
        self._history = initialize_frame_major_history(self._validate_base_observation(first_observation)["policy"])
        self._first_policy_observation = first_observation["policy"].clone()
        self._cached_observations = self._build_snapshot(first_observation)

    @staticmethod
    def _validate_base_observation(observations: TensorDict) -> TensorDict:
        if set(observations.keys()) != {"policy", "critic"}:
            raise ValueError(f"base environment observations must be policy/critic, got {sorted(observations.keys())}")
        expected = {"policy": ACTOR_OBS_DIM, "critic": CRITIC_OBS_DIM}
        for key, width in expected.items():
            value = observations[key]
            if value.ndim != 2 or value.shape[-1] != width or value.dtype != torch.float32:
                raise ValueError(f"{key} observation must be float32 [N,{width}], got {value.dtype} {tuple(value.shape)}")
        return observations

    def _build_snapshot(self, observations: TensorDict) -> TensorDict:
        observations = self._validate_base_observation(observations)
        snapshot = observations.clone(recurse=True)
        snapshot.set("policy_history", flatten_frame_major_history(self._history).clone())
        if snapshot["policy_history"].shape != (self.num_envs, HISTORY_OBS_DIM):
            raise RuntimeError("DreamWaQ history has an invalid shape")
        if not torch.equal(snapshot["policy_history"][:, -ACTOR_OBS_DIM:], snapshot["policy"]):
            raise RuntimeError("DreamWaQ history current frame differs from the policy observation")
        return snapshot

    @property
    def first_policy_observation(self) -> torch.Tensor:
        return self._first_policy_observation.clone()

    def get_observations(self) -> TensorDict:
        return self._cached_observations.clone(recurse=True)

    def reset(self) -> tuple[TensorDict, dict]:
        _, extras = super().reset()
        first_observation = super().get_observations()
        first_observation = self._validate_base_observation(first_observation)
        self._history = initialize_frame_major_history(first_observation["policy"])
        self._first_policy_observation = first_observation["policy"].clone()
        self._cached_observations = self._build_snapshot(first_observation)
        return self.get_observations(), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        observations, rewards, dones, extras = super().step(actions)
        observations = self._validate_base_observation(observations)
        self._history = append_frame_major_history(self._history, observations["policy"], dones)
        self._cached_observations = self._build_snapshot(observations)
        return self.get_observations(), rewards, dones, extras

