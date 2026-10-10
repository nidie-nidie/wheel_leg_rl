from __future__ import annotations

import torch

from wheelleg_dreamwaq.schemas.observation import (
    ACTOR_OBS_DIM,
    HISTORY_LENGTH,
    HISTORY_OBS_DIM,
    append_frame_major_history,
    flatten_frame_major_history,
    initialize_frame_major_history,
)


class FrameMajorHistoryV1:
    """External history runtime shared by play, Isaac evaluation, MuJoCo, and deployment."""

    def __init__(self, first_policy_observation: torch.Tensor) -> None:
        self._history = initialize_frame_major_history(first_policy_observation)

    @property
    def num_envs(self) -> int:
        return int(self._history.shape[0])

    @property
    def device(self) -> torch.device:
        return self._history.device

    @property
    def dtype(self) -> torch.dtype:
        return self._history.dtype

    def reset(self, first_policy_observation: torch.Tensor) -> None:
        replacement = initialize_frame_major_history(first_policy_observation)
        if replacement.shape != self._history.shape:
            raise ValueError("History reset batch shape differs from the existing runtime")
        self._history.copy_(replacement)

    def append(self, next_policy_observation: torch.Tensor, dones: torch.Tensor | None = None) -> None:
        if next_policy_observation.shape != (self.num_envs, ACTOR_OBS_DIM):
            raise ValueError(f"next policy observation must have shape [{self.num_envs},{ACTOR_OBS_DIM}]")
        if next_policy_observation.dtype != self.dtype or next_policy_observation.device != self.device:
            raise ValueError("next policy observation dtype/device differs from history")
        if dones is None:
            self._history.copy_(append_frame_major_history(self._history, next_policy_observation))
            return
        if dones.shape != (self.num_envs,) or dones.dtype != torch.bool or dones.device != self.device:
            raise ValueError("dones must be bool with shape [N] on the history device")
        advanced = append_frame_major_history(self._history, next_policy_observation)
        if dones.any():
            reset_history = initialize_frame_major_history(next_policy_observation)
            advanced = torch.where(dones.view(-1, 1, 1), reset_history, advanced)
        self._history.copy_(advanced)

    def flat(self) -> torch.Tensor:
        flat = flatten_frame_major_history(self._history)
        if flat.shape != (self.num_envs, HISTORY_OBS_DIM):
            raise RuntimeError("FrameMajorHistoryV1 produced an invalid flattened shape")
        return flat.clone()

    def frames(self) -> torch.Tensor:
        if self._history.shape[1:] != (HISTORY_LENGTH, ACTOR_OBS_DIM):
            raise RuntimeError("FrameMajorHistoryV1 internal shape is invalid")
        return self._history.clone()
