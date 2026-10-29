from __future__ import annotations

from typing import NamedTuple

import torch
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage

from wheelleg_dreamwaq.schemas.observation import PROPRIO_RECONSTRUCTION_TARGET_DIM


class DreamWaQMiniBatch(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    target_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_actions_log_prob: torch.Tensor
    old_mu: torch.Tensor
    old_sigma: torch.Tensor
    hidden_states: tuple[None, None]
    masks: None
    next_proprio_target: torch.Tensor
    reconstruction_mask: torch.Tensor


class DreamWaQRolloutStorage(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self) -> None:
            super().__init__()
            self.next_proprio_target: torch.Tensor | None = None
            self.reconstruction_mask: torch.Tensor | None = None

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.next_proprio_target = torch.zeros(
            num_transitions_per_env,
            num_envs,
            PROPRIO_RECONSTRUCTION_TARGET_DIM,
            dtype=torch.float32,
            device=device,
        )
        self.reconstruction_mask = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            dtype=torch.bool,
            device=device,
        )

    def add_transitions(self, transition: Transition) -> None:
        target = transition.next_proprio_target
        mask = transition.reconstruction_mask
        if target is None or mask is None:
            raise ValueError("DreamWaQ transition requires next_proprio_target and reconstruction_mask")
        expected_target_shape = (self.num_envs, PROPRIO_RECONSTRUCTION_TARGET_DIM)
        expected_mask_shape = (self.num_envs, 1)
        if target.shape != expected_target_shape or target.dtype != torch.float32:
            raise ValueError(
                f"next_proprio_target must be float32 {expected_target_shape}, got {target.dtype} {tuple(target.shape)}"
            )
        if mask.shape != expected_mask_shape or mask.dtype != torch.bool:
            raise ValueError(
                f"reconstruction_mask must be bool {expected_mask_shape}, got {mask.dtype} {tuple(mask.shape)}"
            )
        storage_device = self.next_proprio_target.device
        if target.device != storage_device or mask.device != storage_device:
            raise ValueError("DreamWaQ target/mask device must match rollout storage")
        step = self.step
        super().add_transitions(transition)
        self.next_proprio_target[step].copy_(target)
        self.reconstruction_mask[step].copy_(mask)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training")
        batch_size = self.num_envs * self.num_transitions_per_env
        if batch_size % num_mini_batches != 0:
            raise ValueError(f"rollout batch {batch_size} is not divisible by {num_mini_batches}")
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        next_proprio_target = self.next_proprio_target.flatten(0, 1)
        reconstruction_mask = self.reconstruction_mask.flatten(0, 1)

        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                start = index * mini_batch_size
                stop = (index + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                yield DreamWaQMiniBatch(
                    observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                    next_proprio_target[batch_idx],
                    reconstruction_mask[batch_idx],
                )

