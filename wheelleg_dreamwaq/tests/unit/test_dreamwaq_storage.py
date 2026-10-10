from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from wheelleg_dreamwaq.algorithms.dreamwaq.storage import DreamWaQMiniBatch, DreamWaQRolloutStorage


def _sample_observations(num_envs: int) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.zeros(num_envs, 25),
            "policy_history": torch.zeros(num_envs, 125),
            "critic": torch.zeros(num_envs, 41),
        },
        batch_size=[num_envs],
    )


def _transition(num_envs: int, marker: float = 1.0) -> DreamWaQRolloutStorage.Transition:
    transition = DreamWaQRolloutStorage.Transition()
    transition.observations = _sample_observations(num_envs)
    transition.observations["policy"][:, 0] = marker
    transition.actions = torch.full((num_envs, 6), marker)
    transition.rewards = torch.full((num_envs,), marker)
    transition.dones = torch.zeros(num_envs, dtype=torch.long)
    transition.values = torch.full((num_envs, 1), marker)
    transition.actions_log_prob = torch.full((num_envs,), marker)
    transition.action_mean = torch.full((num_envs, 6), marker)
    transition.action_sigma = torch.full((num_envs, 6), marker + 1.0)
    transition.next_proprio_target = torch.full((num_envs, 16), marker, dtype=torch.float32)
    transition.reconstruction_mask = torch.ones(num_envs, 1, dtype=torch.bool)
    return transition


def test_storage_preallocates_and_copies_inference_transition_fields() -> None:
    storage = DreamWaQRolloutStorage("rl", 2, 2, _sample_observations(2), [6], "cpu")
    assert storage.next_proprio_target.shape == (2, 2, 16)
    assert storage.reconstruction_mask.shape == (2, 2, 1)
    assert storage.next_proprio_target.dtype == torch.float32
    assert storage.reconstruction_mask.dtype == torch.bool

    with torch.inference_mode():
        transition = _transition(2, marker=3.0)
        source_target = transition.next_proprio_target
        source_mask = transition.reconstruction_mask
    storage.add_transitions(transition)
    transition.clear()
    with torch.inference_mode():
        source_target.fill_(99.0)
        source_mask.zero_()
    assert torch.equal(storage.next_proprio_target[0], torch.full((2, 16), 3.0))
    assert storage.reconstruction_mask[0].all()
    assert not storage.next_proprio_target.is_inference()
    assert not storage.reconstruction_mask.is_inference()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("next_proprio_target", None, "requires"),
        ("reconstruction_mask", None, "requires"),
        ("next_proprio_target", torch.zeros(2, 15), "next_proprio_target"),
        ("next_proprio_target", torch.zeros(2, 16, dtype=torch.float64), "next_proprio_target"),
        ("reconstruction_mask", torch.ones(2, 1, dtype=torch.long), "reconstruction_mask"),
        ("reconstruction_mask", torch.ones(2, dtype=torch.bool), "reconstruction_mask"),
    ],
)
def test_storage_rejects_missing_or_invalid_target_and_mask(field: str, value, message: str) -> None:
    storage = DreamWaQRolloutStorage("rl", 2, 1, _sample_observations(2), [6], "cpu")
    transition = _transition(2)
    setattr(transition, field, value)
    with pytest.raises(ValueError, match=message):
        storage.add_transitions(transition)


def test_named_minibatch_uses_one_shuffle_index_for_all_fields() -> None:
    storage = DreamWaQRolloutStorage("rl", 2, 2, _sample_observations(2), [6], "cpu")
    for step in range(2):
        transition = _transition(2, marker=float(step + 1))
        transition.observations["policy"][:, 0] = torch.tensor([10.0 * step + 1.0, 10.0 * step + 2.0])
        transition.next_proprio_target[:, 0] = transition.observations["policy"][:, 0]
        storage.add_transitions(transition)
    storage.compute_returns(torch.zeros(2, 1), gamma=0.99, lam=0.95)

    batches = list(storage.mini_batch_generator(num_mini_batches=2, num_epochs=1))
    assert len(batches) == 2
    assert all(isinstance(batch, DreamWaQMiniBatch) for batch in batches)
    for batch in batches:
        assert torch.equal(batch.observations["policy"][:, 0], batch.next_proprio_target[:, 0])
        assert batch.old_mu.shape[-1] == 6
        assert batch.old_sigma.shape[-1] == 6

