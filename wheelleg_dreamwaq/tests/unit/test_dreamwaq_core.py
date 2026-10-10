from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from wheelleg_dreamwaq.algorithms.dreamwaq.actor_critic import DreamWaQActorCritic
from wheelleg_dreamwaq.algorithms.dreamwaq.cenet import CENet
from wheelleg_dreamwaq.deployment.dreamwaq_inference import DreamWaQInferenceActorV1
from wheelleg_dreamwaq.schemas.observation import (
    append_frame_major_history,
    extract_proprio_reconstruction_target,
    extract_velocity_target,
    flatten_frame_major_history,
    initialize_frame_major_history,
)


def _observations(batch_size: int = 4) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, 25),
            "policy_history": torch.randn(batch_size, 125),
            "critic": torch.randn(batch_size, 41),
        },
        batch_size=[batch_size],
    )


def _policy(batch_size: int = 4) -> tuple[DreamWaQActorCritic, TensorDict]:
    observations = _observations(batch_size)
    policy = DreamWaQActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        6,
    )
    return policy, observations


def test_history_layout_is_frame_major_and_done_resets_to_current_frame() -> None:
    current = torch.arange(50, dtype=torch.float32).reshape(2, 25)
    history = initialize_frame_major_history(current)
    assert history.shape == (2, 5, 25)
    assert torch.equal(flatten_frame_major_history(history)[0], current[0].repeat(5))

    numbered = torch.arange(2 * 5 * 25, dtype=torch.float32).reshape(2, 5, 25)
    next_observation = torch.full((2, 25), -3.0)
    advanced = append_frame_major_history(numbered, next_observation, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(advanced[0, :4], numbered[0, 1:])
    assert torch.equal(advanced[0, 4], next_observation[0])
    assert torch.equal(advanced[1], next_observation[1].repeat(5, 1))


def test_velocity_and_reconstruction_targets_use_clean_critic_slices() -> None:
    critic = torch.arange(82, dtype=torch.float32).reshape(2, 41)
    assert torch.equal(extract_velocity_target(critic), critic[:, 25:28])
    expected = torch.cat((critic[:, 0:6], critic[:, 9:19]), dim=-1)
    assert torch.equal(extract_proprio_reconstruction_target(critic), expected)
    assert expected.shape == (2, 16)


def test_network_topology_and_parameter_counts_are_frozen() -> None:
    policy, _ = _policy()
    counts = {
        "encoder": sum(parameter.numel() for parameter in policy.cenet.encoder.parameters()),
        "actor": sum(parameter.numel() for parameter in policy.actor.parameters()),
        "critic": sum(parameter.numel() for parameter in policy.critic.parameters()),
        "decoder": sum(parameter.numel() for parameter in policy.cenet.decoder.parameters()),
    }
    assert counts == {"encoder": 26659, "actor": 53062, "critic": 51969, "decoder": 12048}


def test_actor_is_deterministic_and_does_not_read_context_logvar_rows() -> None:
    torch.manual_seed(7)
    policy, observations = _policy()
    before = policy.act_inference(observations).detach().clone()
    encoder_output = policy.cenet.encoder[-1]
    assert isinstance(encoder_output, torch.nn.Linear)
    with torch.no_grad():
        encoder_output.bias[19:35].add_(100.0)
        encoder_output.weight[19:35].mul_(-4.0)
    after = policy.act_inference(observations)
    assert torch.equal(before, after)


def test_reparameterization_is_explicit_and_decoder_detaches_velocity_and_action() -> None:
    cenet = CENet()
    mu = torch.randn(3, 16, requires_grad=True)
    logvar = torch.randn(3, 16, requires_grad=True)
    epsilon = torch.randn(3, 16)
    first = cenet.sample_context(mu, logvar, epsilon)
    second = cenet.sample_context(mu, logvar, epsilon)
    assert torch.equal(first, second)
    assert torch.allclose(first, mu + torch.exp(0.5 * logvar) * epsilon)

    velocity = torch.randn(3, 3, requires_grad=True)
    action = torch.randn(3, 6, requires_grad=True)
    prediction = cenet.decode(first, velocity, action)
    prediction.sum().backward()
    assert mu.grad is not None
    assert logvar.grad is not None
    assert velocity.grad is None
    assert action.grad is None


def test_batch_forward_uses_stored_action_and_returns_all_contract_outputs() -> None:
    policy, observations = _policy(batch_size=5)
    actions = torch.randn(5, 6)
    output = policy.batch_forward(observations, actions)
    assert output.action_mean.shape == (5, 6)
    assert output.action_sigma.shape == (5, 6)
    assert output.action_log_prob.shape == (5,)
    assert output.entropy.shape == (5,)
    assert output.value.shape == (5, 1)
    assert output.estimated_velocity.shape == (5, 3)
    assert output.context_mu.shape == (5, 16)
    assert output.context_logvar.shape == (5, 16)
    assert output.predicted_next_proprio.shape == (5, 16)


def test_inference_action_and_estimator_share_one_encoder_call(monkeypatch) -> None:
    policy, observations = _policy(batch_size=4)
    calls = 0
    original = policy.cenet.encode

    def counted(history: torch.Tensor):
        nonlocal calls
        calls += 1
        return original(history)

    monkeypatch.setattr(policy.cenet, "encode", counted)
    with torch.inference_mode():
        action, velocity, context_mu = policy.act_inference_with_estimator(observations)

    assert calls == 1
    assert action.shape == (4, 6)
    assert velocity.shape == (4, 3)
    assert context_mu.shape == (4, 16)


def test_action_distribution_rejects_non_finite_raw_std() -> None:
    policy, observations = _policy()
    with torch.no_grad():
        policy.raw_std[0] = torch.nan
    with pytest.raises(FloatingPointError, match="raw action standard deviation"):
        policy.act(observations)


def test_actor_critic_rejects_extra_observation_sets_and_unknown_config() -> None:
    observations = _observations()
    with pytest.raises(ValueError, match="obs_groups"):
        DreamWaQActorCritic(
            observations,
            {"policy": ["policy"], "critic": ["critic"], "extra": ["policy_history"]},
            6,
        )
    with pytest.raises(TypeError, match="Unexpected"):
        DreamWaQActorCritic(
            observations,
            {"policy": ["policy"], "critic": ["critic"]},
            6,
            obsolete_field=True,
        )


def test_torchscript_inference_actor_has_dynamic_batch_and_matches_eager() -> None:
    torch.manual_seed(11)
    policy, _ = _policy()
    inference_actor = DreamWaQInferenceActorV1.from_policy(policy).eval()
    scripted = torch.jit.script(inference_actor)
    for batch_size in (1, 7, 32):
        history = torch.randn(batch_size, 125, dtype=torch.float32)
        with torch.inference_mode():
            expected = inference_actor(history)
            actual = scripted(history)
        assert actual.shape == (batch_size, 6)
        assert torch.max(torch.abs(expected - actual)).item() <= 1.0e-7
