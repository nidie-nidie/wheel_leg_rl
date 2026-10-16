from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict

from wheelleg_dreamwaq.algorithms.dreamwaq.ppo import DreamWaQPPO
from wheelleg_dreamwaq.deployment.dreamwaq_inference import DreamWaQInferenceActorV1
from wheelleg_dreamwaq.schemas.normalization import NormalizationV2
from wheelleg_dreamwaq.schemas.observation import (
    append_frame_major_history, initialize_frame_major_history,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import ControlLimits, compute_action_targets

from debug.no_action_clip.experiment import (
    RawActionActorCritic, RawActionNormalization, raw_action_targets, verify_formal_files,
)


def observations(batch=4):
    current = torch.randn(batch, 25)
    critic = torch.randn(batch, 41)
    critic[:, :25] = current
    return TensorDict({
        "policy": current, "policy_history": current.repeat(1, 5), "critic": critic,
    }, batch_size=[batch])


def test_formal_file_identity():
    assert verify_formal_files() >= 98


def test_in_range_actions_equal_formal_controller():
    action = torch.tensor([[0.2, -0.5, 1.0, -1.0, 0.3, -0.7]])
    nominal = torch.tensor([-0.33367134, 0.33367134, -0.33367134, 0.33367134])
    actual = raw_action_targets(action, nominal, ControlLimits())
    expected = compute_action_targets(action, nominal, ControlLimits())
    for received, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(received, wanted, rtol=0, atol=0)


def test_raw_actions_keep_angle_limits_and_wheel_signs():
    actions = torch.tensor([[8.0, -8.0, 2.0, -2.0, 2.0, 3.0]])
    raw, legs, wheels = raw_action_targets(actions, torch.zeros(4), ControlLimits())
    torch.testing.assert_close(raw, actions, rtol=0, atol=0)
    torch.testing.assert_close(legs, torch.tensor([[1.0, -1.0, 0.7, -0.7]]))
    torch.testing.assert_close(wheels, torch.tensor([[50.0, -75.0]]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_action_rejected(bad):
    actions = torch.zeros(1, 6)
    actions[0, 0] = bad
    with pytest.raises(FloatingPointError):
        raw_action_targets(actions, torch.zeros(4), ControlLimits())
    with pytest.raises(FloatingPointError):
        RawActionNormalization.normalize_previous_action(actions)


def test_normalization_and_partial_history_keep_raw_actions():
    values = torch.tensor([[2.0, -3.0, 4.0, -5.0, 6.0, -7.0]])
    normalization = RawActionNormalization()
    torch.testing.assert_close(normalization.normalize_previous_action(values), values, rtol=0, atol=0)
    assert not torch.equal(NormalizationV2.normalize_previous_action(values), values)
    velocity = torch.tensor([[99.0, -99.0, 0.5]])
    torch.testing.assert_close(normalization.normalize_joint_velocity(velocity),
                               NormalizationV2().normalize_joint_velocity(velocity), rtol=0, atol=0)
    current = torch.zeros(2, 25)
    current[:, 19:25] = values
    history = initialize_frame_major_history(current)
    next_frame = current.clone()
    next_frame[0, 19:25] = 0
    next_frame[1, 19:25] *= 2
    updated = append_frame_major_history(history, next_frame, torch.tensor([1, 0]))
    torch.testing.assert_close(updated[0], next_frame[0].expand(5, 25), rtol=0, atol=0)
    torch.testing.assert_close(updated[1, :-1], history[1, 1:], rtol=0, atol=0)
    torch.testing.assert_close(updated[1, -1, 19:25], values[0] * 2, rtol=0, atol=0)


def test_cenet_decoder_receives_detached_raw_actions():
    obs = observations()
    policy = RawActionActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 6)
    actions = torch.tensor([[2.0, -3.0, 4.0, -5.0, 6.0, -7.0]]).repeat(4, 1).requires_grad_()
    received = []

    def record_input(module, inputs):
        received.append(inputs[0].detach().clone())

    handle = policy.cenet.decoder.register_forward_pre_hook(record_input)
    output = policy.batch_forward(obs, actions)
    handle.remove()
    torch.testing.assert_close(received[0][:, -6:], actions.detach(), rtol=0, atol=0)
    gradient = torch.autograd.grad(output.predicted_next_proprio.sum(), actions, allow_unused=True)[0]
    assert gradient is None
    assert output.action_mean.abs().max() <= 20.0


def test_real_ppo_storage_and_update_preserve_raw_actions():
    torch.manual_seed(73)
    obs = observations(8)
    policy = RawActionActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 6)
    algorithm = DreamWaQPPO(policy, num_learning_epochs=1, num_mini_batches=1,
                           learning_rate=1.0e-3, device="cpu")
    algorithm.init_storage("rl", 8, 2, obs, [6])
    expected = []
    for _ in range(2):
        with torch.inference_mode():
            action = algorithm.act(obs)
            expected.append(action.clone())
            next_obs = observations(8)
            algorithm.process_env_step(next_obs, torch.linspace(-0.5, 0.5, 8), torch.zeros(8, dtype=torch.long), {})
        obs = next_obs
    torch.testing.assert_close(algorithm.storage.actions, torch.stack(expected), rtol=0, atol=0)
    assert algorithm.storage.actions.abs().max() > 1.0
    with torch.inference_mode():
        algorithm.compute_returns(obs)
    losses = algorithm.update()
    assert all(math.isfinite(value) for value in losses.values())
    assert algorithm.estimator_monitor_state.completed_rollouts == 1
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())


def test_torchscript_golden_with_large_previous_actions():
    torch.manual_seed(20261015)
    obs = observations(3)
    obs["policy_history"][:, 19::25] = 3.0
    obs["policy"] = obs["policy_history"][:, -25:].clone()
    policy = RawActionActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 6).eval()
    actor = DreamWaQInferenceActorV1.from_policy(policy).eval()
    scripted = torch.jit.script(actor)
    with torch.inference_mode():
        torch.testing.assert_close(scripted(obs["policy_history"]), policy.act_inference(obs), rtol=1e-6, atol=1e-6)
