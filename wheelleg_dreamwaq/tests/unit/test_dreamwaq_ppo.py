from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict

from wheelleg_dreamwaq.algorithms.dreamwaq.actor_critic import DreamWaQActorCritic
from wheelleg_dreamwaq.algorithms.dreamwaq.ppo import (
    DreamWaQPPO,
    masked_reconstruction_loss,
    velocity_zero_baseline_mse_ratio,
)
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import EstimatorMonitorStateV1


def _observations(batch_size: int) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, 25),
            "policy_history": torch.randn(batch_size, 125),
            "critic": torch.randn(batch_size, 41),
        },
        batch_size=[batch_size],
    )


def _algorithm(num_envs: int = 4) -> tuple[DreamWaQPPO, TensorDict]:
    observations = _observations(num_envs)
    policy = DreamWaQActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        6,
    )
    algorithm = DreamWaQPPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        schedule="adaptive",
        device="cpu",
    )
    algorithm.init_storage("rl", num_envs, 2, observations, [6])
    return algorithm, observations


def test_estimator_monitor_round_trip_and_interrupted_sequence_are_equivalent() -> None:
    values = [0.1 + 0.01 * index for index in range(14)]
    uninterrupted = EstimatorMonitorStateV1()
    for value in values:
        uninterrupted.record_rollout(value)

    interrupted = EstimatorMonitorStateV1()
    for value in values[:6]:
        interrupted.record_rollout(value)
    interrupted = EstimatorMonitorStateV1.from_dict(interrupted.to_dict())
    for value in values[6:]:
        interrupted.record_rollout(value)

    assert interrupted.to_dict() == uninterrupted.to_dict()
    assert interrupted.completed_rollouts == 14
    assert interrupted.context_mu_feature_std_mean_initial == pytest.approx(values[0])
    assert interrupted.recent_context_mu_feature_std_mean == pytest.approx(values[-10:])
    assert interrupted.final_window_mean == pytest.approx(sum(values[-10:]) / 10)


def test_estimator_monitor_rejects_missing_or_inconsistent_state() -> None:
    with pytest.raises(ValueError, match="keys"):
        EstimatorMonitorStateV1.from_dict({})
    payload = EstimatorMonitorStateV1().to_dict()
    payload["completed_rollouts"] = 1
    with pytest.raises(ValueError, match="initial"):
        EstimatorMonitorStateV1.from_dict(payload)


def test_masked_reconstruction_uses_element_count_and_graph_connected_zero() -> None:
    prediction = torch.arange(32, dtype=torch.float32).reshape(2, 16).requires_grad_()
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True], [False]])
    loss = masked_reconstruction_loss(prediction, target, mask)
    assert loss.item() == pytest.approx(float(prediction[0].square().mean().item()))
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad[1]) == 0

    prediction = torch.randn(2, 16, requires_grad=True)
    zero_loss = masked_reconstruction_loss(prediction, torch.zeros_like(prediction), torch.zeros(2, 1, dtype=torch.bool))
    assert zero_loss.item() == 0.0
    zero_loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_velocity_zero_baseline_ratio_uses_floor() -> None:
    estimate = torch.ones(2, 3)
    target = torch.zeros(2, 3)
    mse, zero_mse, ratio = velocity_zero_baseline_mse_ratio(estimate, target, 1.0e-8)
    assert mse.item() == pytest.approx(1.0)
    assert zero_mse.item() == 0.0
    assert ratio.item() == pytest.approx(1.0e8)


def test_act_keeps_pending_snapshot_immutable() -> None:
    algorithm, observations = _algorithm()
    with torch.inference_mode():
        algorithm.act(observations)
    snapshot = algorithm.transition.observations.clone(recurse=True)
    observations["policy"].add_(100.0)
    observations["policy_history"].mul_(0.0)
    observations["critic"].sub_(50.0)
    for key in snapshot.keys():
        assert torch.equal(algorithm.transition.observations[key], snapshot[key])


def test_process_env_step_uses_long_done_mask_and_clean_next_target() -> None:
    algorithm, observations = _algorithm()
    with torch.inference_mode():
        algorithm.act(observations)
        next_observations = _observations(4)
        dones = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        expected_target = torch.cat((next_observations["critic"][:, 0:6], next_observations["critic"][:, 9:19]), dim=-1)
        algorithm.process_env_step(next_observations, torch.ones(4), dones, {"time_outs": dones.to(torch.bool)})
    assert torch.equal(algorithm.storage.reconstruction_mask[0, :, 0], torch.tensor([True, False, True, False]))
    assert torch.equal(algorithm.storage.next_proprio_target[0], expected_target)


def test_one_update_is_finite_increments_monitor_and_preserves_adaptive_kl() -> None:
    torch.manual_seed(23)
    algorithm, observations = _algorithm()
    for step in range(2):
        with torch.inference_mode():
            algorithm.act(observations)
            next_observations = _observations(4)
            dones = torch.zeros(4, dtype=torch.long)
            algorithm.process_env_step(next_observations, torch.randn(4), dones, {})
        observations = next_observations
    with torch.inference_mode():
        algorithm.compute_returns(observations)
    result = algorithm.update()

    assert algorithm.estimator_monitor_state.completed_rollouts == 1
    assert algorithm.storage.step == 0
    assert all(math.isfinite(value) for value in result.values())
    assert result["context_mu_feature_std_mean_rollout"] >= 0.0
    assert result["velocity"] >= 0.0
    assert result["reconstruction"] >= 0.0
    assert algorithm.learning_rate > 0.0
    assert algorithm.optimizer.param_groups[0]["lr"] == algorithm.learning_rate


def test_adaptive_kl_formula_updates_all_parameter_groups() -> None:
    algorithm, _ = _algorithm()
    mean = torch.zeros(4, 6)
    sigma = torch.ones(4, 6)
    algorithm._adapt_learning_rate(sigma, sigma.clone(), mean.clone(), mean)
    assert algorithm.learning_rate == pytest.approx(1.5e-3)
    assert [group["lr"] for group in algorithm.optimizer.param_groups] == pytest.approx([1.5e-3])
