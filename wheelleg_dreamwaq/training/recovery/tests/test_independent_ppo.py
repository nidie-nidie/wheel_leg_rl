import torch

from recovery.bounded_actor_critic import BoundedActorCritic
from recovery.independent_ppo import IndependentPPO
from test_bounded_actor_critic import actual_observations


def ppo_config():
    return {"learning_rate": 3e-4, "gamma": 0.99, "lam": 0.95, "num_mini_batches": 4,
            "num_learning_epochs": 5, "clip_param": 0.2, "value_loss_coef": 1.0,
            "entropy_coef": 0.001, "max_grad_norm": 1.0}


def test_real_reference_ppo_update_changes_finite_parameters_and_restores_optimizer():
    actor, critic, median = actual_observations()
    policy = BoundedActorCritic(median)
    algorithm = IndependentPPO(policy, ppo_config())
    with torch.no_grad():
        sampled = policy.sample(actor, critic)
        batch = {"policy": actor.reshape(24, 4, 110), "critic": critic.reshape(24, 4, 41),
                 "latent": sampled.latent.reshape(24, 4, 6), "action": sampled.action.reshape(24, 4, 6),
                 "normal_log_prob": sampled.normal_log_prob.reshape(24, 4), "mean": sampled.mean.reshape(24, 4, 6),
                 "std": sampled.std.reshape(24, 4, 6), "value": sampled.value.reshape(24, 4)}
        # 真实参考轨迹的相邻价值用于独立检验优化器更新，不作为环境训练成绩。
        returns = sampled.value.roll(-1).reshape(24, 4)
        batch.update(returns=returns, advantages=returns - batch["value"])
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    result = algorithm.update(batch)
    assert result["measured_minibatch_count"] == 20
    assert 1e-5 <= result["learning_rate"] <= 3e-4
    assert any(not torch.equal(before[key], value) for key, value in policy.state_dict().items())
    assert all(torch.isfinite(value).all() for value in policy.state_dict().values())
    resumed = IndependentPPO(policy, ppo_config())
    resumed.load_state_dict(algorithm.state_dict())
    assert resumed.learning_rate == algorithm.learning_rate
    assert resumed.optimizer.state_dict()["param_groups"] == algorithm.optimizer.state_dict()["param_groups"]


def test_finite_horizon_gae_masks_all_terminal_types():
    _, _, median = actual_observations()
    algorithm = IndependentPPO(BoundedActorCritic(median), ppo_config())
    rewards = torch.tensor([[1., 2.], [3., 4.]])
    dones = torch.tensor([[True, False], [False, True]])
    values = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    returns, _ = algorithm.returns(rewards, dones, values, torch.tensor([5., 1000.]))
    assert returns[0, 0] == 1.
    assert returns[1, 1] == 4.
    assert torch.allclose(returns[1, 0], torch.tensor(3. + 0.99 * 5.))
