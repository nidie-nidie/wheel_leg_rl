from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
import unittest

import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from recovery.contracts import GuidanceConfigV1
from recovery.guidance_networks import GuidanceCostCritic
from recovery.guidance_ppo import GuidancePPO
from recovery.guidance_storage import GuidanceRolloutStorage


def observations(num_envs: int, alpha: float) -> TensorDict:
    policy = torch.randn(num_envs, 111)
    critic = torch.randn(num_envs, 35)
    policy[:, -1] = critic[:, -1] = alpha
    return TensorDict({"policy": policy, "critic": critic}, batch_size=[num_envs])


def learning_algorithm(*, beta_force=1e-5, seed=731, num_envs=4, epochs=2, mini_batches=4, force_enabled=True):
    torch.manual_seed(seed)
    configuration = asdict(GuidanceConfigV1(beta_force=beta_force, force_enabled=force_enabled))
    observation = observations(num_envs, 1)
    policy = ActorCritic(observation, {"policy": ["policy"], "critic": ["critic"]}, 6,
                         actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64], noise_std_type="log")
    algorithm = GuidancePPO(policy, configuration, num_learning_epochs=epochs, num_mini_batches=mini_batches,
                            entropy_coef=0, schedule="adaptive", gamma=.99, lam=.95)
    algorithm.init_storage("rl", num_envs, 24, observation, [6])
    return algorithm


def collect_neural_transitions(algorithm, *, start_k=0, zero_cost=False):
    num_envs = algorithm.storage.num_envs
    enabled = algorithm.guidance_config["force_enabled"]
    observation = observations(num_envs, max(1 - start_k / 72000, 0) if enabled else 0)
    original_rewards = []
    with torch.inference_mode():
        for index in range(24):
            k = start_k + index + 1
            alpha = max(1 - (k - 1) / 72000, 0) if enabled else 0
            next_alpha = max(1 - k / 72000, 0) if enabled else 0
            algorithm.act(observation)
            following = observations(num_envs, next_alpha)
            rewards = torch.arange(num_envs, dtype=torch.float32) * .02 - .01 * index
            original_rewards.append(rewards.clone())
            dones = torch.zeros(num_envs, dtype=torch.bool)
            if index in (7, 19):
                dones[0] = True
            force = torch.zeros(num_envs) if zero_cost or alpha == 0 else torch.arange(1, num_envs + 1).float() * alpha
            extra = {"guidance": {
                "schema_version": "RecoveryGuidanceTransitionV1", "cost_force_n": force,
                "alpha_t": torch.full((num_envs,), alpha), "alpha_next": torch.full((num_envs,), next_alpha),
                "global_control_steps": k, "max_force_n": torch.full((num_envs,), 21.563625812),
            }}
            algorithm.process_env_step(following, rewards, dones, extra)
            observation = following
        algorithm.compute_returns(observation)
    return observation, torch.stack(original_rewards).unsqueeze(-1)


class GuidanceLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_math_storage(self, steps=3, num_envs=1):
        storage = GuidanceRolloutStorage("rl", num_envs, steps, observations(num_envs, 1), [6])
        storage.step = steps
        storage.alpha_t.fill_(1)
        storage.alpha_next.fill_(1)
        return storage

    def test_cost_gae_has_physical_units_and_reward_returns_stay_independent(self):
        storage = self.make_math_storage()
        storage.rewards[:, 0, 0] = torch.tensor([1., 2., 3.])
        storage.values[:, 0, 0] = torch.tensor([.4, .5, .6])
        storage.cost_force_n[:, 0, 0] = torch.tensor([4., 2., 1.])
        storage.cost_values[:, 0, 0] = torch.tensor([.2, .3, .4])
        storage.dones[1] = 1
        last_reward, last_cost = torch.tensor([[.8]]), torch.tensor([[.5]])
        storage.compute_guidance_returns(last_reward, last_cost, .9, .8, 10., .002)
        reward_advantage = torch.tensor([.6 + .9 * .5 + .9 * .8 * 1.5, 1.5, 3 + .9 * .8 - .6])
        cost_advantage_scaled = torch.tensor([.4 + .9 * .3 - .2 + .9 * .8 * (.2 - .3), .2 - .3, .1 + .9 * .5 - .4])
        torch.testing.assert_close(storage.reward_advantages[:, 0, 0], reward_advantage)
        torch.testing.assert_close(storage.returns[:, 0, 0], reward_advantage + torch.tensor([.4, .5, .6]))
        torch.testing.assert_close(storage.cost_advantages_physical[:, 0, 0], cost_advantage_scaled * 10)
        expected = reward_advantage - .002 / .1 * cost_advantage_scaled * 10
        torch.testing.assert_close(storage.mixed_advantages[:, 0, 0], expected)
        torch.testing.assert_close(storage.advantages[:, 0, 0], (expected - expected.mean()) / (expected.std(unbiased=False) + 1e-8))

    def test_true_terminal_removes_both_bootstraps(self):
        storage = self.make_math_storage(steps=1)
        storage.dones.fill_(1)
        storage.rewards.fill_(2)
        storage.cost_force_n.fill_(3)
        storage.values.fill_(.5)
        storage.cost_values.fill_(.1)
        storage.compute_guidance_returns(torch.tensor([[999.]]), torch.tensor([[999.]]), .99, .95, 10., 1e-5)
        self.assertEqual(float(storage.returns.item()), 2)
        self.assertAlmostEqual(float(storage.cost_returns.item()), .3, places=6)

    def test_withdrawal_transition_keeps_current_cost_and_zero_next_bootstrap(self):
        storage = self.make_math_storage(steps=2)
        storage.alpha_t[:, 0, 0] = torch.tensor([1 / 72000, 0])
        storage.alpha_next.zero_()
        storage.cost_force_n[0] = .0001
        storage.cost_values[0] = .2
        storage.cost_values[1] = 0
        storage.compute_guidance_returns(torch.zeros(1, 1), torch.tensor([[100.]]), .99, .95, 10., 1e-5)
        self.assertAlmostEqual(float(storage.cost_returns[0].item()), .00001, places=7)
        self.assertNotEqual(float(storage.cost_advantages_physical[0].item()), 0)
        self.assertEqual(float(storage.cost_advantages_physical[1].item()), 0)

    def test_physical_cost_advantage_is_invariant_to_fixed_critic_unit_scale(self):
        first, second = self.make_math_storage(), self.make_math_storage()
        for storage, scale in ((first, 10.), (second, 100.)):
            storage.cost_force_n[:, 0, 0] = torch.tensor([4., 2., 1.])
            storage.cost_values[:, 0, 0] = torch.tensor([2., 3., 4.]) / scale
            storage.compute_guidance_returns(torch.zeros(1, 1), torch.tensor([[5 / scale]]), .99, .95, scale, 1e-5)
        torch.testing.assert_close(first.cost_advantages_physical, second.cost_advantages_physical)
        torch.testing.assert_close(first.mixed_advantages, second.mixed_advantages)

    def test_shared_minibatch_indices_keep_every_signal_with_its_observation(self):
        storage = self.make_math_storage(steps=3, num_envs=4)
        labels = torch.arange(12, dtype=torch.float32).reshape(3, 4, 1)
        storage.observations["policy"][..., :1] = labels
        storage.observations["critic"][..., :1] = labels
        storage.actions[..., :1] = labels
        storage.cost_returns.copy_(labels * 7)
        seen = []
        for obs, fields in storage.guidance_mini_batch_generator(4, 1):
            label = obs["policy"][:, :1]
            torch.testing.assert_close(obs["critic"][:, :1], label)
            torch.testing.assert_close(fields["actions"][:, :1], label)
            torch.testing.assert_close(fields["cost_returns"], label * 7)
            seen.extend(label.flatten().tolist())
        self.assertEqual(sorted(seen), list(range(12)))

    def test_zero_alpha_cost_critic_output_and_parameter_gradient_are_exactly_zero(self):
        critic = GuidanceCostCritic()
        sample = observations(4, 0)["critic"]
        output = critic(sample)
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
        output.sum().backward()
        for parameter in critic.parameters():
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), rtol=0, atol=0)

    def test_real_policy_rollout_update_changes_both_networks_and_retains_task_rewards(self):
        algorithm = learning_algorithm()
        _, rewards = collect_neural_transitions(algorithm)
        torch.testing.assert_close(algorithm.storage.rewards, rewards, rtol=0, atol=0)
        torch.testing.assert_close(algorithm.storage.advantages.mean(), torch.tensor(0.), atol=1e-6, rtol=0)
        torch.testing.assert_close(algorithm.storage.advantages.std(unbiased=False), torch.tensor(1.), atol=1e-6, rtol=0)
        policy_before = {name: parameter.detach().clone() for name, parameter in algorithm.policy.named_parameters()}
        cost_before = {name: parameter.detach().clone() for name, parameter in algorithm.cost_critic.named_parameters()}
        losses = algorithm.update()
        self.assertTrue(all(math.isfinite(value) for value in losses.values()))
        self.assertTrue(any(not torch.equal(parameter, policy_before[name]) for name, parameter in algorithm.policy.named_parameters()))
        self.assertTrue(any(not torch.equal(parameter, cost_before[name]) for name, parameter in algorithm.cost_critic.named_parameters()))
        self.assertEqual(algorithm.storage.step, 0)
        self.assertGreater(losses["guidance_force_mean_n"], 0)

    def test_zero_assistance_rollout_has_zero_cost_loss_and_zero_actor_penalty(self):
        algorithm = learning_algorithm()
        collect_neural_transitions(algorithm)
        algorithm.update()
        algorithm.load_guidance_state_dict(deepcopy(algorithm.guidance_state_dict()))
        collect_neural_transitions(algorithm, start_k=72000)
        for value in (algorithm.storage.cost_values, algorithm.storage.cost_returns,
                      algorithm.storage.cost_advantages_physical, algorithm.storage.penalty_advantages):
            torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)
        torch.testing.assert_close(algorithm.storage.mixed_advantages, algorithm.storage.reward_advantages, rtol=0, atol=0)
        before = deepcopy(algorithm.cost_critic.state_dict())
        losses = algorithm.update()
        self.assertEqual(losses["guidance_cost_value_function"], 0)
        for key, value in before.items():
            torch.testing.assert_close(algorithm.cost_critic.state_dict()[key], value, rtol=0, atol=0)

    def test_ratio_weighted_cost_retains_gradient_when_old_cost_advantage_mean_is_zero(self):
        mean = torch.tensor(0., requires_grad=True)
        action = torch.tensor([-1., 1.])
        old_log_prob = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.)).log_prob(action)
        ratio = (torch.distributions.Normal(mean, torch.tensor(1.)).log_prob(action) - old_log_prob).exp()
        cost_advantage = torch.tensor([-2., 2.])
        reward_advantage = torch.tensor([.2, -.1])
        beta, gamma = .001, .99
        mixed = reward_advantage - beta / (1 - gamma) * cost_advantage
        mixed_gradient = torch.autograd.grad((ratio * mixed).mean(), mean, retain_graph=True)[0]
        penalty_gradient = torch.autograd.grad(
            (ratio * reward_advantage).mean() - beta * (123. + (ratio * cost_advantage).mean() / (1 - gamma)), mean,
            retain_graph=True,
        )[0]
        cost_gradient = torch.autograd.grad((ratio * cost_advantage).mean(), mean)[0]
        self.assertEqual(float(cost_advantage.mean()), 0)
        self.assertNotEqual(float(cost_gradient), 0)
        torch.testing.assert_close(mixed_gradient, penalty_gradient)

    def test_sampling_and_inference_preserve_actions_above_unit_range(self):
        algorithm = learning_algorithm()
        with torch.no_grad():
            algorithm.policy.actor[-1].weight.zero_()
            algorithm.policy.actor[-1].bias.fill_(3)
        observation = observations(4, 1)
        with torch.inference_mode():
            inference = algorithm.policy.act_inference(observation)
            sample = algorithm.act(observation)
        torch.testing.assert_close(inference, torch.full((4, 6), 3.))
        self.assertTrue(bool((sample > 1).any()))

    def test_guidance_checkpoint_restores_cost_parameters_and_adam_state(self):
        algorithm = learning_algorithm()
        collect_neural_transitions(algorithm)
        algorithm.update()
        state = deepcopy(algorithm.guidance_state_dict())
        restored = learning_algorithm(seed=991)
        restored.load_guidance_state_dict(state)
        for key, value in algorithm.cost_critic.state_dict().items():
            torch.testing.assert_close(restored.cost_critic.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(len(restored.cost_optimizer.state), len(algorithm.cost_optimizer.state))
        for parameter, original in zip(restored.cost_optimizer.state.values(), algorithm.cost_optimizer.state.values(), strict=True):
            for key in original:
                torch.testing.assert_close(parameter[key], original[key], rtol=0, atol=0)

    def test_invalid_independent_minibatch_normalization_is_rejected(self):
        algorithm = learning_algorithm()
        with self.assertRaisesRegex(ValueError, "整批标准化"):
            GuidancePPO(algorithm.policy, asdict(GuidanceConfigV1()), normalize_advantage_per_mini_batch=True)

    def test_initial_cost_diagnostic_uses_saved_initial_features_at_requested_alpha(self):
        algorithm = learning_algorithm()
        initial = observations(4, .9)["critic"]
        preserved = initial.clone()
        expected_input = initial.clone()
        expected_input[:, -1] = .3
        with torch.no_grad():
            expected = algorithm.cost_critic(expected_input).mean() * algorithm.cost_scale_n
        torch.testing.assert_close(algorithm.estimate_initial_cost(initial, .3), expected)
        self.assertEqual(float(algorithm.estimate_initial_cost(initial, 0)), 0)
        torch.testing.assert_close(initial, preserved, rtol=0, atol=0)

    def test_force_disabled_has_zero_alpha_and_cost_from_first_transition(self):
        algorithm = learning_algorithm(force_enabled=False)
        collect_neural_transitions(algorithm)
        self.assertEqual(float(algorithm.storage.alpha_t.sum()), 0)
        self.assertEqual(float(algorithm.storage.cost_advantages_physical.sum()), 0)
        self.assertEqual(algorithm.update()["guidance_force_mean_n"], 0)


if __name__ == "__main__":
    unittest.main()
