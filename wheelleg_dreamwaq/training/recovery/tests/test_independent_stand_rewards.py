from copy import deepcopy
import unittest

import torch

from recovery.independent_rewards import compute_independent_reward, deadband_score, smoothstep
from independent_real_data import real_reference, real_reward_state


class IndependentStandingRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference = real_reference()

    def test_real_401_scores_are_finite_and_within_continuous_bound(self):
        state = real_reward_state(self.reference, count=560, index=0)
        reward, terms = compute_independent_reward(**state)
        self.assertTrue(bool(torch.isfinite(reward).all()))
        self.assertLessEqual(float(reward.max()), .107001)
        torch.testing.assert_close(reward, torch.stack(list(terms.values())).sum(dim=0))
        self.assertTrue(bool((terms["success"] == 0).all()))

    def test_independent_velocity_scores_do_not_depend_on_pose_error(self):
        state = real_reward_state(self.reference)
        _, before = compute_independent_reward(**state)
        state["q"] += .6
        _, after = compute_independent_reward(**state)
        torch.testing.assert_close(before["angular_velocity"], after["angular_velocity"])
        torch.testing.assert_close(before["horizontal_velocity"], after["horizontal_velocity"])
        self.assertGreater(float(before["pose"]), float(after["pose"]))

    def test_velocity_penalty_does_not_disable_pose_score(self):
        state = real_reward_state(self.reference)
        _, before = compute_independent_reward(**state)
        state["omega"] *= 50.
        state["dq"] *= 50.
        _, after = compute_independent_reward(**state)
        torch.testing.assert_close(before["pose"], after["pose"])
        self.assertGreater(float(before["angular_velocity"]), float(after["angular_velocity"]))

    def test_nonwheel_support_has_separate_negative_contribution(self):
        state = real_reward_state(self.reference)
        _, before = compute_independent_reward(**state)
        state["nonwheel_force_mean"] += self.reference.identity["weight_n"] * .1
        _, after = compute_independent_reward(**state)
        self.assertGreater(float(before["nonwheel_support"]), float(after["nonwheel_support"]))
        torch.testing.assert_close(before["pose"], after["pose"])

    def test_terminal_rewards_have_no_dt_scaling_and_are_mutually_exclusive(self):
        state = real_reward_state(self.reference)
        _, terms = compute_independent_reward(**state)
        for name, expected in (("success", 20.), ("physical_failure", -10.), ("timeout", -5.)):
            changed = deepcopy(state)
            changed[name][:] = True
            _, terminal = compute_independent_reward(**changed)
            self.assertEqual(float(terminal[name]), expected)
        state["success"][:] = state["physical_failure"][:] = True
        with self.assertRaises(ValueError):
            compute_independent_reward(**state)

    def test_continuous_gates_and_deadband(self):
        theta = torch.linspace(.50, .15, 1001)
        gate = smoothstep((.5 - theta) / .35)
        self.assertTrue(bool((gate[1:] >= gate[:-1]).all()))
        self.assertEqual(float(gate[0]), 0.)
        self.assertEqual(float(gate[-1]), 1.)
        score = deadband_score(torch.tensor([-.04, 0., .04, .14]), .04, .1)
        torch.testing.assert_close(score, torch.tensor([1., 1., 1., .5]))

    def test_executed_action_bounds_are_enforced(self):
        state = real_reward_state(self.reference)
        state["actions"][0, 0] = 2.
        with self.assertRaises(ValueError):
            compute_independent_reward(**state)
