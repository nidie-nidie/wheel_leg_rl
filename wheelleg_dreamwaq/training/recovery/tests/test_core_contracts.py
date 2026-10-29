from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
import unittest

import torch

from recovery.contracts import RecoveryContractV2, default_resolved_config, validate_resolved_config
from recovery.control import check_action_compatibility, decode_action, decode_walker_action, estimate_pd_torque
from recovery.observations import HistoryBuffer, build_actor_frame, build_critic_frame
from recovery.randomization import RecoveryRandomization, RecoveryRandomizationV1
from recovery.rewards import compute_reward


class CoreContractTests(unittest.TestCase):
    def test_raw_action_and_unbounded_targets_are_preserved(self):
        action = torch.tensor([[10., -10., 2., -2., 3., -4.]])
        original = action.clone()
        leg, wheel = decode_action(action, torch.zeros(4))
        torch.testing.assert_close(leg, torch.tensor([[3.5, -3.5, .7, -.7]]))
        torch.testing.assert_close(wheel, torch.tensor([[75., -100.]]))
        torch.testing.assert_close(action, original)

    def test_frozen_walker_keeps_original_targets_and_raw_action(self):
        action = torch.tensor([[10., -10., 2., -2., 3., -4.]])
        original = action.clone()
        leg, wheel = decode_walker_action(action, torch.zeros(4))
        torch.testing.assert_close(leg, torch.tensor([[1., -1., .7, -.7]]))
        torch.testing.assert_close(wheel, torch.tensor([[75., -100.]]))
        torch.testing.assert_close(action, original)

    def test_unbounded_recovery_targets_keep_actuator_effort_limits(self):
        actions = torch.tensor([[100., -100., 100., -100., 100., -100.]])
        leg, wheel = decode_action(actions, torch.zeros(4))
        plan = RecoveryRandomization(1, 1, profile=RecoveryRandomizationV1(enabled=False))
        torque = estimate_pd_torque(leg, wheel, torch.zeros(1, 6), torch.zeros(1, 6),
            plan.realized_stiffness, plan.realized_damping, plan.realized_effort_limit)
        torch.testing.assert_close(torque, torch.tensor([[18., -18., 18., -18., 9., -9.]]))

    def test_each_environment_keeps_its_reference_without_target_clipping(self):
        actions = torch.tensor([[6., -6., 6., -6., 2., -2.], [-6., 6., -6., 6., -2., 2.]])
        reference = torch.tensor([[-.3, .3, -.4, .4], [-.2, .2, -.5, .5]])
        leg, wheel = decode_action(actions, reference)
        torch.testing.assert_close(leg, reference + .35 * actions[:, :4], rtol=0, atol=0)
        torch.testing.assert_close(wheel, 25.0 * actions[:, 4:], rtol=0, atol=0)
        self.assertTrue(bool((leg.abs() > 1.0).all()))

    def test_current_contract_rejects_legacy_action_configuration(self):
        legacy = asdict(RecoveryContractV2())
        legacy.update(schema_version="RecoveryContractV1", leg_target_lower=-1.0, leg_target_upper=1.0)
        with self.assertRaises(ValueError):
            RecoveryContractV2.from_dict(legacy)
        configuration = default_resolved_config()
        configuration["assessment"]["deployment_action_limit"] = 10.0
        with self.assertRaises(ValueError):
            validate_resolved_config(configuration)
        configuration = default_resolved_config()
        configuration["schema_version"] = "RecoveryResolvedConfigV1"
        with self.assertRaises(ValueError):
            validate_resolved_config(configuration)

    def test_actual_pd_parameters_and_effort_are_used(self):
        leg = torch.tensor([[.5, -.5, .1, -.1]])
        wheel = torch.tensor([[5., -10.]])
        q, qd = torch.zeros(1, 6), torch.ones(1, 6)
        kp = torch.tensor([[120., 100., 80., 60., 0., 0.]])
        kd = torch.tensor([[4., 3., 2., 1., .6, .5]])
        limit = torch.tensor([[17.1, 18., 18., 18., 8.55, 9.]])
        result = estimate_pd_torque(leg, wheel, q, qd, kp, kd, limit)
        torch.testing.assert_close(result, torch.tensor([[17.1, -18., 6., -7., 2.4, -5.5]]))

    def test_compatibility_does_not_assume_raw_actions_are_close(self):
        actual, candidate = torch.zeros(1, 6), torch.zeros(1, 6)
        actual[:, 0], candidate[:, 0] = 10., 20.
        plan = RecoveryRandomization(1, 1, profile=RecoveryRandomizationV1(enabled=False))
        result = check_action_compatibility(actual, candidate, torch.zeros(1, 6), torch.zeros(1, 6),
            torch.zeros(4), plan.realized_stiffness, plan.realized_damping,
            plan.realized_effort_limit, torch.ones(6) * .1, torch.ones(6) * .1)
        self.assertFalse(bool(result["compatible"][0]))
        self.assertEqual(float(result["target_difference"].max()), 2.5)
        self.assertEqual(float(result["torque_difference"].max()), 0.)

    def test_walker_compatibility_uses_its_own_decoding_for_both_sides(self):
        actual, candidate = torch.zeros(1, 6), torch.zeros(1, 6)
        actual[:, 0], candidate[:, 0] = 10., 20.
        plan = RecoveryRandomization(1, 1, profile=RecoveryRandomizationV1(enabled=False))
        result = check_action_compatibility(actual, candidate, torch.zeros(1, 6), torch.zeros(1, 6),
            torch.zeros(4), plan.realized_stiffness, plan.realized_damping,
            plan.realized_effort_limit, torch.ones(6) * .1, torch.ones(6) * .1,
            actual_is_walker=True)
        self.assertTrue(bool(result["compatible"][0]))
        self.assertEqual(float(result["target_difference"].max()), 0.)

    def test_observation_indices_gravity_and_previous_raw_action(self):
        gyro = torch.tensor([[4., 8., 12.]])
        gravity = torch.tensor([[0., 0., -1.]])
        q = torch.tensor([[.1, .2, .3, .4, 0., 0.]])
        qd = torch.arange(6).float().view(1, 6) * 10
        previous = torch.tensor([[10., -10., 2., -2., 3., -4.]])
        frame = build_actor_frame(gyro, gravity, q, qd, previous, torch.zeros(4))
        torch.testing.assert_close(frame[:, :3], torch.tensor([[1., 2., 3.]]))
        torch.testing.assert_close(frame[:, 3:6], gravity)
        torch.testing.assert_close(frame[:, 6:10], q[:, :4])
        torch.testing.assert_close(frame[:, 10:16], qd * .05)
        torch.testing.assert_close(frame[:, 16:22], previous)
        critic = build_critic_frame(frame, torch.tensor([[1., 2., 3.]]), torch.tensor([.24]),
                                    torch.ones(1, 6) * 10, torch.tensor([.5]), torch.tensor([.75]))
        self.assertEqual(tuple(critic.shape), (1, 34))
        torch.testing.assert_close(critic[:, 22:25], torch.tensor([[2., 4., 6.]]))
        torch.testing.assert_close(critic[:, 25], torch.tensor([1.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(critic[:, 26:32], torch.ones(1, 6) * .5)
        torch.testing.assert_close(critic[:, 32:], torch.tensor([[.5, .75]]))

    def test_physical_reset_history_and_repeated_reads(self):
        history = HistoryBuffer(2)
        frames = torch.arange(220).float().reshape(2, 5, 22)
        history.reset(torch.arange(2), frames)
        before = history.frames()[1].clone()
        history.reset(torch.tensor([0]), torch.ones(1, 5, 22))
        torch.testing.assert_close(history.frames()[1], before)
        history.flattened().zero_()
        torch.testing.assert_close(history.frames()[1], before)
        current = torch.ones(2, 22) * 500
        history.append(current, 1)
        torch.testing.assert_close(history.frames()[:, -1], current)
        with self.assertRaises(ValueError):
            history.append(current, 1)

    def test_reward_matches_independent_scalar_formula(self):
        cfg = RecoveryContractV2()
        angle = .3
        gravity = torch.tensor([[math.sin(angle), 0., -math.cos(angle)]], dtype=torch.float64)
        q = torch.tensor([[*cfg.q_nominal, 0., 0.]], dtype=torch.float64)
        qd = torch.tensor([[0., 0., 0., 0., 1., -1.]], dtype=torch.float64)
        omega = torch.tensor([[.1, .2, 0.]], dtype=torch.float64)
        action = torch.tensor([[1., 2., 3., 4., 5., 6.]], dtype=torch.float64)
        previous, previous2 = action * .5, action * .25
        torque = torch.tensor([[1., 2., 3., 4., 5., 6.]], dtype=torch.float64)
        limits = torch.tensor([cfg.effort_limits], dtype=torch.float64)
        reward, terms = compute_reward(gravity, q, qd, omega, torch.tensor([.21]), torque, limits,
            action, previous, previous2, torch.tensor([.3]), torch.tensor([False]), torch.tensor([False]),
            torch.tensor([True]), q_nominal=cfg.q_nominal)
        c = (1 + math.cos(angle)) / 2
        expected = (.02 * 2 * c + 2 * (.99 * c - .3)
                    + .02 * math.exp(-((.21 - .20) / .04) ** 2)
                    + .02 * math.exp(-(.1**2 + .2**2) / .5**2 - .25)
                    - .02 * .002 * sum((float(t) / float(l)) ** 2 for t, l in zip(torque[0], limits[0])) / 6
                    - .02 * .01 * sum((.25 * value) ** 2 for value in (1, 2, 3, 4)) / 4
                    - .02 * .01 * sum((.25 * value) ** 2 for value in (5, 6)) / 2 - .002)
        self.assertAlmostEqual(float(reward[0]), expected, places=6)
        self.assertAlmostEqual(float(terms["potential"][0]), c, places=7)

    def test_terminal_potential_discounted_sum_and_dense_bound(self):
        initial = .25
        values = [initial] + [(index % 19) / 18 for index in range(1, 400)] + [0.]
        total = sum(.99 ** index * 2 * (.99 * values[index + 1] - values[index]) for index in range(400))
        self.assertAlmostEqual(total, -2 * initial, places=12)
        cfg = RecoveryContractV2()
        q = torch.tensor([[*cfg.q_nominal, 0., 0.]], dtype=torch.float64)
        zeros = torch.zeros(1, 6, dtype=torch.float64)
        args = (torch.tensor([[0., 0., -1.]], dtype=torch.float64), q, zeros,
                torch.zeros(1, 3, dtype=torch.float64), torch.tensor([.2], dtype=torch.float64),
                zeros, torch.tensor([cfg.effort_limits], dtype=torch.float64), zeros, zeros, zeros,
                torch.tensor([1.], dtype=torch.float64))
        reward, terms = compute_reward(*args, torch.tensor([False]), torch.tensor([False]), torch.tensor([False]))
        self.assertAlmostEqual(float(reward - terms["progress"]), .078, places=7)
        terminal, final_terms = compute_reward(*args, torch.tensor([True]), torch.tensor([False]), torch.tensor([False]))
        self.assertAlmostEqual(float(final_terms["progress"]), -2., places=7)
        self.assertAlmostEqual(float(terminal), 8.078, places=6)

    def test_randomization_preserves_plan_and_exact_noise_resume(self):
        runtime = RecoveryRandomization(3, 42)
        saved = deepcopy(runtime.state_dict())
        expected = runtime.sample_actor_noise(3)
        runtime.load_state_dict(saved)
        actual = runtime.sample_actor_noise(3)
        for name in ("angular_velocity", "projected_gravity", "leg_position_error", "joint_velocity"):
            torch.testing.assert_close(getattr(expected, name), getattr(actual, name), atol=0, rtol=0)
        other = runtime.pose_generator(1).get_state().clone()
        torch.rand(10, generator=runtime.pose_generator(0))
        torch.testing.assert_close(runtime.pose_generator(1).get_state(), other)
        self.assertEqual(runtime.manifest(), saved["manifest"])
        upper = torch.tensor([18., 18., 18., 18., 9., 9.])
        self.assertTrue(bool((runtime.realized_effort_limit <= upper).all()))
        self.assertTrue(bool((runtime.realized_effort_limit >= upper * .95).all()))

    def test_formal_configuration_rejects_missing_experiment_evidence(self):
        config = default_resolved_config()
        validate_resolved_config(config)
        config["environment"]["task_type"] = "formal_training"
        with self.assertRaises(ValueError):
            validate_resolved_config(config)
        with self.assertRaises(ValueError):
            RecoveryRandomizationV1(effort_limit_scale_range=(.95, 1.05))
        with self.assertRaises(ValueError):
            RecoveryRandomizationV1(reset_velocity_perturbation_enabled=True)
        with self.assertRaises(ValueError):
            RecoveryContractV2(effort_limits=(100.,) * 6)
        with self.assertRaises(ValueError):
            RecoveryContractV2(actor_hidden_dims=(512, 256, 128))


if __name__ == "__main__":
    unittest.main()
