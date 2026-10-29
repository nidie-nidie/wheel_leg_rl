from copy import deepcopy
import unittest

import torch

from recovery.independent_state import ContactWindow, REASON_CODES, StandingTask
from independent_real_data import real_reference, real_standing_state


class IndependentStandingStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference = real_reference()

    def make_task(self, count=1, mode="train"):
        return StandingTask(count, "cpu", self.reference, self.reference.identity["weight_n"], mode=mode)

    def test_actual_contact_window_and_local_reset(self):
        data = real_standing_state(self.reference, count=2)
        window = ContactWindow(2, "cpu")
        for index in range(10):
            window.push(data["wheel_forces"], data["nonwheel_force"])
            self.assertTrue(bool((window.count == index + 1).all()))
        before = window.means[1].clone()
        window.reset(torch.tensor([0]))
        self.assertEqual(int(window.count[0]), 0)
        torch.testing.assert_close(window.means[1], before, rtol=0, atol=0)
        self.assertEqual(int(window.count[1]), 10)

    def test_fifty_qualifying_then_one_hundred_hold_steps(self):
        task = self.make_task()
        state = real_standing_state(self.reference)
        for step in range(1, 159):
            result = task.step(**state)
            self.assertFalse(bool(result.success[0]))
            if step == 9:
                self.assertFalse(bool(result.qualified[0]))
            if step == 58:
                self.assertEqual(int(task.qualification_count[0]), 49)
            if step == 59:
                self.assertEqual(int(task.phase[0]), task.HOLD)
                self.assertEqual(int(task.hold_count[0]), 0)
            if step == 158:
                self.assertEqual(int(task.hold_count[0]), 99)
        result = task.step(**state)
        self.assertTrue(bool(result.success[0]))
        self.assertFalse(bool(result.failure[0]))
        repeated = task.step(**state)
        self.assertFalse(bool(repeated.success[0]))
        self.assertFalse(bool(repeated.failure[0]))

    def test_partial_reset_preserves_other_environment_phase_and_origin(self):
        task = self.make_task(2)
        state = real_standing_state(self.reference, count=2)
        for _ in range(70):
            task.step(**state)
        previous = {name: getattr(task, name)[1].clone() for name in (
            "hold_count", "qualification_count", "phase", "hold_origin", "elapsed_steps", "position_delta")}
        task.reset(torch.tensor([0]))
        for name, value in previous.items():
            torch.testing.assert_close(getattr(task, name)[1], value, rtol=0, atol=0)
        self.assertEqual(int(task.phase[0]), task.GETUP)
        self.assertEqual(int(task.contact_window.count[0]), 0)

    def test_checkpoint_restores_actual_contact_history_and_continues_identically(self):
        task = self.make_task(2)
        state = real_standing_state(self.reference, count=2)
        for _ in range(67):
            task.step(**state)
        restored = self.make_task(2)
        restored.load_state_dict(task.state_dict())
        for _ in range(92):
            original, resumed = task.step(**state), restored.step(**state)
            for name in ("qualified", "success", "failure", "timeout", "reason_codes"):
                torch.testing.assert_close(getattr(original, name), getattr(resumed, name), rtol=0, atol=0)
        self.assertTrue(bool(original.success.all()))

    def test_training_hold_interruption_resets_all_continuous_counters(self):
        task = self.make_task()
        state = real_standing_state(self.reference)
        for _ in range(70):
            task.step(**state)
        moved = deepcopy(state)
        moved["world_xy"][:, 0] += .11
        result = task.step(**moved)
        self.assertFalse(bool(result.failure[0]))
        self.assertEqual(int(task.phase[0]), task.GETUP)
        self.assertEqual(int(task.hold_count[0]), 0)
        self.assertEqual(int(task.qualification_count[0]), 0)

    def test_eval_hold_interruption_terminates(self):
        task = self.make_task(mode="eval")
        state = real_standing_state(self.reference)
        for _ in range(70):
            task.step(**state)
        moved = deepcopy(state)
        moved["world_xy"][:, 0] += .21
        result = task.step(**moved)
        self.assertTrue(bool(result.failure[0]))
        self.assertTrue(bool(result.timeout[0]))
        self.assertEqual(int(result.reason_codes[0]), REASON_CODES["hold_interrupted"])

    def test_getup_deadline_400_and_success_at_500(self):
        task = self.make_task()
        state = real_standing_state(self.reference)
        unqualified = deepcopy(state)
        unqualified["gravity"] *= -1.
        for _ in range(350):
            self.assertFalse(bool(task.step(**unqualified).failure[0]))
        for _ in range(50):
            result = task.step(**state)
        self.assertEqual(int(task.elapsed_steps[0]), 400)
        self.assertEqual(int(task.phase[0]), task.HOLD)
        self.assertFalse(bool(result.failure[0]))
        for _ in range(100):
            result = task.step(**state)
        self.assertEqual(int(task.elapsed_steps[0]), 500)
        self.assertTrue(bool(result.success[0]))

    def test_400_step_deadline_without_qualification_is_failure(self):
        task = self.make_task()
        state = real_standing_state(self.reference)
        state["gravity"] *= -1.
        for _ in range(399):
            self.assertFalse(bool(task.step(**state).failure[0]))
        result = task.step(**state)
        self.assertTrue(bool(result.timeout[0]))
        self.assertEqual(int(result.reason_codes[0]), REASON_CODES["getup_timeout"])

    def test_physical_failure_has_priority_over_success(self):
        task = self.make_task()
        state = real_standing_state(self.reference)
        for _ in range(158):
            task.step(**state)
        state["physical_failure"][:] = True
        result = task.step(**state)
        self.assertTrue(bool(result.failure[0]))
        self.assertFalse(bool(result.timeout[0]))
        self.assertFalse(bool(result.success[0]))
        self.assertEqual(int(result.reason_codes[0]), REASON_CODES["physical_failure"])

    def test_eval_500_additional_steps_and_900_boundary(self):
        task = self.make_task(mode="eval")
        state = real_standing_state(self.reference)
        unqualified = deepcopy(state)
        unqualified["gravity"] *= -1.
        for _ in range(350):
            task.step(**unqualified)
        for _ in range(549):
            result = task.step(**state)
            self.assertFalse(bool(result.success[0]))
        self.assertEqual(int(task.elapsed_steps[0]), 899)
        result = task.step(**state)
        self.assertEqual(int(task.elapsed_steps[0]), 900)
        self.assertTrue(bool(result.success[0]))
