import json
import math
import unittest
from copy import deepcopy
from dataclasses import replace

from recovery.metrics import AttemptContext, ObservationEndKind, RecoveryMetrics
from recovery.supervisor import (
    ControlMode,
    RecoverySupervisor,
    SupervisorConfig,
    SupervisorMeasurements,
    physical_conditions,
)


def measurement(frame_id: int, **values) -> SupervisorMeasurements:
    base = SupervisorMeasurements(
        frame_id=frame_id,
        tilt_rad=0.0,
        angular_velocity=(0.0, 0.0, 0.0),
        leg_position_error=(0.0, 0.0, 0.0, 0.0),
        leg_velocity=(0.0, 0.0, 0.0, 0.0),
        wheel_velocity=(0.0, 0.0),
        com_height_m=0.2,
        left_wheel_support=True,
        right_wheel_support=True,
        body_or_leg_support=False,
        action_compatible=True,
    )
    return replace(base, **values)


def advance(supervisor: RecoverySupervisor, count: int = 1, *, manual=False, **values):
    result = supervisor.snapshot()
    for _ in range(count):
        frame_id = 1 if result.frame_id is None else result.frame_id + 1
        result = supervisor.post_step(measurement(frame_id, **values), manual_recovery=manual)
    return result


class SupervisorTests(unittest.TestCase):
    def test_initial_combined_recovery_starts_before_first_action_without_time_credit(self):
        supervisor = RecoverySupervisor(initial_recovery=True)
        initial = supervisor.snapshot()
        self.assertEqual(initial.mode, ControlMode.RECOVERY)
        self.assertEqual((initial.event_steps, initial.recovery_steps, initial.attempt_steps), (0, 0, 0))
        self.assertEqual((initial.event_id, initial.attempt_id), (1, 1))
        self.assertEqual(advance(supervisor).attempt_steps, 1)

    def test_physical_conditions_check_individual_dimensions_and_boundaries(self):
        good = measurement(1)
        self.assertTrue(all(physical_conditions(good).values()))
        checks = (
            ({"tilt_rad": 0.25}, "tilt"),
            ({"angular_velocity": (0.3, 0.4, 0.0)}, "angular_velocity"),
            ({"leg_position_error": (0.0, 0.2, 0.0, 0.0)}, "leg_position/1"),
            ({"leg_velocity": (0.0, 0.0, -1.0, 0.0)}, "leg_velocity/2"),
            ({"wheel_velocity": (0.0, 2.0)}, "wheel_velocity/1"),
            ({"com_height_m": 0.16}, "height"),
            ({"left_wheel_support": False}, "left_wheel_support"),
            ({"right_wheel_support": False}, "right_wheel_support"),
            ({"body_or_leg_support": True}, "body_or_leg_support"),
        )
        for changes, label in checks:
            with self.subTest(label=label):
                self.assertFalse(physical_conditions(replace(good, **changes))[label])
        for height in (0.17, 0.25):
            self.assertTrue(physical_conditions(replace(good, com_height_m=height))["height"])

    def test_fall_requires_ten_actual_steps_and_trigger_state_is_initial(self):
        supervisor = RecoverySupervisor()
        result = advance(supervisor, 9, tilt_rad=1.0)
        self.assertEqual(result.mode, ControlMode.LOCOMOTION)
        result = advance(supervisor, tilt_rad=1.0)
        self.assertEqual(result.transition, "LOCOMOTION->RECOVERY")
        self.assertEqual((result.event_steps, result.recovery_steps, result.attempt_steps), (0, 0, 0))
        result = advance(supervisor, tilt_rad=1.0)
        self.assertEqual((result.event_steps, result.recovery_steps, result.attempt_steps), (1, 1, 1))

    def test_fall_threshold_and_hold_reset(self):
        supervisor = RecoverySupervisor()
        advance(supervisor, 9, tilt_rad=1.0)
        advance(supervisor, tilt_rad=0.9)
        self.assertEqual(advance(supervisor, 9, tilt_rad=1.0).mode, ControlMode.LOCOMOTION)
        self.assertEqual(advance(supervisor, tilt_rad=1.0).mode, ControlMode.RECOVERY)

    def test_manual_requests_in_recovery_do_not_restart_attempt(self):
        supervisor = RecoverySupervisor()
        advance(supervisor, manual=True, tilt_rad=1.0)
        result = advance(supervisor, 30, manual=True, tilt_rad=1.0)
        self.assertEqual((result.attempt_id, result.attempt_steps, result.recovery_steps), (1, 30, 30))
        self.assertEqual(len(supervisor.metrics.attempts), 1)

    def test_recovery_confirm_and_locomotion_hold_are_separate(self):
        supervisor = RecoverySupervisor()
        advance(supervisor, manual=True)
        self.assertEqual(advance(supervisor, 49).mode, ControlMode.RECOVERY)
        handoff = advance(supervisor)
        self.assertTrue(handoff.capture_heading)
        self.assertEqual(handoff.mode, ControlMode.LOCOMOTION)
        self.assertEqual(handoff.zero_stand_hold_steps, 0)
        self.assertEqual(handoff.attempt_ended.confirmation_delay_s, 0.0)
        self.assertTrue(advance(supervisor, 99).zero_command)
        finished = advance(supervisor)
        self.assertTrue(finished.event_finished)
        self.assertFalse(finished.zero_command)
        self.assertEqual((finished.event_steps, finished.recovery_steps), (150, 50))
        self.assertFalse(advance(supervisor).capture_heading)

    def test_zero_stand_does_not_require_action_compatibility(self):
        supervisor = RecoverySupervisor()
        advance(supervisor, manual=True)
        advance(supervisor, 50)
        result = advance(
            supervisor, 100, action_compatible=False,
            compatibility_blockers=("torque_difference/0",),
        )
        self.assertTrue(result.event_finished)
        self.assertEqual(supervisor.metrics.attempts[0].physical_without_compatibility_steps, 0)

    def test_zero_stand_stability_loss_resets_only_its_hold(self):
        supervisor = RecoverySupervisor()
        advance(supervisor, manual=True)
        advance(supervisor, 50)
        advance(supervisor, 70)
        result = advance(supervisor, wheel_velocity=(2.0, 0.0))
        self.assertEqual(result.zero_stand_hold_steps, 0)
        self.assertEqual(len(supervisor.metrics.attempts), 1)
        result = advance(supervisor, 100)
        self.assertTrue(result.event_finished)
        self.assertEqual(supervisor.metrics.events[0].zero_stand_break_count, 1)

    def test_standalone_success_exactly_on_400_wins_deadline(self):
        supervisor = RecoverySupervisor(standalone=True)
        advance(supervisor, 350, tilt_rad=1.0)
        result = advance(supervisor, 50)
        self.assertTrue(result.event_finished)
        self.assertIsNone(result.fault_reason)
        self.assertTrue(supervisor.completed)
        self.assertEqual(result.attempt_ended.handoff_step, 400)
        with self.assertRaises(RuntimeError):
            advance(supervisor)
        supervisor.new_episode()
        self.assertEqual(supervisor.snapshot().attempt_steps, 0)
        self.assertEqual(advance(supervisor, tilt_rad=1.0).attempt_steps, 1)

    def test_400_without_full_confirmation_faults_even_when_upright(self):
        supervisor = RecoverySupervisor(standalone=True)
        advance(supervisor, 379, tilt_rad=1.0)
        result = advance(supervisor, 21)
        self.assertEqual(result.fault_reason, "recovery_deadline")
        self.assertEqual(result.attempt_ended.max_joint_confirm_steps, 21)
        self.assertIsNone(result.attempt_ended.confirmation_delay_s)
        self.assertEqual(result.attempt_ended.observation_end_kind, ObservationEndKind.TASK_FAILURE)

    def test_combined_400_handoff_then_exact_500_zero_stand_succeeds(self):
        supervisor = RecoverySupervisor()
        advance(supervisor, manual=True)
        advance(supervisor, 350, tilt_rad=1.0)
        result = advance(supervisor, 50)
        self.assertEqual(result.transition, "RECOVERY->LOCOMOTION")
        self.assertEqual(result.recovery_steps, 400)
        result = advance(supervisor, 100)
        self.assertEqual(result.event_steps, 500)
        self.assertTrue(result.event_finished)
        self.assertIsNone(result.fault_reason)

    def test_used_recovery_budget_prevents_fall_reentry(self):
        supervisor = RecoverySupervisor()
        advance(supervisor, manual=True)
        advance(supervisor, 350, tilt_rad=1.0)
        advance(supervisor, 50)
        result = advance(supervisor, 10, tilt_rad=1.0)
        self.assertEqual(result.fault_reason, "recovery_budget_exhausted_on_fall")
        self.assertEqual(len(supervisor.metrics.attempts), 1)
        self.assertEqual(result.recovery_steps, 400)

    def test_event_500_handoff_only_is_failure_and_retains_rejected_confirmation(self):
        supervisor = RecoverySupervisor()
        advance(supervisor, manual=True)
        advance(supervisor, 50)
        advance(supervisor, 199, tilt_rad=0.5)
        advance(supervisor, manual=True, tilt_rad=0.5)
        advance(supervisor, 200, tilt_rad=1.0)
        result = advance(supervisor, 50)
        self.assertEqual((result.event_steps, result.recovery_steps), (500, 300))
        self.assertEqual(result.fault_reason, "event_deadline")
        self.assertEqual(result.attempt_ended.joint_ready_step, 250)
        self.assertIsNone(result.attempt_ended.handoff_step)
        self.assertEqual(result.attempt_ended.end_reason, "event_deadline")

    def test_event_deadline_precedes_same_step_ordinary_fall_transition(self):
        supervisor = RecoverySupervisor()
        advance(supervisor, manual=True)
        advance(supervisor, 50)
        advance(supervisor, 440, tilt_rad=0.5)
        result = advance(supervisor, 10, tilt_rad=1.0)
        self.assertEqual(result.fault_reason, "event_deadline")
        self.assertEqual(len(supervisor.metrics.attempts), 1)

    def test_reentry_preserves_event_and_recovery_budgets(self):
        supervisor = RecoverySupervisor()
        advance(supervisor, manual=True)
        advance(supervisor, 50)
        result = advance(supervisor, 10, tilt_rad=1.0)
        self.assertEqual((result.event_id, result.attempt_id), (1, 2))
        self.assertEqual((result.event_steps, result.recovery_steps, result.attempt_steps), (60, 50, 0))
        advance(supervisor, 300, tilt_rad=1.0)
        result = advance(supervisor, 50)
        self.assertEqual((result.event_steps, result.recovery_steps), (410, 400))
        self.assertEqual(result.transition, "RECOVERY->LOCOMOTION")
        advance(supervisor, 89)
        result = advance(supervisor)
        self.assertEqual(result.fault_reason, "event_deadline")
        self.assertEqual(result.zero_stand_hold_steps, 90)
        self.assertEqual(supervisor.metrics.events[0].recurrent_fall_count, 1)

    def test_physical_failure_has_priority_over_success_and_deadline(self):
        supervisor = RecoverySupervisor(standalone=True)
        advance(supervisor, 350, tilt_rad=1.0)
        advance(supervisor, 49)
        result = advance(supervisor, physical_failure_reason="closure_error")
        self.assertEqual(result.fault_reason, "closure_error")
        self.assertIsNone(result.attempt_ended.handoff_step)
        self.assertEqual(result.attempt_ended.end_step, 400)
        self.assertEqual(result.attempt_ended.steps[-1].blockers, ("failure/closure_error",))

    def test_nonfinite_failure_is_immediate_and_latched(self):
        supervisor = RecoverySupervisor(standalone=True)
        result = advance(supervisor, angular_velocity=(math.nan, 0.0, 0.0))
        self.assertEqual(result.fault_reason, "nonfinite_state")
        self.assertTrue(result.halt_requested)
        with self.assertRaises(RuntimeError):
            advance(supervisor)
        supervisor.reenable()
        self.assertEqual(supervisor.snapshot().attempt_id, 2)
        self.assertEqual(advance(supervisor).attempt_steps, 1)

    def test_duplicate_or_skipped_frames_rejected_without_counter_change(self):
        supervisor = RecoverySupervisor(standalone=True)
        supervisor.post_step(measurement(11))
        baseline = supervisor.snapshot()
        for frame in (11, 13, 10):
            with self.assertRaises(ValueError):
                supervisor.post_step(measurement(frame))
            self.assertEqual(supervisor.snapshot(), baseline)
        self.assertEqual(supervisor.post_step(measurement(12)).attempt_steps, 2)

    def test_repeated_snapshot_has_no_timing_or_metric_side_effects(self):
        supervisor = RecoverySupervisor(standalone=True)
        baseline = advance(supervisor, 20)
        for _ in range(100):
            self.assertEqual(supervisor.snapshot(), baseline)
        self.assertEqual(len(supervisor.metrics.attempts[0].steps), 20)
        self.assertEqual(advance(supervisor).joint_confirm_steps, 21)

    def test_nonintegral_durations_and_unexplained_compatibility_fail_fast(self):
        with self.assertRaises(ValueError):
            SupervisorConfig(ready_hold_s=1.01)
        with self.assertRaises(ValueError):
            measurement(1, action_compatible=False)
        with self.assertRaises(ValueError):
            measurement(1, compatibility_blockers=("target_difference/1",))


class MetricsTests(unittest.TestCase):
    def test_checkpoint_metrics_roundtrip_preserves_denominators_and_real_step_records(self):
        supervisor = RecoverySupervisor(standalone=True)
        advance(supervisor, 50)
        supervisor.new_episode()
        advance(
            supervisor, 60, action_compatible=False,
            compatibility_blockers=("torque_difference/2",),
        )
        saved_supervisor = deepcopy(supervisor)
        saved_supervisor.interrupt("checkpoint_resume")
        saved = json.loads(json.dumps(saved_supervisor.metrics.state_dict(), allow_nan=False))
        restored = RecoveryMetrics.from_state_dict(saved)
        self.assertEqual(restored.summary(), saved_supervisor.metrics.summary())
        self.assertIsNone(supervisor.metrics.attempts[-1].observation_end_kind)
        self.assertEqual(supervisor.snapshot().attempt_steps, 60)
        resumed = RecoverySupervisor(standalone=True, metrics=restored)
        self.assertEqual(resumed.snapshot().attempt_id, 3)
        self.assertEqual(advance(resumed, 50).attempt_ended.handoff_step, 50)
        self.assertEqual(restored.summary()[0]["legal_initial_count"], 3)
        self.assertEqual(restored.summary()[0]["censored_count"], 1)

    def test_checkpoint_metrics_recompute_detects_corrupted_counters(self):
        supervisor = RecoverySupervisor(standalone=True)
        advance(supervisor, 50)
        state = supervisor.metrics.state_dict()
        state["attempts"][0]["physical_ready_step"] = 49
        with self.assertRaises(ValueError):
            RecoveryMetrics.from_state_dict(state)
        state = supervisor.metrics.state_dict()
        state["events"][0]["recovery_steps"] = 49
        with self.assertRaises(ValueError):
            RecoveryMetrics.from_state_dict(state)

    def test_delay_uses_first_physical_ready_and_includes_later_instability(self):
        supervisor = RecoverySupervisor(standalone=True)
        advance(
            supervisor, 55, action_compatible=False,
            compatibility_blockers=("target_difference/2", "torque_difference/2"),
        )
        advance(supervisor, 20)
        advance(
            supervisor, action_compatible=False, tilt_rad=0.5,
            compatibility_blockers=("target_difference/2",),
        )
        result = advance(supervisor, 50)
        attempt = result.attempt_ended
        self.assertEqual((attempt.first_physical_step, attempt.physical_ready_step), (1, 50))
        self.assertEqual((attempt.joint_ready_step, attempt.handoff_step), (126, 126))
        self.assertAlmostEqual(attempt.confirmation_delay_s, 1.52)
        self.assertEqual(attempt.physical_without_compatibility_steps, 55)
        self.assertEqual(attempt.max_physical_hold_steps, 75)
        self.assertEqual(attempt.physical_break_count, 1)
        self.assertEqual(attempt.compatibility_break_count, 1)
        self.assertEqual(attempt.steps[75].confirmation_reset_reasons, ("target_difference/2", "tilt"))
        self.assertEqual(attempt.blocker_steps["torque_difference/2"], 55)
        self.assertEqual(attempt.blocker_steps["target_difference/2"], 56)
        self.assertEqual(attempt.blocker_steps["tilt"], 1)
        json.dumps(attempt.to_dict(), allow_nan=False)

    def test_failed_and_censored_samples_remain_in_explicit_denominators(self):
        context = AttemptContext("left_side", "dynamic", "nominal", "independent")
        metrics = RecoveryMetrics()
        success = RecoverySupervisor(standalone=True, context=context, metrics=metrics)
        advance(success, 50)
        failure = RecoverySupervisor(standalone=True, context=context, metrics=metrics)
        advance(
            failure, 400, action_compatible=False,
            compatibility_blockers=("torque_difference/1",),
        )
        censored = RecoverySupervisor(standalone=True, context=context, metrics=metrics)
        advance(
            censored, 75, action_compatible=False,
            compatibility_blockers=("target_difference/3",),
        )
        censored.interrupt("checkpoint_resume")
        never_ready = RecoverySupervisor(standalone=True, context=context, metrics=metrics)
        advance(never_ready, 400, tilt_rad=1.0)
        row = metrics.summary()[0]
        self.assertEqual(row["legal_initial_count"], 4)
        self.assertEqual(row["ever_physical_count"], 3)
        self.assertEqual(row["physical_ready_count"], 3)
        self.assertEqual(row["accepted_handoff_count"], 1)
        self.assertEqual(row["task_failure_count"], 2)
        self.assertEqual(row["censored_count"], 1)
        self.assertEqual(row["success_fraction_of_all_legal"], 0.25)
        self.assertAlmostEqual(row["success_fraction_of_conclusive"], 1 / 3)
        self.assertEqual(row["delay_success_sample_count"], 1)
        self.assertEqual(row["confirmation_delay_median_s"], 0)
        self.assertEqual(row["physical_ready_without_handoff_count"], 2)
        pending = row["physical_ready_without_handoff"]
        self.assertEqual(pending[0]["observation_end_kind"], "task_failure")
        self.assertAlmostEqual(pending[0]["observed_after_physical_ready_s"], 7.0)
        self.assertEqual(pending[1]["observation_end_kind"], "censored")
        self.assertAlmostEqual(pending[1]["observed_after_physical_ready_s"], 0.5)
        self.assertIsNone(metrics.attempts[1].confirmation_delay_s)
        self.assertIsNone(metrics.attempts[2].confirmation_delay_s)
        self.assertIsNone(metrics.attempts[3].physical_ready_step)
        json.dumps(row, allow_nan=False)

    def test_metrics_groups_stage_category_motion_and_randomization(self):
        metrics = RecoveryMetrics()
        contexts = (
            AttemptContext("front", "static", "nominal", "independent"),
            AttemptContext("back", "static", "nominal", "independent"),
            AttemptContext("front", "dynamic", "nominal", "independent"),
            AttemptContext("front", "static", "randomized", "independent"),
        )
        for context in contexts:
            supervisor = RecoverySupervisor(standalone=True, context=context, metrics=metrics)
            advance(supervisor, 50)
        self.assertEqual(len(metrics.summary()), 4)
        self.assertTrue(all(row["legal_initial_count"] == 1 for row in metrics.summary()))

    def test_interrupted_zero_stand_is_event_censoring_after_completed_attempt(self):
        supervisor = RecoverySupervisor()
        advance(supervisor, manual=True)
        advance(supervisor, 50)
        advance(supervisor, 10)
        result = supervisor.interrupt("recording_interrupted")
        self.assertTrue(result.halt_requested)
        self.assertEqual(supervisor.metrics.events[0].observation_end_kind, ObservationEndKind.CENSORED)
        self.assertEqual(supervisor.metrics.attempts[0].observation_end_kind, ObservationEndKind.COMPLETED)
        self.assertEqual(supervisor.metrics.attempts[0].confirmation_delay_s, 0)


if __name__ == "__main__":
    unittest.main()
