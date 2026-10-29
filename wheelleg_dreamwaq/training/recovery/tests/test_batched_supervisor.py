import math
import unittest
from dataclasses import replace

import torch

from recovery.batched_supervisor import (
    BLOCKER_BITS, BLOCKER_LABELS, REASON_CODES, REASON_LABELS,
    BatchedRecoverySupervisor, EndKind,
)
from recovery.metrics import ObservationEndKind
from recovery.supervisor import (
    ControlMode, RecoverySupervisor, SupervisorConfig, SupervisorMeasurements,
    physical_conditions,
)


_KINDS = {None: 0, ObservationEndKind.COMPLETED: 1,
          ObservationEndKind.TASK_FAILURE: 2, ObservationEndKind.CENSORED: 3}
_TRANSITIONS = {None: 0, "LOCOMOTION->RECOVERY": 1, "RECOVERY->LOCOMOTION": 2}


class OracleRun:
    def __init__(self, case, count=3, *, device="cpu", standalone=False,
                 initial=True, config=SupervisorConfig(), dtype=torch.float64):
        self.case, self.count, self.device, self.dtype = case, count, device, dtype
        self.scalar = [RecoverySupervisor(config, standalone=standalone, initial_recovery=initial)
                       for _ in range(count)]
        self.batch = BatchedRecoverySupervisor(count, device, config,
                                              standalone=standalone, initial_recovery=initial)
        self.frames = torch.zeros(count, dtype=torch.int64, device=device)

    def values(self):
        result = torch.zeros((self.count, 15), dtype=self.dtype, device=self.device)
        result[:, 14] = 0.2
        return result

    def step(self, values=None, *, manual=False, blockers=None, failures=None,
             wheels=None, body=None, finite=None):
        values = self.values() if values is None else values
        n, device = self.count, self.device
        integer = lambda: torch.zeros(n, dtype=torch.int64, device=device)
        boolean = lambda value: torch.full((n,), value, dtype=torch.bool, device=device)
        blockers = integer() if blockers is None else blockers
        failures = integer() if failures is None else failures
        wheels = torch.ones((n, 2), dtype=torch.bool, device=device) if wheels is None else wheels
        body = boolean(False) if body is None else body
        finite = boolean(True) if finite is None else finite
        manual = boolean(manual) if isinstance(manual, bool) else manual
        self.frames += 1
        update = self.batch.post_step(values, self.frames, wheel_support=wheels,
                                     body_support=body, action_compatible=blockers == 0,
                                     compatibility_mask=blockers, physical_failure=failures,
                                     finite=finite, manual_recovery=manual)
        rows = values.cpu().tolist()
        input_rows = torch.stack((self.frames, blockers, failures, wheels[:, 0], wheels[:, 1],
                                  body, finite, manual), dim=1).cpu().tolist()
        outputs = []
        for index, (row, flags) in enumerate(zip(rows, input_rows)):
            frame, blocker, failure, left, right, support, isfinite, requested = flags
            labels = tuple(label for bit, label in enumerate(BLOCKER_LABELS) if blocker & (1 << bit))
            measurement = SupervisorMeasurements(
                frame, row[0], tuple(row[1:4]), tuple(row[4:8]), tuple(row[8:12]),
                tuple(row[12:14]), row[14], bool(left), bool(right), bool(support),
                not blocker, labels, REASON_LABELS[failure] if failure else None, bool(isfinite))
            supervisor = self.scalar[index]
            previous_attempt, previous_event = supervisor.attempt, supervisor.event
            previous_mode = supervisor.mode
            out = supervisor.post_step(measurement, manual_recovery=bool(requested))
            outputs.append(out)
            event = supervisor.event or supervisor.metrics.last_event
            attempted = previous_mode is ControlMode.RECOVERY
            attempt = supervisor.attempt or out.attempt_ended
            expected = {
                "frame_id": out.frame_id, "mode": int(out.mode is ControlMode.RECOVERY),
                "transition": _TRANSITIONS[out.transition],
                "fault_reason": REASON_CODES[out.fault_reason] if out.fault_reason else 0,
                "halt_requested": out.halt_requested, "capture_heading": out.capture_heading,
                "zero_command": out.zero_command, "completed": supervisor.completed,
                "success": out.attempt_ended is not None and out.attempt_ended.handoff_step is not None,
                "failure": out.fault_reason is not None,
                "event_active": supervisor.event is not None,
                "attempt_active": supervisor.attempt is not None,
                "event_id": out.event_id if out.event_id is not None else -1,
                "attempt_id": out.attempt_id if out.attempt_id is not None else -1,
                "event_steps": out.event_steps, "recovery_steps": out.recovery_steps,
                "attempt_steps": out.attempt_steps, "physical_hold_steps": out.physical_hold_steps,
                "joint_confirm_steps": out.joint_confirm_steps,
                "zero_stand_hold_steps": out.zero_stand_hold_steps,
                "fall_hold_steps": supervisor.fall_hold_steps,
                "zero_stand_break_count": event.zero_stand_break_count if event else 0,
                "recurrent_fall_count": event.recurrent_fall_count if event else 0,
                "attempt_recorded": attempted,
                "attempt_started": supervisor.attempt is not None and supervisor.attempt is not previous_attempt,
                "event_started": supervisor.event is not None and supervisor.event is not previous_event,
                "attempt_ended": out.attempt_ended is not None,
                "event_finished": out.event_finished,
                "attempt_end_reason": REASON_CODES[out.attempt_ended.end_reason] if out.attempt_ended else 0,
                "attempt_end_kind": _KINDS[out.attempt_ended.observation_end_kind] if out.attempt_ended else 0,
                "event_end_reason": REASON_CODES[event.end_reason] if out.event_finished else 0,
                "event_end_kind": _KINDS[event.observation_end_kind] if out.event_finished else 0,
            }
            failure_label = supervisor._measurement_failure(measurement)
            expected["physical"] = all(physical_conditions(measurement, supervisor.config).values()) and failure_label is None
            expected["action_compatible"] = not blocker
            if attempted:
                expected["blocker_mask"] = sum(BLOCKER_BITS[label] for label in attempt.steps[-1].blockers)
            columns = list(expected)
            actual = torch.stack([getattr(update, name)[index].to(torch.int64) for name in columns]).cpu().tolist()
            for name, value in zip(columns, actual):
                self.case.assertEqual(value, expected[name], f"frame={frame} env={index} field={name}")
        return update, outputs


class BatchedSupervisorTests(unittest.TestCase):
    def test_thresholds_match_original_promoted_samples(self):
        for dtype in (torch.float32, torch.float64):
            scenarios = [(0, .25), (4, .2), (5, -.2), (8, 1.), (11, -1.),
                         (12, 2.), (13, -2.), (14, .17), (14, .25), (0, .9)]
            for column, threshold in scenarios:
                with self.subTest(dtype=dtype, column=column, threshold=threshold):
                    run = OracleRun(self, dtype=dtype)
                    values = run.values()
                    center = torch.tensor(threshold, dtype=dtype)
                    values[:, column] = torch.tensor([
                        torch.nextafter(center, torch.tensor(-math.inf, dtype=dtype)),
                        center, torch.nextafter(center, torch.tensor(math.inf, dtype=dtype))], dtype=dtype)
                    run.step(values)
            run = OracleRun(self, dtype=dtype)
            values = run.values()
            values[:, 1:4] = torch.tensor([[.3, .4, 0.], [.5, 0., 0.], [.3, .4, 1e-8]], dtype=dtype)
            run.step(values)

    def test_confirmation_compatibility_and_all_failure_precedence(self):
        run = OracleRun(self)
        for _ in range(49):
            run.step()
        blockers = torch.tensor([0, BLOCKER_BITS["target_2"], BLOCKER_BITS["torque_5"]])
        run.step(blockers=blockers)
        # 第50帧确认与物理失败同帧时，原监督器优先记录失败。
        for code in (1, 2, 3, 4, 5, 15):
            isolated = OracleRun(self, count=1, standalone=True)
            for _ in range(49):
                isolated.step()
            out, _ = isolated.step(failures=torch.tensor([code]))
            self.assertTrue(bool(out.failure[0]))
            self.assertFalse(bool(out.success[0]))

    def test_400_boundary_ready_then_recurrent_fall_exhausts_budget(self):
        for standalone in (False, True):
            run = OracleRun(self, count=1, standalone=standalone)
            values = run.values()
            values[:, 0] = 1.
            for _ in range(350):
                run.step(values)
            for _ in range(50):
                out, _ = run.step()
            self.assertTrue(bool(out.success[0]))
            self.assertEqual(int(out.recovery_steps[0]), 400)
            if not standalone:
                out, _ = run.step(manual=True)
                self.assertEqual(int(out.fault_reason[0]), REASON_CODES["recovery_budget_exhausted_on_fall"])

    def test_zero_completion_wins_at_500_and_event_deadline_wins_handoff(self):
        run = OracleRun(self, count=1)
        values = run.values()
        values[:, 0] = 1.
        for _ in range(350):
            run.step(values)
        for _ in range(50):
            run.step()
        for _ in range(99):
            run.step()
        out, _ = run.step(manual=True)
        self.assertEqual(int(out.event_end_reason[0]), REASON_CODES["zero_stand_complete"])
        self.assertEqual(int(out.event_steps[0]), 500)
        cfg = replace(SupervisorConfig(), recovery_budget_s=10.)
        run = OracleRun(self, count=1, config=cfg)
        values = run.values()
        values[:, 0] = 1.
        for _ in range(450):
            run.step(values)
        for _ in range(50):
            out, _ = run.step()
        self.assertEqual(int(out.fault_reason[0]), REASON_CODES["event_deadline"])

    def test_fall_hold_recurrence_zero_break_and_context(self):
        run = OracleRun(self, initial=False)
        values = run.values()
        values[:, 0] = 1.
        for _ in range(9):
            run.step(values)
        run.step(values)
        for _ in range(50):
            run.step()
        run.step()
        run.step(values)
        for _ in range(9):
            out, _ = run.step(values)
        self.assertEqual(out.recurrent_fall_count.tolist(), [1, 1, 1])
        self.assertEqual(out.zero_stand_break_count.tolist(), [1, 1, 1])
        self.assertEqual(out.attempt_id.tolist(), [2, 2, 2])

    def test_nonfinite_and_frame_validation_are_atomic(self):
        run = OracleRun(self)
        values = run.values()
        values[0, 1] = math.nan
        values[1, 14] = math.inf
        run.step(values)
        batch = BatchedRecoverySupervisor(3, initial_recovery=True)
        before = batch.state_dict()
        values = torch.zeros((3, 15), dtype=torch.float64)
        kwargs = dict(wheel_support=torch.ones((3, 2), dtype=torch.bool),
                      body_support=torch.zeros(3, dtype=torch.bool),
                      action_compatible=torch.ones(3, dtype=torch.bool),
                      compatibility_mask=torch.zeros(3, dtype=torch.int64),
                      physical_failure=torch.zeros(3, dtype=torch.int64))
        with self.assertRaises(ValueError):
            batch.post_step(values, torch.tensor([0, -1, 0]), **kwargs)
        for key, value in before["state"].items():
            torch.testing.assert_close(value, batch.state_dict()["state"][key])
        batch.post_step(values, torch.tensor([0, 0, 0]), **kwargs)
        with self.assertRaises(ValueError):
            batch.post_step(values, torch.tensor([1, 0, 1]), **kwargs)

    def test_sparse_reset_interrupt_checkpoint_and_ids(self):
        batch = BatchedRecoverySupervisor(3, standalone=True, defer_initial_begin=True)
        self.assertFalse(bool(batch.initial_update.attempt_started.any()))
        contexts = torch.tensor([12, 13, 14])
        out = batch.reset(context_ids=contexts, next_event_ids=torch.tensor([3, 4, 5]),
                          next_attempt_ids=torch.tensor([7, 8, 9]))
        self.assertEqual(out.attempt_id.tolist(), [7, 8, 9])
        self.assertEqual(out.context_id.tolist(), [12, 13, 14])
        with self.assertRaises(RuntimeError):
            batch.reset(torch.tensor([1]))
        old = batch.interrupt("manual_reset", env_ids=torch.tensor([1]))
        self.assertEqual(old.attempt_end_kind.tolist(), [0, int(EndKind.CENSORED), 0])
        new = batch.reset(torch.tensor([1]))
        self.assertEqual(new.generation.tolist(), [1, 2, 1])
        self.assertEqual(new.attempt_started.tolist(), [False, True, False])
        self.assertEqual(new.attempt_id.tolist(), [7, 9, 9])
        restored = BatchedRecoverySupervisor(3, standalone=True, defer_initial_begin=True)
        restored.load_state_dict(batch.state_dict())
        for key, value in batch.snapshot().columns().items():
            torch.testing.assert_close(value, restored.snapshot().columns()[key])
        ended = batch.interrupt("training_run_end")
        self.assertEqual(ended.attempt_end_reason.tolist(), [14, 14, 14])
        resumed = batch.reenable()
        self.assertEqual(resumed.attempt_id.tolist(), [8, 10, 10])

    def test_admin_operations_follow_scalar_frame_and_id_semantics(self):
        run = OracleRun(self, standalone=True)
        for _ in range(50):
            run.step()
        out = run.batch.new_episode()
        for scalar in run.scalar:
            scalar.new_episode()
        self.assertEqual(out.attempt_id.tolist(), [2, 2, 2])
        self.assertEqual(out.frame_id.tolist(), [50, 50, 50])
        run.step()
        ids = torch.tensor([1])
        old = run.batch.interrupt(env_ids=ids)
        scalar_old = run.scalar[1].interrupt()
        self.assertEqual(int(old.attempt_end_reason[1]), REASON_CODES[scalar_old.attempt_ended.end_reason])
        run.batch.reenable(env_ids=ids)
        run.scalar[1].reenable()
        run.step()
        run.batch.interrupt("manual_reset", env_ids=ids)
        run.scalar[1].interrupt("manual_reset")
        old_metrics = run.scalar[1].metrics
        run.scalar[1] = RecoverySupervisor(standalone=True, metrics=old_metrics)
        out = run.batch.reset(ids, frame_ids=run.frames)
        self.assertEqual(int(out.frame_id[1]), 52)
        self.assertEqual(int(out.generation[1]), 1)
        run.step()
        combined = OracleRun(self, count=1)
        combined.step()
        combined.batch.interrupt()
        combined.scalar[0].interrupt()
        out = combined.batch.reenable()
        combined.scalar[0].reenable()
        self.assertEqual(out.mode.tolist(), [0])
        self.assertFalse(bool(out.attempt_started[0]))
        combined.step(manual=True)

    def test_deadline_failure_and_checkpoint_state_are_strict(self):
        run = OracleRun(self, count=1)
        values = run.values()
        values[:, 0] = 1.
        for _ in range(400):
            out, _ = run.step(values)
        self.assertEqual(int(out.fault_reason[0]), REASON_CODES["recovery_deadline"])
        with self.assertRaises(RuntimeError):
            run.step(values)
        batch = BatchedRecoverySupervisor(1, initial_recovery=True)
        valid = batch.state_dict()
        invalid = batch.state_dict()
        invalid["state"]["joint_confirm_steps"][0] = 1
        with self.assertRaises(ValueError):
            batch.load_state_dict(invalid)
        for key, value in valid["state"].items():
            torch.testing.assert_close(value, batch.state_dict()["state"][key])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_gpu_float64_boundaries_and_float32_promotion(self):
        for dtype in (torch.float32, torch.float64):
            for column, threshold in ((0, .25), (4, .2), (8, 1.), (12, 2.), (14, .17), (14, .25)):
                run = OracleRun(self, dtype=dtype, device="cuda:0")
                values = run.values()
                center = torch.tensor(threshold, device="cuda:0", dtype=dtype)
                values[:, column] = torch.stack((torch.nextafter(center, center.new_tensor(-math.inf)),
                                                center, torch.nextafter(center, center.new_tensor(math.inf))))
                run.step(values)
            run = OracleRun(self, dtype=dtype, device="cuda:0")
            values = run.values()
            center = torch.tensor(.5, device="cuda:0", dtype=dtype)
            values[:, 1] = torch.stack((torch.nextafter(center, center.new_tensor(-math.inf)),
                                       center, torch.nextafter(center, center.new_tensor(math.inf))))
            run.step(values)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_gpu_6144_mixed_rows_follow_six_real_oracles(self):
        # 使用真实标量监督器处理相同观测序列，只检验逻辑，不宣称物理能力。
        n, device = 6144, "cuda:0"
        group = torch.arange(n, device=device) % 6
        standalone = group < 3
        batch = BatchedRecoverySupervisor(n, device, standalone=standalone, initial_recovery=True)
        oracles = [OracleRun(self, count=1, standalone=index < 3) for index in range(6)]
        for frame in range(1, 152):
            values = torch.zeros((n, 15), device=device)
            values[:, 14] = .2
            masks = torch.zeros(n, dtype=torch.int64, device=device)
            values[group == 1, 0] = 1.
            values[group == 4, 0] = 1. if frame <= 10 or 62 <= frame <= 71 else 0.
            masks[group == 2] = BLOCKER_BITS["calibration_unavailable"]
            masks[group == 5] = BLOCKER_BITS["torque_0"] if frame % 3 == 0 else 0
            out = batch.post_step(values, torch.full((n,), frame, device=device, dtype=torch.int64),
                                  wheel_support=torch.ones((n, 2), dtype=torch.bool, device=device),
                                  body_support=torch.zeros(n, dtype=torch.bool, device=device),
                                  action_compatible=masks == 0, compatibility_mask=masks,
                                  physical_failure=torch.zeros(n, dtype=torch.int64, device=device))
            prototype_values = values[:6].cpu()
            prototype_masks = masks[:6].cpu()
            expected = []
            for index, oracle in enumerate(oracles):
                scalar_out, _ = oracle.step(prototype_values[index:index + 1], blockers=prototype_masks[index:index + 1])
                expected.append(scalar_out)
            for name in ("mode", "event_steps", "recovery_steps", "attempt_steps", "physical_hold_steps",
                         "joint_confirm_steps", "zero_stand_hold_steps", "zero_stand_break_count",
                         "recurrent_fall_count", "attempt_ended", "event_finished", "fault_reason"):
                pattern = torch.cat([getattr(item, name) for item in expected]).to(device)
                torch.testing.assert_close(getattr(out, name), pattern[group], rtol=0, atol=0)
            done = torch.nonzero(out.completed | out.halt_requested, as_tuple=False).flatten()
            if done.numel():
                batch.reset(done)
                for index, oracle in enumerate(oracles):
                    if bool(expected[index].completed[0]) or bool(expected[index].halt_requested[0]):
                        old = oracle.scalar[0]
                        oracle.scalar[0] = RecoverySupervisor(standalone=index < 3, initial_recovery=True, metrics=old.metrics)
                        oracle.batch.reset()


if __name__ == "__main__":
    unittest.main()
