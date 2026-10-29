from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from recovery.contracts import ARTIFACT_ROOT
from recovery.metrics import (
    AttemptContext,
    AttemptStatistics,
    CompactAttemptSteps,
    RecoveryMetrics,
    recompute_archived_aggregates,
    restore_archived_attempt,
)
from recovery.metrics_archive import MetricsArchive
from recovery.supervisor import RecoverySupervisor
from recovery.tests.test_supervisor_metrics import advance


class MetricsArchiveTests(unittest.TestCase):
    def setUp(self):
        root = ARTIFACT_ROOT / "test_work"
        root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.archive = MetricsArchive(self.root / "archive", segment_records=7)
        self.addCleanup(self.archive.close)

    def supervisor(self, stream_id="0", *, archive=None, metrics=None):
        metrics = metrics or RecoveryMetrics(archive=archive or self.archive, stream_id=stream_id)
        return RecoverySupervisor(standalone=True, metrics=metrics)

    def test_compact_steps_losslessly_reconstruct_all_step_fields(self):
        supervisor = self.supervisor()
        advance(supervisor, 20)
        advance(supervisor, action_compatible=False, tilt_rad=0.5,
                compatibility_blockers=("target/0", "torque/0"))
        advance(supervisor, 50)
        raw = next(record["payload"] for record in self.archive.iter_records() if record["kind"] == "attempt")
        restored = restore_archived_attempt(raw)
        self.assertEqual(restored.steps[20].confirmation_reset_reasons, ("target/0", "tilt", "torque/0"))
        self.assertEqual(restored.steps[20].physical_hold_steps, 0)
        self.assertEqual(restored.steps[20].joint_confirm_steps, 0)
        self.assertEqual(restored.end_step, 71)
        self.assertEqual(restored.to_dict(compact_steps=True), raw)
        self.assertEqual(len(supervisor.metrics.attempts), 0)
        self.assertEqual(len(supervisor.metrics.events), 0)

    def test_many_completed_attempts_are_archived_with_bounded_live_history(self):
        supervisor = self.supervisor()
        for episode in range(1000):
            if episode:
                supervisor.new_episode()
            advance(supervisor, 50)
            self.assertEqual(len(supervisor.metrics.attempts), 0)
            self.assertEqual(len(supervisor.metrics.events), 0)
        state = supervisor.metrics.state_dict()
        row = supervisor.metrics.summary()[0]
        self.assertEqual(row["legal_initial_count"], 1000)
        self.assertEqual(row["accepted_handoff_count"], 1000)
        self.assertEqual(row["delay_steps_histogram"], {"0": 1000})
        self.assertLess(len(json.dumps(state)), 2500)
        recompute_archived_aggregates(state, self.archive)
        records = list(self.archive.iter_records())
        self.assertEqual(len(records), 2000)
        self.assertEqual([record["sequence"] for record in records], list(range(1, 2001)))

    def test_checkpoint_censors_only_snapshot_and_strict_resume_preserves_raw_denominators(self):
        supervisor = self.supervisor()
        advance(supervisor, 50)
        supervisor.new_episode()
        advance(supervisor, 400, action_compatible=False, compatibility_blockers=("torque/1",))
        supervisor.reenable()
        advance(supervisor, 75, action_compatible=False, compatibility_blockers=("target/1",))
        before = supervisor.snapshot()
        state = supervisor.checkpoint_metrics_state(seal_archive=False)
        descriptor = self.archive.snapshot()
        self.assertEqual(supervisor.snapshot(), before)
        self.assertEqual(supervisor.metrics.summary()[0]["censored_count"], 0)
        self.assertEqual(supervisor.metrics.summary()[0]["active_count"], 1)
        recompute_archived_aggregates(state, self.archive, archive_index=descriptor)
        resumed_archive = MetricsArchive.from_index(descriptor, self.root / "resume")
        self.addCleanup(resumed_archive.close)
        metrics = RecoveryMetrics.from_state_dict(state, archive=resumed_archive)
        restored = self.supervisor(archive=resumed_archive, metrics=metrics)
        self.assertEqual(restored.snapshot().attempt_id, 4)
        advance(restored, 50)
        row = metrics.summary()[0]
        self.assertEqual(row["legal_initial_count"], 4)
        self.assertEqual(row["accepted_handoff_count"], 2)
        self.assertEqual(row["task_failure_count"], 1)
        self.assertEqual(row["censored_count"], 1)
        self.assertEqual(row["physical_ready_without_handoff_count"], 2)
        self.assertEqual(row["physical_ready_without_handoff_observation_steps_histogram"]["censored:checkpoint_resume"], {"25": 1})
        final_state = metrics.state_dict()
        recompute_archived_aggregates(final_state, resumed_archive)

    def test_unselected_earlier_checkpoint_snapshots_do_not_inflate_counts(self):
        supervisor = self.supervisor()
        advance(supervisor, 20)
        first = supervisor.checkpoint_metrics_state(seal_archive=False)
        advance(supervisor, 10)
        second = supervisor.checkpoint_metrics_state(seal_archive=False)
        descriptor = self.archive.snapshot()
        recompute_archived_aggregates(first, self.archive, archive_index=descriptor)
        recompute_archived_aggregates(second, self.archive, archive_index=descriptor)
        self.assertNotEqual(first["checkpoint_censor_sequences"], second["checkpoint_censor_sequences"])
        self.assertEqual(second["aggregates"][0]["legal_initial_count"], 1)
        self.assertEqual(second["aggregates"][0]["censored_count"], 1)

    def test_shared_archive_uses_one_global_index_and_distinguishes_environment_streams(self):
        left, right = self.supervisor("left"), self.supervisor("right")
        advance(left, 50)
        advance(right, 50)
        states = [supervisor.checkpoint_metrics_state(seal_archive=False) for supervisor in (left, right)]
        self.assertTrue(all("archive_index" not in state for state in states))
        descriptor = self.archive.snapshot()
        self.assertEqual(descriptor, self.archive.snapshot())
        self.assertEqual(len(list((self.root / "archive").glob("index_*.json"))), 1)
        self.assertEqual(len(list(self.archive.iter_records(stream_id="left"))), 2)
        self.assertEqual(len(list(self.archive.iter_records(stream_id="right"))), 2)
        for state in states:
            recompute_archived_aggregates(state, self.archive, archive_index=descriptor)

    def test_checkpoint_cutoff_excludes_later_live_completion_from_raw_recomputation(self):
        supervisor = self.supervisor()
        advance(supervisor, 20)
        state = supervisor.checkpoint_metrics_state(seal_archive=False)
        advance(supervisor, 30)
        later_index = self.archive.snapshot()
        rows = recompute_archived_aggregates(state, self.archive, archive_index=later_index)
        self.assertEqual(rows[0]["legal_initial_count"], 1)
        self.assertEqual(rows[0]["accepted_handoff_count"], 0)
        self.assertEqual(rows[0]["censored_count"], 1)
        self.assertEqual(supervisor.metrics.summary()[0]["accepted_handoff_count"], 1)

    def test_exact_delay_histograms_reproduce_median_and_nearest_rank_percentiles(self):
        supervisor = self.supervisor()
        for episode, delay in enumerate((0, 1, 3, 70)):
            if episode:
                supervisor.new_episode()
            if delay:
                advance(supervisor, delay, action_compatible=False, compatibility_blockers=("torque/0",))
            advance(supervisor, 50)
        row = supervisor.metrics.summary()[0]
        self.assertEqual(row["delay_steps_histogram"], {"0": 1, "1": 1, "3": 1, "70": 1})
        self.assertEqual(row["confirmation_delay_median_s"], 0.04)
        self.assertAlmostEqual(row["confirmation_delay_p90_s"], 1.4)
        self.assertAlmostEqual(row["confirmation_delay_p95_s"], 1.4)
        self.assertEqual(row["delay_success_sample_count"], 4)
        recompute_archived_aggregates(supervisor.metrics.state_dict(), self.archive)

    def test_modified_segment_and_modified_index_are_rejected(self):
        supervisor = self.supervisor()
        advance(supervisor, 50)
        descriptor = self.archive.snapshot()
        index = MetricsArchive.verify_index(descriptor)
        source = Path(index["segments"][0]["path"])
        with source.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            MetricsArchive.from_index(descriptor, self.root / "bad_resume")
        self.assertFalse((self.root / "bad_resume").exists())

    def test_corrupt_compact_hold_counter_is_detected_by_raw_recomputation(self):
        supervisor = self.supervisor()
        advance(supervisor, 50)
        raw = next(record["payload"] for record in self.archive.iter_records() if record["kind"] == "attempt")
        raw["steps"]["physical_holds"][30] = 1
        with self.assertRaises(ValueError):
            restore_archived_attempt(raw)

    def test_6144_active_399_step_buffers_use_at_most_60_mib_of_column_storage(self):
        attempt = AttemptStatistics(1, 1, AttemptContext(), 0.02, 50)
        for frame in range(399):
            attempt.observe(frame_id=frame, physical=True, action_compatible=False, blockers=("torque/0",))
        buffers = [deepcopy(attempt.steps) for _ in range(6144)]
        payload_bytes = sum(buffer.payload_bytes for buffer in buffers)
        allocated_array_bytes = sum(sum(sys.getsizeof(value) for value in (
            buffer.frame_ids, buffer.physical_holds, buffer.joint_holds, buffer.blocker_masks, buffer.flags
        )) for buffer in buffers)
        self.assertEqual(payload_bytes, 6144 * 399 * 21)
        self.assertLess(allocated_array_bytes, 60 * 1024 * 1024)
        self.assertTrue(all(isinstance(buffer, CompactAttemptSteps) for buffer in buffers))
        self.assertEqual(buffers[-1][-1].physical_hold_steps, 399)


if __name__ == "__main__":
    unittest.main()
