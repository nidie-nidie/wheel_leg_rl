from dataclasses import asdict
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from recovery.batched_supervisor import BatchedRecoverySupervisor, BLOCKER_BITS, REASON_CODES
from recovery.checkpoint import validate_logging_barrier
from recovery.contracts import ARTIFACT_ROOT
from recovery.metrics import AttemptContext, RecoveryMetrics, verify_archived_metrics_states
from recovery.metrics_archive import MetricsArchive
from recovery.metrics_protocol import LOG_COLUMNS, slot_budget
from recovery.metrics_stream import RecoveryMetricsStream
from recovery.supervisor import RecoverySupervisor, SupervisorMeasurements


class MetricsStreamTests(unittest.TestCase):
    def setUp(self):
        directory = ARTIFACT_ROOT / "test_work"
        directory.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        bank = json.loads((ARTIFACT_ROOT / "bootstrap_pose_bank_v1/pose_bank.json").read_text(encoding="utf-8"))
        self.records = [record for record in bank["records"] if record["split"] == "train"]
        self.context = AttemptContext(initial_category=self.records[0]["initial_category"],
            motion_kind=self.records[0]["motion_kind"], randomization_profile="RecoveryBootstrapNominalV1", stage="independent")

    def stream(self, count=3, *, restore_state=None, name="stream", **kwargs):
        stream = RecoveryMetricsStream(count, "cpu", self.root / name, {0: self.context},
                                       restore_state=restore_state, timeout_s=30, **kwargs)
        self.addCleanup(stream.abort)
        return stream

    def values(self, count):
        values = []
        for index in range(count):
            record = self.records[index % len(self.records)]
            frame = record["clean_history"][-1]
            tilt = np.arccos(np.clip(-frame[5], -1, 1))
            values.append([tilt, *frame[:3], *frame[6:10], *frame[10:16], record["metadata"]["com_height_m"]])
        return torch.tensor(values, dtype=torch.float32)

    def post(self, supervisor, frame, *, failure=None):
        count = supervisor.num_envs
        return supervisor.post_step(self.values(count), torch.full((count,), frame, dtype=torch.int64),
            wheel_support=torch.zeros(count, 2, dtype=torch.bool), body_support=torch.ones(count, dtype=torch.bool),
            action_compatible=torch.zeros(count, dtype=torch.bool),
            compatibility_mask=torch.full((count,), BLOCKER_BITS["calibration_unavailable"], dtype=torch.int64),
            physical_failure=torch.zeros(count, dtype=torch.int64) if failure is None else failure)

    def begin(self, stream, count=3):
        supervisor = BatchedRecoverySupervisor(count, "cpu", standalone=True, defer_initial_begin=True)
        stream.submit(supervisor.reset(), kind="reset")
        return supervisor

    def assert_recomputable(self, state, directory):
        archive = MetricsArchive.from_index(state["metrics_archive"], directory)
        self.addCleanup(archive.close)
        verify_archived_metrics_states(state["metrics"], archive, state["metrics_archive"])

    def test_protocol_budget_includes_worst_case_lifecycle_packets(self):
        budget = slot_budget(6144)
        self.assertEqual(len(LOG_COLUMNS), 37)
        self.assertEqual(budget["pinned_bytes"], 7274496)
        self.assertEqual(budget["pinned_allocator_rounded_budget_bytes"], 8 * 1024 * 1024)
        self.assertLess(budget["pinned_bytes"], 16 * 1024 * 1024)
        with self.assertRaises(ValueError):
            slot_budget(100000)

    def test_partial_initialization_failure_closes_all_allocated_shared_slots(self):
        stream = RecoveryMetricsStream.__new__(RecoveryMetricsStream)
        with self.assertRaises(KeyError):
            stream.__init__(3, "cpu", self.root / "failed_initialization", {0: self.context}, restore_state={})
        self.assertTrue(stream._closed)
        self.assertIsNone(stream._process)
        for slot in stream._slots:
            self.assertIsNone(slot["host"])
            self.assertIsNone(slot["array"])
            self.assertIsNone(slot["shared"]._mmap)

    def test_actual_worker_checkpoint_copy_censor_and_later_completion(self):
        stream = self.stream()
        supervisor = self.begin(stream)
        for frame in range(1, 21):
            stream.submit(self.post(supervisor, frame))
        checkpoint = stream.checkpoint(torch.full((3,), 20, dtype=torch.int64))
        self.assertEqual(checkpoint["metrics"][0]["aggregates"][0]["censored_count"], 1)
        self.assertTrue(bool(supervisor.attempt_active.all()))
        self.assert_recomputable(checkpoint, self.root / "checkpoint_verify")
        for frame in range(21, 25):
            stream.submit(self.post(supervisor, frame))
        final = stream.close(torch.full((3,), 24, dtype=torch.int64))
        self.assertEqual(final["metrics"][0]["aggregates"][0]["legal_initial_count"], 1)
        self.assertEqual(final["metrics"][0]["aggregates"][0]["censored_count"], 1)
        self.assertEqual(final["logging_barrier"]["pending_packets"], 0)
        self.assert_recomputable(final, self.root / "final_verify")
        validate_logging_barrier({**final, "frame_ids": torch.full((3,), 24, dtype=torch.int64)})
        for key, bad_value in (("pending_packets", 1), ("consumed_packet_sequence", 0), ("archive_last_sequence", -1)):
            corrupted = deepcopy(final)
            corrupted["logging_barrier"][key] = bad_value
            with self.assertRaises(ValueError):
                validate_logging_barrier({**corrupted, "frame_ids": torch.full((3,), 24, dtype=torch.int64)})
        with self.assertRaises(ValueError):
            validate_logging_barrier({**final, "frame_ids": torch.full((3,), 25, dtype=torch.int64)})

    def test_backpressure_all_environments_end_and_restart_every_step(self):
        stream = self.stream(segment_records=7)
        supervisor = self.begin(stream)
        for frame in range(1, 17):
            stream.submit(self.post(supervisor, frame, failure=torch.full((3,), REASON_CODES["loop_closure"], dtype=torch.int64)))
            stream.submit(supervisor.reset(frame_ids=torch.full((3,), frame, dtype=torch.int64)), kind="reset")
        final = stream.close(torch.full((3,), 16, dtype=torch.int64))
        row = final["metrics"][0]["aggregates"][0]
        self.assertEqual(row["legal_initial_count"], 17)
        self.assertEqual(row["task_failure_count"], 16)
        self.assertEqual(row["censored_count"], 1)
        self.assertLessEqual(stream.statistics["maximum_owned_slots"], 2)
        self.assertEqual(stream.statistics["submitted_packets"], 33)
        self.assert_recomputable(final, self.root / "dense_verify")

    def test_worker_exit_and_real_disk_path_failure_propagate(self):
        stream = self.stream(name="exit")
        stream._process.terminate()
        stream._process.wait(timeout=5)
        with self.assertRaises(RuntimeError):
            stream.poll()
        disk = self.stream(name="disk")
        (disk.directory / "archive/tensor_packets").write_text("blocking file", encoding="utf-8")
        supervisor = self.begin(disk)
        disk.submit(self.post(supervisor, 1))
        with self.assertRaises((RuntimeError, EOFError, ConnectionError)):
            disk.flush()

    def test_source_binary_packets_and_hashes_are_preserved(self):
        stream = self.stream(count=1)
        supervisor = self.begin(stream, 1)
        stream.submit(self.post(supervisor, 1))
        result = stream.close(torch.tensor([1]))
        index = MetricsArchive.verify_index(result["metrics_archive"])
        self.assertEqual(len(index["sidecars"]), 1)
        sidecar = index["sidecars"][0]
        with gzip.open(sidecar["path"], "rb") as source:
            packets = np.load(source, allow_pickle=False)
        self.assertEqual(packets.shape, (2, 1, 37))
        self.assertEqual(int(packets[-1, 0, LOG_COLUMNS.index("frame_id")]), 1)
        with Path(sidecar["path"]).open("ab") as source:
            source.write(b"changed")
        with self.assertRaises(ValueError):
            MetricsArchive.verify_index(result["metrics_archive"])

    def test_scalar_statistics_and_all_raw_steps_match_worker(self):
        stream = self.stream(count=1)
        supervisor = self.begin(stream, 1)
        scalar_archive = MetricsArchive(self.root / "scalar", segment_records=7)
        self.addCleanup(scalar_archive.close)
        metric = RecoveryMetrics(archive=scalar_archive)
        scalar = RecoverySupervisor(standalone=True, context=self.context, metrics=metric)
        values = self.values(1)[0].tolist()
        for frame in range(1, 81):
            stream.submit(self.post(supervisor, frame))
            scalar.post_step(SupervisorMeasurements(frame_id=frame, tilt_rad=values[0], angular_velocity=tuple(values[1:4]),
                leg_position_error=tuple(values[4:8]), leg_velocity=tuple(values[8:12]), wheel_velocity=tuple(values[12:14]),
                com_height_m=values[14], left_wheel_support=False, right_wheel_support=False, body_or_leg_support=True,
                action_compatible=False, compatibility_blockers=("calibration_unavailable",)))
        state = stream.checkpoint(torch.tensor([80]))
        expected = scalar.checkpoint_metrics_state(seal_archive=False)
        self.assertEqual(state["metrics"][0]["aggregates"], expected["aggregates"])
        worker_archive = MetricsArchive.from_index(state["metrics_archive"], self.root / "raw_compare")
        self.addCleanup(worker_archive.close)
        raw = next(item["payload"] for item in worker_archive.iter_records(stream_id="0") if item["kind"] == "checkpoint_attempt")
        expected_raw = next(item["payload"] for item in scalar_archive.iter_records(stream_id="0") if item["kind"] == "checkpoint_attempt")
        self.assertEqual(raw, expected_raw)
        stream.close(torch.tensor([80]))

    def test_earlier_snapshot_restore_retains_cutoff_and_new_denominator_once(self):
        stream = self.stream(count=1)
        supervisor = self.begin(stream, 1)
        for frame in range(1, 5):
            stream.submit(self.post(supervisor, frame))
        earlier = stream.checkpoint(torch.tensor([4]))
        for frame in range(5, 9):
            stream.submit(self.post(supervisor, frame))
        stream.close(torch.tensor([8]))
        restored = self.stream(count=1, restore_state={**earlier, "frame_ids": torch.tensor([4])}, name="restored")
        ids = restored.next_ids()
        restarted = BatchedRecoverySupervisor(1, "cpu", standalone=True, defer_initial_begin=True,
            next_event_ids=torch.tensor(ids["next_event_id"]), next_attempt_ids=torch.tensor(ids["next_attempt_id"]))
        restored.submit(restarted.reset(frame_ids=torch.tensor([4])), kind="reset")
        restored.submit(self.post(restarted, 5))
        final = restored.close(torch.tensor([5]))
        rows = final["metrics"][0]["aggregates"]
        self.assertEqual(rows[0]["legal_initial_count"], 2)
        self.assertEqual(rows[0]["censored_count"], 2)
        self.assert_recomputable(final, self.root / "restored_verify")


if __name__ == "__main__":
    unittest.main()
