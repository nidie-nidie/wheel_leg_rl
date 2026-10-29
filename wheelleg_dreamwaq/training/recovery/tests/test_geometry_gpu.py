from copy import deepcopy
import json
from multiprocessing import shared_memory
from pathlib import Path
import subprocess
from time import time_ns

import numpy as np
import pytest
import torch

from recovery.geometry_binary import CAPACITY, SLOT_BYTES, GeometryBinaryClient
from recovery.geometry_gpu import GeometryGpuGuard, PlaneSupportCalculator
from recovery.geometry_guard import GeometryGuardClient
from recovery.geometry_archive import verify_geometry_barrier


ROOT = Path(__file__).resolve().parents[3]
MUJOCO_PYTHON = ROOT / "sim2sim/mujoco/.venv/Scripts/python.exe"


@pytest.fixture(scope="module")
def geometry():
    directory = ROOT / "recovery/artifacts/geometry_gpu_tests" / str(time_ns())
    command = [str(MUJOCO_PYTHON), "-m", "recovery.tests.geometry_reference_fixture", "--directory", str(directory)]
    bank = ROOT / "recovery/artifacts/bootstrap_pose_bank_v1/pose_bank.json"
    if bank.is_file():
        command += ["--bank", str(bank)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    metadata = json.loads((directory / "reference.json").read_text(encoding="utf-8"))
    with np.load(directory / "reference.npz", allow_pickle=False) as arrays:
        metadata.update({name: arrays[name].copy() for name in arrays.files})
    return directory, metadata


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_same_original_float32_body_inputs_match_production_reference(geometry, device):
    directory, source = geometry
    guard = GeometryGpuGuard(source["config"], directory / device.replace(":", "_"), device, source["body_names"])
    try:
        p, q = torch.tensor(source["positions"], device=device), torch.tensor(source["quaternions"], device=device)
        result = guard.inspect(source["body_names"], p, q)
        actual = result.maximum_ground_penetration_m.cpu().numpy()
        error = result.absolute_distance_error_bound_m.cpu().numpy()
        assert np.all(actual >= source["native_penetration"])
        assert np.all(actual - source["native_penetration"] <= 2 * error)
        np.testing.assert_array_equal(result.ground_failure.cpu(), source["native_penetration"] > .005)
        barrier = guard.flush()
        assert barrier["barrier"]["all_requests_acknowledged"]
        assert barrier["statistics"]["internal_clearance_inspected_states"] == min(32, len(p))
        assert guard.transport.statistics["maximum_pending"] <= 2
    finally:
        guard.close()


def test_full_6144_critical_fallback_is_bounded_and_does_not_advance_diagnostic_twice(geometry):
    directory, source = geometry
    guard = GeometryGpuGuard(source["config"], directory / "critical_6144", "cuda:0", source["body_names"])
    try:
        p = torch.tensor(source["positions"][:1], device="cuda:0")
        q = torch.tensor(source["quaternions"][:1], device="cuda:0")
        signed, _ = guard.calculator.calculate(p, q, .05)
        p[:, :, 2] += -.005 - signed.min()
        p, q = p.repeat(6144, 1, 1), q.repeat(6144, 1, 1)
        result = guard.inspect(source["body_names"], p, q, frame_id=7, substep_id=3)
        assert result.fallback_env_count == 6144
        assert guard.fallback_batches == 6144 // CAPACITY
        assert guard.diagnostic_offset == 32
        final = guard.flush()
        assert final["statistics"]["internal_clearance_inspected_states"] == 32
        assert final["statistics"]["transport"]["shared_slot_bytes"] == SLOT_BYTES
        assert final["statistics"]["transport"]["maximum_pending"] <= 2
    finally:
        guard.close()


def test_unsampled_dummy_invalid_input_is_rejected(geometry):
    directory, source = geometry
    guard = GeometryGpuGuard(source["config"], directory / "invalid", "cuda:0", source["body_names"])
    try:
        p = torch.tensor(source["positions"][:1], device="cuda:0").repeat(33, 1, 1)
        q = torch.tensor(source["quaternions"][:1], device="cuda:0").repeat(33, 1, 1)
        p[32, -1, 0] = float("nan")
        with pytest.raises(ValueError, match="全部真实body"):
            guard.inspect(source["body_names"], p, q)
        p[32, -1, 0] = 0
        q[32, -1] *= 1.001
        with pytest.raises(ValueError, match="全部真实body"):
            guard.inspect(source["body_names"], p, q)
    finally:
        guard.close()


def test_possible_violation_always_uses_native_reference(geometry):
    directory, source = geometry
    guard = GeometryGpuGuard(source["config"], directory / "native_violation", "cuda:0", source["body_names"])
    reference = GeometryGuardClient(source["config"], directory / "native_violation_reference")
    try:
        p = torch.tensor(source["positions"][:1], device="cuda:0").repeat(257, 1, 1)
        q = torch.tensor(source["quaternions"][:1], device="cuda:0").repeat(257, 1, 1)
        p[:, :, 2] -= 1
        result = guard.inspect(source["body_names"], p, q)
        expected = reference.inspect(source["body_names"], p.cpu().numpy(), q.cpu().numpy())
        assert result.fallback_env_count == 257
        assert guard.fallback_batches == 2
        np.testing.assert_array_equal(result.maximum_ground_penetration_m.cpu(),
            [row["maximum_ground_penetration_m"] for row in expected])
        assert result.ground_failure.all()
        assert guard.diagnostic_offset == 32
        guard.flush()
    finally:
        guard.close()
        reference.close()


def test_binary_backpressure_ack_and_worker_exit(geometry):
    directory, source = geometry
    client = GeometryBinaryClient(source["config"], directory / "binary")
    try:
        tickets = [client.submit("floor", source["body_names"], source["positions"][:1], source["quaternions"][:1]) for _ in range(2)]
        with pytest.raises(BufferError, match="双槽位"):
            client.submit("floor", source["body_names"], source["positions"][:1], source["quaternions"][:1])
        for ticket in tickets:
            assert client.collect(ticket)["values"][0]["penetration"] == source["native_penetration"][0]
        client.process.terminate()
        client.process.wait(timeout=10)
        with pytest.raises(RuntimeError, match="worker"):
            client.check_health()
    finally:
        client.close()


def test_strict_checks_entire_frozen_input_and_diagnostic_rotation(geometry):
    directory, source = geometry
    strict = deepcopy(source["config"])
    strict["internal_clearance_mode"] = "strict"
    guard = GeometryGpuGuard(strict, directory / "strict", "cuda:0", source["body_names"])
    try:
        p = torch.tensor(source["positions"][:1], device="cuda:0").repeat(257, 1, 1)
        q = torch.tensor(source["quaternions"][:1], device="cuda:0").repeat(257, 1, 1)
        result = guard.inspect(source["body_names"], p, q)
        assert result.internal_failure.all()
        assert guard.transport.statistics["completed_requests"] == 2
        assert guard.diagnostic_offset == 0
        assert guard.flush()["statistics"]["internal_clearance_inspected_states"] == 257
    finally:
        guard.close()


def test_threshold_neighbors_and_round_robin_archive(geometry):
    directory, source = geometry
    guard = GeometryGpuGuard(source["config"], directory / "neighbors", "cuda:0", source["body_names"])
    reference = GeometryGuardClient(source["config"], directory / "neighbors_reference")
    try:
        p = torch.tensor(source["positions"][:1], device="cuda:0").repeat(33, 1, 1)
        q = torch.tensor(source["quaternions"][:1], device="cuda:0").repeat(33, 1, 1)
        q *= 1.000001
        signed, _ = guard.calculator.calculate(p, q, .05)
        p[:, :, 2] += -.005 - signed.amin(dim=1, keepdim=True).float()
        p[0, :, 2] = p[0, :, 2].nextafter(torch.full_like(p[0, :, 2], -float("inf")))
        p[2, :, 2] = p[2, :, 2].nextafter(torch.full_like(p[2, :, 2], float("inf")))
        previous_barrier = None
        for frame in range(3):
            result = guard.inspect(source["body_names"], p, q, env_ids=torch.arange(33, device="cuda:0") + 100,
                                   generations=torch.arange(33, device="cuda:0") + 200, frame_id=frame)
            expected = reference.inspect(source["body_names"], p.cpu().numpy(), q.cpu().numpy())
            assert result.ground_failure.cpu().tolist() == [row["maximum_ground_penetration_m"] > .005 for row in expected]
            final = guard.flush()
            if previous_barrier is not None:
                verify_geometry_barrier(previous_barrier)
            previous_barrier = final["barrier"]
            records = final["diagnostics"]
            assert len(records) == 1
            selected = [(frame * 32 + index) % 33 for index in range(32)]
            assert records[0]["env_ids"] == [index + 100 for index in selected]
            assert records[0]["generations"] == [index + 200 for index in selected]
            assert records[0]["frame_id"] == frame
            assert records[0]["population_count"] == 33
        with guard.archive_path.open("r+b") as stream:
            stream.seek(0)
            stream.write(b"X")
        with pytest.raises(ValueError, match="哈希"):
            verify_geometry_barrier(previous_barrier)
    finally:
        guard.close()
        reference.close()


def test_compiled_identity_is_stable_across_run_directories(geometry):
    directory, source = geometry
    first = GeometryBinaryClient(source["config"], directory / "identity_one")
    second = GeometryBinaryClient(source["config"], directory / "identity_two")
    try:
        assert first.constants["identity_sha256"] == second.constants["identity_sha256"]
    finally:
        first.close()
        second.close()


def test_sixty_real_bank_reset_requests_drain_without_control_steps(geometry):
    directory, source = geometry
    bank_indices = [index for index, label in enumerate(source["labels"])
                    if label != "nominal" and not label.startswith("physical:")]
    assert len(bank_indices) == 60
    guard = GeometryGpuGuard(source["config"], directory / "sixty_resets", "cuda:0", source["body_names"])
    try:
        p = torch.tensor(source["positions"][bank_indices], device="cuda:0")
        q = torch.tensor(source["quaternions"][bank_indices], device="cuda:0")
        env_ids = torch.arange(4, device="cuda:0")
        for generation in range(60):
            result = guard.inspect(source["body_names"], p[generation:generation + 1].repeat(4, 1, 1),
                q[generation:generation + 1].repeat(4, 1, 1), env_ids=env_ids,
                generations=torch.full((4,), generation, device="cuda:0"),
                frame_id=0, substep_id=0, restoring=True)
            assert not result.ground_failure.any()
            guard.check_health()
            assert len(guard.completed) <= 32
        final = guard.flush()
        verify_geometry_barrier(final["barrier"])
        records = [json.loads(line) for line in guard.archive_path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 60
        assert len({record["sequence"] for record in records}) == 60
        for generation, record in enumerate(records):
            assert record["restoring"] is True
            assert record["frame_id"] == record["substep_id"] == 0
            assert record["env_ids"] == list(range(4))
            assert record["generations"] == [generation] * 4
        assert final["statistics"]["inspected_states"] == 240
        assert final["statistics"]["internal_clearance_inspected_states"] == 240
        assert final["statistics"]["transport"]["maximum_pending"] <= 2
    finally:
        guard.close()


def test_floor_cache_reuses_only_same_physics_and_write_generation(geometry):
    directory, source = geometry
    guard = GeometryGpuGuard(source["config"], directory / "floor_cache", "cuda:0", source["body_names"])
    try:
        p = torch.tensor(source["positions"][:2], device="cuda:0")
        q = torch.tensor(source["quaternions"][:2], device="cuda:0")
        first = guard.inspect(source["body_names"], p, q, state_key=(0, 0))
        repeated = guard.inspect(source["body_names"], p, q, state_key=(0, 0))
        torch.testing.assert_close(first.maximum_ground_penetration_m, repeated.maximum_ground_penetration_m, rtol=0, atol=0)
        assert torch.equal(first.ground_failure, repeated.ground_failure)
        assert guard.floor_computed_states == guard.floor_reused_states == 2
        p[:, :, 2] -= .1
        updated = guard.inspect(source["body_names"], p, q, state_key=(0, 1))
        assert updated.ground_failure.all()
        assert guard.floor_computed_states == 4
        guard.check_health()
        guard.inspect(source["body_names"], p, q, state_key=(1, 1))
        guard.inspect(source["body_names"], p, q)
        guard.inspect(source["body_names"], p, q, state_key=(1, 1))
        final = guard.flush()
        assert guard.floor_computed_states == 10
        assert final["statistics"]["internal_clearance_inspected_states"] == 12
        records = [json.loads(line) for line in guard.archive_path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 6
    finally:
        guard.close()


def _assert_binary_resources_released(client):
    assert client.closed
    assert client.log is None or client.log.closed
    assert client.process is None or client.process.poll() is not None
    assert client.worker_handle is None or client.worker_handle.handle is None
    for segment in client.segments:
        assert segment._mmap is None
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=segment.name)


def test_binary_actual_process_launch_failure_releases_allocated_slots(geometry):
    directory, source = geometry
    config = deepcopy(source["config"])
    config["python_executable"] = str(directory / "executable_does_not_exist.exe")
    client = GeometryBinaryClient.__new__(GeometryBinaryClient)
    with pytest.raises(FileNotFoundError):
        client.__init__(config, directory / "launch_failure")
    _assert_binary_resources_released(client)


def test_binary_actual_log_open_failure_releases_allocated_slots(geometry):
    directory, source = geometry
    failed_directory = directory / "log_open_failure"
    (failed_directory / "geometry_worker.log").mkdir(parents=True)
    client = GeometryBinaryClient.__new__(GeometryBinaryClient)
    with pytest.raises(OSError):
        client.__init__(source["config"], failed_directory)
    _assert_binary_resources_released(client)


def test_gpu_guard_actual_body_mapping_failure_releases_started_worker(geometry):
    directory, source = geometry
    guard = GeometryGpuGuard.__new__(GeometryGpuGuard)
    with pytest.raises(ValueError, match="body名称无效"):
        guard.__init__(source["config"], directory / "calculator_failure", "cpu", ["base", "base"])
    _assert_binary_resources_released(guard.transport)
