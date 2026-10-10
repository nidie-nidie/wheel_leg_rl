from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
import torch

from conftest import PROJECT_ROOT, run_project_script
from wheelleg_dreamwaq.schemas.randomization import (
    PROCESS_START_RANDOMIZATION_STREAMS,
    RANDOMIZATION_STREAMS,
    RUNTIME_RANDOMIZATION_STREAMS,
    canonical_tensor_sha256,
)
from wheelleg_dreamwaq.training.checkpoint import CHECKPOINT_METADATA_VERSION, sha256_file


RESUME_METADATA_FIELDS = {
    "schema_version",
    "mode",
    "cache_resume_mode",
    "source_checkpoint",
    "source_checkpoint_sha256",
    "source_run_manifest",
    "source_run_manifest_sha256",
    "source_completed_iterations",
    "source_hardware_profile_name",
    "target_hardware_profile_name",
    "source_num_envs",
    "target_num_envs",
    "same_num_envs",
    "source_cache_file_sha256",
    "source_cache_tensor_sha256",
    "target_cache_file_sha256",
    "target_cache_tensor_sha256",
    "source_realized_plan_hash",
    "target_realized_plan_hash",
    "rng_stream_sources",
}
TRACE_TENSOR_FIELDS = (
    "post_restore_command",
    "post_restore_root_velocity_world_usd",
    "post_restore_reset_discarded_policy_obs",
    "first_policy_observation",
)


def _run_training(arguments: list[str]) -> Path:
    run_project_script(["scripts/train_ppo.py", *arguments, "--headless"], timeout=360)
    latest_path = PROJECT_ROOT / "artifacts" / "phase1" / "latest_run.txt"
    return Path(latest_path.read_text(encoding="utf-8").strip())


def _load_run(run_dir: Path) -> tuple[dict, dict, Path, dict]:
    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(summary["final_checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return summary, manifest, checkpoint, payload


def _load_resume_trace(run_dir: Path, summary: dict, *, num_envs: int) -> dict:
    metadata = summary["resume_sequence_trace"]
    assert set(metadata) == {"schema_version", "relative_path", "file_sha256", "tensor_sha256"}
    assert metadata["schema_version"] == "ResumeSequenceTraceV1"
    trace_path = run_dir / metadata["relative_path"]
    assert trace_path.name == "resume_sequence_trace.pt"
    assert metadata["file_sha256"] == sha256_file(trace_path)
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)
    assert set(trace) == {
        "schema_version",
        "cache_resume_mode",
        "rng_stream_sources",
        *TRACE_TENSOR_FIELDS,
        "tensor_sha256",
    }
    expected_shapes = {
        "post_restore_command": (num_envs, 3),
        "post_restore_root_velocity_world_usd": (num_envs, 6),
        "post_restore_reset_discarded_policy_obs": (num_envs, 25),
        "first_policy_observation": (num_envs, 25),
    }
    tensors = {}
    for name, shape in expected_shapes.items():
        values = trace[name]
        assert values.device.type == "cpu"
        assert values.dtype == torch.float32
        assert values.is_contiguous()
        assert tuple(values.shape) == shape
        assert torch.isfinite(values).all()
        tensors[name] = values
    assert trace["tensor_sha256"] == canonical_tensor_sha256(tensors)
    assert metadata["tensor_sha256"] == trace["tensor_sha256"]
    return trace


@pytest.mark.integration
def test_rtx5070_checkpoint_resume_same_and_cross_environment_counts() -> None:
    source_name = f"pytest-source-{uuid.uuid4().hex[:8]}"
    source_dir = _run_training(
        [
            "--profile",
            "rtx5070",
            "--max-iterations",
            "2",
            "--run-name",
            source_name,
        ]
    )
    source_summary, source_manifest, source_checkpoint, source_payload = _load_run(source_dir)
    source_metadata = source_payload["infos"]
    source_optimizer_lr = float(source_payload["optimizer_state_dict"]["param_groups"][0]["lr"])
    source_cache = source_dir / "closed_chain_reset_cache.pt"

    assert source_name in source_dir.name
    assert source_summary["completed_iterations"] == 2
    assert source_summary["resume_sequence_trace"] is None
    assert source_metadata["metadata_version"] == CHECKPOINT_METADATA_VERSION
    assert source_metadata["completed_iterations"] == 2
    assert source_metadata["run_manifest_sha256"] == sha256_file(source_dir / "run_manifest.json")
    assert source_metadata["algorithm_learning_rate"] == pytest.approx(source_optimizer_lr)
    assert set(source_metadata["environment_rng_state"]["streams"]) == set(RANDOMIZATION_STREAMS)
    assert source_manifest["contract"]["manifest_version"] == "Phase1RandomizedContractV3"
    assert source_manifest["contract"]["runtime"]["collision_policy"] == "WheelOnlyCollisionV2"
    assert source_manifest["contract"]["schemas"]["reward"] == "RewardSchemaV2"
    assert source_manifest["contract"]["schemas"]["closed_chain_reset_cache"] == (
        "ClosedChainResetCacheSchemaV2"
    )
    assert source_manifest["closed_chain_reset_cache"]["file_sha256"] == sha256_file(source_cache)
    assert source_metadata["closed_chain_reset_cache"]["file_sha256"] == sha256_file(source_cache)
    source_cache_artifact = torch.load(source_cache, map_location="cpu", weights_only=False)
    assert source_cache_artifact["q_reset_projected_env"].shape[0] == 256
    assert source_cache_artifact["root_height_offset_env"].shape == (256,)

    same_name = f"pytest-same-{uuid.uuid4().hex[:8]}"
    same_dir = _run_training(
        [
            "--profile",
            "portable",
            "--num-envs",
            "256",
            "--max-iterations",
            "3",
            "--run-name",
            same_name,
            "--resume",
            str(source_checkpoint),
        ]
    )
    same_summary, same_manifest, same_checkpoint, same_payload = _load_run(same_dir)
    same_resume = same_manifest["resume"]
    same_cache = same_dir / "closed_chain_reset_cache.pt"

    assert set(same_resume) == RESUME_METADATA_FIELDS
    assert same_resume["schema_version"] == "ResumeMetadataV2"
    assert same_resume["cache_resume_mode"] == "reused_source_exactly"
    assert same_resume["source_hardware_profile_name"] == "rtx5070"
    assert same_resume["target_hardware_profile_name"] == "portable"
    assert same_resume["source_num_envs"] == same_resume["target_num_envs"] == 256
    assert same_resume["same_num_envs"] is True
    assert set(same_resume["rng_stream_sources"]) == set(RANDOMIZATION_STREAMS)
    assert set(same_resume["rng_stream_sources"].values()) == {"source_checkpoint"}
    assert sha256_file(same_cache) == sha256_file(source_cache)
    assert same_payload["infos"]["closed_chain_reset_cache"] == source_metadata["closed_chain_reset_cache"]
    assert same_summary["completed_iterations"] == 3
    assert same_summary["restored_learning_rate"] == pytest.approx(source_optimizer_lr)
    assert same_payload["iter"] == 2
    same_trace = _load_resume_trace(same_dir, same_summary, num_envs=256)
    assert same_trace["cache_resume_mode"] == "reused_source_exactly"

    cross_runs = []
    for _ in range(2):
        cross_name = f"pytest-cross-{uuid.uuid4().hex[:8]}"
        cross_dir = _run_training(
            [
                "--profile",
                "portable",
                "--max-iterations",
                "3",
                "--run-name",
                cross_name,
                "--resume",
                str(source_checkpoint),
            ]
        )
        summary, manifest, checkpoint, payload = _load_run(cross_dir)
        resume = manifest["resume"]
        target_cache = cross_dir / "closed_chain_reset_cache.pt"
        trace = _load_resume_trace(cross_dir, summary, num_envs=32)

        assert set(resume) == RESUME_METADATA_FIELDS
        assert resume["schema_version"] == "ResumeMetadataV2"
        assert resume["cache_resume_mode"] == "regenerated_for_target_num_envs"
        assert resume["source_hardware_profile_name"] == "rtx5070"
        assert resume["target_hardware_profile_name"] == "portable"
        assert resume["source_num_envs"] == 256
        assert resume["target_num_envs"] == 32
        assert resume["same_num_envs"] is False
        assert resume["source_cache_file_sha256"] == sha256_file(source_cache)
        assert resume["target_cache_file_sha256"] == sha256_file(target_cache)
        assert resume["source_cache_tensor_sha256"] == source_manifest["closed_chain_reset_cache"][
            "tensor_sha256"
        ]
        assert resume["target_cache_tensor_sha256"] == manifest["closed_chain_reset_cache"][
            "tensor_sha256"
        ]
        assert resume["source_realized_plan_hash"] == source_manifest["randomization_audit"][
            "realized_plan_hash"
        ]
        assert resume["target_realized_plan_hash"] == manifest["randomization_audit"][
            "realized_plan_hash"
        ]
        assert all(
            resume["rng_stream_sources"][stream] == "target_process_start"
            for stream in PROCESS_START_RANDOMIZATION_STREAMS
        )
        assert all(
            resume["rng_stream_sources"][stream] == "source_checkpoint"
            for stream in RUNTIME_RANDOMIZATION_STREAMS
        )
        assert trace["rng_stream_sources"] == resume["rng_stream_sources"]
        assert payload["infos"]["closed_chain_reset_cache"]["file_sha256"] == sha256_file(target_cache)
        assert payload["infos"]["closed_chain_reset_cache"] != source_metadata["closed_chain_reset_cache"]
        assert summary["completed_iterations"] == 3
        assert summary["restored_learning_rate"] == pytest.approx(source_optimizer_lr)
        assert payload["iter"] == 2
        cross_runs.append((cross_dir, summary, checkpoint, trace))

    first_trace = cross_runs[0][3]
    repeated_trace = cross_runs[1][3]
    assert first_trace["tensor_sha256"] == repeated_trace["tensor_sha256"]
    for field in TRACE_TENSOR_FIELDS:
        assert torch.equal(first_trace[field], repeated_trace[field])

    run_project_script(
        [
            "scripts/play.py",
            "--checkpoint",
            str(cross_runs[0][2]),
            "--num-envs",
            "2",
            "--steps",
            "10",
            "--headless",
        ],
        timeout=240,
    )
