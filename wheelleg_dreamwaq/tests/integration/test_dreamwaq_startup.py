from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
import torch

from conftest import PROJECT_ROOT, run_project_script
from wheelleg_dreamwaq.schemas.isaac_evaluation import (
    EVALUATION_ACTION_STEPS,
    EVALUATION_ENV_COUNT,
    FORMAL_SCENARIOS,
    PHASE1R_ISAAC_BASELINE,
)
from wheelleg_dreamwaq.schemas.randomization import (
    PROCESS_START_RANDOMIZATION_STREAMS,
    RANDOMIZATION_STREAMS,
    RUNTIME_RANDOMIZATION_STREAMS,
    canonical_tensor_sha256,
)
from wheelleg_dreamwaq.training.checkpoint import sha256_file


def _run_training(arguments: list[str]) -> Path:
    result = run_project_script(["scripts/train_dreamwaq.py", *arguments, "--headless"], timeout=900)
    prefix = "[INFO] DreamWaQ run directory: "
    for line in result.stdout.splitlines():
        if line.startswith(prefix):
            run_dir = Path(line[len(prefix) :].strip())
            if run_dir.is_dir():
                return run_dir
    pytest.fail(f"DreamWaQ training output did not identify a run directory:\n{result.stdout[-4000:]}")


def _load_run(run_dir: Path) -> tuple[dict, dict, Path, dict]:
    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(summary["final_checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return summary, manifest, checkpoint, payload


def _load_resume_trace(run_dir: Path, summary: dict, *, num_envs: int) -> dict:
    metadata = summary["resume_sequence_trace"]
    trace_path = run_dir / metadata["relative_path"]
    assert metadata["file_sha256"] == sha256_file(trace_path)
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)
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


@pytest.fixture(scope="module")
def dreamwaq_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    artifact_root = tmp_path_factory.mktemp("dreamwaq-integration")
    suffix = uuid.uuid4().hex[:8]
    seed = 20261007
    source_dir = _run_training(
        [
            "--profile",
            "portable",
            "--max-iterations",
            "2",
            "--seed",
            str(seed),
            "--run-name",
            f"pytest-dream-source-{suffix}",
        ]
    )
    source = _load_run(source_dir)
    same_dir = _run_training(
        [
            "--profile",
            "portable",
            "--num-envs",
            "32",
            "--max-iterations",
            "3",
            "--seed",
            str(seed),
            "--run-name",
            f"pytest-dream-same-{suffix}",
            "--resume",
            str(source[2]),
        ]
    )
    cross_dir = _run_training(
        [
            "--profile",
            "portable",
            "--num-envs",
            "16",
            "--max-iterations",
            "3",
            "--seed",
            str(seed),
            "--run-name",
            f"pytest-dream-cross-{suffix}",
            "--resume",
            str(source[2]),
        ]
    )
    return {
        "root": artifact_root,
        "source": source,
        "same": _load_run(same_dir),
        "cross": _load_run(cross_dir),
    }


@pytest.mark.integration
def test_dreamwaq_fresh_same_and_cross_scale_resume(dreamwaq_runs: dict[str, object]) -> None:
    source_summary, source_manifest, _, source_payload = dreamwaq_runs["source"]
    same_summary, same_manifest, _, same_payload = dreamwaq_runs["same"]
    cross_summary, cross_manifest, _, cross_payload = dreamwaq_runs["cross"]

    assert source_summary["completed_iterations"] == 2
    assert source_manifest["resume_provenance"] == {"schema_version": "ResumeMetadataV2", "mode": "fresh"}
    source_monitor = source_payload["infos"]["estimator_monitor_state"]
    assert source_monitor["completed_rollouts"] == 2

    for summary, manifest, payload, same_num_envs in (
        (same_summary, same_manifest, same_payload, True),
        (cross_summary, cross_manifest, cross_payload, False),
    ):
        resume = manifest["resume_provenance"]
        assert summary["completed_iterations"] == 3
        assert summary["starting_completed_iterations"] == 2
        assert summary["iterations_executed"] == 1
        assert resume["same_num_envs"] is same_num_envs
        assert resume["cache_resume_mode"] == (
            "reused_source_exactly" if same_num_envs else "regenerated_for_target_num_envs"
        )
        assert summary["resume_sequence_trace"] is not None
        monitor = payload["infos"]["estimator_monitor_state"]
        assert monitor["completed_rollouts"] == 3
        assert monitor["context_mu_feature_std_mean_initial"] == pytest.approx(
            source_monitor["context_mu_feature_std_mean_initial"], rel=0.0, abs=0.0
        )
        source_recent = source_monitor["recent_context_mu_feature_std_mean"]
        assert monitor["recent_context_mu_feature_std_mean"][: len(source_recent)] == pytest.approx(
            source_recent, rel=0.0, abs=0.0
        )
        trace = _load_resume_trace(Path(summary["run_dir"]), summary, num_envs=32 if same_num_envs else 16)
        assert trace["cache_resume_mode"] == resume["cache_resume_mode"]
        assert trace["rng_stream_sources"] == resume["rng_stream_sources"]
        if same_num_envs:
            assert set(trace["rng_stream_sources"].values()) == {"source_checkpoint"}
        else:
            assert all(
                trace["rng_stream_sources"][stream] == "target_process_start"
                for stream in PROCESS_START_RANDOMIZATION_STREAMS
            )
            assert all(
                trace["rng_stream_sources"][stream] == "source_checkpoint"
                for stream in RUNTIME_RANDOMIZATION_STREAMS
            )
        assert set(trace["rng_stream_sources"]) == set(RANDOMIZATION_STREAMS)

    assert same_manifest["closed_chain_reset_cache"]["file_sha256"] == source_manifest[
        "closed_chain_reset_cache"
    ]["file_sha256"]
    assert cross_manifest["closed_chain_reset_cache"]["tensor_sha256"] != source_manifest[
        "closed_chain_reset_cache"
    ]["tensor_sha256"]


@pytest.mark.integration
def test_dreamwaq_play_export_and_torchscript_golden(dreamwaq_runs: dict[str, object]) -> None:
    root = dreamwaq_runs["root"]
    _, _, checkpoint, _ = dreamwaq_runs["same"]
    play_reports = []
    for index in range(2):
        output = root / f"play-{index}.json"
        run_project_script(
            [
                "scripts/play_dreamwaq.py",
                "--checkpoint",
                str(checkpoint),
                "--num-envs",
                "2",
                "--steps",
                "8",
                "--fixed-command",
                "0.2",
                "0.0",
                "0.2",
                "--output",
                str(output),
                "--headless",
            ],
            timeout=600,
        )
        play_reports.append(json.loads(output.read_text(encoding="utf-8")))
    assert all(report["finite"] and report["completed_steps"] == 8 for report in play_reports)
    assert play_reports[0]["reset_count"] == play_reports[1]["reset_count"]
    assert play_reports[0]["mean_reward_per_step"] == pytest.approx(
        play_reports[1]["mean_reward_per_step"], rel=0.0, abs=1.0e-7
    )

    export_dir = root / "export"
    run_project_script(
        [
            "scripts/export_dreamwaq_actor.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(export_dir),
        ]
    )
    assert {path.name for path in export_dir.iterdir()} == {
        "actor.ts",
        "golden_vectors.pt",
        "policy_manifest.json",
    }
    manifest = json.loads((export_dir / "policy_manifest.json").read_text(encoding="utf-8"))
    assert manifest["network"]["input_dimension"] == 125
    assert manifest["network"]["output_dimension"] == 6
    assert manifest["history_adapter_version"] == "MujocoFrameMajorHistoryAdapterV1"
    golden = torch.load(export_dir / "golden_vectors.pt", map_location="cpu", weights_only=False)
    actor = torch.jit.load(str(export_dir / "actor.ts"), map_location="cpu").eval()
    with torch.inference_mode():
        actual = actor(golden["history"])
        single = actor(golden["history"][:1])
        dynamic = actor(golden["history"][:3])
    assert actual.shape == (32, 6)
    assert single.shape == (1, 6)
    assert dynamic.shape == (3, 6)
    assert torch.max(torch.abs(actual - golden["expected_action_mean"])).item() <= 1.0e-7


@pytest.mark.integration
def test_dreamwaq_and_phase1r_share_frozen_isaac_evaluation(dreamwaq_runs: dict[str, object]) -> None:
    root = dreamwaq_runs["root"]
    _, _, checkpoint, _ = dreamwaq_runs["same"]
    reset_cache = root / "evaluation-reset-cache.pt"
    baseline_dir = root / "isaac-baseline"
    candidate_dir = root / "isaac-dreamwaq"
    run_project_script(
        [
            "scripts/evaluate_isaac.py",
            "--checkpoint",
            PHASE1R_ISAAC_BASELINE["checkpoint"],
            "--reset-cache",
            str(reset_cache),
            "--output",
            str(baseline_dir),
            "--headless",
        ],
        timeout=900,
    )
    run_project_script(
        [
            "scripts/evaluate_isaac.py",
            "--checkpoint",
            str(checkpoint),
            "--reset-cache",
            str(reset_cache),
            "--baseline-report",
            str(baseline_dir / "summary.json"),
            "--output",
            str(candidate_dir),
            "--headless",
        ],
        timeout=900,
    )

    baseline = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    candidate = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    assert baseline["policy_kind"] == "phase1r_baseline"
    assert candidate["policy_kind"] == "dreamwaq"
    assert baseline["evaluation_contract"] == candidate["evaluation_contract"]
    assert baseline["reset_cache"] == candidate["reset_cache"]
    assert baseline["reset_cache"]["file_sha256"] == sha256_file(reset_cache)
    control = candidate["evaluation_contract"]["control"]
    assert control["action_steps"] == EVALUATION_ACTION_STEPS == 499
    assert control["expected_done"] == "truncated_on_action_step_499"
    assert candidate["aggregate"]["scenario_count"] == EVALUATION_ENV_COUNT == 8
    assert [item["name"] for item in candidate["aggregate"]["scenarios"]] == [
        name for name, _ in FORMAL_SCENARIOS
    ]
    for scenario in candidate["aggregate"]["scenarios"]:
        assert 0 <= scenario["survival_steps"] <= EVALUATION_ACTION_STEPS
        if scenario["completed"]:
            assert scenario["survival_steps"] == EVALUATION_ACTION_STEPS
            assert scenario["failure_reason"] is None
        else:
            assert scenario["failure_reason"] is not None
    estimator = candidate["estimator"]
    assert 0 < estimator["velocity_sample_frames"] <= EVALUATION_ENV_COUNT * EVALUATION_ACTION_STEPS
    assert estimator["velocity_scalar_samples"] == 3 * estimator["velocity_sample_frames"]
    assert candidate["baseline_comparison"]["baseline_report_hash"] == baseline["report_hash"]
