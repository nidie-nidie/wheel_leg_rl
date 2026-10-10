from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import tensordict  # noqa: F401
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESUME_SEQUENCE_TRACE_VERSION = "ResumeSequenceTraceV1"
RESUME_SEQUENCE_TRACE_FILENAME = "resume_sequence_trace.pt"
PROFILE_NAMES = ("portable", "rtx4060", "rtx5070")

from isaaclab.app import AppLauncher

from wheelleg_dreamwaq.training.runtime import validate_runtime

parser = argparse.ArgumentParser(description="Train the frozen WheelLeg Phase 2 DreamWaQ policy.")
parser.add_argument("--profile", choices=PROFILE_NAMES, default="portable")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--max-iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--resume", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
RUNTIME_INFO = validate_runtime(PROJECT_ROOT, device=args_cli.device)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import yaml

from isaaclab.utils.io import dump_yaml

from wheelleg_dreamwaq.algorithms.dreamwaq.history_wrapper import DreamWaQHistoryVecEnvWrapper
from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
    build_base_task_contract_from_configs,
    build_dreamwaq_algorithm_contract,
    build_dreamwaq_export_contract,
)
from wheelleg_dreamwaq.schemas.randomization import (
    FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
    PROCESS_START_RANDOMIZATION_STREAMS,
    RANDOMIZATION_STREAMS,
    RUNTIME_RANDOMIZATION_STREAMS,
    canonical_tensor_sha256,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.agents import WheelLegFlatDreamWaQRunnerCfg
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
from wheelleg_dreamwaq.training.checkpoint import (
    CLOSED_CHAIN_RESET_CACHE_FILENAME,
    checkpoint_cache_binding,
    closed_chain_reset_cache_metadata,
    load_closed_chain_reset_cache_artifact,
    sha256_file,
)
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import (
    DREAMWAQ_RUNNER_CLASS,
    DreamWaQContractOnPolicyRunner,
    fresh_resume_provenance,
    resume_dreamwaq_runner_from_checkpoint,
    validate_dreamwaq_checkpoint_metadata,
)


def _load_profile(name: str) -> dict[str, int]:
    path = PROJECT_ROOT / "configs" / "hardware" / f"{name}.yaml"
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(profile) != {"num_envs", "num_mini_batches"}:
        raise ValueError(f"Hardware profile {path} has unexpected fields")
    if profile["num_envs"] <= 0 or profile["num_mini_batches"] <= 0:
        raise ValueError(f"Hardware profile values must be positive: {profile}")
    return {key: int(value) for key, value in profile.items()}


def _git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _package_version(name: str) -> str:
    return importlib.metadata.version(name)


def _project_source_sha256() -> str:
    roots = (PROJECT_ROOT / "source", PROJECT_ROOT / "scripts", PROJECT_ROOT / "configs")
    files = [path for root in roots for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    files.extend(PROJECT_ROOT / name for name in ("pyproject.toml", "dependency-manifest.toml", "uv.lock"))
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix()):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest().upper()


def _atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _write_manifest(
    path: Path,
    *,
    profile_name: str,
    profile: dict[str, int],
    env_cfg: WheelLegFlatEnvCfg,
    agent_cfg: WheelLegFlatDreamWaQRunnerCfg,
    base_task_contract: dict,
    algorithm_contract: dict,
    export_contract: dict,
    randomization_audit: dict,
    cache_metadata: dict,
    resume_provenance: dict,
) -> None:
    architecture_path = PROJECT_ROOT.parent / "docs" / "2026-10-03-wheelleg-dreamwaq-architecture.md"
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv,
        "runner_class": DREAMWAQ_RUNNER_CLASS,
        "profile_name": profile_name,
        "profile": profile,
        "seed": agent_cfg.seed,
        "num_envs": env_cfg.scene.num_envs,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
        "max_iterations": agent_cfg.max_iterations,
        "device": agent_cfg.device,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "tensordict": _package_version("tensordict"),
        "isaac_sim": _package_version("isaacsim"),
        "rsl_rl": _package_version("rsl-rl-lib"),
        "isaac_lab_commit": _git_commit(PROJECT_ROOT / "dependencies" / "IsaacLab-v2.3.2"),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "asset_bundle_version": ASSET_BUNDLE_V2.version,
        "asset_bundle_hash": ASSET_BUNDLE_V2.bundle_hash,
        "architecture_sha256": sha256_file(architecture_path),
        "dependency_manifest_sha256": sha256_file(PROJECT_ROOT / "dependency-manifest.toml"),
        "pyproject_sha256": sha256_file(PROJECT_ROOT / "pyproject.toml"),
        "uv_lock_sha256": sha256_file(PROJECT_ROOT / "uv.lock"),
        "project_source_sha256": _project_source_sha256(),
        "runtime_validation": RUNTIME_INFO,
        "resolved_config_sha256": {
            "environment": sha256_file(path.parent / "params" / "env.yaml"),
            "agent": sha256_file(path.parent / "params" / "agent.yaml"),
            "hardware": sha256_file(path.parent / "params" / "hardware.yaml"),
        },
        "base_task_contract": base_task_contract,
        "dreamwaq_algorithm_contract": algorithm_contract,
        "dreamwaq_export_contract": export_contract,
        "randomization_audit": randomization_audit,
        "closed_chain_reset_cache": cache_metadata,
        "resume_provenance": resume_provenance,
        "randomness": {
            "master_seed": agent_cfg.seed,
            "effective_process_seed": agent_cfg.seed,
            "world_size": 1,
            "global_rank": 0,
            "local_rank": 0,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "environment_streams": randomization_audit["streams"],
        },
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> Path:
    if (int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))) != (1, 0, 0):
        raise RuntimeError("DreamWaQ Phase 2 supports one local GPU process")
    profile = _load_profile(args_cli.profile)
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else profile["num_envs"]
    if num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args_cli.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args_cli.run_name and re.fullmatch(r"[A-Za-z0-9._-]+", args_cli.run_name) is None:
        raise ValueError("--run-name may contain only letters, digits, dot, underscore, and hyphen")

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    env_cfg = WheelLegFlatEnvCfg()
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.randomization = FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    agent_cfg = WheelLegFlatDreamWaQRunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    agent_cfg.algorithm.num_mini_batches = profile["num_mini_batches"]
    agent_cfg.run_name = args_cli.run_name
    if args_cli.max_iterations is not None:
        if args_cli.max_iterations <= 0:
            raise ValueError("--max-iterations must be positive")
        agent_cfg.max_iterations = args_cli.max_iterations
    batch_size = num_envs * agent_cfg.num_steps_per_env
    if batch_size % agent_cfg.algorithm.num_mini_batches:
        raise ValueError("DreamWaQ rollout batch is not divisible by the mini-batch count")

    agent_dict = agent_cfg.to_dict()
    base_task_contract = build_base_task_contract_from_configs(
        asset_bundle_hash=asset_report.bundle_hash,
        env_cfg=env_cfg,
        clip_actions=agent_cfg.clip_actions,
    )
    algorithm_contract = build_dreamwaq_algorithm_contract(agent_dict)
    export_contract = build_dreamwaq_export_contract()

    resume_checkpoint = args_cli.resume.resolve() if args_cli.resume is not None else None
    source_manifest = None
    source_metadata = None
    source_cache_artifact = None
    source_cache_metadata = None
    source_cache_path = None
    source_num_envs = None
    same_num_envs = False
    starting_completed_iterations = 0
    if resume_checkpoint is not None:
        source_manifest_path = resume_checkpoint.parent / "run_manifest.json"
        source_metadata = validate_dreamwaq_checkpoint_metadata(
            resume_checkpoint,
            source_manifest_path,
            base_task_contract,
            algorithm_contract,
            export_contract,
            requested_seed=args_cli.seed,
        )
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_cache_artifact, source_cache_metadata, source_cache_path = load_closed_chain_reset_cache_artifact(
            source_manifest_path,
            manifest=source_manifest,
        )
        source_num_envs = int(source_metadata["hardware_profile"]["effective_num_envs"])
        same_num_envs = source_num_envs == num_envs
        starting_completed_iterations = int(source_metadata["completed_iterations"])
        if agent_cfg.max_iterations <= starting_completed_iterations:
            raise ValueError("--max-iterations is a total target and must exceed the checkpoint")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "_".join(part for part in (args_cli.profile, "fudan-v1", args_cli.run_name) if part)
    run_dir = PROJECT_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name / f"{timestamp}_{suffix}"
    (run_dir / "params").mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(run_dir)
    dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "params" / "agent.yaml"), agent_cfg)
    (run_dir / "params" / "hardware.yaml").write_text(yaml.safe_dump(profile, sort_keys=True), encoding="utf-8")
    print(f"[INFO] DreamWaQ run directory: {run_dir}", flush=True)
    print(f"[INFO] Run directory: {run_dir}", flush=True)

    start_time = time.perf_counter()
    direct_env = WheelLegFlatEnv(
        env_cfg,
        closed_chain_reset_cache=source_cache_artifact if same_num_envs else None,
    )
    randomization_audit = direct_env.randomization_audit
    cache_path = run_dir / CLOSED_CHAIN_RESET_CACHE_FILENAME
    if source_cache_path is not None and same_num_envs:
        _atomic_copy(source_cache_path, cache_path)
    else:
        _atomic_save(cache_path, direct_env.closed_chain_reset_cache_artifact)
    cache_metadata = closed_chain_reset_cache_metadata(cache_path, run_dir=run_dir)

    if resume_checkpoint is None:
        resume_provenance = fresh_resume_provenance()
    else:
        assert source_manifest is not None and source_metadata is not None
        assert source_cache_metadata is not None and source_num_envs is not None
        source_audit = source_manifest["randomization_audit"]
        if same_num_envs:
            if cache_metadata != source_cache_metadata:
                raise RuntimeError("Same-scale DreamWaQ resume did not preserve the cache exactly")
            cache_resume_mode = "reused_source_exactly"
            stream_sources = {stream: "source_checkpoint" for stream in RANDOMIZATION_STREAMS}
        else:
            cache_resume_mode = "regenerated_for_target_num_envs"
            stream_sources = {
                stream: "target_process_start" if stream in PROCESS_START_RANDOMIZATION_STREAMS else "source_checkpoint"
                for stream in RANDOMIZATION_STREAMS
            }
        source_manifest_path = resume_checkpoint.parent / "run_manifest.json"
        resume_provenance = {
            "schema_version": "ResumeMetadataV2",
            "mode": "weights_optimizer_rng_new_environment",
            "cache_resume_mode": cache_resume_mode,
            "source_checkpoint": str(resume_checkpoint),
            "source_checkpoint_sha256": sha256_file(resume_checkpoint),
            "source_run_manifest": str(source_manifest_path.resolve()),
            "source_run_manifest_sha256": sha256_file(source_manifest_path),
            "source_completed_iterations": starting_completed_iterations,
            "source_hardware_profile_name": source_metadata["hardware_profile"]["name"],
            "target_hardware_profile_name": args_cli.profile,
            "source_num_envs": source_num_envs,
            "target_num_envs": num_envs,
            "same_num_envs": same_num_envs,
            "source_cache_file_sha256": source_cache_metadata["file_sha256"],
            "source_cache_tensor_sha256": source_cache_metadata["tensor_sha256"],
            "target_cache_file_sha256": cache_metadata["file_sha256"],
            "target_cache_tensor_sha256": cache_metadata["tensor_sha256"],
            "source_realized_plan_hash": source_audit["realized_plan_hash"],
            "target_realized_plan_hash": randomization_audit["realized_plan_hash"],
            "rng_stream_sources": stream_sources,
        }

    run_manifest_path = run_dir / "run_manifest.json"
    _write_manifest(
        run_manifest_path,
        profile_name=args_cli.profile,
        profile=profile,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        base_task_contract=base_task_contract,
        algorithm_contract=algorithm_contract,
        export_contract=export_contract,
        randomization_audit=randomization_audit,
        cache_metadata=cache_metadata,
        resume_provenance=resume_provenance,
    )
    run_manifest_hash = sha256_file(run_manifest_path)
    env = DreamWaQHistoryVecEnvWrapper(direct_env, clip_actions=agent_cfg.clip_actions)
    checkpoint_metadata = {
        "seed": agent_cfg.seed,
        "run_manifest_sha256": run_manifest_hash,
        "base_task_contract_version": base_task_contract["manifest_version"],
        "base_task_contract_hash": base_task_contract["contract_hash"],
        "dreamwaq_algorithm_contract_version": algorithm_contract["manifest_version"],
        "dreamwaq_algorithm_contract_hash": algorithm_contract["contract_hash"],
        "dreamwaq_export_contract_version": export_contract["manifest_version"],
        "dreamwaq_export_contract_hash": export_contract["contract_hash"],
        "hardware_profile": {
            "name": args_cli.profile,
            "declared_num_envs": profile["num_envs"],
            "effective_num_envs": num_envs,
            "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "world_size": 1,
            "global_rank": 0,
            "local_rank": 0,
        },
        "reset_cache_binding": checkpoint_cache_binding(cache_metadata),
        "resume_provenance": resume_provenance,
    }
    runner = DreamWaQContractOnPolicyRunner(
        env,
        agent_dict,
        log_dir=str(run_dir),
        device=agent_cfg.device,
        checkpoint_metadata_factory=lambda: checkpoint_metadata,
        environment_rng_state_factory=direct_env.get_randomization_rng_state,
    )

    restored_learning_rate = None
    if resume_checkpoint is not None:
        assert source_metadata is not None
        streams_to_restore = RANDOMIZATION_STREAMS if same_num_envs else RUNTIME_RANDOMIZATION_STREAMS
        restored = resume_dreamwaq_runner_from_checkpoint(
            runner,
            resume_checkpoint,
            source_metadata,
            map_location=agent_cfg.device,
            requested_seed=args_cli.seed,
            target_total_iterations=agent_cfg.max_iterations,
            restore_environment_rng_state=lambda payload: direct_env.set_randomization_rng_state(
                payload,
                stream_names=streams_to_restore,
            ),
        )
        if restored != starting_completed_iterations:
            raise RuntimeError("DreamWaQ resume restored an unexpected iteration count")
        restored_learning_rate = float(runner.alg.learning_rate)
        direct_env.begin_resume_sequence_capture()
        env.reset()

    iterations_to_run = agent_cfg.max_iterations - starting_completed_iterations
    runner.learn(num_learning_iterations=iterations_to_run, init_at_random_ep_len=True)

    resume_trace_metadata = None
    if resume_checkpoint is not None:
        trace_tensors = direct_env.get_resume_sequence_trace_tensors()
        expected_shapes = {
            "post_restore_command": (num_envs, 3),
            "post_restore_root_velocity_world_usd": (num_envs, 6),
            "post_restore_reset_discarded_policy_obs": (num_envs, 25),
            "first_policy_observation": (num_envs, 25),
        }
        for name, shape in expected_shapes.items():
            value = trace_tensors.get(name)
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32:
                raise RuntimeError(f"DreamWaQ resume trace tensor is invalid: {name}")
            if tuple(value.shape) != shape or not value.is_contiguous() or not torch.isfinite(value).all():
                raise RuntimeError(f"DreamWaQ resume trace tensor shape/value is invalid: {name}")
        trace_hash = canonical_tensor_sha256(trace_tensors)
        trace_path = run_dir / RESUME_SEQUENCE_TRACE_FILENAME
        _atomic_save(
            trace_path,
            {
                "schema_version": RESUME_SEQUENCE_TRACE_VERSION,
                "cache_resume_mode": resume_provenance["cache_resume_mode"],
                "rng_stream_sources": resume_provenance["rng_stream_sources"],
                **trace_tensors,
                "tensor_sha256": trace_hash,
            },
        )
        resume_trace_metadata = {
            "schema_version": RESUME_SEQUENCE_TRACE_VERSION,
            "relative_path": trace_path.relative_to(run_dir).as_posix(),
            "file_sha256": sha256_file(trace_path),
            "tensor_sha256": trace_hash,
        }

    if runner.writer is not None:
        runner.writer.flush()
        runner.writer.close()
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda item: int(item.stem.split("_")[-1]))
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not checkpoints or not event_files:
        raise RuntimeError("DreamWaQ training did not produce checkpoint and TensorBoard artifacts")
    for checkpoint in checkpoints:
        validate_dreamwaq_checkpoint_metadata(
            checkpoint,
            run_manifest_path,
            base_task_contract,
            algorithm_contract,
            export_contract,
            requested_seed=args_cli.seed,
        )
    monitor = runner.alg.estimator_monitor_state
    summary = {
        "run_dir": str(run_dir.resolve()),
        "iterations_requested": agent_cfg.max_iterations,
        "starting_completed_iterations": starting_completed_iterations,
        "iterations_executed": iterations_to_run,
        "completed_iterations": runner.current_learning_iteration + 1,
        "runner_final_iteration_index": runner.current_learning_iteration,
        "duration_seconds": time.perf_counter() - start_time,
        "final_checkpoint": str(checkpoints[-1].resolve()),
        "event_files": [str(path.resolve()) for path in event_files],
        "run_manifest_sha256": run_manifest_hash,
        "base_task_contract_hash": base_task_contract["contract_hash"],
        "dreamwaq_algorithm_contract_hash": algorithm_contract["contract_hash"],
        "dreamwaq_export_contract_hash": export_contract["contract_hash"],
        "closed_chain_reset_cache": {
            "path": str(cache_path.resolve()),
            "file_sha256": cache_metadata["file_sha256"],
            "tensor_sha256": cache_metadata["tensor_sha256"],
        },
        "resumed_from": None if resume_checkpoint is None else str(resume_checkpoint),
        "restored_learning_rate": restored_learning_rate,
        "resume_sequence_trace": resume_trace_metadata,
        "estimator_monitor_state": monitor.to_dict(),
        "context_final_window_mean": monitor.final_window_mean,
        "context_acceptance_threshold": monitor.acceptance_threshold(1.0e-3, 0.10),
        "context_acceptance": monitor.passes_context_gate(1.0e-3, 0.10, require_full_window=True),
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    latest = PROJECT_ROOT / "artifacts" / "phase2_dreamwaq" / "latest_run.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(run_dir.resolve()), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return run_dir


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if exit_code == 0:
            simulation_app.close(skip_cleanup=True)
        else:
            os._exit(exit_code)
    raise SystemExit(exit_code)
