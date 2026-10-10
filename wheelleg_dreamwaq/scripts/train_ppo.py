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

# On Windows, preload the PyTorch/TensorDict native DLLs before Kit changes the DLL search state.
from isaaclab.app import AppLauncher

from wheelleg_dreamwaq.schemas.randomization import (
    CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
    FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
    NOMINAL_TRAINING_PROFILE_V1,
    PROCESS_START_RANDOMIZATION_STREAMS,
    RANDOMIZATION_STREAMS,
    RANDOMIZATION_SCHEMA_VERSION,
    RUNTIME_RANDOMIZATION_STREAMS,
    canonical_tensor_sha256,
)
from wheelleg_dreamwaq.training.runtime import validate_runtime

PROFILE_NAMES = ("portable", "rtx4060", "rtx5070")
RANDOMIZATION_PROFILES = {
    "nominal": NOMINAL_TRAINING_PROFILE_V1,
    "fudan-v1": FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
}
RESUME_METADATA_VERSION = "ResumeMetadataV2"
RESUME_SEQUENCE_TRACE_VERSION = "ResumeSequenceTraceV1"
RESUME_SEQUENCE_TRACE_FILENAME = "resume_sequence_trace.pt"

parser = argparse.ArgumentParser(description="Train the WheelLeg Phase 1 asymmetric PPO baseline.")
parser.add_argument("--profile", choices=PROFILE_NAMES, default="portable")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--max-iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--resume", type=Path, default=None)
parser.add_argument("--randomization-profile", choices=tuple(RANDOMIZATION_PROFILES), default="fudan-v1")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
RUNTIME_INFO = validate_runtime(PROJECT_ROOT, device=args_cli.device)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import yaml

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.kinematics.virtual_leg import VIRTUAL_LEG_KINEMATICS_VERSION
from wheelleg_dreamwaq.schemas.action import ACTION_SCHEMA_VERSION, CANONICAL_JOINT_ORDER
from wheelleg_dreamwaq.schemas.command import COMMAND_SCHEMA_VERSION
from wheelleg_dreamwaq.schemas.frames import CONTROL_FRAME_VERSION
from wheelleg_dreamwaq.schemas.manifest import build_phase1_contract_from_configs
from wheelleg_dreamwaq.schemas.normalization import NORMALIZATION_SCHEMA_VERSION
from wheelleg_dreamwaq.schemas.observation import ACTOR_OBS_SCHEMA_VERSION, CRITIC_OBS_SCHEMA_VERSION
from wheelleg_dreamwaq.schemas.physics import PHYSICS_SCHEMA_VERSION
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.agents import WheelLegFlatPPORunnerCfg
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import COMMAND_SAMPLING_VERSION
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import REWARD_SCHEMA_VERSION
from wheelleg_dreamwaq.training.checkpoint import (
    CLOSED_CHAIN_RESET_CACHE_FILENAME,
    ContractOnPolicyRunner,
    checkpoint_cache_binding,
    closed_chain_reset_cache_metadata,
    load_closed_chain_reset_cache_artifact,
    resume_runner_from_checkpoint,
    sha256_file,
    validate_checkpoint_metadata,
)


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _project_source_sha256() -> str:
    roots = (PROJECT_ROOT / "source", PROJECT_ROOT / "scripts", PROJECT_ROOT / "configs")
    files = [path for root in roots for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    files.extend(PROJECT_ROOT / name for name in ("pyproject.toml", "dependency-manifest.toml", "uv.lock"))
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix()):
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest().upper()


def _atomic_save_cache(path: Path, artifact: dict) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(artifact, temporary_path)
    os.replace(temporary_path, path)


def _atomic_copy_cache(source: Path, destination: Path) -> None:
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary_path)
    os.replace(temporary_path, destination)


def _load_profile(name: str) -> dict[str, int]:
    path = PROJECT_ROOT / "configs" / "hardware" / f"{name}.yaml"
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_keys = {"num_envs", "num_mini_batches"}
    if set(profile) != expected_keys:
        raise ValueError(f"Hardware profile {path} must contain exactly {sorted(expected_keys)}")
    if profile["num_envs"] <= 0 or profile["num_mini_batches"] <= 0:
        raise ValueError(f"Hardware profile values must be positive: {profile}")
    return {key: int(value) for key, value in profile.items()}


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _package_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def _write_manifest(
    path: Path,
    *,
    profile_name: str,
    profile: dict[str, int],
    env_cfg: WheelLegFlatEnvCfg,
    agent_cfg: WheelLegFlatPPORunnerCfg,
    asset_bundle_hash: str,
    contract: dict,
    randomization_audit: dict,
    closed_chain_reset_cache: dict,
    resume: dict | None,
) -> None:
    architecture_path = PROJECT_ROOT.parent / "docs" / "2026-10-03-wheelleg-dreamwaq-architecture.md"
    isaaclab_checkout = PROJECT_ROOT / "dependencies" / "IsaacLab-v2.3.2"
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv,
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
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "isaac_sim": _package_version("isaacsim"),
        "rsl_rl": _package_version("rsl-rl-lib"),
        "isaac_lab_commit": _git_commit(isaaclab_checkout),
        "asset_bundle_version": ASSET_BUNDLE_V2.version,
        "asset_bundle_hash": asset_bundle_hash,
        "architecture_sha256": _sha256(architecture_path),
        "dependency_manifest_sha256": _sha256(PROJECT_ROOT / "dependency-manifest.toml"),
        "pyproject_sha256": _sha256(PROJECT_ROOT / "pyproject.toml"),
        "uv_lock_sha256": _sha256(PROJECT_ROOT / "uv.lock"),
        "project_source_sha256": _project_source_sha256(),
        "runtime_validation": RUNTIME_INFO,
        "resolved_config_sha256": {
            "environment": _sha256(path.parent / "params" / "env.yaml"),
            "agent": _sha256(path.parent / "params" / "agent.yaml"),
            "hardware": _sha256(path.parent / "params" / "hardware.yaml"),
        },
        "contract": contract,
        "randomization_audit": randomization_audit,
        "closed_chain_reset_cache": closed_chain_reset_cache,
        "schemas": {
            "action": ACTION_SCHEMA_VERSION,
            "command": COMMAND_SCHEMA_VERSION,
            "actor_observation": ACTOR_OBS_SCHEMA_VERSION,
            "critic_observation": CRITIC_OBS_SCHEMA_VERSION,
            "normalization": NORMALIZATION_SCHEMA_VERSION,
            "control_frame": CONTROL_FRAME_VERSION,
            "command_sampling": COMMAND_SAMPLING_VERSION,
            "physics": PHYSICS_SCHEMA_VERSION,
            "reward": REWARD_SCHEMA_VERSION,
            "virtual_leg_kinematics": VIRTUAL_LEG_KINEMATICS_VERSION,
            "randomization": RANDOMIZATION_SCHEMA_VERSION,
            "closed_chain_reset_cache": CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
        },
        "canonical_joint_order": list(CANONICAL_JOINT_ORDER),
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
        "resume": resume,
    }
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary_path, path)


def main() -> Path:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 1 or global_rank != 0 or local_rank != 0:
        raise RuntimeError("Phase 1 supports one local GPU process; distributed launch variables must be unset")

    profile = _load_profile(args_cli.profile)
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else profile["num_envs"]
    if num_envs <= 0:
        raise ValueError("--num-envs must be positive")

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
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
    env_cfg.randomization = RANDOMIZATION_PROFILES[args_cli.randomization_profile]

    agent_cfg = WheelLegFlatPPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    agent_cfg.algorithm.num_mini_batches = profile["num_mini_batches"]
    if args_cli.max_iterations is not None:
        if args_cli.max_iterations <= 0:
            raise ValueError("--max-iterations must be positive")
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.run_name = args_cli.run_name

    if args_cli.run_name and re.fullmatch(r"[A-Za-z0-9._-]+", args_cli.run_name) is None:
        raise ValueError("--run-name may contain only letters, digits, dot, underscore, and hyphen")

    batch_size = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env
    if batch_size % agent_cfg.algorithm.num_mini_batches != 0:
        raise ValueError(
            f"Rollout batch {batch_size} is not divisible by {agent_cfg.algorithm.num_mini_batches} mini-batches"
        )

    resume_checkpoint = args_cli.resume.resolve() if args_cli.resume is not None else None
    resume_metadata = None
    resume_manifest_path = None
    source_manifest = None
    source_cache_artifact = None
    source_cache_metadata = None
    source_cache_path = None
    source_num_envs = None
    same_num_envs = False
    starting_completed_iterations = 0
    resume_info = None
    resume_sequence_trace_metadata = None
    if resume_checkpoint is not None:
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
        resume_manifest_path = resume_checkpoint.parent / "run_manifest.json"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix_parts = [args_cli.profile, args_cli.randomization_profile]
    if args_cli.run_name:
        suffix_parts.append(args_cli.run_name)
    run_dir = PROJECT_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name / f"{timestamp}_{'_'.join(suffix_parts)}"
    (run_dir / "params").mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(run_dir)

    dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "params" / "agent.yaml"), agent_cfg)
    (run_dir / "params" / "hardware.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=True), encoding="utf-8"
    )

    print(f"[INFO] Run directory: {run_dir}", flush=True)
    print(
        f"[INFO] PPO batch: {env_cfg.scene.num_envs} envs x {agent_cfg.num_steps_per_env} steps "
        f"/ {agent_cfg.algorithm.num_mini_batches} mini-batches",
        flush=True,
    )

    contract = build_phase1_contract_from_configs(
        asset_bundle_hash=asset_report.bundle_hash,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
    )
    if resume_checkpoint is not None:
        assert resume_manifest_path is not None
        resume_metadata = validate_checkpoint_metadata(resume_checkpoint, resume_manifest_path, contract)
        if int(resume_metadata["seed"]) != agent_cfg.seed:
            raise ValueError(
                f"Resume seed {resume_metadata['seed']} does not match requested seed {agent_cfg.seed}"
            )
        source_num_envs = int(resume_metadata["hardware_profile"]["effective_num_envs"])
        same_num_envs = source_num_envs == env_cfg.scene.num_envs
        source_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
        source_cache_artifact, source_cache_metadata, source_cache_path = (
            load_closed_chain_reset_cache_artifact(
                resume_manifest_path,
                manifest=source_manifest,
            )
        )
        starting_completed_iterations = int(resume_metadata["completed_iterations"])
        if agent_cfg.max_iterations <= starting_completed_iterations:
            raise ValueError(
                f"--max-iterations is a total target and must exceed the checkpoint's "
                f"{starting_completed_iterations} completed iterations"
            )

    start_time = time.perf_counter()
    direct_env = WheelLegFlatEnv(
        env_cfg,
        closed_chain_reset_cache=source_cache_artifact if same_num_envs else None,
    )
    randomization_audit = direct_env.randomization_audit
    cache_path = run_dir / CLOSED_CHAIN_RESET_CACHE_FILENAME
    if source_cache_path is not None and same_num_envs:
        _atomic_copy_cache(source_cache_path, cache_path)
    else:
        _atomic_save_cache(cache_path, direct_env.closed_chain_reset_cache_artifact)
    cache_metadata = closed_chain_reset_cache_metadata(cache_path, run_dir=run_dir)
    if resume_checkpoint is not None:
        assert resume_manifest_path is not None and resume_metadata is not None
        assert source_manifest is not None and source_cache_metadata is not None
        assert source_num_envs is not None
        source_audit = source_manifest.get("randomization_audit")
        if not isinstance(source_audit, dict):
            raise ValueError("Resume source manifest has no randomization audit")
        source_hardware_profile_name = resume_metadata["hardware_profile"]["name"]
        if source_manifest.get("profile_name") != source_hardware_profile_name:
            raise ValueError("Resume source hardware profile metadata is inconsistent")
        if source_manifest.get("num_envs") != source_num_envs:
            raise ValueError("Resume source num_envs metadata is inconsistent")
        if same_num_envs:
            if source_audit.get("realized_plan_hash") != randomization_audit["realized_plan_hash"]:
                raise ValueError("Resume process-start randomization plan does not match the source run")
            if cache_metadata != source_cache_metadata:
                raise ValueError("Resume cache copy does not exactly match the source artifact metadata")
            cache_resume_mode = "reused_source_exactly"
            rng_stream_sources = {stream: "source_checkpoint" for stream in RANDOMIZATION_STREAMS}
        else:
            if cache_metadata["identity"]["num_envs"] != env_cfg.scene.num_envs:
                raise ValueError("Cross-scale resume target cache does not match target num_envs")
            cache_resume_mode = "regenerated_for_target_num_envs"
            rng_stream_sources = {
                stream: (
                    "target_process_start"
                    if stream in PROCESS_START_RANDOMIZATION_STREAMS
                    else "source_checkpoint"
                )
                for stream in RANDOMIZATION_STREAMS
            }
        resume_info = {
            "schema_version": RESUME_METADATA_VERSION,
            "mode": "weights_optimizer_rng_new_environment",
            "cache_resume_mode": cache_resume_mode,
            "source_checkpoint": str(resume_checkpoint),
            "source_checkpoint_sha256": _sha256(resume_checkpoint),
            "source_run_manifest": str(resume_manifest_path.resolve()),
            "source_run_manifest_sha256": _sha256(resume_manifest_path),
            "source_completed_iterations": starting_completed_iterations,
            "source_hardware_profile_name": source_hardware_profile_name,
            "target_hardware_profile_name": args_cli.profile,
            "source_num_envs": source_num_envs,
            "target_num_envs": env_cfg.scene.num_envs,
            "same_num_envs": same_num_envs,
            "source_cache_file_sha256": source_cache_metadata["file_sha256"],
            "source_cache_tensor_sha256": source_cache_metadata["tensor_sha256"],
            "target_cache_file_sha256": cache_metadata["file_sha256"],
            "target_cache_tensor_sha256": cache_metadata["tensor_sha256"],
            "source_realized_plan_hash": source_audit["realized_plan_hash"],
            "target_realized_plan_hash": randomization_audit["realized_plan_hash"],
            "rng_stream_sources": rng_stream_sources,
        }

    iterations_to_run = agent_cfg.max_iterations - starting_completed_iterations
    run_manifest_path = run_dir / "run_manifest.json"
    _write_manifest(
        run_manifest_path,
        profile_name=args_cli.profile,
        profile=profile,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        asset_bundle_hash=asset_report.bundle_hash,
        contract=contract,
        randomization_audit=randomization_audit,
        closed_chain_reset_cache=cache_metadata,
        resume=resume_info,
    )
    run_manifest_hash = _sha256(run_manifest_path)

    env = RslRlVecEnvWrapper(direct_env, clip_actions=agent_cfg.clip_actions)
    agent_dict = agent_cfg.to_dict()
    checkpoint_metadata = {
        "run_manifest_sha256": run_manifest_hash,
        "contract_hash": contract["contract_hash"],
        "closed_chain_reset_cache": checkpoint_cache_binding(cache_metadata),
        "seed": agent_cfg.seed,
        "hardware_profile": {
            "name": args_cli.profile,
            "declared_num_envs": profile["num_envs"],
            "effective_num_envs": env_cfg.scene.num_envs,
            "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "world_size": 1,
            "global_rank": 0,
            "local_rank": 0,
        },
    }
    runner = ContractOnPolicyRunner(
        env,
        agent_dict,
        log_dir=str(run_dir),
        device=agent_cfg.device,
        checkpoint_metadata_factory=lambda: checkpoint_metadata,
        environment_rng_state_factory=direct_env.get_randomization_rng_state,
    )
    restored_learning_rate = None
    if resume_checkpoint is not None and resume_metadata is not None:
        assert resume_info is not None
        streams_to_restore = (
            RANDOMIZATION_STREAMS if same_num_envs else RUNTIME_RANDOMIZATION_STREAMS
        )
        restored_iterations = resume_runner_from_checkpoint(
            runner,
            resume_checkpoint,
            resume_metadata,
            map_location=agent_cfg.device,
            restore_environment_rng_state=lambda payload: direct_env.set_randomization_rng_state(
                payload,
                stream_names=streams_to_restore,
            ),
        )
        if restored_iterations != starting_completed_iterations:
            raise RuntimeError("Resume helper restored an unexpected iteration count")
        restored_learning_rate = float(runner.alg.learning_rate)
        print(
            f"[INFO] Resuming after {starting_completed_iterations} completed iterations; "
            f"running {iterations_to_run} more to reach {agent_cfg.max_iterations}; "
            f"restored learning rate={restored_learning_rate:.12g}",
            flush=True,
        )
        direct_env.begin_resume_sequence_capture()
        env.reset()
    runner.learn(num_learning_iterations=iterations_to_run, init_at_random_ep_len=True)
    if resume_info is not None:
        trace_tensors = direct_env.get_resume_sequence_trace_tensors()
        expected_shapes = {
            "post_restore_command": (env_cfg.scene.num_envs, 3),
            "post_restore_root_velocity_world_usd": (env_cfg.scene.num_envs, 6),
            "post_restore_reset_discarded_policy_obs": (env_cfg.scene.num_envs, 25),
            "first_policy_observation": (env_cfg.scene.num_envs, 25),
        }
        for name, shape in expected_shapes.items():
            values = trace_tensors.get(name)
            if (
                not isinstance(values, torch.Tensor)
                or values.device.type != "cpu"
                or values.dtype != torch.float32
                or not values.is_contiguous()
                or tuple(values.shape) != shape
                or not torch.isfinite(values).all()
            ):
                raise RuntimeError(f"Resume-sequence trace tensor is invalid: {name}")
        trace_tensor_hash = canonical_tensor_sha256(trace_tensors)
        trace_artifact = {
            "schema_version": RESUME_SEQUENCE_TRACE_VERSION,
            "cache_resume_mode": resume_info["cache_resume_mode"],
            "rng_stream_sources": resume_info["rng_stream_sources"],
            **trace_tensors,
            "tensor_sha256": trace_tensor_hash,
        }
        trace_path = run_dir / RESUME_SEQUENCE_TRACE_FILENAME
        _atomic_save_cache(trace_path, trace_artifact)
        resume_sequence_trace_metadata = {
            "schema_version": RESUME_SEQUENCE_TRACE_VERSION,
            "relative_path": trace_path.relative_to(run_dir).as_posix(),
            "file_sha256": _sha256(trace_path),
            "tensor_sha256": trace_tensor_hash,
        }
    if runner.writer is not None:
        runner.writer.flush()
        runner.writer.close()

    duration_s = time.perf_counter() - start_time
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda item: int(item.stem.split("_")[-1]))
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not checkpoints:
        raise RuntimeError("PPO training finished without producing a checkpoint")
    if not event_files:
        raise RuntimeError("PPO training finished without producing a TensorBoard event file")
    for checkpoint in checkpoints:
        validate_checkpoint_metadata(checkpoint, run_manifest_path, contract)

    summary = {
        "run_dir": str(run_dir.resolve()),
        "iterations_requested": agent_cfg.max_iterations,
        "starting_completed_iterations": starting_completed_iterations,
        "iterations_executed": iterations_to_run,
        "completed_iterations": runner.current_learning_iteration + 1,
        "runner_final_iteration_index": runner.current_learning_iteration,
        "duration_seconds": duration_s,
        "final_checkpoint": str(checkpoints[-1].resolve()),
        "event_files": [str(path.resolve()) for path in event_files],
        "run_manifest_sha256": run_manifest_hash,
        "contract_hash": contract["contract_hash"],
        "closed_chain_reset_cache": {
            "path": str(cache_path.resolve()),
            "file_sha256": cache_metadata["file_sha256"],
            "tensor_sha256": cache_metadata["tensor_sha256"],
        },
        "resumed_from": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "restored_learning_rate": restored_learning_rate,
        "resume_sequence_trace": resume_sequence_trace_metadata,
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    latest_path = PROJECT_ROOT / "artifacts" / "phase1" / "latest_run.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir.resolve()), encoding="utf-8")
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
