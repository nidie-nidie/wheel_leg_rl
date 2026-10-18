from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import tensordict  # noqa: F401
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CHECKPOINT = (
    PROJECT_ROOT
    / "logs"
    / "rsl_rl"
    / "wheelleg_flat_dreamwaq"
    / "2026-10-17_17-56-29_rtx5070_fudan-v1_baseline-yaw1-headingoff-env4096-fresh701-20261017"
    / "model_700.pt"
)
DEFAULT_RUN_DIR = PROJECT_ROOT / "debug" / "sim2sim" / "runs" / "constraint-finetune-4096-fresh20-20261018"
OLD_DEBUG_CHECKPOINT_SHA = "A351C42E999F873FDA0921940BA9F2C57FB2143485971DF50A8757B067507FB9"
FORMAL_CHECKPOINT_SHA = "21DEB4222A2B260008DE6B9829447C92E64B21BE268CF0AB66AF6E163062E984"
BASELINE_SPEED_MAE = 0.03731354048572227
BASELINE_QDDOT_RMS = 0.17272677856097493
QDDOT_BUDGET = 0.181363117489024
QDDOT_WINDOW_STEPS = 50
QDDOT_PENALTY_COEF = 1.0
REFERENCE_KL_COEF = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git_status() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT.parent), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    return [line for line in result.stdout.splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug-only constrained WheelLeg fine-tune.")
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=41882943)
    parser.add_argument("--profile", choices=("rtx5070",), default="rtx5070")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-profile", default="stop_reverse_raw_split_straight_yaw1_headingoff_v1")
    parser.add_argument("--qddot-budget", type=float, default=QDDOT_BUDGET)
    parser.add_argument("--qddot-penalty-coef", type=float, default=QDDOT_PENALTY_COEF)
    parser.add_argument("--reference-kl-coef", type=float, default=REFERENCE_KL_COEF)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_checkpoint.resolve()
    run_dir = args.run_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source}")
    source_sha = sha256_file(source)
    if source_sha != FORMAL_CHECKPOINT_SHA:
        raise ValueError(f"formal source checkpoint SHA mismatch: {source_sha}")
    if args.num_envs != 4096 or args.seed != 41882943 or args.max_iterations != 20:
        raise ValueError("debug smoke must use 4096 envs, seed 41882943, and 20 iterations")
    if args.reference_kl_coef != REFERENCE_KL_COEF:
        raise ValueError("reference KL coefficient is fixed for this smoke")
    if args.qddot_budget != QDDOT_BUDGET or args.qddot_penalty_coef != QDDOT_PENALTY_COEF:
        raise ValueError("qddot constraint constants are fixed for this smoke")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {run_dir}")
    return {
        "preflight_ok": True,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_sha,
        "old_debug_checkpoint_sha256_anchor": OLD_DEBUG_CHECKPOINT_SHA,
        "run_dir": str(run_dir),
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "device": args.device,
        "task_profile": args.task_profile,
        "qddot_window_steps": QDDOT_WINDOW_STEPS,
        "qddot_budget": args.qddot_budget,
        "qddot_penalty_coef": args.qddot_penalty_coef,
        "reference_kl_coef": args.reference_kl_coef,
        "reset_adam": True,
        "formal_tree_status_before": git_status(),
    }


def _copy_source_artifacts(source: Path, run_dir: Path) -> None:
    source_dir = source.parent
    for name in ("run_manifest.json", "closed_chain_reset_cache.pt"):
        shutil.copy2(source_dir / name, run_dir / name)
    source_params = source_dir / "params"
    if source_params.is_dir():
        shutil.copytree(source_params, run_dir / "params")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def run_training(args: argparse.Namespace, app: Any) -> Path:
    import numpy as np
    import torch
    from isaaclab.utils.io import dump_yaml

    from wheelleg_dreamwaq.algorithms.dreamwaq.registration import register_dreamwaq_rsl_rl_classes
    from wheelleg_dreamwaq.algorithms.dreamwaq.history_wrapper import DreamWaQHistoryVecEnvWrapper
    from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
    from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
        build_base_task_contract_from_configs,
        build_dreamwaq_algorithm_contract,
        build_dreamwaq_export_contract,
    )
    from wheelleg_dreamwaq.schemas.randomization import (
        FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
        canonical_tensor_sha256,
    )
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.agents import WheelLegFlatDreamWaQRunnerCfg
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import (
        apply_profile_to_runner,
        apply_task_profile,
    )
    from wheelleg_dreamwaq.training.checkpoint import (
        checkpoint_cache_binding,
        closed_chain_reset_cache_metadata,
        load_closed_chain_reset_cache_artifact,
        sha256_file as checkpoint_sha256_file,
    )
    import wheelleg_dreamwaq.training.dreamwaq_checkpoint as checkpoint_module
    from wheelleg_dreamwaq.training.dreamwaq_checkpoint import (
        DreamWaQContractOnPolicyRunner,
        fresh_resume_provenance,
        load_dreamwaq_checkpoint_artifact,
    )
    from debug.sim2sim.constraint_finetune_env import make_constrained_wheelleg_env_class
    from debug.sim2sim.constraint_finetune_ppo import ConstrainedDreamWaQPPO

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    source = args.source_checkpoint.resolve()
    payload, source_manifest, source_metadata = load_dreamwaq_checkpoint_artifact(source)
    source_manifest_path = source.parent / "run_manifest.json"
    source_cache_artifact, source_cache_metadata, source_cache_path = load_closed_chain_reset_cache_artifact(
        source_manifest_path, manifest=source_manifest
    )
    initial_model_state = payload["model_state_dict"]
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    _copy_source_artifacts(source, run_dir)
    copied_manifest_path = run_dir / "run_manifest.json"
    copied_manifest_hash = checkpoint_sha256_file(copied_manifest_path)
    if copied_manifest_hash != checkpoint_sha256_file(source_manifest_path):
        raise RuntimeError("copied source run manifest hash changed")
    copied_cache_path = run_dir / "closed_chain_reset_cache.pt"
    copied_cache_metadata = closed_chain_reset_cache_metadata(copied_cache_path, run_dir=run_dir)
    if copied_cache_metadata != source_cache_metadata:
        raise RuntimeError("copied reset cache metadata differs from source")

    profile = {"num_envs": 4096, "num_mini_batches": 8}
    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    env_cfg = WheelLegFlatEnvCfg()
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.randomization = FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    env_cfg.log_dir = str(run_dir)
    apply_task_profile(env_cfg, args.task_profile)
    agent_cfg = WheelLegFlatDreamWaQRunnerCfg()
    apply_profile_to_runner(agent_cfg, args.task_profile)
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.algorithm.num_mini_batches = profile["num_mini_batches"]
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.run_name = "constraint-finetune-4096-fresh20-20261018"
    base_task_contract = build_base_task_contract_from_configs(
        asset_bundle_hash=asset_report.bundle_hash,
        env_cfg=env_cfg,
        clip_actions=agent_cfg.clip_actions,
    )
    algorithm_contract = build_dreamwaq_algorithm_contract(agent_cfg.to_dict())
    export_contract = build_dreamwaq_export_contract(
        heading_hold=base_task_contract["task"].get("heading_hold")
    )
    for label, current, saved in (
        ("base task", base_task_contract, source_manifest["base_task_contract"]),
        ("algorithm", algorithm_contract, source_manifest["dreamwaq_algorithm_contract"]),
        ("export", export_contract, source_manifest["dreamwaq_export_contract"]),
    ):
        if current["contract_hash"] != saved["contract_hash"]:
            raise RuntimeError(f"{label} contract differs from source checkpoint")
    if source_metadata["completed_iterations"] != 701:
        raise RuntimeError("source checkpoint is not the completed 701-update model")

    params_dir = run_dir / "params"
    params_dir.mkdir(exist_ok=True)
    dump_yaml(str(params_dir / "env_debug.yaml"), env_cfg)
    dump_yaml(str(params_dir / "agent_debug.yaml"), agent_cfg)
    (params_dir / "hardware_debug.yaml").write_text("num_envs: 4096\nnum_mini_batches: 8\n", encoding="utf-8")

    direct_env_class = make_constrained_wheelleg_env_class()
    direct_env = direct_env_class(
        env_cfg,
        closed_chain_reset_cache=source_cache_artifact,
        qddot_budget=args.qddot_budget,
        qddot_penalty_coef=args.qddot_penalty_coef,
    )
    randomization_audit = direct_env.randomization_audit
    if randomization_audit["realized_plan_hash"] != source_manifest["randomization_audit"]["realized_plan_hash"]:
        raise RuntimeError("debug environment randomization plan differs from 701 source")
    env = DreamWaQHistoryVecEnvWrapper(direct_env, clip_actions=agent_cfg.clip_actions)

    original_register = register_dreamwaq_rsl_rl_classes

    def register_debug_classes() -> None:
        original_register()
        import rsl_rl.runners.on_policy_runner as on_policy_runner
        on_policy_runner.DreamWaQPPO = ConstrainedDreamWaQPPO

    checkpoint_module.register_dreamwaq_rsl_rl_classes = register_debug_classes

    checkpoint_metadata = {
        "seed": agent_cfg.seed,
        "run_manifest_sha256": copied_manifest_hash,
        "base_task_contract_version": base_task_contract["manifest_version"],
        "base_task_contract_hash": base_task_contract["contract_hash"],
        "dreamwaq_algorithm_contract_version": algorithm_contract["manifest_version"],
        "dreamwaq_algorithm_contract_hash": algorithm_contract["contract_hash"],
        "dreamwaq_export_contract_version": export_contract["manifest_version"],
        "hardware_profile": {
            "name": args.profile,
            "declared_num_envs": profile["num_envs"],
            "effective_num_envs": args.num_envs,
            "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "world_size": 1,
            "global_rank": 0,
            "local_rank": 0,
        },
        "reset_cache_binding": checkpoint_cache_binding(copied_cache_metadata),
        "resume_provenance": fresh_resume_provenance(),
    }
    runner = DreamWaQContractOnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=str(run_dir),
        device=agent_cfg.device,
        checkpoint_metadata_factory=lambda: checkpoint_metadata,
        environment_rng_state_factory=direct_env.get_randomization_rng_state,
    )
    runner.alg.policy.load_state_dict(initial_model_state, strict=True)
    runner.alg.set_reference_from_policy()
    if len(runner.alg.optimizer.state) != 0:
        raise RuntimeError("debug fine-tune optimizer was not fresh")
    source_model_sha = checkpoint_sha256_file(source)
    debug_manifest = {
        "schema_version": "WheelLegDebugConstraintFineTuneV1",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_model_sha,
        "old_debug_checkpoint_sha256_anchor": OLD_DEBUG_CHECKPOINT_SHA,
        "source_completed_iterations": int(source_metadata["completed_iterations"]),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "max_iterations": args.max_iterations,
        "profile": args.profile,
        "task_profile": args.task_profile,
        "device": args.device,
        "reset_adam": True,
        "qddot_window_steps": QDDOT_WINDOW_STEPS,
        "qddot_budget": args.qddot_budget,
        "qddot_penalty_coef": args.qddot_penalty_coef,
        "reference_kl_coef": args.reference_kl_coef,
        "baseline_speed_mae": BASELINE_SPEED_MAE,
        "baseline_qddot_rms": BASELINE_QDDOT_RMS,
        "source_base_task_contract_hash": source_manifest["base_task_contract"]["contract_hash"],
        "target_base_task_contract_hash": base_task_contract["contract_hash"],
        "formal_tree_status_before": git_status(),
        "command": sys.argv,
    }
    _write_json(run_dir / "debug_constraint_manifest.json", debug_manifest)

    update_metrics: list[dict[str, Any]] = []
    original_update = runner.alg.update

    def recording_update() -> dict[str, float]:
        result = original_update()
        update_metrics.append({
            "completed_update": len(update_metrics) + 1,
            **{key: float(value) for key, value in result.items()},
        })
        return result

    runner.alg.update = recording_update
    started = time.perf_counter()
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        final_checkpoint = run_dir / f"model_{runner.current_learning_iteration}.pt"
        runner.save(str(final_checkpoint))
        records = direct_env.drain_debug_metric_records()
        with (run_dir / "constraint_metrics.jsonl").open("w", encoding="utf-8") as stream:
            for index, record in enumerate(records, start=1):
                stream.write(json.dumps({"rollout_step": index, **record}) + "\n")
        with (run_dir / "update_metrics.jsonl").open("w", encoding="utf-8") as stream:
            for record in update_metrics:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        summary = {
            "run_dir": str(run_dir),
            "iterations_requested": args.max_iterations,
            "iterations_executed": runner.current_learning_iteration + 1,
            "completed_iterations": runner.current_learning_iteration + 1,
            "final_checkpoint": str(final_checkpoint),
            "optimizer_state_entries": len(runner.alg.optimizer.state),
            "optimizer_steps": sorted({
                int(value["step"].item())
                for value in runner.alg.optimizer.state.values()
                if isinstance(value.get("step"), torch.Tensor)
            }),
            "reset_adam": True,
            "source_checkpoint_sha256": source_model_sha,
            "duration_seconds": time.perf_counter() - started,
        }
        _write_json(run_dir / "training_summary.json", summary)
        (run_dir / "final_model_path.txt").write_text(str(final_checkpoint.resolve()), encoding="utf-8")
        return run_dir
    except Exception as error:
        _write_json(run_dir / "failure.json", {
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_updates_observed": len(update_metrics),
        })
        raise


def main() -> int:
    args = parse_args()
    if args.preflight:
        print(json.dumps(run_preflight(args), indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        print(json.dumps({**run_preflight(args), "dry_run_ok": True}, indent=2, sort_keys=True))
        return 0
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        run_training(args, simulation_app)
    finally:
        simulation_app.close(skip_cleanup=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





