from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import tensordict  # noqa: F401
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CHECKPOINT = (
    PROJECT_ROOT / "logs" / "rsl_rl" / "wheelleg_flat_dreamwaq"
    / "2026-10-17_17-56-29_rtx5070_fudan-v1_baseline-yaw1-headingoff-env4096-fresh701-20261017"
    / "model_700.pt"
)
DEFAULT_RUN_DIR = PROJECT_ROOT / "debug" / "sim2sim" / "runs" / "isaac-frozen-policy-probe-20261019"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug-only frozen DreamWaQ Isaac rollout probe.")
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=41882943)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window-steps", type=int, default=50)
    parser.add_argument("--policy-mode", choices=("mean", "sampled", "both"), default="both")
    parser.add_argument("--enable-external-forces-every-iteration", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def make_runner(args: argparse.Namespace, run_dir: Path):
    import numpy as np
    from wheelleg_dreamwaq.algorithms.dreamwaq.history_wrapper import DreamWaQHistoryVecEnvWrapper
    from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
    from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
        build_base_task_contract_from_configs,
        build_dreamwaq_algorithm_contract,
        build_dreamwaq_export_contract,
    )
    from wheelleg_dreamwaq.schemas.randomization import FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.agents import WheelLegFlatDreamWaQRunnerCfg
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import apply_profile_to_runner, apply_task_profile
    from wheelleg_dreamwaq.training.checkpoint import load_closed_chain_reset_cache_artifact
    from wheelleg_dreamwaq.training.dreamwaq_checkpoint import (
        DreamWaQContractOnPolicyRunner,
        load_dreamwaq_checkpoint_artifact,
    )

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
    if int(source_metadata["completed_iterations"]) != 701:
        raise RuntimeError("source checkpoint is not the completed 701-update model")
    source_manifest_path = source.parent / "run_manifest.json"
    cache_artifact, _, _ = load_closed_chain_reset_cache_artifact(source_manifest_path, manifest=source_manifest)

    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    env_cfg = WheelLegFlatEnvCfg()
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.randomization = FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    env_cfg.log_dir = str(run_dir)
    apply_task_profile(env_cfg, "stop_reverse_raw_split_straight_yaw1_headingoff_v1")
    agent_cfg = WheelLegFlatDreamWaQRunnerCfg()
    apply_profile_to_runner(agent_cfg, "stop_reverse_raw_split_straight_yaw1_headingoff_v1")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.algorithm.num_mini_batches = 8
    agent_cfg.max_iterations = 1
    agent_cfg.run_name = "isaac-frozen-policy-probe-20261019"

    current_base = build_base_task_contract_from_configs(
        asset_bundle_hash=asset_report.bundle_hash,
        env_cfg=env_cfg,
        clip_actions=agent_cfg.clip_actions,
    )
    current_algorithm = build_dreamwaq_algorithm_contract(agent_cfg.to_dict())
    current_export = build_dreamwaq_export_contract(heading_hold=current_base["task"].get("heading_hold"))
    for label, current, saved in (
        ("base task", current_base, source_manifest["base_task_contract"]),
        ("algorithm", current_algorithm, source_manifest["dreamwaq_algorithm_contract"]),
        ("export", current_export, source_manifest["dreamwaq_export_contract"]),
    ):
        if current["contract_hash"] != saved["contract_hash"]:
            raise RuntimeError(f"{label} contract differs from source checkpoint")

    if args.enable_external_forces_every_iteration:
        env_cfg.sim.physx.enable_external_forces_every_iteration = True
    direct_env = WheelLegFlatEnv(env_cfg, closed_chain_reset_cache=cache_artifact)
    env = DreamWaQHistoryVecEnvWrapper(direct_env, clip_actions=agent_cfg.clip_actions)
    runner = DreamWaQContractOnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=str(run_dir),
        device=agent_cfg.device,
        checkpoint_metadata_factory=lambda: {},
        environment_rng_state_factory=direct_env.get_randomization_rng_state,
    )
    runner.alg.policy.load_state_dict(payload["model_state_dict"], strict=True)
    runner.alg.policy.eval()
    return runner, direct_env, env, source


def set_command(direct_env: Any, command: tuple[float, float, float]) -> None:
    command_tensor = torch.tensor(command, dtype=torch.float32, device=direct_env.device).expand(direct_env.num_envs, -1).clone()
    direct_env.set_source_command(command_tensor)


def run_segment(runner: Any, direct_env: Any, env: Any, *, label: str, command: tuple[float, float, float], policy_mode: str, steps: int, window_steps: int) -> dict[str, Any]:
    obs, _ = env.reset()
    set_command(direct_env, command)
    policy = runner.alg.policy
    qddot_window = torch.zeros((window_steps, direct_env.num_envs, 4), device=direct_env.device)
    write_index = 0
    previous_velocity = None
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(steps):
        with torch.inference_mode():
            if policy_mode == "mean":
                action = policy.act_inference(obs)
            elif policy_mode == "sampled":
                action, _ = policy.act_with_context(obs)
            else:
                raise ValueError(policy_mode)
        obs, reward, dones, extras = env.step(action)
        state = direct_env._current_state()
        qddot = state.joint_acceleration[:, :4].detach()
        velocity = state.joint_velocity[:, :4].detach()
        qddot_window[write_index] = qddot
        write_index = (write_index + 1) % window_steps
        count = min(step + 1, window_steps)
        if step + 1 < window_steps:
            window_values = qddot_window[:count]
        else:
            window_values = qddot_window
        window_rms_per_env = window_values.square().mean(dim=(0, 2)).sqrt()
        manual_diff = None
        if previous_velocity is not None:
            manual_diff = (velocity - previous_velocity) / float(direct_env.step_dt)
        previous_velocity = velocity.clone()
        speed_target = state.command[:, 0]
        speed = state.root_com_linear_velocity[:, 0]
        reward_terms = {
            str(key): float(value.item())
            for key, value in direct_env.extras.get("log", {}).items()
            if str(key).startswith("Reward/") and isinstance(value, torch.Tensor) and value.ndim == 0
        }
        records.append({
            "step": step + 1,
            "qddot_rms_all_env": float(qddot.square().mean().sqrt().item()),
            "qddot_window_rms_mean_env": float(window_rms_per_env.mean().item()),
            "qddot_window_rms_p50_env": float(torch.quantile(window_rms_per_env, 0.50).item()),
            "qddot_window_rms_p95_env": float(torch.quantile(window_rms_per_env, 0.95).item()),
            "qddot_window_rms_p99_env": float(torch.quantile(window_rms_per_env, 0.99).item()),
            "qddot_abs_p95": float(torch.quantile(qddot.abs().reshape(-1), 0.95).item()),
            "reward_mean": float(reward.mean().item()),
            "reward_terms": reward_terms,
            "speed_mae": float((speed - speed_target).abs().mean().item()),
            "done_fraction": float(dones.float().mean().item()),
            "manual_diff_rms_error": None if manual_diff is None else float((manual_diff - qddot).square().mean().sqrt().item()),
            "manual_diff_rms": None if manual_diff is None else float(manual_diff.square().mean().sqrt().item()),
            "joint_velocity_rms": float(velocity.square().mean().sqrt().item()),
            "joint_acceleration_rms": float(qddot.square().mean().sqrt().item()),
        })
    tail = records[window_steps:]
    if not tail:
        tail = records
    def avg(key: str) -> float:
        return float(sum(float(r[key]) for r in tail if r[key] is not None) / max(1, sum(r[key] is not None for r in tail)))
    reward_term_keys = sorted({key for record in tail for key in record.get("reward_terms", {})})
    reward_terms_tail = {
        key: float(sum(record["reward_terms"].get(key, 0.0) for record in tail) / max(1, len(tail)))
        for key in reward_term_keys
    }
    return {
        "label": label,
        "command": list(command),
        "policy_mode": policy_mode,
        "steps": steps,
        "warmup_excluded_steps": min(window_steps, steps),
        "physics_dt": float(direct_env.physics_dt),
        "control_step_dt": float(direct_env.step_dt),
        "control_decimation": int(direct_env.cfg.decimation),
        "qddot_window_rms_mean_env_tail": avg("qddot_window_rms_mean_env"),
        "qddot_window_rms_p50_env_tail": avg("qddot_window_rms_p50_env"),
        "qddot_window_rms_p95_env_tail": avg("qddot_window_rms_p95_env"),
        "qddot_window_rms_p99_env_tail": avg("qddot_window_rms_p99_env"),
        "qddot_rms_all_env_tail": avg("qddot_rms_all_env"),
        "qddot_abs_p95_tail": avg("qddot_abs_p95"),
        "reward_mean_tail": avg("reward_mean"),
        "reward_terms_tail": reward_terms_tail,
        "speed_mae_tail": avg("speed_mae"),
        "done_fraction_tail": avg("done_fraction"),
        "manual_diff_rms_error_tail": avg("manual_diff_rms_error"),
        "manual_diff_rms_tail": avg("manual_diff_rms"),
        "joint_velocity_rms_tail": avg("joint_velocity_rms"),
        "joint_acceleration_rms_tail": avg("joint_acceleration_rms"),
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {run_dir}")
    if args.num_envs != 4096 or args.seed != 41882943:
        raise ValueError("probe is fixed to 4096 envs and seed 41882943")
    run_dir.mkdir(parents=True, exist_ok=False)
    import isaaclab.app
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        runner, direct_env, env, source = make_runner(args, run_dir)
        segments = []
        for label, command in (("nominal_stand", (0.0, 0.0, 0.2)), ("forward", (1.0, 0.0, 0.2))):
            for mode in (("mean", "sampled") if args.policy_mode == "both" else (args.policy_mode,)):
                result = run_segment(
                    runner,
                    direct_env,
                    env,
                    label=label,
                    command=command,
                    policy_mode=mode,
                    steps=args.steps,
                    window_steps=args.window_steps,
                )
                records_path = run_dir / f"{label}_{mode}_records.jsonl"
                with records_path.open("w", encoding="utf-8") as f:
                    for record in result.pop("records"):
                        f.write(json.dumps(record, sort_keys=True) + "\n")
                segments.append(result)
        manifest = {
            "schema_version": "IsaacFrozenPolicyProbeV1",
            "created_at": datetime.now().astimezone().isoformat(),
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": sha256_file(source),
            "num_envs": args.num_envs,
            "seed": args.seed,
            "device": args.device,
            "task_profile": "stop_reverse_raw_split_straight_yaw1_headingoff_v1",
            "policy_updates": 0,
            "optimizer_updates": 0,
            "policy_modes": (["mean", "sampled"] if args.policy_mode == "both" else [args.policy_mode]),
            "window_steps": args.window_steps,
            "enable_external_forces_every_iteration": bool(args.enable_external_forces_every_iteration),
            "segments": segments,
            "command": sys.argv,
            "script_sha256": sha256_file(Path(__file__).resolve()),
        }
        (run_dir / "probe_summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({k: v for k, v in manifest.items() if k != "segments"}, indent=2, sort_keys=True))
        for segment in segments:
            print(json.dumps(segment, sort_keys=True))
    finally:
        simulation_app.close(skip_cleanup=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





