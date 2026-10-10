from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import tensordict  # noqa: F401
import torch
from tensordict import TensorDict

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from isaaclab.app import AppLauncher

from wheelleg_dreamwaq.training.runtime import validate_runtime

parser = argparse.ArgumentParser(description="Run deterministic WheelLeg DreamWaQ inference in Isaac Lab.")
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--fixed-command",
    type=float,
    nargs=3,
    metavar=("VX", "YAW_RATE", "BASE_HEIGHT"),
    default=None,
)
parser.add_argument("--real-time", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
validate_runtime(PROJECT_ROOT, device=args_cli.device)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheelleg_dreamwaq.algorithms.dreamwaq.actor_critic import DreamWaQActorCritic
from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.deployment.observation_adapter import FrameMajorHistoryV1
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
    build_base_task_contract_from_configs,
    build_dreamwaq_algorithm_contract,
    build_dreamwaq_export_contract,
)
from wheelleg_dreamwaq.schemas.randomization import (
    FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
    NOMINAL_EVALUATION_PROFILE_V1,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.agents import WheelLegFlatDreamWaQRunnerCfg
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import (
    apply_task_profile, disable_command_practice, task_profile_from_contract,
)
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import validate_dreamwaq_checkpoint_metadata


def _latest_checkpoint() -> Path:
    latest = PROJECT_ROOT / "artifacts" / "phase2_dreamwaq" / "latest_run.txt"
    if not latest.is_file():
        raise FileNotFoundError("No latest DreamWaQ run is recorded; pass --checkpoint")
    run_dir = Path(latest.read_text(encoding="utf-8").strip())
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda item: int(item.stem.split("_")[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No DreamWaQ checkpoints found in {run_dir}")
    return checkpoints[-1]


def _build_policy(state_dict: dict[str, torch.Tensor], device: str) -> DreamWaQActorCritic:
    sample = TensorDict(
        {
            "policy": torch.zeros(1, 25, device=device),
            "policy_history": torch.zeros(1, 125, device=device),
            "critic": torch.zeros(1, 41, device=device),
        },
        batch_size=[1],
        device=device,
    )
    policy = DreamWaQActorCritic(sample, {"policy": ["policy"], "critic": ["critic"]}, 6).to(device)
    policy.load_state_dict(state_dict, strict=True)
    return policy.eval()


def _fixed_command(direct_env: WheelLegFlatEnv) -> torch.Tensor | None:
    if args_cli.fixed_command is None:
        return None
    values = tuple(float(value) for value in args_cli.fixed_command)
    ranges = (direct_env.cfg.commands.vx, direct_env.cfg.commands.yaw_rate, direct_env.cfg.commands.base_height)
    if any(not bounds[0] <= value <= bounds[1] for value, bounds in zip(values, ranges, strict=True)):
        raise ValueError("Fixed command is outside the frozen training command ranges")
    return torch.tensor(values, dtype=torch.float32, device=direct_env.device)


def _apply_command(direct_env: WheelLegFlatEnv, command: torch.Tensor | None) -> None:
    if command is not None:
        direct_env._commands.copy_(command.view(1, 3).expand_as(direct_env._commands))
        direct_env._current_state().command.copy_(direct_env._commands)


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("--num-envs and --steps must be positive")
    checkpoint = (args_cli.checkpoint or _latest_checkpoint()).resolve()
    run_manifest_path = checkpoint.parent / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    env_cfg = WheelLegFlatEnvCfg()
    env_cfg.seed = int(run_manifest["seed"])
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.randomization = FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    saved_profile = task_profile_from_contract(run_manifest["base_task_contract"])
    apply_task_profile(env_cfg, saved_profile)
    agent_cfg = WheelLegFlatDreamWaQRunnerCfg()
    agent_cfg.seed = int(run_manifest["seed"])
    agent_cfg.device = args_cli.device
    agent_cfg.algorithm.num_mini_batches = int(run_manifest["num_mini_batches"])
    agent_dict = agent_cfg.to_dict()
    base_contract = build_base_task_contract_from_configs(
        asset_bundle_hash=asset_report.bundle_hash,
        env_cfg=env_cfg,
        clip_actions=agent_cfg.clip_actions,
    )
    algorithm_contract = build_dreamwaq_algorithm_contract(agent_dict)
    export_contract = build_dreamwaq_export_contract()
    metadata = validate_dreamwaq_checkpoint_metadata(
        checkpoint,
        run_manifest_path,
        base_contract,
        algorithm_contract,
        export_contract,
        requested_seed=agent_cfg.seed,
    )
    payload = torch.load(checkpoint, map_location=args_cli.device, weights_only=False)
    policy = _build_policy(payload["model_state_dict"], args_cli.device)

    env_cfg.randomization = NOMINAL_EVALUATION_PROFILE_V1
    if args_cli.fixed_command is not None:
        disable_command_practice(env_cfg)
    direct_env = WheelLegFlatEnv(env_cfg)
    env = RslRlVecEnvWrapper(direct_env, clip_actions=agent_cfg.clip_actions)
    command = _fixed_command(direct_env)
    _apply_command(direct_env, command)
    base_observations = env.get_observations()
    history = FrameMajorHistoryV1(base_observations["policy"])
    reward_sum = torch.zeros(args_cli.num_envs, device=direct_env.device)
    reset_count = 0
    completed_steps = 0
    try:
        while simulation_app.is_running() and completed_steps < args_cli.steps:
            start = time.perf_counter()
            observations = TensorDict(
                {
                    "policy": base_observations["policy"],
                    "policy_history": history.flat(),
                    "critic": base_observations["critic"],
                },
                batch_size=[args_cli.num_envs],
            )
            with torch.inference_mode():
                actions = policy.act_inference(observations)
                next_observations, rewards, dones, _ = env.step(actions)
                _apply_command(direct_env, command)
                if command is not None:
                    next_observations = env.get_observations()
            if not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
                raise RuntimeError("DreamWaQ play produced NaN or Inf")
            history.append(next_observations["policy"], dones.to(dtype=torch.bool))
            base_observations = next_observations
            reward_sum += rewards
            reset_count += int(dones.sum().item())
            completed_steps += 1
            if args_cli.real_time:
                remaining = direct_env.step_dt - (time.perf_counter() - start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        env.close()
    report = {
        "schema_version": "DreamWaQPlayV1",
        "checkpoint": str(checkpoint),
        "completed_iterations": metadata["completed_iterations"],
        "num_envs": args_cli.num_envs,
        "task_profile": saved_profile,
        "runtime_command_practice": env_cfg.commands.practice_schedule,
        "completed_steps": completed_steps,
        "fixed_command": None if command is None else command.tolist(),
        "mean_reward_per_step": float((reward_sum / completed_steps).mean().item()),
        "reset_count": reset_count,
        "finite": True,
        "base_task_contract_hash": base_contract["contract_hash"],
        "dreamwaq_algorithm_contract_hash": algorithm_contract["contract_hash"],
        "dreamwaq_export_contract_hash": export_contract["contract_hash"],
        "estimator_monitor_state": metadata["estimator_monitor_state"],
    }
    if args_cli.output is not None:
        output = args_cli.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


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
