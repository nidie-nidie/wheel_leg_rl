from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import tensordict  # noqa: F401
import torch
from tensordict import TensorDict

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from isaaclab.app import AppLauncher

from wheelleg_dreamwaq.training.runtime import validate_runtime

parser = argparse.ArgumentParser(description="Run the frozen deterministic WheelLeg Isaac evaluation.")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--reset-cache", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--baseline-report", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
validate_runtime(PROJECT_ROOT, device=args_cli.device)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.modules import ActorCritic

from wheelleg_dreamwaq.algorithms.dreamwaq.actor_critic import DreamWaQActorCritic
from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.deployment.observation_adapter import FrameMajorHistoryV1
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
    build_base_task_contract_from_configs,
    build_base_task_contract_from_phase1_contract,
    build_dreamwaq_algorithm_contract,
    build_dreamwaq_export_contract,
    stable_contract_hash,
)
from wheelleg_dreamwaq.schemas.isaac_evaluation import (
    EVALUATION_ACTION_STEPS,
    EVALUATION_ENV_COUNT,
    EVALUATION_SEED,
    FORMAL_SCENARIOS,
    ISAAC_EVALUATION_SCHEMA_VERSION,
    PHASE1R_ISAAC_BASELINE,
    SampleWeightedVelocityMSE,
    aggregate_scenarios,
    build_evaluation_reset_cache_identity,
    build_evaluation_source_fingerprint,
    build_isaac_evaluation_contract,
    candidate_not_worse_than_baseline,
    summarize_scenario,
    validate_matching_isaac_evaluation_contracts,
)
from wheelleg_dreamwaq.schemas.manifest import build_phase1_contract_from_configs
from wheelleg_dreamwaq.schemas.randomization import (
    NOMINAL_EVALUATION_PROFILE_V1,
    RandomizationProfileV1,
    profile_contract_hash,
    validate_closed_chain_reset_cache_artifact,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.agents import (
    WheelLegFlatDreamWaQRunnerCfg,
    WheelLegFlatPPORunnerCfg,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
from wheelleg_dreamwaq.training.checkpoint import (
    load_checkpoint_artifact,
    sha256_file,
    validate_checkpoint_metadata,
)
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import validate_dreamwaq_checkpoint_metadata


def _atomic_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _source_fingerprint() -> dict:
    files = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "source" / "wheelleg_dreamwaq" / "wheelleg_dreamwaq" / "schemas" / "isaac_evaluation.py",
        PROJECT_ROOT / "source" / "wheelleg_dreamwaq" / "wheelleg_dreamwaq" / "schemas" / "observation.py",
        PROJECT_ROOT / "source" / "wheelleg_dreamwaq" / "wheelleg_dreamwaq" / "deployment" / "observation_adapter.py",
    )
    return build_evaluation_source_fingerprint(
        {path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in files}
    )


def _dreamwaq_policy(state: dict[str, torch.Tensor], device: str) -> DreamWaQActorCritic:
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
    policy.load_state_dict(state, strict=True)
    return policy.eval()


def _phase1_policy(contract: dict, state: dict[str, torch.Tensor], device: str) -> ActorCritic:
    sample = TensorDict(
        {
            "policy": torch.zeros(1, contract["observation"]["actor_dimension"], device=device),
            "critic": torch.zeros(1, contract["observation"]["critic_dimension"], device=device),
        },
        batch_size=[1],
        device=device,
    )
    policy_cfg = contract["training"]["policy"]
    network = contract["network"]
    policy = ActorCritic(
        sample,
        contract["training"]["runner"]["obs_groups"],
        contract["action"]["dimension"],
        actor_obs_normalization=network["actor_obs_normalization"],
        critic_obs_normalization=network["critic_obs_normalization"],
        actor_hidden_dims=network["actor_hidden_dims"],
        critic_hidden_dims=network["critic_hidden_dims"],
        activation=network["activation"],
        init_noise_std=policy_cfg["init_noise_std"],
        noise_std_type=policy_cfg["noise_std_type"],
        state_dependent_std=policy_cfg["state_dependent_std"],
    ).to(device)
    policy.load_state_dict(state)
    return policy.eval()


def _load_policy_and_contracts(checkpoint: Path, asset_hash: str) -> tuple[str, torch.nn.Module, dict, dict]:
    run_manifest_path = checkpoint.parent / "run_manifest.json"
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if "base_task_contract" in manifest:
        base = manifest["base_task_contract"]
        env_cfg = WheelLegFlatEnvCfg()
        env_cfg.seed = int(manifest["seed"])
        env_cfg.scene.num_envs = EVALUATION_ENV_COUNT
        env_cfg.sim.device = args_cli.device
        profile_values = base["randomization"]["profile_contract"]["profile"]
        env_cfg.randomization = RandomizationProfileV1(**profile_values)
        agent_cfg = WheelLegFlatDreamWaQRunnerCfg()
        agent_cfg.seed = int(manifest["seed"])
        agent_cfg.device = args_cli.device
        agent_cfg.algorithm.num_mini_batches = int(manifest["num_mini_batches"])
        agent_dict = agent_cfg.to_dict()
        current_base = build_base_task_contract_from_configs(
            asset_bundle_hash=asset_hash,
            env_cfg=env_cfg,
            clip_actions=agent_cfg.clip_actions,
        )
        current_algorithm = build_dreamwaq_algorithm_contract(agent_dict)
        current_export = build_dreamwaq_export_contract()
        metadata = validate_dreamwaq_checkpoint_metadata(
            checkpoint,
            run_manifest_path,
            current_base,
            current_algorithm,
            current_export,
            requested_seed=int(manifest["seed"]),
        )
        payload = torch.load(checkpoint, map_location=args_cli.device, weights_only=False)
        return "dreamwaq", _dreamwaq_policy(payload["model_state_dict"], args_cli.device), base, metadata

    payload, manifest, metadata = load_checkpoint_artifact(checkpoint, run_manifest_path)
    contract = manifest["contract"]
    if sha256_file(checkpoint) != PHASE1R_ISAAC_BASELINE["checkpoint_sha256"]:
        raise ValueError("Isaac Phase 1R evaluation only accepts the frozen run-03 baseline checkpoint")
    if contract["contract_hash"] != PHASE1R_ISAAC_BASELINE["phase1_contract_hash"]:
        raise ValueError("Frozen Phase 1R baseline contract hash differs")
    if sha256_file(run_manifest_path) != PHASE1R_ISAAC_BASELINE["run_manifest_sha256"]:
        raise ValueError("Frozen Phase 1R baseline run manifest hash differs")
    profile_values = contract["randomization"]["profile_contract"]["profile"]
    env_cfg = WheelLegFlatEnvCfg()
    env_cfg.seed = int(metadata["seed"])
    env_cfg.scene.num_envs = EVALUATION_ENV_COUNT
    env_cfg.sim.device = args_cli.device
    env_cfg.randomization = RandomizationProfileV1(**profile_values)
    agent_cfg = WheelLegFlatPPORunnerCfg()
    agent_cfg.seed = int(metadata["seed"])
    agent_cfg.device = args_cli.device
    agent_cfg.algorithm.num_mini_batches = int(metadata["hardware_profile"]["num_mini_batches"])
    current_contract = build_phase1_contract_from_configs(
        asset_bundle_hash=asset_hash,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
    )
    validate_checkpoint_metadata(checkpoint, run_manifest_path, current_contract)
    base = build_base_task_contract_from_phase1_contract(contract)
    return "phase1r_baseline", _phase1_policy(contract, payload["model_state_dict"], args_cli.device), base, metadata


def _termination_reason(extras: dict, env_index: int) -> str:
    diagnostics = extras.get("termination_diagnostics", {})
    for name in (
        "invalid",
        "height_terminated",
        "tilt_terminated",
        "root_linear_terminated",
        "root_angular_terminated",
        "joint_velocity_terminated",
    ):
        values = diagnostics.get(name)
        if isinstance(values, torch.Tensor) and bool(values[env_index].item()):
            return name
    return "terminated"


def _load_baseline_report(path: Path, evaluation_contract: dict) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    embedded_hash = report.get("report_hash")
    calculated = stable_contract_hash({key: value for key, value in report.items() if key != "report_hash"})
    if embedded_hash != calculated:
        raise ValueError("Isaac baseline report hash is invalid")
    if report.get("policy_kind") != "phase1r_baseline":
        raise ValueError("Isaac comparison report is not Phase1RIsaacBaselineV1")
    if report.get("source_checkpoint_sha256") != PHASE1R_ISAAC_BASELINE["checkpoint_sha256"]:
        raise ValueError("Isaac baseline report checkpoint identity differs")
    saved_contract = report.get("evaluation_contract")
    matching_hash = validate_matching_isaac_evaluation_contracts(saved_contract, evaluation_contract)
    if report.get("evaluation_contract_hash") != matching_hash:
        raise ValueError("Isaac baseline report evaluation contract hash is invalid")
    return report


def main() -> None:
    checkpoint = args_cli.checkpoint.resolve()
    cache_path = args_cli.reset_cache.resolve()
    output = args_cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    policy_kind, policy, base_task_contract, checkpoint_metadata = _load_policy_and_contracts(
        checkpoint, asset_report.bundle_hash
    )

    cache_artifact = None
    if cache_path.is_file():
        cache_artifact = validate_closed_chain_reset_cache_artifact(
            torch.load(cache_path, map_location="cpu", weights_only=False)
        )
    env_cfg = WheelLegFlatEnvCfg()
    env_cfg.seed = EVALUATION_SEED
    env_cfg.scene.num_envs = EVALUATION_ENV_COUNT
    env_cfg.sim.device = args_cli.device
    env_cfg.randomization = NOMINAL_EVALUATION_PROFILE_V1
    direct_env = WheelLegFlatEnv(env_cfg, closed_chain_reset_cache=cache_artifact)
    if cache_artifact is None:
        cache_artifact = direct_env.closed_chain_reset_cache_artifact
        _atomic_save(cache_path, cache_artifact)
    if cache_artifact["identity"]["master_seed"] != EVALUATION_SEED:
        raise ValueError("Isaac evaluation reset cache seed differs from the frozen seed")
    if cache_artifact["identity"]["num_envs"] != EVALUATION_ENV_COUNT:
        raise ValueError("Isaac evaluation reset cache must contain eight environments")
    if cache_artifact["identity"]["randomization_profile_hash"] != profile_contract_hash(
        NOMINAL_EVALUATION_PROFILE_V1
    ):
        raise ValueError("Isaac evaluation reset cache was not generated with NominalEvaluationProfileV1")
    cache_identity = build_evaluation_reset_cache_identity(
        path=str(cache_path),
        file_sha256=sha256_file(cache_path),
        cache_payload=cache_artifact,
    )
    evaluation_contract = build_isaac_evaluation_contract(
        base_task_contract=base_task_contract,
        evaluation_source=_source_fingerprint(),
        reset_cache_identity=cache_identity,
    )

    env = RslRlVecEnvWrapper(direct_env, clip_actions=1.0)
    commands = torch.tensor([command for _, command in FORMAL_SCENARIOS], dtype=torch.float32, device=direct_env.device)
    direct_env._commands.copy_(commands)
    base_observations = env.get_observations()
    expected_commands = env_cfg.normalization.normalize_command(commands)
    if not torch.allclose(base_observations["policy"][:, 6:9], expected_commands, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("Fixed evaluation commands were not present in the first policy observation")
    history = FrameMajorHistoryV1(base_observations["policy"]) if policy_kind == "dreamwaq" else None
    velocity = SampleWeightedVelocityMSE() if policy_kind == "dreamwaq" else None
    active = torch.ones(EVALUATION_ENV_COUNT, dtype=torch.bool, device=direct_env.device)
    reward_sums = torch.zeros(EVALUATION_ENV_COUNT, dtype=torch.float64, device=direct_env.device)
    survival_steps = torch.zeros(EVALUATION_ENV_COUNT, dtype=torch.int64, device=direct_env.device)
    done_steps: list[int | None] = [None] * EVALUATION_ENV_COUNT
    terminated_flags = [False] * EVALUATION_ENV_COUNT
    truncated_flags = [False] * EVALUATION_ENV_COUNT
    failure_reasons: list[str | None] = [None] * EVALUATION_ENV_COUNT
    try:
        for action_step in range(1, EVALUATION_ACTION_STEPS + 1):
            active_before = active.clone()
            if not active_before.any():
                break
            if policy_kind == "dreamwaq":
                assert history is not None and velocity is not None
                observations = TensorDict(
                    {
                        "policy": base_observations["policy"],
                        "policy_history": history.flat(),
                        "critic": base_observations["critic"],
                    },
                    batch_size=[EVALUATION_ENV_COUNT],
                )
                with torch.inference_mode():
                    actions, estimated_velocity, _ = policy.act_inference_with_estimator(observations)
                velocity.update(estimated_velocity, base_observations["critic"][:, 25:28], active_before)
            else:
                with torch.inference_mode():
                    actions = policy.act_inference(base_observations)
            actions = actions.clone()
            actions[~active_before] = 0.0
            next_observations, rewards, dones, extras = env.step(actions)
            if not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
                raise RuntimeError("Isaac evaluation produced NaN or Inf")
            reward_sums[active_before] += rewards[active_before].to(dtype=torch.float64)
            survival_steps[active_before] += 1
            truncated = extras["time_outs"].to(dtype=torch.bool)
            done = dones.to(dtype=torch.bool)
            terminated = done & ~truncated
            for index in torch.nonzero(active_before & done, as_tuple=False).flatten().tolist():
                done_steps[index] = action_step
                terminated_flags[index] = bool(terminated[index].item())
                truncated_flags[index] = bool(truncated[index].item())
                if terminated_flags[index]:
                    failure_reasons[index] = _termination_reason(extras, index)
                elif action_step != EVALUATION_ACTION_STEPS:
                    failure_reasons[index] = "early_timeout"
            active &= ~done
            if history is not None:
                history.append(next_observations["policy"], done)
            base_observations = next_observations
    finally:
        env.close()

    scenario_summaries = []
    for index, (name, command) in enumerate(FORMAL_SCENARIOS):
        summary = summarize_scenario(
            name=name,
            command=command,
            reward_sum=float(reward_sums[index].item()),
            survival_steps=int(survival_steps[index].item()),
            done_step=done_steps[index],
            terminated=terminated_flags[index],
            truncated=truncated_flags[index],
        )
        if failure_reasons[index] is not None:
            summary["failure_reason"] = failure_reasons[index]
        scenario_summaries.append(summary)
    aggregate = aggregate_scenarios(scenario_summaries)
    estimator = None if velocity is None else velocity.result()
    baseline_comparison = None
    if policy_kind == "dreamwaq":
        if args_cli.baseline_report is None:
            raise ValueError("DreamWaQ Isaac evaluation requires --baseline-report")
        baseline_report = _load_baseline_report(args_cli.baseline_report.resolve(), evaluation_contract)
        baseline_comparison = {
            "baseline_report": str(args_cli.baseline_report.resolve()),
            "baseline_report_hash": baseline_report["report_hash"],
            "baseline_performance_key": baseline_report["aggregate"]["performance_key"],
            "candidate_performance_key": aggregate["performance_key"],
            "candidate_not_worse": candidate_not_worse_than_baseline(aggregate, baseline_report["aggregate"]),
        }
    elif args_cli.baseline_report is not None:
        raise ValueError("Phase1R baseline generation cannot consume --baseline-report")

    report = {
        "evaluation_schema_version": ISAAC_EVALUATION_SCHEMA_VERSION,
        "evaluation_contract": evaluation_contract,
        "evaluation_contract_hash": evaluation_contract["contract_hash"],
        "policy_kind": policy_kind,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_run_manifest": str((checkpoint.parent / "run_manifest.json").resolve()),
        "source_run_manifest_sha256": sha256_file(checkpoint.parent / "run_manifest.json"),
        "source_seed": int(checkpoint_metadata["seed"]),
        "source_completed_iterations": int(checkpoint_metadata["completed_iterations"]),
        "base_task_contract_version": base_task_contract["manifest_version"],
        "base_task_contract_hash": base_task_contract["contract_hash"],
        "reset_cache": cache_identity,
        "aggregate": aggregate,
        "estimator": estimator,
        "baseline_comparison": baseline_comparison,
    }
    if policy_kind == "phase1r_baseline":
        report["baseline_identity"] = PHASE1R_ISAAC_BASELINE
    report["report_hash"] = stable_contract_hash(report)
    report_path = output / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
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
