from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import tensordict

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher
from wheelleg_dreamwaq.training.runtime import validate_runtime
from debug.no_action_clip.rewards import (
    BASE_REWARD_PROFILE, REWARD_PROFILES, resolve_reward_weights, reward_manifest_fields,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
RUN_DIRECTORY = None
parser = argparse.ArgumentParser(description="独立 debug 六维原始动作训练")
parser.add_argument("--validate-only", action="store_true")
parser.add_argument("--output", type=Path)
parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default=BASE_REWARD_PROFILE)
parser.add_argument("--wheel-differential-weight", type=float,
                    help="新奖励 profile 的差速惩罚正系数，必须显式填写")
AppLauncher.add_app_launcher_args(parser)
arguments = parser.parse_args()
selected_reward_weights = resolve_reward_weights(arguments.reward_profile, arguments.wheel_differential_weight)
runtime_info = validate_runtime(PROJECT_ROOT, device=arguments.device)
app_launcher = AppLauncher(arguments)
simulation_app = app_launcher.app

import numpy as np
from isaaclab.utils.io import dump_yaml

from wheelleg_dreamwaq.algorithms.dreamwaq.history_wrapper import DreamWaQHistoryVecEnvWrapper
from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
from wheelleg_dreamwaq.schemas.randomization import FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1, canonical_tensor_sha256
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.agents import WheelLegFlatDreamWaQRunnerCfg
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import apply_task_profile
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import load_dreamwaq_checkpoint_artifact
from wheelleg_dreamwaq.schemas.manifest import ContractMismatchError

from debug.no_action_clip.environment import RawActionEnv
from debug.no_action_clip.experiment import (
    ACTION_CONTRACT, BASELINE_EXPORT, GEOMETRY_GUARD_CONTRACT, RawActionNormalization,
    RawActionRunner, load_experiment_checkpoint, sha256, verify_formal_files, write_json,
)
from debug.no_action_clip.export import export_checkpoint


def validate_data_flow(env):
    observation = env.get_observations()
    history = observation["policy_history"].reshape(env.num_envs, 5, 25)
    torch.testing.assert_close(history, observation["policy"][:, None, :].expand_as(history), rtol=0, atol=0)
    assert torch.count_nonzero(observation["policy"][:, 19:25]) == 0
    probe = torch.tensor([2.5, -2.5, 3.0, -3.0, 2.0, -2.0], device=env.device).repeat(env.num_envs, 1)
    with torch.inference_mode():
        next_obs, _, dones, _ = env.step(probe)
        continuing = dones == 0
        assert continuing.any(), "原始动作探测导致全部环境终止"
        torch.testing.assert_close(next_obs["policy"][continuing, 19:25], probe[continuing], rtol=0, atol=0)
        torch.testing.assert_close(next_obs["critic"][continuing, 19:25], probe[continuing], rtol=0, atol=0)
        before_history = next_obs["policy_history"].reshape(env.num_envs, 5, 25).clone()
        env.episode_length_buf[0] = env.max_episode_length
        reset_obs, _, reset_dones, reset_extras = env.step(probe)
        assert reset_dones[0] != 0
        after_history = reset_obs["policy_history"].reshape(env.num_envs, 5, 25)
        torch.testing.assert_close(after_history[0], reset_obs["policy"][0].expand(5, 25), rtol=0, atol=0)
        assert torch.count_nonzero(after_history[0, :, 19:25]) == 0
        continuing = reset_dones == 0
        assert continuing.any()
        torch.testing.assert_close(after_history[continuing, :-1], before_history[continuing, 1:], rtol=0, atol=0)
        torch.testing.assert_close(after_history[continuing, -1, 19:25], probe[continuing], rtol=0, atol=0)
        for key in ("Debug/fk_disagreement_fraction", "Debug/fk_invalid_fraction", "Debug/fk_length_error_max_m"):
            assert key in reset_extras["log"], f"缺少 FK 诊断日志：{key}"
        state = env.unwrapped._current_state()
        nonfinite_state = replace(state, virtual_leg_phi0_fk=torch.full_like(state.virtual_leg_phi0_fk, float("nan")))
        try:
            nonfinite_state.assert_finite()
        except RuntimeError as error:
            assert "virtual_leg_phi0_fk" in str(error)
        else:
            raise AssertionError("实验取消了 FK 字段的非有限数值检查")
    return {"raw_actor_and_critic_action": True, "partial_reset_and_history": True,
            "remaining_envs_after_partial_reset": int(continuing.sum()),
            "fk_diagnostic_logs_present": True, "fk_nonfinite_state_rejected": True}


def main():
    global RUN_DIRECTORY
    if any(int(os.environ.get(key, "1" if key == "WORLD_SIZE" else "0")) != expected
           for key, expected in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0))):
        raise RuntimeError("本实验仅支持单个 GPU 进程")
    protected_count = verify_formal_files()
    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    seed = 41882943
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    singapore_timezone = timezone(timedelta(hours=8), "Asia/Singapore")
    timestamp = datetime.now(singapore_timezone).strftime("%Y%m%d-%H%M%S")
    run_dir = (arguments.output or EXPERIMENT_ROOT / "runs" / timestamp).resolve()
    if not run_dir.is_relative_to((EXPERIMENT_ROOT / "runs").resolve()):
        raise ValueError("实验输出必须在本目录 runs 内")
    (run_dir / "params").mkdir(parents=True, exist_ok=False)
    RUN_DIRECTORY = run_dir
    env_cfg = WheelLegFlatEnvCfg()
    env_cfg.seed = seed
    env_cfg.scene.num_envs = 8 if arguments.validate_only else 4096
    env_cfg.sim.device = arguments.device
    env_cfg.randomization = FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    apply_task_profile(env_cfg, "stop_reverse_v1")
    env_cfg.normalization = RawActionNormalization(**asdict(env_cfg.normalization))
    env_cfg.log_dir = str(run_dir)
    agent_cfg = WheelLegFlatDreamWaQRunnerCfg()
    agent_cfg.seed = seed
    agent_cfg.device = arguments.device
    agent_cfg.algorithm.num_mini_batches = 8
    agent_cfg.clip_actions = None
    agent_cfg.max_iterations = 1 if arguments.validate_only else 1000
    if arguments.validate_only:
        agent_cfg.num_steps_per_env = 2
    agent_config = agent_cfg.to_dict()
    agent_config["policy"]["class_name"] = "RawActionActorCritic"
    baseline_manifest_path = BASELINE_EXPORT / "policy_manifest.json"
    baseline = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    if asdict(env_cfg.reward_weights) != baseline["training_reward_weights"]:
        raise RuntimeError("实验奖励配置与 run-03 不同")
    env_cfg.reward_weights = selected_reward_weights
    reward_identity = reward_manifest_fields(env_cfg.reward_weights)
    agent_config["class_name"] = reward_identity["experiment_id"]
    dump_yaml(str(run_dir / "params/env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "params/agent.yaml"), agent_config)
    direct_env = RawActionEnv(env_cfg)
    torch.save(direct_env.closed_chain_reset_cache_artifact, run_dir / "closed_chain_reset_cache.pt")
    env = DreamWaQHistoryVecEnvWrapper(direct_env, clip_actions=None)
    if env.clip_actions is not None or direct_env.cfg.normalization.schema_version != "DebugRawActionNormalizationV1":
        raise RuntimeError("实验 wrapper 或 normalization 不符")
    validation = validate_data_flow(env) if arguments.validate_only else None
    manifest = {
        **reward_identity, "action_contract": ACTION_CONTRACT,
        "geometry_guard": GEOMETRY_GUARD_CONTRACT,
        "created_at": datetime.now(singapore_timezone).isoformat(),
        "seed": seed, "fresh_initialization": True, "validation_only": arguments.validate_only,
        "max_iterations": agent_cfg.max_iterations, "num_envs": env.num_envs,
        "agent_config": copy.deepcopy(agent_config), "runtime": runtime_info,
        "asset_bundle_hash": asset_report.bundle_hash,
        "baseline_export": {
            "directory": str(BASELINE_EXPORT), "actor_sha256": sha256(BASELINE_EXPORT / "actor.ts"),
            "manifest_sha256": sha256(baseline_manifest_path),
            "source_checkpoint": baseline["source_checkpoint"],
            "source_checkpoint_sha256": baseline["source_checkpoint_sha256"],
        },
        "protected_formal_file_count": protected_count,
        "formal_files_before_sha256": sha256(EXPERIMENT_ROOT / "formal_files_before.json"),
        "environment_config_sha256": sha256(run_dir / "params/env.yaml"),
        "agent_config_sha256": sha256(run_dir / "params/agent.yaml"),
        "reset_cache_sha256": sha256(run_dir / "closed_chain_reset_cache.pt"),
        "randomization_audit": direct_env.randomization_audit,
        "debug_file_sha256": {path.name: sha256(path) for path in sorted(EXPERIMENT_ROOT.iterdir())
                               if path.suffix in (".py", ".md")},
    }
    shutil.copyfile(EXPERIMENT_ROOT / "formal_files_before.json", run_dir / "formal_files_before.json")
    write_json(run_dir / "manifest.json", manifest)
    runner = RawActionRunner(env, agent_config, run_dir)
    initial_hash = canonical_tensor_sha256(runner.alg.policy.state_dict())
    write_json(run_dir / "initialization.json", {
        "seed": seed, "fresh_initialization": True, "initial_model_sha256": initial_hash,
        "optimizer_parameter_states": len(runner.alg.optimizer.state), "starting_completed_iterations": 0,
    })
    print(f"DEBUG_TRAIN_READY {run_dir} fresh seed={seed} num_envs={env.num_envs} iterations={agent_cfg.max_iterations}", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    runner.writer.flush()
    runner.writer.close()
    final_checkpoint = run_dir / f"model_{agent_cfg.max_iterations - 1}.pt"
    payload, _ = load_experiment_checkpoint(final_checkpoint)
    assert payload["infos"]["completed_iterations"] == agent_cfg.max_iterations
    assert list(run_dir.glob("events.out.tfevents.*")), "缺少 TensorBoard 记录"
    assert canonical_tensor_sha256(payload["model_state_dict"]) != initial_hash, "训练未更新模型参数"
    if arguments.validate_only:
        # 正式 loader 必须拒绝实验身份，不能把它当成正式 checkpoint。
        try:
            load_dreamwaq_checkpoint_artifact(final_checkpoint, run_dir / "manifest.json")
        except ContractMismatchError:
            validation["formal_loader_rejects_experiment"] = True
        else:
            raise RuntimeError("正式 loader 错误接受了实验 checkpoint")
    export_report = export_checkpoint(final_checkpoint)
    summary = {
        "experiment_id": reward_identity["experiment_id"],
        "completed_iterations": payload["infos"]["completed_iterations"],
        "validation_only": arguments.validate_only, "data_flow_checks": validation,
        "initial_model_sha256": initial_hash,
        "final_model_sha256": canonical_tensor_sha256(payload["model_state_dict"]),
        "final_checkpoint": str(final_checkpoint), "export": export_report,
        "protected_formal_files_verified": verify_formal_files(),
        "estimator_monitor_state": payload["infos"]["estimator_monitor_state"],
    }
    write_json(run_dir / "training_summary.json", summary)
    write_json(run_dir / "progress.json", {"state": "completed", **summary})
    print("DEBUG_TRAIN_COMPLETED " + json.dumps(summary), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        if RUN_DIRECTORY is not None:
            write_json(RUN_DIRECTORY / "progress.json", {"state": "failed", "error": traceback.format_exc()})
        os._exit(1)
    simulation_app.close(skip_cleanup=True)
