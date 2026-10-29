from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

import numpy as np
import torch

from .contracts import file_sha256, source_sha256, write_json
from .independent_preflight import current_training_resources, independent_preflight, periodic_evaluation_ids


def _digest(policy):
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def run_independent_training(arguments, config):
    from isaaclab.app import AppLauncher

    if arguments.import_compatible or arguments.import_evidence or arguments.profile:
        raise ValueError("独立站立入口只接受fresh或同身份checkpoint恢复")
    if arguments.device != config["environment"]["device"]:
        raise ValueError("启动设备与新任务完整配置不同")
    if any(int(os.environ.get(key, str(value))) != value for key, value in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0))):
        raise ValueError("独立站立只接受单进程训练")
    identity = independent_preflight(config)
    if arguments.preflight_only:
        print(json.dumps({"identity": identity, "status": "passed"}, ensure_ascii=False), flush=True)
        return
    interface = config["environment"]["task_type"] == "interface_validation"
    if arguments.verify_resume and not interface:
        raise ValueError("额外保存恢复检查只允许接口验证")
    run_dir = Path(config["environment"]["run_dir"])
    if arguments.resume is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "resolved_config.json", config)
    elif not run_dir.is_dir():
        raise FileNotFoundError("恢复要求原运行目录与小index日志cutoff一致")
    seed = config["environment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    if config["environment"]["task_type"] == "formal_training":
        resources = current_training_resources()
        write_json(run_dir / "launch_resources.json", resources)
        if not resources["sufficient"]:
            raise RuntimeError("资源不足，保持其他项目运行并停止本次启动")
    app = AppLauncher(arguments).app
    try:
        _run(arguments, config, identity, started, started_utc)
    except BaseException:
        failure = traceback.format_exc()
        write_json(run_dir / "training_failure.json", {"passed": False, "traceback": failure,
                   "identity": identity, "recorded_utc": datetime.now(timezone.utc).isoformat()})
        sys.__stderr__.write(failure)
        sys.__stderr__.flush()
        os._exit(1)
    app.close(skip_cleanup=True)


def _run(arguments, config, identity, started, started_utc):
    from .independent_env import IndependentStandingEnv
    from .independent_env_cfg import build_independent_env_cfg
    from .independent_runner import IndependentRunner
    from .standing_reference import StandingReference

    run_dir = Path(config["environment"]["run_dir"])
    ref = config["reference"]
    reference = StandingReference.load(ref["manifest_path"], ref["bank_path"], ref["bank_sha256"],
                                       expected_manifest_sha256=ref["manifest_sha256"])
    env = IndependentStandingEnv(build_independent_env_cfg(config))
    runner = None
    try:
        observation, _ = env.reset()
        if set(observation) != {"policy", "critic"}:
            raise RuntimeError("物理环境未提供新契约观测")
        local_reset = _verify_local_reset(env) if arguments.verify_resume else None
        if local_reset is not None:
            env.reset()
        runner = IndependentRunner(env, config, identity, reference.q_median)
        if arguments.resume is not None:
            runner.load(arguments.resume)
        before = _digest(runner.policy)
        manifest = {"schema_version": "IndependentStandingRunManifestV1", "identity": identity,
                    "started_utc": started_utc, "run_dir": str(run_dir),
                    "statistics_scope": "this_run", "physical_restore": "explicit_root_joint_and_task_state",
                    "physics_contact_solver_cache_restored": False,
                    "guidance_enabled": False, "cost_network": False,
                    "resume": None if arguments.resume is None else str(arguments.resume)}
        if arguments.resume is None:
            write_json(run_dir / "run_manifest.json", manifest)
        target = config["ppo"]["max_iterations"]
        restore_check = None
        if arguments.verify_resume:
            if target != 4 or runner.completed_updates != 0:
                raise ValueError("保存恢复接口检查要求fresh4环境4次更新配置")
            runner.learn(2)
            path = run_dir / "model_2.pt"
            runner.save(path)
            saved_observation = {k: v.clone() for k, v in env.get_observations().items()}
            runner.close(reason="interface_segment")
            env.step(torch.zeros(env.num_envs, 6, device=env.device))
            runner = IndependentRunner(env, config, identity, reference.q_median)
            payload = runner.load(path)
            for key, value in env.get_observations().items():
                torch.testing.assert_close(value, saved_observation[key], rtol=0, atol=0)
            restored_parameters = _digest(runner.policy)
            runner.learn(2, start_update=2)
            if _digest(runner.policy) == restored_parameters:
                raise RuntimeError("恢复后的两次PPO更新没有改变策略参数")
            restore_check = {"num_envs": env.num_envs, "updates_before_save": 2,
                             "updates_after_restore": 2, "restored_update": payload["completed_updates"],
                             "checkpoint_sha256": file_sha256(path), "observation_exactly_restored": True,
                             "parameters_changed_after_restore": True}
        else:
            while runner.completed_updates < target:
                runner.learn(min(config["assessment"]["evaluation_interval"], target - runner.completed_updates))
                if config["environment"]["task_type"] == "formal_training":
                    _periodic_evaluation(runner, config)
        changed = _digest(runner.policy) != before
        if not changed or source_sha256() != identity["source_sha256"]:
            raise RuntimeError("参数未改变或运行中源码身份改变")
        export = runner.export_actor(run_dir / "export" / "actor.ts")
        physical = env.physical_failure_counts()
        barrier = runner.close()
        report = {"schema_version": "IndependentStandingTrainingVerificationV1", "passed": True,
                  "identity": identity, "completed_updates": runner.completed_updates,
                  "parameters_changed": changed, "performance": runner.performance,
                  "wall_seconds": perf_counter() - started, "physical_failure_counts": physical,
                  "save_restore": restore_check, "local_reset": local_reset,
                  "export": export, "logging_barrier": barrier,
                  "behavior_success_verified": False, "formal_training": config["environment"]["task_type"] == "formal_training"}
        write_json(run_dir / "training_verification.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        if runner is not None and not runner.closed:
            runner.close(reason="training_failure")
        env.close()


def _verify_local_reset(env):
    if env.num_envs != 4:
        raise ValueError("局部reset接口检查要求真实4环境")
    for _ in range(2):
        action = torch.zeros(env.num_envs, 6, device=env.device)
        action[:, :4] = env._read_state()["joint_position"][:, :4]
        env.step(action)
    before = env.environment_state()
    env._reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
    after = env.environment_state()
    checked = []
    for group in ("tensors", "task_tensors", "contact_window"):
        for name, value in before[group].items():
            retained = (slice(None), slice(1, None)) if group == "contact_window" and name == "buffer" else slice(1, None)
            torch.testing.assert_close(after[group][name][retained], value[retained], rtol=0, atol=0)
            checked.append(f"{group}.{name}")
    for name in ("root_state", "joint_position", "joint_velocity", "history", "episode_length_buf"):
        torch.testing.assert_close(after[name][1:], before[name][1:], rtol=0, atol=0)
        checked.append(name)
    if (after["reset_record_ids"][1:] != before["reset_record_ids"][1:]
            or after["episode_sources"][1:] != before["episode_sources"][1:]
            or int(env.task.contact_window.count[0]) != 0
            or int(env.task.elapsed_steps[0]) != 0):
        raise RuntimeError("局部reset影响其他回合或未清空本环境窗口")
    return {"passed": True, "num_envs": 4, "reset_env_ids": [0],
            "unchanged_env_ids": [1, 2, 3], "actual_physics_steps_before_reset": 2,
            "checked_fields": checked}


def _periodic_evaluation(runner, config):
    import subprocess

    update = runner.completed_updates
    directory = Path(config["environment"]["run_dir"]) / "evaluations" / f"update_{update}"
    export = directory / "actor.ts"
    runner.export_actor(export)
    from .independent_poses import IndependentPoseBank

    bank = IndependentPoseBank.load(config["environment"]["pose_bank_path"])
    selected = periodic_evaluation_ids(bank)
    write_json(directory / "selection.json", {"record_ids": selected, "pose_bank_sha256": bank.sha256})
    command = [config["geometry_guard"]["python_executable"], "-m", "recovery.independent_evaluate",
               "--config", str(Path(config["environment"]["run_dir"]) / "resolved_config.json"),
               "--actor", str(export), "--output", str(directory / "results"), "--record-ids", *selected]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"独立CPU定期测评失败: exit_code={completed.returncode}")
