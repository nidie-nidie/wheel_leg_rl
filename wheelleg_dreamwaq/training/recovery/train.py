from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

import numpy as np
import tensordict
import torch

from .contracts import (
    GuidanceConfigV1, GuidanceTrainingContractV1, file_sha256, load_resolved_config,
    source_sha256, stable_hash, verify_walker_export, write_json,
)
from .pose_bank import PoseBank, source_pose_records
from .randomization import RecoveryRandomization, RecoveryRandomizationV1


def preflight(config: dict) -> dict:
    if config["schema_version"] == "RecoveryIndependentStandV1":
        from .independent_preflight import independent_preflight

        return independent_preflight(config)
    environment = config["environment"]
    for key in ("pose_bank_path", "walker_export_dir", "run_dir"):
        if not environment[key]:
            raise ValueError(f"起身运行必须显式填写 environment.{key}")
    bank = PoseBank.load(environment["pose_bank_path"])
    bank.validate_coverage(formal=environment["task_type"] == "formal_training")
    walker_dir = Path(environment["walker_export_dir"])
    manifest_path = walker_dir / "policy_manifest.json"
    manifest = verify_walker_export(walker_dir)
    actor_path = walker_dir / "actor.ts"
    if manifest["actor_sha256"] != file_sha256(actor_path):
        raise ValueError("冻结行走 Actor 文件与源导出清单身份不一致")
    randomization = RecoveryRandomization(environment["num_envs"], environment["seed"], environment["device"],
                                           RecoveryRandomizationV1(**config["randomization"]), config["contract"]["q_nominal"])
    for index in range(environment["num_envs"]):
        if not any(record.split == "train" and record.parameter_plan_sha256 == randomization.plan(index).sha256 for record in bank.records):
            raise ValueError(f"姿态库没有环境 {index} 的实际物理参数计划")
    contact_hashes = sorted({record.contact_model_sha256 for record in bank.records})
    identity = {
        "training_config_sha256": stable_hash(config),
        "walker_actor_sha256": file_sha256(actor_path),
        "walker_manifest_sha256": file_sha256(manifest_path),
        "asset_bundle_sha256": manifest["asset_bundle_hash"],
        "mujoco_contact_model_sha256": contact_hashes[0] if len(contact_hashes) == 1 else stable_hash(contact_hashes),
        "isaac_contact_model_sha256": environment["isaac_contact_model_sha256"],
        "pose_bank_sha256": bank.sha256,
        "source_pose_records_sha256": stable_hash(source_pose_records(bank)),
        "parameter_plan_sha256": randomization.manifest()["parameter_plan_sha256"],
        "num_envs": environment["num_envs"], "device": environment["device"],
        "source_sha256": source_sha256(),
        "geometry_guard_sha256": stable_hash(config["readiness"]["geometry_guard"]),
        "curriculum_sha256": (None if config["pose_sampling"]["curriculum_path"] is None else file_sha256(config["pose_sampling"]["curriculum_path"])),
    }
    identity["contact_model_sha256"] = stable_hash({"isaac": identity["isaac_contact_model_sha256"],
                                                  "mujoco": identity["mujoco_contact_model_sha256"]})
    if config["schema_version"] == "RecoveryResolvedConfigV3":
        identity["guidance_config_sha256"] = GuidanceConfigV1.from_dict(config["guidance"]).sha256
        identity["training_contract_sha256"] = GuidanceTrainingContractV1.from_dict(config["training_contract"]).sha256
    if environment["task_type"] == "formal_training":
        from .preflight import verify_formal_evidence

        verify_formal_evidence(config, bank, identity)
    return identity


def _parameter_digest(policy) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def _assert_restored_state(actual, expected, path="state") -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype:
            raise RuntimeError(f"checkpoint 恢复张量类型不同: {path}")
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"checkpoint 恢复字段不同: {path}")
        for key in expected:
            _assert_restored_state(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise RuntimeError(f"checkpoint 恢复序列不同: {path}")
        for index, value in enumerate(expected):
            _assert_restored_state(actual[index], value, f"{path}[{index}]")
    elif type(actual) is not type(expected) or actual != expected:
        raise RuntimeError(f"checkpoint 恢复标量不同: {path}")


def _guidance_progress(runner) -> dict:
    guidance = runner.env.unwrapped._guidance
    expected_steps = runner.completed_updates * runner.num_steps_per_env
    if guidance is None or guidance.pending or guidance.global_control_steps != expected_steps:
        raise RuntimeError("辅助进度必须在完整更新边界保存和验收")
    expected_alpha = (max(1.0 - expected_steps / guidance.config.end_control_steps, 0.0)
                      if guidance.config.force_enabled else 0.0)
    torch.testing.assert_close(guidance.alpha, torch.full_like(guidance.alpha, expected_alpha), rtol=0, atol=1e-7)
    return {"global_control_steps": expected_steps, "alpha": float(guidance.alpha[0]),
            "expected_alpha": expected_alpha, "pending_transition": False}


def _optimizer_steps(optimizer, *, allow_empty: bool = False) -> list[int]:
    steps = []
    for state in optimizer.state.values():
        step = state["step"]
        value = float(step.detach().cpu()) if isinstance(step, torch.Tensor) else float(step)
        if not value.is_integer() or value <= 0:
            raise RuntimeError("恢复验收要求成本Adam具有真实完整更新计数")
        steps.append(int(value))
    if not steps and not allow_empty:
        raise RuntimeError("恢复验收缺少成本Adam已更新状态")
    return steps


def verify_guidance_data_flow(wrapper) -> dict:
    env = wrapper.unwrapped
    guidance = env._guidance
    if guidance is None or guidance.pending:
        raise RuntimeError("辅助只读接口验证要求已初始化且没有待提交转移")
    before = {"control_steps": guidance.global_control_steps, "physics_steps": env._sim_step_counter,
              "history": env._history.frames(), "walker_history": env._walker_history.frames(),
              "frame_ids": env._frame_ids.clone()}
    observation = wrapper.get_observations()
    repeated = wrapper.get_observations()
    if observation["policy"].shape != (env.num_envs, 111) or observation["critic"].shape != (env.num_envs, 35):
        raise RuntimeError("辅助接口观测形状与训练契约不同")
    for key in observation.keys():
        torch.testing.assert_close(observation[key], repeated[key], rtol=0, atol=0)
        if not torch.isfinite(observation[key]).all():
            raise FloatingPointError("辅助接口观测包含非有限数值")
    torch.testing.assert_close(observation["policy"][:, -1], guidance.alpha, rtol=0, atol=0)
    torch.testing.assert_close(observation["critic"][:, -1], guidance.alpha, rtol=0, atol=0)
    torch.testing.assert_close(observation["policy"][:, 104:110], env._action, rtol=0, atol=0)
    torch.testing.assert_close(observation["critic"][:, 16:22], env._action, rtol=0, atol=0)
    transition = guidance.last_transition
    if transition is not None:
        if transition["schema_version"] != "RecoveryGuidanceTransitionV1":
            raise RuntimeError("辅助只读接口包含未知转移版本")
        for key in ("cost_force_n", "peak_force_n", "alpha_t", "alpha_next", "max_force_n"):
            value = transition[key]
            if not isinstance(value, torch.Tensor) or value.shape != (env.num_envs,) or not torch.isfinite(value).all():
                raise RuntimeError(f"辅助转移的真实字段无效: {key}")
        if (transition["cost_force_n"] < 0).any() or (transition["cost_force_n"] > transition["peak_force_n"] + 1e-5).any():
            raise RuntimeError("辅助原始成本与子步峰值不一致")
    if guidance.global_control_steps != before["control_steps"] or env._sim_step_counter != before["physics_steps"]:
        raise RuntimeError("辅助只读接口验证推进了训练或物理时间")
    for actual, expected in ((env._history.frames(), before["history"]),
                             (env._walker_history.frames(), before["walker_history"]),
                             (env._frame_ids, before["frame_ids"])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return {"schema_version": "RecoveryGuidanceReadonlyInterfaceVerificationV1",
            "observation_shapes_verified": True, "actor_critic_alpha_equal": True,
            "raw_action_slice": [104, 110], "raw_action_observation_preserved": True,
            "repeated_observation_readonly": True, "physics_steps_added": 0,
            "control_steps_added": 0, "collected_transition_fields_verified": transition is not None,
            "physical_force_application_validated": False,
            "scope": "只读接口；真实施力与采集由独立物理及训练验证提供证据"}


def verify_data_flow(wrapper) -> dict:
    env = wrapper.unwrapped
    plan_hashes = {plan.sha256 for plan in env._randomization.plans}
    expected_pose_ids = {record.record_id for record in env.pose_bank.records
                         if record.split == "train" and record.parameter_plan_sha256 in plan_hashes}
    restored_pose_ids = set(env._reset_records)
    for _ in range(1000):
        if expected_pose_ids <= restored_pose_ids:
            break
        env._reset_idx(torch.arange(env.num_envs, device=env.device))
        restored_pose_ids.update(env._reset_records)
    if not expected_pose_ids <= restored_pose_ids:
        raise RuntimeError("接口真实复位回放未覆盖当前参数计划的全部训练姿态")
    observation = wrapper.get_observations()
    clean_state = env._read_state()
    if env._read_state() is not clean_state:
        raise RuntimeError("同一物理与写入代数重复读取没有复用clean state")
    repeated = wrapper.get_observations()
    for key in observation.keys():
        torch.testing.assert_close(observation[key], repeated[key], rtol=0, atol=0)
    before_frames = env._history.frames()
    before_walker = env._walker_history.frames()
    with torch.inference_mode():
        candidate_input = env._walker_history.flattened()
        env._walker(candidate_input.clone())
        env._walker(candidate_input.clone())
    torch.testing.assert_close(before_frames, env._history.frames(), rtol=0, atol=0)
    torch.testing.assert_close(before_walker, env._walker_history.frames(), rtol=0, atol=0)
    probe = torch.tensor([6., -6., 6., -6., 1.1, -1.1], device=env.device).repeat(env.num_envs, 1)
    with torch.inference_mode():
        after, _, done, extras = wrapper.step(probe)
    if "time_outs" in extras:
        raise RuntimeError("起身 wrapper 添加了 timeout bootstrap")
    continuing = done == 0
    if not continuing.any():
        raise RuntimeError("原始动作验证导致所有真实环境终止")
    torch.testing.assert_close(after["policy"][continuing, -6:], probe[continuing], rtol=0, atol=0)
    torch.testing.assert_close(after["critic"][continuing, 16:22], probe[continuing], rtol=0, atol=0)
    expected_leg = env._q_reference[continuing] + env.contract.leg_action_scale * probe[continuing, :4]
    torch.testing.assert_close(env._leg_targets[continuing], expected_leg, rtol=0, atol=0)
    torch.testing.assert_close(env.robot.data.joint_pos_target[continuing][:, env._leg_ids], expected_leg, rtol=0, atol=0)
    if not (expected_leg.abs() > 1.0).any():
        raise RuntimeError("真实接口验证未覆盖绝对值超过1rad的腿部目标")
    torch.testing.assert_close(env._history.frames()[continuing, :-1], before_frames[continuing, 1:], rtol=0, atol=0)
    torch.testing.assert_close(env._walker_history.frames()[:, -1], env._walker_frame(env._clean_frame), rtol=0, atol=0)
    actor_noise_active = not torch.equal(env._history.frames()[:, -1], env._clean_frame)
    if env._randomization.profile.enabled and not actor_noise_active:
        raise RuntimeError("启用随机化后没有实测到 Actor 观测噪声")
    if env.num_envs > 1:
        positions = env.robot.data.root_link_pos_w[1:].clone()
        velocities = env.robot.data.joint_vel[1:].clone()
        histories = env._history.frames()[1:].clone()
        frames = env._frame_ids[1:].clone()
        sim_steps = env._sim_step_counter
        previous_cache = env._read_state()
        env._reset_idx(torch.tensor([0], device=env.device))
        if env._read_state() is previous_cache:
            raise RuntimeError("局部reset未使clean state缓存失效")
        torch.testing.assert_close(env.robot.data.root_link_pos_w[1:], positions, rtol=0, atol=0)
        torch.testing.assert_close(env.robot.data.joint_vel[1:], velocities, rtol=0, atol=0)
        torch.testing.assert_close(env._history.frames()[1:], histories, rtol=0, atol=0)
        torch.testing.assert_close(env._frame_ids[1:], frames, rtol=0, atol=0)
        if env._sim_step_counter != sim_steps:
            raise RuntimeError("局部 reset 推进了世界物理时间")
    return {"repeated_observation_readonly": True, "candidate_readonly": True,
            "raw_action_above_unit_preserved": True, "unbounded_leg_target_written_to_sim": True,
            "history_order_verified": True,
            "partial_reset_does_not_advance_physics": env.num_envs > 1,
            "clean_state_cache_readonly_and_reset_invalidates": True,
            "finite_horizon_no_timeout_bootstrap": True,
            "walker_history_uses_clean_observation": True, "actor_noise_active": actor_noise_active,
            "restored_training_pose_ids": sorted(restored_pose_ids),
            "restored_training_categories": sorted({record.initial_category for record in env.pose_bank.records
                                                     if record.record_id in restored_pose_ids}),
            "physical_mechanical_limits_calibrated": False,
            "readiness_calibrated": env.resolved_config["readiness"]["calibration_path"] is not None}


def main() -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="独立倒地起身 PPO 训练")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--import-compatible", type=Path)
    parser.add_argument("--import-evidence", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-interface", action="store_true")
    parser.add_argument("--verify-resume", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    arguments = parser.parse_args()
    if arguments.resume is not None and arguments.import_compatible is not None:
        raise ValueError("严格恢复与兼容导入不能同时使用")
    if (arguments.import_compatible is None) != (arguments.import_evidence is None):
        raise ValueError("兼容导入必须同时指定父checkpoint和完整等价证据")
    config = load_resolved_config(arguments.config)
    if config["schema_version"] == "RecoveryIndependentStandV1":
        from .independent_train import run_independent_training

        run_independent_training(arguments, config)
        return
    if config["schema_version"] == "RecoveryResolvedConfigV3" and arguments.import_compatible is not None:
        raise ValueError("辅助训练禁止兼容导入，必须使用新网络或同身份严格恢复")
    if arguments.device != config["environment"]["device"]:
        raise ValueError("AppLauncher device 与完整解析配置不同")
    if any(int(os.environ.get(name, str(expected))) != expected for name, expected in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0))):
        raise RuntimeError("起身训练只接受单个 GPU 进程")
    identity = preflight(config)
    if arguments.preflight_only:
        print(json.dumps({"identity": identity, "preflight": "passed"}, ensure_ascii=False), flush=True)
        return
    environment = config["environment"]
    updates = config["assessment"]["updates"]
    if not isinstance(updates, int) or updates <= 0:
        raise ValueError("运行更新次数必须写入完整解析配置")
    if environment["task_type"] == "interface_validation" and updates > 2:
        raise ValueError("接口验证最多进行两次真实 PPO 更新")
    if arguments.verify_resume and environment["task_type"] != "interface_validation":
        raise ValueError("额外一步断点验证只允许接口验证运行")
    run_dir = Path(environment["run_dir"])
    if arguments.resume is None:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif not run_dir.is_dir():
        raise FileNotFoundError("resume 必须使用原运行目录")
    seed = environment["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if arguments.resume is None:
        write_json(run_dir / "resolved_config.json", config)
    app = AppLauncher(arguments).app
    try:
        with ExitStack() as cleanup:
            _run_training(arguments, config, identity, app, cleanup)
    except BaseException:
        # Kit 快速关闭可能直接以零状态退出，必须先保存真实异常。
        failure = traceback.format_exc()
        sys.__stderr__.write(failure)
        sys.__stderr__.flush()
        with (run_dir / "training_failure.json").open("w", encoding="utf-8") as stream:
            json.dump({"passed": False, "traceback": failure}, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os._exit(1)
    app.close(skip_cleanup=True)


def _run_training(arguments, config, identity, app, cleanup):
    environment = config["environment"]
    guided = config["schema_version"] == "RecoveryResolvedConfigV3"
    run_dir = Path(environment["run_dir"])
    updates = config["assessment"]["updates"]
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from .checkpoint import load_checkpoint, save_checkpoint, prepare_compatible_import, restore_global_rng_state
    from .env import RecoveryEnv
    from .env_cfg import build_env_cfg
    from .ppo_runner import RecoveryPPORunner

    direct_env = RecoveryEnv(build_env_cfg(config))
    cleanup.callback(direct_env.close)
    if arguments.resume is None:
        write_json(run_dir / "joint_limits_readback.json", {
            "joint_names": direct_env.robot.joint_names,
            "joint_position_limits_rad": direct_env.robot.data.joint_pos_limits[0].tolist(),
            "source": "Isaac_PhysX_actual_joint_position_limits", "physical_mechanical_limits_calibrated": False,
        })
    actual_isaac_contact = stable_hash({"asset_bundle_sha256": direct_env.asset_report.bundle_hash,
                                       "collision_manifest": direct_env.contact_manifest,
                                       "ground_size_xy_m": environment["ground_size_xy_m"],
                                       "ground_top_z_m": 0.0,
                                       "physics_dt_s": config["contract"]["isaac_physics_dt_s"],
                                       "decimation": config["contract"]["isaac_decimation"]})
    expected_contact = environment["isaac_contact_model_sha256"]
    if expected_contact is not None and expected_contact != actual_isaac_contact:
        raise ValueError("Isaac 完整接触模型实际读回身份与配置不同")
    identity["isaac_contact_model_sha256"] = actual_isaac_contact
    identity["contact_model_sha256"] = stable_hash({"isaac": actual_isaac_contact,
                                                  "mujoco": identity["mujoco_contact_model_sha256"]})
    execution = {"implementation": "RecoveryGpuExecutionV1", "geometry": None if direct_env._geometry_guard is None
                 else direct_env._geometry_guard.identity, "logging_protocol": direct_env._metrics_stream.protocol}
    if guided:
        if direct_env.guidance_execution is None:
            raise RuntimeError("辅助训练没有建立实际质量与施力接口身份")
        execution["guidance"] = direct_env.guidance_execution
    identity["execution_implementation_sha256"] = stable_hash(execution)
    env = RslRlVecEnvWrapper(direct_env, clip_actions=None)
    runner = RecoveryPPORunner(env, config)
    cleanup.callback(runner.close)
    if arguments.resume is None:
        write_json(run_dir / "initial_body_pose.json", {
            "record_id": direct_env._reset_records[0], "body_names": direct_env.robot.body_names,
            "positions": (direct_env.robot.data.body_link_pos_w[:1] - direct_env.scene.env_origins[:1, None, :]).tolist(),
            "quaternions_wxyz": direct_env.robot.data.body_link_quat_w[:1].tolist(),
        })
    manifest = {"schema_version": "RecoveryTrainingRunV1", "created_at": datetime.now().astimezone().isoformat(),
                "identity": identity, "resolved_config": config, "randomization": direct_env._randomization.manifest(),
                "isaac_contact_manifest": direct_env.contact_manifest, "material_readback": direct_env.material_readback.tolist(),
                "ground_coverage": direct_env.ground_coverage,
                "physical_mechanical_limits_calibrated": False,
                "geometry_guard_identity": None if direct_env._geometry_guard is None else direct_env._geometry_guard.identity,
                "execution": execution, "logging_memory_budget": direct_env._metrics_stream.budget,
                "purpose": environment["task_type"], "versions": {"torch": torch.__version__,
                "rsl_rl": importlib.metadata.version("rsl-rl-lib"), "isaac_sim": importlib.metadata.version("isaacsim")},
                "fresh": arguments.resume is None, "resume_checkpoint": None if arguments.resume is None else str(arguments.resume)}
    if arguments.resume is not None:
        payload = load_checkpoint(arguments.resume, identity, restore_rng=False)
        runner.load_state_dict(payload)
        direct_env.restore_environment_state(payload["environment_state"])
        runner.refresh_observation_after_resume()
        load_checkpoint(arguments.resume, identity, restore_rng=True)
    if arguments.import_compatible is not None:
        if guided:
            raise ValueError("辅助训练禁止兼容导入")
        parent_manifest = json.loads((arguments.import_compatible.parent / "run_manifest.json").read_text(encoding="utf-8"))
        envelope = prepare_compatible_import(arguments.import_compatible, parent_manifest["identity"], identity,
            arguments.import_evidence, run_dir / "compatible_import_manifest.json", new_config=config)
        payload = envelope["payload"]
        runner.load_state_dict(payload)
        direct_env.restore_environment_state(payload["environment_state"])
        runner.refresh_observation_after_resume()
        restore_global_rng_state(envelope["global_rng_state"])
        manifest["compatible_parent_checkpoint"] = str(arguments.import_compatible.resolve())
        manifest["compatible_parent_checkpoint_sha256"] = file_sha256(arguments.import_compatible)
    if guided:
        manifest["guidance"] = {"schema_version": "RecoveryGuidanceTrainingRunV1",
            "training_contract": config["training_contract"], "guidance_config": config["guidance"],
            "training_contract_sha256": identity["training_contract_sha256"],
            "guidance_config_sha256": identity["guidance_config_sha256"],
            "execution": direct_env.guidance_execution, "initial_progress": _guidance_progress(runner)}
    write_json(run_dir / ("run_manifest.json" if arguments.resume is None else "resume_manifest.json"), manifest)
    validation = ((verify_guidance_data_flow(env) if guided else verify_data_flow(env))
                  if arguments.verify_interface else None)
    if arguments.verify_interface:
        runner.refresh_observation_after_resume()
    initial_digest = _parameter_digest(runner.policy)
    initial_cost_digest = _parameter_digest(runner.algorithm.cost_critic) if guided else None
    saved = []

    def writer(active_runner):
        path = run_dir / f"model_{active_runner.completed_updates}.pt"
        index = save_checkpoint(path, active_runner.state_dict(), identity)
        saved.append({"path": str(path), **index})

    remaining_updates = updates - runner.completed_updates
    if remaining_updates <= 0:
        print(f"RECOVERY_TRAINING_TARGET_ALREADY_REACHED completed={runner.completed_updates} target={updates}", flush=True)
        return
    learning_started = perf_counter()
    if arguments.profile:
        profile_dir = run_dir / "torch_profile"
        profile_dir.mkdir(exist_ok=False)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir))) as profiler:
            runner.learn(remaining_updates, checkpoint_writer=writer, profiler=profiler)
    else:
        runner.learn(remaining_updates, checkpoint_writer=writer)
    updated_digest = _parameter_digest(runner.policy)
    updated_cost_digest = _parameter_digest(runner.algorithm.cost_critic) if guided else None
    observed_force_cost = (any(record["guidance"]["mean_force_cost_n"] > 0 for record in runner.update_records)
                           if guided else False)
    measured_update_count = len(runner.update_records)
    if initial_digest == updated_digest:
        raise RuntimeError("真实 PPO 更新没有改变网络参数")
    if guided and observed_force_cost and initial_cost_digest == updated_cost_digest:
        raise RuntimeError("真实辅助成本已进入采集，但成本critic参数没有更新")
    resume_verified = False
    guidance_resume_verification = None
    if arguments.verify_resume:
        selected = Path(saved[-1]["path"])
        payload = load_checkpoint(selected, identity, restore_rng=False)
        runner.close()
        reloaded = RecoveryPPORunner(env, config)
        cleanup.callback(reloaded.close)
        reloaded.load_state_dict(payload)
        if _parameter_digest(reloaded.policy) != updated_digest:
            raise RuntimeError("checkpoint 重载后的网络参数不同")
        _assert_restored_state(reloaded.algorithm.optimizer.state_dict(), payload["optimizer_state_dict"], "ppo_optimizer")
        if guided:
            if _parameter_digest(reloaded.algorithm.cost_critic) != updated_cost_digest:
                raise RuntimeError("checkpoint 重载后的成本critic参数不同")
            _assert_restored_state(reloaded.algorithm.cost_optimizer.state_dict(),
                                   payload["guidance"]["cost_optimizer_state_dict"], "cost_optimizer")
            restored_guidance_progress = _guidance_progress(reloaded)
            restored_cost_optimizer_steps = _optimizer_steps(reloaded.algorithm.cost_optimizer,
                allow_empty=not config["guidance"]["force_enabled"])
        direct_env.restore_environment_state(payload["environment_state"])
        reloaded.refresh_observation_after_resume()
        load_checkpoint(selected, identity, restore_rng=True)
        reloaded.learn(1, checkpoint_writer=writer)
        measured_update_count += len(reloaded.update_records)
        if _parameter_digest(reloaded.policy) == updated_digest:
            raise RuntimeError("checkpoint 恢复后没有继续优化")
        if guided:
            resumed_cost_digest = _parameter_digest(reloaded.algorithm.cost_critic)
            resumed_force_cost = any(record["guidance"]["mean_force_cost_n"] > 0 for record in reloaded.update_records)
            if resumed_force_cost and resumed_cost_digest == updated_cost_digest:
                raise RuntimeError("checkpoint 恢复后成本critic没有继续优化")
            continued_cost_optimizer_steps = _optimizer_steps(reloaded.algorithm.cost_optimizer,
                allow_empty=not config["guidance"]["force_enabled"])
            cost_optimizer_advanced = (bool(continued_cost_optimizer_steps)
                and len(continued_cost_optimizer_steps) == len(restored_cost_optimizer_steps)
                and all(after > before for after, before in zip(continued_cost_optimizer_steps, restored_cost_optimizer_steps)))
            if resumed_force_cost and not cost_optimizer_advanced:
                raise RuntimeError("checkpoint 恢复后成本Adam没有继续更新")
            observed_force_cost = observed_force_cost or resumed_force_cost
            guidance_resume_verification = {
                "cost_critic_restored_exactly": True, "cost_optimizer_restored_exactly": True,
                "ppo_optimizer_restored_exactly": True, "restored_progress": restored_guidance_progress,
                "continued_progress": _guidance_progress(reloaded),
                "continued_cost_parameters_changed": resumed_cost_digest != updated_cost_digest,
                "continued_nonzero_force_cost_observed": resumed_force_cost,
                "restored_cost_optimizer_steps": restored_cost_optimizer_steps,
                "continued_cost_optimizer_steps": continued_cost_optimizer_steps,
                "continued_cost_optimizer_advanced": cost_optimizer_advanced,
            }
        runner = reloaded
        resume_verified = True
    drain_started = perf_counter()
    final_metrics = direct_env.finalize_metrics()
    final_drain_seconds = perf_counter() - drain_started
    learning_and_drain_seconds = perf_counter() - learning_started
    write_json(run_dir / "metrics.json", {"environments": final_metrics["metrics"],
        "metrics_archive": final_metrics["metrics_archive"], "logging_barrier": final_metrics["logging_barrier"]})
    report = {"schema_version": "RecoveryTrainingVerificationV1", "purpose": environment["task_type"],
              "passed": True, "identity": identity,
              "completed_updates": runner.completed_updates, "initial_parameter_sha256": initial_digest,
              "updated_parameter_sha256": _parameter_digest(runner.policy), "parameters_changed": True,
              "resume_and_continued_optimization_verified": resume_verified, "interface_checks": validation,
              "saved_checkpoints": saved, "recovery_ability_validated": False,
              "pose_reset_readback": direct_env.reset_validation_reports,
              "pose_reset_readback_statistics": direct_env.reset_validation_statistics,
              "geometry_statistics": direct_env.geometry_statistics,
              "performance": {"learning_and_drain_seconds": learning_and_drain_seconds,
                  "final_drain_seconds": final_drain_seconds,
                  "measured_update_count": measured_update_count,
                  "environment_steps": measured_update_count * direct_env.num_envs * runner.num_steps_per_env,
                  "environment_steps_per_full_second": measured_update_count * direct_env.num_envs * runner.num_steps_per_env / learning_and_drain_seconds,
                  "logging_statistics": dict(direct_env._metrics_stream.statistics)},
              "physical_mechanical_limits_calibrated": False,
              "end_to_end_handoff_validated": False}
    if guided:
        final_cost_digest = _parameter_digest(runner.algorithm.cost_critic)
        if not all(torch.isfinite(parameter).all() for parameter in runner.algorithm.cost_critic.parameters()):
            raise FloatingPointError("辅助训练结束时成本critic参数包含非有限数值")
        report["guidance"] = {
            "schema_version": "RecoveryGuidanceTrainingVerificationV1",
            "training_contract": config["training_contract"], "guidance_config": config["guidance"],
            "training_contract_sha256": identity["training_contract_sha256"],
            "guidance_config_sha256": identity["guidance_config_sha256"],
            "execution": direct_env.guidance_execution, "final_progress": _guidance_progress(runner),
            "initial_cost_parameter_sha256": initial_cost_digest,
            "updated_cost_parameter_sha256": final_cost_digest,
            "cost_parameters_changed": final_cost_digest != initial_cost_digest,
            "nonzero_force_cost_observed": observed_force_cost,
            "resume_verification": guidance_resume_verification,
            "cost_parameters_finite": True,
        }
    write_json(run_dir / "training_verification.json", report)
    print("RECOVERY_TRAINING_COMPLETED " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
