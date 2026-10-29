from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    launcher = AppLauncher(arguments)
    try:
        run(arguments)
    except BaseException:
        failure = traceback.format_exc()
        (arguments.output / "failure.txt").write_text(failure, encoding="utf-8")
        sys.__stderr__.write(failure)
        sys.__stderr__.flush()
        os._exit(1)
    launcher.app.close(skip_cleanup=True)


def run(arguments):
    import torch
    from torch import nn

    from recovery.contracts import ARTIFACT_ROOT, file_sha256, load_resolved_config, source_sha256, stable_hash, write_json
    from recovery.env import RecoveryEnv
    from recovery.env_cfg import build_env_cfg
    from recovery.metrics_archive import MetricsArchive

    config = deepcopy(load_resolved_config(arguments.config))
    if config["environment"]["num_envs"] != 4 or config["schema_version"] != "RecoveryResolvedConfigV3":
        raise ValueError("真实短轨迹检查要求四环境辅助配置")
    config["environment"]["run_dir"] = str(arguments.output / "run")
    env = RecoveryEnv(build_env_cfg(config))
    previous_path = ARTIFACT_ROOT / "unclipped-actions-20261024/training4096x2000/model_2000.pt"
    previous_index = json.loads(Path(str(previous_path) + ".index.json").read_text(encoding="utf-8"))
    previous_sha256 = file_sha256(previous_path)
    if previous_index["checkpoint_sha256"] != previous_sha256:
        raise ValueError("既有2000更新模型文件与索引哈希不同")
    envelope = torch.load(previous_path, map_location=env.device, weights_only=False)
    if set(envelope) != {"schema_version", "identity", "payload", "global_rng_state"}:
        raise ValueError("既有2000更新模型 envelope 字段不同")
    payload = envelope["payload"]
    if payload["completed_updates"] != 2000 or "model_state_dict" not in payload:
        raise ValueError("既有动作对照必须使用实际2000更新模型")
    previous = payload["model_state_dict"]
    actor = nn.Sequential(nn.Linear(110, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                          nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 6)).to(env.device)
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in previous.items() if key.startswith("actor.")}, strict=True)
    if not all(torch.isfinite(parameter).all() for parameter in actor.parameters()):
        raise FloatingPointError("既有动作对照模型包含非有限参数")
    actor.eval()
    modes = ("zero_action", "previous_2000_actor", "positive_leg_action_2", "negative_leg_action_2")
    gamma = config["guidance"]["gamma"]
    coefficient = config["guidance"]["beta_force"] / (1 - gamma)
    rows = []
    initial_categories = {record.record_id: record.initial_category for record in env.pose_bank.records}
    randomization_state = deepcopy(env._randomization.state_dict())
    curriculum_level = env._curriculum_level
    curriculum_outcomes = deepcopy(env._curriculum_outcomes)
    initial_physical = selected_initial = None
    mode_records = []
    verification_control_steps = 0
    physical_steps_before = env._sim_step_counter
    readiness_calibrated = all(config["readiness"][key] is not None
        for key in ("support_force_n", "target_difference_limits", "torque_difference_limits"))
    with torch.inference_mode():
        env.reset()
        env.step(torch.zeros(4, 6, device=env.device))
        verification_control_steps += 1
        frozen_transition = deepcopy(env._guidance.last_transition)
        before_partial_reset = {
            "root": env.robot.data.root_link_state_w[1:].clone(),
            "joint_position": env.robot.data.joint_pos[1:].clone(),
            "joint_velocity": env.robot.data.joint_vel[1:].clone(),
            "history": env._history.frames()[1:].clone(),
            "walker_history": env._walker_history.frames()[1:].clone(),
            "frame_ids": env._frame_ids[1:].clone(),
            "alpha": env._guidance.alpha.clone(),
            "control_steps": env._guidance.global_control_steps,
            "physics_steps": env._sim_step_counter,
        }
        env._reset_idx(torch.tensor([0], device=env.device))
        for key, actual in (
            ("root", env.robot.data.root_link_state_w[1:]),
            ("joint_position", env.robot.data.joint_pos[1:]),
            ("joint_velocity", env.robot.data.joint_vel[1:]),
            ("history", env._history.frames()[1:]),
            ("walker_history", env._walker_history.frames()[1:]),
            ("frame_ids", env._frame_ids[1:]),
            ("alpha", env._guidance.alpha),
        ):
            torch.testing.assert_close(actual, before_partial_reset[key], rtol=0, atol=0)
        if (env._guidance.global_control_steps != before_partial_reset["control_steps"]
                or env._sim_step_counter != before_partial_reset["physics_steps"]
                or env._guidance.pending):
            raise RuntimeError("真实局部reset改变了辅助或物理时间")
        if set(env._guidance.last_transition) != set(frozen_transition):
            raise RuntimeError("真实局部reset改变了已冻结辅助转移字段")
        for key, value in frozen_transition.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(env._guidance.last_transition[key], value, rtol=0, atol=0)
            elif env._guidance.last_transition[key] != value:
                raise RuntimeError("真实局部reset改变了已冻结辅助转移")
        partial_reset_checks = {
            "passed": True, "reset_env_ids": [0], "untouched_env_ids": [1, 2, 3],
            "root_joint_states_and_histories_unchanged": True,
            "alpha_and_progress_unchanged": True, "frozen_transition_unchanged": True,
            "control_steps_before_and_after": before_partial_reset["control_steps"],
            "frozen_force_cost_n": frozen_transition["cost_force_n"].tolist(),
            "frozen_alpha_t": frozen_transition["alpha_t"].tolist(),
            "frozen_alpha_next": frozen_transition["alpha_next"].tolist(),
            "formal_training_steps_added": 0,
        }
        for mode in modes:
            if env._guidance.pending:
                raise RuntimeError("短轨迹分组不能在物理子步尚未完成时重置")
            env._randomization.load_state_dict(randomization_state)
            env._curriculum_level = curriculum_level
            env._curriculum_outcomes = deepcopy(curriculum_outcomes)
            env._guidance.restore_progress(0)
            observation, _ = env.reset()
            current_initial = list(env._reset_records)
            physical = {"root_link_state_w": env.robot.data.root_link_state_w.clone(),
                        "joint_position": env.robot.data.joint_pos.clone(),
                        "joint_velocity": env.robot.data.joint_vel.clone(),
                        "base_com_height_m": env._read_state()["height"].clone(),
                        "actor_observation": observation["policy"].clone()}
            if initial_physical is None:
                initial_physical = physical
                selected_initial = current_initial
            else:
                if current_initial != selected_initial:
                    raise RuntimeError("恢复相同采样随机流后，动作分组的初态record仍然不同")
                for key in physical:
                    torch.testing.assert_close(physical[key], initial_physical[key], rtol=0, atol=1e-6)
            attempt_ids = env._supervisor.attempt_id.clone().cpu().tolist()
            mode_records.append({"mode": mode, "initial_records": current_initial,
                                 "initial_attempt_ids": attempt_ids,
                                 "guidance_control_steps_at_start": env._guidance.global_control_steps,
                                 "guidance_alpha_at_start": float(env._guidance.alpha[0]),
                                 "matched_physical_initial_states": True})
            returns = torch.zeros(4, dtype=torch.float64, device=env.device)
            costs = torch.zeros_like(returns)
            lengths = torch.zeros(4, dtype=torch.long, device=env.device)
            finished = torch.zeros(4, dtype=torch.bool, device=env.device)
            discounted_terms = {}

            def record(index, *, terminated, truncated, observed):
                task_return = float(returns[index])
                cost_return = float(costs[index])
                return {"env_id": index, "mode": mode, "initial_record": current_initial[index],
                    "initial_attempt_id": attempt_ids[index],
                    "initial_category": initial_categories[current_initial[index]], "length": int(lengths[index]),
                    "initial_base_com_height_m": float(physical["base_com_height_m"][index]),
                    "discounted_task_return": task_return, "discounted_force_cost_n": cost_return,
                    "discounted_reward_terms": {key: float(value[index]) for key, value in discounted_terms.items()},
                    "force_penalty": coefficient * cost_return,
                    "penalized_return": task_return - coefficient * cost_return,
                    "termination_observed": observed, "terminated": terminated, "truncated": truncated,
                    "success": None, "terminal_kind": None, "terminal_reason": None,
                    "censored_by_verification_horizon": not observed}

            for control_step in range(400):
                actions = torch.zeros(4, 6, device=env.device)
                if mode == "previous_2000_actor":
                    actions.copy_(actor(observation["policy"][:, :110]))
                elif mode == "positive_leg_action_2":
                    actions[:, :4] = 2.0
                elif mode == "negative_leg_action_2":
                    actions[:, :4] = -2.0
                observation, reward, terminated, truncated, extras = env.step(actions)
                verification_control_steps += 1
                cost = extras["guidance"]["cost_force_n"]
                if not torch.isfinite(reward).all() or not torch.isfinite(cost).all() or (cost < 0).any():
                    raise FloatingPointError("真实轨迹奖励或辅助成本无效")
                active = ~finished
                discount = gamma ** lengths
                returns.add_(torch.where(active, discount * reward, 0))
                costs.add_(torch.where(active, discount * cost, 0))
                for key, value in env._reward_terms.items():
                    if key != "potential":
                        if key not in discounted_terms:
                            discounted_terms[key] = torch.zeros_like(returns)
                        discounted_terms[key].add_(torch.where(active, discount * value, 0))
                lengths.add_(active)
                newly_done = active & (terminated | truncated)
                for index in newly_done.nonzero().flatten().cpu().tolist():
                    rows.append(record(index, terminated=bool(terminated[index]),
                                       truncated=bool(truncated[index]), observed=True))
                finished |= newly_done
                if bool(finished.all()):
                    break
            for index in (~finished).nonzero().flatten().cpu().tolist():
                rows.append(record(index, terminated=False, truncated=False, observed=False))
            mode_records[-1]["guidance_control_steps_at_end"] = env._guidance.global_control_steps
            if env._guidance.global_control_steps != control_step + 1:
                raise RuntimeError("短轨迹验证的辅助控制步计数不连续")
    metrics = env.finalize_metrics()
    env.close()
    index = MetricsArchive.verify_index(metrics["metrics_archive"])
    selected_attempts = {(str(row["env_id"]), row["initial_attempt_id"]): row for row in rows}
    for segment in index["segments"]:
        with gzip.open(segment["path"], "rt", encoding="utf-8") as source:
            for line in source:
                archive_record = json.loads(line)
                if archive_record["kind"] != "attempt":
                    continue
                attempt = archive_record["payload"]
                key = (archive_record["stream_id"], attempt["attempt_id"])
                if key not in selected_attempts:
                    continue
                row = selected_attempts[key]
                if row["terminal_kind"] is not None:
                    raise RuntimeError("同一选定轨迹出现重复终止归档")
                row["terminal_kind"] = attempt["observation_end_kind"]
                row["terminal_reason"] = attempt["end_reason"]
                row["terminal_outcome_source"] = "verified_metrics_archive_attempt"
                row["metrics_archive_sequence"] = archive_record["sequence"]
                if row["termination_observed"]:
                    if attempt["end_step"] != row["length"]:
                        raise RuntimeError("归档终止步数与实际首回合采集不同")
                    row["observed_completed"] = attempt["observation_end_kind"] == "completed"
                    row["success"] = row["observed_completed"] if readiness_calibrated else None
    for row in rows:
        if row["terminal_kind"] is None:
            raise RuntimeError("选定首回合缺少真实日志终止或删失证据")
        if abs(sum(row["discounted_reward_terms"].values()) - row["discounted_task_return"]) > 1e-4:
            raise RuntimeError("真实轨迹任务回报与未改动的奖励分项累计不同")
    physical_failures = {"nonfinite_state", "loop_closure", "joint_position_limit", "nonallowed_body_clearance",
                         "excessive_ground_penetration", "closure_error"}
    continued = [row for row in rows if row["termination_observed"] and row["length"] >= 20]
    immediate = [row for row in rows if row["termination_observed"] and row["length"] < 20
                 and row["terminal_kind"] == "task_failure" and row["terminal_reason"] in physical_failures]
    comparisons = []
    for left in continued:
        for right in immediate:
            if left["env_id"] != right["env_id"] or left["initial_record"] != right["initial_record"]:
                continue
            margin = left["discounted_task_return"] - right["discounted_task_return"]
            penalized_margin = left["penalized_return"] - right["penalized_return"]
            comparisons.append({"env_id": left["env_id"], "initial_record": left["initial_record"],
                "continued_mode": left["mode"], "immediate_physical_failure_mode": right["mode"],
                "continued_length": left["length"], "immediate_length": right["length"],
                "immediate_failure_reason": right["terminal_reason"],
                "task_reward_margin": margin, "penalized_reward_margin": penalized_margin,
                "original_task_prefers_continued_attempt": margin > 1e-9,
                "new_failure_preference_reversal": margin > 1e-9 and penalized_margin <= 1e-9})
    eligible = [item for item in comparisons if item["original_task_prefers_continued_attempt"]]
    reversals = [item for item in eligible if item["new_failure_preference_reversal"]]
    status = "failed" if reversals else ("passed" if eligible else "inconclusive")
    all_terminated = all(row["termination_observed"] for row in rows)
    if status == "passed" and not all_terminated:
        status = "inconclusive"
    if len(rows) != 16:
        raise RuntimeError("真实短轨迹检查必须包含四种动作乘四个初态")
    force_coverage = []
    for env_id in range(4):
        matching = [row for row in rows if row["env_id"] == env_id]
        force_coverage.append({"env_id": env_id, "initial_record": matching[0]["initial_record"],
            "initial_category": matching[0]["initial_category"],
            "initial_base_com_height_m": matching[0]["initial_base_com_height_m"],
            "initial_height_at_or_above_target": matching[0]["initial_base_com_height_m"] >= config["guidance"]["h_target"],
            "modes_with_nonzero_force_cost": [row["mode"] for row in matching if row["discounted_force_cost_n"] > 0],
            "discounted_force_cost_n_by_mode": {row["mode"]: row["discounted_force_cost_n"] for row in matching}})
    report = {"schema_version": "RecoveryGuidanceTrajectoryVerificationV1", "passed": status == "passed",
        "status": status, "source_sha256": source_sha256(), "training_config_sha256": stable_hash(config),
        "source_config_file_sha256": file_sha256(arguments.config),
        "reference_checkpoint": {"path": str(previous_path), "sha256": previous_sha256,
                                 "completed_updates": payload["completed_updates"]},
        "trajectories": rows, "mode_initial_state_checks": mode_records,
        "partial_reset_checks": partial_reset_checks,
        "force_coverage_by_initial_state": force_coverage,
        "same_initial_state_comparisons": comparisons, "eligible_comparison_count": len(eligible),
        "new_failure_preference_reversals": reversals,
        "inconclusive_reason": ("没有同一初态下原任务奖励偏好持续尝试的早期物理失败对照"
            if not eligible else ("存在未观察到完整终止的轨迹" if not all_terminated else None)),
        "readiness_calibrated": readiness_calibrated,
        "success_comparison_available": readiness_calibrated and any(row["success"] is True for row in rows),
        "success_comparison_limitation": (None if readiness_calibrated else "原就绪参数未标定，成功门控阻止通过；逐轨迹success为null"),
        "scope": "同一四个真实初态各执行四种确定动作，只检查新增辅助成本是否逆转已观察到的持续尝试与早期物理失败排序；未证明全部初态或学习策略不偏好失败，未覆盖快慢成功排序",
        "validation_progress": {"formal_training_updates": 0, "formal_training_control_steps": 0,
            "verification_control_steps": verification_control_steps,
            "verification_physics_steps": env._sim_step_counter - physical_steps_before,
            "guidance_progress_reset_per_mode_for_matched_conditions": True,
            "initial_episode_trajectories": len(rows),
            "automatically_reset_subsequent_episodes_excluded_from_comparisons": True},
        "logging_barrier": metrics["logging_barrier"], "metrics_archive": metrics["metrics_archive"]}
    write_json(arguments.output / "verification.json", report)
    if reversals:
        raise RuntimeError("新增辅助成本逆转了真实持续尝试与早期失败的奖励排序")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
