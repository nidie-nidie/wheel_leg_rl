from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from recovery.contracts import PROJECT_ROOT, file_sha256, source_sha256, write_json
from recovery.independent_checkpoint import load_independent_checkpoint


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "数值记录包含NaN或Inf")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def verify(artifact):
    run = artifact / "isaac-interface-v3"
    unit = read_json(artifact / "final-unit-v2/verification.json")
    review = read_json(artifact / "independent_review_final.json")
    report = read_json(run / "training_verification.json")
    identity = report["identity"]
    current = source_sha256()
    require(unit["passed"] and unit["source_sha256"] == unit["source_after_sha256"] == current,
            "完整单元验收身份或结果不匹配")
    require(review["source_sha256"] == identity["source_sha256"] == current,
            "审阅、物理验证与当前源码身份不同")
    require(review["conclusion"]["unresolved_p1"] == review["conclusion"]["unresolved_p2"] == 0,
            "独立复核仍有P1/P2")
    for name, sha in review["reviewed_files"].items():
        require(file_sha256(PROJECT_ROOT / name) == sha, f"受审文件改变: {name}")
    require(read_json(artifact / "isaac-supervisor-v3/exit.json")["exit_code"] == 0,
            "真实四环境进程退出异常")
    require(not (run / "training_failure.json").exists(), "本次成功目录仍有故障记录")
    require(report["passed"] and report["completed_updates"] == 4 and report["parameters_changed"],
            "真实PPO完成或参数变化检查未通过")
    require(report["performance"]["measured_update_count"] == 4
            and report["performance"]["environment_steps"] == 384, "更新或采样预算不符")
    require(report["save_restore"]["observation_exactly_restored"]
            and report["save_restore"]["parameters_changed_after_restore"]
            and report["save_restore"]["updates_before_save"] == report["save_restore"]["updates_after_restore"] == 2,
            "保存恢复后两次实际更新检查不完整")
    require(report["local_reset"]["passed"], "真实多环境局部reset未通过")
    updates = [json.loads(line) for line in (run / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
    require([item["update"] for item in updates] == [1, 2, 3, 4], "更新记录不连续")
    for item in updates:
        finite(item)
        require(item["global_control_steps"] == 24 * item["update"] and item["guidance_alpha"] == 0.,
                "真实步骤或辅助力标记不符")
        require(all(-1. <= value <= 1. for key in ("action_min", "action_max") for value in item[key]),
                "执行动作超出有界契约")
    metrics = read_json(run / "metrics.json")["logging_barrier"]
    require(metrics == report["logging_barrier"] and metrics["finalized"]
            and metrics["reason"] == "training_run_end" and metrics["pending"] == 0
            and metrics["submitted"] == metrics["consumed"] == metrics["last_sequence"]
            and metrics["worker_state"] == "synchronous", "日志最终确认不完整")
    for name, expected in metrics["stream_cutoffs"].items():
        data = (run / f"{name}.jsonl").read_bytes()
        require(len(data) == expected["bytes"]
                and hashlib.sha256(data[-4096:]).hexdigest().upper() == expected["tail_sha256"],
                f"日志cutoff不一致: {name}")
    require(metrics["last_frame_hash"] == updates[-1]["frame_hash"], "最终日志帧哈希不一致")
    saved = [load_independent_checkpoint(run / f"model_{step}.pt", identity, restore_rng=False) for step in (2, 4)]
    require(any(not torch.equal(saved[0]["policy"][key], saved[1]["policy"][key]) for key in saved[0]["policy"]),
            "保存的第2和第4次更新策略完全相同")
    events = EventAccumulator(str(run / "tensorboard"), size_guidance={"scalars": 0}).Reload()
    require([event.step for event in events.Scalars("Train/mean_step_reward")] == [1, 2, 3, 4],
            "TensorBoard真实更新不连续")
    evaluation = read_json(artifact / "exported-actor-interface-v1/evaluation.json")
    require(len(evaluation["episodes"]) == 2 and evaluation["actor_sha256"] == report["export"]["actor_sha256"],
            "CPU跨模拟器接口评估未使用本次导出Actor")
    finite(evaluation)
    counts = {name: sum(suite[name] for suite in unit["suites"])
              for name in ("tests", "failures", "errors", "skipped")}
    result = {"schema_version": "IndependentStandingImplementationAcceptanceV1", "passed": True,
        "source_sha256": current, "identity": identity, "unit_statistics": counts,
        "actual_isaac_updates": 4, "actual_isaac_environment_steps": 384,
        "local_reset_passed": True, "save_restore_2_plus_2_passed": True,
        "export_maximum_action_error": report["export"]["maximum_action_error"],
        "logging_barrier": metrics, "cpu_export_interface_episodes": 2,
        "cpu_export_interface_successes": evaluation["successes"],
        "formal_training_started": False, "behavior_success_verified": False,
        "limitations": ["四次PPO更新只验证接口；两次CPU回合成功0，不代表已学会起身",
                        "四环境96个训练控制步骤未产生完整回合，未在线覆盖GPU HOLD成功或中断",
                        "HOLD状态边界经单元检查与401实际轨迹校准，正式GPU阶段仍需补充覆盖",
                        "未执行4096环境资源实测、完整正式准入、4000更新训练及700回合最终行为评估",
                        "显式恢复根/关节/任务/RNG，不恢复PhysX接触求解器缓存"],
        "evidence": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in {
            "unit": artifact / "final-unit-v2/verification.json", "independent_review": artifact / "independent_review_final.json",
            "isaac": run / "training_verification.json", "exit": artifact / "isaac-supervisor-v3/exit.json",
            "cpu_export": artifact / "exported-actor-interface-v1/evaluation.json"}.items()}}
    write_json(artifact / "implementation_acceptance.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.artifact.resolve())
    print(json.dumps({"passed": result["passed"], "source_sha256": result["source_sha256"],
                      "unit_statistics": result["unit_statistics"], "behavior_success_verified": False}))


if __name__ == "__main__":
    main()
