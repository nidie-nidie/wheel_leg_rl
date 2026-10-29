from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from wheelleg_dreamwaq.schemas.heading_control import HeadingHold, HeadingHoldConfig, yaw_from_rotation

from .calibrate import CalibrationConfig, compatibility, load_calibration
from .contact_model import file_sha256, payload_sha256
from .contracts import CATEGORIES, write_json
from .export import load_export
from .metrics import AttemptContext
from .mujoco_runtime import MujocoRecoveryRuntime, supervisor_measurements
from .pose_bank import PoseBank, pose_record_physical_sha256
from .supervisor import ControlMode, RecoverySupervisor, SupervisorConfig


@dataclass(frozen=True)
class EvaluationProtocol:
    name: str
    randomization_profile: str
    maximum_case_steps: int
    minimum_cases_per_fallen_category: int
    command_segments: list[dict]
    diagnostic: bool = False

    def __post_init__(self):
        if self.maximum_case_steps < 500 or self.minimum_cases_per_fallen_category < 1:
            raise ValueError("评估预算和类别分母无效")
        if not self.diagnostic and self.minimum_cases_per_fallen_category < 100:
            raise ValueError("正式能力验收要求每个倒地类别至少100个固定测试初态")
        if not self.name or not self.randomization_profile or not self.command_segments:
            raise ValueError("评估协议需要名称、随机化配置和命令序列")
        for segment in self.command_segments:
            if set(segment) != {"command", "steps"} or len(segment["command"]) != 3 or segment["steps"] < 1:
                raise ValueError("命令阶段必须明确三维命令及动作步数")
            if not np.isfinite(segment["command"]).all():
                raise ValueError("评估命令必须为有限数值")
        if not self.diagnostic:
            commands = [segment["command"] for segment in self.command_segments]
            if not any(command[0] > 0 for command in commands) or not any(command[0] < 0 for command in commands) or not any(command[0] == 0 and command[1] == 0 for command in commands) or not any(command[1] != 0 for command in commands):
                raise ValueError("正式组合评估必须覆盖前进、后退、停止及转向")
            required_steps = 400 + max(500, 100 + sum(segment["steps"] for segment in self.command_segments))
            if self.maximum_case_steps < required_steps:
                raise ValueError("正式评估预算不足以覆盖最晚交接、零速确认和全部命令及十秒存活")


def _yaw(runtime):
    matrix = runtime.data.xmat[runtime.model_map.base_body_id].reshape(1, 3, 3).copy()
    return yaw_from_rotation(torch.from_numpy(matrix))


def create_case_supervisor(record, protocol: EvaluationProtocol, stage: str,
                            config: SupervisorConfig | None = None) -> RecoverySupervisor:
    if stage not in ("standalone", "combined"):
        raise ValueError("未知评估阶段")
    context_stage = "independent" if stage == "standalone" else "combined"
    context = AttemptContext(record.initial_category, record.motion_kind, protocol.randomization_profile, context_stage)
    return RecoverySupervisor(config=config or SupervisorConfig(), standalone=stage == "standalone", initial_recovery=True, context=context)


def select_source_command(update, protocol: EvaluationProtocol, segment_index: int, segment_step: int):
    if update.zero_command or segment_index >= len(protocol.command_segments):
        return np.array([0.0, 0.0, 0.20]), segment_index, segment_step
    segment = protocol.command_segments[segment_index]
    command = np.asarray(segment["command"], dtype=np.float64)
    segment_step += 1
    if segment_step >= segment["steps"]:
        segment_index += 1
        segment_step = 0
    return command, segment_index, segment_step


def verify_training_physics_identity(runtime: MujocoRecoveryRuntime, source: dict) -> None:
    if source["walker_actor_sha256"] != runtime.walker_identity["actor_sha256"] or source["walker_manifest_sha256"] != runtime.walker_identity["manifest_sha256"]:
        raise ValueError("组合行走策略与起身训练时冻结专家不同")
    if source["asset_bundle_sha256"] != runtime.walker_manifest["asset_bundle_hash"]:
        raise ValueError("训练资产身份与组合源资产不同")
    if source["mujoco_contact_model_sha256"] != file_sha256(runtime.contact_manifest_path):
        raise ValueError("MuJoCo接触模型不在源训练配置绑定的跨仿真模型中")


def create_combination_manifest(recovery_export: Path, walker_export: Path, contact_manifest: Path,
                                actual_parameters: dict, calibration_path: Path, protocol: EvaluationProtocol,
                                *, heading_enabled: bool = True) -> dict:
    _, recovery_manifest = load_export(recovery_export)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    config = CalibrationConfig(**calibration["config"])
    runtime = MujocoRecoveryRuntime(contact_manifest, walker_export, allowed_body_pairs=config.allowed_body_pairs,
                                    actual_parameters=actual_parameters)
    load_calibration(calibration_path, runtime, allow_unvalidated_recovery=protocol.diagnostic)
    source = recovery_manifest["identity"]
    verify_training_physics_identity(runtime, source)
    if payload_sha256(recovery_manifest["contract"]) != payload_sha256(asdict(runtime.contract)):
        raise ValueError("组合MuJoCo运行契约与起身导出不同")
    result = {"schema_version": "RecoveryCombinationV1", "source_training_identity": source,
              "training_config_sha256": recovery_manifest["training_config_sha256"],
              "training_contact_model_sha256": source["contact_model_sha256"],
              "recovery_export": {"directory": str(recovery_export.resolve()),
                                  "manifest_sha256": file_sha256(recovery_export / "manifest.json"),
                                  "actor_sha256": recovery_manifest["actor_sha256"]},
              "walker_export": runtime.walker_identity,
              "mujoco_contact_manifest": {"path": str(contact_manifest.resolve()), "sha256": file_sha256(contact_manifest)},
              "actual_parameters": actual_parameters, "parameter_plan_sha256": runtime.parameter_sha256,
              "calibration": {"path": str(calibration_path.resolve()), "sha256": file_sha256(calibration_path)},
              "supervisor": asdict(SupervisorConfig()),
              "heading": {"enabled": heading_enabled, "kp": 9.0, "kd": 0.0, "max_yaw_rate": 1.0, "control_dt_s": 0.02},
              "protocol": asdict(protocol)}
    result["combination_config_sha256"] = payload_sha256(result)
    return result


def load_combination(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "RecoveryCombinationV1":
        raise ValueError("组合清单版本无效")
    expected_hash = manifest["combination_config_sha256"]
    if payload_sha256({key: value for key, value in manifest.items() if key != "combination_config_sha256"}) != expected_hash:
        raise ValueError("组合配置身份无效")
    for field in ("mujoco_contact_manifest", "calibration"):
        if file_sha256(manifest[field]["path"]) != manifest[field]["sha256"]:
            raise ValueError(f"组合依赖身份变化：{field}")
    export = Path(manifest["recovery_export"]["directory"])
    if file_sha256(export / "manifest.json") != manifest["recovery_export"]["manifest_sha256"]:
        raise ValueError("起身导出清单变化")
    actor, recovery_manifest = load_export(export, expected_identity=manifest["source_training_identity"])
    if recovery_manifest["actor_sha256"] != manifest["recovery_export"]["actor_sha256"]:
        raise ValueError("起身Actor身份变化")
    protocol = EvaluationProtocol(**manifest["protocol"])
    calibration_data = json.loads(Path(manifest["calibration"]["path"]).read_text(encoding="utf-8"))
    config = CalibrationConfig(**calibration_data["config"])
    runtime = MujocoRecoveryRuntime(manifest["mujoco_contact_manifest"]["path"],
                                    manifest["walker_export"]["directory"], allowed_body_pairs=config.allowed_body_pairs,
                                    actual_parameters=manifest["actual_parameters"])
    if runtime.walker_identity != manifest["walker_export"] or runtime.parameter_sha256 != manifest["parameter_plan_sha256"]:
        raise ValueError("实际运行参数或行走来源与组合清单不同")
    verify_training_physics_identity(runtime, manifest["source_training_identity"])
    calibration = load_calibration(manifest["calibration"]["path"], runtime, allow_unvalidated_recovery=protocol.diagnostic)
    if payload_sha256(recovery_manifest["contract"]) != payload_sha256(asdict(runtime.contract)):
        raise ValueError("运行契约与导出源契约不同")
    SupervisorConfig(**manifest["supervisor"])
    HeadingHoldConfig(**manifest["heading"])
    return runtime, actor, calibration, config, protocol, manifest, recovery_manifest


@torch.inference_mode()
def evaluate_case(runtime, actor, calibration, calibration_config, protocol, record, stage, trace_path,
                  *, supervisor_config: SupervisorConfig | None = None, heading_config: dict | None = None):
    runtime.restore_pose(record)
    supervisor = create_case_supervisor(record, protocol, stage, supervisor_config)
    fields = heading_config or {"enabled": True, "kp": 9.0, "kd": 0.0, "max_yaw_rate": 1.0, "control_dt_s": 0.02}
    heading = HeadingHold(HeadingHoldConfig(**fields),
                          1, dtype=torch.float64)
    heading.reset(_yaw(runtime))
    first_handoff = None
    zero_complete = None
    segment_index = segment_step = 0
    recurrent_falls = 0
    succeeded = False
    reason = None
    with gzip.open(trace_path, "xt", encoding="utf-8") as stream:
        for step_index in range(protocol.maximum_case_steps):
            before = supervisor.snapshot()
            source_command = np.array([0.0, 0.0, 0.20])
            if before.mode is ControlMode.RECOVERY:
                action = actor(torch.from_numpy(runtime.recovery_observation()).view(1, 110)).squeeze(0).numpy()
                effective_command = source_command
            else:
                source_command, segment_index, segment_step = select_source_command(before, protocol, segment_index, segment_step)
                state = runtime.state()
                effective_command = heading.update(_yaw(runtime), torch.tensor([state.angular_velocity_control[2]], dtype=torch.float64),
                                                    torch.from_numpy(source_command).view(1, 3)).numpy()[0]
                action = runtime.walker_action(effective_command)
            locomotion = before.mode is ControlMode.LOCOMOTION
            result = runtime.step(action, effective_command, locomotion=locomotion)
            candidate = runtime.walker_action(np.array([0.0, 0.0, 0.20]))
            compatible, blockers, differences = compatibility(runtime, action, candidate, calibration,
                                                               actual_is_walker=locomotion)
            measurement = supervisor_measurements(runtime, result,
                support_force_threshold_n=calibration_config.support_force_n,
                maximum_contact_penetration_m=calibration_config.maximum_contact_penetration_m,
                minimum_clearance_m=calibration_config.minimum_clearance_m,
                action_compatible=compatible, compatibility_blockers=blockers)
            after = supervisor.post_step(measurement)
            if after.capture_heading:
                heading.reset(_yaw(runtime))
                if first_handoff is None:
                    first_handoff = step_index + 1
            if after.transition == "LOCOMOTION->RECOVERY":
                recurrent_falls += 1
            if after.event_finished and after.fault_reason is None and stage == "combined" and zero_complete is None:
                zero_complete = step_index + 1
            stream.write(json.dumps({"step": step_index + 1, "mode_executed": before.mode,
                                      "source_command": source_command.tolist(), "effective_command": effective_command.tolist(),
                                      "runtime": asdict(result), "measurement": asdict(measurement),
                                      "supervisor": asdict(after), "differences": differences}, default=_json_value, allow_nan=False) + "\n")
            if after.halt_requested:
                reason = after.fault_reason
                break
            if stage == "standalone" and after.attempt_ended is not None:
                succeeded = after.attempt_ended.handoff_step is not None
                reason = after.attempt_ended.end_reason
                break
            if stage == "combined" and first_handoff is not None and zero_complete is not None and not after.zero_command and segment_index == len(protocol.command_segments) and step_index + 1 - first_handoff >= 500:
                succeeded = recurrent_falls == 0 and after.mode is ControlMode.LOCOMOTION
                reason = "combined_complete" if succeeded else "recurrent_fall"
                break
        else:
            reason = "evaluation_observation_limit"
            supervisor.interrupt(reason)
    return {"record_id": record.record_id, "trajectory_id": record.trajectory_id, "initial_category": record.initial_category,
            "motion_kind": record.motion_kind, "stage": stage, "success": succeeded, "end_reason": reason,
            "steps": step_index + 1, "first_handoff_step": first_handoff, "zero_complete_step": zero_complete,
            "post_handoff_ten_second_survival": first_handoff is not None and step_index + 1 - first_handoff >= 500 and recurrent_falls == 0,
            "recurrent_falls": recurrent_falls, "trace_path": str(trace_path.resolve()),
            "metrics": supervisor.metrics.state_dict(), "metrics_summary": supervisor.metrics.summary()}


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"不能序列化运行记录：{type(value).__name__}")


def verify_test_isolation(records, source_pose_records: list[dict]) -> None:
    training_records = [record for record in source_pose_records if record["split"] == "train"]
    training_ids = {record["record_id"] for record in training_records}
    training_trajectories = {record["trajectory_id"] for record in training_records}
    training_content = {record["physical_state_sha256"] for record in training_records}
    for record in records:
        if record.record_id in training_ids or record.trajectory_id in training_trajectories or pose_record_physical_sha256(record) in training_content:
            raise ValueError("独立测试初态与源训练记录、产生轨迹或完整物理内容重叠")


def run_evaluation(manifest_path: Path, bank_path: Path, output_dir: Path) -> dict:
    runtime, actor, calibration, calibration_config, protocol, manifest, recovery_manifest = load_combination(manifest_path)
    bank = PoseBank.load(bank_path)
    records = [record for record in bank.records if record.split == "test" and record.parameter_plan_sha256 == runtime.parameter_sha256]
    if not records:
        raise ValueError("没有与实际参数一致的独立测试姿态")
    counts = {category: sum(record.initial_category == category for record in records) for category in CATEGORIES}
    if not protocol.diagnostic and any(counts[category] < protocol.minimum_cases_per_fallen_category for category in CATEGORIES[:5]):
        raise ValueError("测试初态未达到各类固定分母要求")
    if not protocol.diagnostic:
        verify_test_isolation(records, recovery_manifest["source_pose_records"])
    output_dir.mkdir(parents=True, exist_ok=False)
    cases = []
    for index, record in enumerate(records):
        for stage in ("standalone", "combined"):
            cases.append(evaluate_case(runtime, actor, calibration, calibration_config, protocol, record, stage,
                                       output_dir / f"case-{index}-{stage}.jsonl.gz",
                                       supervisor_config=SupervisorConfig(**manifest["supervisor"]),
                                       heading_config=manifest["heading"]))
    groups = []
    for category in CATEGORIES:
        for stage in ("standalone", "combined"):
            subset = [case for case in cases if case["initial_category"] == category and case["stage"] == stage]
            groups.append({"initial_category": category, "stage": stage, "total_legal_initial_states": len(subset),
                           "successes": sum(case["success"] for case in subset),
                           "success_rate": sum(case["success"] for case in subset) / len(subset) if subset else None,
                           "failures": [case["end_reason"] for case in subset if not case["success"]]})
    report = {"schema_version": "RecoveryCombinedEvaluationV1", "combination_manifest_sha256": file_sha256(manifest_path),
              "pose_bank_sha256": bank.sha256, "protocol": asdict(protocol), "counts": counts,
              "category_results": groups, "cases": cases, "diagnostic": protocol.diagnostic}
    write_json(output_dir / "evaluation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    compose = sub.add_parser("compose")
    for name in ("recovery-export", "walker-export", "contact-manifest", "actual-parameters", "calibration", "protocol", "output"):
        compose.add_argument("--" + name, type=Path, required=True)
    compose.add_argument("--disable-heading", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--combination-manifest", type=Path, required=True)
    run.add_argument("--pose-bank", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "compose":
        protocol = EvaluationProtocol(**json.loads(args.protocol.read_text(encoding="utf-8")))
        parameters = json.loads(args.actual_parameters.read_text(encoding="utf-8"))
        if args.output.exists():
            raise FileExistsError("组合清单必须写入新文件")
        result = create_combination_manifest(args.recovery_export, args.walker_export, args.contact_manifest,
                                              parameters, args.calibration, protocol, heading_enabled=not args.disable_heading)
        write_json(args.output, result)
        print(args.output)
    else:
        report = run_evaluation(args.combination_manifest, args.pose_bank, args.output_dir)
        print(json.dumps(report["category_results"], indent=2))


if __name__ == "__main__":
    main()
