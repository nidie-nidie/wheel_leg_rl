from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .contact_model import file_sha256, payload_sha256
from .contracts import write_json
from .control import check_action_compatibility
from .mujoco_runtime import MujocoRecoveryRuntime, supervisor_measurements
from .mujoco_runtime import RuntimeStep
from .observations import normalize_physical_frame
from .pose_bank import PoseBank, pose_record_physical_sha256
from .export import load_export
from .supervisor import SupervisorConfig, physical_conditions


@dataclass(frozen=True)
class CalibrationConfig:
    seed: int
    training_samples: int
    validation_samples: int
    warmup_steps: int
    takeover_steps: int
    maximum_raw_action_perturbation: tuple[float, ...]
    quantile: float
    minimum_successful_training_samples: int
    minimum_accepted_validation_samples: int
    allowed_validation_failure_rate: float
    support_force_n: float
    maximum_contact_penetration_m: float
    minimum_clearance_m: float
    allowed_body_pairs: list[list[str]]

    def __post_init__(self):
        if self.seed < 0 or min(self.training_samples, self.validation_samples, self.warmup_steps, self.takeover_steps) < 1:
            raise ValueError("校准样本数、种子与物理步数无效")
        if len(self.maximum_raw_action_perturbation) != 6 or any(v <= 0 for v in self.maximum_raw_action_perturbation):
            raise ValueError("必须指定六维非零合法目标扰动幅值")
        if not 0 < self.quantile <= 1 or not 0 <= self.allowed_validation_failure_rate < 1:
            raise ValueError("校准分位数或允许失败率无效")
        if self.support_force_n <= 0 or self.maximum_contact_penetration_m < 0:
            raise ValueError("支撑及接触压缩门限需要显式指定")
        if not 1 <= self.minimum_successful_training_samples <= self.training_samples:
            raise ValueError("训练校准有效样本数量要求无效")
        if not 1 <= self.minimum_accepted_validation_samples <= self.validation_samples:
            raise ValueError("独立校准验收数量要求无效")
        payload_sha256(asdict(self))


def action_differences(runtime: MujocoRecoveryRuntime, actual: np.ndarray, candidate: np.ndarray, *,
                       actual_is_walker: bool = False) -> tuple[np.ndarray, np.ndarray]:
    actual_leg, actual_wheel = runtime.decoded(actual, locomotion=actual_is_walker)
    candidate_leg, candidate_wheel = runtime.decoded(candidate, locomotion=True)
    return (np.abs(np.concatenate((actual_leg - candidate_leg, actual_wheel - candidate_wheel))),
            np.abs(runtime.estimated_torque(actual, locomotion=actual_is_walker)
                   - runtime.estimated_torque(candidate, locomotion=True)))


def compatibility(runtime: MujocoRecoveryRuntime, actual: np.ndarray, candidate: np.ndarray, calibration: dict, *,
                  actual_is_walker: bool = False) -> tuple[bool, tuple[str, ...], dict]:
    state = runtime.state()
    tensor = lambda values: torch.as_tensor(values, dtype=torch.float32)
    checked = check_action_compatibility(tensor(actual), tensor(candidate), tensor(state.joint_position_control),
        tensor(state.joint_velocity_control), tensor(runtime.q_reference), tensor(runtime.kp), tensor(runtime.kd),
        tensor(runtime.effort_limits), tensor(calibration["target_difference_limits"]),
        tensor(calibration["torque_difference_limits"]), runtime.contract, actual_is_walker=actual_is_walker)
    blockers = tuple([f"target_difference/joint{index}" for index in np.flatnonzero(checked["target_blocked"].numpy())] +
                     [f"torque_difference/joint{index}" for index in np.flatnonzero(checked["torque_blocked"].numpy())])
    return bool(checked["compatible"]), blockers, {"target_difference": checked["target_difference"].tolist(),
                                                  "torque_difference_nm": checked["torque_difference"].tolist()}


def _measurement(runtime, step, config):
    return supervisor_measurements(runtime, step, support_force_threshold_n=config.support_force_n,
                                   maximum_contact_penetration_m=config.maximum_contact_penetration_m,
                                   minimum_clearance_m=config.minimum_clearance_m, action_compatible=True)


def _takeover(runtime: MujocoRecoveryRuntime, config: CalibrationConfig) -> dict:
    hold = maximum_hold = 0
    reason = None
    errors = []
    for index in range(config.takeover_steps):
        step = runtime.step(runtime.walker_action(), locomotion=True)
        measurement = _measurement(runtime, step, config)
        hold = hold + 1 if all(physical_conditions(measurement).values()) else 0
        maximum_hold = max(maximum_hold, hold)
        if measurement.physical_failure_reason is not None or measurement.tilt_rad > SupervisorConfig().fall_tilt_rad:
            reason = measurement.physical_failure_reason or "walker_fall"
            break
        errors.append(float(np.linalg.norm(runtime.state().com_linear_velocity_control)))
    return {"success": reason is None and maximum_hold >= 100, "reason": reason,
            "steps_executed": index + 1, "maximum_stable_hold_steps": maximum_hold,
            "maximum_com_speed_m_s": max(errors) if errors else None}


def recovery_terminal_pair(runtime: MujocoRecoveryRuntime, record) -> dict:
    runtime.restore_pose(record)
    actual = np.asarray(record.prior_actions[-1], dtype=np.float64)
    if not np.allclose(runtime.previous_action, actual, atol=1e-7, rtol=0):
        raise ValueError("起身终态上一动作与真实历史不一致")
    candidate = runtime.walker_action()
    targets, torques = action_differences(runtime, actual, candidate)
    return {"actual_action": actual.tolist(), "candidate_action": candidate.tolist(),
            "target_difference": targets.tolist(), "torque_difference_nm": torques.tolist(),
            "state": runtime.snapshot()}


def _validate_terminal_action(runtime: MujocoRecoveryRuntime, record, step: RuntimeStep) -> None:
    if not np.allclose(step.action, record.prior_actions[-1], atol=1e-7, rtol=0) or len(step.substeps) != runtime.contract.mujoco_decimation:
        raise ValueError("终态子步记录不对应最后实际起身动作")
    expected_leg, expected_wheel = runtime.decoded(np.asarray(step.action), locomotion=False)
    targets = np.concatenate((step.leg_target, step.wheel_target))
    if not np.allclose(targets, np.concatenate((expected_leg, expected_wheel)), atol=1e-6, rtol=0):
        raise ValueError("终态电机目标无法由当前起身动作契约重算")
    if not np.allclose(targets, record.motor_targets, atol=1e-6, rtol=0):
        raise ValueError("终态实际电机目标与姿态库记录不同")


def validate_recovery_terminal_records(runtime: MujocoRecoveryRuntime, bank: PoseBank | None) -> list:
    if bank is None:
        return []
    records = [record for record in bank.records if record.parameter_plan_sha256 == runtime.parameter_sha256
               and record.metadata.get("origin") == "recovery_terminal"]
    train = [record for record in records if record.split == "train"]
    test = [record for record in records if record.split == "test"]
    train_ids = {record.record_id for record in train}
    train_trajectories = {record.trajectory_id for record in train}
    train_content = {pose_record_physical_sha256(record) for record in train}
    if any(record.record_id in train_ids or record.trajectory_id in train_trajectories or
           pose_record_physical_sha256(record) in train_content for record in test):
        raise ValueError("起身终态训练与校准测试记录的轨迹或物理内容重叠")
    for partition in (train, test):
        if (len({record.trajectory_id for record in partition}) != len(partition) or
            len({pose_record_physical_sha256(record) for record in partition}) != len(partition)):
            raise ValueError("每个起身终态校准样本必须来自不同真实轨迹和物理状态")
    for record in records:
        metadata = record.metadata
        required = {"recovery_export_dir", "recovery_actor_sha256", "recovery_actor_physical_history", "recovery_terminal_step"}
        if not required <= metadata.keys() or metadata.get("history_source") != "physical_trajectory":
            raise ValueError("起身终态缺少可核验策略、真实Actor输入及实际结束动作记录")
        actor, manifest = load_export(metadata["recovery_export_dir"])
        if manifest["actor_sha256"] != metadata["recovery_actor_sha256"]:
            raise ValueError("起身终态Actor身份与导出文件不同")
        if payload_sha256(manifest["contract"]) != payload_sha256(asdict(runtime.contract)):
            raise ValueError("起身终态导出动作契约与当前运行不同")
        source = manifest["identity"]
        if (source["walker_actor_sha256"] != runtime.walker_identity["actor_sha256"] or
            source["walker_manifest_sha256"] != runtime.walker_identity["manifest_sha256"] or
            source["asset_bundle_sha256"] != record.asset_bundle_sha256 or
            record.asset_bundle_sha256 != runtime.walker_manifest["asset_bundle_hash"] or
            source.get("mujoco_contact_model_sha256") != record.contact_model_sha256 or
            record.contact_model_sha256 != file_sha256(runtime.contact_manifest_path)):
            raise ValueError("起身终态策略的物理或冻结专家来源不一致")
        history = np.asarray(metadata["recovery_actor_physical_history"], dtype=np.float32)
        if history.shape != (5, 22) or not np.isfinite(history).all():
            raise ValueError("起身终态缺少实际动作前的五帧物理输入")
        if not np.allclose(history[1:], np.asarray(record.clean_history)[:-1], atol=1e-6, rtol=0):
            raise ValueError("起身终态动作前后历史不连续")
        with torch.inference_mode():
            inferred = actor(normalize_physical_frame(torch.from_numpy(history)).reshape(1, 110)).squeeze(0).numpy()
        if not np.allclose(inferred, record.prior_actions[-1], atol=1e-7, rtol=0):
            raise ValueError("记录中的最后起身动作无法由指定Actor和真实历史重算")
        step = RuntimeStep(**metadata["recovery_terminal_step"])
        _validate_terminal_action(runtime, record, step)
    return records


def _record_sample(runtime, config, *, sample_id, partition, source, pair, initial_failure,
                    pose_record=None, pose_bank=None) -> dict:
    label = _takeover(runtime, config)
    runtime.restore_snapshot(pair["state"])
    return {"sample_id": sample_id, "partition": partition, "source": source,
            "record_id": None if pose_record is None else pose_record.record_id,
            "pose_record_id": None if pose_record is None else pose_record.record_id,
            "trajectory_id": None if pose_record is None else pose_record.trajectory_id,
            "recovery_actor_sha256": None if pose_record is None else pose_record.metadata["recovery_actor_sha256"],
            "physical_state_sha256": None if pose_record is None else pose_record_physical_sha256(pose_record),
            "pose_bank_sha256": None if pose_bank is None else pose_bank.sha256,
            **pair, "takeover": label, "initial_physical_validation": initial_failure}


def accepted_samples(samples: list[dict], target_limits: np.ndarray, torque_limits: np.ndarray) -> tuple[list[dict], bool]:
    accepted = [sample for sample in samples if
                np.all(np.asarray(sample["target_difference"]) < target_limits) and
                np.all(np.asarray(sample["torque_difference_nm"]) < torque_limits)]
    recovery_validated = all(any(sample["partition"] == partition and sample["source"] == "recovery_terminal"
                                and sample["takeover"]["success"] and sample["initial_physical_validation"] is None
                                for sample in accepted) for partition in ("train", "test"))
    return accepted, recovery_validated


def run_calibration(runtime: MujocoRecoveryRuntime, config: CalibrationConfig, *,
                    recovery_pose_bank: PoseBank | None = None) -> dict:
    samples = []
    identity = {"walker": runtime.walker_identity, "parameter_plan_sha256": runtime.parameter_sha256,
                "contact_manifest_sha256": file_sha256(runtime.contact_manifest_path),
                "asset_bundle_sha256": runtime.walker_manifest["asset_bundle_hash"],
                "recovery_contract_sha256": runtime.contract.sha256}
    source_records = validate_recovery_terminal_records(runtime, recovery_pose_bank)
    for partition, count, seed_offset in (("train", config.training_samples, 0), ("test", config.validation_samples, 1)):
        generator = np.random.default_rng(np.random.SeedSequence([config.seed, seed_offset]))
        for sample_index in range(count):
            runtime.reset_nominal()
            for _ in range(config.warmup_steps):
                runtime.step(runtime.walker_action(), locomotion=True)
            perturbation = (np.zeros(6) if sample_index == 0 else
                            generator.uniform(-np.asarray(config.maximum_raw_action_perturbation),
                                              np.asarray(config.maximum_raw_action_perturbation)))
            actual = runtime.walker_action() + perturbation
            executed = runtime.step(actual)
            candidate = runtime.walker_action()
            targets, torques = action_differences(runtime, actual, candidate)
            pair = {"state": runtime.snapshot(), "actual_action": actual.tolist(), "candidate_action": candidate.tolist(),
                    "target_difference": targets.tolist(), "torque_difference_nm": torques.tolist()}
            samples.append(_record_sample(runtime, config, sample_id=f"expert-{partition}-{sample_index}",
                partition=partition, source="expert_control" if sample_index == 0 else "expert_target_perturbation",
                pair=pair, initial_failure=_measurement(runtime, executed, config).physical_failure_reason))
        for record in [record for record in source_records if record.split == partition]:
            pair = recovery_terminal_pair(runtime, record)
            terminal_step = RuntimeStep(**record.metadata["recovery_terminal_step"])
            measurement = _measurement(runtime, terminal_step, config)
            initial_failure = measurement.physical_failure_reason
            if initial_failure is None and not all(physical_conditions(measurement).values()):
                initial_failure = "recovery_terminal_not_physically_ready"
            samples.append(_record_sample(runtime, config, sample_id=f"recovery-{partition}-{record.record_id}",
                partition=partition, source="recovery_terminal", pair=pair, initial_failure=initial_failure,
                pose_record=record, pose_bank=recovery_pose_bank))
    positive = [sample for sample in samples if sample["partition"] == "train" and sample["source"] != "expert_control" and sample["takeover"]["success"]
                and sample["initial_physical_validation"] is None]
    if len(positive) < config.minimum_successful_training_samples:
        return {"schema_version": "RecoveryCompatibilityCalibrationV2", "status": "rejected",
                "reason": "insufficient_successful_training_samples", "samples": samples, "config": asdict(config),
                "identity": identity, "recovery_terminal_validation": False, "formal_validation": "required"}
    target_limits = np.quantile(np.asarray([sample["target_difference"] for sample in positive]), config.quantile, axis=0).astype(np.float32)
    torque_limits = np.quantile(np.asarray([sample["torque_difference_nm"] for sample in positive]), config.quantile, axis=0).astype(np.float32)
    if np.any(target_limits <= 0) or np.any(torque_limits <= 0):
        raise ValueError("真实正样本产生零容差维度，采集未覆盖该维度")
    accepted_all, recovery_validated = accepted_samples(samples, target_limits, torque_limits)
    accepted = [sample for sample in accepted_all if sample["partition"] == "test"]
    failures = sum(not sample["takeover"]["success"] or sample["initial_physical_validation"] is not None for sample in accepted)
    passed = len(accepted) >= config.minimum_accepted_validation_samples and failures <= config.allowed_validation_failure_rate * len(accepted)
    status = ("accepted" if recovery_validated else "diagnostic_accepted") if passed else "rejected"
    return {"schema_version": "RecoveryCompatibilityCalibrationV2", "status": status,
            "target_difference_limits": target_limits.tolist(), "torque_difference_limits": torque_limits.tolist(),
            "target_difference_units": ["rad"] * 4 + ["rad/s"] * 2, "torque_difference_unit": "Nm",
            "support_force_n": config.support_force_n,
            "accepted_validation_samples": len(accepted), "accepted_validation_failures": failures,
            "training_successful_samples": len(positive), "config": asdict(config), "samples": samples,
            "identity": identity,
            "recovery_terminal_validation": recovery_validated,
            "formal_validation": "required"}


def load_calibration(path: str | Path, runtime: MujocoRecoveryRuntime, *, allow_unvalidated_recovery: bool = False) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    accepted_states = ("accepted", "diagnostic_accepted") if allow_unvalidated_recovery else ("accepted",)
    if record["schema_version"] != "RecoveryCompatibilityCalibrationV2":
        raise ValueError("交接校准动作契约版本无效")
    identity = record["identity"]
    if identity.get("recovery_contract_sha256") != runtime.contract.sha256:
        raise ValueError("交接校准与当前起身动作契约不同")
    if record["status"] not in accepted_states:
        raise ValueError("交接校准尚未通过真实独立样本验证")
    if identity["walker"] != runtime.walker_identity or identity["parameter_plan_sha256"] != runtime.parameter_sha256 or identity["contact_manifest_sha256"] != file_sha256(runtime.contact_manifest_path):
        raise ValueError("交接校准与当前模型、参数或行走策略身份不一致")
    if not allow_unvalidated_recovery and not record["recovery_terminal_validation"]:
        raise ValueError("校准尚未包含真实起身终态验证")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--walker-export", type=Path, required=True)
    parser.add_argument("--actual-parameters", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recovery-pose-bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = CalibrationConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    runtime = MujocoRecoveryRuntime(args.contact_manifest, args.walker_export, allowed_body_pairs=config.allowed_body_pairs,
                                    actual_parameters=json.loads(args.actual_parameters.read_text(encoding="utf-8")))
    bank = None if args.recovery_pose_bank is None else PoseBank.load(args.recovery_pose_bank)
    if args.output.exists():
        raise FileExistsError("校准产物必须写入新文件")
    result = run_calibration(runtime, config, recovery_pose_bank=bank)
    write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
