from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .contracts import RecoveryContractV2, file_sha256, load_resolved_config, stable_hash, validate_resolved_config, write_json
from .pose_bank import PoseBank
from .randomization import RecoveryRandomization, RecoveryRandomizationV1


REQUIRED_CHECKS = (
    "complete_contact_isaac", "complete_contact_mujoco", "walker_full_contact_baseline",
    "mechanical_recovery_by_category", "pose_classification", "contact_and_clearance",
    "handoff_calibration", "runtime_resources",
)


def _verified_json(reference: dict, base: Path) -> tuple[dict, Path]:
    if set(reference) != {"path", "sha256"}:
        raise ValueError("证据引用必须包含path和sha256")
    path = Path(reference["path"])
    if not path.is_absolute():
        path = base / path
    if not path.is_file() or file_sha256(path) != reference["sha256"]:
        raise ValueError(f"验证证据缺失或SHA256不一致: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("验证证据必须为JSON对象")
    return result, path


def verify_formal_evidence(config: dict, bank: PoseBank, identity: dict) -> dict:
    validate_resolved_config(config)
    if config["environment"]["task_type"] != "formal_training":
        raise ValueError("正式训练预检只接受formal_training配置")
    if identity["training_config_sha256"] != stable_hash(config) or identity["pose_bank_sha256"] != bank.sha256:
        raise ValueError("正式训练配置或姿态库身份不一致")
    coverage = bank.validate_coverage(formal=True)
    path = Path(config["assessment"]["formal_evidence_path"])
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != "RecoveryFormalEvidenceV1":
        raise ValueError("正式训练前置实验记录版本无效")
    if evidence.get("identity") != identity:
        raise ValueError("前置实验记录不对应当前源码、模型、策略、姿态库或实际参数计划")
    checks = evidence.get("checks", {})
    if set(checks) != set(REQUIRED_CHECKS):
        raise ValueError("正式训练前置实验记录缺少必需检查")
    verified = {}
    for name in REQUIRED_CHECKS:
        report, report_path = _verified_json(checks[name], path.parent)
        if report.get("status") != "passed":
            raise ValueError(f"正式训练前置实验尚未通过: {name}")
        if report.get("identity") != identity:
            raise ValueError(f"前置实验身份不匹配: {name}")
        verified[name] = {"path": str(report_path.resolve()), "sha256": file_sha256(report_path)}
        if name == "mechanical_recovery_by_category":
            categories = report.get("categories", {})
            if set(categories) != {"front", "back", "left", "right", "supine"}:
                raise ValueError("机械恢复证据必须覆盖五类倒地")
            for category, entry in categories.items():
                trajectory, _ = _verified_json(entry, report_path.parent)
                if (trajectory.get("unassisted") is not True or trajectory.get("status") != "passed"
                        or trajectory.get("initial_category") != category or not trajectory.get("substeps")):
                    raise ValueError(f"缺少真实无辅助合法恢复子步轨迹: {category}")
                for step in trajectory["substeps"]:
                    limits = (step["max_abs_leg_torque_nm"] <= 18,
                              step["max_abs_wheel_torque_nm"] <= 9,
                              0 <= step["loop_error_m"] < config["contract"]["max_loop_error_m"],
                              step["clearance_valid"] is True, step["contact_valid"] is True)
                    if not all(limits) or not all(math.isfinite(float(step[key])) for key in (
                        "max_abs_leg_target_rad", "max_abs_leg_torque_nm", "max_abs_wheel_torque_nm", "loop_error_m")):
                        raise ValueError(f"恢复子步轨迹违反执行器、闭链、接触或间隙约束: {category}")
        if name == "runtime_resources" and report.get("num_envs") != config["environment"]["num_envs"]:
            raise ValueError("资源实测不对应正式并行环境数量")
        if name == "handoff_calibration":
            _verify_calibration_sources(report, report_path, config, identity)
        if name == "contact_and_clearance":
            guard = config["readiness"]["geometry_guard"]
            if (report.get("geometry_guard_config_sha256") != stable_hash(guard)
                    or report.get("isaac_body_geometry_mapping_verified") is not True
                    or file_sha256(guard["contact_manifest"]) != identity["mujoco_contact_model_sha256"]):
                raise ValueError("逐子步几何检查缺少同源Isaac刚体与接触几何映射验收")
    calibration_path = Path(config["readiness"]["calibration_path"])
    if file_sha256(calibration_path) != verified["handoff_calibration"]["sha256"]:
        raise ValueError("成功条件必须来自本次已验证的交接标定文件")
    return {"status": "passed", "schema_version": "RecoveryPreflightResultV1", "identity": identity,
            "coverage": coverage, "verified_checks": verified, "formal_evidence_sha256": file_sha256(path)}


def _verify_calibration_sources(report: dict, path: Path, config: dict, identity: dict) -> None:
    if report.get("schema_version") != "RecoveryHandoffCalibrationEvidenceV1":
        raise ValueError("正式交接证据需要包装真实后端标定报告并绑定完整训练身份")
    readiness = config["readiness"]
    for key in ("support_force_n", "target_difference_limits", "torque_difference_limits"):
        if report.get(key) != readiness[key]:
            raise ValueError(f"正式交接证据与训练成功条件不同: {key}")
    references = report.get("source_calibrations", [])
    if not references:
        raise ValueError("正式交接证据缺少真实标定报告")
    env = config["environment"]
    plans = RecoveryRandomization(env["num_envs"], env["seed"], "cpu",
        RecoveryRandomizationV1(**config["randomization"]), config["contract"]["q_nominal"])
    required_plans = {plan.sha256 for plan in plans.plans}
    covered = set()
    for reference in references:
        source, _ = _verified_json(reference, path.parent)
        if (source.get("schema_version") != "RecoveryCompatibilityCalibrationV2"
                or source.get("status") != "accepted" or source.get("recovery_terminal_validation") is not True):
            raise ValueError("正式交接证据必须包含真实起身终态上的独立接管标定")
        source_identity = source["identity"]
        walker = source_identity["walker"]
        if (walker["actor_sha256"] != identity["walker_actor_sha256"]
                or walker["manifest_sha256"] != identity["walker_manifest_sha256"]
                or source_identity["contact_manifest_sha256"] != identity["mujoco_contact_model_sha256"]
                or source_identity.get("recovery_contract_sha256") != RecoveryContractV2.from_dict(config["contract"]).sha256):
            raise ValueError("标定来源的行走策略、MuJoCo完整接触模型或起身动作契约不一致")
        plan = source_identity["parameter_plan_sha256"]
        if plan not in required_plans or plan in covered:
            raise ValueError("标定来源参数计划无效或重复")
        if source["config"]["support_force_n"] != readiness["support_force_n"]:
            raise ValueError("源标定支撑门限与训练条件不同")
        accepted = []
        accepted_terminals = set()
        for sample in source["samples"]:
            if (all(value < limit for value, limit in zip(sample["target_difference"], readiness["target_difference_limits"]))
                    and all(value < limit for value, limit in zip(sample["torque_difference_nm"], readiness["torque_difference_limits"]))):
                if sample["partition"] == "test":
                    accepted.append(sample)
                if (sample["source"] == "recovery_terminal" and sample["initial_physical_validation"] is None
                        and sample["takeover"]["success"]):
                    accepted_terminals.add(sample["partition"])
        failures = sum(sample["initial_physical_validation"] is not None or not sample["takeover"]["success"] for sample in accepted)
        if (len(accepted) < source["config"]["minimum_accepted_validation_samples"]
                or failures > source["config"]["allowed_validation_failure_rate"] * len(accepted)):
            raise ValueError("统一六维门限未在该参数计划的独立真实接管数据上通过")
        if accepted_terminals != {"train", "test"}:
            raise ValueError("统一六维门限未同时接受训练和独立验证的真实起身终态")
        covered.add(plan)
    if covered != required_plans:
        raise ValueError("交接标定未覆盖全部实际训练参数计划")


def main() -> None:
    parser = argparse.ArgumentParser(description="检查倒地起身正式训练的真实实验前置条件")
    parser.add_argument("--config", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_resolved_config(args.config)
    identity = json.loads(Path(args.identity).read_text(encoding="utf-8"))
    result = verify_formal_evidence(config, PoseBank.load(config["environment"]["pose_bank_path"]), identity)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
