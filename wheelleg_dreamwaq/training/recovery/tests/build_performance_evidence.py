from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from recovery.contracts import file_sha256, load_resolved_config, source_sha256, stable_hash, write_json
from recovery.tests.run_unit_verification import junit_statistics, test_inventory


ALLOWED_CONFIG = {("environment", "run_dir"), ("assessment", "updates"), ("ppo", "save_interval")}
FIXED_IDENTITY = {"walker_actor_sha256", "walker_manifest_sha256", "asset_bundle_sha256",
    "mujoco_contact_model_sha256", "isaac_contact_model_sha256", "pose_bank_sha256", "source_pose_records_sha256",
    "parameter_plan_sha256", "num_envs", "device", "geometry_guard_sha256", "curriculum_sha256", "contact_model_sha256"}


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"证据必须是JSON对象: {path}")
    return value


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def verified_reference(record, base):
    if set(record) != {"path", "sha256"}:
        raise ValueError("证据文件引用字段不完整")
    path = Path(record["path"])
    path = path if path.is_absolute() else base / path
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"证据文件SHA改变: {path}")
    return path


def compare_config(old, new, allowed=ALLOWED_CONFIG, prefix=()):
    changes = []
    if isinstance(old, dict) and isinstance(new, dict):
        if set(old) != set(new):
            raise ValueError(f"兼容配置字段改变: {'.'.join(prefix)}")
        for key in sorted(old):
            changes.extend(compare_config(old[key], new[key], allowed, (*prefix, key)))
    elif old != new:
        if prefix not in allowed:
            raise ValueError(f"兼容配置改变训练物理或随机契约: {'.'.join(prefix)}")
        changes.append({"path": ".".join(prefix), "parent": old, "new": new})
    return changes


def require_report(path, source, schema):
    report = read_json(path)
    identity = report.get("identity", {})
    if (report.get("passed") is not True or report.get("schema_version") != schema
            or report.get("source_sha256", identity.get("source_sha256")) != source):
        raise ValueError(f"报告未通过或没有绑定当前源码: {path}")
    return report


def verify_unit(path, source):
    report = require_report(path, source, "RecoveryUnitVerificationV1")
    if (report.get("all_tests_verified") is not True or report.get("source_after_sha256") != source
            or set(report.get("runs", {})) != {"main", "mujoco"}):
        raise ValueError("单元报告没有确认全部测试及源码冻结")
    inventory = test_inventory()
    expected_suites = []
    for name, run in report["runs"].items():
        if run.get("passed") is not True or run.get("returncode") != 0:
            raise ValueError(f"单元子进程未通过: {name}")
        xml = verified_reference(run["xml"], path.parent)
        if junit_statistics(xml, inventory[name]) != run.get("statistics"):
            raise ValueError(f"JUnit统计与验收报告不同: {name}")
        expected_suites.append({"name": name, **run["statistics"]})
        for field in ("stdout", "stderr"):
            verified_reference(run[field], path.parent)
    if sorted(report.get("suites", []), key=lambda suite: suite["name"]) != sorted(expected_suites, key=lambda suite: suite["name"]):
        raise ValueError("单元suites列表与真实JUnit结果不同")
    return report


def verify_manifest(path, source):
    manifest = read_json(path)
    identity, config = manifest["identity"], manifest["resolved_config"]
    if (manifest.get("schema_version") != "RecoveryTrainingRunV1" or identity["source_sha256"] != source
            or identity["training_config_sha256"] != stable_hash(config)):
        raise ValueError(f"真实运行清单身份无效: {path}")
    environment = config["environment"]
    actual = stable_hash({"asset_bundle_sha256": identity["asset_bundle_sha256"],
        "collision_manifest": manifest["isaac_contact_manifest"], "ground_size_xy_m": environment["ground_size_xy_m"],
        "ground_top_z_m": 0.0, "physics_dt_s": config["contract"]["isaac_physics_dt_s"],
        "decimation": config["contract"]["isaac_decimation"]})
    execution = manifest["execution"]
    if (actual != identity["isaac_contact_model_sha256"]
            or stable_hash({"isaac": actual, "mujoco": identity["mujoco_contact_model_sha256"]}) != identity["contact_model_sha256"]
            or stable_hash(execution) != identity["execution_implementation_sha256"]
            or execution["geometry"] != manifest["geometry_guard_identity"]
            or execution["geometry"]["config_sha256"] != stable_hash(config["readiness"]["geometry_guard"])):
        raise ValueError(f"真实接触或执行实现SHA不一致: {path}")
    expected_contact = environment["isaac_contact_model_sha256"]
    if expected_contact is not None and expected_contact != actual:
        raise ValueError("清单实际接触与配置要求不一致")
    if read_json(path.parent / "resolved_config.json") != config:
        raise ValueError("清单配置与真实运行resolved_config不同")
    return manifest


def verify_interface(path, count, source, target, reference_manifest):
    report = require_report(path, source, "RecoveryTrainingVerificationV1")
    if (report.get("parameters_changed") is not True
            or report.get("resume_and_continued_optimization_verified") is not True
            or report["identity"]["num_envs"] != count):
        raise ValueError(f"真实{count}环境接口没有证明优化与恢复继续优化")
    manifest = verify_manifest(path.parent / "run_manifest.json", source)
    if manifest["identity"] != report["identity"] or manifest["resolved_config"]["environment"]["task_type"] != "interface_validation":
        raise ValueError("接口报告与真实运行清单不同")
    compare_config(manifest["resolved_config"], target,
        ALLOWED_CONFIG | {("environment", "num_envs"), ("environment", "task_type")})
    for key in FIXED_IDENTITY - {"num_envs", "parameter_plan_sha256"}:
        if manifest["identity"].get(key) != reference_manifest["identity"].get(key):
            raise ValueError(f"接口与6144参考运行的固定身份不同: {key}")
    if manifest["execution"] != reference_manifest["execution"]:
        raise ValueError("接口和6144参考运行的执行实现不同")
    return report


def main():
    parser = argparse.ArgumentParser(description="核验实际性能验收产物并生成兼容导入证据，不启动训练")
    for name in ("unit-report", "logging-report", "interface4-report", "interface256-report", "reference-manifest",
                 "target-config", "parent-checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("性能验收证据必须使用新文件路径")
    source = source_sha256()
    verify_unit(args.unit_report.resolve(), source)
    logging = require_report(args.logging_report, source, "RecoveryMetricsPipelineVerificationV1")
    component = logging.get("component", {})
    if component.get("num_envs") != 6144 or component.get("post_step_calls") != 6144 * 24 or not component.get("device", "").startswith("cuda"):
        raise ValueError("日志报告必须包含6144环境24步真实CUDA组件验收")
    target = load_resolved_config(args.target_config)
    reference_manifest = verify_manifest(args.reference_manifest.resolve(), source)
    reference_identity = reference_manifest["identity"]
    reference_report_path = args.reference_manifest.parent / "training_verification.json"
    reference_report = require_report(reference_report_path, source, "RecoveryTrainingVerificationV1")
    if reference_report.get("parameters_changed") is not True or reference_report["identity"] != reference_identity:
        raise ValueError("6144参考清单没有对应真实通过的PPO报告")
    if reference_identity["num_envs"] != 6144 or target["environment"]["num_envs"] != 6144:
        raise ValueError("性能参考与目标配置必须均为6144环境")
    reference_changes = compare_config(reference_manifest["resolved_config"], target)
    verify_interface(args.interface4_report.resolve(), 4, source, target, reference_manifest)
    verify_interface(args.interface256_report.resolve(), 256, source, target, reference_manifest)
    # preflight只校验资产、参数计划和身份；不创建Isaac场景、不导入checkpoint、不调用训练。
    from recovery.train import preflight

    new_identity = preflight(target)
    actual_contact = reference_identity["isaac_contact_model_sha256"]
    expected_contact = target["environment"]["isaac_contact_model_sha256"]
    if expected_contact is not None and expected_contact != actual_contact:
        raise ValueError("目标配置接触身份与参考实测不同")
    new_identity["isaac_contact_model_sha256"] = actual_contact
    new_identity["contact_model_sha256"] = stable_hash({"isaac": actual_contact, "mujoco": new_identity["mujoco_contact_model_sha256"]})
    new_identity["execution_implementation_sha256"] = reference_identity["execution_implementation_sha256"]
    for key, value in new_identity.items():
        if key != "training_config_sha256" and reference_identity.get(key) != value:
            raise ValueError(f"目标preflight身份与参考实测身份不同: {key}")
    parent = args.parent_checkpoint.resolve()
    parent_manifest_path = parent.parent / "run_manifest.json"
    parent_manifest = read_json(parent_manifest_path)
    parent_identity = parent_manifest.get("identity", parent_manifest.get("training_identity"))
    parent_config = read_json(parent.parent / "resolved_config.json")
    if not isinstance(parent_identity, dict) or stable_hash(parent_config) != parent_identity["training_config_sha256"]:
        raise ValueError("父运行配置与身份不同")
    index_path = Path(str(parent) + ".index.json")
    index = read_json(index_path)
    if (index.get("schema_version") != "RecoveryCheckpointIndexV1" or index["checkpoint_file"] != parent.name
            or index["identity_sha256"] != stable_hash(parent_identity) or index["checkpoint_sha256"] != file_sha256(parent)):
        raise ValueError("父checkpoint索引或文件SHA不一致")
    parent_changes = compare_config(parent_config, target)
    if not FIXED_IDENTITY <= parent_identity.keys() or not FIXED_IDENTITY <= new_identity.keys():
        raise ValueError("父运行或目标运行缺少固定物理身份")
    for key in FIXED_IDENTITY:
        if parent_identity[key] != new_identity[key]:
            raise ValueError(f"兼容导入固定身份改变: {key}")
    legacy = logging.get("legacy", {})
    if (legacy.get("parent_checkpoint_sha256") != index["checkpoint_sha256"]
            or any(legacy.get(key) is not True for key in ("all_stream_raw_recomputation_passed", "next_ids_preserved",
                                                        "aggregate_and_censored_denominators_preserved"))):
        raise ValueError("日志验收没有验证同一父checkpoint的原始归档与删失统计")
    if source_sha256() != source:
        raise ValueError("生成证据期间源码发生改变")
    checks = {name: {"passed": True, **reference(path)} for name, path in (
        ("geometry", args.unit_report), ("fsm", args.unit_report), ("logging", args.logging_report),
        ("interface4", args.interface4_report), ("interface256", args.interface256_report))}
    evidence = {"schema_version": "RecoveryPerformanceEquivalenceEvidenceV1", "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": source,
        "parent_identity_sha256": stable_hash(parent_identity), "new_identity_sha256": stable_hash(new_identity),
        "checks": checks, "new_identity": new_identity, "parent_identity": parent_identity,
        "reference_manifest": reference(args.reference_manifest), "target_config": reference(args.target_config),
        "reference_training_report": reference(reference_report_path),
        "parent_checkpoint": reference(parent), "parent_checkpoint_index": reference(index_path),
        "reference_configuration_changes": reference_changes, "parent_configuration_changes": parent_changes}
    write_json(args.output, evidence)
    print(json.dumps({"passed": True, **reference(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
