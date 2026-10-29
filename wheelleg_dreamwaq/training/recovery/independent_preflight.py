from __future__ import annotations

import json
from pathlib import Path

from .contracts import file_sha256, source_sha256, stable_hash
from .independent_config import SCHEMA, validate_independent_config


REQUIRED_CHECKS = (
    "reference_identity", "standing_gate_positive_negative", "reward_ordering",
    "bounded_distribution_export", "isaac_updates_save_restore", "cross_sim_physics",
    "initialization_geometry_splits", "runtime_resources",
)


def periodic_evaluation_ids(bank):
    selected = []
    for source in ("front", "back", "left", "right", "supine", "near_upright"):
        candidates = sorted(r["record_id"] for r in bank.records if r["source"] == source and r["split"] == "test")
        if len(candidates) < 2:
            raise ValueError(f"定期测评需要每类至少两个固定留出姿态: {source}")
        selected.extend(candidates[:2])
    return selected


def independent_preflight(config: dict) -> dict:
    from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
    from .independent_poses import IndependentPoseBank
    from .standing_reference import StandingReference
    from .standing_calibration import load_standing_calibration
    from .randomization import RecoveryRandomization, RecoveryRandomizationV1

    validate_independent_config(config)
    ref = config["reference"]
    StandingReference.load(ref["manifest_path"], ref["bank_path"], ref["bank_sha256"],
                           expected_manifest_sha256=ref["manifest_sha256"])
    bank = IndependentPoseBank.load(config["environment"]["pose_bank_path"])
    coverage = bank.validate_split_ids(config["initialization"])
    periodic_evaluation_ids(bank)
    asset = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)
    plans = RecoveryRandomization(1, config["environment"]["seed"], "cpu",
        RecoveryRandomizationV1(**config["randomization"]), config["contract"]["q_nominal"])
    contact_sha = file_sha256(config["geometry_guard"]["contact_manifest"])
    for record in bank.records:
        if (record["asset_bundle_sha256"] != asset.bundle_hash
                or record["contact_model_sha256"] != contact_sha
                or record["parameter_plan_sha256"] != plans.plan(0).sha256):
            raise ValueError(f"初态资产、接触或名义参数身份不一致: {record['record_id']}")
    assessment = config["assessment"]
    calibration = load_standing_calibration(assessment["calibration_path"], assessment["calibration_sha256"],
        require_formal=config["environment"]["task_type"] == "formal_training")
    if (calibration["reference"]["bank_sha256"] != ref["bank_sha256"]
            or calibration["reference"]["manifest_sha256"] != ref["manifest_sha256"]):
        raise ValueError("站立校准与成对参考身份不一致")
    identity = {
        "schema_version": SCHEMA, "source_sha256": source_sha256(),
        "config_sha256": stable_hash(config), "contract_sha256": stable_hash(config["contract"]),
        "reference_sha256": ref["bank_sha256"], "reference_manifest_sha256": ref["manifest_sha256"],
        "calibration_sha256": assessment["calibration_sha256"],
        "geometry_sha256": stable_hash(config["geometry_guard"]),
        "contact_manifest_sha256": file_sha256(config["geometry_guard"]["contact_manifest"]),
        "pose_bank_sha256": file_sha256(config["environment"]["pose_bank_path"]),
        "asset_bundle_sha256": asset.bundle_hash,
        "num_envs": config["environment"]["num_envs"], "seed": config["environment"]["seed"],
    }
    if config["environment"]["task_type"] == "formal_training":
        if calibration.get("formal_ready") is not True:
            raise ValueError("完整站立正负样本校准尚未通过，不能启动正式训练")
        verify_independent_evidence(config, identity, coverage)
    return identity


def verify_independent_evidence(config: dict, identity: dict, coverage: dict) -> dict:
    path_value = config["assessment"]["evidence_path"]
    if not path_value:
        raise ValueError("独立站立正式训练缺少专用准入证据")
    path = Path(path_value)
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (evidence.get("schema_version") != "IndependentStandingFormalEvidenceV1"
            or evidence.get("identity") != identity or set(evidence.get("checks", {})) != set(REQUIRED_CHECKS)):
        raise ValueError("新任务准入证据版本、身份或必需检查不完整")
    verified = {}
    for name in REQUIRED_CHECKS:
        item = evidence["checks"][name]
        if set(item) != {"path", "sha256"}:
            raise ValueError(f"准入证据引用无效: {name}")
        report_path = Path(item["path"])
        if not report_path.is_absolute():
            report_path = path.parent / report_path
        if file_sha256(report_path) != item["sha256"]:
            raise ValueError(f"准入证据身份变化: {name}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if name == "isaac_updates_save_restore":
            measured_identity = report.get("identity", {})
            shared_keys = set(identity) - {"config_sha256", "num_envs"}
            if (report.get("schema_version") != "IndependentStandingTrainingVerificationV1"
                    or report.get("passed") is not True
                    or any(measured_identity.get(key) != identity[key] for key in shared_keys)
                    or measured_identity.get("num_envs") != 4):
                raise ValueError("真实4环境报告的源码、物理、参考或校准身份不一致")
            restored = report.get("save_restore", {})
            if (restored.get("num_envs") != 4 or restored.get("updates_before_save", 0) < 2
                    or restored.get("updates_after_restore", 0) < 2 or report.get("parameters_changed") is not True
                    or restored.get("observation_exactly_restored") is not True):
                raise ValueError("缺少真实4环境2次更新、保存恢复后再2次更新的证据")
        elif report.get("status") != "passed" or report.get("identity") != identity:
            raise ValueError(f"准入检查尚未通过或身份不一致: {name}")
        if name == "runtime_resources" and report.get("num_envs") != 4096:
            raise ValueError("资源证据必须来自4096环境的实际测量")
        verified[name] = {"path": str(report_path), "sha256": item["sha256"]}
    return {"schema_version": "IndependentStandingPreflightV1", "status": "passed",
            "identity": identity, "coverage": coverage, "verified_checks": verified}


def current_training_resources() -> dict:
    import csv
    from datetime import datetime, timezone
    import io
    import subprocess

    memory_command = ("Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory | "
                      "Select-Object AvailableMBytes,CommittedBytes,CommitLimit | ConvertTo-Json -Compress")
    memory = json.loads(subprocess.run(["pwsh", "-NoProfile", "-Command", memory_command],
        capture_output=True, text=True, check=True).stdout)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    rows = list(csv.reader(io.StringIO(gpu.stdout)))
    device = next(row for row in rows if int(row[0]) == 0)
    commit_free = int(memory["CommitLimit"]) - int(memory["CommittedBytes"])
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "commit_free_bytes": commit_free,
              "physical_free_mib": int(memory["AvailableMBytes"]), "gpu0_free_mib": int(device[1]),
              "gpu0_total_mib": int(device[2]), "other_processes_modified": False}
    report["sufficient"] = (commit_free >= 14 * 1024 ** 3 and report["physical_free_mib"] >= 6144
                            and report["gpu0_free_mib"] >= 4096)
    return report
