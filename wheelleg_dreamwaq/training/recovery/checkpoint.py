from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .contracts import GuidanceConfigV1, GuidanceTrainingContractV1, file_sha256, stable_hash


def capture_global_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
        "cuda_device_count": torch.cuda.device_count(),
    }


def restore_global_rng_state(state: dict[str, Any]) -> None:
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda", "cuda_device_count"}:
        raise ValueError("Global RNG state fields are incomplete")
    if state["cuda_device_count"] != torch.cuda.device_count():
        raise ValueError("Strict RNG restore requires the same CUDA device count")
    if len(state["torch_cuda"]) != state["cuda_device_count"]:
        raise ValueError("CUDA RNG states do not match the recorded device count")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def checkpoint_index_path(path: str | Path) -> Path:
    return Path(str(Path(path)) + ".index.json")


def save_checkpoint(path: str | Path, payload: dict, identity: dict) -> dict:
    destination = Path(path)
    index_path = checkpoint_index_path(destination)
    _validate_payload(payload)
    _validate_identity(identity)
    has_guidance = validate_guidance_checkpoint(payload, identity)
    validate_logging_barrier(payload["environment_state"])
    validate_geometry_barrier(payload["environment_state"])
    if destination.exists() or index_path.exists():
        raise FileExistsError("Checkpoint and index files must use a new path")
    envelope = {
        "schema_version": "RecoveryCheckpointV2" if has_guidance else "RecoveryCheckpointV1",
        "identity": deepcopy(identity),
        "payload": payload,
        "global_rng_state": capture_global_rng_state(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save(envelope, stream)
        stream.flush()
        os.fsync(stream.fileno())
    index = {
        "schema_version": "RecoveryCheckpointIndexV2" if has_guidance else "RecoveryCheckpointIndexV1",
        "checkpoint_file": destination.name,
        "checkpoint_sha256": file_sha256(destination),
        "identity_sha256": stable_hash(identity),
        "completed_updates": payload["completed_updates"],
    }
    with index_path.open("x", encoding="utf-8") as stream:
        json.dump(index, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return index


def load_checkpoint(
    path: str | Path,
    expected_identity: dict,
    *,
    restore_rng: bool = True,
) -> dict:
    envelope = read_checkpoint_envelope(path, expected_identity)
    if restore_rng:
        restore_global_rng_state(envelope["global_rng_state"])
    return envelope["payload"]


def read_checkpoint_envelope(path: str | Path, expected_identity: dict) -> dict:
    source = Path(path)
    _validate_identity(expected_identity)
    index = json.loads(checkpoint_index_path(source).read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version", "checkpoint_file", "checkpoint_sha256", "identity_sha256", "completed_updates"
    }
    if (set(index) != expected_fields
            or index["schema_version"] not in ("RecoveryCheckpointIndexV1", "RecoveryCheckpointIndexV2")):
        raise ValueError("Unsupported or incomplete checkpoint index")
    if index["checkpoint_file"] != source.name or file_sha256(source) != index["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes disagree with the external index")
    if stable_hash(expected_identity) != index["identity_sha256"]:
        raise ValueError("Checkpoint identity does not exactly match the requested training identity")
    envelope = torch.load(source, map_location="cpu", weights_only=False)
    if set(envelope) != {"schema_version", "identity", "payload", "global_rng_state"}:
        raise ValueError("Checkpoint envelope fields are incomplete")
    if envelope["schema_version"] not in ("RecoveryCheckpointV1", "RecoveryCheckpointV2"):
        raise ValueError("Unsupported recovery checkpoint schema")
    if stable_hash(envelope["identity"]) != index["identity_sha256"] or envelope["identity"] != expected_identity:
        raise ValueError("Checkpoint identity disagrees with its index or expected identity")
    _validate_payload(envelope["payload"])
    has_guidance = validate_guidance_checkpoint(envelope["payload"], envelope["identity"])
    expected_schema = "RecoveryCheckpointV2" if has_guidance else "RecoveryCheckpointV1"
    expected_index = "RecoveryCheckpointIndexV2" if has_guidance else "RecoveryCheckpointIndexV1"
    if envelope["schema_version"] != expected_schema or index["schema_version"] != expected_index:
        raise ValueError("Checkpoint版本与辅助学习器字段不一致")
    validate_logging_barrier(envelope["payload"]["environment_state"])
    validate_geometry_barrier(envelope["payload"]["environment_state"])
    if envelope["payload"]["completed_updates"] != index["completed_updates"]:
        raise ValueError("Checkpoint update count disagrees with its index")
    return envelope


def validate_logging_barrier(environment_state: dict) -> None:
    if "logging_barrier" not in environment_state:
        return
    barrier = environment_state["logging_barrier"]
    if barrier.get("schema_version") != "RecoveryLoggingBarrierV1":
        raise ValueError("Checkpoint日志屏障版本无效")
    frames = environment_state["frame_ids"]
    frames = frames.detach().cpu().tolist() if isinstance(frames, torch.Tensor) else list(frames)
    if (barrier["pending_packets"] != 0 or barrier["submitted_packet_sequence"] != barrier["consumed_packet_sequence"]
            or barrier["frame_ids"] != frames or barrier["frame_ids_sha256"] != stable_hash(frames)):
        raise ValueError("Checkpoint模型环境帧与原始日志消费ACK不同")
    archive = environment_state["metrics_archive"]
    if barrier["archive_last_sequence"] != archive["last_sequence"]:
        raise ValueError("Checkpoint日志归档屏障与索引不同")
    states = environment_state["metrics"]
    if len(states) != len(frames) or barrier["stream_cutoffs"] != [state["record_cutoff_sequence"] for state in states]:
        raise ValueError("Checkpoint逐stream截止序号与日志屏障不同")
    if any(state["record_cutoff_sequence"] > archive["last_sequence"] for state in states):
        raise ValueError("Checkpoint统计越过已持久化归档序号")


def prepare_compatible_import(checkpoint_path: str | Path, expected_parent_identity: dict, new_identity: dict,
                              evidence_path: str | Path, manifest_path: str | Path, *, new_config: dict) -> dict:
    parent = Path(checkpoint_path).resolve()
    manifest_destination = Path(manifest_path)
    if manifest_destination.exists():
        raise FileExistsError("兼容导入清单必须使用新的路径")
    envelope = read_checkpoint_envelope(parent, expected_parent_identity)
    parent_config_path = parent.parent / "resolved_config.json"
    parent_manifest_path = parent.parent / "run_manifest.json"
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if stable_hash(parent_config) != expected_parent_identity["training_config_sha256"]:
        raise ValueError("旧配置SHA与已核验父运行身份不同")
    manifest_identity = parent_manifest.get("identity", parent_manifest.get("training_identity"))
    if manifest_identity != expected_parent_identity:
        raise ValueError("父运行清单身份与显式预期身份不同")
    if stable_hash(new_config) != new_identity["training_config_sha256"]:
        raise ValueError("新配置SHA与新运行身份不同")
    allowed = {("environment", "run_dir"), ("assessment", "updates"), ("ppo", "save_interval")}
    changes = []
    def compare(old, new, prefix=()):
        if isinstance(old, dict) and isinstance(new, dict):
            if set(old) != set(new):
                raise ValueError(f"兼容导入配置字段改变: {'.'.join(prefix)}")
            for key in sorted(old):
                compare(old[key], new[key], (*prefix, key))
        elif old != new:
            if prefix not in allowed:
                raise ValueError(f"兼容导入改变了训练物理或随机契约: {'.'.join(prefix)}")
            changes.append({"path": ".".join(prefix), "parent": old, "new": new})
    compare(parent_config, new_config)
    fixed_identity_keys = {"walker_actor_sha256", "walker_manifest_sha256", "asset_bundle_sha256",
        "mujoco_contact_model_sha256", "isaac_contact_model_sha256", "pose_bank_sha256", "source_pose_records_sha256",
        "parameter_plan_sha256", "num_envs", "device", "geometry_guard_sha256", "curriculum_sha256", "contact_model_sha256"}
    if not fixed_identity_keys.issubset(expected_parent_identity) or not fixed_identity_keys.issubset(new_identity):
        raise ValueError("兼容导入缺少完整固定物理身份")
    for key in fixed_identity_keys:
        if expected_parent_identity.get(key) != new_identity.get(key):
            raise ValueError(f"兼容导入固定身份改变: {key}")
    evidence_source = Path(evidence_path).resolve()
    evidence = json.loads(evidence_source.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != "RecoveryPerformanceEquivalenceEvidenceV1":
        raise ValueError("性能等价验收证据版本无效")
    if (evidence.get("source_sha256") != new_identity["source_sha256"]
            or evidence.get("parent_identity_sha256") != stable_hash(expected_parent_identity)
            or evidence.get("new_identity_sha256") != stable_hash(new_identity)):
        raise ValueError("兼容导入证据未绑定父身份和当前受验源码身份")
    required = {"geometry", "fsm", "logging", "interface4", "interface256"}
    if set(evidence.get("checks", {})) != required:
        raise ValueError("兼容导入缺少完整几何/状态机/日志/真实接口验收")
    checks = {}
    for key, check in evidence["checks"].items():
        if set(check) != {"passed", "path", "sha256"} or check["passed"] is not True:
            raise ValueError(f"兼容导入验收未通过: {key}")
        path = Path(check["path"])
        if not path.is_absolute():
            path = evidence_source.parent / path
        if file_sha256(path) != check["sha256"]:
            raise ValueError(f"兼容导入验收产物SHA改变: {key}")
        _validate_performance_report(key, path.resolve(), new_identity, new_config, fixed_identity_keys)
        checks[key] = {**check, "path": str(path.resolve())}
    from .metrics_archive import MetricsArchive

    environment = envelope["payload"]["environment_state"]
    MetricsArchive.verify_index(environment["metrics_archive"])
    source_states = environment["metrics"]
    if len(source_states) != expected_parent_identity["num_envs"] or any(
            state["events"] or state["attempts"] or state["record_cutoff_sequence"] > environment["metrics_archive"]["last_sequence"]
            for state in source_states):
        raise ValueError("父checkpoint统计必须是有截止序号的行政删失截面")
    manifest = {"schema_version": "RecoveryCompatibleImportV1", "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": file_sha256(parent), "parent_config_sha256": file_sha256(parent_config_path),
        "parent_run_manifest_sha256": file_sha256(parent_manifest_path), "parent_identity": expected_parent_identity,
        "new_identity": new_identity, "configuration_changes": changes,
        "equivalence_evidence": {"path": str(evidence_source), "sha256": file_sha256(evidence_source), "checks": checks},
        "completed_updates": envelope["payload"]["completed_updates"],
        "import_scope": ["model_state_dict", "optimizer_state_dict", "completed_updates", "learning_rate", "global_rng_state",
                         "environment_named_rng_states", "curriculum", "checkpoint_censored_metrics"],
        "statistics_origin": {"metrics_archive": environment["metrics_archive"],
                              "stream_cutoffs_sha256": stable_hash([state["record_cutoff_sequence"] for state in source_states]),
                              "selected_censor_sequences_sha256": stable_hash([state["checkpoint_censor_sequences"] for state in source_states])},
        "physical_episode_restore": "按父checkpoint已行政删失截面开始新物理回合；不继承截面之后的更新或尾部日志"}
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    with manifest_destination.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return envelope


def _validate_performance_report(key, path, identity, config, fixed_identity_keys):
    report = json.loads(path.read_text(encoding="utf-8"))
    source = report.get("source_sha256", report.get("identity", {}).get("source_sha256"))
    if report.get("passed") is not True or source != identity["source_sha256"]:
        raise ValueError(f"验收子报告未绑定当前受验源码或未通过: {key}")
    if key in {"geometry", "fsm"}:
        suites = report.get("suites", [])
        if (report.get("schema_version") != "RecoveryUnitVerificationV1"
                or {suite.get("name") for suite in suites} != {"main", "mujoco"}
                or any(suite.get("tests", 0) <= 0 or any(suite.get(field) != 0
                    for field in ("failures", "errors", "skipped")) for suite in suites)):
            raise ValueError("几何及状态机需要完整同源实际测试报告")
    elif key == "logging":
        component = report.get("component", {})
        statistics = component.get("stream_statistics", {})
        if (report.get("schema_version") != "RecoveryMetricsPipelineVerificationV1"
                or component.get("num_envs") != identity["num_envs"]
                or statistics.get("submitted_packets", 0) <= 0
                or statistics.get("submitted_packets") != statistics.get("acknowledged_packets")):
            raise ValueError("日志验收需要完整规模及全部消费ACK")
    else:
        expected_count = {"interface4": 4, "interface256": 256}[key]
        actual_identity = report.get("identity", {})
        actual_config = json.loads((path.parent / "resolved_config.json").read_text(encoding="utf-8"))
        manifest = json.loads((path.parent / "run_manifest.json").read_text(encoding="utf-8"))
        if (report.get("schema_version") != "RecoveryTrainingVerificationV1"
                or report.get("purpose") != "interface_validation"
                or report.get("parameters_changed") is not True
                or report.get("resume_and_continued_optimization_verified") is not True
                or actual_identity.get("num_envs") != expected_count
                or actual_config["environment"]["num_envs"] != expected_count
                or stable_hash(actual_config) != actual_identity.get("training_config_sha256")
                or actual_identity != manifest.get("identity") or actual_config != manifest.get("resolved_config")
                or stable_hash(manifest.get("execution")) != actual_identity.get("execution_implementation_sha256")):
            raise ValueError(f"真实接口报告身份、规模或恢复更新验收不完整: {key}")
        for field in fixed_identity_keys - {"num_envs", "parameter_plan_sha256"} | {"execution_implementation_sha256"}:
            if field not in actual_identity or actual_identity[field] != identity[field]:
                raise ValueError(f"接口与导入运行的固定物理身份不同: {key}.{field}")
        excluded = {("environment", "num_envs"), ("environment", "run_dir"), ("environment", "task_type"),
                    ("assessment", "updates"), ("ppo", "save_interval")}
        def compare(left, right, prefix=()):
            if isinstance(left, dict) and isinstance(right, dict):
                if set(left) != set(right):
                    raise ValueError("接口验收配置字段不同")
                for field in left:
                    compare(left[field], right[field], (*prefix, field))
            elif prefix not in excluded and left != right:
                raise ValueError(f"接口验收物理配置不同: {'.'.join(prefix)}")
        compare(actual_config, config)


def validate_geometry_barrier(environment_state: dict) -> None:
    barrier = environment_state.get("geometry_barrier")
    if barrier is None:
        return
    from .geometry_archive import verify_geometry_barrier

    verify_geometry_barrier(barrier)


def _validate_identity(identity: dict) -> None:
    if not isinstance(identity, dict) or not identity:
        raise ValueError("A nonempty complete identity is required")
    stable_hash(identity)


def _validate_payload(payload: dict) -> None:
    required = {
        "model_state_dict", "optimizer_state_dict", "completed_updates", "learning_rate", "environment_state"
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Checkpoint payload lacks model, optimizer, progress, or environment state")
    updates = payload["completed_updates"]
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("Checkpoint requires a positive completed update count")
    if not math.isfinite(payload["learning_rate"]) or payload["learning_rate"] <= 0:
        raise ValueError("Checkpoint learning rate must be finite and positive")
    for field in ("model_state_dict", "optimizer_state_dict", "environment_state"):
        if not isinstance(payload[field], dict):
            raise ValueError(f"{field} must be a dictionary")
    for value in payload["model_state_dict"].values():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise FloatingPointError("Checkpoint model contains nonfinite parameters")


def validate_guidance_checkpoint(payload: dict, identity: dict) -> bool:
    identity_fields = {"training_contract_sha256", "guidance_config_sha256"}
    has_guidance = "guidance" in payload
    indicated = bool(identity_fields.intersection(identity)) or "training_contract" in payload
    if not has_guidance and not indicated:
        return False
    if not has_guidance or not identity_fields.issubset(identity):
        raise ValueError("辅助checkpoint缺少完整学习器状态或训练身份")
    state = payload["guidance"]
    expected = {"schema_version", "cost_critic_state_dict", "cost_optimizer_state_dict",
                "global_control_steps", "alpha", "guidance_config", "training_contract"}
    if (not isinstance(state, dict) or set(state) != expected
            or state["schema_version"] != "RecoveryGuidanceLearnerStateV1"):
        raise ValueError("辅助学习器恢复字段不完整或版本无效")
    config = GuidanceConfigV1.from_dict(state["guidance_config"])
    contract = GuidanceTrainingContractV1.from_dict(state["training_contract"])
    if (identity["guidance_config_sha256"] != config.sha256
            or identity["training_contract_sha256"] != contract.sha256):
        raise ValueError("辅助checkpoint配置与训练契约身份不同")
    source = identity.get("source_sha256")
    if (not isinstance(source, str) or len(source) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in source)):
        raise ValueError("辅助checkpoint缺少受验源码身份")
    steps = state["global_control_steps"]
    if (type(steps) is not int or steps != payload["completed_updates"] * config.rollout_steps):
        raise ValueError("辅助控制步进度与已完成更新数量不同")
    expected_alpha = max(1.0 - steps / config.end_control_steps, 0.0) if config.force_enabled else 0.0
    alpha = state["alpha"]
    if (type(alpha) not in (int, float) or not math.isfinite(alpha)
            or not 0.0 <= alpha <= 1.0 or abs(alpha - expected_alpha) > 1e-7):
        raise ValueError("辅助alpha与更新边界进度不同")
    network = state["cost_critic_state_dict"]
    optimizer = state["cost_optimizer_state_dict"]
    if not isinstance(network, dict) or not network or not isinstance(optimizer, dict):
        raise ValueError("辅助成本网络及优化器必须完整保存")
    if (set(optimizer) != {"state", "param_groups"} or not isinstance(optimizer["state"], dict)
            or not isinstance(optimizer["param_groups"], list) or not optimizer["param_groups"]):
        raise ValueError("辅助成本优化器状态不完整")
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
           or not torch.isfinite(value).all() for value in network.values()):
        raise ValueError("辅助成本网络参数必须为有限float32张量")
    cost_shapes = _mlp_parameter_shapes("network", contract.critic_input_dim, 1)
    model_shapes = {**_mlp_parameter_shapes("actor", contract.actor_input_dim, 6),
                    **_mlp_parameter_shapes("critic", contract.critic_input_dim, 1), "log_std": (6,)}
    for parameters, shapes in ((network, cost_shapes), (payload["model_state_dict"], model_shapes)):
        if set(parameters) != set(shapes) or any(
                not isinstance(parameters[name], torch.Tensor) or tuple(parameters[name].shape) != shape
                or parameters[name].dtype != torch.float32 for name, shape in shapes.items()):
            raise ValueError("辅助checkpoint网络字段、维度或类型不符合训练契约")
    for group in optimizer["param_groups"]:
        if (not isinstance(group, dict) or not group.get("params")
                or type(group.get("lr")) not in (int, float)
                or not math.isfinite(group["lr"]) or group["lr"] != config.cost_learning_rate):
            raise ValueError("辅助成本优化器参数组无效")
    _validate_finite_optimizer_state(optimizer["state"])
    parameter_ids = [parameter for group in optimizer["param_groups"] for parameter in group["params"]]
    if len(parameter_ids) != len(cost_shapes) or len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError("辅助成本优化器参数身份与成本网络不同")
    if not config.force_enabled and not optimizer["state"]:
        return True
    if set(optimizer["state"]) != set(parameter_ids):
        raise ValueError("辅助成本优化器缺少已更新参数的恢复状态")
    for parameter_id, shape in zip(parameter_ids, cost_shapes.values()):
        parameter_state = optimizer["state"][parameter_id]
        if (not isinstance(parameter_state, dict)
                or not {"step", "exp_avg", "exp_avg_sq"}.issubset(parameter_state)
                or set(parameter_state) - {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}):
            raise ValueError("辅助Adam恢复状态不完整")
        for name in set(parameter_state) - {"step"}:
            if (not isinstance(parameter_state[name], torch.Tensor)
                    or tuple(parameter_state[name].shape) != shape):
                raise ValueError("辅助Adam矩估计维度与成本网络不同")
        step = parameter_state["step"]
        if (not isinstance(step, torch.Tensor) or step.numel() != 1
                or not torch.isfinite(step).all() or float(step) <= 0 or float(step) % 1 != 0):
            raise ValueError("辅助Adam更新计数必须为正有限整数")
    return True


def _mlp_parameter_shapes(prefix: str, input_dim: int, output_dim: int) -> dict[str, tuple[int, ...]]:
    shapes = {}
    dimensions = (input_dim, 256, 128, 64, output_dim)
    for index, (previous, following) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        shapes[f"{prefix}.{index * 2}.weight"] = (following, previous)
        shapes[f"{prefix}.{index * 2}.bias"] = (following,)
    return shapes


def _validate_finite_optimizer_state(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("辅助成本优化器包含非有限状态")
    elif isinstance(value, dict):
        for item in value.values():
            _validate_finite_optimizer_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_finite_optimizer_state(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("辅助成本优化器包含非有限标量")
