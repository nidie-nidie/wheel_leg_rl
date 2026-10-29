from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wheelleg_mujoco.control import ActionTargets, MixedActionController
from wheelleg_mujoco.contract import AdapterContract
from wheelleg_mujoco.runner import WheelLegMujocoRuntime

from debug.no_action_clip.contract import (
    ACTION_CONTRACT, EXPERIMENT_ID, GEOMETRY_GUARD_CONTRACT, PROJECT_ROOT, sha256,
)
from debug.no_action_clip.rewards import (
    SPLIT_STRAIGHT_EXPERIMENT_ID, validate_reward_manifest,
)


COMPATIBILITY_RECORD_SHA256 = "1F4CD984A204A3E007A0DF2973AC62615790F633FFE46084771B95906F3EE42D"


def _verified_runtime_compatibility(export_dir: Path, path: Path, expected: str, actual: str) -> bool:
    record_path = Path(__file__).with_name("baseline_runtime_compatibility.json")
    if sha256(record_path) != COMPATIBILITY_RECORD_SHA256:
        raise ValueError("MuJoCo 兼容记录自身指纹不符")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record["schema_version"] != "BaselineRuntimeCompatibilityV1" or record["exact_equal"] is not True
            or any(v != 0.0 for result in record["max_abs_difference"].values() for v in result.values())):
        raise ValueError("MuJoCo 兼容记录没有通过精确轨迹比较")
    verified_export = record["after_identity"]["debug701"]
    if export_dir.resolve() != Path(verified_export["export"]).resolve():
        return False
    if any(sha256(export_dir / filename) != value for filename, value in verified_export["files"].items()):
        return False
    relative = str(path.relative_to(PROJECT_ROOT))
    return (record["before_identity"]["runtime_files"].get(relative) == expected
            and record["after_identity"]["runtime_files"].get(relative) == actual)


class RawActionController(MixedActionController):
    def prepare(self, action):
        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError("MuJoCo 原始动作必须为有限六维向量")
        legs = np.clip(self.contract.q_nominal + self.contract.leg_action_scale * raw[:4],
                       self.contract.leg_target_lower, self.contract.leg_target_upper)
        wheels = self.contract.wheel_action_scale * raw[4:6] * self.contract.wheel_signs
        if not np.isfinite(legs).all() or not np.isfinite(wheels).all():
            raise FloatingPointError("MuJoCo 动作目标溢出")
        return ActionTargets(raw.copy(), legs, wheels)


class RawActionRuntime(WheelLegMujocoRuntime):
    def __init__(self, model_path, contract, model_manifest):
        super().__init__(model_path, contract, model_manifest)
        self.controller = RawActionController(self.model_map, contract)

    def _append_policy_observation(self, current):
        # 物理观测归一化复用正式实现；上一动作按实验契约保留原始值。
        current[19:25] = self.previous_action.astype(np.float32)
        if not np.isfinite(current).all():
            raise FloatingPointError("MuJoCo 实验观测出现非有限数值")
        return super()._append_policy_observation(current)

    def step(self, action, command):
        result = super().step(action, command)
        result.metrics.pop("action_saturation_fraction")
        result.metrics["action_outside_former_unit_interval_fraction"] = float(np.mean(np.abs(action) > 1.0))
        return result


def load_runtime(export_dir: Path):
    metadata = json.loads((export_dir / "experiment_export.json").read_text(encoding="utf-8"))
    if (metadata["experiment_id"] not in (EXPERIMENT_ID, SPLIT_STRAIGHT_EXPERIMENT_ID)
            or metadata["action_contract"] != ACTION_CONTRACT
            or metadata.get("geometry_guard") != GEOMETRY_GUARD_CONTRACT):
        raise ValueError("MuJoCo 实验身份不符")
    for filename, field in (("actor.ts", "actor_sha256"), ("encoder.ts", "encoder_sha256"),
                            ("golden_vectors.pt", "golden_sha256")):
        if sha256(export_dir / filename) != metadata[field]:
            raise ValueError(f"MuJoCo 实验导出文件指纹不符：{filename}")
    checkpoint_path = Path(metadata["source_checkpoint"])
    if sha256(checkpoint_path) != metadata["source_checkpoint_sha256"]:
        raise ValueError("MuJoCo 实验源 checkpoint 指纹不符")
    training_manifest_path = checkpoint_path.parent / "manifest.json"
    if sha256(training_manifest_path) != metadata["training_manifest_sha256"]:
        raise ValueError("MuJoCo 实验训练 manifest 指纹不符")
    training = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    validate_reward_manifest(training)
    if (training["experiment_id"] != metadata["experiment_id"] or training["action_contract"] != ACTION_CONTRACT
            or training.get("reward_contract") != metadata.get("reward_contract")
            or ("reward_contract" in training
                and metadata.get("training_reward_weights") != training["reward_weights"])
            or training.get("geometry_guard") != GEOMETRY_GUARD_CONTRACT
            or training["baseline_export"] != metadata["baseline_export"]
            or not 0 < metadata["completed_iterations"] <= training["max_iterations"]):
        raise ValueError("MuJoCo 实验训练身份、参考物理契约或轮次不符")
    frozen_path = checkpoint_path.parent / "formal_files_before.json"
    if sha256(frozen_path) != training["formal_files_before_sha256"]:
        raise ValueError("MuJoCo 实验训练时正式文件指纹清单不符")
    frozen = {str(Path(record["path"]).resolve()): record["sha256"]
              for record in json.loads(frozen_path.read_text(encoding="utf-8-sig"))}
    baseline_dir = Path(metadata["baseline_export"]["directory"])
    baseline_path = baseline_dir / "policy_manifest.json"
    if sha256(baseline_path) != metadata["baseline_export"]["manifest_sha256"]:
        raise ValueError("MuJoCo 参考物理契约指纹不符")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    contract = AdapterContract.from_policy_manifest(baseline)
    model_manifest_path = PROJECT_ROOT / "sim2sim/mujoco/model_manifest.json"
    adapter_root = PROJECT_ROOT / "sim2sim/mujoco/wheelleg_mujoco"
    for path in [model_manifest_path, *adapter_root.glob("*.py")]:
        expected, actual = frozen.get(str(path.resolve())), sha256(path)
        if expected != actual and not _verified_runtime_compatibility(export_dir, path, expected, actual):
            raise ValueError(f"MuJoCo 当前正式文件与训练时身份不同：{path}")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    model_path = PROJECT_ROOT / "sim2sim/mujoco/models" / model_manifest["model_xml"]["path"]
    if (sha256(model_path) != model_manifest["model_xml"]["sha256"]
            or frozen.get(str(model_path.resolve())) != sha256(model_path)):
        raise ValueError("MuJoCo 模型指纹不符")
    return RawActionRuntime(model_path, contract, model_manifest)
