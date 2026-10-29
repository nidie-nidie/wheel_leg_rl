from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import CATEGORIES, stable_hash, write_json


@dataclass(frozen=True)
class PoseRecord:
    record_id: str
    trajectory_id: str
    split: str
    initial_category: str
    motion_kind: str
    source_simulator: str
    joint_names: tuple[str, ...]
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    root_state_usd: tuple[float, ...]
    clean_history: tuple[tuple[float, ...], ...]
    prior_actions: tuple[tuple[float, ...], ...]
    motor_targets: tuple[float, ...]
    parameter_plan_sha256: str
    asset_bundle_sha256: str
    contact_model_sha256: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        stable_hash(asdict(self))
        if not self.record_id or not self.trajectory_id or self.split not in ("train", "test"):
            raise ValueError("姿态记录身份或数据集划分无效")
        if self.initial_category not in CATEGORIES or self.motion_kind not in ("static", "dynamic"):
            raise ValueError("姿态类别或运动类别无效")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("姿态记录必须按名称登记全部关节")
        if len(self.joint_position) != len(self.joint_names) or len(self.joint_velocity) != len(self.joint_names):
            raise ValueError("姿态关节数据不完整")
        root = np.asarray(self.root_state_usd, dtype=np.float64)
        if root.shape != (13,) or not np.isfinite(root).all() or abs(np.linalg.norm(root[3:7]) - 1) > 1e-5:
            raise ValueError("姿态根状态或WXYZ四元数无效")
        if np.asarray(self.clean_history).shape != (5, 22):
            raise ValueError("姿态必须保存5帧物理观测历史")
        actions = np.asarray(self.prior_actions)
        if actions.ndim != 2 or actions.shape[1] != 6 or len(actions) < 2 or len(self.motor_targets) != 6:
            raise ValueError("姿态必须保存两阶真实动作及六维电机目标")
        for value in (self.joint_position, self.joint_velocity, self.clean_history, self.prior_actions, self.motor_targets):
            if not np.isfinite(np.asarray(value)).all():
                raise ValueError("姿态记录包含非有限数值")
        if self.motion_kind == "dynamic" and self.metadata.get("history_source") != "physical_trajectory":
            raise ValueError("动态姿态必须来自真实物理轨迹历史")
        gravity = np.asarray(self.clean_history)[:, 3:6]
        if not np.allclose(np.linalg.norm(gravity, axis=-1), 1, atol=1e-4):
            raise ValueError("姿态历史重力必须为单位向量")
        if not np.allclose(np.asarray(self.clean_history)[-1, 16:22], actions[-1], atol=1e-6):
            raise ValueError("当前帧上一动作与真实前序动作不一致")
        if any(len(value) != 64 for value in (self.parameter_plan_sha256, self.asset_bundle_sha256,
                                             self.contact_model_sha256)):
            raise ValueError("姿态缺少参数计划或模型身份")


def pose_record_physical_sha256(record: PoseRecord) -> str:
    joints = sorted(zip(record.joint_names, record.joint_position, record.joint_velocity))
    return stable_hash({"root_state_usd": record.root_state_usd, "joints": joints,
                        "clean_history": record.clean_history, "prior_actions": record.prior_actions,
                        "motor_targets": record.motor_targets})


def source_pose_records(bank: PoseBank) -> list[dict[str, str]]:
    return [{"record_id": record.record_id, "trajectory_id": record.trajectory_id,
             "split": record.split, "physical_state_sha256": pose_record_physical_sha256(record)}
            for record in bank.records]


class PoseBank:
    def __init__(self, records: list[PoseRecord], metadata: dict[str, Any]) -> None:
        if not records:
            raise ValueError("姿态库不能为空")
        self.records, self.metadata = list(records), dict(metadata)
        self.validate_coverage(formal=False)
        self._sampling_cache = {}

    def payload(self) -> dict[str, Any]:
        return {"schema_version": "RecoveryPoseBankV1", "metadata": self.metadata,
                "records": [asdict(record) for record in self.records]}

    @property
    def sha256(self) -> str:
        return stable_hash(self.payload())

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError("姿态库版本必须写入新文件，不能覆盖既有数据")
        write_json(destination, self.payload())

    @classmethod
    def load(cls, path: str | Path) -> PoseBank:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(payload) != {"schema_version", "metadata", "records"} or payload["schema_version"] != "RecoveryPoseBankV1":
            raise ValueError("姿态库格式不符合契约")
        return cls([PoseRecord(**record) for record in payload["records"]], payload["metadata"])

    def sample(self, env_id: int, parameter_plan_sha256: str, generator: torch.Generator,
               split: str = "train", category: str | None = None) -> PoseRecord:
        if env_id < 0 or split not in ("train", "test") or (category is not None and category not in CATEGORIES):
            raise ValueError("姿态采样条件无效")
        key = (parameter_plan_sha256, split, category)
        if key not in self._sampling_cache:
            eligible = [record for record in self.records if record.split == split
                        and record.parameter_plan_sha256 == parameter_plan_sha256
                        and (category is None or record.initial_category == category)]
            if not eligible:
                raise ValueError(f"没有匹配实际物理参数计划的姿态: env={env_id}, split={split}, category={category}")
            categories = [name for name in CATEGORIES if any(record.initial_category == name for record in eligible)]
            weights = torch.tensor([0.10 if name == "near_upright" else 0.18 for name in categories])
            candidates = {name: [record for record in eligible if record.initial_category == name] for name in categories}
            self._sampling_cache[key] = (categories, weights, candidates)
        categories, weights, by_category = self._sampling_cache[key]
        selected = categories[int(torch.multinomial(weights, 1, generator=generator))]
        candidates = by_category[selected]
        return candidates[int(torch.randint(len(candidates), (1,), generator=generator))]

    def validate_coverage(self, formal: bool = False) -> dict[str, Any]:
        ids = [record.record_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("姿态记录身份重复")
        trajectory_splits: dict[str, str] = {}
        for record in self.records:
            previous = trajectory_splits.setdefault(record.trajectory_id, record.split)
            if previous != record.split:
                raise ValueError("同一产生轨迹不能跨训练和测试数据集")
        coverage = {name: sum(record.initial_category == name for record in self.records) for name in CATEGORIES}
        fallen = [record for record in self.records if record.initial_category != "near_upright"]
        walker_falls = sum(record.metadata.get("origin") == "frozen_walker_fall" for record in fallen)
        report = {"category_counts": coverage, "records": len(ids), "trajectories": len(trajectory_splits),
                  "walker_fall_count": walker_falls, "fallen_count": len(fallen)}
        if formal:
            if any(count == 0 for count in coverage.values()) or not fallen or walker_falls / len(fallen) < 0.30:
                raise ValueError("正式姿态库缺少类别或冻结行走真实倒地样本不足30%")
            for plan in {record.parameter_plan_sha256 for record in self.records if record.split == "train"}:
                subset = [record for record in self.records if record.parameter_plan_sha256 == plan and record.split == "train"]
                if {record.initial_category for record in subset} != set(CATEGORIES):
                    raise ValueError("正式训练每个参数计划必须覆盖全部初态类别")
                for category in CATEGORIES:
                    if {record.motion_kind for record in subset if record.initial_category == category} != {"static", "dynamic"}:
                        raise ValueError("正式训练每个参数计划的各类别必须覆盖静态和动态姿态")
            train_plans = {record.parameter_plan_sha256 for record in self.records if record.split == "train"}
            test_plans = {record.parameter_plan_sha256 for record in self.records if record.split == "test"}
            if train_plans & test_plans:
                raise ValueError("正式评估必须使用独立的实际物理参数计划")
            if not self.metadata.get("classification_sha256") or not self.metadata.get("curriculum_sha256"):
                raise ValueError("正式姿态库缺少冻结的分类或课程身份")
        return report
