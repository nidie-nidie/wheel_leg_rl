from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .contracts import file_sha256, stable_hash, write_json
from .pose_bank import PoseBank


TRAIN_SOURCES = ("front", "back", "left", "right", "supine", "standing", "perturbed")
SOURCE_PROBABILITIES = (.12, .12, .12, .12, .12, .20, .20)


class IndependentPoseBank:
    def __init__(self, records, metadata):
        self.records, self.metadata = list(records), dict(metadata)
        self.by_id = {record["record_id"]: record for record in self.records}
        if len(self.by_id) != len(self.records) or not self.records:
            raise ValueError("独立站立初态身份重复或数据为空")
        trajectory_split = {}
        for record in self.records:
            if record["source"] not in (*TRAIN_SOURCES, "near_upright") or record["split"] not in ("train", "test"):
                raise ValueError("独立初态来源与划分无效")
            old = trajectory_split.setdefault(record["trajectory_id"], record["split"])
            if old != record["split"]:
                raise ValueError("同一物理轨迹不能跨训练与测试")
            n = len(record["joint_names"])
            if n == 0 or n != len(set(record["joint_names"])):
                raise ValueError("独立初态需要具名完整关节")
            for key, shape in (("joint_position", (n,)), ("joint_velocity", (n,)),
                               ("root_state_usd", (13,)), ("bounded_action", (6,))):
                value = np.asarray(record[key])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"独立初态物理字段无效: {key}")
            root = np.asarray(record["root_state_usd"])
            if not np.isclose(np.linalg.norm(root[3:7]), 1., atol=1e-5):
                raise ValueError("独立初态四元数无效")
            action = np.asarray(record["bounded_action"])
            if np.any(np.abs(action) > 1.) or np.any(action[4:] != 0.):
                raise ValueError("独立初态使用有界腿目标和零轮目标")
            for key in ("asset_bundle_sha256", "contact_model_sha256"):
                if len(record[key]) != 64:
                    raise ValueError("独立初态缺少物理身份")
            if not record["metadata"]["physical_validation_passed"]:
                raise ValueError("独立初态缺少真实物理核验")
            if record["source"] == "standing" and not record["metadata"].get("standing_gate_passed"):
                raise ValueError("稳定站立初态需要真实连续门控证据")
        self.sha256 = stable_hash({"schema_version": "IndependentStandingPoseBankV1",
                                  "metadata": self.metadata, "records": self.records})

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["schema_version"] != "IndependentStandingPoseBankV1":
            raise ValueError("独立站立初态库契约无效")
        return cls(payload["records"], payload["metadata"])

    def split_ids(self):
        return {split: {source: [r["record_id"] for r in self.records
                                if r["split"] == split and r["source"] == source]
                        for source in TRAIN_SOURCES} for split in ("train", "test")}

    def validate_split_ids(self, initialization):
        selected, trajectory_sets = {}, {}
        for split in ("train", "test"):
            groups = initialization[f"{split}_ids"]
            if set(groups) != set(TRAIN_SOURCES) or any(not values for values in groups.values()):
                raise ValueError("训练与测试初态分别需要完整七组来源")
            ids = [value for values in groups.values() for value in values]
            if len(ids) != len(set(ids)):
                raise ValueError("同一划分的初态ID重复")
            for source, values in groups.items():
                for value in values:
                    if value not in self.by_id or self.by_id[value]["split"] != split or self.by_id[value]["source"] != source:
                        raise ValueError("初态ID来源或划分不一致")
            selected[split] = set(ids)
            trajectory_sets[split] = {self.by_id[value]["trajectory_id"] for value in ids}
        if selected["train"] & selected["test"] or trajectory_sets["train"] & trajectory_sets["test"]:
            raise ValueError("初态及产生轨迹必须独立划分")
        if [initialization[name] for name in ("fallen_fraction", "standing_fraction", "perturbed_fraction")] != [.6, .2, .2]:
            raise ValueError("独立站立初态比例固定为60/20/20")
        return {"passed": True, "counts": {split: {source: len(values) for source, values in
                    initialization[f"{split}_ids"].items()} for split in ("train", "test")},
                "pose_bank_sha256": self.sha256}

    def sample(self, env_ids, randomization, initialization):
        records = []
        probabilities = torch.tensor(SOURCE_PROBABILITIES)
        for env_id in env_ids:
            generator = randomization.pose_generator(int(env_id))
            source = TRAIN_SOURCES[int(torch.multinomial(probabilities, 1, generator=generator))]
            candidates = initialization["train_ids"][source]
            record = self.by_id[candidates[int(torch.randint(len(candidates), (1,), generator=generator))]]
            if record["parameter_plan_sha256"] != randomization.plan(int(env_id)).sha256:
                raise ValueError("初态实际物理参数与训练环境不同")
            records.append(record)
        return records


class IndependentPoseTensorPool:
    def __init__(self, bank, joint_names, asset_sha256, device):
        self.record_indices = {r["record_id"]: index for index, r in enumerate(bank.records)}
        positions, velocities = [], []
        for record in bank.records:
            if set(record["joint_names"]) != set(joint_names) or record["asset_bundle_sha256"] != asset_sha256:
                raise ValueError("独立初态的完整关节或资产身份不同")
            order = [record["joint_names"].index(name) for name in joint_names]
            positions.append([record["joint_position"][i] for i in order])
            velocities.append([record["joint_velocity"][i] for i in order])
        self.joint_position = torch.tensor(positions, device=device)
        self.joint_velocity = torch.tensor(velocities, device=device)
        self.root_state = torch.tensor([r["root_state_usd"] for r in bank.records], device=device)
        self.actions = torch.tensor([r["bounded_action"] for r in bank.records], device=device)

    def indices(self, records):
        return torch.tensor([self.record_indices[r["record_id"]] for r in records],
                            dtype=torch.long, device=self.joint_position.device)


def import_fallen_records(path):
    bank = PoseBank.load(path)
    converted = []
    for old in bank.records:
        record = asdict(old)
        record["source"] = record["initial_category"]
        q = np.asarray(record["clean_history"])[-1, 6:10] + np.asarray(
            record["metadata"]["actual_parameters"]["q_reference"])
        if np.any(np.abs(q) > 1.):
            raise ValueError("倒地初态腿关节位置超出新动作映射，需要重新生成实际初态")
        record["bounded_action"] = np.r_[q, np.zeros(2)].tolist()
        record["metadata"]["q_control_rad"] = q.tolist()
        record["metadata"]["projected_gravity"] = list(record["clean_history"][-1][3:6])
        converted.append(record)
    return converted, {"source_fallen_bank_sha256": file_sha256(path), "source_fallen_bank_identity": bank.sha256}


def solve_passive_joints(model, data, controlled_joint_names):
    import mujoco
    from scipy.optimize import least_squares

    passive = [index for index in range(model.njnt)
               if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE
               and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) not in controlled_joint_names]
    addresses = np.array([model.jnt_qposadr[index] for index in passive], dtype=int)
    initial = data.qpos[addresses].copy()

    def residual(values):
        data.qpos[addresses] = values
        mujoco.mj_forward(model, data)
        selected = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        return np.r_[data.efc_pos[:data.nefc][selected], 1e-7 * (values - initial)]

    result = least_squares(residual, initial, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500)
    if not result.success or np.max(np.abs(residual(result.x)[:model.neq * 3])) > 1e-5:
        raise ValueError("近站立初态闭链从动关节求解失败")
    data.qpos[addresses] = result.x
    data.qvel[[model.jnt_dofadr[index] for index in passive]] = 0.
    mujoco.mj_forward(model, data)
    return {"success": True, "evaluations": result.nfev, "maximum_residual": float(np.max(np.abs(result.fun)))}


def solve_passive_velocity(model, data, controlled_joint_names):
    import mujoco

    addresses = np.array([model.jnt_dofadr[j] for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) not in controlled_joint_names], dtype=int)
    selected = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    jacobian = data.efc_J.reshape(data.nefc, model.nv)[selected]
    data.qvel[addresses] = 0.
    data.qvel[addresses] = np.linalg.lstsq(jacobian[:, addresses], -(jacobian @ data.qvel), rcond=None)[0]
    residual = float(np.max(np.abs(jacobian @ data.qvel)))
    if not np.isfinite(residual) or residual > 1e-5:
        raise ValueError("从动关节速度无法满足真实闭链速度约束")
    mujoco.mj_forward(model, data)
    return residual
