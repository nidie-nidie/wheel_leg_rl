from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ReferenceMatch:
    q: torch.Tensor
    height: torch.Tensor
    index: torch.Tensor
    distance: torch.Tensor


class StandingReference:
    def __init__(self, q, height, *, identity, arrays, device="cpu"):
        self.q = torch.as_tensor(q, dtype=torch.float32, device=device)
        self.height = torch.as_tensor(height, dtype=torch.float32, device=device)
        self.q_median = torch.quantile(self.q, 0.5, dim=0)
        self.identity = identity
        self.arrays = arrays

    @classmethod
    def load(cls, manifest_path, bank_path, expected_sha256, *, device="cpu",
             expected_manifest_sha256=None):
        manifest_path, bank_path = Path(manifest_path), Path(bank_path)
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest().upper()
        bank_hash = hashlib.sha256(bank_path.read_bytes()).hexdigest().upper()
        if expected_manifest_sha256 is not None and manifest_hash != expected_manifest_sha256.upper():
            raise ValueError("站立参考 manifest 身份不匹配")
        if len(expected_sha256) != 64 or bank_hash != expected_sha256.upper():
            raise ValueError("站立参考 NPZ 身份不匹配")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads((manifest_path.parent / "summary.json").read_text(encoding="utf-8"))
        if (manifest["schema"] != "StandingReference401V1" or manifest["source_completed_updates"] != 401
                or manifest["height_semantics"] != "base_body_COM_above_floor"
                or manifest["joint_order"] != ["jIJ", "jIO", "jAB", "jAG", "jwheel_left", "jwheel_right"]
                or manifest["control_signs"] != [1., 1., 1., 1., 1., -1.]
                or manifest["control_dt_s"] != .02 or manifest["policy_updates"] != 0
                or manifest["optimizer_updates"] != 0 or not summary["completed"]):
            raise ValueError("站立参考物理与策略契约不匹配")
        identity = manifest["identity"]
        if any(len(identity[key]) != 64 for key in ("checkpoint_sha256", "actor_sha256")):
            raise ValueError("站立参考缺少冻结策略身份")
        model = summary["models"]["recovery_contact"]
        if model["paired_reference_sha256"].upper() != bank_hash or model["paired_reference_samples"] != 560:
            raise ValueError("站立参考必须为已登记的 560 组 recovery_contact 配套状态")
        with np.load(bank_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        if any(len(value) != 560 or not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("站立参考字段数量或有限性无效")
        if arrays["q_control_rad"].shape != (560, 6) or arrays["height_m"].shape != (560,):
            raise ValueError("站立参考角度与高度形状无效")
        if not arrays["reference_candidate"].all() or np.any(arrays["nonwheel_normal_force_n"] > 1e-6):
            raise ValueError("站立参考包含未接受状态或非轮支撑")
        return cls(arrays["q_control_rad"][:, :4], arrays["height_m"],
                   identity={"manifest_sha256": manifest_hash, "bank_sha256": bank_hash,
                             "checkpoint_sha256": identity["checkpoint_sha256"],
                             "actor_sha256": identity["actor_sha256"],
                             "contact_manifest_sha256": manifest["recovery_contact_manifest_sha256"],
                             "contact_model_sha256": manifest["recovery_xml_sha256"],
                             "weight_n": model["weight_n"], "samples": 560}, arrays=arrays, device=device)

    def nearest(self, q, height):
        if q.ndim != 2 or q.shape[1] != 4 or height.shape != (q.shape[0],):
            raise ValueError("配套站姿查询必须提供 N×4 角度与 N 高度")
        if q.device != self.q.device or height.device != self.q.device:
            raise ValueError("配套站姿查询设备不同")
        if not torch.isfinite(q).all() or not torch.isfinite(height).all():
            raise FloatingPointError("配套站姿查询包含非有限数值")
        distance = ((q[:, None] - self.q[None]) / .10).square().mean(dim=-1)
        distance += ((height[:, None] - self.height[None]) / .02).square()
        minimum, index = distance.min(dim=1)
        return ReferenceMatch(self.q[index], self.height[index], index, minimum)
