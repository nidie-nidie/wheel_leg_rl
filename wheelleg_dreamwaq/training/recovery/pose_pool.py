from __future__ import annotations

import torch

from .pose_bank import PoseBank


class PoseTensorPool:
    def __init__(self, bank: PoseBank, joint_names: list[str], asset_sha256: str, device) -> None:
        self.record_indices = {record.record_id: index for index, record in enumerate(bank.records)}
        positions, velocities = [], []
        for record in bank.records:
            if record.asset_bundle_sha256 != asset_sha256 or set(record.joint_names) != set(joint_names):
                raise ValueError("姿态池的资产身份或完整关节集合不同")
            order = [record.joint_names.index(name) for name in joint_names]
            positions.append([record.joint_position[column] for column in order])
            velocities.append([record.joint_velocity[column] for column in order])
        tensor = lambda values: torch.tensor(values, dtype=torch.float32, device=device)
        self.joint_position = tensor(positions)
        self.joint_velocity = tensor(velocities)
        self.root_state = tensor([record.root_state_usd for record in bank.records])
        self.history = tensor([record.clean_history for record in bank.records])
        self.action = tensor([record.prior_actions[-1] for record in bank.records])
        self.previous_action = tensor([record.prior_actions[-2] for record in bank.records])
        self.prior_count = torch.tensor([int(record.metadata.get("valid_prior_action_count", len(record.prior_actions)))
                                         for record in bank.records], dtype=torch.int64, device=device)
        self.motor_targets = tensor([record.motor_targets for record in bank.records])

    def indices(self, records) -> torch.Tensor:
        return torch.tensor([self.record_indices[record.record_id] for record in records],
                            dtype=torch.int64, device=self.joint_position.device)
