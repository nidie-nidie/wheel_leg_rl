from pathlib import Path

import pytest
import torch

from recovery.contracts import ARTIFACT_ROOT, CATEGORIES
from recovery.pose_bank import PoseBank
from recovery.pose_pool import PoseTensorPool


BANK = ARTIFACT_ROOT / "bootstrap_pose_bank_v1/pose_bank.json"


@pytest.mark.parametrize("category", [None, *CATEGORIES])
def test_cached_sampling_preserves_real_bank_sequence_and_rng(category):
    bank = PoseBank.load(BANK)
    plan = bank.records[0].parameter_plan_sha256
    actual_rng = torch.Generator().manual_seed(721)
    expected_rng = torch.Generator().manual_seed(721)
    eligible = [record for record in bank.records if record.split == "train"
                and record.parameter_plan_sha256 == plan and (category is None or record.initial_category == category)]
    categories = [name for name in CATEGORIES if any(record.initial_category == name for record in eligible)]
    weights = torch.tensor([0.10 if name == "near_upright" else 0.18 for name in categories])
    for _ in range(64):
        selected = categories[int(torch.multinomial(weights, 1, generator=expected_rng))]
        candidates = [record for record in eligible if record.initial_category == selected]
        expected = candidates[int(torch.randint(len(candidates), (1,), generator=expected_rng))]
        actual = bank.sample(0, plan, actual_rng, category=category)
        assert actual.record_id == expected.record_id
        assert torch.equal(actual_rng.get_state(), expected_rng.get_state())


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_real_bank_tensor_pool_preserves_full_reset_fields(device):
    bank = PoseBank.load(BANK)
    names = list(reversed(bank.records[0].joint_names))
    pool = PoseTensorPool(bank, names, bank.records[0].asset_bundle_sha256, device)
    selected = list(reversed(bank.records))
    indices = pool.indices(selected)
    for index, record in enumerate(selected):
        order = [record.joint_names.index(name) for name in names]
        for field, values in (("joint_position", [record.joint_position[column] for column in order]),
                              ("joint_velocity", [record.joint_velocity[column] for column in order]),
                              ("root_state", record.root_state_usd), ("history", record.clean_history),
                              ("action", record.prior_actions[-1]), ("previous_action", record.prior_actions[-2]),
                              ("motor_targets", record.motor_targets)):
            actual = getattr(pool, field)[indices[index]]
            expected = torch.tensor(values, dtype=torch.float32, device=device)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
