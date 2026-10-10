from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from test_isaac_evaluation import _base_contract, _cache_payload
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import stable_contract_hash
from wheelleg_dreamwaq.schemas.isaac_evaluation import (
    PhysicalTrackingMAE, build_current_isaac_evaluation_contract,
    build_evaluation_source_fingerprint, build_evaluation_reset_cache_identity,
    validate_matching_isaac_evaluation_contracts, speed_tracking_target_failures, FORMAL_SCENARIOS,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import CommandRanges, command_contract_payload
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RewardWeights


def test_physical_tracking_counts_active_pre_action_units_without_reset_aliases():
    tracking = PhysicalTrackingMAE(2)
    physical = torch.tensor([[10.0, 0.4], [99.0, 99.0]])
    commands = torch.zeros(2, 3)
    tracking.update(physical, commands, torch.tensor([True, False]))
    # step() may replace these tensors with reset values; previous samples remain intact.
    physical[:] = 999.0
    tracking.update(torch.tensor([[1.0, -0.2], [2.0, 0.3]]), commands,
                    torch.tensor([True, True]))
    assert tracking.result(0)["vx_mae"] == pytest.approx(5.5)
    assert tracking.result(0)["yaw_rate_mae"] == pytest.approx(0.3)
    assert tracking.result(0)["tracking_sample_frames"] == 2
    assert tracking.result(1)["vx_mae"] == 2.0
    assert tracking.result(1)["tracking_sample_frames"] == 1


def test_current_isaac_contract_is_explicit_and_cannot_mix_with_v1():
    base = _base_contract()
    base["schemas"] = {"physics": "PhysicsV5"}
    base["contract_hash"] = stable_contract_hash({k: v for k, v in base.items() if k != "contract_hash"})
    arguments = dict(
        base_task_contract=base,
        evaluation_source=build_evaluation_source_fingerprint({"evaluation.py": "B" * 64}),
        reset_cache_identity=build_evaluation_reset_cache_identity(
            path="cache.pt", file_sha256="C" * 64, cache_payload=_cache_payload()),
        evaluation_reward_weights=asdict(RewardWeights()),
        evaluation_commands=command_contract_payload(CommandRanges()),
    )
    contract = build_current_isaac_evaluation_contract(**arguments)
    assert contract["manifest_version"] == "IsaacEvaluationContractV2"
    assert contract["baseline"] is None
    assert contract["baseline_status"] == "not_comparable"
    assert contract["physical_tracking"]["sample_timing"] == "active_pre_action_frame"
    assert contract["evaluation_reward_weights"]["tracking_vx"] == 1.0
    with pytest.raises(ValueError, match="version"):
        validate_matching_isaac_evaluation_contracts(contract, contract)
    wrong = deepcopy(arguments)
    wrong["evaluation_reward_weights"]["tracking_vx"] = 2.0
    with pytest.raises(ValueError, match="scoring"):
        build_current_isaac_evaluation_contract(**wrong)


def test_speed_target_gate_cannot_accept_an_incomplete_low_error_prefix():
    aggregate = {"scenarios": [{"name": name, "completed": True, "vx_mae": 0.0}
                              for name, _ in FORMAL_SCENARIOS]}
    assert speed_tracking_target_failures(aggregate) == []
    aggregate["scenarios"][0]["completed"] = False
    aggregate["scenarios"][4]["vx_mae"] = 0.3
    assert speed_tracking_target_failures(aggregate) == [
        "nominal_stand:incomplete", "reverse:vx_mae_above_0.2"]
