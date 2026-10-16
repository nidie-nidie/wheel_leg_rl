from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from test_phase1_math import _state
from test_dreamwaq_checkpoint import _write_checkpoint
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import CommandRanges, command_contract_payload
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import (
    RewardWeights, SplitSmoothRewardWeights, compute_reward, compute_reward_terms, reward_schema_version)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import (
    SPLIT_SMOOTH_PROFILE_WEIGHTS, apply_task_profile, task_profile_from_contract, resolve_task_profile)
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import stable_contract_hash
from wheelleg_dreamwaq.schemas.manifest import ContractMismatchError
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import validate_dreamwaq_checkpoint_metadata


def profile(name):
    cfg = SimpleNamespace(commands=CommandRanges(), reward_weights=RewardWeights())
    apply_task_profile(cfg, name)
    contract = {"schemas": {"reward": reward_schema_version(cfg.reward_weights)}, "task": {
        "commands": command_contract_payload(cfg.commands), "reward_weights": asdict(cfg.reward_weights)}}
    return cfg, contract


@pytest.mark.parametrize("joint", range(6))
def test_split_isolates_all_six_executed_action_coordinates(joint):
    state = _state(1)
    state.last_applied_action[0, joint] = .6
    state.previous_applied_action[0, joint] = -.2
    state.previous_previous_applied_action[0, joint] = .1
    terms = compute_reward_terms(state, SplitSmoothRewardWeights(), torch.zeros(1, dtype=torch.bool))
    assert "action_smooth" not in terms
    assert terms["action_smooth_leg"].item() == pytest.approx(1.21 if joint < 4 else 0.)
    assert terms["action_smooth_wheel"].item() == pytest.approx(0. if joint < 4 else 1.21)


def test_equal_weights_recover_legacy_and_doubling_only_changes_leg_contribution():
    state = _state(3)
    state.last_applied_action[:] = torch.tensor([.2, -.5, .3, .1, -.7, .4])
    state.previous_applied_action[:] = -.15
    state.previous_previous_applied_action[:] = .05
    done = torch.zeros(3, dtype=torch.bool)
    old, old_terms = compute_reward(state, RewardWeights(), control_dt=.02, terminated=done)
    equal, equal_terms = compute_reward(state, SplitSmoothRewardWeights(action_smooth_leg=-.01),
                                       control_dt=.02, terminated=done)
    new, new_terms = compute_reward(state, SplitSmoothRewardWeights(), control_dt=.02, terminated=done)
    assert torch.allclose(equal, old, atol=1e-7)
    assert torch.allclose(equal_terms["action_smooth_leg"] + equal_terms["action_smooth_wheel"],
                          old_terms["action_smooth"])
    assert torch.allclose(new, equal + equal_terms["action_smooth_leg"], atol=1e-7)
    assert torch.equal(new_terms["action_smooth_wheel"], equal_terms["action_smooth_wheel"])
    assert torch.allclose(new, torch.stack(list(new_terms.values())).sum(0), atol=1e-7)
    reset_terms = compute_reward_terms(_state(3), SplitSmoothRewardWeights(), done)
    assert not reset_terms["action_smooth_leg"].any() and not reset_terms["action_smooth_wheel"].any()


@pytest.mark.parametrize("name", SPLIT_SMOOTH_PROFILE_WEIGHTS)
def test_profile_roundtrip_and_switch_back_removes_subclass_fields(name):
    cfg, saved = profile(name)
    assert task_profile_from_contract(deepcopy(saved)) == name
    assert not cfg.leg_reversal.enabled and not cfg.leg_acceleration.enabled
    expected, old_contract = profile("stop_reverse_v1")
    for key, value in old_contract["task"]["reward_weights"].items():
        if key != "action_smooth":
            assert saved["task"]["reward_weights"][key] == value
    apply_task_profile(cfg, "stop_reverse_v1")
    assert type(cfg.reward_weights) is RewardWeights
    assert cfg.reward_weights == expected.reward_weights
    assert asdict(cfg.reward_weights) == old_contract["task"]["reward_weights"]
    assert reward_schema_version(cfg.reward_weights) == "RewardSchemaV2"
    with pytest.raises(ValueError, match="differs"):
        resolve_task_profile("stop_reverse_v1", saved)
    with pytest.raises(ValueError, match="differs"):
        resolve_task_profile(name, old_contract)


@pytest.mark.parametrize("mutation", ("schema", "missing", "wheel", "leg", "composite", "other",
                                      "schedule", "held", "acceleration", "reversal"))
def test_saved_split_identity_rejects_incomplete_or_conflicting_contract(mutation):
    _, saved = profile("stop_reverse_split_smooth_v1")
    if mutation == "schema":
        saved["schemas"]["reward"] = "RewardSchemaV2"
    elif mutation == "missing":
        saved["task"]["reward_weights"].pop("action_smooth_wheel")
    elif mutation in ("wheel", "leg", "composite", "other"):
        key = {"wheel": "action_smooth_wheel", "leg": "action_smooth_leg",
               "composite": "action_smooth", "other": "dof_acc"}[mutation]
        saved["task"]["reward_weights"][key] = -.123
    elif mutation == "schedule":
        saved["task"]["commands"]["practice_schedule"] = "disabled"
    elif mutation == "held":
        saved["task"]["commands"]["hold_for_episode"] = True
    else:
        saved["task"]["leg_" + mutation] = {}
    with pytest.raises(ValueError, match="split smooth"):
        task_profile_from_contract(saved)


@pytest.mark.parametrize("kwargs", ({"action_smooth": -.01}, {"action_smooth_leg": 0.},
                                    {"action_smooth_wheel": float("nan")}, {"action_smooth_leg": float("inf")}))
def test_invalid_split_weights_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SplitSmoothRewardWeights(**kwargs)


def test_checkpoint_validator_preserves_old_contract_and_rejects_reward_change(tmp_path):
    checkpoint, manifest, contracts, _ = _write_checkpoint(tmp_path)
    base, algorithm, export = contracts
    validate_dreamwaq_checkpoint_metadata(checkpoint, manifest, base, algorithm, export)
    changed = deepcopy(base)
    changed["schemas"]["reward"] = "RewardSchemaV3SplitSmooth"
    changed["task"] = profile("stop_reverse_split_smooth_v1")[1]["task"]
    changed.pop("contract_hash")
    changed["contract_hash"] = stable_contract_hash(changed)
    with pytest.raises(ContractMismatchError, match="runtime|contract"):
        validate_dreamwaq_checkpoint_metadata(checkpoint, manifest, changed, algorithm, export)
