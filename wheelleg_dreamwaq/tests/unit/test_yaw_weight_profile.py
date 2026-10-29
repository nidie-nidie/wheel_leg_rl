from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import torch

from test_phase1_math import _state
from test_split_action_smooth import profile
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import dreamwaq_contract_versions
from wheelleg_dreamwaq.schemas.heading_control import (
    RAW_BASELINE_NO_HEADING_PROFILE, RAW_BASELINE_PROFILE, RUNTIME_GEOMETRY, heading_contract,
)
from wheelleg_dreamwaq.schemas.normalization import RawActionNormalizationV1
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import command_contract_payload
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import ControlLimits
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import compute_reward, reward_schema_version
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import (
    RAW_BASELINE_YAW1_NO_HEADING_PROFILE, apply_task_profile, resolve_task_profile, task_profile_from_contract,
)


def saved_contract(cfg):
    return {
        "manifest_version": "WheelLegBaseTaskContractV2RawHeading",
        "schemas": {"reward": reward_schema_version(cfg.reward_weights), "action": "RawActionV1",
                    "normalization": RawActionNormalizationV1.schema_version},
        "action": {"clip_actions": None},
        "normalization": asdict(RawActionNormalizationV1()),
        "task": {"commands": command_contract_payload(cfg.commands),
                 "reward_weights": asdict(cfg.reward_weights), "heading_hold": heading_contract(cfg.heading_hold),
                 "runtime_geometry": dict(RUNTIME_GEOMETRY), "control": asdict(ControlLimits()), "control_dt": .02},
    }


def test_yaw_one_only_changes_two_weights_and_preserves_old_profiles():
    old, _ = profile(RAW_BASELINE_NO_HEADING_PROFILE)
    new, _ = profile(RAW_BASELINE_YAW1_NO_HEADING_PROFILE)
    expected = deepcopy(old.__dict__)
    expected["reward_weights"] = replace(old.reward_weights, tracking_yaw_rate=1., tracking_yaw_rate_enhance=1.)
    assert new.__dict__ == expected
    assert task_profile_from_contract(saved_contract(new)) == RAW_BASELINE_YAW1_NO_HEADING_PROFILE
    assert dreamwaq_contract_versions(saved_contract(new)) == (
        "WheelLegBaseTaskContractV2RawHeading", "DreamWaQAlgorithmContractV2RawActions", "DreamWaQExportContractV2RawHeading")
    assert resolve_task_profile(None, saved_contract(new)) == RAW_BASELINE_YAW1_NO_HEADING_PROFILE
    for name in (RAW_BASELINE_PROFILE, RAW_BASELINE_NO_HEADING_PROFILE):
        apply_task_profile(new, name)
        assert new.reward_weights.tracking_yaw_rate == new.reward_weights.tracking_yaw_rate_enhance == 2.
        assert task_profile_from_contract(saved_contract(new)) == name
        assert dreamwaq_contract_versions(saved_contract(new))[0] == "WheelLegBaseTaskContractV2RawHeading"
    with pytest.raises(ValueError, match="differs"):
        resolve_task_profile(RAW_BASELINE_NO_HEADING_PROFILE, saved_contract(profile(RAW_BASELINE_YAW1_NO_HEADING_PROFILE)[0]))


@pytest.mark.parametrize("case", ("heading_on", "one_weight_only", "unknown_weight", "missing_weight"))
def test_yaw_one_rejects_unknown_saved_combinations(case):
    cfg, _ = profile(RAW_BASELINE_YAW1_NO_HEADING_PROFILE)
    contract = saved_contract(cfg)
    if case == "heading_on":
        contract["task"]["heading_hold"]["enabled"] = True
    elif case == "missing_weight":
        del contract["task"]["reward_weights"]["tracking_yaw_rate"]
    else:
        contract["task"]["reward_weights"]["tracking_yaw_rate"] = 2. if case == "one_weight_only" else .5
    with pytest.raises(ValueError):
        task_profile_from_contract(contract)


def test_yaw_one_halves_both_actual_yaw_terms_without_changing_other_terms():
    old, _ = profile(RAW_BASELINE_NO_HEADING_PROFILE)
    new, _ = profile(RAW_BASELINE_YAW1_NO_HEADING_PROFILE)
    state = _state(1)
    state.root_angular_velocity[:, 2] = .7
    terminated = torch.zeros(1, dtype=torch.bool)
    _, before = compute_reward(state, old.reward_weights, control_dt=.02, terminated=terminated)
    _, after = compute_reward(state, new.reward_weights, control_dt=.02, terminated=terminated)
    assert before.keys() == after.keys()
    for name in before:
        expected = before[name] * .5 if name in ("tracking_yaw_rate", "tracking_yaw_rate_enhance") else before[name]
        torch.testing.assert_close(after[name], expected, rtol=0, atol=0)
