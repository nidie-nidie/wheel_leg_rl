from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from test_dreamwaq_checkpoint import _write_checkpoint, _algorithm, _initialize_optimizer
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import stable_contract_hash
from wheelleg_dreamwaq.schemas.manifest import ContractMismatchError
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import CommandRanges, command_contract_payload
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RewardWeights
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.leg_acceleration import (
    LegAccelerationReward, acceleration_contract_payload,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import (
    ACCELERATION_PROFILE_FRACTIONS, apply_task_profile, task_profile_from_contract,
)
from wheelleg_dreamwaq.training.checkpoint import sha256_file
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import (
    DreamWaQContractOnPolicyRunner, validate_dreamwaq_checkpoint_metadata,
    resume_dreamwaq_runner_from_checkpoint,
)
from wheelleg_dreamwaq.training.weights_initialization import require_initialization_compatibility


def profile(name):
    cfg = SimpleNamespace(commands=CommandRanges(), reward_weights=RewardWeights())
    apply_task_profile(cfg, name)
    task = {"commands": command_contract_payload(cfg.commands), "reward_weights": asdict(cfg.reward_weights)}
    if cfg.leg_acceleration.enabled:
        task["leg_acceleration"] = acceleration_contract_payload(cfg.leg_acceleration)
    return cfg, json.loads(json.dumps(task))


def rehash(base):
    base.pop("contract_hash", None)
    base["contract_hash"] = stable_contract_hash(base)
    return base


@pytest.mark.parametrize("name", ACCELERATION_PROFILE_FRACTIONS)
def test_profiles_roundtrip_json_and_keep_existing_rewards(name):
    cfg, task = profile(name)
    baseline, original = profile("stop_reverse_v1")
    assert cfg.commands == baseline.commands
    assert cfg.reward_weights == baseline.reward_weights
    assert not cfg.leg_reversal.enabled
    assert {k: v for k, v in task.items() if k != "leg_acceleration"} == original
    assert task_profile_from_contract({"task": task}) == name
    state = LegAccelerationReward(1, cfg.leg_acceleration, "cpu").state_dict()
    assert state["contract"] == task["leg_acceleration"]
    apply_task_profile(cfg, "stop_reverse_v1")
    assert not cfg.leg_acceleration.enabled


def test_weights_initialization_only_accepts_added_acceleration_contract(tmp_path):
    _, _, contracts, _ = _write_checkpoint(tmp_path)
    source, algorithm, export = deepcopy(contracts)
    source["task"] = profile("stop_reverse_v1")[1]
    rehash(source)
    target = deepcopy(source)
    target["task"] = profile("stop_reverse_accel80_v1")[1]
    rehash(target)
    require_initialization_compatibility(source, target, algorithm, algorithm, export, export)
    target["task"]["reward_weights"]["dof_acc"] *= 2
    with pytest.raises(ContractMismatchError, match="only add"):
        require_initialization_compatibility(source, target, algorithm, algorithm, export, export)


def test_checkpoint_json_contract_resume_and_missing_or_wrong_budget_rejected(tmp_path):
    cp, manifest_path, contracts, _ = _write_checkpoint(tmp_path)
    base, algorithm, export = deepcopy(contracts)
    cfg, task = profile("stop_reverse_accel80_v1")
    base["task"] = task
    rehash(base)
    controller = LegAccelerationReward(8, cfg.leg_acceleration, "cpu")
    for _ in range(48):
        controller.step(torch.full((8, 6), 100.0))
    state = controller.state_dict()
    assert state["gain"] == 1.0 and state["completed_rollouts"] == 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base_task_contract"] = base
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload = torch.load(cp, weights_only=False, map_location="cpu")
    payload["infos"]["base_task_contract_hash"] = base["contract_hash"]
    payload["infos"]["run_manifest_sha256"] = sha256_file(manifest_path)
    payload["infos"]["leg_acceleration_gain_state"] = state
    torch.save(payload, cp)
    metadata = validate_dreamwaq_checkpoint_metadata(cp, manifest_path, base, algorithm, export)
    runner = DreamWaQContractOnPolicyRunner.__new__(DreamWaQContractOnPolicyRunner)
    runner.alg = _algorithm(learning_rate=1e-3)
    _initialize_optimizer(runner.alg)
    runner._leg_acceleration_reward = LegAccelerationReward(8, cfg.leg_acceleration, "cpu")
    restored = resume_dreamwaq_runner_from_checkpoint(
        runner, cp, metadata, map_location="cpu", requested_seed=42,
        target_total_iterations=3, restore_environment_rng_state=lambda payload: None,
    )
    assert restored == 2
    assert runner._leg_acceleration_reward.state_dict() == state
    for mutation in ("missing", "budget", "count"):
        bad = deepcopy(payload)
        if mutation == "missing":
            bad["infos"].pop("leg_acceleration_gain_state")
        elif mutation == "budget":
            bad["infos"]["leg_acceleration_gain_state"]["contract"]["budget_fraction"] = .75
        else:
            bad["infos"]["leg_acceleration_gain_state"]["completed_rollouts"] = 1
        torch.save(bad, cp)
        with pytest.raises(ContractMismatchError, match="acceleration"):
            validate_dreamwaq_checkpoint_metadata(cp, manifest_path, base, algorithm, export)

