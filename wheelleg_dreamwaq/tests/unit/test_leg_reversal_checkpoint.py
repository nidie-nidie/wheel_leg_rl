from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch

from test_dreamwaq_checkpoint import _write_checkpoint, _algorithm, _initialize_optimizer
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import stable_contract_hash
from wheelleg_dreamwaq.schemas.manifest import ContractMismatchError
from wheelleg_dreamwaq.training.checkpoint import sha256_file
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import (
    validate_dreamwaq_checkpoint_metadata, DreamWaQContractOnPolicyRunner,
    resume_dreamwaq_runner_from_checkpoint,
)
from wheelleg_dreamwaq.training.weights_initialization import (
    require_initialization_compatibility, load_weights_initialization,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import CommandRanges, command_contract_payload
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RewardWeights
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.leg_reversal import LegReversalReward, reversal_contract_payload
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import (
    apply_task_profile, task_profile_from_contract, resolve_task_profile,
)


def profile(name):
    cfg = SimpleNamespace(commands=CommandRanges(), reward_weights=RewardWeights())
    apply_task_profile(cfg, name)
    task = {"commands": command_contract_payload(cfg.commands), "reward_weights": asdict(cfg.reward_weights)}
    if cfg.leg_reversal.enabled:
        task["leg_reversal"] = reversal_contract_payload(cfg.leg_reversal)
    return cfg, json.loads(json.dumps(task))


def rehash(base):
    base.pop("contract_hash", None)
    base["contract_hash"] = stable_contract_hash(base)
    return base


def rewrite(checkpoint, manifest_path, contracts, profile_name, gain_state=None, initialization=None):
    base, algorithm, export = deepcopy(contracts)
    cfg, task = profile(profile_name)
    base["task"] = task
    rehash(base)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base_task_contract"] = base
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    info = payload["infos"]
    info["base_task_contract_hash"] = base["contract_hash"]
    if gain_state is not None:
        info["leg_reversal_gain_state"] = gain_state
    if initialization is not None:
        manifest["weights_initialization"] = initialization
        info["weights_initialization"] = initialization
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    info["run_manifest_sha256"] = sha256_file(manifest_path)
    torch.save(payload, checkpoint)
    return (base, algorithm, export)


@pytest.mark.parametrize("name", ["stop_reverse_reversal12_v1", "stop_reverse_reversal8_v1"])
def test_profiles_only_add_explicit_contract_and_do_not_change_original_rewards(name):
    baseline, old_task = profile("stop_reverse_v1")
    current, task = profile(name)
    assert current.commands == baseline.commands and current.reward_weights == baseline.reward_weights
    assert {k: v for k, v in task.items() if k != "leg_reversal"} == old_task
    assert task_profile_from_contract({"task": task}) == name
    with pytest.raises(ValueError, match="profile"):
        resolve_task_profile("stop_reverse_v1", {"task": task})
    task["leg_reversal"]["amplitude_rad"] *= 2
    with pytest.raises(ValueError, match="profile"):
        task_profile_from_contract({"task": task})
    apply_task_profile(current, "stop_reverse_v1")
    assert current.leg_reversal.enabled is False


def test_weights_only_compatibility_rejects_other_changes_and_validates_source(tmp_path):
    cp, manifest_path, contracts, _ = _write_checkpoint(tmp_path)
    source_contracts = rewrite(cp, manifest_path, contracts, "stop_reverse_v1")
    target = deepcopy(source_contracts[0])
    target["task"] = profile("stop_reverse_reversal12_v1")[1]
    rehash(target)
    weights, source, metadata, provenance = load_weights_initialization(
        cp, target, source_contracts[1], source_contracts[2])
    assert provenance["source_completed_iterations"] == 2
    assert provenance["source_checkpoint_sha256"] == sha256_file(cp)
    assert weights["raw_std"].equal(torch.load(cp, weights_only=False)["model_state_dict"]["raw_std"])
    for key, value in [("q_nominal", [0] * 4), ("reward_weights", asdict(RewardWeights()))]:
        bad = deepcopy(target)
        bad["task"][key] = value
        with pytest.raises(ContractMismatchError, match="only add"):
            require_initialization_compatibility(source_contracts[0], bad, source_contracts[1],
                source_contracts[1], source_contracts[2], source_contracts[2])
    payload = torch.load(cp, weights_only=False)
    payload["infos"]["run_manifest_sha256"] = "F" * 64
    torch.save(payload, cp)
    with pytest.raises(ContractMismatchError, match="manifest hash"):
        load_weights_initialization(cp, target, source_contracts[1], source_contracts[2])


def test_reversal_checkpoint_requires_state_and_resume_restores_gain(tmp_path):
    cp, manifest_path, contracts, _ = _write_checkpoint(tmp_path)
    cfg = profile("stop_reverse_reversal12_v1")[0].leg_reversal
    controller = LegReversalReward(8, cfg, "cpu")
    for i in range(72):
        controller.step(torch.full((8, 4), .05 if i % 2 == 0 else -.05))
    state = controller.state_dict()
    assert state["gain"] == 1.0
    new_contracts = rewrite(cp, manifest_path, contracts, "stop_reverse_reversal12_v1", state)
    payload = torch.load(cp, weights_only=False)
    payload["iter"] = 2
    payload["infos"]["runner_iteration"] = 2
    payload["infos"]["completed_iterations"] = 3
    payload["infos"]["total_timesteps"] = 576
    from wheelleg_dreamwaq.schemas.dreamwaq_manifest import EstimatorMonitorStateV1
    monitor = EstimatorMonitorStateV1()
    for value in (.2, .3, .4):
        monitor.record_rollout(value)
    payload["infos"]["estimator_monitor_state"] = monitor.to_dict()
    torch.save(payload, cp)
    metadata = validate_dreamwaq_checkpoint_metadata(cp, manifest_path, *new_contracts)
    runner = DreamWaQContractOnPolicyRunner.__new__(DreamWaQContractOnPolicyRunner)
    runner.alg = _algorithm(learning_rate=1e-3)
    _initialize_optimizer(runner.alg)
    runner._leg_reversal_reward = LegReversalReward(8, cfg, "cpu")
    restored = resume_dreamwaq_runner_from_checkpoint(
        runner, cp, metadata, map_location="cpu", requested_seed=42,
        target_total_iterations=4, restore_environment_rng_state=lambda payload: None)
    assert restored == 3
    assert runner._leg_reversal_reward.state_dict() == state
    assert not runner._leg_reversal_reward.counter.eligible.any()
    payload = torch.load(cp, weights_only=False)
    payload["infos"].pop("leg_reversal_gain_state")
    torch.save(payload, cp)
    with pytest.raises(ContractMismatchError, match="reversal"):
        validate_dreamwaq_checkpoint_metadata(cp, manifest_path, *new_contracts)
