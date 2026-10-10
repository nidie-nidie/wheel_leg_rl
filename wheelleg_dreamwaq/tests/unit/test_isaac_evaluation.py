from __future__ import annotations

from copy import deepcopy
import json

import pytest
import torch

from wheelleg_dreamwaq.schemas.dreamwaq_manifest import BASE_TASK_CONTRACT_VERSION, stable_contract_hash
from wheelleg_dreamwaq.schemas.isaac_evaluation import (
    EVALUATION_ACTION_STEPS,
    EVALUATION_DURATION_S,
    EVALUATION_ENV_COUNT,
    EVALUATION_SEED,
    FORMAL_SCENARIOS,
    ISAAC_EVALUATION_CONTRACT_VERSION,
    PHASE1R_ISAAC_BASELINE,
    SampleWeightedVelocityMSE,
    aggregate_scenarios,
    build_evaluation_reset_cache_identity,
    build_evaluation_source_fingerprint,
    build_isaac_evaluation_contract,
    candidate_not_worse_than_baseline,
    policy_performance_key,
    summarize_scenario,
    validate_matching_isaac_evaluation_contracts,
)


def _base_contract() -> dict:
    payload = {"manifest_version": BASE_TASK_CONTRACT_VERSION, "frozen": True}
    payload["contract_hash"] = stable_contract_hash(payload)
    return payload


def _cache_payload() -> dict:
    return {
        "schema_version": "ClosedChainResetCacheSchemaV2",
        "algorithm_version": "BoundaryClampedPhysXRelaxationV1",
        "root_height_algorithm_version": "ClosedChainRootHeightAlignmentV1",
        "tensor_sha256": "A" * 64,
        "q_reset_projected_env": torch.zeros(EVALUATION_ENV_COUNT, 18),
    }


def test_isaac_evaluation_contract_freezes_horizon_scenarios_cache_and_baseline() -> None:
    source = build_evaluation_source_fingerprint({"scripts/evaluate_isaac.py": "B" * 64})
    cache = build_evaluation_reset_cache_identity(
        path="evaluation-reset-cache.pt",
        file_sha256="C" * 64,
        cache_payload=_cache_payload(),
    )
    contract = build_isaac_evaluation_contract(
        base_task_contract=_base_contract(), evaluation_source=source, reset_cache_identity=cache
    )

    assert contract["manifest_version"] == ISAAC_EVALUATION_CONTRACT_VERSION
    assert contract["seed"] == EVALUATION_SEED
    assert contract["environment_count"] == EVALUATION_ENV_COUNT
    assert contract["control"]["action_steps"] == EVALUATION_ACTION_STEPS == 499
    assert contract["control"]["duration_s"] == EVALUATION_DURATION_S == 9.98
    assert [(item["name"], tuple(item["command"])) for item in contract["scenarios"]] == list(FORMAL_SCENARIOS)
    assert contract["reset_cache"]["file_sha256"] == "C" * 64
    assert contract["baseline"] == PHASE1R_ISAAC_BASELINE
    assert contract["contract_hash"] == stable_contract_hash(
        {key: value for key, value in contract.items() if key != "contract_hash"}
    )


def test_evaluation_contract_identity_accepts_json_list_tuple_round_trip() -> None:
    source = build_evaluation_source_fingerprint({"scripts/evaluate_isaac.py": "B" * 64})
    cache = build_evaluation_reset_cache_identity(
        path="evaluation-reset-cache.pt",
        file_sha256="C" * 64,
        cache_payload=_cache_payload(),
    )
    current = build_isaac_evaluation_contract(
        base_task_contract=_base_contract(), evaluation_source=source, reset_cache_identity=cache
    )
    saved = json.loads(json.dumps(current))

    assert saved != current
    assert validate_matching_isaac_evaluation_contracts(saved, current) == current["contract_hash"]

    tampered = deepcopy(saved)
    tampered["control"]["action_steps"] = 500
    with pytest.raises(ValueError, match="saved evaluation contract hash"):
        validate_matching_isaac_evaluation_contracts(tampered, current)


def test_velocity_mse_is_sample_weighted_over_active_frames_and_features() -> None:
    accumulator = SampleWeightedVelocityMSE()
    accumulator.update(
        torch.tensor([[1.0, 1.0, 1.0], [100.0, 100.0, 100.0]]),
        torch.tensor([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]),
        torch.tensor([True, False]),
    )
    accumulator.update(
        torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
        torch.tensor([True, True]),
    )
    result = accumulator.result()

    assert result["velocity_sample_frames"] == 3
    assert result["velocity_scalar_samples"] == 9
    assert result["velocity_eval_mse"] == pytest.approx(2.0)
    assert result["velocity_zero_baseline_mse"] == pytest.approx(1.0 / 3.0)
    assert result["velocity_eval_mse_ratio"] == pytest.approx(6.0)
    assert result["velocity_acceptance"] is False


def test_expected_step_499_timeout_is_completion() -> None:
    summary = summarize_scenario(
        name="nominal_stand",
        command=(0.0, 0.0, 0.20),
        reward_sum=10.0,
        survival_steps=499,
        done_step=499,
        terminated=False,
        truncated=True,
    )
    assert summary["completed"] is True
    assert summary["failure_reason"] is None
    assert summary["survival_fraction"] == 1.0


@pytest.mark.parametrize(
    ("done_step", "survival_steps", "terminated", "truncated", "reason"),
    [
        (100, 100, False, True, "early_timeout"),
        (499, 499, True, False, "terminated"),
        (None, 499, False, False, "missing_expected_timeout"),
    ],
)
def test_unexpected_done_semantics_are_failures(
    done_step: int | None,
    survival_steps: int,
    terminated: bool,
    truncated: bool,
    reason: str,
) -> None:
    summary = summarize_scenario(
        name="nominal_stand",
        command=(0.0, 0.0, 0.20),
        reward_sum=1.0,
        survival_steps=survival_steps,
        done_step=done_step,
        terminated=terminated,
        truncated=truncated,
    )
    assert summary["completed"] is False
    assert summary["failure_reason"] == reason


def _scenario_summaries(*, completed: int, reward: float, survival: int = 499) -> list[dict]:
    results = []
    for index, (name, command) in enumerate(FORMAL_SCENARIOS):
        is_complete = index < completed
        results.append(
            summarize_scenario(
                name=name,
                command=command,
                reward_sum=reward,
                survival_steps=499 if is_complete else survival,
                done_step=499 if is_complete else survival,
                terminated=not is_complete,
                truncated=is_complete,
            )
        )
    return results


def test_performance_key_and_baseline_comparison_are_lexicographic() -> None:
    baseline = aggregate_scenarios(_scenario_summaries(completed=8, reward=10.0))
    better = aggregate_scenarios(_scenario_summaries(completed=8, reward=10.1))
    partial = aggregate_scenarios(_scenario_summaries(completed=7, reward=1000.0, survival=498))

    assert policy_performance_key(baseline) == (0, -10.0)
    assert policy_performance_key(partial)[0] == 1
    assert candidate_not_worse_than_baseline(better, baseline) is True
    assert candidate_not_worse_than_baseline(partial, baseline) is False


def test_evaluation_contract_rejects_tampered_source_or_cache_identity() -> None:
    source = build_evaluation_source_fingerprint({"scripts/evaluate_isaac.py": "B" * 64})
    cache = build_evaluation_reset_cache_identity(
        path="evaluation-reset-cache.pt",
        file_sha256="C" * 64,
        cache_payload=_cache_payload(),
    )
    tampered = deepcopy(cache)
    tampered["num_envs"] = 7

    with pytest.raises(ValueError, match="cache identity hash"):
        build_isaac_evaluation_contract(
            base_task_contract=_base_contract(), evaluation_source=source, reset_cache_identity=tampered
        )
