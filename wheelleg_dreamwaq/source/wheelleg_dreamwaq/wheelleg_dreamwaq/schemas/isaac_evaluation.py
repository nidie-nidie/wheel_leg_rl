from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
    BASE_TASK_CONTRACT_VERSION,
    stable_contract_hash,
)
from wheelleg_dreamwaq.schemas.randomization import (
    NOMINAL_EVALUATION_PROFILE_V1,
    profile_contract_hash,
    profile_contract_payload,
)


ISAAC_EVALUATION_CONTRACT_VERSION = "IsaacEvaluationContractV1"
ISAAC_EVALUATION_SCHEMA_VERSION = "IsaacEvaluationV1"
ISAAC_EVALUATION_SOURCE_VERSION = "IsaacEvaluationSourceV1"
ISAAC_EVALUATION_CACHE_IDENTITY_VERSION = "IsaacEvaluationResetCacheIdentityV1"
PHASE1R_ISAAC_BASELINE_VERSION = "Phase1RIsaacBaselineV1"
EVALUATION_SEED = 20261007
EVALUATION_ENV_COUNT = 8
EVALUATION_ACTION_STEPS = 499
EVALUATION_CONTROL_DT_S = 0.02
EVALUATION_DURATION_S = EVALUATION_ACTION_STEPS * EVALUATION_CONTROL_DT_S
VELOCITY_MSE_DENOMINATOR_FLOOR = 1.0e-8
VELOCITY_MSE_MAX_ZERO_BASELINE_RATIO = 0.80

FORMAL_SCENARIOS = (
    ("nominal_stand", (0.0, 0.0, 0.20)),
    ("low_stand", (0.0, 0.0, 0.17)),
    ("high_stand", (0.0, 0.0, 0.23)),
    ("forward", (1.0, 0.0, 0.20)),
    ("reverse", (-1.0, 0.0, 0.20)),
    ("left_turn", (0.0, 0.6, 0.20)),
    ("right_turn", (0.0, -0.6, 0.20)),
    ("combined", (0.8, 0.5, 0.20)),
)

PHASE1R_ISAAC_BASELINE = {
    "schema_version": PHASE1R_ISAAC_BASELINE_VERSION,
    "checkpoint": (
        "E:\\wheel_leg_rl-main\\wheelleg_dreamwaq\\logs\\rsl_rl\\wheelleg_flat_ppo\\"
        "2026-10-07_07-32-32_rtx5070_fudan-v1_phase1r-v3-suite-20261007-062147-run03-seed1884612625\\"
        "model_999.pt"
    ),
    "checkpoint_sha256": "31C1A7DB00AD00B68794D721B5B7DDD0E4765B583E3F916710D917D5053B56B4",
    "run_manifest_sha256": "026B6D8F151A9DFC3B13E6342EAE5B6851E34C78F2B01EABC2577F1ED4F54984",
    "phase1_contract_version": "Phase1RandomizedContractV3",
    "phase1_contract_hash": "CC57F16B1BB75E169CCF21CD706E10C8D2D23498434ACED982AFFA8FD1F72848",
    "seed": 1884612625,
    "completed_iterations": 1000,
}


def _require_upper_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.upper() != value:
        raise ValueError(f"{label} must be an uppercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be an uppercase SHA256") from error
    return value


def build_evaluation_source_fingerprint(files: dict[str, str]) -> dict[str, Any]:
    if not files:
        raise ValueError("Isaac evaluation source fingerprint requires at least one file")
    normalized = {}
    for path, digest in sorted(files.items()):
        if not isinstance(path, str) or not path:
            raise ValueError("Isaac evaluation source paths must be non-empty strings")
        normalized[path.replace("\\", "/")] = _require_upper_sha256(digest, f"source file {path}")
    payload = {"schema_version": ISAAC_EVALUATION_SOURCE_VERSION, "files": normalized}
    payload["hash"] = stable_contract_hash(payload)
    return payload


def build_evaluation_reset_cache_identity(
    *,
    path: str,
    file_sha256: str,
    cache_payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(cache_payload, dict):
        raise ValueError("Evaluation reset cache payload must be a dictionary")
    q_reset = cache_payload.get("q_reset_projected_env")
    if not isinstance(q_reset, torch.Tensor) or q_reset.ndim != 2 or q_reset.shape[0] != EVALUATION_ENV_COUNT:
        raise ValueError("Evaluation reset cache must contain exactly eight environments")
    identity = {
        "schema_version": ISAAC_EVALUATION_CACHE_IDENTITY_VERSION,
        "path": str(path),
        "file_sha256": _require_upper_sha256(file_sha256, "evaluation reset cache file"),
        "tensor_sha256": _require_upper_sha256(cache_payload.get("tensor_sha256"), "evaluation reset cache tensor"),
        "cache_schema_version": cache_payload.get("schema_version"),
        "relaxation_algorithm_version": cache_payload.get("algorithm_version"),
        "root_height_algorithm_version": cache_payload.get("root_height_algorithm_version"),
        "num_envs": int(q_reset.shape[0]),
    }
    identity["identity_hash"] = stable_contract_hash(identity)
    return identity


def build_isaac_evaluation_contract(
    *,
    base_task_contract: dict[str, Any],
    evaluation_source: dict[str, Any],
    reset_cache_identity: dict[str, Any],
) -> dict[str, Any]:
    if base_task_contract.get("manifest_version") != BASE_TASK_CONTRACT_VERSION:
        raise ValueError("Isaac evaluation requires WheelLegBaseTaskContractV1")
    base_hash = base_task_contract.get("contract_hash")
    _require_upper_sha256(base_hash, "base-task contract")
    if evaluation_source.get("schema_version") != ISAAC_EVALUATION_SOURCE_VERSION:
        raise ValueError("Isaac evaluation source fingerprint version mismatch")
    source_hash = evaluation_source.get("hash")
    if source_hash != stable_contract_hash({key: value for key, value in evaluation_source.items() if key != "hash"}):
        raise ValueError("Isaac evaluation source fingerprint hash mismatch")
    if reset_cache_identity.get("schema_version") != ISAAC_EVALUATION_CACHE_IDENTITY_VERSION:
        raise ValueError("Isaac evaluation reset cache identity version mismatch")
    cache_hash = reset_cache_identity.get("identity_hash")
    if cache_hash != stable_contract_hash(
        {key: value for key, value in reset_cache_identity.items() if key != "identity_hash"}
    ):
        raise ValueError("Isaac evaluation reset cache identity hash mismatch")
    payload = {
        "manifest_version": ISAAC_EVALUATION_CONTRACT_VERSION,
        "evaluation_schema_version": ISAAC_EVALUATION_SCHEMA_VERSION,
        "seed": EVALUATION_SEED,
        "environment_count": EVALUATION_ENV_COUNT,
        "randomization_profile": profile_contract_payload(NOMINAL_EVALUATION_PROFILE_V1),
        "randomization_profile_hash": profile_contract_hash(NOMINAL_EVALUATION_PROFILE_V1),
        "control": {
            "action_steps": EVALUATION_ACTION_STEPS,
            "control_dt_s": EVALUATION_CONTROL_DT_S,
            "duration_s": EVALUATION_DURATION_S,
            "expected_done": "truncated_on_action_step_499",
            "policy_action": "deterministic_effective_mean",
            "runtime_action_clip": [-1.0, 1.0],
        },
        "scenarios": [{"name": name, "command": list(command)} for name, command in FORMAL_SCENARIOS],
        "first_observation_sequence": [
            "discard_upstream_wrapper_reset_observation",
            "write_eight_fixed_commands",
            "call_get_observations_once",
            "fill_five_history_frames_from_first_policy_observation",
        ],
        "estimator": {
            "sample_timing": "active_pre_action_frame",
            "target": "critic[:,25:28]",
            "aggregation": "sample_weighted_over_frames_and_three_features",
            "denominator_floor": VELOCITY_MSE_DENOMINATOR_FLOOR,
            "maximum_zero_baseline_ratio": VELOCITY_MSE_MAX_ZERO_BASELINE_RATIO,
        },
        "done_semantics": {
            "terminated": "failure",
            "truncated_before_step_499": "protocol_failure",
            "truncated_on_step_499": "completed",
            "post_reset_observation_counted": False,
        },
        "performance_key": {
            "full_survival": [0, "-round(mean_scenario_return,6)"],
            "partial_survival": [
                1,
                "-completed_scenarios",
                "-round(mean_survival_fraction,6)",
                "-round(mean_scenario_return,6)",
            ],
            "comparison": "candidate_key <= baseline_key",
        },
        "base_task_contract_hash": base_hash,
        "evaluation_source": evaluation_source,
        "reset_cache": reset_cache_identity,
        "baseline": dict(PHASE1R_ISAAC_BASELINE),
    }
    payload["contract_hash"] = stable_contract_hash(payload)
    return payload


def validate_matching_isaac_evaluation_contracts(saved: dict[str, Any], current: dict[str, Any]) -> str:
    for label, contract in (("saved", saved), ("current", current)):
        if not isinstance(contract, dict) or contract.get("manifest_version") != ISAAC_EVALUATION_CONTRACT_VERSION:
            raise ValueError(f"Isaac {label} evaluation contract version is invalid")
        embedded = contract.get("contract_hash")
        calculated = stable_contract_hash({key: value for key, value in contract.items() if key != "contract_hash"})
        if embedded != calculated:
            raise ValueError(f"Isaac {label} evaluation contract hash is invalid")
    if saved["contract_hash"] != current["contract_hash"]:
        raise ValueError("Isaac baseline and candidate evaluation contracts differ")
    return saved["contract_hash"]


@dataclass
class SampleWeightedVelocityMSE:
    squared_error_sum: float = 0.0
    zero_baseline_squared_error_sum: float = 0.0
    scalar_sample_count: int = 0
    frame_count: int = 0

    def update(self, estimated: torch.Tensor, target: torch.Tensor, active: torch.Tensor) -> None:
        if estimated.shape != target.shape or estimated.ndim != 2 or estimated.shape[1] != 3:
            raise ValueError("Velocity tensors must have identical [N,3] shapes")
        if active.shape != (estimated.shape[0],) or active.dtype != torch.bool:
            raise ValueError("active mask must be bool with shape [N]")
        selected_estimated = estimated[active]
        selected_target = target[active]
        if selected_target.numel() == 0:
            return
        if not torch.isfinite(selected_estimated).all() or not torch.isfinite(selected_target).all():
            raise ValueError("Velocity evaluation samples must be finite")
        self.squared_error_sum += float((selected_estimated - selected_target).square().sum().item())
        self.zero_baseline_squared_error_sum += float(selected_target.square().sum().item())
        self.scalar_sample_count += int(selected_target.numel())
        self.frame_count += int(selected_target.shape[0])

    def result(self) -> dict[str, float | int | bool]:
        if self.scalar_sample_count <= 0:
            raise ValueError("Velocity evaluation contains no active pre-action frames")
        mse = self.squared_error_sum / self.scalar_sample_count
        zero_mse = self.zero_baseline_squared_error_sum / self.scalar_sample_count
        ratio = mse / max(zero_mse, VELOCITY_MSE_DENOMINATOR_FLOOR)
        return {
            "velocity_eval_mse": mse,
            "velocity_zero_baseline_mse": zero_mse,
            "velocity_eval_mse_ratio": ratio,
            "velocity_sample_frames": self.frame_count,
            "velocity_scalar_samples": self.scalar_sample_count,
            "velocity_acceptance": ratio <= VELOCITY_MSE_MAX_ZERO_BASELINE_RATIO,
        }


def summarize_scenario(
    *,
    name: str,
    command: tuple[float, float, float],
    reward_sum: float,
    survival_steps: int,
    done_step: int | None,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    expected = dict(FORMAL_SCENARIOS).get(name)
    if expected != command:
        raise ValueError(f"Scenario {name!r} command differs from IsaacEvaluationContractV1")
    if not math.isfinite(reward_sum):
        raise ValueError("Scenario reward sum must be finite")
    if not 0 <= survival_steps <= EVALUATION_ACTION_STEPS:
        raise ValueError("Scenario survival_steps is outside the evaluation horizon")
    if terminated and truncated:
        raise ValueError("A scenario cannot be both terminated and truncated")
    completed = False
    failure_reason = None
    if terminated:
        failure_reason = "terminated"
    elif truncated:
        if done_step == EVALUATION_ACTION_STEPS and survival_steps == EVALUATION_ACTION_STEPS:
            completed = True
        else:
            failure_reason = "early_timeout"
    elif done_step is not None:
        failure_reason = "done_without_terminated_or_truncated"
    elif survival_steps == EVALUATION_ACTION_STEPS:
        failure_reason = "missing_expected_timeout"
    else:
        failure_reason = "incomplete_without_done"
    return {
        "name": name,
        "command": list(command),
        "reward_sum": float(reward_sum),
        "survival_steps": int(survival_steps),
        "survival_fraction": survival_steps / EVALUATION_ACTION_STEPS,
        "completed": completed,
        "failure_reason": failure_reason,
    }


def policy_performance_key(aggregate: dict[str, Any]) -> tuple[float | int, ...]:
    completed = aggregate.get("completed_scenarios")
    scenario_count = aggregate.get("scenario_count")
    mean_survival = aggregate.get("mean_survival_fraction")
    mean_return = aggregate.get("mean_scenario_return")
    if not isinstance(completed, int) or not isinstance(scenario_count, int) or scenario_count <= 0:
        raise ValueError("Isaac aggregate scenario counts are invalid")
    if not 0 <= completed <= scenario_count:
        raise ValueError("Isaac aggregate completed scenario count is invalid")
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (mean_survival, mean_return)):
        raise ValueError("Isaac aggregate means must be finite")
    rounded_return = round(float(mean_return), 6)
    if completed == scenario_count:
        return (0, -rounded_return)
    return (1, -completed, -round(float(mean_survival), 6), -rounded_return)


def aggregate_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    if len(scenarios) != len(FORMAL_SCENARIOS):
        raise ValueError("Isaac evaluation aggregate requires all eight scenarios")
    expected_names = [name for name, _ in FORMAL_SCENARIOS]
    if [scenario.get("name") for scenario in scenarios] != expected_names:
        raise ValueError("Isaac evaluation scenario order differs from the frozen contract")
    completed = sum(bool(scenario.get("completed")) for scenario in scenarios)
    aggregate = {
        "scenario_count": len(scenarios),
        "completed_scenarios": completed,
        "full_survival": completed == len(scenarios),
        "mean_survival_fraction": sum(float(item["survival_fraction"]) for item in scenarios) / len(scenarios),
        "mean_scenario_return": sum(float(item["reward_sum"]) for item in scenarios) / len(scenarios),
        "scenarios": scenarios,
    }
    aggregate["performance_key"] = list(policy_performance_key(aggregate))
    return aggregate


def candidate_not_worse_than_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return policy_performance_key(candidate) <= policy_performance_key(baseline)
