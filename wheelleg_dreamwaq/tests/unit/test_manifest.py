from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2
from wheelleg_dreamwaq.schemas.manifest import (
    ContractMismatchError,
    PHASE1_CONTRACT_VERSION,
    build_phase1_contract,
    validate_contract,
)
from wheelleg_dreamwaq.schemas.normalization import NormalizationV1, NormalizationV2
from wheelleg_dreamwaq.schemas.physics import (
    DECIMATION,
    SIM_DT_S,
    SOLVER_POSITION_ITERATIONS,
    SOLVER_VELOCITY_ITERATIONS,
    unrestricted_velocity_policy,
)
from wheelleg_dreamwaq.schemas.randomization import (
    FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
    NOMINAL_TRAINING_PROFILE_V1,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import CommandRanges
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import ControlLimits
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RewardWeights
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.terminations import TerminationLimits


def _contract(
    *,
    clip_actions: float = 1.0,
    is_finite_horizon: bool = False,
    sim_dt: float = SIM_DT_S,
    decimation: int = DECIMATION,
    solver_position_iterations: int = SOLVER_POSITION_ITERATIONS,
    solver_velocity_iterations: int = SOLVER_VELOCITY_ITERATIONS,
    runtime: dict | None = None,
    training: dict | None = None,
    randomization=FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
) -> dict:
    return build_phase1_contract(
        asset_bundle_hash=ASSET_BUNDLE_V2.bundle_hash,
        sim_dt=sim_dt,
        decimation=decimation,
        solver_position_iterations=solver_position_iterations,
        solver_velocity_iterations=solver_velocity_iterations,
        episode_length_s=10.0,
        q_nominal=(-0.3, 0.3, -0.3, 0.3),
        control=ControlLimits(),
        commands=CommandRanges(),
        normalization=NormalizationV2(),
        randomization=randomization,
        reward_weights=RewardWeights(),
        termination=TerminationLimits(),
        actor_hidden_dims=(256, 128, 64),
        critic_hidden_dims=(256, 128, 64),
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        clip_actions=clip_actions,
        is_finite_horizon=is_finite_horizon,
        training=training
        or {
            "runner": {"num_steps_per_env": 24},
            "algorithm_except_hardware_batching": {"gamma": 0.99},
        },
        runtime=runtime or {"collision_policy": "test", "robot": {"disable_gravity": False}},
    )


def test_contract_hash_is_stable_and_self_validating() -> None:
    first = _contract()
    second = _contract()

    assert first == second
    assert first["manifest_version"] == "Phase1RandomizedContractV3"
    assert first["asset_bundle_version"] == "AssetBundleV2"
    assert first["asset_bundle_hash"] == ASSET_BUNDLE_V2.bundle_hash
    assert first["schemas"]["normalization"] == "NormalizationV2"
    assert first["schemas"]["physics"] == "PhysicsV5"
    assert first["schemas"]["reward"] == "RewardSchemaV2"
    assert first["schemas"]["virtual_leg_kinematics"] == "VirtualLegKinematicsV1"
    assert first["schemas"]["randomization"] == "RandomizationSchemaV1"
    assert first["schemas"]["closed_chain_reset_cache"] == "ClosedChainResetCacheSchemaV2"
    assert first["randomization"]["profile_contract"]["profile"]["name"] == "FudanStyleDomainRandomizationV1"
    assert first["task"]["commands"]["hold_for_episode"] is True
    assert first["task"]["physics"] == {
        "sim_dt": 0.005,
        "decimation": 4,
        "control_dt": 0.02,
        "solver_position_iterations": 96,
        "solver_velocity_iterations": 4,
        "velocity_limit_policy": unrestricted_velocity_policy(),
    }
    assert first["virtual_leg"]["asset_bundle_hash"] == ASSET_BUNDLE_V2.bundle_hash
    assert len(first["virtual_leg"]["semantic_sha256"]) == 64
    assert first["virtual_leg"]["runtime_thresholds"]["max_loop_closure_error_m"] == 0.005
    assert first["task"]["reward_weights"]["phi0_symmetry"] == -1.0
    assert first["randomization"]["reset_joint_state"]["passive_position"] == "closed_chain_reset_cache"
    assert first["randomization"]["reset_joint_state"]["passive_velocity_target"] == "zero"
    assert first["randomization"]["closed_chain_reset_cache"]["algorithm_version"] == (
        "BoundaryClampedPhysXRelaxationV1"
    )
    assert first["randomization"]["closed_chain_reset_cache"]["root_height_algorithm_version"] == (
        "ClosedChainRootHeightAlignmentV1"
    )
    assert first["randomization"]["reset_joint_state"]["root_height"] == (
        "closed_chain_reset_cache"
    )
    assert PHASE1_CONTRACT_VERSION == "Phase1RandomizedContractV3"
    assert len(first["contract_hash"]) == 64
    validate_contract(first, second)


def test_contract_rejects_normalization_drift_even_if_version_is_unchanged() -> None:
    saved = _contract()
    changed = deepcopy(saved)
    changed["normalization"]["joint_velocity_scale"] = 0.1

    with pytest.raises(ContractMismatchError, match="embedded hash"):
        validate_contract(changed, saved)


def test_contract_rejects_correctly_hashed_historical_braking_physics() -> None:
    current = _contract()
    historical = deepcopy(current)
    historical["schemas"]["physics"] = "PhysicsV4"
    del historical["task"]["physics"]["velocity_limit_policy"]
    payload = {key: value for key, value in historical.items() if key != "contract_hash"}
    historical["contract_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest().upper()
    with pytest.raises(ContractMismatchError, match="does not match the current runtime"):
        validate_contract(historical, current)


def test_contract_rejects_historical_v3_contract() -> None:
    current = _contract()
    historical = deepcopy(current)
    historical["manifest_version"] = "Phase1ContractV3"
    historical["schemas"]["normalization"] = NormalizationV1.schema_version
    historical["contract_hash"] = "0" * 64

    with pytest.raises(ContractMismatchError, match="version"):
        validate_contract(historical, current)


def test_contract_rejects_checkpoint_from_another_schema_contract() -> None:
    saved = _contract()
    current = deepcopy(saved)
    current["task"]["q_nominal"][0] = -0.4
    current["contract_hash"] = "0" * 64

    with pytest.raises(ContractMismatchError):
        validate_contract(saved, current)


@pytest.mark.parametrize(
    "changed",
    [
        _contract(clip_actions=0.5),
        _contract(is_finite_horizon=True),
        _contract(sim_dt=0.01, decimation=2),
        _contract(solver_position_iterations=32),
        _contract(runtime={"collision_policy": "test", "robot": {"disable_gravity": True}}),
        _contract(runtime={"collision_policy": "test", "robot": {"root_lin_vel": [0.1, 0.0, 0.0]}}),
        _contract(
            training={
                "runner": {"num_steps_per_env": 24},
                "algorithm_except_hardware_batching": {"gamma": 0.95},
            }
        ),
        _contract(randomization=NOMINAL_TRAINING_PROFILE_V1),
    ],
)
def test_contract_hash_changes_for_behavioral_runtime_or_ppo_drift(changed: dict) -> None:
    assert changed["contract_hash"] != _contract()["contract_hash"]
