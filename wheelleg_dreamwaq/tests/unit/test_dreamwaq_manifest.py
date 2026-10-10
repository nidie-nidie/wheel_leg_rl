from __future__ import annotations

from copy import deepcopy

import pytest

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
    BASE_TASK_CONTRACT_VERSION,
    DREAMWAQ_ALGORITHM_CONTRACT_VERSION,
    DREAMWAQ_EXPORT_CONTRACT_VERSION,
    GOLDEN_VECTOR_SEED,
    GOLDEN_VECTOR_SHAPE,
    TORCHSCRIPT_MAX_ABS_ERROR,
    DreamWaQContractMismatchError,
    build_base_task_contract_from_phase1_contract,
    build_dreamwaq_algorithm_contract,
    build_dreamwaq_export_contract,
    validate_named_contract,
    stable_contract_hash,
)
from wheelleg_dreamwaq.schemas.manifest import build_phase1_contract
from wheelleg_dreamwaq.schemas.normalization import NormalizationV2
from wheelleg_dreamwaq.schemas.physics import (
    DECIMATION,
    SIM_DT_S,
    SOLVER_POSITION_ITERATIONS,
    SOLVER_VELOCITY_ITERATIONS,
)
from wheelleg_dreamwaq.schemas.randomization import FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import CommandRanges
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import ControlLimits
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RewardWeights
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.terminations import TerminationLimits


def _phase1_contract() -> dict:
    return build_phase1_contract(
        asset_bundle_hash=ASSET_BUNDLE_V2.bundle_hash,
        sim_dt=SIM_DT_S,
        decimation=DECIMATION,
        solver_position_iterations=SOLVER_POSITION_ITERATIONS,
        solver_velocity_iterations=SOLVER_VELOCITY_ITERATIONS,
        episode_length_s=10.0,
        q_nominal=(-0.3, 0.3, -0.3, 0.3),
        control=ControlLimits(),
        commands=CommandRanges(),
        normalization=NormalizationV2(),
        randomization=FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
        reward_weights=RewardWeights(),
        termination=TerminationLimits(),
        actor_hidden_dims=(256, 128, 64),
        critic_hidden_dims=(256, 128, 64),
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        clip_actions=1.0,
        is_finite_horizon=False,
        training={"phase1_only": True},
        runtime={"collision_policy": "test"},
    )


def _runner_config(*, num_mini_batches: int = 4) -> dict:
    return {
        "class_name": "DreamWaQContractOnPolicyRunner",
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "clip_actions": 1.0,
        "num_steps_per_env": 24,
        "policy": {
            "class_name": "DreamWaQActorCritic",
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "state_dependent_std": False,
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "activation": "elu",
            "current_obs_group": "policy",
            "history_obs_group": "policy_history",
            "history_length": 5,
            "current_obs_dim": 25,
            "history_obs_dim": 125,
            "critic_obs_dim": 41,
            "velocity_dim": 3,
            "context_dim": 16,
            "reconstruction_target_dim": 16,
            "cenet_encoder_hidden_dims": [128, 64],
            "cenet_decoder_hidden_dims": [64, 128],
            "actor_context_mode": "deterministic_context_mu",
            "action_mean_clip": 20.0,
            "action_std_min": 1.0e-4,
            "action_std_max": 2.0,
        },
        "algorithm": {
            "class_name": "DreamWaQPPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": num_mini_batches,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "velocity_coef": 1.0,
            "reconstruction_coef": 1.0,
            "kl_beta": 1.0,
            "strict_finite_checks": True,
            "context_mu_min_feature_std": 1.0e-3,
            "context_mu_min_initial_std_ratio": 0.10,
            "context_monitor_final_window_iterations": 10,
            "velocity_mse_max_zero_baseline_ratio": 0.80,
            "velocity_mse_denominator_floor": 1.0e-8,
        },
    }


def test_base_task_contract_extracts_only_frozen_environment_semantics() -> None:
    base = build_base_task_contract_from_phase1_contract(_phase1_contract())

    assert base["manifest_version"] == BASE_TASK_CONTRACT_VERSION
    assert "network" not in base
    assert "training" not in base
    assert len(base["randomization"]["profile_contract"]["streams"]) == 7
    assert base["randomization"]["profile_contract"]["profile"]["angular_velocity_noise_rad_s"] == 0.2
    assert base["randomization"]["profile_contract"]["profile"]["projected_gravity_noise"] == 0.05
    assert base["randomization"]["profile_contract"]["profile"]["leg_position_noise_rad"] == 0.02
    assert base["randomization"]["profile_contract"]["profile"]["joint_velocity_noise_rad_s"] == 1.5
    validate_named_contract(base, deepcopy(base), BASE_TASK_CONTRACT_VERSION)


def test_algorithm_contract_ignores_only_hardware_batching_choice() -> None:
    first = build_dreamwaq_algorithm_contract(_runner_config(num_mini_batches=4))
    second = build_dreamwaq_algorithm_contract(_runner_config(num_mini_batches=8))

    assert first == second
    assert first["manifest_version"] == DREAMWAQ_ALGORITHM_CONTRACT_VERSION
    assert first["hardware_mutable_fields"] == ["device", "num_envs", "num_mini_batches"]
    validate_named_contract(first, second, DREAMWAQ_ALGORITHM_CONTRACT_VERSION)


def test_dreamwaq_resume_rejects_valid_historical_physics_contract() -> None:
    current = build_base_task_contract_from_phase1_contract(_phase1_contract())
    historical = deepcopy(current)
    historical["schemas"]["physics"] = "PhysicsV4"
    del historical["task"]["physics"]["velocity_limit_policy"]
    historical["contract_hash"] = stable_contract_hash(
        {key: value for key, value in historical.items() if key != "contract_hash"}
    )
    with pytest.raises(DreamWaQContractMismatchError, match="does not match the current runtime"):
        validate_named_contract(historical, current, BASE_TASK_CONTRACT_VERSION)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("obs_groups",), {"policy": ["policy", "policy_history"], "critic": ["critic"]}),
        (("policy", "history_length"), 4),
        (("algorithm", "velocity_coef"), 0.5),
        (("algorithm", "num_mini_batches"), 0),
    ],
)
def test_algorithm_contract_rejects_frozen_field_drift(path: tuple[str, ...], value: object) -> None:
    config = _runner_config()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(DreamWaQContractMismatchError):
        build_dreamwaq_algorithm_contract(config)


def test_export_contract_freezes_torchscript_golden_vector_acceptance() -> None:
    contract = build_dreamwaq_export_contract()

    assert contract["manifest_version"] == DREAMWAQ_EXPORT_CONTRACT_VERSION
    assert contract["golden_vectors"]["seed"] == GOLDEN_VECTOR_SEED
    assert contract["golden_vectors"]["shape"] == list(GOLDEN_VECTOR_SHAPE)
    assert contract["golden_vectors"]["max_abs_error"] == TORCHSCRIPT_MAX_ABS_ERROR
    assert contract["actor"]["input"]["dynamic_batch"] is True
    validate_named_contract(contract, deepcopy(contract), DREAMWAQ_EXPORT_CONTRACT_VERSION)


def test_named_contract_rejects_tampering_before_comparison() -> None:
    current = build_dreamwaq_export_contract()
    tampered = deepcopy(current)
    tampered["actor"]["output"]["dimension"] = 5

    with pytest.raises(DreamWaQContractMismatchError, match="embedded hash"):
        validate_named_contract(tampered, current, DREAMWAQ_EXPORT_CONTRACT_VERSION)
