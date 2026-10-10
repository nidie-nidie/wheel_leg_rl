from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2
from wheelleg_dreamwaq.schemas.manifest import ContractMismatchError
from wheelleg_dreamwaq.training.checkpoint import (
    CLOSED_CHAIN_RESET_CACHE_FILENAME,
    CHECKPOINT_METADATA_VERSION,
    capture_rng_state,
    checkpoint_cache_binding,
    closed_chain_reset_cache_metadata,
    expose_episode_info_key_union,
    resume_runner_from_checkpoint,
    restore_rng_state,
    sha256_file,
    validate_checkpoint_metadata,
)

from wheelleg_dreamwaq.schemas.manifest import build_phase1_contract
from wheelleg_dreamwaq.schemas.normalization import NormalizationV2
from wheelleg_dreamwaq.schemas.physics import (
    DECIMATION,
    SIM_DT_S,
    SOLVER_POSITION_ITERATIONS,
    SOLVER_VELOCITY_ITERATIONS,
)
from wheelleg_dreamwaq.schemas.action import CANONICAL_JOINT_ORDER
from wheelleg_dreamwaq.schemas.randomization import FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
from wheelleg_dreamwaq.schemas.randomization import (
    CLOSED_CHAIN_NOMINAL_BRANCH_SIGNATURE,
    CLOSED_CHAIN_PHYSICAL_PASSIVE_BRANCH_JOINTS,
    CLOSED_CHAIN_RELAXATION_VERSION,
    CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
    CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION,
    RANDOMIZATION_SEED_DERIVATION_VERSION,
    RANDOMIZATION_STREAMS,
    canonical_tensor_sha256,
    closed_chain_reset_contract_payload,
    profile_contract_hash,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import CommandRanges
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import ControlLimits
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RewardWeights
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.terminations import TerminationLimits


TEST_PASSIVE_JOINT_NAMES = tuple(f"passive_{index}" for index in range(20))


def _contract() -> dict:
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
        training={"runner": {"num_steps_per_env": 24}, "algorithm": {"gamma": 0.99}},
        runtime={"collision_policy": "test"},
    )


def _write_checkpoint(tmp_path: Path) -> tuple[Path, Path, dict]:
    contract = _contract()
    num_envs = 32
    q_reset = torch.zeros(
        (num_envs, len(CANONICAL_JOINT_ORDER) + len(TEST_PASSIVE_JOINT_NAMES)), dtype=torch.float32
    )
    root_height_offset = torch.linspace(-0.003, 0.007, num_envs, dtype=torch.float32)
    actuator_plan = {
        "active_joint_realized_stiffness": torch.ones((num_envs, 6), dtype=torch.float32),
        "active_joint_realized_damping": torch.ones((num_envs, 6), dtype=torch.float32),
        "active_joint_realized_effort_limit": torch.ones((num_envs, 6), dtype=torch.float32),
    }
    actuator_plan_hash = canonical_tensor_sha256(actuator_plan)
    realized_plan_hash = "A" * 64
    artifact = {
        "schema_version": CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
        "algorithm_version": CLOSED_CHAIN_RELAXATION_VERSION,
        "root_height_algorithm_version": CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION,
        "contract": closed_chain_reset_contract_payload(sim_dt=SIM_DT_S),
        "identity": {
            "asset_bundle_version": ASSET_BUNDLE_V2.version,
            "asset_bundle_hash": ASSET_BUNDLE_V2.bundle_hash,
            "physics_schema_version": contract["schemas"]["physics"],
            "randomization_profile_hash": profile_contract_hash(FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1),
            "realized_plan_hash": realized_plan_hash,
            "actuator_plan_hash": actuator_plan_hash,
            "master_seed": 42,
            "num_envs": num_envs,
            "joint_names": list(CANONICAL_JOINT_ORDER) + list(TEST_PASSIVE_JOINT_NAMES),
            "passive_joint_names": list(TEST_PASSIVE_JOINT_NAMES),
            "physical_passive_branch_joint_names": list(CLOSED_CHAIN_PHYSICAL_PASSIVE_BRANCH_JOINTS),
        },
        "q_reset_projected_env": q_reset,
        "root_height_offset_env": root_height_offset,
        "tensor_sha256": canonical_tensor_sha256(
            {
                "q_reset_projected_env": q_reset,
                "root_height_offset_env": root_height_offset,
            }
        ),
        "actuator_plan": actuator_plan,
        "metrics": {
            "physical_passive_branch_signature": [list(CLOSED_CHAIN_NOMINAL_BRANCH_SIGNATURE)] * num_envs,
        },
        "trace": [],
    }
    cache_path = tmp_path / CLOSED_CHAIN_RESET_CACHE_FILENAME
    torch.save(artifact, cache_path)
    cache_metadata = closed_chain_reset_cache_metadata(cache_path, run_dir=tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "contract": contract,
                "seed": 42,
                "num_envs": num_envs,
                "randomization_audit": {
                    "realized_plan_hash": realized_plan_hash,
                    "actuator_plan_hash": actuator_plan_hash,
                },
                "closed_chain_reset_cache": cache_metadata,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    optimizer_state = optimizer.state_dict()
    environment_rng_state = {
        "seed_derivation_version": RANDOMIZATION_SEED_DERIVATION_VERSION,
        "streams": {},
    }
    for index, stream in enumerate(RANDOMIZATION_STREAMS):
        generator = torch.Generator(device="cpu").manual_seed(100 + index)
        environment_rng_state["streams"][stream] = {
            "device": "cpu",
            "state": generator.get_state(),
        }
    metadata = {
        "metadata_version": CHECKPOINT_METADATA_VERSION,
        "run_manifest_sha256": sha256_file(manifest_path),
        "contract_hash": contract["contract_hash"],
        "closed_chain_reset_cache": checkpoint_cache_binding(cache_metadata),
        "seed": 42,
        "hardware_profile": {
            "name": "portable",
            "declared_num_envs": 32,
            "effective_num_envs": num_envs,
            "num_mini_batches": 4,
            "num_steps_per_env": 24,
            "world_size": 1,
            "global_rank": 0,
            "local_rank": 0,
        },
        "runner_iteration": 1,
        "completed_iterations": 2,
        "total_timesteps": 1536,
        "total_time_seconds": 1.25,
        "optimizer_learning_rates": [2.5e-4],
        "algorithm_learning_rate": 2.5e-4,
        "rng_state": capture_rng_state(),
        "environment_rng_state": environment_rng_state,
    }
    checkpoint_path = tmp_path / "model_1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "iter": 1,
            "infos": metadata,
        },
        checkpoint_path,
    )
    return checkpoint_path, manifest_path, contract


def test_checkpoint_metadata_binds_checkpoint_to_run_manifest(tmp_path: Path) -> None:
    checkpoint_path, manifest_path, contract = _write_checkpoint(tmp_path)

    metadata = validate_checkpoint_metadata(checkpoint_path, manifest_path, contract)

    assert metadata["seed"] == 42
    assert metadata["contract_hash"] == contract["contract_hash"]
    assert metadata["completed_iterations"] == 2
    assert metadata["algorithm_learning_rate"] == pytest.approx(2.5e-4)
    assert set(metadata["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert set(metadata["environment_rng_state"]["streams"]) == set(RANDOMIZATION_STREAMS)
    assert metadata["closed_chain_reset_cache"]["q_reset_shape"] == [32, 26]
    assert metadata["closed_chain_reset_cache"]["root_offset_shape"] == [32]
    assert metadata["closed_chain_reset_cache"]["root_height_algorithm_version"] == (
        CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION
    )


def test_checkpoint_metadata_rejects_tampered_run_manifest(tmp_path: Path) -> None:
    checkpoint_path, manifest_path, contract = _write_checkpoint(tmp_path)
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ContractMismatchError, match="run manifest hash"):
        validate_checkpoint_metadata(checkpoint_path, manifest_path, contract)


def test_checkpoint_metadata_rejects_tampered_reset_cache(tmp_path: Path) -> None:
    checkpoint_path, manifest_path, contract = _write_checkpoint(tmp_path)
    cache_path = tmp_path / CLOSED_CHAIN_RESET_CACHE_FILENAME
    with cache_path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ContractMismatchError, match="cache metadata"):
        validate_checkpoint_metadata(checkpoint_path, manifest_path, contract)


def test_checkpoint_metadata_rejects_runtime_contract_drift(tmp_path: Path) -> None:
    checkpoint_path, manifest_path, contract = _write_checkpoint(tmp_path)
    current = deepcopy(contract)
    current["task"]["q_nominal"][0] = -0.4
    current["contract_hash"] = "0" * 64

    with pytest.raises(ContractMismatchError):
        validate_checkpoint_metadata(checkpoint_path, manifest_path, current)


def test_restore_rng_state_replays_python_numpy_and_torch_streams() -> None:
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(()).item()))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), float(torch.rand(()).item()))

    assert actual == pytest.approx(expected)


def test_checkpoint_metadata_rejects_off_by_one_completed_iteration(tmp_path: Path) -> None:
    checkpoint_path, manifest_path, contract = _write_checkpoint(tmp_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["infos"]["completed_iterations"] = 1
    torch.save(payload, checkpoint_path)

    with pytest.raises(ContractMismatchError, match="completed iteration"):
        validate_checkpoint_metadata(checkpoint_path, manifest_path, contract)


def test_checkpoint_metadata_rejects_missing_optimizer_state(tmp_path: Path) -> None:
    checkpoint_path, manifest_path, contract = _write_checkpoint(tmp_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload.pop("optimizer_state_dict")
    torch.save(payload, checkpoint_path)

    with pytest.raises(ContractMismatchError, match="optimizer state"):
        validate_checkpoint_metadata(checkpoint_path, manifest_path, contract)


def test_resume_restores_adaptive_learning_rate_from_optimizer(tmp_path: Path) -> None:
    checkpoint_path, manifest_path, contract = _write_checkpoint(tmp_path)
    metadata = validate_checkpoint_metadata(checkpoint_path, manifest_path, contract)

    class FakeRunner:
        def __init__(self) -> None:
            self.model = torch.nn.Linear(2, 1)
            self.alg = SimpleNamespace(
                optimizer=torch.optim.Adam(self.model.parameters(), lr=1.0e-3),
                learning_rate=1.0e-3,
            )
            self.current_learning_iteration = 0
            self.tot_timesteps = 0
            self.tot_time = 0.0

        def load(self, path: str, load_optimizer: bool, map_location: str) -> dict:
            payload = torch.load(path, map_location=map_location, weights_only=False)
            self.model.load_state_dict(payload["model_state_dict"])
            if load_optimizer:
                self.alg.optimizer.load_state_dict(payload["optimizer_state_dict"])
            return payload["infos"]

    runner = FakeRunner()
    restored_environment_rng_state = []
    completed = resume_runner_from_checkpoint(
        runner,
        checkpoint_path,
        metadata,
        map_location="cpu",
        restore_environment_rng_state=restored_environment_rng_state.append,
    )

    assert completed == 2
    assert runner.current_learning_iteration == 2
    assert runner.alg.learning_rate == pytest.approx(2.5e-4)
    assert runner.alg.optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)
    assert restored_environment_rng_state[0] is metadata["environment_rng_state"]


def test_checkpoint_metadata_rejects_missing_environment_rng_state(tmp_path: Path) -> None:
    checkpoint_path, manifest_path, contract = _write_checkpoint(tmp_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["infos"].pop("environment_rng_state")
    torch.save(payload, checkpoint_path)

    with pytest.raises(ContractMismatchError, match="environment RNG state"):
        validate_checkpoint_metadata(checkpoint_path, manifest_path, contract)


@pytest.mark.parametrize("missing_stream", RANDOMIZATION_STREAMS)
def test_checkpoint_metadata_rejects_any_missing_environment_rng_stream(
    tmp_path: Path,
    missing_stream: str,
) -> None:
    checkpoint_path, manifest_path, contract = _write_checkpoint(tmp_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["infos"]["environment_rng_state"]["streams"].pop(missing_stream)
    torch.save(payload, checkpoint_path)

    with pytest.raises(ContractMismatchError, match="streams are missing or incomplete"):
        validate_checkpoint_metadata(checkpoint_path, manifest_path, contract)


def test_checkpoint_metadata_rejects_unusable_environment_rng_state(tmp_path: Path) -> None:
    checkpoint_path, manifest_path, contract = _write_checkpoint(tmp_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["infos"]["environment_rng_state"]["streams"]["material_rng"]["state"] = torch.tensor(
        [1], dtype=torch.uint8
    )
    torch.save(payload, checkpoint_path)

    with pytest.raises(ContractMismatchError, match="material_rng"):
        validate_checkpoint_metadata(checkpoint_path, manifest_path, contract)


def test_episode_logging_exposes_keys_that_first_appeared_after_rollout_step_zero() -> None:
    infos = [
        {"Reward/tracking_vx": torch.tensor(1.0)},
        {"Reward/tracking_vx": torch.tensor(2.0), "Episode/tracking_vx": torch.tensor(3.0)},
    ]
    patched = expose_episode_info_key_union(infos, device="cpu")

    assert set(patched[0]) == {"Reward/tracking_vx", "Episode/tracking_vx"}
    assert patched[0]["Episode/tracking_vx"].numel() == 0
    assert patched[1] is infos[1]
