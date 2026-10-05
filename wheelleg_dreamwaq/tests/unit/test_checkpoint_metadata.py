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
    CHECKPOINT_METADATA_VERSION,
    capture_rng_state,
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
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import CommandRanges
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import ControlLimits
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RewardWeights
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.terminations import TerminationLimits


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
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps({"contract": contract}, sort_keys=True), encoding="utf-8")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    optimizer_state = optimizer.state_dict()
    metadata = {
        "metadata_version": CHECKPOINT_METADATA_VERSION,
        "run_manifest_sha256": sha256_file(manifest_path),
        "contract_hash": contract["contract_hash"],
        "seed": 42,
        "hardware_profile": {
            "name": "portable",
            "declared_num_envs": 32,
            "effective_num_envs": 32,
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


def test_checkpoint_metadata_rejects_tampered_run_manifest(tmp_path: Path) -> None:
    checkpoint_path, manifest_path, contract = _write_checkpoint(tmp_path)
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ContractMismatchError, match="run manifest hash"):
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
    completed = resume_runner_from_checkpoint(runner, checkpoint_path, metadata, map_location="cpu")

    assert completed == 2
    assert runner.current_learning_iteration == 2
    assert runner.alg.learning_rate == pytest.approx(2.5e-4)
    assert runner.alg.optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)


def test_episode_logging_exposes_keys_that_first_appeared_after_rollout_step_zero() -> None:
    infos = [
        {"Reward/tracking_vx": torch.tensor(1.0)},
        {"Reward/tracking_vx": torch.tensor(2.0), "Episode/tracking_vx": torch.tensor(3.0)},
    ]
    patched = expose_episode_info_key_union(infos, device="cpu")

    assert set(patched[0]) == {"Reward/tracking_vx", "Episode/tracking_vx"}
    assert patched[0]["Episode/tracking_vx"].numel() == 0
    assert patched[1] is infos[1]
