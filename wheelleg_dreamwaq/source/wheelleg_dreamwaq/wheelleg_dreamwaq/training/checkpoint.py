from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from wheelleg_dreamwaq.schemas.manifest import ContractMismatchError, validate_contract


CHECKPOINT_METADATA_VERSION = "CheckpointMetadataV3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    expected = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, dict) or set(state) != expected:
        raise ContractMismatchError("Checkpoint RNG state is missing or incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = state["torch_cuda"]
        if not isinstance(cuda_state, list) or not cuda_state:
            raise ContractMismatchError("Checkpoint CUDA RNG state is missing")
        if len(cuda_state) != torch.cuda.device_count():
            raise ContractMismatchError(
                f"Checkpoint CUDA RNG device count {len(cuda_state)} does not match runtime {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def expose_episode_info_key_union(
    episode_infos: list[dict[str, Any]],
    *,
    device: str | torch.device,
) -> list[dict[str, Any]]:
    if not episode_infos:
        return episode_infos
    all_keys = set().union(*(info.keys() for info in episode_infos))
    if all_keys.issubset(episode_infos[0]):
        return episode_infos
    first = dict(episode_infos[0])
    for key in all_keys - set(first):
        first[key] = torch.empty(0, device=device)
    return [first, *episode_infos[1:]]


def _optimizer_learning_rates_from_state(optimizer_state: dict[str, Any]) -> list[float]:
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise ContractMismatchError("Checkpoint optimizer parameter groups are missing")
    learning_rates: list[float] = []
    for group in param_groups:
        if not isinstance(group, dict) or not isinstance(group.get("lr"), (int, float)):
            raise ContractMismatchError("Checkpoint optimizer learning rate is missing")
        learning_rate = float(group["lr"])
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ContractMismatchError("Checkpoint optimizer learning rate is invalid")
        learning_rates.append(learning_rate)
    return learning_rates


def _require_single_learning_rate(learning_rates: list[float]) -> float:
    first = learning_rates[0]
    if any(value != first for value in learning_rates[1:]):
        raise ContractMismatchError("PPO optimizer parameter groups use different learning rates")
    return first


class ContractOnPolicyRunner(OnPolicyRunner):
    """RSL-RL runner that attaches the frozen run contract to every checkpoint."""

    def __init__(
        self,
        *args,
        checkpoint_metadata_factory: Callable[[], dict[str, Any]],
        **kwargs,
    ) -> None:
        self._checkpoint_metadata_factory = checkpoint_metadata_factory
        super().__init__(*args, **kwargs)

    def save(self, path: str, infos: dict | None = None) -> None:
        checkpoint_infos = copy.deepcopy(self._checkpoint_metadata_factory())
        checkpoint_infos["metadata_version"] = CHECKPOINT_METADATA_VERSION
        checkpoint_infos["saved_at"] = datetime.now().astimezone().isoformat()
        checkpoint_infos["runner_iteration"] = self.current_learning_iteration
        checkpoint_infos["completed_iterations"] = self.current_learning_iteration + 1
        checkpoint_infos["total_timesteps"] = self.tot_timesteps
        checkpoint_infos["total_time_seconds"] = self.tot_time
        optimizer_learning_rates = [float(group["lr"]) for group in self.alg.optimizer.param_groups]
        optimizer_learning_rate = _require_single_learning_rate(optimizer_learning_rates)
        algorithm_learning_rate = float(self.alg.learning_rate)
        if optimizer_learning_rate != algorithm_learning_rate:
            raise RuntimeError(
                "PPO algorithm learning rate does not match its optimizer before checkpoint save: "
                f"{algorithm_learning_rate} != {optimizer_learning_rate}"
            )
        checkpoint_infos["optimizer_learning_rates"] = optimizer_learning_rates
        checkpoint_infos["algorithm_learning_rate"] = algorithm_learning_rate
        checkpoint_infos["rng_state"] = capture_rng_state()
        if infos is not None:
            checkpoint_infos["rsl_rl_infos"] = infos
        super().save(path, infos=checkpoint_infos)

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        patched_locs = dict(locs)
        patched_locs["ep_infos"] = expose_episode_info_key_union(
            locs.get("ep_infos", []),
            device=self.device,
        )
        super().log(patched_locs, width=width, pad=pad)


def validate_checkpoint_metadata(
    checkpoint_path: Path,
    run_manifest_path: Path,
    current_contract: dict[str, Any],
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not run_manifest_path.is_file():
        raise ContractMismatchError(f"Checkpoint run manifest is missing: {run_manifest_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("model_state_dict"), dict) or not payload["model_state_dict"]:
        raise ContractMismatchError("Checkpoint model state is missing or invalid")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict) or not isinstance(optimizer_state.get("state"), dict):
        raise ContractMismatchError("Checkpoint optimizer state is missing or invalid")
    optimizer_learning_rates = _optimizer_learning_rates_from_state(optimizer_state)
    metadata = payload.get("infos")
    if not isinstance(metadata, dict):
        raise ContractMismatchError("Checkpoint has no WheelLeg metadata dictionary")
    if metadata.get("metadata_version") != CHECKPOINT_METADATA_VERSION:
        raise ContractMismatchError(
            f"Checkpoint metadata version is {metadata.get('metadata_version')!r}, "
            f"expected {CHECKPOINT_METADATA_VERSION!r}"
        )

    expected_manifest_hash = metadata.get("run_manifest_sha256")
    actual_manifest_hash = sha256_file(run_manifest_path)
    if expected_manifest_hash != actual_manifest_hash:
        raise ContractMismatchError(
            f"Checkpoint run manifest hash mismatch: {expected_manifest_hash!r} != {actual_manifest_hash!r}"
        )

    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    saved_contract = manifest.get("contract")
    if not isinstance(saved_contract, dict):
        raise ContractMismatchError("Run manifest does not contain a contract dictionary")
    validate_contract(saved_contract, current_contract)
    if metadata.get("contract_hash") != saved_contract["contract_hash"]:
        raise ContractMismatchError("Checkpoint contract hash does not match its run manifest")

    hardware_profile = metadata.get("hardware_profile")
    required_hardware_fields = {
        "name",
        "declared_num_envs",
        "effective_num_envs",
        "num_mini_batches",
        "num_steps_per_env",
        "world_size",
        "global_rank",
        "local_rank",
    }
    if not isinstance(hardware_profile, dict) or set(hardware_profile) != required_hardware_fields:
        raise ContractMismatchError("Checkpoint hardware profile metadata is missing or incomplete")
    if any(
        not isinstance(hardware_profile[field], int) or hardware_profile[field] <= 0
        for field in ("declared_num_envs", "effective_num_envs", "num_mini_batches", "num_steps_per_env", "world_size")
    ):
        raise ContractMismatchError("Checkpoint hardware profile contains invalid positive integer fields")
    if hardware_profile["global_rank"] != 0 or hardware_profile["local_rank"] != 0:
        raise ContractMismatchError("Checkpoint hardware profile is not a supported single-process run")

    metadata_optimizer_lrs = metadata.get("optimizer_learning_rates")
    if metadata_optimizer_lrs != optimizer_learning_rates:
        raise ContractMismatchError("Checkpoint optimizer learning-rate metadata does not match optimizer state")
    optimizer_learning_rate = _require_single_learning_rate(optimizer_learning_rates)
    algorithm_learning_rate = metadata.get("algorithm_learning_rate")
    if not isinstance(algorithm_learning_rate, (int, float)) or float(algorithm_learning_rate) != optimizer_learning_rate:
        raise ContractMismatchError("Checkpoint adaptive learning-rate state is missing or inconsistent")

    rng_state = metadata.get("rng_state")
    if not isinstance(rng_state, dict) or set(rng_state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ContractMismatchError("Checkpoint RNG state is missing or incomplete")
    checkpoint_iteration = payload.get("iter")
    if not isinstance(checkpoint_iteration, int):
        raise ContractMismatchError("Checkpoint iteration is missing or invalid")
    if metadata.get("runner_iteration") != checkpoint_iteration:
        raise ContractMismatchError("Checkpoint metadata runner iteration does not match checkpoint payload")
    if metadata.get("completed_iterations") != checkpoint_iteration + 1:
        raise ContractMismatchError("Checkpoint completed iteration count is inconsistent")
    if not isinstance(metadata.get("total_timesteps"), int) or metadata["total_timesteps"] < 0:
        raise ContractMismatchError("Checkpoint total timesteps are missing or invalid")
    if not isinstance(metadata.get("total_time_seconds"), (int, float)) or metadata["total_time_seconds"] < 0:
        raise ContractMismatchError("Checkpoint total training time is missing or invalid")
    return metadata


def load_checkpoint_artifact(
    checkpoint_path: Path,
    run_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load a checkpoint after validating it against its own frozen run contract."""

    manifest_path = run_manifest_path or checkpoint_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ContractMismatchError(f"Checkpoint run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise ContractMismatchError("Run manifest does not contain a contract dictionary")
    metadata = validate_checkpoint_metadata(checkpoint_path, manifest_path, contract)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return payload, manifest, metadata


def resume_runner_from_checkpoint(
    runner: OnPolicyRunner,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    *,
    map_location: str,
) -> int:
    runner.load(str(checkpoint_path), load_optimizer=True, map_location=map_location)
    restored_learning_rates = [float(group["lr"]) for group in runner.alg.optimizer.param_groups]
    restored_learning_rate = _require_single_learning_rate(restored_learning_rates)
    if restored_learning_rates != metadata["optimizer_learning_rates"]:
        raise ContractMismatchError("Restored optimizer learning rates do not match checkpoint metadata")
    runner.alg.learning_rate = restored_learning_rate
    completed_iterations = int(metadata["completed_iterations"])
    runner.current_learning_iteration = completed_iterations
    runner.tot_timesteps = int(metadata["total_timesteps"])
    runner.tot_time = float(metadata["total_time_seconds"])
    restore_rng_state(metadata["rng_state"])
    return completed_iterations
