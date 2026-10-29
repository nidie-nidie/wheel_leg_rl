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

from wheelleg_dreamwaq.schemas.manifest import (
    HISTORICAL_PHASE1_CONTRACT_VERSION,
    HISTORICAL_RANDOMIZED_PHASE1_CONTRACT_VERSION,
    LEGACY_RANDOMIZED_PHASE1_CONTRACT_VERSION,
    PHASE1_CONTRACT_VERSION,
    ContractMismatchError,
    validate_contract,
)
from wheelleg_dreamwaq.schemas.randomization import (
    CLOSED_CHAIN_RELAXATION_VERSION,
    CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
    CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION,
    validate_closed_chain_reset_cache_artifact,
    validate_randomization_rng_state_payload,
)


CHECKPOINT_METADATA_VERSION = "CheckpointMetadataV6"
HISTORICAL_RANDOMIZED_CHECKPOINT_METADATA_VERSION = "CheckpointMetadataV5"
LEGACY_RANDOMIZED_CHECKPOINT_METADATA_VERSION = "CheckpointMetadataV4"
HISTORICAL_CHECKPOINT_METADATA_VERSION = "CheckpointMetadataV3"
CLOSED_CHAIN_RESET_CACHE_FILENAME = "closed_chain_reset_cache.pt"
CLOSED_CHAIN_CACHE_BINDING_FIELDS = (
    "relative_path",
    "file_sha256",
    "tensor_sha256",
    "schema_version",
    "relaxation_algorithm_version",
    "root_height_algorithm_version",
    "q_reset_shape",
    "q_reset_dtype",
    "root_offset_shape",
    "root_offset_dtype",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def closed_chain_reset_cache_metadata(cache_path: Path, *, run_dir: Path) -> dict[str, Any]:
    """Validate a reset-cache artifact and build its run-manifest metadata."""

    cache_path = cache_path.resolve()
    run_dir = run_dir.resolve()
    try:
        relative_path = cache_path.relative_to(run_dir).as_posix()
    except ValueError as error:
        raise ContractMismatchError("Closed-chain reset cache must be stored inside the run directory") from error
    if relative_path != CLOSED_CHAIN_RESET_CACHE_FILENAME:
        raise ContractMismatchError(
            f"Closed-chain reset cache path must be {CLOSED_CHAIN_RESET_CACHE_FILENAME!r}, got {relative_path!r}"
        )
    if not cache_path.is_file():
        raise ContractMismatchError(f"Closed-chain reset cache is missing: {cache_path}")
    try:
        artifact = validate_closed_chain_reset_cache_artifact(
            torch.load(cache_path, map_location="cpu", weights_only=False)
        )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ContractMismatchError(f"Closed-chain reset cache is invalid: {error}") from error
    q_reset = artifact["q_reset_projected_env"]
    root_height_offset = artifact["root_height_offset_env"]
    return {
        "relative_path": relative_path,
        "file_sha256": sha256_file(cache_path),
        "tensor_sha256": artifact["tensor_sha256"],
        "schema_version": artifact["schema_version"],
        "relaxation_algorithm_version": artifact["algorithm_version"],
        "root_height_algorithm_version": artifact["root_height_algorithm_version"],
        "q_reset_shape": list(q_reset.shape),
        "q_reset_dtype": str(q_reset.dtype),
        "root_offset_shape": list(root_height_offset.shape),
        "root_offset_dtype": str(root_height_offset.dtype),
        "identity": copy.deepcopy(artifact["identity"]),
        "metrics": copy.deepcopy(artifact["metrics"]),
    }


def checkpoint_cache_binding(cache_metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cache_metadata, dict):
        raise ContractMismatchError("Closed-chain reset cache metadata is missing")
    missing = set(CLOSED_CHAIN_CACHE_BINDING_FIELDS) - set(cache_metadata)
    if missing:
        raise ContractMismatchError(f"Closed-chain reset cache metadata is incomplete: {sorted(missing)}")
    return {field: copy.deepcopy(cache_metadata[field]) for field in CLOSED_CHAIN_CACHE_BINDING_FIELDS}


def load_closed_chain_reset_cache_artifact(
    run_manifest_path: Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Load and validate the cache artifact bound to a finalized run manifest."""

    if manifest is None:
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    cache_metadata = manifest.get("closed_chain_reset_cache")
    if not isinstance(cache_metadata, dict):
        raise ContractMismatchError("Run manifest has no closed-chain reset cache metadata")
    relative_path = cache_metadata.get("relative_path")
    if relative_path != CLOSED_CHAIN_RESET_CACHE_FILENAME:
        raise ContractMismatchError("Run manifest closed-chain reset cache path is invalid")
    cache_path = (run_manifest_path.parent / relative_path).resolve()
    if cache_path.parent != run_manifest_path.parent.resolve():
        raise ContractMismatchError("Closed-chain reset cache path escapes the run directory")
    actual_metadata = closed_chain_reset_cache_metadata(cache_path, run_dir=run_manifest_path.parent)
    if actual_metadata != cache_metadata:
        raise ContractMismatchError("Closed-chain reset cache metadata does not match the artifact")
    artifact = validate_closed_chain_reset_cache_artifact(
        torch.load(cache_path, map_location="cpu", weights_only=False)
    )
    return artifact, actual_metadata, cache_path


def validate_cache_identity(
    *,
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    saved_contract: dict[str, Any],
) -> None:
    identity = artifact["identity"]
    expected = {
        "asset_bundle_version": saved_contract.get("asset_bundle_version"),
        "asset_bundle_hash": saved_contract.get("asset_bundle_hash"),
        "physics_schema_version": saved_contract.get("schemas", {}).get("physics"),
        "randomization_profile_hash": saved_contract.get("randomization", {}).get("profile_hash"),
        "master_seed": metadata.get("seed"),
        "num_envs": metadata.get("hardware_profile", {}).get("effective_num_envs"),
    }
    for field, expected_value in expected.items():
        if identity.get(field) != expected_value:
            raise ContractMismatchError(
                f"Closed-chain reset cache identity mismatch for {field}: "
                f"{identity.get(field)!r} != {expected_value!r}"
            )
    audit = manifest.get("randomization_audit")
    if not isinstance(audit, dict):
        raise ContractMismatchError("Run manifest randomization audit is missing")
    if identity.get("realized_plan_hash") != audit.get("realized_plan_hash"):
        raise ContractMismatchError("Closed-chain reset cache realized-plan hash mismatch")
    if identity.get("actuator_plan_hash") != audit.get("actuator_plan_hash"):
        raise ContractMismatchError("Closed-chain reset cache actuator-plan hash mismatch")


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


def optimizer_learning_rates_from_state(optimizer_state: dict[str, Any]) -> list[float]:
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


def require_single_learning_rate(learning_rates: list[float]) -> float:
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
        environment_rng_state_factory: Callable[[], dict[str, Any]],
        **kwargs,
    ) -> None:
        self._checkpoint_metadata_factory = checkpoint_metadata_factory
        self._environment_rng_state_factory = environment_rng_state_factory
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
        optimizer_learning_rate = require_single_learning_rate(optimizer_learning_rates)
        algorithm_learning_rate = float(self.alg.learning_rate)
        if optimizer_learning_rate != algorithm_learning_rate:
            raise RuntimeError(
                "PPO algorithm learning rate does not match its optimizer before checkpoint save: "
                f"{algorithm_learning_rate} != {optimizer_learning_rate}"
            )
        checkpoint_infos["optimizer_learning_rates"] = optimizer_learning_rates
        checkpoint_infos["algorithm_learning_rate"] = algorithm_learning_rate
        checkpoint_infos["rng_state"] = capture_rng_state()
        checkpoint_infos["environment_rng_state"] = self._environment_rng_state_factory()
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
    optimizer_learning_rates = optimizer_learning_rates_from_state(optimizer_state)
    metadata = payload.get("infos")
    if not isinstance(metadata, dict):
        raise ContractMismatchError("Checkpoint has no WheelLeg metadata dictionary")

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
    expected_metadata_version = {
        HISTORICAL_PHASE1_CONTRACT_VERSION: HISTORICAL_CHECKPOINT_METADATA_VERSION,
        HISTORICAL_RANDOMIZED_PHASE1_CONTRACT_VERSION: HISTORICAL_RANDOMIZED_CHECKPOINT_METADATA_VERSION,
        LEGACY_RANDOMIZED_PHASE1_CONTRACT_VERSION: LEGACY_RANDOMIZED_CHECKPOINT_METADATA_VERSION,
        PHASE1_CONTRACT_VERSION: CHECKPOINT_METADATA_VERSION,
    }.get(saved_contract.get("manifest_version"))
    if expected_metadata_version is None or metadata.get("metadata_version") != expected_metadata_version:
        raise ContractMismatchError(
            f"Checkpoint metadata version is {metadata.get('metadata_version')!r}, "
            f"expected {expected_metadata_version!r} for contract {saved_contract.get('manifest_version')!r}"
        )
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
    optimizer_learning_rate = require_single_learning_rate(optimizer_learning_rates)
    algorithm_learning_rate = metadata.get("algorithm_learning_rate")
    if not isinstance(algorithm_learning_rate, (int, float)) or float(algorithm_learning_rate) != optimizer_learning_rate:
        raise ContractMismatchError("Checkpoint adaptive learning-rate state is missing or inconsistent")

    rng_state = metadata.get("rng_state")
    if not isinstance(rng_state, dict) or set(rng_state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ContractMismatchError("Checkpoint RNG state is missing or incomplete")
    if expected_metadata_version == CHECKPOINT_METADATA_VERSION:
        artifact, cache_metadata, _ = load_closed_chain_reset_cache_artifact(
            run_manifest_path,
            manifest=manifest,
        )
        expected_cache_binding = checkpoint_cache_binding(cache_metadata)
        if metadata.get("closed_chain_reset_cache") != expected_cache_binding:
            raise ContractMismatchError(
                "Checkpoint closed-chain reset cache binding does not match its run manifest"
            )
        if cache_metadata["schema_version"] != CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION:
            raise ContractMismatchError("Checkpoint closed-chain reset cache schema is invalid")
        if cache_metadata["relaxation_algorithm_version"] != CLOSED_CHAIN_RELAXATION_VERSION:
            raise ContractMismatchError("Checkpoint closed-chain reset cache algorithm is invalid")
        if cache_metadata["root_height_algorithm_version"] != CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION:
            raise ContractMismatchError("Checkpoint closed-chain reset cache root-height algorithm is invalid")
        validate_cache_identity(
            artifact=artifact,
            manifest=manifest,
            metadata=metadata,
            saved_contract=saved_contract,
        )
        environment_rng_state = metadata.get("environment_rng_state")
        try:
            validate_randomization_rng_state_payload(environment_rng_state)
        except (ValueError, RuntimeError, TypeError) as error:
            raise ContractMismatchError(f"Checkpoint environment RNG state is invalid: {error}") from error
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
    restore_environment_rng_state: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    if metadata.get("metadata_version") != CHECKPOINT_METADATA_VERSION:
        raise ContractMismatchError(
            f"Only {CHECKPOINT_METADATA_VERSION} checkpoints can resume Phase 1R V3 training"
        )
    runner.load(str(checkpoint_path), load_optimizer=True, map_location=map_location)
    restored_learning_rates = [float(group["lr"]) for group in runner.alg.optimizer.param_groups]
    restored_learning_rate = require_single_learning_rate(restored_learning_rates)
    if restored_learning_rates != metadata["optimizer_learning_rates"]:
        raise ContractMismatchError("Restored optimizer learning rates do not match checkpoint metadata")
    runner.alg.learning_rate = restored_learning_rate
    completed_iterations = int(metadata["completed_iterations"])
    runner.current_learning_iteration = completed_iterations
    runner.tot_timesteps = int(metadata["total_timesteps"])
    runner.tot_time = float(metadata["total_time_seconds"])
    restore_rng_state(metadata["rng_state"])
    if "environment_rng_state" in metadata:
        if restore_environment_rng_state is None:
            raise ContractMismatchError("Environment RNG restore callback is required for this checkpoint")
        restore_environment_rng_state(metadata["environment_rng_state"])
    return completed_iterations


# Compatibility aliases for historical internal imports. New code uses the public names above.
_validate_cache_identity = validate_cache_identity
_optimizer_learning_rates_from_state = optimizer_learning_rates_from_state
_require_single_learning_rate = require_single_learning_rate
