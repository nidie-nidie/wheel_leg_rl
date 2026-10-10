from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from rsl_rl.runners import OnPolicyRunner

from wheelleg_dreamwaq.algorithms.dreamwaq.actor_critic import (
    EXPECTED_OBS_GROUPS,
    DreamWaQActorCritic,
)
from wheelleg_dreamwaq.algorithms.dreamwaq.ppo import DreamWaQPPO
from wheelleg_dreamwaq.algorithms.dreamwaq.registration import register_dreamwaq_rsl_rl_classes
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
    BASE_TASK_CONTRACT_VERSION,
    DREAMWAQ_ALGORITHM_CONTRACT_VERSION,
    DREAMWAQ_EXPORT_CONTRACT_VERSION,
    EstimatorMonitorStateV1,
    validate_named_contract,
)
from wheelleg_dreamwaq.schemas.manifest import ContractMismatchError
from wheelleg_dreamwaq.schemas.randomization import (
    PROCESS_START_RANDOMIZATION_STREAMS,
    RANDOMIZATION_STREAMS,
    RUNTIME_RANDOMIZATION_STREAMS,
    validate_randomization_rng_state_payload,
)
from wheelleg_dreamwaq.training.checkpoint import (
    ContractOnPolicyRunner,
    capture_rng_state,
    checkpoint_cache_binding,
    expose_episode_info_key_union,
    load_closed_chain_reset_cache_artifact,
    optimizer_learning_rates_from_state,
    require_single_learning_rate,
    restore_rng_state,
    sha256_file,
    validate_cache_identity,
)


DREAMWAQ_CHECKPOINT_METADATA_VERSION = "DreamWaQCheckpointMetadataV1"
DREAMWAQ_RUNNER_CLASS = "DreamWaQContractOnPolicyRunner"
RESUME_METADATA_VERSION = "ResumeMetadataV2"
FRESH_RESUME_PROVENANCE = {"schema_version": RESUME_METADATA_VERSION, "mode": "fresh"}

RESUME_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "cache_resume_mode",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_run_manifest",
        "source_run_manifest_sha256",
        "source_completed_iterations",
        "source_hardware_profile_name",
        "target_hardware_profile_name",
        "source_num_envs",
        "target_num_envs",
        "same_num_envs",
        "source_cache_file_sha256",
        "source_cache_tensor_sha256",
        "target_cache_file_sha256",
        "target_cache_tensor_sha256",
        "source_realized_plan_hash",
        "target_realized_plan_hash",
        "rng_stream_sources",
    }
)

DREAMWAQ_METADATA_FIELDS = frozenset(
    {
        "metadata_version",
        "saved_at",
        "runner_class",
        "seed",
        "runner_iteration",
        "completed_iterations",
        "total_timesteps",
        "total_time_seconds",
        "optimizer_learning_rates",
        "algorithm_learning_rate",
        "rng_state",
        "environment_rng_state",
        "estimator_monitor_state",
        "run_manifest_sha256",
        "base_task_contract_version",
        "base_task_contract_hash",
        "dreamwaq_algorithm_contract_version",
        "dreamwaq_algorithm_contract_hash",
        "dreamwaq_export_contract_version",
        "dreamwaq_export_contract_hash",
        "hardware_profile",
        "reset_cache_binding",
        "resume_provenance",
    }
)

DREAMWAQ_DYNAMIC_METADATA_FIELDS = frozenset(
    {
        "metadata_version",
        "saved_at",
        "runner_class",
        "runner_iteration",
        "completed_iterations",
        "total_timesteps",
        "total_time_seconds",
        "optimizer_learning_rates",
        "algorithm_learning_rate",
        "rng_state",
        "environment_rng_state",
        "estimator_monitor_state",
    }
)

HARDWARE_PROFILE_FIELDS = frozenset(
    {
        "name",
        "declared_num_envs",
        "effective_num_envs",
        "num_mini_batches",
        "num_steps_per_env",
        "world_size",
        "global_rank",
        "local_rank",
    }
)


def _require_upper_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.upper():
        raise ContractMismatchError(f"{label} must be an uppercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ContractMismatchError(f"{label} must be an uppercase SHA256") from error
    return value


def fresh_resume_provenance() -> dict[str, str]:
    return dict(FRESH_RESUME_PROVENANCE)


def validate_resume_provenance(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContractMismatchError("DreamWaQ resume provenance must be a dictionary")
    mode = payload.get("mode")
    if mode == "fresh":
        if payload != FRESH_RESUME_PROVENANCE:
            raise ContractMismatchError("Fresh resume provenance may contain only schema_version and mode")
        return copy.deepcopy(payload)
    if mode != "weights_optimizer_rng_new_environment" or set(payload) != RESUME_PROVENANCE_FIELDS:
        raise ContractMismatchError("DreamWaQ resume provenance fields or mode are invalid")
    if payload.get("schema_version") != RESUME_METADATA_VERSION:
        raise ContractMismatchError("DreamWaQ resume provenance schema version is invalid")
    for field in (
        "source_checkpoint",
        "source_run_manifest",
        "source_hardware_profile_name",
        "target_hardware_profile_name",
    ):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ContractMismatchError(f"DreamWaQ resume provenance {field} is invalid")
    for field in (
        "source_checkpoint_sha256",
        "source_run_manifest_sha256",
        "source_cache_file_sha256",
        "source_cache_tensor_sha256",
        "target_cache_file_sha256",
        "target_cache_tensor_sha256",
        "source_realized_plan_hash",
        "target_realized_plan_hash",
    ):
        _require_upper_sha256(payload[field], f"resume provenance {field}")
    for field in ("source_completed_iterations", "source_num_envs", "target_num_envs"):
        if not isinstance(payload[field], int) or isinstance(payload[field], bool) or payload[field] <= 0:
            raise ContractMismatchError(f"DreamWaQ resume provenance {field} must be a positive integer")
    if not isinstance(payload["same_num_envs"], bool):
        raise ContractMismatchError("DreamWaQ resume provenance same_num_envs must be bool")
    if payload["same_num_envs"] != (payload["source_num_envs"] == payload["target_num_envs"]):
        raise ContractMismatchError("DreamWaQ resume provenance same_num_envs is inconsistent")
    stream_sources = payload["rng_stream_sources"]
    if not isinstance(stream_sources, dict) or set(stream_sources) != set(RANDOMIZATION_STREAMS):
        raise ContractMismatchError("DreamWaQ resume provenance RNG stream sources are incomplete")
    if payload["same_num_envs"]:
        if payload["cache_resume_mode"] != "reused_source_exactly":
            raise ContractMismatchError("Same-scale DreamWaQ resume must reuse the source cache exactly")
        if any(source != "source_checkpoint" for source in stream_sources.values()):
            raise ContractMismatchError("Same-scale DreamWaQ resume must restore all seven RNG streams")
        if (
            payload["source_cache_file_sha256"] != payload["target_cache_file_sha256"]
            or payload["source_cache_tensor_sha256"] != payload["target_cache_tensor_sha256"]
        ):
            raise ContractMismatchError("Same-scale DreamWaQ resume cache hashes must match")
    else:
        if payload["cache_resume_mode"] != "regenerated_for_target_num_envs":
            raise ContractMismatchError("Cross-scale DreamWaQ resume must regenerate the target cache")
        for stream in PROCESS_START_RANDOMIZATION_STREAMS:
            if stream_sources[stream] != "target_process_start":
                raise ContractMismatchError("Cross-scale DreamWaQ process-start RNG provenance is invalid")
        for stream in RUNTIME_RANDOMIZATION_STREAMS:
            if stream_sources[stream] != "source_checkpoint":
                raise ContractMismatchError("Cross-scale DreamWaQ runtime RNG provenance is invalid")
    return copy.deepcopy(payload)


def _validate_hardware_profile(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != HARDWARE_PROFILE_FIELDS:
        raise ContractMismatchError("DreamWaQ checkpoint hardware profile is missing or incomplete")
    if not isinstance(payload["name"], str) or not payload["name"]:
        raise ContractMismatchError("DreamWaQ checkpoint hardware profile name is invalid")
    for field in ("declared_num_envs", "effective_num_envs", "num_mini_batches", "num_steps_per_env", "world_size"):
        if not isinstance(payload[field], int) or isinstance(payload[field], bool) or payload[field] <= 0:
            raise ContractMismatchError(f"DreamWaQ checkpoint hardware field {field} is invalid")
    if payload["world_size"] != 1 or payload["global_rank"] != 0 or payload["local_rank"] != 0:
        raise ContractMismatchError("DreamWaQ checkpoint only supports a single-process hardware profile")
    return payload


def _validate_model_state_dict(
    state: Any,
    *,
    expected_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict) or not state:
        raise ContractMismatchError("DreamWaQ checkpoint model state is missing or invalid")
    keys = set(state)
    if expected_keys is not None and keys != expected_keys:
        raise ContractMismatchError("DreamWaQ checkpoint model keys differ from the current policy")
    required = ("raw_std", "cenet.encoder.", "cenet.decoder.", "actor.", "critic.")
    for prefix in required:
        if prefix == "raw_std":
            present = prefix in keys
        else:
            present = any(key.startswith(prefix) for key in keys)
        if not present:
            raise ContractMismatchError(f"DreamWaQ checkpoint model state lacks {prefix}")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ContractMismatchError("DreamWaQ checkpoint model state contains invalid tensors")
    return state


def optimizer_parameter_group_signature(optimizer_state: dict[str, Any]) -> list[dict[str, Any]]:
    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ContractMismatchError("DreamWaQ requires exactly one Adam optimizer parameter group")
    signatures = []
    for group in groups:
        parameters = group.get("params")
        if not isinstance(parameters, list) or not parameters:
            raise ContractMismatchError("DreamWaQ optimizer parameter group is empty")
        signature = {
            key: copy.deepcopy(value)
            for key, value in group.items()
            if key not in {"params", "lr", "initial_lr"}
        }
        signature["parameter_count"] = len(parameters)
        signatures.append(signature)
    return signatures


def _validate_optimizer_state(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict) or not payload["state"]:
        raise ContractMismatchError("DreamWaQ checkpoint optimizer state is missing or invalid")
    optimizer_learning_rates_from_state(payload)
    optimizer_parameter_group_signature(payload)
    for parameter_state in payload["state"].values():
        if not isinstance(parameter_state, dict):
            raise ContractMismatchError("DreamWaQ optimizer parameter state is invalid")
        for value in parameter_state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ContractMismatchError("DreamWaQ optimizer state contains NaN or Inf")
    return payload


class DreamWaQContractOnPolicyRunner(ContractOnPolicyRunner):
    """RSL-RL runner with the frozen Phase 2 checkpoint contract."""

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
        *,
        checkpoint_metadata_factory: Callable[[], dict[str, Any]],
        environment_rng_state_factory: Callable[[], dict[str, Any]],
    ) -> None:
        if train_cfg.get("class_name") != DREAMWAQ_RUNNER_CLASS:
            raise ValueError(f"DreamWaQ runner class_name must be {DREAMWAQ_RUNNER_CLASS}")
        if train_cfg.get("obs_groups") != EXPECTED_OBS_GROUPS:
            raise ValueError(f"DreamWaQ obs_groups must equal {EXPECTED_OBS_GROUPS}")
        register_dreamwaq_rsl_rl_classes()
        super().__init__(
            env,
            train_cfg,
            log_dir=log_dir,
            device=device,
            checkpoint_metadata_factory=checkpoint_metadata_factory,
            environment_rng_state_factory=environment_rng_state_factory,
        )
        if not isinstance(self.alg, DreamWaQPPO) or not isinstance(self.alg.policy, DreamWaQActorCritic):
            raise TypeError("DreamWaQContractOnPolicyRunner constructed an unexpected algorithm or policy")

    def save(self, path: str, infos: dict | None = None) -> None:
        checkpoint_infos = copy.deepcopy(self._checkpoint_metadata_factory())
        if set(checkpoint_infos) & DREAMWAQ_DYNAMIC_METADATA_FIELDS:
            raise RuntimeError("DreamWaQ checkpoint metadata factory attempted to set reserved dynamic fields")
        optimizer_learning_rates = [float(group["lr"]) for group in self.alg.optimizer.param_groups]
        optimizer_learning_rate = require_single_learning_rate(optimizer_learning_rates)
        algorithm_learning_rate = float(self.alg.learning_rate)
        if optimizer_learning_rate != algorithm_learning_rate:
            raise RuntimeError("DreamWaQ algorithm and optimizer learning rates differ before checkpoint save")
        completed_iterations = self.current_learning_iteration + 1
        monitor = self.alg.estimator_monitor_state
        if monitor.completed_rollouts != completed_iterations:
            raise RuntimeError(
                "DreamWaQ estimator monitor rollout count differs from completed iterations: "
                f"{monitor.completed_rollouts} != {completed_iterations}"
            )
        checkpoint_infos.update(
            {
                "metadata_version": DREAMWAQ_CHECKPOINT_METADATA_VERSION,
                "saved_at": datetime.now().astimezone().isoformat(),
                "runner_class": DREAMWAQ_RUNNER_CLASS,
                "runner_iteration": self.current_learning_iteration,
                "completed_iterations": completed_iterations,
                "total_timesteps": self.tot_timesteps,
                "total_time_seconds": self.tot_time,
                "optimizer_learning_rates": optimizer_learning_rates,
                "algorithm_learning_rate": algorithm_learning_rate,
                "rng_state": capture_rng_state(),
                "environment_rng_state": self._environment_rng_state_factory(),
                "estimator_monitor_state": monitor.to_dict(),
            }
        )
        validate_resume_provenance(checkpoint_infos.get("resume_provenance"))
        try:
            validate_randomization_rng_state_payload(checkpoint_infos["environment_rng_state"])
        except (ValueError, RuntimeError, TypeError) as error:
            raise RuntimeError(f"DreamWaQ environment RNG state is invalid before save: {error}") from error
        if infos is not None:
            checkpoint_infos["rsl_rl_infos"] = copy.deepcopy(infos)
        allowed = DREAMWAQ_METADATA_FIELDS | {"rsl_rl_infos"}
        if set(checkpoint_infos) - allowed or not DREAMWAQ_METADATA_FIELDS.issubset(checkpoint_infos):
            raise RuntimeError("DreamWaQ checkpoint metadata fields are incomplete or unexpected")
        OnPolicyRunner.save(self, path, infos=checkpoint_infos)

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        patched_locs = dict(locs)
        patched_locs["ep_infos"] = expose_episode_info_key_union(
            locs.get("ep_infos", []),
            device=self.device,
        )
        OnPolicyRunner.log(self, patched_locs, width=width, pad=pad)


def validate_dreamwaq_checkpoint_metadata(
    checkpoint_path: Path,
    run_manifest_path: Path,
    current_base_task_contract: dict[str, Any],
    current_algorithm_contract: dict[str, Any],
    current_export_contract: dict[str, Any],
    *,
    requested_seed: int | None = None,
    expected_model_state_keys: set[str] | None = None,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not run_manifest_path.is_file():
        raise ContractMismatchError(f"DreamWaQ run manifest is missing: {run_manifest_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(payload) != {"model_state_dict", "optimizer_state_dict", "iter", "infos"}:
        raise ContractMismatchError("DreamWaQ checkpoint top-level fields are invalid")
    _validate_model_state_dict(payload["model_state_dict"], expected_keys=expected_model_state_keys)
    optimizer_state = _validate_optimizer_state(payload["optimizer_state_dict"])
    optimizer_learning_rates = optimizer_learning_rates_from_state(optimizer_state)

    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if "contract" in manifest:
        raise ContractMismatchError("DreamWaQ run manifest must not contain a generic contract field")
    saved_contracts = {
        "base_task_contract": (BASE_TASK_CONTRACT_VERSION, current_base_task_contract),
        "dreamwaq_algorithm_contract": (DREAMWAQ_ALGORITHM_CONTRACT_VERSION, current_algorithm_contract),
        "dreamwaq_export_contract": (DREAMWAQ_EXPORT_CONTRACT_VERSION, current_export_contract),
    }
    for name, (version, current) in saved_contracts.items():
        saved = manifest.get(name)
        if not isinstance(saved, dict):
            raise ContractMismatchError(f"DreamWaQ run manifest lacks {name}")
        try:
            validate_named_contract(saved, current, version)
        except RuntimeError as error:
            raise ContractMismatchError(str(error)) from error

    metadata = payload.get("infos")
    if not isinstance(metadata, dict):
        raise ContractMismatchError("DreamWaQ checkpoint metadata dictionary is missing")
    allowed = DREAMWAQ_METADATA_FIELDS | {"rsl_rl_infos"}
    if not DREAMWAQ_METADATA_FIELDS.issubset(metadata) or set(metadata) - allowed:
        raise ContractMismatchError("DreamWaQ checkpoint metadata fields are incomplete or unexpected")
    if metadata["metadata_version"] != DREAMWAQ_CHECKPOINT_METADATA_VERSION:
        raise ContractMismatchError("DreamWaQ checkpoint metadata version is invalid")
    if metadata["runner_class"] != DREAMWAQ_RUNNER_CLASS:
        raise ContractMismatchError("DreamWaQ checkpoint runner class is invalid")
    try:
        datetime.fromisoformat(metadata["saved_at"])
    except (TypeError, ValueError) as error:
        raise ContractMismatchError("DreamWaQ checkpoint saved_at is invalid") from error
    if not isinstance(metadata["seed"], int) or isinstance(metadata["seed"], bool) or metadata["seed"] < 0:
        raise ContractMismatchError("DreamWaQ checkpoint seed is invalid")
    if requested_seed is not None and metadata["seed"] != requested_seed:
        raise ContractMismatchError("DreamWaQ checkpoint seed differs from the requested seed")
    if metadata["run_manifest_sha256"] != sha256_file(run_manifest_path):
        raise ContractMismatchError("DreamWaQ checkpoint run manifest hash mismatch")
    expected_contract_metadata = {
        "base_task_contract_version": BASE_TASK_CONTRACT_VERSION,
        "base_task_contract_hash": current_base_task_contract["contract_hash"],
        "dreamwaq_algorithm_contract_version": DREAMWAQ_ALGORITHM_CONTRACT_VERSION,
        "dreamwaq_algorithm_contract_hash": current_algorithm_contract["contract_hash"],
        "dreamwaq_export_contract_version": DREAMWAQ_EXPORT_CONTRACT_VERSION,
        "dreamwaq_export_contract_hash": current_export_contract["contract_hash"],
    }
    for field, expected in expected_contract_metadata.items():
        if metadata[field] != expected:
            raise ContractMismatchError(f"DreamWaQ checkpoint {field} differs from its run manifest")
    _validate_hardware_profile(metadata["hardware_profile"])

    artifact, cache_metadata, _ = load_closed_chain_reset_cache_artifact(
        run_manifest_path,
        manifest=manifest,
    )
    if metadata["reset_cache_binding"] != checkpoint_cache_binding(cache_metadata):
        raise ContractMismatchError("DreamWaQ checkpoint reset cache binding differs from its run manifest")
    validate_cache_identity(
        artifact=artifact,
        manifest=manifest,
        metadata=metadata,
        saved_contract=manifest["base_task_contract"],
    )
    try:
        validate_randomization_rng_state_payload(metadata["environment_rng_state"])
    except (ValueError, RuntimeError, TypeError) as error:
        raise ContractMismatchError(f"DreamWaQ checkpoint environment RNG state is invalid: {error}") from error
    provenance = validate_resume_provenance(metadata["resume_provenance"])
    if provenance != manifest.get("resume_provenance"):
        raise ContractMismatchError("DreamWaQ checkpoint resume provenance differs from its run manifest")

    checkpoint_iteration = payload["iter"]
    if not isinstance(checkpoint_iteration, int) or isinstance(checkpoint_iteration, bool) or checkpoint_iteration < 0:
        raise ContractMismatchError("DreamWaQ checkpoint iteration is invalid")
    completed_iterations = checkpoint_iteration + 1
    if metadata["runner_iteration"] != checkpoint_iteration or metadata["completed_iterations"] != completed_iterations:
        raise ContractMismatchError("DreamWaQ checkpoint iteration metadata is inconsistent")
    if not isinstance(metadata["total_timesteps"], int) or metadata["total_timesteps"] < 0:
        raise ContractMismatchError("DreamWaQ checkpoint total_timesteps is invalid")
    if not isinstance(metadata["total_time_seconds"], (int, float)) or not math.isfinite(
        float(metadata["total_time_seconds"])
    ) or metadata["total_time_seconds"] < 0:
        raise ContractMismatchError("DreamWaQ checkpoint total_time_seconds is invalid")
    if metadata["optimizer_learning_rates"] != optimizer_learning_rates:
        raise ContractMismatchError("DreamWaQ checkpoint optimizer learning-rate metadata is inconsistent")
    optimizer_learning_rate = require_single_learning_rate(optimizer_learning_rates)
    if not isinstance(metadata["algorithm_learning_rate"], (int, float)) or float(
        metadata["algorithm_learning_rate"]
    ) != optimizer_learning_rate:
        raise ContractMismatchError("DreamWaQ checkpoint adaptive learning rate is inconsistent")
    rng_state = metadata["rng_state"]
    if not isinstance(rng_state, dict) or set(rng_state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ContractMismatchError("DreamWaQ checkpoint RNG state is missing or incomplete")
    try:
        monitor = EstimatorMonitorStateV1.from_dict(metadata["estimator_monitor_state"])
    except (TypeError, ValueError) as error:
        raise ContractMismatchError(f"DreamWaQ estimator monitor state is invalid: {error}") from error
    if monitor.completed_rollouts != completed_iterations:
        raise ContractMismatchError("DreamWaQ estimator monitor rollout count differs from completed iterations")
    return metadata


def load_dreamwaq_checkpoint_artifact(
    checkpoint_path: Path,
    run_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = run_manifest_path or checkpoint_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ContractMismatchError(f"DreamWaQ run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contracts = (
        manifest.get("base_task_contract"),
        manifest.get("dreamwaq_algorithm_contract"),
        manifest.get("dreamwaq_export_contract"),
    )
    if not all(isinstance(contract, dict) for contract in contracts):
        raise ContractMismatchError("DreamWaQ run manifest contract dictionaries are missing")
    metadata = validate_dreamwaq_checkpoint_metadata(
        checkpoint_path,
        manifest_path,
        contracts[0],
        contracts[1],
        contracts[2],
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return payload, manifest, metadata


def resume_dreamwaq_runner_from_checkpoint(
    runner: DreamWaQContractOnPolicyRunner,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    *,
    map_location: str,
    requested_seed: int,
    target_total_iterations: int,
    restore_environment_rng_state: Callable[[dict[str, Any]], None],
) -> int:
    if not isinstance(runner, DreamWaQContractOnPolicyRunner):
        raise ContractMismatchError("DreamWaQ resume requires DreamWaQContractOnPolicyRunner")
    if metadata.get("metadata_version") != DREAMWAQ_CHECKPOINT_METADATA_VERSION:
        raise ContractMismatchError("Only DreamWaQCheckpointMetadataV1 can resume DreamWaQ training")
    if metadata.get("runner_class") != DREAMWAQ_RUNNER_CLASS:
        raise ContractMismatchError("DreamWaQ resume checkpoint runner class is invalid")
    if metadata.get("seed") != requested_seed:
        raise ContractMismatchError("DreamWaQ resume requested seed differs from the checkpoint seed")
    completed_iterations = int(metadata["completed_iterations"])
    if target_total_iterations <= completed_iterations:
        raise ContractMismatchError("DreamWaQ resume target total iterations must exceed completed iterations")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_model_state_dict(
        payload.get("model_state_dict"),
        expected_keys=set(runner.alg.policy.state_dict()),
    )
    saved_optimizer = _validate_optimizer_state(payload.get("optimizer_state_dict"))
    current_optimizer = runner.alg.optimizer.state_dict()
    if optimizer_parameter_group_signature(saved_optimizer) != optimizer_parameter_group_signature(current_optimizer):
        raise ContractMismatchError("DreamWaQ optimizer parameter-group structure differs from the current runner")

    runner.load(str(checkpoint_path), load_optimizer=True, map_location=map_location)
    restored_learning_rates = [float(group["lr"]) for group in runner.alg.optimizer.param_groups]
    restored_learning_rate = require_single_learning_rate(restored_learning_rates)
    if restored_learning_rates != metadata["optimizer_learning_rates"]:
        raise ContractMismatchError("DreamWaQ restored optimizer learning rates differ from checkpoint metadata")
    runner.alg.learning_rate = restored_learning_rate
    runner.current_learning_iteration = completed_iterations
    runner.tot_timesteps = int(metadata["total_timesteps"])
    runner.tot_time = float(metadata["total_time_seconds"])
    runner.alg.set_estimator_monitor_state(
        EstimatorMonitorStateV1.from_dict(metadata["estimator_monitor_state"])
    )
    restore_rng_state(metadata["rng_state"])
    restore_environment_rng_state(metadata["environment_rng_state"])
    return completed_iterations
