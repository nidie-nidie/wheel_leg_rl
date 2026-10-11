"""Explicit experimental weights-only initialization, separate from ordinary resume."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from wheelleg_dreamwaq.schemas.manifest import ContractMismatchError
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import task_profile_from_contract
from wheelleg_dreamwaq.training.checkpoint import sha256_file


INITIALIZATION_FIELDS = {
    "schema_version", "mode", "source_checkpoint", "source_checkpoint_sha256",
    "source_run_manifest", "source_run_manifest_sha256", "source_completed_iterations",
    "source_base_task_contract_hash", "target_base_task_contract_hash",
}


def validate_weights_initialization(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != INITIALIZATION_FIELDS:
        raise ContractMismatchError("weights initialization fields are invalid")
    if payload["schema_version"] != "WeightsOnlyInitializationV1" or payload["mode"] != "weights_only_new_optimizer_rng_environment":
        raise ContractMismatchError("weights initialization mode is invalid")
    if type(payload["source_completed_iterations"]) is not int or payload["source_completed_iterations"] <= 0:
        raise ContractMismatchError("weights initialization source iteration is invalid")
    for key in ("source_checkpoint", "source_run_manifest"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ContractMismatchError("weights initialization source path is invalid")
    for key in ("source_checkpoint_sha256", "source_run_manifest_sha256",
                "source_base_task_contract_hash", "target_base_task_contract_hash"):
        value = payload[key]
        if not isinstance(value, str) or len(value) != 64 or value != value.upper():
            raise ContractMismatchError("weights initialization hash is invalid")
        try:
            int(value, 16)
        except ValueError as error:
            raise ContractMismatchError("weights initialization hash is invalid") from error


def require_initialization_compatibility(source: dict, target: dict,
                                         source_algorithm: dict, target_algorithm: dict,
                                         source_export: dict, target_export: dict) -> None:
    if source_algorithm != target_algorithm or source_export != target_export:
        raise ContractMismatchError("weights initialization algorithm/export contracts differ")
    if task_profile_from_contract(source) != "stop_reverse_v1":
        raise ContractMismatchError("weights initialization source must be current stop_reverse_v1")
    if task_profile_from_contract(target) not in (
        "stop_reverse_v1", "stop_reverse_reversal12_v1", "stop_reverse_reversal8_v1"
    ):
        raise ContractMismatchError("weights initialization target profile is not an approved experiment")
    source_payload, target_payload = copy.deepcopy(source), copy.deepcopy(target)
    source_payload.pop("contract_hash")
    target_payload.pop("contract_hash")
    target_payload["task"].pop("leg_reversal", None)
    if source_payload != target_payload:
        raise ContractMismatchError("weights initialization may only add leg_reversal; other task contracts differ")


def load_weights_initialization(checkpoint: Path, target_base: dict,
                                target_algorithm: dict, target_export: dict) -> tuple[dict, dict, dict, dict]:
    # Local import avoids a checkpoint/provenance module cycle.
    from .dreamwaq_checkpoint import load_dreamwaq_checkpoint_artifact
    checkpoint = checkpoint.resolve()
    payload, manifest, metadata = load_dreamwaq_checkpoint_artifact(checkpoint)
    require_initialization_compatibility(
        manifest["base_task_contract"], target_base,
        manifest["dreamwaq_algorithm_contract"], target_algorithm,
        manifest["dreamwaq_export_contract"], target_export)
    manifest_path = checkpoint.parent / "run_manifest.json"
    provenance = {
        "schema_version": "WeightsOnlyInitializationV1",
        "mode": "weights_only_new_optimizer_rng_environment",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_run_manifest": str(manifest_path),
        "source_run_manifest_sha256": sha256_file(manifest_path),
        "source_completed_iterations": metadata["completed_iterations"],
        "source_base_task_contract_hash": manifest["base_task_contract"]["contract_hash"],
        "target_base_task_contract_hash": target_base["contract_hash"],
    }
    validate_weights_initialization(provenance)
    return payload["model_state_dict"], manifest, metadata, provenance
