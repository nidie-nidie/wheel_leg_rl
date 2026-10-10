from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from wheelleg_dreamwaq.algorithms.dreamwaq.actor_critic import DreamWaQActorCritic
from wheelleg_dreamwaq.algorithms.dreamwaq.ppo import DreamWaQPPO
from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2
from wheelleg_dreamwaq.schemas.action import CANONICAL_JOINT_ORDER
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
    BASE_TASK_CONTRACT_VERSION,
    DREAMWAQ_ALGORITHM_CONTRACT_VERSION,
    DREAMWAQ_EXPORT_CONTRACT_VERSION,
    EstimatorMonitorStateV1,
    stable_contract_hash,
)
from wheelleg_dreamwaq.schemas.manifest import ContractMismatchError
from wheelleg_dreamwaq.schemas.randomization import (
    CLOSED_CHAIN_NOMINAL_BRANCH_SIGNATURE,
    CLOSED_CHAIN_PHYSICAL_PASSIVE_BRANCH_JOINTS,
    CLOSED_CHAIN_RELAXATION_VERSION,
    CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
    CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION,
    FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
    PROCESS_START_RANDOMIZATION_STREAMS,
    RANDOMIZATION_SEED_DERIVATION_VERSION,
    RANDOMIZATION_STREAMS,
    RUNTIME_RANDOMIZATION_STREAMS,
    canonical_tensor_sha256,
    closed_chain_reset_contract_payload,
    profile_contract_hash,
)
from wheelleg_dreamwaq.schemas.physics import PHYSICS_SCHEMA_VERSION, SIM_DT_S
from wheelleg_dreamwaq.training.checkpoint import (
    CLOSED_CHAIN_RESET_CACHE_FILENAME,
    capture_rng_state,
    checkpoint_cache_binding,
    closed_chain_reset_cache_metadata,
    sha256_file,
)
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import (
    DREAMWAQ_CHECKPOINT_METADATA_VERSION,
    DREAMWAQ_RUNNER_CLASS,
    DreamWaQContractOnPolicyRunner,
    fresh_resume_provenance,
    resume_dreamwaq_runner_from_checkpoint,
    validate_dreamwaq_checkpoint_metadata,
    validate_resume_provenance,
)


TEST_PASSIVE_JOINT_NAMES = tuple(f"passive_{index}" for index in range(20))


def _with_hash(payload: dict) -> dict:
    result = deepcopy(payload)
    result["contract_hash"] = stable_contract_hash(result)
    return result


def _contracts() -> tuple[dict, dict, dict]:
    base = _with_hash(
        {
            "manifest_version": BASE_TASK_CONTRACT_VERSION,
            "asset_bundle_version": ASSET_BUNDLE_V2.version,
            "asset_bundle_hash": ASSET_BUNDLE_V2.bundle_hash,
            "schemas": {"physics": PHYSICS_SCHEMA_VERSION},
            "randomization": {"profile_hash": profile_contract_hash(FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1)},
        }
    )
    algorithm = _with_hash(
        {"manifest_version": DREAMWAQ_ALGORITHM_CONTRACT_VERSION, "runner_class": DREAMWAQ_RUNNER_CLASS}
    )
    export = _with_hash(
        {"manifest_version": DREAMWAQ_EXPORT_CONTRACT_VERSION, "schema_version": "DreamWaQPolicyExportV1"}
    )
    return base, algorithm, export


def _policy() -> DreamWaQActorCritic:
    observations = TensorDict(
        {
            "policy": torch.zeros(2, 25),
            "policy_history": torch.zeros(2, 125),
            "critic": torch.zeros(2, 41),
        },
        batch_size=[2],
    )
    return DreamWaQActorCritic(observations, {"policy": ["policy"], "critic": ["critic"]}, 6)


def _algorithm(*, learning_rate: float) -> DreamWaQPPO:
    return DreamWaQPPO(
        _policy(),
        learning_rate=learning_rate,
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
    )


def _initialize_optimizer(algorithm: DreamWaQPPO) -> None:
    algorithm.optimizer.zero_grad()
    loss = sum(parameter.square().mean() for parameter in algorithm.policy.parameters())
    loss.backward()
    algorithm.optimizer.step()


def _environment_rng_state() -> dict:
    payload = {"seed_derivation_version": RANDOMIZATION_SEED_DERIVATION_VERSION, "streams": {}}
    for index, stream in enumerate(RANDOMIZATION_STREAMS):
        generator = torch.Generator(device="cpu").manual_seed(1000 + index)
        payload["streams"][stream] = {"device": "cpu", "state": generator.get_state()}
    return payload


def _write_cache(tmp_path: Path, base_contract: dict, *, seed: int, num_envs: int) -> tuple[dict, dict]:
    q_reset = torch.zeros(
        (num_envs, len(CANONICAL_JOINT_ORDER) + len(TEST_PASSIVE_JOINT_NAMES)), dtype=torch.float32
    )
    root_height_offset = torch.linspace(-0.002, 0.003, num_envs, dtype=torch.float32)
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
            "asset_bundle_version": base_contract["asset_bundle_version"],
            "asset_bundle_hash": base_contract["asset_bundle_hash"],
            "physics_schema_version": base_contract["schemas"]["physics"],
            "randomization_profile_hash": base_contract["randomization"]["profile_hash"],
            "realized_plan_hash": realized_plan_hash,
            "actuator_plan_hash": actuator_plan_hash,
            "master_seed": seed,
            "num_envs": num_envs,
            "joint_names": list(CANONICAL_JOINT_ORDER) + list(TEST_PASSIVE_JOINT_NAMES),
            "passive_joint_names": list(TEST_PASSIVE_JOINT_NAMES),
            "physical_passive_branch_joint_names": list(CLOSED_CHAIN_PHYSICAL_PASSIVE_BRANCH_JOINTS),
        },
        "q_reset_projected_env": q_reset,
        "root_height_offset_env": root_height_offset,
        "tensor_sha256": canonical_tensor_sha256(
            {"q_reset_projected_env": q_reset, "root_height_offset_env": root_height_offset}
        ),
        "actuator_plan": actuator_plan,
        "metrics": {
            "physical_passive_branch_signature": [list(CLOSED_CHAIN_NOMINAL_BRANCH_SIGNATURE)] * num_envs
        },
        "trace": [],
    }
    cache_path = tmp_path / CLOSED_CHAIN_RESET_CACHE_FILENAME
    torch.save(artifact, cache_path)
    return closed_chain_reset_cache_metadata(cache_path, run_dir=tmp_path), {
        "realized_plan_hash": realized_plan_hash,
        "actuator_plan_hash": actuator_plan_hash,
    }


def _write_checkpoint(tmp_path: Path) -> tuple[Path, Path, tuple[dict, dict, dict], dict]:
    seed = 42
    num_envs = 8
    base, algorithm_contract, export_contract = _contracts()
    cache_metadata, randomization_audit = _write_cache(tmp_path, base, seed=seed, num_envs=num_envs)
    provenance = fresh_resume_provenance()
    manifest = {
        "base_task_contract": base,
        "dreamwaq_algorithm_contract": algorithm_contract,
        "dreamwaq_export_contract": export_contract,
        "randomization_audit": randomization_audit,
        "closed_chain_reset_cache": cache_metadata,
        "resume_provenance": provenance,
    }
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    saved_algorithm = _algorithm(learning_rate=2.5e-4)
    _initialize_optimizer(saved_algorithm)
    monitor = EstimatorMonitorStateV1()
    monitor.record_rollout(0.2)
    monitor.record_rollout(0.3)
    metadata = {
        "metadata_version": DREAMWAQ_CHECKPOINT_METADATA_VERSION,
        "saved_at": "2026-10-07T12:00:00+08:00",
        "runner_class": DREAMWAQ_RUNNER_CLASS,
        "seed": seed,
        "runner_iteration": 1,
        "completed_iterations": 2,
        "total_timesteps": 384,
        "total_time_seconds": 1.25,
        "optimizer_learning_rates": [2.5e-4],
        "algorithm_learning_rate": 2.5e-4,
        "rng_state": capture_rng_state(),
        "environment_rng_state": _environment_rng_state(),
        "estimator_monitor_state": monitor.to_dict(),
        "run_manifest_sha256": sha256_file(manifest_path),
        "base_task_contract_version": BASE_TASK_CONTRACT_VERSION,
        "base_task_contract_hash": base["contract_hash"],
        "dreamwaq_algorithm_contract_version": DREAMWAQ_ALGORITHM_CONTRACT_VERSION,
        "dreamwaq_algorithm_contract_hash": algorithm_contract["contract_hash"],
        "dreamwaq_export_contract_version": DREAMWAQ_EXPORT_CONTRACT_VERSION,
        "dreamwaq_export_contract_hash": export_contract["contract_hash"],
        "hardware_profile": {
            "name": "portable",
            "declared_num_envs": num_envs,
            "effective_num_envs": num_envs,
            "num_mini_batches": 1,
            "num_steps_per_env": 24,
            "world_size": 1,
            "global_rank": 0,
            "local_rank": 0,
        },
        "reset_cache_binding": checkpoint_cache_binding(cache_metadata),
        "resume_provenance": provenance,
    }
    checkpoint_path = tmp_path / "model_1.pt"
    torch.save(
        {
            "model_state_dict": saved_algorithm.policy.state_dict(),
            "optimizer_state_dict": saved_algorithm.optimizer.state_dict(),
            "iter": 1,
            "infos": metadata,
        },
        checkpoint_path,
    )
    return checkpoint_path, manifest_path, (base, algorithm_contract, export_contract), metadata


def test_dreamwaq_checkpoint_binds_three_contracts_cache_rng_and_monitor(tmp_path: Path) -> None:
    checkpoint, manifest, contracts, _ = _write_checkpoint(tmp_path)

    metadata = validate_dreamwaq_checkpoint_metadata(
        checkpoint,
        manifest,
        *contracts,
        requested_seed=42,
        expected_model_state_keys=set(_policy().state_dict()),
    )

    assert metadata["runner_class"] == DREAMWAQ_RUNNER_CLASS
    assert metadata["estimator_monitor_state"]["completed_rollouts"] == 2
    assert metadata["resume_provenance"] == fresh_resume_provenance()
    assert metadata["optimizer_learning_rates"] == [pytest.approx(2.5e-4)]


@pytest.mark.parametrize(
    "contract_index",
    [0, 1, 2],
)
def test_dreamwaq_checkpoint_rejects_any_contract_drift(tmp_path: Path, contract_index: int) -> None:
    checkpoint, manifest, contracts, _ = _write_checkpoint(tmp_path)
    current = list(deepcopy(contracts))
    current[contract_index]["drift"] = True
    current[contract_index]["contract_hash"] = stable_contract_hash(
        {key: value for key, value in current[contract_index].items() if key != "contract_hash"}
    )

    with pytest.raises(ContractMismatchError):
        validate_dreamwaq_checkpoint_metadata(checkpoint, manifest, *current)


def test_dreamwaq_checkpoint_rejects_phase1_metadata_and_generic_contract(tmp_path: Path) -> None:
    checkpoint, manifest_path, contracts, _ = _write_checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["infos"]["metadata_version"] = "CheckpointMetadataV6"
    torch.save(payload, checkpoint)

    with pytest.raises(ContractMismatchError, match="metadata version"):
        validate_dreamwaq_checkpoint_metadata(checkpoint, manifest_path, *contracts)

    checkpoint, manifest_path, contracts, _ = _write_checkpoint(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract"] = manifest["base_task_contract"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ContractMismatchError, match="generic contract"):
        validate_dreamwaq_checkpoint_metadata(checkpoint, manifest_path, *contracts)


def test_dreamwaq_checkpoint_rejects_monitor_iteration_or_seed_mismatch(tmp_path: Path) -> None:
    checkpoint, manifest, contracts, _ = _write_checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["infos"]["estimator_monitor_state"]["completed_rollouts"] = 1
    torch.save(payload, checkpoint)

    with pytest.raises(ContractMismatchError, match="estimator monitor"):
        validate_dreamwaq_checkpoint_metadata(checkpoint, manifest, *contracts)

    checkpoint, manifest, contracts, _ = _write_checkpoint(tmp_path)
    with pytest.raises(ContractMismatchError, match="requested seed"):
        validate_dreamwaq_checkpoint_metadata(checkpoint, manifest, *contracts, requested_seed=99)


def _resume_provenance(*, same_num_envs: bool) -> dict:
    source_num_envs = 8
    target_num_envs = 8 if same_num_envs else 16
    stream_sources = {
        stream: (
            "source_checkpoint"
            if same_num_envs or stream in RUNTIME_RANDOMIZATION_STREAMS
            else "target_process_start"
        )
        for stream in RANDOMIZATION_STREAMS
    }
    return {
        "schema_version": "ResumeMetadataV2",
        "mode": "weights_optimizer_rng_new_environment",
        "cache_resume_mode": "reused_source_exactly" if same_num_envs else "regenerated_for_target_num_envs",
        "source_checkpoint": "source.pt",
        "source_checkpoint_sha256": "1" * 64,
        "source_run_manifest": "run_manifest.json",
        "source_run_manifest_sha256": "2" * 64,
        "source_completed_iterations": 2,
        "source_hardware_profile_name": "portable",
        "target_hardware_profile_name": "rtx5070",
        "source_num_envs": source_num_envs,
        "target_num_envs": target_num_envs,
        "same_num_envs": same_num_envs,
        "source_cache_file_sha256": "3" * 64,
        "source_cache_tensor_sha256": "4" * 64,
        "target_cache_file_sha256": ("3" if same_num_envs else "5") * 64,
        "target_cache_tensor_sha256": ("4" if same_num_envs else "6") * 64,
        "source_realized_plan_hash": "7" * 64,
        "target_realized_plan_hash": "8" * 64,
        "rng_stream_sources": stream_sources,
    }


def test_resume_provenance_strictly_separates_fresh_same_scale_and_cross_scale() -> None:
    assert validate_resume_provenance(fresh_resume_provenance())["mode"] == "fresh"
    assert validate_resume_provenance(_resume_provenance(same_num_envs=True))["same_num_envs"] is True
    cross = validate_resume_provenance(_resume_provenance(same_num_envs=False))
    assert all(cross["rng_stream_sources"][stream] == "target_process_start" for stream in PROCESS_START_RANDOMIZATION_STREAMS)
    assert all(cross["rng_stream_sources"][stream] == "source_checkpoint" for stream in RUNTIME_RANDOMIZATION_STREAMS)

    invalid = fresh_resume_provenance()
    invalid["source_checkpoint"] = "forbidden.pt"
    with pytest.raises(ContractMismatchError, match="only schema_version and mode"):
        validate_resume_provenance(invalid)


def test_resume_restores_model_optimizer_learning_rate_iteration_rng_and_monitor(tmp_path: Path) -> None:
    checkpoint, _, _, metadata = _write_checkpoint(tmp_path)
    runner = DreamWaQContractOnPolicyRunner.__new__(DreamWaQContractOnPolicyRunner)
    runner.alg = _algorithm(learning_rate=1.0e-3)
    runner.current_learning_iteration = 0
    runner.tot_timesteps = 0
    runner.tot_time = 0.0
    restored_environment_rng = []

    completed = resume_dreamwaq_runner_from_checkpoint(
        runner,
        checkpoint,
        metadata,
        map_location="cpu",
        requested_seed=42,
        target_total_iterations=3,
        restore_environment_rng_state=restored_environment_rng.append,
    )

    assert completed == 2
    assert runner.current_learning_iteration == 2
    assert runner.tot_timesteps == 384
    assert runner.tot_time == pytest.approx(1.25)
    assert runner.alg.learning_rate == pytest.approx(2.5e-4)
    assert runner.alg.optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)
    assert runner.alg.estimator_monitor_state.to_dict() == metadata["estimator_monitor_state"]
    assert restored_environment_rng[0] is metadata["environment_rng_state"]


def test_resume_rejects_non_increasing_total_iteration_target(tmp_path: Path) -> None:
    checkpoint, _, _, metadata = _write_checkpoint(tmp_path)
    runner = DreamWaQContractOnPolicyRunner.__new__(DreamWaQContractOnPolicyRunner)
    runner.alg = _algorithm(learning_rate=1.0e-3)
    runner.current_learning_iteration = 0
    runner.tot_timesteps = 0
    runner.tot_time = 0.0

    with pytest.raises(ContractMismatchError, match="must exceed"):
        resume_dreamwaq_runner_from_checkpoint(
            runner,
            checkpoint,
            metadata,
            map_location="cpu",
            requested_seed=42,
            target_total_iterations=2,
            restore_environment_rng_state=lambda _: None,
        )
