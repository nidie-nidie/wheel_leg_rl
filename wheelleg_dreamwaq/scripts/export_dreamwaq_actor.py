from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tensordict import TensorDict

from wheelleg_dreamwaq.algorithms.dreamwaq.actor_critic import DreamWaQActorCritic
from wheelleg_dreamwaq.deployment.dreamwaq_inference import DreamWaQInferenceActorV1
from wheelleg_dreamwaq.deployment.manifest import export_dreamwaq_inference_package
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
    DREAMWAQ_EXPORT_CONTRACT_VERSION,
)
from wheelleg_dreamwaq.training.checkpoint import sha256_file
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import load_dreamwaq_checkpoint_artifact
from wheelleg_dreamwaq.schemas.physics import PHYSICS_SCHEMA_VERSION, unrestricted_velocity_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST_PATH = PROJECT_ROOT / "sim2sim" / "mujoco" / "model_manifest.json"


def _build_policy(state_dict: dict[str, torch.Tensor]) -> DreamWaQActorCritic:
    observations = TensorDict(
        {
            "policy": torch.zeros(1, 25, dtype=torch.float32),
            "policy_history": torch.zeros(1, 125, dtype=torch.float32),
            "critic": torch.zeros(1, 41, dtype=torch.float32),
        },
        batch_size=[1],
    )
    policy = DreamWaQActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        6,
    )
    policy.load_state_dict(state_dict, strict=True)
    return policy.cpu().eval()


def _manifest_payload(
    *,
    checkpoint: Path,
    run_manifest_path: Path,
    run_manifest: dict,
    metadata: dict,
    model_manifest: dict,
) -> dict:
    base = run_manifest["base_task_contract"]
    if (
        base["schemas"].get("physics") != PHYSICS_SCHEMA_VERSION
        or base["task"]["physics"].get("velocity_limit_policy") != unrestricted_velocity_policy()
    ):
        raise ValueError("Checkpoint physics differs from PhysicsV5; old checkpoints cannot be relabeled")
    algorithm = run_manifest["dreamwaq_algorithm_contract"]
    export = run_manifest["dreamwaq_export_contract"]
    return {
        "dreamwaq_export_contract_version": DREAMWAQ_EXPORT_CONTRACT_VERSION,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_run_manifest": str(run_manifest_path),
        "source_run_manifest_sha256": sha256_file(run_manifest_path),
        "source_completed_iterations": int(metadata["completed_iterations"]),
        "source_seed": int(metadata["seed"]),
        "hardware_profile": metadata["hardware_profile"],
        "base_task_contract_version": base["manifest_version"],
        "base_task_contract_hash": base["contract_hash"],
        "dreamwaq_algorithm_contract_version": algorithm["manifest_version"],
        "dreamwaq_algorithm_contract_hash": algorithm["contract_hash"],
        "dreamwaq_export_contract_hash": export["contract_hash"],
        "history_adapter_version": "MujocoFrameMajorHistoryAdapterV1",
        "asset_bundle_version": base["asset_bundle_version"],
        "asset_bundle_hash": base["asset_bundle_hash"],
        "asset": base["asset"],
        "schemas": base["schemas"],
        "action": base["action"],
        "observation": base["observation"],
        "normalization": base["normalization"],
        "training_randomization": base["randomization"],
        "frames": base["frames"],
        "command_sampling": base["task"]["commands"],
        "q_nominal": base["task"]["q_nominal"],
        "control": base["task"]["control"],
        "actuators": base["runtime"]["robot_except_asset_absolute_path"]["actuators"],
        "rigid_body_properties": base["runtime"]["robot_except_asset_absolute_path"]["spawn"]["rigid_props"],
        "velocity_limit_policy": base["task"]["physics"]["velocity_limit_policy"],
        "initial_state": base["runtime"]["robot_except_asset_absolute_path"]["init_state"],
        "network": {
            "input_dimension": 125,
            "output_dimension": 6,
            "dynamic_batch": True,
            "history_layout": "frame_major",
            "history_length": 5,
            "current_observation_dimension": 25,
            "encoder": algorithm["network"]["encoder"],
            "actor": algorithm["network"]["actor"],
            "encoder_output_slices": algorithm["network"]["encoder_output_slices"],
            "actor_context_mode": algorithm["network"]["actor_context_mode"],
            "effective_mean_clip": [-20.0, 20.0],
            "runtime_action_clip": [-1.0, 1.0],
        },
        "timing": {
            "isaac_sim_dt_s": base["task"]["sim_dt"],
            "isaac_decimation": base["task"]["decimation"],
            "control_dt_s": base["task"]["control_dt"],
            "mujoco_physics_dt_s": model_manifest["physics_dt_s"],
            "mujoco_physics_steps_per_action": model_manifest["physics_steps_per_policy_action"],
        },
        "mujoco_model": {
            "model_version": model_manifest["model_version"],
            "model_xml_sha256": model_manifest["model_xml"]["sha256"],
            "model_manifest_sha256": sha256_file(MODEL_MANIFEST_PATH),
            "dynamics_semantics_hash": model_manifest["dynamics_semantics_hash"],
            "actuator_effort_limits_nm": model_manifest["actuator_effort_limits_nm"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a validated deterministic DreamWaQ actor as TorchScript.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    run_manifest_path = checkpoint.parent / "run_manifest.json"
    payload, run_manifest, metadata = load_dreamwaq_checkpoint_artifact(checkpoint, run_manifest_path)
    if not MODEL_MANIFEST_PATH.is_file():
        raise FileNotFoundError(MODEL_MANIFEST_PATH)
    model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    base = run_manifest["base_task_contract"]
    if model_manifest["asset_bundle_hash"] != base["asset_bundle_hash"]:
        raise ValueError("MuJoCo model and DreamWaQ checkpoint use different AssetBundleV2 identities")
    policy = _build_policy(payload["model_state_dict"])
    inference_actor = DreamWaQInferenceActorV1.from_policy(policy)
    summary = export_dreamwaq_inference_package(
        inference_actor,
        args.output,
        _manifest_payload(
            checkpoint=checkpoint,
            run_manifest_path=run_manifest_path.resolve(),
            run_manifest=run_manifest,
            metadata=metadata,
            model_manifest=model_manifest,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
