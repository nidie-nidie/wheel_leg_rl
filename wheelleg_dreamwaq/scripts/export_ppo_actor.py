from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from wheelleg_dreamwaq.training.checkpoint import load_checkpoint_artifact, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST_PATH = PROJECT_ROOT / "sim2sim" / "mujoco" / "model_manifest.json"
EXPORT_SCHEMA_VERSION = "PpoActorExportV1"


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def _build_policy(contract: dict, state_dict: dict[str, torch.Tensor]) -> ActorCritic:
    observation = contract["observation"]
    action = contract["action"]
    network = contract["network"]
    training_policy = contract["training"]["policy"]
    if network["actor_obs_normalization"] or training_policy["state_dependent_std"]:
        raise ValueError("PpoActorExportV1 only supports the frozen unnormalized feed-forward Phase 1 actor")
    sample = TensorDict(
        {
            "policy": torch.zeros(1, observation["actor_dimension"]),
            "critic": torch.zeros(1, observation["critic_dimension"]),
        },
        batch_size=[1],
    )
    policy = ActorCritic(
        sample,
        contract["training"]["runner"]["obs_groups"],
        action["dimension"],
        actor_obs_normalization=network["actor_obs_normalization"],
        critic_obs_normalization=network["critic_obs_normalization"],
        actor_hidden_dims=network["actor_hidden_dims"],
        critic_hidden_dims=network["critic_hidden_dims"],
        activation=network["activation"],
        init_noise_std=training_policy["init_noise_std"],
        noise_std_type=training_policy["noise_std_type"],
        state_dependent_std=training_policy["state_dependent_std"],
    )
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a validated deterministic PPO actor as TorchScript.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    payload, run_manifest, metadata = load_checkpoint_artifact(checkpoint)
    contract = run_manifest["contract"]
    if contract.get("manifest_version") != "Phase1ContractV4":
        raise ValueError("Only Phase1ContractV4 checkpoints can be exported")
    if contract["schemas"] != {
        "action": "ActionV1",
        "actor_observation": "ActorObsV1",
        "command": "CommandV1",
        "command_sampling": "CommandSamplingV2",
        "control_frame": "ControlFrameV1",
        "critic_observation": "CriticObsV1",
        "normalization": "NormalizationV2",
        "physics": "PhysicsV4",
        "reward": "RewardSchemaV2",
        "virtual_leg_kinematics": "VirtualLegKinematicsV1",
    }:
        raise ValueError("Checkpoint schema set is not the frozen ContractV4 schema set")
    if not MODEL_MANIFEST_PATH.is_file():
        raise FileNotFoundError(MODEL_MANIFEST_PATH)
    model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if model_manifest["asset_bundle_hash"] != contract["asset_bundle_hash"]:
        raise ValueError("MuJoCo model and PPO checkpoint use different AssetBundleV2 identities")

    policy = _build_policy(contract, payload["model_state_dict"])
    actor = policy.actor.cpu().eval()
    output.mkdir(parents=True, exist_ok=True)
    actor_path = output / "actor.ts"
    scripted_actor = torch.jit.script(actor)
    scripted_actor.save(str(actor_path))

    generator = torch.Generator(device="cpu").manual_seed(20261004)
    test_observation = torch.randn(32, contract["observation"]["actor_dimension"], generator=generator)
    critic_observation = torch.zeros(32, contract["observation"]["critic_dimension"])
    test_tensor_dict = TensorDict(
        {"policy": test_observation, "critic": critic_observation},
        batch_size=[32],
    )
    with torch.inference_mode():
        expected = policy.cpu().act_inference(test_tensor_dict)
        actual = torch.jit.load(str(actor_path))(test_observation)
    max_abs_error = float(torch.max(torch.abs(expected - actual)).item())
    if max_abs_error > 1.0e-7:
        raise RuntimeError(f"Exported Actor disagrees with RSL-RL inference: {max_abs_error}")

    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_run_manifest": str((checkpoint.parent / "run_manifest.json").resolve()),
        "source_run_manifest_sha256": sha256_file(checkpoint.parent / "run_manifest.json"),
        "source_completed_iterations": int(metadata["completed_iterations"]),
        "phase1_contract_version": contract["manifest_version"],
        "phase1_contract_hash": contract["contract_hash"],
        "asset_bundle_version": contract["asset_bundle_version"],
        "asset_bundle_hash": contract["asset_bundle_hash"],
        "schemas": contract["schemas"],
        "actor_file": actor_path.name,
        "actor_sha256": sha256_file(actor_path),
        "actor_dtype": "float32",
        "network": {
            "input_dimension": contract["observation"]["actor_dimension"],
            "hidden_dimensions": contract["network"]["actor_hidden_dims"],
            "output_dimension": contract["action"]["dimension"],
            "activation": contract["network"]["activation"],
            "actor_observation_normalization": contract["network"]["actor_obs_normalization"],
        },
        "action": contract["action"],
        "observation": contract["observation"],
        "normalization": contract["normalization"],
        "frames": contract["frames"],
        "command_sampling": contract["task"]["commands"],
        "q_nominal": contract["task"]["q_nominal"],
        "control": contract["task"]["control"],
        "actuators": contract["runtime"]["robot_except_asset_absolute_path"]["actuators"],
        "initial_state": contract["runtime"]["robot_except_asset_absolute_path"]["init_state"],
        "timing": {
            "isaac_sim_dt_s": contract["task"]["sim_dt"],
            "isaac_decimation": contract["task"]["decimation"],
            "control_dt_s": contract["task"]["control_dt"],
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
        "export_verification_max_abs_error": max_abs_error,
        "torch_version": torch.__version__,
    }
    manifest["manifest_hash"] = _stable_hash(manifest)
    manifest_path = output / "policy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "actor": str(actor_path),
                "policy_manifest": str(manifest_path),
                "contract_hash": contract["contract_hash"],
                "verification_max_abs_error": max_abs_error,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
