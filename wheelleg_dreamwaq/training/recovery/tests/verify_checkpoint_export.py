import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from recovery.checkpoint import load_checkpoint, save_checkpoint
from recovery.contracts import default_resolved_config, file_sha256, source_sha256, stable_hash, write_json
from recovery.export import export_actor, golden_inputs_from_pose_bank, load_export
from recovery.pose_bank import PoseBank, source_pose_records


def verify(pose_bank_path: Path, output_dir: Path, walker_export: Path) -> dict:
    inputs, provenance = golden_inputs_from_pose_bank(pose_bank_path)
    bank = PoseBank.load(pose_bank_path)
    config = default_resolved_config()
    config["environment"]["device"] = "cpu"
    config["environment"]["num_envs"] = 1
    config["environment"]["pose_bank_path"] = str(pose_bank_path.resolve())
    config["environment"]["run_dir"] = str(output_dir.resolve())
    first = bank.records[0]
    identity = {
        "training_config_sha256": stable_hash(config),
        "walker_actor_sha256": file_sha256(walker_export / "actor.ts"),
        "asset_bundle_sha256": first.asset_bundle_sha256,
        "contact_model_sha256": first.contact_model_sha256,
        "pose_bank_sha256": bank.sha256,
        "source_pose_records_sha256": stable_hash(source_pose_records(bank)),
        "parameter_plan_sha256": first.parameter_plan_sha256,
        "num_envs": 1,
        "device": "cpu",
        "source_sha256": source_sha256(),
    }
    observations = TensorDict({"policy": inputs, "critic": torch.zeros(len(inputs), 34)}, batch_size=[len(inputs)])
    policy = ActorCritic(
        observations, {"policy": ["policy"], "critic": ["critic"]}, 6,
        actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64],
        activation="elu", init_noise_std=0.5, noise_std_type="log",
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    # 使用真实记录的上一实际动作核验优化器与导出组件，不作为起身 PPO 能力证据。
    targets = torch.tensor([record.prior_actions[-1] for record in bank.records], dtype=torch.float32)
    before = deepcopy(policy.actor.state_dict())
    optimizer.zero_grad()
    loss = (policy.actor(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    parameter_changed = any(not torch.equal(before[key], value) for key, value in policy.actor.state_dict().items())
    if not parameter_changed:
        raise AssertionError("The real gradient update did not change Actor parameters")
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "component_model_1.pt"
    save_checkpoint(checkpoint, {
        "model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "completed_updates": 1, "learning_rate": 0.001,
        "environment_state": {"source_pose_bank_sha256": bank.sha256,
                              "validation_kind": "component_gradient_on_recorded_physical_histories"},
    }, identity)
    loaded = load_checkpoint(checkpoint, identity, restore_rng=False)
    if loaded["completed_updates"] != 1:
        raise AssertionError("Checkpoint completed update count changed")
    manifest = export_actor(
        checkpoint, output_dir / "export", resolved_config=config, expected_identity=identity,
        golden_inputs=inputs, golden_provenance=provenance,
    )
    actor, _ = load_export(output_dir / "export", expected_identity=identity)
    with torch.inference_mode():
        error = float((actor(inputs) - policy.actor(inputs)).abs().max())
    if error > 1e-7:
        raise AssertionError("Reloaded Actor differs from the original policy")
    # 部署包迁移后，来源训练目录与姿态库路径可以失效。
    portable_dir = output_dir / "portable_export"
    portable_dir.mkdir()
    for source in (output_dir / "export").iterdir():
        (portable_dir / source.name).write_bytes(source.read_bytes())
    # 本检查直接移走源 checkpoint；部署加载只使用包内文件。
    moved_checkpoint = checkpoint.with_suffix(".saved")
    checkpoint.rename(moved_checkpoint)
    load_export(portable_dir, expected_identity=identity)
    moved_checkpoint.rename(checkpoint)
    corrupt_actor = portable_dir / "actor.ts"
    with corrupt_actor.open("ab") as stream:
        stream.write(b"corrupted")
    rejected_tamper = False
    try:
        load_export(portable_dir, expected_identity=identity)
    except ValueError:
        rejected_tamper = True
    if not rejected_tamper:
        raise AssertionError("Changed Actor bytes were accepted")
    report = {
        "schema_version": "RecoveryCheckpointExportVerificationV1",
        "validation_kind": "component_gradient_on_recorded_physical_histories",
        "recovery_ppo_training_verified": False,
        "recovery_capability_verified": False,
        "golden_sample_count": len(inputs),
        "pose_bank_sha256": bank.sha256,
        "parameter_changed": parameter_changed,
        "component_loss": float(loss),
        "golden_batch_errors": manifest["golden_max_abs_error"],
        "reload_max_abs_error": error,
        "portable_without_checkpoint": True,
        "changed_actor_rejected": rejected_tamper,
        "export_dir": str((output_dir / "export").resolve()),
        "identity": identity,
    }
    write_json(output_dir / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--walker-export", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.pose_bank, args.output_dir, args.walker_export), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
