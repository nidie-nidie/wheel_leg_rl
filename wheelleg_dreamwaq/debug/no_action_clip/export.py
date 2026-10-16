from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tensordict import TensorDict

from wheelleg_dreamwaq.deployment.dreamwaq_inference import DreamWaQInferenceActorV1

from debug.no_action_clip.experiment import (
    ACTION_CONTRACT, GEOMETRY_GUARD_CONTRACT, RawActionActorCritic,
    load_experiment_checkpoint, sha256, write_json,
)


def empty_observations(batch=1):
    return TensorDict({
        "policy": torch.zeros(batch, 25), "policy_history": torch.zeros(batch, 125),
        "critic": torch.zeros(batch, 41),
    }, batch_size=[batch])


def export_checkpoint(checkpoint: Path):
    payload, manifest = load_experiment_checkpoint(checkpoint)
    config = dict(manifest["agent_config"]["policy"])
    if config.pop("class_name") != "RawActionActorCritic":
        raise ValueError("导出策略配置不符")
    policy = RawActionActorCritic(empty_observations(), {"policy": ["policy"], "critic": ["critic"]}, 6, **config)
    policy.load_state_dict(payload["model_state_dict"], strict=True)
    policy.eval()
    actor = DreamWaQInferenceActorV1.from_policy(policy).eval()
    destination = checkpoint.parent / "export"
    destination.mkdir(exist_ok=True)
    scripted = torch.jit.script(actor)
    torch.jit.save(scripted, str(destination / "actor.ts"))
    torch.jit.save(torch.jit.script(policy.cenet.encoder), str(destination / "encoder.ts"))
    loaded = torch.jit.load(str(destination / "actor.ts")).eval()
    history = torch.randn(32, 5, 25, generator=torch.Generator().manual_seed(20261015)) * 0.3
    history[:, :, 19:25] = torch.linspace(-4.0, 4.0, 32 * 5 * 6).reshape(32, 5, 6)
    history = history.reshape(32, 125)
    obs = empty_observations(32)
    obs["policy_history"] = history
    obs["policy"] = history[:, -25:]
    errors = {}
    with torch.inference_mode():
        expected = policy.act_inference(obs)
        for count in (1, 3, 32):
            output = loaded(history[:count])
            torch.testing.assert_close(output, expected[:count], rtol=1.0e-6, atol=1.0e-6)
            torch.testing.assert_close(scripted(history[:count]), actor(history[:count]), rtol=0, atol=0)
            errors[str(count)] = float((output - expected[:count]).abs().max())
        torch.save({"history": history, "expected_action_mean": expected}, destination / "golden_vectors.pt")
    report = {
        "experiment_id": manifest["experiment_id"], "action_contract": ACTION_CONTRACT,
        "geometry_guard": GEOMETRY_GUARD_CONTRACT,
        "source_checkpoint": str(checkpoint.resolve()), "source_checkpoint_sha256": sha256(checkpoint),
        "training_manifest_sha256": sha256(checkpoint.parent / "manifest.json"),
        "actor_sha256": sha256(destination / "actor.ts"),
        "encoder_sha256": sha256(destination / "encoder.ts"),
        "golden_sha256": sha256(destination / "golden_vectors.pt"),
        "golden_max_abs_errors": errors,
        "completed_iterations": payload["infos"]["completed_iterations"],
        "baseline_export": manifest["baseline_export"],
    }
    if "reward_contract" in manifest:
        report["reward_contract"] = manifest["reward_contract"]
        report["training_reward_weights"] = manifest["reward_weights"]
    write_json(destination / "experiment_export.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    arguments = parser.parse_args()
    print(export_checkpoint(arguments.checkpoint))
