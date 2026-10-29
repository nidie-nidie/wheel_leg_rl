from __future__ import annotations

from copy import deepcopy
import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .checkpoint import read_checkpoint_envelope
from .contracts import (
    GuidanceConfigV1,
    GuidanceTrainingContractV1,
    RecoveryContractV2,
    file_sha256,
    stable_hash,
    validate_resolved_config,
)
from .observations import normalize_physical_frame
from .pose_bank import PoseBank, source_pose_records


class RecoveryMeanActor(nn.Module):
    def __init__(self, actor: nn.Sequential) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.dim() != 2 or observation.shape[1] != 110:
            raise ValueError("Recovery Actor requires batch-major 110-dimensional input")
        if not torch.isfinite(observation).all():
            raise ValueError("Recovery Actor input must be finite")
        result = self.actor(observation)
        if not torch.isfinite(result).all():
            raise ValueError("Recovery Actor output must be finite")
        return result


class GuidanceRecoveryMeanActor(nn.Module):
    def __init__(self, actor: nn.Sequential) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.dim() != 2 or observation.shape[1] != 110:
            raise ValueError("辅助起身部署输入必须为110维")
        if not torch.isfinite(observation).all():
            raise ValueError("辅助起身部署输入必须为有限数值")
        alpha = torch.zeros_like(observation[:, :1])
        result = self.actor(torch.cat((observation, alpha), dim=1))
        if not torch.isfinite(result).all():
            raise ValueError("辅助起身部署动作必须为有限数值")
        return result


def golden_inputs_from_pose_bank(
    path: str | Path,
    record_ids: list[str] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    bank = PoseBank.load(path)
    ids = list(record_ids) if record_ids is not None else [record.record_id for record in bank.records]
    if len(ids) < 32 or len(ids) != len(set(ids)):
        raise ValueError("Golden validation requires at least 32 distinct recorded physical samples")
    records = {record.record_id: record for record in bank.records}
    if any(record_id not in records for record_id in ids):
        raise ValueError("Golden sample ID is not present in the pose bank")
    physical = torch.tensor([records[record_id].clean_history for record_id in ids], dtype=torch.float32)
    inputs = normalize_physical_frame(physical).reshape(len(ids), 110).contiguous()
    return inputs, {
        "kind": "pose_bank",
        "path": str(Path(path).resolve()),
        "sha256": bank.sha256,
        "record_ids": ids,
    }


def export_actor(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    resolved_config: dict,
    expected_identity: dict,
    golden_inputs: torch.Tensor,
    golden_provenance: dict,
) -> dict:
    validate_resolved_config(resolved_config)
    contract = RecoveryContractV2.from_dict(resolved_config["contract"])
    if stable_hash(resolved_config) != expected_identity["training_config_sha256"]:
        raise ValueError("Export configuration differs from the checkpoint training configuration")
    _validate_golden_provenance(golden_inputs, golden_provenance, expected_identity)
    source_bank = PoseBank.load(golden_provenance["path"])
    source_index = source_pose_records(source_bank)
    if expected_identity.get("source_pose_records_sha256") != stable_hash(source_index):
        raise ValueError("训练checkpoint身份未绑定实际训练/测试姿态索引")
    envelope = read_checkpoint_envelope(checkpoint_path, expected_identity)
    guided = resolved_config["schema_version"] == "RecoveryResolvedConfigV3"
    training_contract = GuidanceTrainingContractV1.from_dict(resolved_config["training_contract"]) if guided else None
    if guided != ("guidance" in envelope["payload"]):
        raise ValueError("导出配置与checkpoint辅助训练机制不同")
    actor = _load_mean_actor(envelope["payload"]["model_state_dict"], contract, training_contract)
    inputs = golden_inputs.detach().to(device="cpu", dtype=torch.float32).contiguous()
    with torch.inference_mode():
        raw_mean = actor(inputs)
        if not torch.isfinite(raw_mean).all():
            raise FloatingPointError("Checkpoint produces nonfinite deployment means")
        max_mean = float(raw_mean.abs().max())
    scripted = torch.jit.script(actor).eval()
    reference_outputs: dict[str, torch.Tensor] = {}
    errors: dict[str, float] = {}
    with torch.inference_mode():
        for batch_size in (1, 3, 32):
            reference = actor(inputs[:batch_size]).clone()
            actual = scripted(inputs[:batch_size])
            error = float((actual - reference).abs().max())
            if error > 1e-7:
                raise ValueError(f"TorchScript golden error exceeds 1e-7 for batch {batch_size}: {error}")
            reference_outputs[str(batch_size)] = reference
            errors[str(batch_size)] = error
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    actor_path = destination / "actor.ts"
    golden_path = destination / "golden_vectors.pt"
    config_path = destination / "resolved_config.json"
    scripted.save(str(actor_path))
    if guided:
        torch.jit.script(actor.actor).eval().save(str(destination / "actor_111.ts"))
    with golden_path.open("xb") as stream:
        torch.save({
            "schema_version": "RecoveryGoldenVectorsV1",
            "inputs": inputs,
            "outputs_by_batch_size": reference_outputs,
            "provenance": deepcopy(golden_provenance),
        }, stream)
    _write_new_json(config_path, resolved_config)
    manifest = {
        "schema_version": "RecoveryExportV3" if guided else "RecoveryExportV2",
        "contract": deepcopy(resolved_config["contract"]),
        "training_config_sha256": stable_hash(resolved_config),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "actor_sha256": file_sha256(actor_path),
        "identity": deepcopy(expected_identity),
        "golden_sha256": file_sha256(golden_path),
        "actor_file": actor_path.name,
        "golden_file": golden_path.name,
        "config_file": config_path.name,
        "config_file_sha256": file_sha256(config_path),
        "action_output": "raw_mean",
        "golden_provenance_sha256": stable_hash(golden_provenance),
        "golden_batch_sizes": [1, 3, 32],
        "golden_max_abs_error": errors,
        "golden_max_observed_mean_abs": max_mean,
        "completed_updates": envelope["payload"]["completed_updates"],
        "source_pose_records": source_index,
    }
    if guided:
        manifest.update({
            "training_contract": deepcopy(resolved_config["training_contract"]),
            "guidance": deepcopy(resolved_config["guidance"]),
            "training_actor_input_dim": training_contract.actor_input_dim,
            "external_input_dim": training_contract.external_input_dim,
            "deployment_alpha": training_contract.deployment_alpha,
            "training_actor_file": "actor_111.ts",
            "training_actor_sha256": file_sha256(destination / "actor_111.ts"),
        })
    _write_new_json(destination / "manifest.json", manifest)
    # 从磁盘重新加载，核验交付文件及黄金向量。
    load_export(destination, expected_identity=expected_identity)
    return manifest


def load_export(
    export_dir: str | Path,
    expected_identity: dict | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.jit.ScriptModule, dict]:
    root = Path(export_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    fields = {
        "schema_version", "contract", "training_config_sha256", "checkpoint_sha256",
        "actor_sha256", "identity", "golden_sha256", "actor_file", "golden_file", "config_file",
        "config_file_sha256", "action_output", "golden_provenance_sha256",
        "golden_batch_sizes", "golden_max_abs_error", "golden_max_observed_mean_abs", "completed_updates",
        "source_pose_records",
    }
    guided = manifest.get("schema_version") == "RecoveryExportV3"
    if guided:
        fields.update({"training_contract", "guidance", "training_actor_input_dim", "external_input_dim",
                       "deployment_alpha", "training_actor_file", "training_actor_sha256"})
    if (set(manifest) != fields
            or manifest["schema_version"] not in ("RecoveryExportV2", "RecoveryExportV3")):
        raise ValueError("Unsupported or incomplete recovery export manifest")
    contract = RecoveryContractV2.from_dict(manifest["contract"])
    records = manifest["source_pose_records"]
    if manifest["identity"].get("source_pose_records_sha256") != stable_hash(records):
        raise ValueError("源训练姿态索引与checkpoint身份不同")
    if (not isinstance(records, list) or not records
            or any(set(record) != {"record_id", "trajectory_id", "split", "physical_state_sha256"}
                   or record["split"] not in ("train", "test")
                   or len(record["physical_state_sha256"]) != 64 for record in records)
            or len({record["record_id"] for record in records}) != len(records)):
        raise ValueError("起身导出缺少可核验的源训练姿态索引")
    if expected_identity is not None and manifest["identity"] != expected_identity:
        raise ValueError("Deployment identity does not exactly match the requested identity")
    if manifest["identity"]["training_config_sha256"] != manifest["training_config_sha256"]:
        raise ValueError("Source identity and export training configuration disagree")
    for filename_field, hash_field in (
        ("actor_file", "actor_sha256"),
        ("golden_file", "golden_sha256"),
        ("config_file", "config_file_sha256"),
    ):
        filename = manifest[filename_field]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Export files must be simple package-local file names")
        if file_sha256(root / filename) != manifest[hash_field]:
            raise ValueError(f"Export package file identity differs: {filename}")
    config = json.loads((root / manifest["config_file"]).read_text(encoding="utf-8"))
    validate_resolved_config(config, check_paths=False)
    if stable_hash(config) != manifest["training_config_sha256"] or config["contract"] != manifest["contract"]:
        raise ValueError("Export contract or source configuration identity is inconsistent")
    if guided != (config["schema_version"] == "RecoveryResolvedConfigV3"):
        raise ValueError("导出版本与训练配置版本不一致")
    training_actor = None
    if guided:
        training_contract = GuidanceTrainingContractV1.from_dict(manifest["training_contract"])
        guidance = GuidanceConfigV1.from_dict(manifest["guidance"])
        if (config["training_contract"] != manifest["training_contract"] or config["guidance"] != manifest["guidance"]
                or manifest["training_actor_input_dim"] != training_contract.actor_input_dim
                or manifest["external_input_dim"] != training_contract.external_input_dim
                or manifest["deployment_alpha"] != 0.0
                or manifest["identity"].get("training_contract_sha256") != training_contract.sha256
                or manifest["identity"].get("guidance_config_sha256") != guidance.sha256):
            raise ValueError("辅助导出契约、维度或配置身份不一致")
        filename = manifest["training_actor_file"]
        if (not isinstance(filename, str) or Path(filename).name != filename
                or file_sha256(root / filename) != manifest["training_actor_sha256"]):
            raise ValueError("辅助训练Actor文件身份无效")
        training_actor = torch.jit.load(str(root / filename), map_location=device).eval()
    if manifest["action_output"] != "raw_mean":
        raise ValueError("起身导出必须直接返回原始动作均值")
    if manifest["golden_batch_sizes"] != [1, 3, 32]:
        raise ValueError("Export must include batch 1, 3 and 32 golden validation")
    golden = torch.load(root / manifest["golden_file"], map_location="cpu", weights_only=True)
    if set(golden) != {"schema_version", "inputs", "outputs_by_batch_size", "provenance"}:
        raise ValueError("Golden vector fields are incomplete")
    if golden["schema_version"] != "RecoveryGoldenVectorsV1":
        raise ValueError("Unsupported golden vector schema")
    if stable_hash(golden["provenance"]) != manifest["golden_provenance_sha256"]:
        raise ValueError("Golden provenance identity is inconsistent")
    if golden["provenance"]["sha256"] != manifest["identity"]["pose_bank_sha256"]:
        raise ValueError("Golden source differs from the recorded source pose bank")
    inputs = golden["inputs"]
    _validate_inputs(inputs, manifest["external_input_dim"] if guided else contract.actor_input_dim)
    actor = torch.jit.load(str(root / manifest["actor_file"]), map_location=device).eval()
    if set(golden["outputs_by_batch_size"]) != {"1", "3", "32"}:
        raise ValueError("Golden output batch set is incomplete")
    with torch.inference_mode():
        for batch_size in manifest["golden_batch_sizes"]:
            expected = golden["outputs_by_batch_size"][str(batch_size)].to(device)
            actual = actor(inputs[:batch_size].to(device))
            if expected.shape != (batch_size, 6) or not torch.isfinite(expected).all():
                raise ValueError("Golden output has invalid dimensions or values")
            if actual.shape != expected.shape or not torch.isfinite(actual).all():
                raise ValueError("Loaded Actor produces invalid golden outputs")
            if float((actual - expected).abs().max()) > 1e-7:
                raise ValueError(f"Loaded Actor fails golden validation for batch {batch_size}")
            if training_actor is not None:
                states = inputs[:batch_size].to(device)
                training_mean = training_actor(torch.cat((states, torch.zeros_like(states[:, :1])), dim=1))
                if (training_mean.shape != actual.shape or not torch.isfinite(training_mean).all()
                        or float((training_mean - actual).abs().max()) > 1e-7):
                    raise ValueError("训练111维Actor与零辅助部署包装器不一致")
    return actor, manifest


def _validate_inputs(inputs: torch.Tensor, input_dim: int = 110) -> None:
    if not isinstance(inputs, torch.Tensor) or inputs.dtype != torch.float32:
        raise ValueError("Golden observations must be explicit float32 tensors")
    if inputs.ndim != 2 or inputs.shape[0] < 32 or inputs.shape[1] != input_dim:
        raise ValueError("Golden observations require at least 32 real 110-dimensional states")
    if not torch.isfinite(inputs).all():
        raise ValueError("Golden observations must contain finite values")


def _validate_golden_provenance(inputs: torch.Tensor, provenance: dict, identity: dict) -> None:
    _validate_inputs(inputs)
    if set(provenance) != {"kind", "path", "sha256", "record_ids"} or provenance["kind"] != "pose_bank":
        raise ValueError("Golden validation requires a verifiable recorded pose-bank source")
    expected, source = golden_inputs_from_pose_bank(provenance["path"], provenance["record_ids"])
    if source["sha256"] != provenance["sha256"] or source["sha256"] != identity["pose_bank_sha256"]:
        raise ValueError("Golden provenance differs from the checkpoint pose-bank identity")
    if inputs.shape != expected.shape or not torch.equal(inputs.detach().cpu(), expected):
        raise ValueError("Golden inputs differ from the normalized recorded physical histories")


def _load_mean_actor(state: dict, contract: RecoveryContractV2,
                     training_contract: GuidanceTrainingContractV1 | None = None) -> nn.Module:
    if contract.actor_hidden_dims != (256, 128, 64) or contract.critic_hidden_dims != (256, 128, 64):
        raise ValueError("RecoveryExportV2 requires the approved 256/128/64 Actor and Critic")
    with torch.random.fork_rng(devices=[]):
        actor = _build_mlp(training_contract.actor_input_dim if training_contract else contract.actor_input_dim,
                           6, contract.actor_hidden_dims)
        critic = _build_mlp(training_contract.critic_input_dim if training_contract else contract.critic_input_dim,
                            1, contract.critic_hidden_dims)
    expected = {f"actor.{key}" for key in actor.state_dict()}
    expected.update(f"critic.{key}" for key in critic.state_dict())
    expected.add("log_std")
    if set(state) != expected or state["log_std"].shape != (6,):
        raise ValueError("Checkpoint is not the approved RSL-RL log-noise ActorCritic")
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
           or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("RecoveryExportV2 requires float32 ActorCritic parameters")
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in state.items()
                           if key.startswith("actor.")}, strict=True)
    critic.load_state_dict({key.removeprefix("critic."): value for key, value in state.items()
                            if key.startswith("critic.")}, strict=True)
    wrapper = GuidanceRecoveryMeanActor(actor) if training_contract else RecoveryMeanActor(actor)
    return wrapper.eval()


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def _write_new_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    from .contracts import load_resolved_config

    parser = argparse.ArgumentParser(description="从真实起身checkpoint导出确定性均值网络")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--golden-pose-bank", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    args = parser.parse_args()
    config = load_resolved_config(args.config)
    run_manifest = args.run_manifest or args.checkpoint.parent / "run_manifest.json"
    identity = json.loads(run_manifest.read_text(encoding="utf-8"))["identity"]
    bank = args.golden_pose_bank or config["environment"]["pose_bank_path"]
    inputs, provenance = golden_inputs_from_pose_bank(bank)
    result = export_actor(args.checkpoint, args.output_dir, resolved_config=config, expected_identity=identity,
                          golden_inputs=inputs, golden_provenance=provenance)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
