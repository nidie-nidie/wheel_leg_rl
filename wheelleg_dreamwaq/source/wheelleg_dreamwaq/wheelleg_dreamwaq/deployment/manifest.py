from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
    DREAMWAQ_EXPORT_CONTRACT_VERSION,
    GOLDEN_VECTOR_SEED,
    GOLDEN_VECTOR_SHAPE,
    TORCHSCRIPT_MAX_ABS_ERROR,
)


DREAMWAQ_POLICY_EXPORT_VERSION = "DreamWaQPolicyExportV1"
GOLDEN_VECTOR_VERSION = "DreamWaQGoldenVectorsV1"
EXPORT_ARTIFACT_NAMES = frozenset({"actor.ts", "policy_manifest.json", "golden_vectors.pt"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_manifest_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def _prepare_output_directory(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.iterdir()}
    if existing:
        raise FileExistsError(f"DreamWaQ export directory must be empty, found {sorted(existing)}")


def export_dreamwaq_inference_package(
    inference_actor: torch.nn.Module,
    output: Path,
    manifest_payload: dict[str, Any],
) -> dict[str, Any]:
    output = output.resolve()
    _prepare_output_directory(output)
    actor = inference_actor.cpu().eval()
    generator = torch.Generator(device="cpu").manual_seed(GOLDEN_VECTOR_SEED)
    history = torch.randn(GOLDEN_VECTOR_SHAPE, generator=generator, dtype=torch.float32, device="cpu").contiguous()
    with torch.inference_mode():
        expected = actor(history).contiguous()
    if expected.shape != (GOLDEN_VECTOR_SHAPE[0], 6) or expected.dtype != torch.float32:
        raise RuntimeError("DreamWaQ inference actor produced an invalid golden output")

    actor_path = output / "actor.ts"
    scripted = torch.jit.script(actor)
    scripted.save(str(actor_path))
    reloaded = torch.jit.load(str(actor_path), map_location="cpu").eval()
    with torch.inference_mode():
        actual = reloaded(history).contiguous()
        dynamic_one = reloaded(history[:1])
        dynamic_three = reloaded(history[:3])
    max_abs_error = float(torch.max(torch.abs(expected - actual)).item())
    if max_abs_error > TORCHSCRIPT_MAX_ABS_ERROR:
        raise RuntimeError(f"Reloaded DreamWaQ actor exceeds golden tolerance: {max_abs_error}")
    if dynamic_one.shape != (1, 6) or dynamic_three.shape != (3, 6):
        raise RuntimeError("Reloaded DreamWaQ actor does not support dynamic batch")

    golden_path = output / "golden_vectors.pt"
    torch.save(
        {
            "schema_version": GOLDEN_VECTOR_VERSION,
            "seed": GOLDEN_VECTOR_SEED,
            "history": history,
            "expected_action_mean": expected,
        },
        golden_path,
    )
    manifest = dict(manifest_payload)
    if "manifest_hash" in manifest:
        raise ValueError("DreamWaQ manifest payload must not predefine manifest_hash")
    if manifest.get("schema_version") not in (None, DREAMWAQ_POLICY_EXPORT_VERSION):
        raise ValueError("DreamWaQ manifest payload schema_version is invalid")
    if manifest.get("dreamwaq_export_contract_version") != DREAMWAQ_EXPORT_CONTRACT_VERSION:
        raise ValueError("DreamWaQ manifest payload export contract version is invalid")
    manifest.update(
        {
            "schema_version": DREAMWAQ_POLICY_EXPORT_VERSION,
            "actor_file": actor_path.name,
            "actor_sha256": sha256_file(actor_path),
            "golden_vectors_file": golden_path.name,
            "golden_vectors_sha256": sha256_file(golden_path),
            "export_verification_max_abs_error": max_abs_error,
            "torch_version": torch.__version__,
        }
    )
    manifest["manifest_hash"] = stable_manifest_hash(manifest)
    manifest_path = output / "policy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    actual_names = {path.name for path in output.iterdir()}
    if actual_names != EXPORT_ARTIFACT_NAMES:
        raise RuntimeError(f"DreamWaQ export directory contents differ: {sorted(actual_names)}")
    return {
        "actor": str(actor_path),
        "actor_sha256": manifest["actor_sha256"],
        "policy_manifest": str(manifest_path),
        "policy_manifest_sha256": sha256_file(manifest_path),
        "policy_manifest_hash": manifest["manifest_hash"],
        "golden_vectors": str(golden_path),
        "golden_vectors_sha256": manifest["golden_vectors_sha256"],
        "verification_max_abs_error": max_abs_error,
    }
