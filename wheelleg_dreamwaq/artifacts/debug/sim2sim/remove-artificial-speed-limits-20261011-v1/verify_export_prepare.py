"""Prepare a fresh, untrained PhysicsV5 package solely for export/loader verification."""
from pathlib import Path
import importlib.util
import json
from copy import deepcopy
import torch

from wheelleg_dreamwaq.deployment.dreamwaq_inference import DreamWaQInferenceActorV1
from wheelleg_dreamwaq.deployment.manifest import export_dreamwaq_inference_package
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
    build_dreamwaq_algorithm_contract,
    build_dreamwaq_export_contract,
)


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    config_fixture = load_module("config_fixture", ROOT / "tests/unit/test_dreamwaq_manifest.py")
    policy_fixture = load_module("policy_fixture", ROOT / "tests/unit/test_dreamwaq_export.py")
    exporter = load_module("exporter", ROOT / "scripts/export_dreamwaq_actor.py")
    base = json.loads((ARTIFACT / "isaac-randomized.json").read_text(encoding="utf-8"))["base_task_contract"]
    run = {
        "verification_only": "fresh untrained initialization; not a training checkpoint",
        "base_task_contract": base,
        "dreamwaq_algorithm_contract": build_dreamwaq_algorithm_contract(config_fixture._runner_config()),
        "dreamwaq_export_contract": build_dreamwaq_export_contract(),
    }
    work = ARTIFACT / "untrained-export-loader-smoke"
    work.mkdir(exist_ok=False)
    run_path = work / "verification-run-manifest.json"
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")
    torch.manual_seed(20261011)
    policy = policy_fixture._policy()
    checkpoint = work / "untrained-initialization.pt"
    torch.save({"verification_only": True, "model_state_dict": policy.state_dict()}, checkpoint)
    model_manifest = json.loads(exporter.MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload = exporter._manifest_payload(
        checkpoint=checkpoint,
        run_manifest_path=run_path,
        run_manifest=run,
        metadata={"completed_iterations": 0, "seed": 20261011, "hardware_profile": {"name": "verification_only_cpu"}},
        model_manifest=model_manifest,
    )
    old_run = deepcopy(run)
    old_run["base_task_contract"]["schemas"]["physics"] = "PhysicsV4"
    try:
        exporter._manifest_payload(
            checkpoint=checkpoint, run_manifest_path=run_path, run_manifest=old_run,
            metadata={"completed_iterations": 0}, model_manifest=model_manifest,
        )
    except ValueError as error:
        assert "old checkpoints cannot be relabeled" in str(error)
    else:
        raise AssertionError("DreamWaQ exporter relabeled old physics")
    payload["verification_only"] = "untrained PhysicsV5 metadata/loader smoke; no performance claim"
    summary = export_dreamwaq_inference_package(
        DreamWaQInferenceActorV1.from_policy(policy), work / "export", payload,
    )
    assert summary["verification_max_abs_error"] == 0.0
    (ARTIFACT / "export-prepare-verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("UNTRAINED_EXPORT_PREPARED", summary["verification_max_abs_error"])


if __name__ == "__main__":
    main()
