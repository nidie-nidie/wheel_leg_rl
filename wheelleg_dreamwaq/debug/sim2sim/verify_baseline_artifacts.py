from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.export_dreamwaq_actor import _build_policy
from wheelleg_dreamwaq.deployment.dreamwaq_inference import DreamWaQInferenceActorV1
from wheelleg_dreamwaq.schemas.dreamwaq_manifest import (
    build_dreamwaq_algorithm_contract, build_dreamwaq_export_contract, dreamwaq_contract_versions,
    _base_task_payload, _with_hash,
)
from wheelleg_dreamwaq.schemas.normalization import NormalizationV2, RawActionNormalizationV1
from wheelleg_dreamwaq.schemas.randomization import FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import CommandRanges
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import ControlLimits
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RewardWeights
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.terminations import TerminationLimits
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import RAW_BASELINE_PROFILE, resolve_task_profile
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import load_dreamwaq_checkpoint_artifact
from wheelleg_dreamwaq.training.checkpoint import sha256_file


PROJECT = Path(__file__).resolve().parents[2]
SIM = PROJECT / "sim2sim/mujoco"


def rejected(operation):
    try:
        operation()
    except (ValueError, RuntimeError):
        return True
    raise AssertionError("应拒绝的契约被接受")


def old_contracts_unchanged(before: Path):
    spec = importlib.util.spec_from_file_location("baseline_original_manifest", before / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/dreamwaq_manifest.py")
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    export = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/exports/run-03"
    policy = json.loads((export / "policy_manifest.json").read_text(encoding="utf-8"))
    manifest_path = Path(policy["source_run_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    agent = yaml.safe_load((manifest_path.parent / "params/agent.yaml").read_text(encoding="utf-8"))
    assert build_dreamwaq_algorithm_contract(agent) == original.build_dreamwaq_algorithm_contract(agent)
    assert build_dreamwaq_export_contract() == original.build_dreamwaq_export_contract()
    base = manifest["base_task_contract"]
    task = base["task"]
    arguments = {"asset_bundle_hash": base["asset_bundle_hash"], "sim_dt": task["sim_dt"],
        "decimation": task["decimation"], "solver_position_iterations": task["physics"]["solver_position_iterations"],
        "solver_velocity_iterations": task["physics"]["solver_velocity_iterations"],
        "episode_length_s": task["episode_length_s"], "q_nominal": tuple(task["q_nominal"]),
        "control": ControlLimits(**task["control"]),
        "commands": CommandRanges(**{f.name: task["commands"][f.name] for f in fields(CommandRanges)}),
        "normalization": NormalizationV2(**base["normalization"]), "randomization": FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1,
        "reward_weights": RewardWeights(**task["reward_weights"]), "termination": TerminationLimits(**task["termination"]),
        "clip_actions": base["action"]["clip_actions"], "is_finite_horizon": task["is_finite_horizon"],
        "runtime": base["runtime"]}
    current = _with_hash(_base_task_payload(**arguments))
    assert current == original._with_hash(original._base_task_payload(**arguments)) == base
    return {"base": current["contract_hash"], "algorithm": build_dreamwaq_algorithm_contract(agent)["contract_hash"],
            "export": build_dreamwaq_export_contract()["contract_hash"], "payloads_exact_equal": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fresh, fresh_manifest, fresh_info = load_dreamwaq_checkpoint_artifact(args.fresh)
    resumed, manifest, info = load_dreamwaq_checkpoint_artifact(args.resumed)
    assert fresh_info["completed_iterations"] == 1 and info["completed_iterations"] == 2
    base = manifest["base_task_contract"]
    assert resolve_task_profile(None, base) == RAW_BASELINE_PROFILE
    assert fresh_manifest["base_task_contract"] == base
    assert info["reset_cache_binding"] == fresh_info["reset_cache_binding"]
    steps = [[float(v["step"]) for v in p["optimizer_state_dict"]["state"].values()] for p in (fresh, resumed)]
    assert set(steps[0]) == {40.} and set(steps[1]) == {80.}
    assert info["estimator_monitor_state"]["completed_rollouts"] == 2
    summary = json.loads((args.resumed.parent / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["starting_completed_iterations"] == 1 and summary["iterations_executed"] == 1
    assert summary["restored_learning_rate"] == fresh_info["algorithm_learning_rate"]
    trace = torch.load(args.resumed.parent / "resume_sequence_trace.pt", map_location="cpu", weights_only=False)
    torch.testing.assert_close(trace["first_policy_observation"][:, 6:9],
        RawActionNormalizationV1().normalize_command(trace["post_restore_command"].to(manifest["device"])).cpu(),
        rtol=0, atol=0)
    rejected_cases = {}
    for key, value in (("kp", 3.), ("enabled", False), ("orientation", "different"),
                       ("command_timing", "after_policy"), ("wheel_differential_gate", "source_command")):
        changed = deepcopy(base)
        changed["task"]["heading_hold"][key] = value
        rejected_cases[key] = rejected(lambda: dreamwaq_contract_versions(changed))
    changed = deepcopy(base)
    del changed["task"]["heading_hold"]
    rejected_cases["missing_heading"] = rejected(lambda: dreamwaq_contract_versions(changed))
    rejected_cases["cross_profile"] = rejected(lambda: resolve_task_profile("stop_reverse_v1", base))
    agent = yaml.safe_load((args.resumed.parent / "params/agent.yaml").read_text(encoding="utf-8"))
    agent["policy"]["decoder_action_clip"] = 1.
    rejected_cases["mixed_decoder_clip"] = rejected(lambda: build_dreamwaq_algorithm_contract(agent))
    subprocess.run([str(SIM / ".venv/Scripts/python.exe"), "-m", "debug.sim2sim.verify_heading_mujoco_artifacts",
                    "--export", str(args.export), "--output", str(args.output / "mujoco")], check=True, cwd=PROJECT)
    mujoco_report = json.loads((args.output / "mujoco/summary.json").read_text(encoding="utf-8"))
    rejected_cases.update(mujoco_report["rejected_contract_cases"])
    policy = _build_policy(resumed["model_state_dict"], raw_actions=True)
    eager = DreamWaQInferenceActorV1.from_policy(policy).eval()
    actor = torch.jit.load(str(args.export / "actor.ts"), map_location="cpu").eval()
    golden = torch.load(args.export / "golden_vectors.pt", weights_only=False)
    errors = {}
    with torch.inference_mode():
        for batch in (32, 1, 3):
            data = golden["history"][:batch]
            errors[str(batch)] = float((eager(data) - actor(data)).abs().max())
            assert errors[str(batch)] <= 1e-7
    vectors = torch.load(args.output / "mujoco/heading_vectors.pt", weights_only=False)
    prepared_batch = torch.as_tensor([v["history"].tolist() for v in vectors], dtype=torch.float32)
    with torch.inference_mode():
        errors["prepared_heading_vectors"] = float((eager(prepared_batch) - actor(prepared_batch)).abs().max())
    assert errors["prepared_heading_vectors"] <= 1e-7
    log = EventAccumulator(str(args.fresh.parent)).Reload()
    tags = log.Tags()["scalars"]
    required = ["Reward/wheel_differential", "Reward/action_smooth_leg", "Reward/action_smooth_wheel"]
    assert all(any(tag.endswith(key) for tag in tags) for key in required)
    assert all(math.isfinite(event.value) for tag in tags for event in log.Scalars(tag))
    report = {"schema_version": "RawHeadingBaselineArtifactVerificationV1", "golden_max_abs_error": errors,
              "optimizer_step_counts": [sorted(set(v)) for v in steps], "resume_first_command": "exact_equal",
              "reset_cache_reused": True, "rejected_contract_cases": rejected_cases,
              "heading_vectors": mujoco_report["heading_vectors"],
              "old_contracts": old_contracts_unchanged(args.before), "all_training_scalars_finite": True,
              "checkpoint_sha256": sha256_file(args.resumed)}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
