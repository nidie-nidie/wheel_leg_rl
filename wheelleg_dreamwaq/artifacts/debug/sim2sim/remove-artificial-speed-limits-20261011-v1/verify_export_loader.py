"""CPU verification of the current loader and live debug entry points; no training."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wheelleg_mujoco.contract import load_policy_contract
from debug.sim2sim.collect_dreamwaq_mujoco_trace import collect_dreamwaq_mujoco_trace
from debug.sim2sim.dreamwaq_debug_contract import TIMING_PROFILES
from debug.sim2sim.evaluate_debug_mujoco import evaluate_debug_mujoco


def main():
    export = ARTIFACT / "untrained-export-loader-smoke/export"
    model_manifest = ROOT / "sim2sim/mujoco/model_manifest.json"
    contract, manifest, _ = load_policy_contract(export / "policy_manifest.json", export / "actor.ts", model_manifest)
    assert manifest["schemas"]["physics"] == "PhysicsV5"
    assert manifest["source_completed_iterations"] == 0
    assert contract.policy_input_dimension == 125
    actor = torch.jit.load(str(export / "actor.ts"), map_location="cpu")
    golden = torch.load(export / "golden_vectors.pt", map_location="cpu", weights_only=False)
    with torch.inference_mode():
        max_error = float((actor(golden["history"]) - golden["expected_action_mean"]).abs().max())
    assert max_error == 0.0
    traces = {}
    for name in ("formal_1ms", "isaac_sync_5ms"):
        output = ARTIFACT / ("untrained-debug-trace-" + name)
        result = collect_dreamwaq_mujoco_trace(
            actor_path=export / "actor.ts", manifest_path=export / "policy_manifest.json",
            output_directory=output, timing=TIMING_PROFILES[name], control_ticks=2, mode="closed_loop",
        )
        with np.load(output / "substep_trace.npz") as arrays:
            assert not arrays["joint_velocity_limit_exceeded"].any()
            assert not arrays["mujoco_velocity_guard_active"].any()
        assert result["completed_control_ticks"] == 2
        traces[name] = {"completed_control_ticks": 2, "speed_limit_guard_events": 0}
    evaluation = evaluate_debug_mujoco(
        actor_path=export / "actor.ts", manifest_path=export / "policy_manifest.json",
        output_directory=ARTIFACT / "untrained-debug-evaluation", timing=TIMING_PROFILES["formal_1ms"],
        expected_ticks=2, scenarios=(("nominal_stand", (0.0, 0.0, 0.20)),),
    )
    assert evaluation["debug_only"] is True and evaluation["formal_ranking_eligible"] is False
    old_root = ROOT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2/exports"
    rejections = {}
    for run in ("run-01", "run-02", "run-03", "run-04"):
        old = old_root / run
        try:
            load_policy_contract(old / "policy_manifest.json", old / "actor.ts", model_manifest)
        except ValueError as error:
            assert "Unsupported policy manifest schema" in str(error)
            rejections[run] = str(error)
        else:
            raise AssertionError(f"Historical PhysicsV4 package {run} was accepted")
    report = {
        "verification_only": "fresh untrained weights, zero optimization updates, no performance claim",
        "current_formal_loader_accepted": True, "physics_schema_version": "PhysicsV5",
        "torchscript_golden_max_abs_error": max_error, "debug_traces": traces,
        "debug_evaluation_ran": True, "historical_packages_rejected": rejections,
    }
    (ARTIFACT / "export-loader-verification.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("EXPORT_LOADER_AND_DEBUG_VERIFIED", max_error)


if __name__ == "__main__":
    main()
