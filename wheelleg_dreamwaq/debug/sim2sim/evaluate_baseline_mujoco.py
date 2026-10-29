from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import wheelleg_dreamwaq.schemas.heading_control as heading_module
from debug.no_action_clip.mujoco_adapter import load_runtime
from scripts.evaluate_mujoco import _failure_reason, _write_csv
from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS, aggregate_run, summarize_scenario
from wheelleg_mujoco.model_semantics import stable_hash
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


PROJECT = Path(__file__).resolve().parents[2]
SIM = PROJECT / "sim2sim/mujoco"


def run_protocol(runtime, actor, name, commands, output, events=None):
    source = np.asarray(commands[0], dtype=np.float64)
    observation = runtime.reset(source)
    initial_yaw = runtime.last_reset_metrics["yaw_rad"]
    rows, valid = [], []
    failure = None
    for tick, values in enumerate(commands):
        source = np.asarray(values, dtype=np.float64)
        event = ""
        if events is not None and tick in events:
            event = events[tick]
            if event == "reset":
                observation = runtime.reset(source)
            else:
                runtime.set_heading_enabled(event == "heading_on")
        if runtime.heading is not None:
            observation = runtime.prepare_policy_observation(source)
        velocity_before = runtime.model_map.joint_velocity_control(runtime.data)[:4].copy()
        with torch.inference_mode():
            action = actor(torch.from_numpy(observation).view(1, -1)).numpy()[0]
        result = runtime.step(action, source)
        metrics = dict(result.metrics)
        metrics.setdefault("action_saturation_fraction", 0.0)
        effective = source if runtime.heading is None else runtime.effective_command
        metrics["effective_yaw_tracking_abs_error_rad_s"] = abs(metrics["yaw_rate_rad_s"] - effective[1])
        actual_velocity = runtime.model_map.joint_velocity_control(runtime.data)[:4].copy()
        acceleration = (actual_velocity - velocity_before) / runtime.contract.control_dt_s
        failure = _failure_reason(runtime, metrics)
        row = {"tick": tick + 1, "time_s": runtime.data.time, "event": event,
               "target_vx_mps": source[0], "target_yaw_rate_rad_s": source[1],
               "target_base_height_m": source[2], "effective_target_vx_mps": effective[0],
               "effective_target_yaw_rate_rad_s": effective[1],
               "effective_target_base_height_m": effective[2], **metrics,
               **{f"raw_action_{i}": float(v) for i, v in enumerate(action)},
               **{f"leg_velocity_{i}_rad_s": float(v) for i, v in enumerate(actual_velocity)},
               **{f"leg_acceleration_{i}_rad_s2": float(v) for i, v in enumerate(acceleration)},
               "failure_reason": failure or ""}
        rows.append(row)
        if failure is not None:
            break
        valid.append(row)
        observation = result.observation
    path = output / (name + ".csv")
    _write_csv(path, rows)
    summary = summarize_scenario(name, tuple(commands[0]), valid, expected_ticks=len(commands),
                                 failure_reason=failure, initial_yaw_rad=initial_yaw)
    summary["protocol"] = "command_transitions" if events is not None or len(set(commands)) > 1 else "fixed_command"
    summary["csv_sha256"] = sha256_file(path)
    summary["actual_yaw_rate_abs_mean_rad_s"] = float(np.mean([abs(r["yaw_rate_rad_s"]) for r in valid])) if valid else None
    summary["effective_yaw_tracking_mae_rad_s"] = float(np.mean([
        r["effective_yaw_tracking_abs_error_rad_s"] for r in valid])) if valid else None
    summary["leg_acceleration_rms_rad_s2"] = [float(np.sqrt(np.mean([
        r[f"leg_acceleration_{i}_rad_s2"] ** 2 for r in valid]))) for i in range(4)] if valid else None
    summary["heading_reference_error_rms_rad"] = float(np.sqrt(np.mean([
        np.arctan2(np.sin(r["heading_reference_rad"] - r["yaw_rad"]),
                   np.cos(r["heading_reference_rad"] - r["yaw_rad"])) ** 2
        for r in valid]))) if valid and runtime.heading is not None else None
    summary["heading_command_saturated_fraction"] = float(np.mean([
        r["heading_command_saturated"] for r in valid])) if valid and runtime.heading is not None else None
    summary["source_vx_tracking_mae_mps"] = float(np.mean([
        abs(r["vx_mps"] - r["target_vx_mps"]) for r in valid])) if valid else None
    return summary


def main():
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--formal-export", type=Path)
    inputs.add_argument("--debug-export", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heading-mode", choices=("both", "enabled", "disabled"), default="both")
    parser.add_argument("--transitions", action="store_true")
    args = parser.parse_args()
    export = (args.formal_export or args.debug_export).resolve()
    if args.formal_export:
        contract, metadata, model = load_policy_contract(export / "policy_manifest.json", export / "actor.ts",
                                                        SIM / "model_manifest.json")
        runtime = WheelLegMujocoRuntime(SIM / "models" / model["model_xml"]["path"], contract, model)
        checkpoint = Path(metadata["source_checkpoint"])
        expected_checkpoint_sha = metadata["source_checkpoint_sha256"]
        iterations = metadata["source_completed_iterations"]
    else:
        runtime = load_runtime(export)
        metadata = json.loads((export / "experiment_export.json").read_text(encoding="utf-8"))
        checkpoint = Path(metadata["source_checkpoint"])
        expected_checkpoint_sha = metadata["source_checkpoint_sha256"]
        iterations = metadata["completed_iterations"]
    if sha256_file(checkpoint) != expected_checkpoint_sha:
        raise ValueError("源 checkpoint 身份不符")
    actor = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    modes = ("enabled", "disabled") if args.heading_mode == "both" else (args.heading_mode,)
    if runtime.heading is None:
        if args.heading_mode != "disabled":
            raise ValueError("旧 profile 只接受显式 disabled 诊断")
        modes = ("disabled",)
    groups = {}
    for mode in modes:
        if runtime.heading is not None:
            runtime.heading.enabled = mode == "enabled"
        directory = output / mode
        directory.mkdir()
        summaries = [run_protocol(runtime, actor, name, [values] * 500, directory)
                     for name, values in FORMAL_SCENARIOS]
        transitions = []
        if args.transitions and runtime.heading is not None:
            protocols = {
                "forward_stop_reverse": [(1., 0., .2)] * 100 + [(0., 0., .2)] * 100
                    + [(-1., 0., .2)] * 100 + [(0., 0., .2)] * 100 + [(1., 0., .2)] * 100,
                "turn_release": [(0., .6, .2)] * 100 + [(1., 0., .2)] * 150
                    + [(0., -.6, .2)] * 100 + [(1., 0., .2)] * 150,
                "reset_toggle": [(0., 0., .2)] * 500,
            }
            for name, commands in protocols.items():
                events = {100: "heading_off", 200: "heading_on", 300: "reset"} if name == "reset_toggle" else {}
                transitions.append(run_protocol(runtime, actor, name, commands, directory, events))
        groups[mode] = {"aggregate": aggregate_run(expected_checkpoint_sha[:12], summaries),
                        "scenarios": summaries, "transitions": transitions}
    contract_record = {"scenarios": FORMAL_SCENARIOS, "ticks": 500, "control_dt_s": runtime.contract.control_dt_s,
                       "acceleration": "difference_of_real_joint_velocity_over_control_dt",
                       "heading_hold": runtime.contract.heading_hold, "heading_modes": modes,
                       "heading_source_sha256": sha256_file(Path(heading_module.__file__)),
                       "diagnostic_source_sha256": sha256_file(Path(__file__)),
                       "failure_source_sha256": sha256_file(PROJECT / "training/scripts/evaluate_mujoco.py"),
                       "runtime_source_sha256": {p.name: sha256_file(p) for p in (SIM / "wheelleg_mujoco").glob("*.py")},
                       "model_manifest_sha256": sha256_file(SIM / "model_manifest.json"),
                       "model_xml_sha256": sha256_file(runtime.model_path)}
    report = {"schema_version": "DreamWaQSingleModelDiagnosticV1", "export": str(export),
              "source_completed_iterations": iterations, "source_checkpoint_sha256": expected_checkpoint_sha,
              "export_files": {p.name: sha256_file(p) for p in export.iterdir() if p.is_file()},
              "evaluation_contract": contract_record, "evaluation_contract_hash": stable_hash(contract_record),
              "groups": groups}
    report["report_hash"] = stable_hash(report)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.json"), "completed_iterations": iterations,
                      "groups": {key: value["aggregate"] for key, value in groups.items()}}, indent=2))


if __name__ == "__main__":
    main()
