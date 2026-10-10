"""Independent six-stage stop/reverse diagnostics; frozen eight-scenario evaluation stays separate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_mujoco import DEFAULT_MODEL, DEFAULT_MODEL_MANIFEST, _failure_reason, _write_csv
from wheelleg_mujoco.command_practice import command_at_tick, summarize_practice, validate_command_practice
from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.model_semantics import stable_hash
from wheelleg_mujoco.runner import WheelLegMujocoRuntime

SCENARIOS = (
    ("forward_stop_reverse", (0.5, 0.0, 0.20)),
    ("reverse_stop_forward", (-0.5, 0.0, 0.20)),
    ("left_stop_right", (0.0, 0.6, 0.20)),
    ("right_stop_left", (0.0, -0.6, 0.20)),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    contract, manifest, model_manifest = load_policy_contract(args.manifest, args.policy, args.model_manifest)
    payload = validate_command_practice(manifest)
    if payload is None:
        raise ValueError("Dynamic stop/reverse diagnostics require an enabled training practice profile")
    if contract.control_dt_s != payload["control_dt_s"]:
        raise ValueError("Runtime and command practice control clocks differ")
    if sha256_file(args.model) != model_manifest["model_xml"]["sha256"]:
        raise ValueError("MuJoCo XML identity differs")
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    runtime = WheelLegMujocoRuntime(args.model, contract, model_manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, scenario_files = [], {}
    for name, values in SCENARIOS:
        initial = np.asarray(values, dtype=np.float64)
        observation = runtime.reset(initial)
        valid_rows, csv_rows = [], []
        failure_reason = None
        for tick in range(20 if args.smoke else 500):
            current_command = command_at_tick(initial, tick, payload)
            next_command = command_at_tick(initial, tick + 1, payload)
            before_xy = runtime.data.xipos[runtime.model_map.base_body_id, :2].copy()
            with torch.inference_mode():
                action = policy(torch.from_numpy(observation).unsqueeze(0)).detach().cpu().numpy().reshape(-1)
            if action.shape != (6,) or not np.isfinite(action).all():
                failure_reason = "non_finite_action"
                break
            try:
                result = runtime.step(action, next_command)
            except FloatingPointError:
                failure_reason = "non_finite_metric"
                break
            after_xy = runtime.data.xipos[runtime.model_map.base_body_id, :2].copy()
            failure_reason = _failure_reason(runtime, result.metrics)
            row = {"tick": tick, "time_s": float(runtime.data.time),
                   "target_vx_mps": float(current_command[0]),
                   "target_yaw_rate_rad_s": float(current_command[1]),
                   "target_height_m": float(current_command[2]),
                   "next_target_vx_mps": float(next_command[0]),
                   "next_target_yaw_rate_rad_s": float(next_command[1]),
                   "com_x_before": float(before_xy[0]), "com_y_before": float(before_xy[1]),
                   "com_x_after": float(after_xy[0]), "com_y_after": float(after_xy[1]),
                   **result.metrics, "survived": int(failure_reason is None),
                   "failure_reason": failure_reason or ""}
            for index, value in enumerate(action):
                row[f"raw_action_{index}"] = float(value)
            csv_rows.append(row)
            if failure_reason is not None:
                break
            valid_rows.append({"tick": tick, **result.metrics,
                               "com_xy_before": before_xy.tolist(), "com_xy_after": after_xy.tolist()})
            observation = result.observation
        csv_path = args.output / f"{name}.csv"
        _write_csv(csv_path, csv_rows)
        scenario_files[name] = {"path": str(csv_path.resolve()), "sha256": sha256_file(csv_path)}
        summaries.append(summarize_practice(name, initial, valid_rows, failure_reason))
    report = {
        "schema_version": "MujocoCommandPracticeEvaluationV1", "smoke": args.smoke,
        "command_practice_contract": payload, "command_practice_contract_hash": stable_hash(payload),
        "policy_sha256": sha256_file(args.policy), "policy_manifest_sha256": sha256_file(args.manifest),
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "base_task_contract_hash": manifest["base_task_contract_hash"],
        "model_xml_sha256": model_manifest["model_xml"]["sha256"],
        "dynamics_semantics_hash": model_manifest["dynamics_semantics_hash"],
        "evaluation_source_sha256": sha256_file(Path(__file__)),
        "statistics_source_sha256": sha256_file(Path(__import__("wheelleg_mujoco.command_practice",
            fromlist=["__file__"]).__file__)),
        "valid_sample_rule": "survived_post_action_sample_against_current_executed_command",
        "scenario_files": scenario_files, "scenarios": summaries,
        "performance_accepted": not args.smoke and all(item["performance_accepted"] for item in summaries),
    }
    report["report_hash"] = stable_hash(report)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
