from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim2sim" / "mujoco"))
from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS
from wheelleg_mujoco.model_semantics import stable_hash
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from evaluate_mujoco import _failure_reason

torch.set_num_threads(1)
SCENARIOS = FORMAL_SCENARIOS
LEG_NAMES = ("jIJ", "jIO", "jAB", "jAG")
METRIC_FIELDS = (
    "base_height_m", "roll_rad", "pitch_rad", "yaw_rad", "vx_mps", "yaw_rate_rad_s",
    "phi0_left_rad", "phi0_right_rad", "phi0_delta_abs_rad", "l0_left_m", "l0_right_m",
    "max_loop_closure_error_m", "tilt_rad", "root_linear_speed_mps",
    "root_angular_speed_rad_s", "max_controlled_joint_speed_rad_s", "max_hinge_speed_rad_s",
    "action_saturation_fraction", "effort_saturation_fraction",
)
CSV_FIELDS = ("tick", "time_s", "target_vx_mps", "target_yaw_rate_rad_s",
              "target_base_height_m", *METRIC_FIELDS,
              *(f"{joint}_{suffix}" for joint in LEG_NAMES
                for suffix in ("q_rad", "qdot_rad_s", "qddot_rad_s2", "target_rad")),
              "survived", "failure_reason")


def _failed_row(runtime, command, tick, reason):
    return {"tick": tick + 1, "time_s": float(runtime.data.time),
            "target_vx_mps": float(command[0]),
            "target_yaw_rate_rad_s": float(command[1]),
            "target_base_height_m": float(command[2]),
            "survived": 0, "failure_reason": reason}



def _step_or_failure(runtime, action, command):
    try:
        return runtime.step(action, command), None
    except FloatingPointError:
        return None, "non_finite_state"


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _window_stats(values: np.ndarray, window: int) -> dict:
    if values.shape[0] < window:
        return {"window_count": 0, "rms_mean": [None] * 4, "rms_p90": [None] * 4}
    rms = np.stack(
        [np.sqrt(np.mean(values[start : start + window] ** 2, axis=0))
         for start in range(0, values.shape[0] - window + 1, window)],
        axis=0,
    )
    return {"window_count": len(rms), "rms_mean": rms.mean(axis=0).tolist(), "rms_p90": np.percentile(rms, 90, axis=0).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo-only actual leg acceleration budget evaluator.")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"))
    parser.add_argument("--model-manifest", type=Path, default=Path("sim2sim/mujoco/model_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--warmup-ticks", type=int, default=100)
    parser.add_argument("--window-ticks", type=int, default=50)
    parser.add_argument("--budget-fraction", type=float, default=None)
    args = parser.parse_args()
    if args.warmup_ticks < 0 or args.ticks <= args.warmup_ticks or args.window_ticks <= 0:
        parser.error("ticks must exceed warmup-ticks and window-ticks must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    actor_path = args.policy.resolve()
    manifest_path = args.manifest.resolve()
    model_path = args.model.resolve()
    model_manifest_path = args.model_manifest.resolve()
    contract, policy_manifest, model_manifest = load_policy_contract(
        manifest_path, actor_path, model_manifest_path
    )
    if sha256_file(model_path) != model_manifest["model_xml"]["sha256"]:
        raise ValueError("MuJoCo XML hash differs from model manifest")
    acceleration_contract = policy_manifest.get("training_leg_acceleration")
    saved_fraction = acceleration_contract["budget_fraction"] if acceleration_contract else None
    if args.budget_fraction is not None and args.budget_fraction != saved_fraction:
        raise ValueError("budget fraction must match saved training contract")
    policy = torch.jit.load(str(actor_path), map_location="cpu").eval()
    runtime = WheelLegMujocoRuntime(model_path, contract, model_manifest)
    scenario_summaries = []
    scenario_files = {}

    for name, command_values in SCENARIOS:
        command = np.asarray(command_values, dtype=np.float64)
        observation = runtime.reset(command)
        rows = []
        qddot_rows = []
        failure_reason = None
        for tick in range(args.ticks):
            with torch.inference_mode():
                action = policy(torch.from_numpy(observation).unsqueeze(0)).cpu().numpy().reshape(-1)
            if action.shape != (6,) or not np.isfinite(action).all():
                failure_reason = "invalid_action"
                rows.append(_failed_row(runtime, command, tick, failure_reason))
                break
            qdot_before = runtime.model_map.joint_velocity_mujoco(runtime.data).copy()
            targets = runtime.controller.prepare(action).leg_position_target_mujoco.copy()
            result, failure_reason = _step_or_failure(runtime, action, command)
            if failure_reason is not None:
                rows.append(_failed_row(runtime, command, tick, failure_reason))
                break
            q_after = runtime.model_map.joint_position_mujoco(runtime.data).copy()
            qdot_after = runtime.model_map.joint_velocity_mujoco(runtime.data).copy()
            qddot = (qdot_after - qdot_before) / contract.control_dt_s
            row: dict[str, float | int | str] = {
                "tick": tick + 1,
                "time_s": float(runtime.data.time),
                "target_vx_mps": float(command[0]),
                "target_yaw_rate_rad_s": float(command[1]),
                "target_base_height_m": float(command[2]),
            }
            row.update({key: float(value) for key, value in result.metrics.items()})
            for index, joint in enumerate(LEG_NAMES):
                row[f"{joint}_q_rad"] = float(q_after[index])
                row[f"{joint}_qdot_rad_s"] = float(qdot_after[index])
                row[f"{joint}_qddot_rad_s2"] = float(qddot[index])
                row[f"{joint}_target_rad"] = float(targets[index])
            row["action_saturation_fraction"] = float(np.mean(np.abs(action) >= 0.999))
            failure_reason = _failure_reason(runtime, result.metrics)
            row["survived"] = int(failure_reason is None)
            row["failure_reason"] = failure_reason or ""
            rows.append(row)
            if failure_reason is not None:
                break
            if tick >= args.warmup_ticks:
                qddot_rows.append(qddot[:4].copy())
            observation = result.observation

        csv_path = output / f"{name}.csv"
        _write_csv(csv_path, rows)
        scenario_files[name] = {"path": str(csv_path), "sha256": sha256_file(csv_path)}
        values = np.asarray(qddot_rows, dtype=np.float64)
        stats = _window_stats(values, args.window_ticks)
        valid = [row for row in rows if row["survived"]]
        vx_mae = float(np.mean([abs(float(row["vx_mps"]) - command[0]) for row in valid])) if valid else None
        yaw_mae = float(np.mean([abs(float(row["yaw_rate_rad_s"]) - command[1]) for row in valid])) if valid else None
        scenario_summaries.append({
            "scenario": name,
            "command": list(map(float, command_values)),
            "survived": failure_reason is None and len(rows) == args.ticks,
            "failure_reason": failure_reason,
            "valid_ticks": len(valid),
            "vx_mae_mps": vx_mae,
            "yaw_rate_mae_rad_s": yaw_mae,
            "steady_vx_mae_mps": float(np.mean([abs(float(row["vx_mps"]) - command[0])
                for row in valid if row["tick"] > args.warmup_ticks]))
                if any(row["tick"] > args.warmup_ticks for row in valid) else None,
            "actual_acceleration_rms_mean_rad_s2": stats["rms_mean"],
            "actual_acceleration_rms_p90_rad_s2": stats["rms_p90"],
            "metric_valid": stats["window_count"] > 0,
            "window_count": stats["window_count"],
            "actual_acceleration_rms_aggregate_mean_rad_s2": float(np.mean(stats["rms_mean"])) if stats["window_count"] else None,
            "action_saturation_fraction": float(np.mean([row["action_saturation_fraction"] for row in valid])) if valid else None,
            "effort_saturation_fraction": float(np.mean([row["effort_saturation_fraction"] for row in valid])) if valid else None,
            "max_tilt_rad": max([row["tilt_rad"] for row in valid], default=None),
        })

    report = {
        "schema_version": "MujocoLegAccelerationBudgetEvaluationV1",
        "diagnostic_only": True,
        "isaac_used_for_data_collection": False,
        "muJoCo_scenarios": [name for name, _ in SCENARIOS],
        "control_dt_s": float(contract.control_dt_s),
        "ticks": args.ticks,
        "warmup_ticks": args.warmup_ticks,
        "window_ticks": args.window_ticks,
        "budget_fraction": saved_fraction,
        "training_acceleration_contract": acceleration_contract,
        "base_task_hash": policy_manifest.get("base_task_contract_hash"),
        "source_checkpoint": policy_manifest.get("source_checkpoint"),
        "source_checkpoint_sha256": policy_manifest.get("source_checkpoint_sha256"),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "failure_rules_sha256": sha256_file(Path(__file__).with_name("evaluate_mujoco.py")),
        "policy": str(actor_path),
        "policy_sha256": sha256_file(actor_path),
        "policy_manifest_sha256": sha256_file(manifest_path),
        "model_sha256": model_manifest["model_xml"]["sha256"],
        "scenario_files": scenario_files,
        "scenarios": scenario_summaries,
    }
    report["report_hash"] = stable_hash(report)
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

