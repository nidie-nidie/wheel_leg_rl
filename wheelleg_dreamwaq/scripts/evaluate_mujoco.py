from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    FORMAL_SCENARIOS,
    aggregate_run,
    build_evaluation_contract,
    summarize_scenario,
)
from wheelleg_mujoco.model_semantics import stable_hash
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM2SIM_ROOT = PROJECT_ROOT / "sim2sim" / "mujoco"
DEFAULT_MODEL = SIM2SIM_ROOT / "models" / "wheel_leg_urdf4_v1.xml"
DEFAULT_MODEL_MANIFEST = SIM2SIM_ROOT / "model_manifest.json"
def _object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name is not None else f"unnamed_{object_id}"


def _unexpected_contact(runtime: WheelLegMujocoRuntime) -> str | None:
    allowed = {
        frozenset(("floor", "left_wheel_proxy")),
        frozenset(("floor", "right_wheel_proxy")),
    }
    for index in range(runtime.data.ncon):
        contact = runtime.data.contact[index]
        pair = frozenset(
            (
                _object_name(runtime.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                _object_name(runtime.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
            )
        )
        if pair not in allowed:
            return "+".join(sorted(pair))
    return None


def _failure_reason(runtime: WheelLegMujocoRuntime, metrics: dict[str, float]) -> str | None:
    if not np.isfinite(np.asarray(list(metrics.values()), dtype=np.float64)).all():
        return "non_finite_metric"
    if not 0.10 <= metrics["base_height_m"] <= 0.40:
        return "base_height"
    if metrics["tilt_rad"] > 0.80:
        return "tilt"
    if metrics["root_linear_speed_mps"] > 20.0:
        return "root_linear_velocity"
    if metrics["root_angular_speed_rad_s"] > 35.0:
        return "root_angular_velocity"
    if metrics["max_hinge_speed_rad_s"] > runtime.contract.passive_velocity_limit:
        return "joint_velocity"
    if metrics["max_loop_closure_error_m"] > 5.0e-3:
        return "loop_closure"
    if min(metrics["l0_left_m"], metrics["l0_right_m"]) <= 0.05:
        return "virtual_leg_length"
    unexpected = _unexpected_contact(runtime)
    return None if unexpected is None else f"unexpected_contact:{unexpected}"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic WheelLeg PPO evaluation in MuJoCo.")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Run 20 ticks of nominal stand only.")
    parser.add_argument("--suite-context", type=Path, default=None)
    parser.add_argument("--run-index", type=int, default=None)
    parser.add_argument("--completed-iterations", type=int, default=None)
    args = parser.parse_args()

    training_suite = None
    training_run = None
    if args.smoke:
        if args.suite_context is not None or args.run_index is not None or args.completed_iterations is not None:
            parser.error("--smoke cannot be combined with formal suite context arguments")
    else:
        if args.suite_context is None or args.run_index is None or args.completed_iterations is None:
            parser.error("Formal evaluation requires --suite-context, --run-index, and --completed-iterations")
        training_suite = json.loads(args.suite_context.resolve().read_text(encoding="utf-8"))
        context_hash = training_suite.get("context_hash")
        calculated_context_hash = stable_hash(
            {key: value for key, value in training_suite.items() if key != "context_hash"}
        )
        if context_hash != calculated_context_hash:
            raise ValueError("Training evaluation context hash is invalid")
        if (
            training_suite.get("schema_version") != "TrainingEvaluationContextV1"
            or training_suite.get("suite_mode") != "formal"
            or training_suite.get("run_count") != 4
            or training_suite.get("iterations_per_run") != 1000
        ):
            raise ValueError("Formal evaluation requires the frozen four-run 1000-iteration context")
        if not 1 <= args.run_index <= training_suite["run_count"]:
            raise ValueError("Formal evaluation run index is outside the suite")
        if args.completed_iterations != training_suite["iterations_per_run"]:
            raise ValueError("Formal evaluation checkpoint iteration count differs from the suite")
        training_run = {
            "schema_version": "TrainingEvaluationRunV1",
            "suite_id": training_suite["suite_id"],
            "run_index": args.run_index,
            "run_count": training_suite["run_count"],
            "completed_iterations": args.completed_iterations,
            "training_fingerprint_hash": training_suite["training_fingerprint_hash"],
            "suite_source_fingerprint_hash": training_suite["suite_source_fingerprint_hash"],
        }

    actor_path = args.policy.resolve()
    policy_manifest_path = args.manifest.resolve()
    model_path = args.model.resolve()
    model_manifest_path = args.model_manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    contract, policy_manifest, model_manifest = load_policy_contract(
        policy_manifest_path,
        actor_path,
        model_manifest_path,
    )
    if sha256_file(model_path) != model_manifest["model_xml"]["sha256"]:
        raise ValueError("MuJoCo XML hash differs from model manifest")
    policy = torch.jit.load(str(actor_path), map_location="cpu").eval()
    runtime = WheelLegMujocoRuntime(model_path, contract, model_manifest)
    scenarios = FORMAL_SCENARIOS[:1] if args.smoke else FORMAL_SCENARIOS
    expected_ticks = 20 if args.smoke else round(10.0 / contract.control_dt_s)
    evaluation_contract = build_evaluation_contract(
        policy_manifest=policy_manifest,
        model_manifest=model_manifest,
        scenarios=scenarios,
        expected_ticks=expected_ticks,
        smoke=args.smoke,
        training_suite=training_suite,
    )
    scenario_summaries = []
    scenario_files = {}

    for scenario_name, command_values in scenarios:
        command = np.asarray(command_values, dtype=np.float64)
        observation = runtime.reset(command)
        if runtime.last_reset_metrics is None:
            raise RuntimeError("MuJoCo runtime did not expose reset metrics")
        initial_yaw_rad = runtime.last_reset_metrics["yaw_rad"]
        if abs(initial_yaw_rad - evaluation_contract["reset_initial_yaw_rad"]) > 1.0e-8:
            raise ValueError("MuJoCo reset yaw differs from the evaluation contract")
        csv_rows = []
        valid_rows = []
        failure_reason = None
        for tick in range(expected_ticks):
            with torch.inference_mode():
                output_tensor = policy(torch.from_numpy(observation).unsqueeze(0))
            action = output_tensor.detach().cpu().numpy().reshape(-1)
            if action.shape != (6,) or not np.isfinite(action).all():
                failure_reason = "non_finite_action"
                break
            result = runtime.step(action, command)
            row = {
                "tick": tick + 1,
                "time_s": runtime.data.time,
                "target_vx_mps": command[0],
                "target_yaw_rate_rad_s": command[1],
                "target_base_height_m": command[2],
            }
            row.update(result.metrics)
            for index, value in enumerate(action):
                row[f"raw_action_{index}"] = float(value)
            for index, value in enumerate(result.clipped_action):
                row[f"clipped_action_{index}"] = float(value)
            for index, value in enumerate(result.applied_torque):
                row[f"applied_torque_{index}_nm"] = float(value)
            failure_reason = _failure_reason(runtime, result.metrics)
            row["survived"] = int(failure_reason is None)
            row["failure_reason"] = failure_reason or ""
            csv_rows.append(row)
            if failure_reason is not None:
                break
            valid_rows.append(result.metrics)
            observation = result.observation

        csv_path = output / f"{scenario_name}.csv"
        _write_csv(csv_path, csv_rows)
        scenario_files[scenario_name] = {
            "path": str(csv_path.resolve()),
            "sha256": sha256_file(csv_path),
        }
        summary = summarize_scenario(
            scenario_name,
            command_values,
            valid_rows,
            expected_ticks=expected_ticks,
            failure_reason=failure_reason,
            initial_yaw_rad=initial_yaw_rad,
        )
        scenario_summaries.append(summary)

    run_id = policy_manifest["source_checkpoint_sha256"][:12]
    aggregate = aggregate_run(run_id, scenario_summaries)
    report = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_contract": evaluation_contract,
        "evaluation_contract_hash": stable_hash(evaluation_contract),
        "smoke": args.smoke,
        "policy": str(actor_path),
        "policy_sha256": sha256_file(actor_path),
        "policy_manifest": str(policy_manifest_path),
        "policy_manifest_sha256": sha256_file(policy_manifest_path),
        "policy_manifest_hash": policy_manifest["manifest_hash"],
        "source_checkpoint": policy_manifest["source_checkpoint"],
        "source_checkpoint_sha256": policy_manifest["source_checkpoint_sha256"],
        "phase1_contract_hash": policy_manifest["phase1_contract_hash"],
        "model": str(model_path),
        "model_sha256": model_manifest["model_xml"]["sha256"],
        "model_manifest_sha256": policy_manifest["mujoco_model"]["model_manifest_sha256"],
        "expected_ticks_per_scenario": expected_ticks,
        "scenario_order": [name for name, _ in scenarios],
        "scenario_files": scenario_files,
        "training_run": training_run,
        "aggregate": aggregate,
    }
    report["report_hash"] = stable_hash(report)
    report_path = output / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
