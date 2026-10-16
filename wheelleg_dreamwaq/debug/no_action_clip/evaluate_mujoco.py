from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS, aggregate_run, summarize_scenario

from debug.no_action_clip.contract import sha256
from debug.no_action_clip.mujoco_adapter import RawActionRuntime, load_runtime


JOINT_SPEED_FAILURE_THRESHOLD_RAD_S = 80.0
EXPECTED_TICKS = 500


def _object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name is not None else f"unnamed_{object_id}"


def _unexpected_contact(runtime: RawActionRuntime) -> str | None:
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


def _failure_reason(runtime: RawActionRuntime, metrics: dict[str, float]) -> str | None:
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
    if metrics["max_hinge_speed_rad_s"] > JOINT_SPEED_FAILURE_THRESHOLD_RAD_S:
        return "joint_velocity"
    if metrics["max_loop_closure_error_m"] > 5.0e-3:
        return "loop_closure"
    if min(metrics["l0_left_m"], metrics["l0_right_m"]) <= 0.05:
        return "virtual_leg_length"
    unexpected = _unexpected_contact(runtime)
    return None if unexpected is None else f"unexpected_contact:{unexpected}"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DebugNoActionClipV2 MuJoCo evaluation.")
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    export_dir = args.export.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(export_dir)
    policy = torch.jit.load(str(export_dir / "actor.ts"), map_location="cpu").eval()
    metadata = json.loads((export_dir / "experiment_export.json").read_text(encoding="utf-8"))

    summaries = []
    scenario_files = {}
    for scenario_name, command_values in FORMAL_SCENARIOS:
        command = np.asarray(command_values, dtype=np.float64)
        observation = runtime.reset(command)
        if runtime.last_reset_metrics is None:
            raise RuntimeError("MuJoCo runtime did not expose reset metrics")
        initial_yaw_rad = float(runtime.last_reset_metrics["yaw_rad"])
        rows = []
        valid_rows = []
        failure_reason = None
        for tick in range(EXPECTED_TICKS):
            with torch.inference_mode():
                action_tensor = policy(torch.from_numpy(observation).unsqueeze(0))
            action = action_tensor.detach().cpu().numpy().reshape(-1)
            if action.shape != (6,) or not np.isfinite(action).all():
                failure_reason = "non_finite_action"
                break
            result = runtime.step(action, command)
            row = {
                "tick": tick + 1,
                "time_s": float(runtime.data.time),
                "target_vx_mps": float(command[0]),
                "target_yaw_rate_rad_s": float(command[1]),
                "target_base_height_m": float(command[2]),
                "action_saturation_fraction": float(np.mean(np.abs(action) >= 1.0)),
            }
            row.update(result.metrics)
            row["survived"] = 0
            for index, value in enumerate(action):
                row[f"raw_action_{index}"] = float(value)
            for index, value in enumerate(result.clipped_action):
                row[f"clipped_action_{index}"] = float(value)
            for index, value in enumerate(result.applied_torque):
                row[f"applied_torque_{index}_nm"] = float(value)
            failure_reason = _failure_reason(runtime, result.metrics)
            row["survived"] = int(failure_reason is None)
            row["failure_reason"] = failure_reason or ""
            rows.append(row)
            if failure_reason is not None:
                break
            valid_rows.append(row)
            observation = result.observation

        csv_path = output / f"{scenario_name}.csv"
        _write_csv(csv_path, rows)
        scenario_files[scenario_name] = {
            "path": str(csv_path),
            "sha256": sha256(csv_path),
        }
        summaries.append(
            summarize_scenario(
                scenario_name,
                command_values,
                valid_rows,
                expected_ticks=EXPECTED_TICKS,
                failure_reason=failure_reason,
                initial_yaw_rad=initial_yaw_rad,
            )
        )

    aggregate = aggregate_run(metadata["source_checkpoint_sha256"][:12], summaries)
    report = {
        "schema_version": "DebugNoActionClipMuJoCoEvaluationV1",
        "experiment_id": metadata["experiment_id"],
        "completed_iterations": metadata["completed_iterations"],
        "export": str(export_dir),
        "actor_sha256": sha256(export_dir / "actor.ts"),
        "expected_ticks_per_scenario": EXPECTED_TICKS,
        "scenario_order": [name for name, _ in FORMAL_SCENARIOS],
        "scenario_files": scenario_files,
        "aggregate": aggregate,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
