from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from wheelleg_mujoco.evaluation import (
    FORMAL_SCENARIOS,
    JOINT_SPEED_FAILURE_THRESHOLD_RAD_S,
    aggregate_run,
    summarize_scenario,
)
from wheelleg_mujoco.metrics import collect_metrics
from wheelleg_mujoco.observation import build_actor_observation, collect_kinematic_state
from wheelleg_mujoco.runner import StepResult, WheelLegMujocoRuntime

from .dreamwaq_debug_contract import (
    DebugPolicyAdapter,
    DebugTimingProfile,
    TIMING_PROFILES,
    contract_for_timing,
    write_debug_timing_model_copy,
)
from .trace_schema import sha256_file, stable_payload_hash, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_MODEL = PROJECT_ROOT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
MODEL_MANIFEST = PROJECT_ROOT / "sim2sim/mujoco/model_manifest.json"


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
    if metrics["max_hinge_speed_rad_s"] > JOINT_SPEED_FAILURE_THRESHOLD_RAD_S:
        return "joint_velocity"
    if metrics["max_loop_closure_error_m"] > 5.0e-3:
        return "loop_closure"
    if min(metrics["l0_left_m"], metrics["l0_right_m"]) <= 0.05:
        return "virtual_leg_length"
    unexpected = _unexpected_contact(runtime)
    return None if unexpected is None else f"unexpected_contact:{unexpected}"


def step_with_debug_timing(
    runtime: WheelLegMujocoRuntime,
    timing: DebugTimingProfile,
    action: np.ndarray,
    command: np.ndarray,
) -> StepResult:
    targets = runtime.controller.prepare(action)
    torque = np.zeros(6, dtype=np.float64)
    effort_saturation_count = 0
    for substep in range(timing.physics_steps_per_action):
        if timing.refreshes_at(substep):
            torque = runtime.controller.compute_torque(runtime.data, targets).copy()
        effort_saturation_count += int(
            np.isclose(np.abs(torque), runtime.controller.effort_limits).sum()
        )
        runtime.controller.apply_torque(runtime.data, torque)
        mujoco.mj_step(runtime.model, runtime.data)
    runtime.previous_action = targets.clipped_action.copy()
    state = collect_kinematic_state(runtime.model, runtime.data, runtime.model_map, runtime.contract)
    current_observation = build_actor_observation(
        state,
        command,
        runtime.previous_action,
        runtime.contract,
    )
    observation = runtime._append_policy_observation(current_observation)
    metrics = collect_metrics(runtime.model, runtime.data, runtime.model_map, runtime.contract, state)
    metrics.update(
        {
            "action_saturation_fraction": float(np.mean(np.abs(np.asarray(action)) >= 1.0)),
            "effort_saturation_fraction": effort_saturation_count
            / (timing.physics_steps_per_action * 6),
            "velocity_limit_event_fraction": 0.0,  # PhysicsV5: no speed-limit force/torque.
        }
    )
    return StepResult(
        observation=observation,
        clipped_action=runtime.previous_action.copy(),
        applied_torque=torque.copy(),
        metrics=metrics,
        physics_steps=timing.physics_steps_per_action,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_debug_mujoco(
    *,
    actor_path: str | Path,
    manifest_path: str | Path,
    output_directory: str | Path,
    timing: DebugTimingProfile,
    expected_ticks: int = 500,
    scenarios: Iterable[tuple[str, tuple[float, float, float]]] = FORMAL_SCENARIOS,
    model_manifest_path: str | Path = MODEL_MANIFEST,
    formal_model_path: str | Path = FORMAL_MODEL,
) -> dict:
    if expected_ticks <= 0:
        raise ValueError("expected_ticks must be positive")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    formal_model = Path(formal_model_path).resolve()
    formal_model_hash_before = sha256_file(formal_model)
    adapter = DebugPolicyAdapter(
        actor_path=actor_path,
        manifest_path=manifest_path,
        model_manifest_path=model_manifest_path,
    )
    debug_model = write_debug_timing_model_copy(
        formal_model,
        output / "models" / f"wheel_leg_{timing.name}_debug.xml",
        timing,
    )
    runtime = WheelLegMujocoRuntime(
        debug_model,
        contract_for_timing(adapter.contract, timing),
        model_manifest=None,
    )
    scenario_summaries = []
    scenario_files = {}
    scenario_list = tuple(scenarios)
    for scenario_name, command_values in scenario_list:
        command = np.asarray(command_values, dtype=np.float64)
        observation = runtime.reset(command)
        if runtime.last_reset_metrics is None:
            raise RuntimeError("Debug runtime did not expose reset metrics")
        initial_yaw_rad = runtime.last_reset_metrics["yaw_rad"]
        rows = []
        valid_metrics = []
        failure_reason = None
        for tick in range(expected_ticks):
            policy_output = adapter.infer(observation)
            result = step_with_debug_timing(
                runtime,
                timing,
                policy_output.raw_action,
                command,
            )
            row = {
                "tick": tick + 1,
                "time_s": float(runtime.data.time),
                "target_vx_mps": float(command[0]),
                "target_yaw_rate_rad_s": float(command[1]),
                "target_base_height_m": float(command[2]),
                **result.metrics,
            }
            for index, value in enumerate(policy_output.raw_action):
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
            valid_metrics.append(result.metrics)
            observation = result.observation
        csv_path = output / f"{scenario_name}.csv"
        _write_csv(csv_path, rows)
        scenario_files[scenario_name] = {
            "path": str(csv_path),
            "sha256": sha256_file(csv_path),
        }
        scenario_summaries.append(
            summarize_scenario(
                scenario_name,
                command_values,
                valid_metrics,
                expected_ticks=expected_ticks,
                failure_reason=failure_reason,
                initial_yaw_rad=initial_yaw_rad,
            )
        )
    aggregate = aggregate_run(adapter.manifest["source_checkpoint_sha256"][:12], scenario_summaries)
    report = {
        "schema_version": "DebugMujocoTimingEvaluationV1",
        "debug_only": True,
        "formal_ranking_eligible": False,
        "policy_kind": adapter.policy_kind,
        "policy": str(adapter.actor_path),
        "policy_sha256": sha256_file(adapter.actor_path),
        "policy_manifest": str(adapter.manifest_path),
        "policy_manifest_sha256": sha256_file(adapter.manifest_path),
        "source_checkpoint_sha256": adapter.manifest["source_checkpoint_sha256"],
        "formal_model": str(formal_model),
        "formal_model_sha256_before": formal_model_hash_before,
        "formal_model_sha256_after": sha256_file(formal_model),
        "debug_model": str(debug_model),
        "debug_model_sha256": sha256_file(debug_model),
        "timing_profile": {
            "name": timing.name,
            "physics_dt_s": timing.physics_dt_s,
            "physics_steps_per_action": timing.physics_steps_per_action,
            "torque_refresh_substeps": timing.torque_refresh_substeps,
            "control_dt_s": 0.020,
        },
        "expected_ticks_per_scenario": expected_ticks,
        "scenario_order": [name for name, _ in scenario_list],
        "scenario_files": scenario_files,
        "aggregate": aggregate,
    }
    if report["formal_model_sha256_before"] != report["formal_model_sha256_after"]:
        raise RuntimeError("Formal MuJoCo model changed during debug evaluation")
    report["report_hash"] = stable_payload_hash(report)
    write_json(output / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a non-formal MuJoCo timing diagnosis.")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-profile", choices=tuple(TIMING_PROFILES), required=True)
    parser.add_argument("--expected-ticks", type=int, default=500)
    args = parser.parse_args()
    report = evaluate_debug_mujoco(
        actor_path=args.policy,
        manifest_path=args.manifest,
        output_directory=args.output,
        timing=TIMING_PROFILES[args.timing_profile],
        expected_ticks=args.expected_ticks,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
