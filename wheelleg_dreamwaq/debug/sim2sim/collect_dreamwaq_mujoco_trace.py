from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from wheelleg_mujoco.observation import build_actor_observation
from wheelleg_mujoco.runner import WheelLegMujocoRuntime

from .collect_mujoco_trace import (
    TERMINATION_REASON_CODES,
    _aggregate_torque,
    _capture_state,
    _diagnostic_flags,
    _native_termination_reason,
    _resolve_all_hinges,
    _rows_to_arrays,
    _substep_row,
    _torque_reference,
)
from .dreamwaq_debug_contract import (
    DebugPolicyAdapter,
    DebugTimingProfile,
    TIMING_PROFILES,
    contract_for_timing,
    write_debug_timing_model_copy,
)
from .trace_schema import (
    engine_native_to_canonical,
    load_action_sequence,
    save_npz,
    sha256_file,
    validate_control_trace,
    validate_substep_trace,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_MODEL = PROJECT_ROOT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
MODEL_MANIFEST = PROJECT_ROOT / "sim2sim/mujoco/model_manifest.json"
COMMAND = np.asarray((0.0, 0.0, 0.20), dtype=np.float64)


def _select_action(
    mode: str,
    tick: int,
    policy_action: np.ndarray,
    sequence: np.ndarray | None,
) -> np.ndarray:
    if mode == "closed_loop":
        return policy_action
    if mode == "zero_action":
        return np.zeros(6, dtype=np.float32)
    if sequence is None or tick >= len(sequence):
        raise ValueError(f"Action sequence is missing tick {tick}")
    return sequence[tick]


def collect_dreamwaq_mujoco_trace(
    *,
    actor_path: str | Path,
    manifest_path: str | Path,
    output_directory: str | Path,
    timing: DebugTimingProfile,
    control_ticks: int,
    mode: str,
    action_sequence: np.ndarray | None = None,
    action_sequence_identity: dict[str, Any] | None = None,
    command: np.ndarray = COMMAND,
) -> dict[str, Any]:
    if mode not in {"closed_loop", "zero_action", "action_replay"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if not 1 <= control_ticks <= 500:
        raise ValueError("control_ticks must be in [1, 500]")
    if mode == "action_replay" and action_sequence is None:
        raise ValueError("action_replay requires an action sequence")
    if action_sequence is not None:
        action_sequence = np.asarray(action_sequence)
        if action_sequence.dtype != np.float32 or action_sequence.ndim != 2 or action_sequence.shape[1] != 6:
            raise ValueError("Action sequence must have shape (N, 6) and dtype float32")
        if len(action_sequence) < control_ticks or not np.isfinite(action_sequence).all():
            raise ValueError("Action sequence is too short or non-finite")

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    formal_hash_before = sha256_file(FORMAL_MODEL)
    adapter = DebugPolicyAdapter(
        actor_path=actor_path,
        manifest_path=manifest_path,
        model_manifest_path=MODEL_MANIFEST,
    )
    debug_model = write_debug_timing_model_copy(
        FORMAL_MODEL,
        output / "models" / f"wheel_leg_{timing.name}_debug.xml",
        timing,
    )
    runtime = WheelLegMujocoRuntime(
        debug_model,
        contract_for_timing(adapter.contract, timing),
        model_manifest=None,
    )
    all_hinge_names = list(adapter.model_manifest["joint_order"])
    all_hinge_qpos, all_hinge_dof = _resolve_all_hinges(runtime.model, all_hinge_names)
    command_values = np.asarray(command, dtype=np.float64)
    if command_values.shape != (3,):
        raise ValueError("Command must have shape (3,)")

    runtime_policy_input = runtime.reset(command_values)
    initial_com_world = np.asarray(runtime.data.xipos[runtime.model_map.base_body_id]).copy()
    reset_state = _capture_state(runtime, all_hinge_qpos, all_hinge_dof, initial_com_world)
    reset_current = build_actor_observation(
        reset_state["kinematic"], command_values, runtime.previous_action, runtime.contract
    )
    policy_input = adapter.initialize_policy_input(reset_current)
    if not np.array_equal(runtime_policy_input, policy_input):
        raise RuntimeError("Debug policy history differs from the formal MuJoCo history adapter")
    write_json(
        output / "reset_snapshot.json",
        {
            "schema_version": "DreamWaQSim2SimTraceV1",
            "engine": "mujoco",
            "current_actor_observation": reset_current.tolist(),
            "policy_input": policy_input.tolist(),
            "history_frames_identical": bool(
                adapter.policy_kind != "dreamwaq"
                or np.array_equal(policy_input.reshape(5, 25), np.tile(reset_current, (5, 1)))
            ),
            "active_joint_position_canonical": reset_state[
                "active_joint_position_canonical"
            ].tolist(),
            "base_com_position_engine_world": reset_state[
                "base_com_position_engine_world"
            ].tolist(),
        },
    )

    control_rows: list[dict[str, Any]] = []
    substep_rows: list[dict[str, Any]] = []
    stopped_reason = 0
    for tick in range(control_ticks):
        previous_action = runtime.previous_action.copy()
        pre_control = _capture_state(runtime, all_hinge_qpos, all_hinge_dof, initial_com_world)
        current_observation = build_actor_observation(
            pre_control["kinematic"], command_values, previous_action, runtime.contract
        )
        if not np.array_equal(current_observation, adapter.current_observation(policy_input)):
            raise RuntimeError("MuJoCo current observation differs from the latest history frame")
        inference = adapter.infer(policy_input)
        selected_action = _select_action(mode, tick, inference.raw_action, action_sequence)
        targets = runtime.controller.prepare(selected_action)
        target_native = np.concatenate(
            (targets.leg_position_target_mujoco, targets.wheel_velocity_target_mujoco)
        )
        target_canonical = engine_native_to_canonical(target_native)

        unclipped_rows: list[np.ndarray] = []
        clipped_rows: list[np.ndarray] = []
        commanded_rows: list[np.ndarray] = []
        host_nan_rows: list[np.ndarray] = []
        effort_events: list[np.ndarray] = []
        velocity_events: list[np.ndarray] = []
        guard_events: list[np.ndarray] = []
        final_post: dict[str, Any] | None = None
        commanded_native = np.zeros(6, dtype=np.float64)
        for substep in range(timing.physics_steps_per_action):
            pre = _capture_state(runtime, all_hinge_qpos, all_hinge_dof, initial_com_world)
            unclipped_native, clipped_native = _torque_reference(runtime, pre, targets)
            if timing.refreshes_at(substep):
                commanded_native = runtime.controller.compute_torque(runtime.data, targets).copy()
            effort_event = (np.abs(unclipped_native) > runtime.contract.effort_limits).astype(np.int8)
            # PhysicsV5 has no joint-speed limiter; retain explicit inactive trace fields.
            velocity_event = np.zeros(6, dtype=np.int8)
            guard_event = np.zeros(6, dtype=np.int8)
            runtime.controller.apply_torque(runtime.data, commanded_native)
            runtime.data.xfrc_applied[:] = 0.0
            mujoco.mj_step(runtime.model, runtime.data)
            post = _capture_state(runtime, all_hinge_qpos, all_hinge_dof, initial_com_world)
            final_post = post
            substep_rows.append(
                _substep_row(
                    tick=tick,
                    substep=substep,
                    time_s=runtime.data.time,
                    pre=pre,
                    post=post,
                    target_canonical=target_canonical,
                    target_native=target_native,
                    unclipped_native=unclipped_native,
                    clipped_native=clipped_native,
                    commanded_native=commanded_native.copy(),
                    effort_event=effort_event,
                    velocity_event=velocity_event,
                    guard_event=guard_event,
                    external_torque_control=np.zeros(3),
                    external_torque_world=np.zeros(3),
                )
            )
            unclipped_rows.append(unclipped_native)
            clipped_rows.append(clipped_native)
            commanded_rows.append(commanded_native.copy())
            host_nan_rows.append(np.full(6, np.nan))
            effort_events.append(effort_event)
            velocity_events.append(velocity_event)
            guard_events.append(guard_event)
        if final_post is None:
            raise RuntimeError("MuJoCo executed no physics substeps")

        runtime.previous_action = targets.clipped_action.copy()
        next_current = build_actor_observation(
            final_post["kinematic"], command_values, runtime.previous_action, runtime.contract
        )
        diagnostic_flags = _diagnostic_flags(final_post)
        reason_code = _native_termination_reason(final_post)
        next_policy_input = adapter.advance_policy_input(
            policy_input,
            next_current,
            done=reason_code != 0,
        )
        row: dict[str, Any] = {
            "control_tick": np.int64(tick),
            "control_time_s": np.float64(runtime.data.time),
            "command": command_values,
            "actor_obs_current_pre_step": current_observation,
            "actor_obs_policy_pre_step": policy_input,
            "previous_action_before_inference": previous_action,
            "cenet_estimated_velocity": inference.estimated_velocity,
            "cenet_context_mu": inference.context_mu,
            "cenet_context_logvar": inference.context_logvar,
            "actor_output_raw": inference.raw_action,
            "action_clipped": targets.clipped_action,
            "target_command_canonical": target_canonical,
            "target_command_engine_native": target_native,
            "active_joint_position_canonical_post_step_pre_reset": final_post[
                "active_joint_position_canonical"
            ],
            "active_joint_velocity_canonical_post_step_pre_reset": final_post[
                "active_joint_velocity_canonical"
            ],
            "all_hinge_position_named_post_step_pre_reset": final_post[
                "all_hinge_position_named"
            ],
            "all_hinge_velocity_named_post_step_pre_reset": final_post[
                "all_hinge_velocity_named"
            ],
            "base_com_position_diag_post_step_pre_reset": final_post["base_com_position_diag"],
            "base_linear_velocity_control_post_step_pre_reset": final_post[
                "base_linear_velocity_control"
            ],
            "base_angular_velocity_control_post_step_pre_reset": final_post[
                "base_angular_velocity_control"
            ],
            "projected_gravity_post_step_pre_reset": final_post["projected_gravity"],
            "base_height_post_step_pre_reset": final_post["base_height"],
            "virtual_leg_length_post_step_pre_reset": final_post["virtual_leg_length"],
            "virtual_leg_phi0_post_step_pre_reset": final_post["virtual_leg_phi0"],
            "loop_closure_error_post_step_pre_reset": final_post["loop_closure_error"],
            "effort_limit_event": np.max(np.stack(effort_events), axis=0),
            "joint_velocity_limit_exceeded": np.max(np.stack(velocity_events), axis=0),
            "mujoco_velocity_guard_active": np.max(np.stack(guard_events), axis=0),
            "native_terminated_int8": np.int8(reason_code != 0),
            "native_truncated_int8": np.int8(0),
            "native_termination_reason_code_int16": np.int16(reason_code),
            "common_diagnostic_flags_int8": diagnostic_flags,
            "next_actor_obs_current_returned": next_current,
            "next_actor_obs_policy_returned": next_policy_input,
            "next_obs_is_reset_int8": np.int8(reason_code != 0),
        }
        _aggregate_torque(
            row, "pd_torque_unclipped", unclipped_rows, timing.physics_dt_s
        )
        _aggregate_torque(
            row, "pd_torque_effort_clipped", clipped_rows, timing.physics_dt_s
        )
        _aggregate_torque(
            row, "mujoco_commanded_torque", commanded_rows, timing.physics_dt_s
        )
        _aggregate_torque(
            row, "isaac_host_pd_torque_estimate", host_nan_rows, timing.physics_dt_s
        )
        control_rows.append(row)
        policy_input = next_policy_input
        if reason_code:
            stopped_reason = reason_code
            break

    control_trace = _rows_to_arrays(control_rows)
    substep_trace = _rows_to_arrays(substep_rows)
    validate_control_trace(
        control_trace,
        policy_observation_dimension=adapter.input_dimension,
    )
    validate_substep_trace(
        substep_trace,
        physics_steps_per_action=timing.physics_steps_per_action,
        continuity_atol=1.0e-12,
    )
    save_npz(output / "control_trace.npz", control_trace)
    save_npz(output / "substep_trace.npz", substep_trace)
    if action_sequence is not None:
        save_npz(output / "input_action_sequence.npz", {"action_sequence": action_sequence})
    metadata = {
        "schema_version": "DreamWaQSim2SimTraceV1",
        "debug_only": True,
        "formal_ranking_eligible": False,
        "engine": "mujoco",
        "engine_version": mujoco.__version__,
        "python_version": platform.python_version(),
        "policy_kind": adapter.policy_kind,
        "policy_input_dimension": adapter.input_dimension,
        "actor_sha256": sha256_file(adapter.actor_path),
        "policy_manifest_sha256": sha256_file(adapter.manifest_path),
        "formal_model_sha256_before": formal_hash_before,
        "formal_model_sha256_after": sha256_file(FORMAL_MODEL),
        "debug_model": str(debug_model),
        "debug_model_sha256": sha256_file(debug_model),
        "timing_profile": {
            "name": timing.name,
            "physics_dt_s": timing.physics_dt_s,
            "physics_steps_per_action": timing.physics_steps_per_action,
            "torque_refresh_substeps": timing.torque_refresh_substeps,
            "control_dt_s": 0.020,
        },
        "mode": mode,
        "command": command_values.tolist(),
        "requested_control_ticks": control_ticks,
        "completed_control_ticks": len(control_rows),
        "stopped_reason_code": stopped_reason,
        "stopped_reason": TERMINATION_REASON_CODES[stopped_reason],
        "action_sequence_identity": action_sequence_identity,
        "all_hinge_order": all_hinge_names,
    }
    if metadata["formal_model_sha256_before"] != metadata["formal_model_sha256_after"]:
        raise RuntimeError("Formal MuJoCo model changed during debug collection")
    write_json(output / "metadata.json", metadata)
    write_json(
        output / "file_hashes.json",
        {
            name: sha256_file(path)
            for name, path in {
                "collector": Path(__file__),
                "debug_contract": Path(__file__).with_name("dreamwaq_debug_contract.py"),
                "metadata": output / "metadata.json",
                "reset_snapshot": output / "reset_snapshot.json",
                "control_trace": output / "control_trace.npz",
                "substep_trace": output / "substep_trace.npz",
            }.items()
        },
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a DreamWaQ-aware MuJoCo debug trace.")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-profile", choices=tuple(TIMING_PROFILES), default="formal_1ms")
    parser.add_argument("--control-ticks", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=("closed_loop", "zero_action", "action_replay"),
        default="closed_loop",
    )
    parser.add_argument("--action-sequence", type=Path)
    parser.add_argument("--action-key", default="action_sequence")
    args = parser.parse_args()
    sequence = None
    identity = None
    if args.action_sequence is not None:
        sequence, identity = load_action_sequence(
            args.action_sequence,
            args.action_key,
            requested_ticks=args.control_ticks,
        )
    metadata = collect_dreamwaq_mujoco_trace(
        actor_path=args.policy,
        manifest_path=args.manifest,
        output_directory=args.output,
        timing=TIMING_PROFILES[args.timing_profile],
        control_ticks=args.control_ticks,
        mode=args.mode,
        action_sequence=sequence,
        action_sequence_identity=identity,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
