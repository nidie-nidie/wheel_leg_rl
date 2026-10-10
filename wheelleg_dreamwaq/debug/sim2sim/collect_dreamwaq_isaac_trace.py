from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_MANIFEST = PROJECT_ROOT / "sim2sim/mujoco/model_manifest.json"

from isaaclab.app import AppLauncher

from wheelleg_dreamwaq.training.runtime import validate_runtime


def _to_numpy(value: Any, *, remove_env: bool = True) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if remove_env and array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array).copy()


def _rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Cannot serialize an empty Isaac trace")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows[1:]):
        raise ValueError("Isaac trace rows do not share one fixed schema")
    return {
        key: np.stack([_to_numpy(row[key]) for row in rows])
        if _to_numpy(rows[0][key]).ndim
        else np.asarray([_to_numpy(row[key]) for row in rows])
        for key in keys
    }


def _aggregate_torque(row: dict[str, Any], prefix: str, values: np.ndarray, dt: float) -> None:
    row[f"{prefix}_last_substep"] = values[-1]
    if np.isnan(values).all():
        row[f"{prefix}_mean"] = np.full(values.shape[1], np.nan)
        row[f"{prefix}_peak_abs"] = np.full(values.shape[1], np.nan)
        row[f"{prefix}_impulse"] = np.full(values.shape[1], np.nan)
        return
    row[f"{prefix}_mean"] = np.mean(values, axis=0)
    row[f"{prefix}_peak_abs"] = np.max(np.abs(values), axis=0)
    row[f"{prefix}_impulse"] = np.sum(values, axis=0) * dt


def _diagnostic_flags(state: dict[str, torch.Tensor]) -> np.ndarray:
    projected = _to_numpy(state["projected_gravity"])
    tilt = float(np.arccos(np.clip(-projected[2], -1.0, 1.0)))
    unexpected = int(_to_numpy(state["unexpected_contact"]))
    return np.asarray(
        (
            float(_to_numpy(state["base_height"])) < 0.10
            or float(_to_numpy(state["base_height"])) > 0.40,
            tilt > 0.80,
            np.linalg.norm(_to_numpy(state["base_linear_velocity_control"])) > 20.0,
            np.linalg.norm(_to_numpy(state["base_angular_velocity_control"])) > 35.0,
            np.max(np.abs(_to_numpy(state["all_hinge_velocity_named"]))) > 80.0,
            np.max(_to_numpy(state["loop_closure_error"])) > 5.0e-3,
            np.min(_to_numpy(state["virtual_leg_length"])) <= 0.05,
            -1 if unexpected < 0 else unexpected,
        ),
        dtype=np.int8,
    )


def _reason_code(env: Any, terminated: bool, truncated: bool) -> int:
    if truncated:
        return 7
    if not terminated:
        return 0
    names = (
        "invalid",
        "height_terminated",
        "tilt_terminated",
        "root_linear_terminated",
        "root_angular_terminated",
        "joint_velocity_terminated",
    )
    for code, name in enumerate(names, start=1):
        value = env._debug_termination_diagnostics.get(name)
        if value is not None and bool(value[0].item()):
            return code
    return 1


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a DreamWaQ-aware Isaac debug trace.")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-ticks", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=("closed_loop", "zero_action", "action_replay"),
        default="closed_loop",
    )
    parser.add_argument("--action-sequence", type=Path)
    parser.add_argument("--action-key", default="action_sequence")
    parser.add_argument("--contact-sensors", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.control_ticks <= 500:
        parser.error("--control-ticks must be in [1, 500]")
    if args.mode == "action_replay" and args.action_sequence is None:
        parser.error("--mode action_replay requires --action-sequence")
    if args.mode != "action_replay" and args.action_sequence is not None:
        parser.error("--action-sequence is only valid with --mode action_replay")

    validate_runtime(PROJECT_ROOT, device=args.device)
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    from debug.sim2sim.dreamwaq_debug_contract import DebugPolicyAdapter
    from debug.sim2sim.isaac_debug_env import WheelLegSim2SimDebugEnv, make_debug_env_cfg
    from debug.sim2sim.trace_schema import (
        build_file_hashes,
        load_action_sequence,
        save_npz,
        sha256_file,
        validate_control_trace,
        validate_substep_trace,
        write_json,
    )
    from wheelleg_dreamwaq.schemas.randomization import NOMINAL_EVALUATION_PROFILE_V1

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sequence = None
    sequence_identity = None
    if args.action_sequence is not None:
        sequence, sequence_identity = load_action_sequence(
            args.action_sequence,
            args.action_key,
            requested_ticks=args.control_ticks,
        )
    actor = DebugPolicyAdapter(
        actor_path=args.policy,
        manifest_path=args.manifest,
        model_manifest_path=MODEL_MANIFEST,
        load_mujoco_contract=False,
    )
    all_hinge_names = list(actor.model_manifest["joint_order"])
    cfg = make_debug_env_cfg(device=args.device, enable_contact_sensors=args.contact_sensors)
    cfg.randomization = NOMINAL_EVALUATION_PROFILE_V1
    if args.control_ticks == 500:
        cfg.episode_length_s = 10.02
    env = WheelLegSim2SimDebugEnv(cfg, enable_contact_sensors=args.contact_sensors)
    env.configure_debug_joint_order(all_hinge_names)
    observations, _ = env.reset(seed=0)
    current_observation = _to_numpy(observations["policy"]).astype(np.float32, copy=False)
    policy_input = actor.initialize_policy_input(current_observation)
    reset_state = env._debug_reset_snapshot["reset_forwarded_post_forward"]
    write_json(
        output / "reset_snapshot.json",
        {
            "schema_version": "DreamWaQSim2SimTraceV1",
            "engine": "isaac_sim",
            "current_actor_observation": current_observation.tolist(),
            "policy_input": policy_input.tolist(),
            "history_frames_identical": bool(
                actor.policy_kind != "dreamwaq"
                or np.array_equal(policy_input.reshape(5, 25), np.tile(current_observation, (5, 1)))
            ),
            "active_joint_position_canonical": _to_numpy(
                reset_state["active_joint_position_canonical"]
            ).tolist(),
            "base_com_position_engine_world": _to_numpy(
                reset_state["base_com_position_engine_world"]
            ).tolist(),
        },
    )

    control_rows: list[dict[str, Any]] = []
    substep_rows: list[dict[str, Any]] = []
    stopped_reason = 0
    try:
        for tick in range(args.control_ticks):
            previous_action = env._canonical_action.detach().clone()
            current_observation = _to_numpy(env.obs_buf["policy"]).astype(np.float32, copy=False)
            if not np.array_equal(current_observation, actor.current_observation(policy_input)):
                raise RuntimeError("Isaac current observation differs from the latest history frame")
            inference = actor.infer(policy_input)
            if args.mode == "closed_loop":
                selected = inference.raw_action.astype(np.float32)
            elif args.mode == "zero_action":
                selected = np.zeros(6, dtype=np.float32)
            else:
                selected = sequence[tick]
            selected_tensor = torch.from_numpy(selected).to(device=env.device).unsqueeze(0)
            next_observations, _, terminated, truncated, _ = env.step(selected_tensor)
            substeps = env._debug_substeps
            if len(substeps) != env.cfg.decimation:
                raise RuntimeError("Isaac debug environment returned the wrong substep count")
            for source in substeps:
                substep_rows.append({name: _to_numpy(value) for name, value in source.items()})
            terminal_state = env._debug_terminal_state
            if terminal_state is None or env._debug_next_obs is None or env._debug_action_clipped is None:
                raise RuntimeError("Isaac debug step did not expose terminal diagnostics")
            terminated_bool = bool(terminated[0].item())
            truncated_bool = bool(truncated[0].item())
            done = terminated_bool or truncated_bool
            reason_code = _reason_code(env, terminated_bool, truncated_bool)
            next_current = _to_numpy(env._debug_next_obs).astype(np.float32, copy=False)
            next_policy_input = actor.advance_policy_input(
                policy_input,
                next_current,
                done=done,
            )
            first_substep = substeps[0]
            row: dict[str, Any] = {
                "control_tick": np.int64(tick),
                "control_time_s": np.float64(substeps[-1]["physics_time_s"]),
                "command": _to_numpy(env._commands),
                "actor_obs_current_pre_step": current_observation,
                "actor_obs_policy_pre_step": policy_input,
                "previous_action_before_inference": _to_numpy(previous_action),
                "cenet_estimated_velocity": inference.estimated_velocity,
                "cenet_context_mu": inference.context_mu,
                "cenet_context_logvar": inference.context_logvar,
                "actor_output_raw": inference.raw_action,
                "action_clipped": _to_numpy(env._debug_action_clipped),
                "target_command_canonical": _to_numpy(first_substep["target_command_canonical"]),
                "target_command_engine_native": _to_numpy(
                    first_substep["target_command_engine_native"]
                ),
                "active_joint_position_canonical_post_step_pre_reset": _to_numpy(
                    terminal_state["active_joint_position_canonical"]
                ),
                "active_joint_velocity_canonical_post_step_pre_reset": _to_numpy(
                    terminal_state["active_joint_velocity_canonical"]
                ),
                "all_hinge_position_named_post_step_pre_reset": _to_numpy(
                    terminal_state["all_hinge_position_named"]
                ),
                "all_hinge_velocity_named_post_step_pre_reset": _to_numpy(
                    terminal_state["all_hinge_velocity_named"]
                ),
                "base_com_position_diag_post_step_pre_reset": _to_numpy(
                    terminal_state["base_com_position_diag"]
                ),
                "base_linear_velocity_control_post_step_pre_reset": _to_numpy(
                    terminal_state["base_linear_velocity_control"]
                ),
                "base_angular_velocity_control_post_step_pre_reset": _to_numpy(
                    terminal_state["base_angular_velocity_control"]
                ),
                "projected_gravity_post_step_pre_reset": _to_numpy(
                    terminal_state["projected_gravity"]
                ),
                "base_height_post_step_pre_reset": np.float64(
                    _to_numpy(terminal_state["base_height"])
                ),
                "virtual_leg_length_post_step_pre_reset": _to_numpy(
                    terminal_state["virtual_leg_length"]
                ),
                "virtual_leg_phi0_post_step_pre_reset": _to_numpy(
                    terminal_state["virtual_leg_phi0"]
                ),
                "loop_closure_error_post_step_pre_reset": _to_numpy(
                    terminal_state["loop_closure_error"]
                ),
                "effort_limit_event": np.max(
                    np.stack([_to_numpy(item["effort_limit_event"]) for item in substeps]), axis=0
                ),
                "joint_velocity_limit_exceeded": np.max(
                    np.stack(
                        [_to_numpy(item["joint_velocity_limit_exceeded"]) for item in substeps]
                    ),
                    axis=0,
                ),
                "mujoco_velocity_guard_active": np.max(
                    np.stack([_to_numpy(item["mujoco_velocity_guard_active"]) for item in substeps]),
                    axis=0,
                ),
                "native_terminated_int8": np.int8(terminated_bool),
                "native_truncated_int8": np.int8(truncated_bool),
                "native_termination_reason_code_int16": np.int16(reason_code),
                "common_diagnostic_flags_int8": _diagnostic_flags(terminal_state),
                "next_actor_obs_current_returned": next_current,
                "next_actor_obs_policy_returned": next_policy_input,
                "next_obs_is_reset_int8": np.int8(done),
            }
            for quantity in (
                "pd_torque_unclipped",
                "pd_torque_effort_clipped",
                "isaac_host_pd_torque_estimate",
                "mujoco_commanded_torque",
            ):
                for frame in ("canonical", "engine_native"):
                    values = np.stack(
                        [_to_numpy(item[f"{quantity}_{frame}"]) for item in substeps]
                    )
                    _aggregate_torque(row, f"{quantity}_{frame}", values, env.physics_dt)
            control_rows.append(row)
            policy_input = next_policy_input
            if done:
                stopped_reason = reason_code
                break

        control_trace = _rows_to_arrays(control_rows)
        substep_trace = _rows_to_arrays(substep_rows)
        validate_control_trace(
            control_trace,
            policy_observation_dimension=actor.input_dimension,
        )
        validate_substep_trace(
            substep_trace,
            physics_steps_per_action=env.cfg.decimation,
            continuity_atol=1.0e-6,
        )
        save_npz(output / "control_trace.npz", control_trace)
        save_npz(output / "substep_trace.npz", substep_trace)
        if sequence is not None:
            save_npz(output / "input_action_sequence.npz", {"action_sequence": sequence})
        metadata = {
            "schema_version": "DreamWaQSim2SimTraceV1",
            "debug_only": True,
            "formal_ranking_eligible": False,
            "engine": "isaac_sim",
            "engine_version": _distribution_version("isaacsim"),
            "isaac_lab_version": _distribution_version("isaaclab"),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": str(env.device),
            "policy_kind": actor.policy_kind,
            "policy_input_dimension": actor.input_dimension,
            "actor_sha256": sha256_file(actor.actor_path),
            "policy_manifest_sha256": sha256_file(actor.manifest_path),
            "timing_profile": {
                "name": "isaac_5ms",
                "physics_dt_s": env.physics_dt,
                "physics_steps_per_action": env.cfg.decimation,
                "torque_refresh_substeps": 1,
                "control_dt_s": env.step_dt,
            },
            "mode": args.mode,
            "command": [0.0, 0.0, 0.20],
            "requested_control_ticks": args.control_ticks,
            "completed_control_ticks": len(control_rows),
            "stopped_reason_code": stopped_reason,
            "action_sequence_identity": sequence_identity,
            "all_hinge_order": all_hinge_names,
            "randomization_profile": NOMINAL_EVALUATION_PROFILE_V1.name,
            "contact_sensors": args.contact_sensors,
        }
        write_json(output / "metadata.json", metadata)
        write_json(
            output / "file_hashes.json",
            build_file_hashes(
                {
                    "collector": Path(__file__),
                    "debug_environment": Path(__file__).with_name("isaac_debug_env.py"),
                    "debug_contract": Path(__file__).with_name("dreamwaq_debug_contract.py"),
                    "metadata": output / "metadata.json",
                    "reset_snapshot": output / "reset_snapshot.json",
                    "control_trace": output / "control_trace.npz",
                    "substep_trace": output / "substep_trace.npz",
                }
            ),
        )
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
