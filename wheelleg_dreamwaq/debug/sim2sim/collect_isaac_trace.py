from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import tomllib
import uuid
from pathlib import Path
from typing import Any

import h5py  # noqa: F401
import numpy as np
import tensordict  # noqa: F401
import torch

from .pitch_torque_pulse import BasePitchTorquePulse

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTOR_RELATIVE = Path("artifacts/phase1_v4/formal-4x1000-20261005/exports/run-01/actor.ts")
POLICY_MANIFEST_RELATIVE = Path(
    "artifacts/phase1_v4/formal-4x1000-20261005/exports/run-01/policy_manifest.json"
)
MODEL_MANIFEST_RELATIVE = Path("sim2sim/mujoco/model_manifest.json")

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
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        values = [_to_numpy(row[key]) for row in rows]
        arrays[key] = np.stack(values) if values[0].ndim else np.asarray(values)
    return arrays


def _physical_observation_from_direct(
    state: dict[str, torch.Tensor],
    command: torch.Tensor,
    previous_action: torch.Tensor,
    q_nominal: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            state["base_angular_velocity_control"],
            state["projected_gravity"],
            command,
            state["active_joint_position_canonical"][:, :4] - q_nominal.unsqueeze(0),
            state["active_joint_velocity_canonical"],
            previous_action,
        ),
        dim=-1,
    )


def _physical_observation_from_cached(env: Any) -> torch.Tensor:
    state = env._current_state()
    return torch.cat(
        (
            state.root_angular_velocity,
            state.projected_gravity,
            state.command,
            state.joint_position[:, :4] - env._q_nominal.unsqueeze(0),
            state.joint_velocity,
            state.last_applied_action,
        ),
        dim=-1,
    )


def _policy_observation_from_direct(env: Any, state: dict[str, torch.Tensor], previous_action: torch.Tensor) -> torch.Tensor:
    norm = env.cfg.normalization
    leg_error = state["active_joint_position_canonical"][:, :4] - env._q_nominal.unsqueeze(0)
    return torch.cat(
        (
            norm.normalize_angular_velocity(state["base_angular_velocity_control"]),
            norm.normalize_projected_gravity(state["projected_gravity"]),
            norm.normalize_command(env._commands),
            norm.normalize_leg_position_error(leg_error),
            norm.normalize_joint_velocity(state["active_joint_velocity_canonical"]),
            norm.normalize_previous_action(previous_action),
        ),
        dim=-1,
    )


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


def _native_reason_code(env: Any, terminated: bool, truncated: bool) -> int:
    if truncated:
        return 7
    if not terminated:
        return 0
    diagnostics = env._debug_termination_diagnostics
    names = (
        "invalid",
        "height_terminated",
        "tilt_terminated",
        "root_linear_terminated",
        "root_angular_terminated",
        "joint_velocity_terminated",
    )
    for code, name in enumerate(names, start=1):
        value = diagnostics.get(name)
        if value is not None and bool(value[0].item()):
            return code
    return 1


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


def _serialize_state(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {name: _to_numpy(value).tolist() for name, value in state.items()}


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a deterministic WheelLeg Isaac Sim debug trace.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-ticks", type=int, default=500)
    parser.add_argument(
        "--mode",
        choices=("closed_loop", "zero_action", "channel_pulse", "isaac_policy_replay"),
        default="closed_loop",
    )
    parser.add_argument("--action-sequence", type=Path)
    parser.add_argument("--action-key")
    parser.add_argument("--contact-sensors", action="store_true")
    parser.add_argument("--base-pitch-torque-nm", type=float, default=0.0)
    parser.add_argument("--pitch-torque-start-tick", type=int, default=0)
    parser.add_argument("--pitch-torque-end-tick", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    pitch_torque_pulse = BasePitchTorquePulse(
        magnitude_nm=args.base_pitch_torque_nm,
        start_tick=args.pitch_torque_start_tick,
        end_tick=args.pitch_torque_end_tick,
    )
    root = args.project_root.resolve()
    validate_runtime(root, device=args.device)
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    from debug.sim2sim.isaac_debug_env import WheelLegSim2SimDebugEnv, make_debug_env_cfg
    from debug.sim2sim.stand_scenario import StandScenarioV1, verify_frozen_files
    from debug.sim2sim.trace_schema import (
        SCHEMA_VERSION,
        build_file_hashes,
        closed_loop_action_identity,
        generated_action_sequence_identity,
        load_json,
        load_action_sequence,
        save_npz,
        sha256_file,
        validate_control_trace,
        validate_substep_trace,
        write_json,
    )

    scenario = StandScenarioV1()
    frozen_files = verify_frozen_files(root)
    if not 1 <= args.control_ticks <= scenario.control_ticks:
        raise ValueError(f"--control-ticks must be in [1, {scenario.control_ticks}]")
    pitch_torque_pulse.validate_control_ticks(args.control_ticks)
    sequence = None
    action_sequence_identity = closed_loop_action_identity()
    if args.action_sequence is not None:
        if args.mode == "closed_loop":
            parser.error("--action-sequence cannot be used with --mode closed_loop")
        if args.action_key is None:
            parser.error("--action-key is required with --action-sequence")
        sequence, action_sequence_identity = load_action_sequence(
            args.action_sequence,
            args.action_key,
            requested_ticks=args.control_ticks,
            require_replay_sidecar=args.mode == "isaac_policy_replay",
        )
    if args.mode == "zero_action" and sequence is None:
        sequence = np.zeros((args.control_ticks, 6), dtype=np.float32)
        action_sequence_identity = generated_action_sequence_identity(
            sequence,
            source="generated_zero",
            key="zero_action",
        )
    if args.mode != "closed_loop" and sequence is None:
        parser.error(f"--mode {args.mode} requires an action sequence")
    if sequence is not None:
        if sequence.ndim != 2 or sequence.shape[1] != 6:
            raise ValueError(f"Action sequence must have shape (N, 6), got {sequence.shape}")
        if len(sequence) < args.control_ticks:
            raise ValueError(
                f"Action sequence has {len(sequence)} rows for {args.control_ticks} requested ticks"
            )
        if not np.isfinite(sequence).all():
            raise ValueError("Action sequence contains NaN or Inf")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    embedded_action_path = None
    if sequence is not None:
        embedded_action_path = save_npz(
            output / "input_action_sequence.npz",
            {"action_sequence": np.asarray(sequence)},
        )
    actor_path = root / ACTOR_RELATIVE
    policy_manifest_path = root / POLICY_MANIFEST_RELATIVE
    model_manifest_path = root / MODEL_MANIFEST_RELATIVE
    policy_manifest = load_json(policy_manifest_path)
    model_manifest = load_json(model_manifest_path)
    dependency_manifest = tomllib.loads((root / "dependency-manifest.toml").read_text(encoding="utf-8"))
    if sha256_file(actor_path) != policy_manifest["actor_sha256"]:
        raise ValueError("TorchScript Actor hash differs from policy manifest")
    if sha256_file(model_manifest_path) != policy_manifest["mujoco_model"]["model_manifest_sha256"]:
        raise ValueError("Model manifest hash differs from policy manifest")
    all_hinge_names = list(model_manifest["joint_order"])

    env_cfg = make_debug_env_cfg(device=args.device, enable_contact_sensors=args.contact_sensors)
    if args.control_ticks == scenario.control_ticks:
        # The formal task times out at max_episode_length - 1. Add one control
        # period so the 500-tick diagnostic remains inside one episode.
        env_cfg.episode_length_s = (scenario.control_ticks + 1) * scenario.control_dt_s
    env = WheelLegSim2SimDebugEnv(
        env_cfg,
        enable_contact_sensors=args.contact_sensors,
    )
    env.configure_debug_joint_order(all_hinge_names)
    env.configure_debug_pitch_torque(pitch_torque_pulse)
    policy = torch.jit.load(str(actor_path), map_location=env.device).eval()
    observations, _ = env.reset(seed=scenario.random_seed)
    observation = observations["policy"]

    reset_debug = env._debug_reset_snapshot
    if reset_debug is None:
        raise RuntimeError("Debug reset did not expose the three reset phases")
    zero_action = torch.zeros((1, 6), dtype=torch.float32, device=env.device)
    pre_state = reset_debug["reset_written_pre_forward"]
    post_state = reset_debug["reset_forwarded_post_forward"]
    reset_snapshot = {
        "schema_version": SCHEMA_VERSION,
        "reset_written_pre_forward": {
            **_serialize_state(pre_state),
            "actor_obs_physical_rebuilt": _to_numpy(
                _physical_observation_from_direct(pre_state, env._commands, zero_action, env._q_nominal)
            ).tolist(),
            "actor_obs_policy_rebuilt": _to_numpy(
                _policy_observation_from_direct(env, pre_state, zero_action)
            ).tolist(),
            "actor_obs_policy_cached": _to_numpy(
                reset_debug["reset_written_cached_actor_obs_policy"]
            ).tolist(),
        },
        "reset_forwarded_post_forward": {
            **_serialize_state(post_state),
            "actor_obs_physical_rebuilt": _to_numpy(
                _physical_observation_from_direct(post_state, env._commands, zero_action, env._q_nominal)
            ).tolist(),
            "actor_obs_policy_rebuilt": _to_numpy(
                _policy_observation_from_direct(env, post_state, zero_action)
            ).tolist(),
        },
        "reset_returned_to_policy": {
            "actor_obs_policy_returned": _to_numpy(
                reset_debug["reset_returned_actor_obs_policy"]
            ).tolist(),
            "previous_action": [0.0] * 6,
        },
    }
    write_json(output / "reset_snapshot.json", reset_snapshot)

    control_rows: list[dict[str, Any]] = []
    substep_rows: list[dict[str, Any]] = []
    stopped_reason = 0
    try:
        for tick in range(args.control_ticks):
            previous_action = env._canonical_action.detach().clone()
            physical_observation = _physical_observation_from_cached(env)
            with torch.inference_mode():
                raw_action = policy(observation)
            if raw_action.shape != (1, 6) or not torch.isfinite(raw_action).all():
                raise FloatingPointError("TorchScript Actor returned an invalid action")
            if args.mode == "closed_loop":
                selected_action = raw_action
            else:
                if tick >= len(sequence):
                    raise ValueError(f"Action sequence ended at tick {tick}")
                selected_action = torch.as_tensor(
                    sequence[tick], dtype=torch.float32, device=env.device
                ).unsqueeze(0)

            next_observations, _, terminated, truncated, _ = env.step(selected_action)
            substeps = env._debug_substeps
            if len(substeps) != env.cfg.decimation:
                raise RuntimeError(f"Expected {env.cfg.decimation} Isaac substeps, got {len(substeps)}")
            substep_rows.extend(substeps)
            terminal_state = env._debug_terminal_state
            if terminal_state is None or env._debug_next_obs is None:
                raise RuntimeError("Debug step did not preserve terminal state and returned observation")
            diagnostic_flags = _diagnostic_flags(terminal_state)
            terminated_bool = bool(terminated[0].item())
            truncated_bool = bool(truncated[0].item())
            reason_code = _native_reason_code(env, terminated_bool, truncated_bool)
            if env._debug_action_clipped is None:
                raise RuntimeError("Debug step did not preserve the applied clipped action")

            target_canonical = _to_numpy(substeps[0]["target_command_canonical"])
            target_native = _to_numpy(substeps[0]["target_command_engine_native"])
            control_row: dict[str, Any] = {
                "control_tick": np.int64(tick),
                "control_time_s": np.float64(substeps[-1]["physics_time_s"]),
                "command": _to_numpy(env._commands),
                "actor_obs_physical_pre_step": _to_numpy(physical_observation),
                "actor_obs_policy_pre_step": _to_numpy(observation),
                "previous_action_before_inference": _to_numpy(previous_action),
                "actor_output_raw": _to_numpy(raw_action),
                "action_clipped": _to_numpy(env._debug_action_clipped),
                "target_command_canonical": target_canonical,
                "target_command_engine_native": target_native,
                "external_base_torque_control": _to_numpy(
                    substeps[-1]["external_base_torque_control"]
                ),
                "external_base_torque_engine_world_mean": np.mean(
                    np.stack(
                        [
                            _to_numpy(row["external_base_torque_engine_world"])
                            for row in substeps
                        ]
                    ),
                    axis=0,
                ),
                "active_joint_position_canonical_post_step_pre_reset": _to_numpy(
                    terminal_state["active_joint_position_canonical"]
                ),
                "active_joint_velocity_canonical_post_step_pre_reset": _to_numpy(
                    terminal_state["active_joint_velocity_canonical"]
                ),
                "active_joint_position_engine_native_post_step_pre_reset": _to_numpy(
                    terminal_state["active_joint_position_engine_native"]
                ),
                "active_joint_velocity_engine_native_post_step_pre_reset": _to_numpy(
                    terminal_state["active_joint_velocity_engine_native"]
                ),
                "all_hinge_position_named_post_step_pre_reset": _to_numpy(
                    terminal_state["all_hinge_position_named"]
                ),
                "all_hinge_velocity_named_post_step_pre_reset": _to_numpy(
                    terminal_state["all_hinge_velocity_named"]
                ),
                "effort_limit_event": np.max(
                    np.stack([_to_numpy(row["effort_limit_event"]) for row in substeps]), axis=0
                ).astype(np.int8),
                "joint_velocity_limit_exceeded": np.max(
                    np.stack(
                        [_to_numpy(row["joint_velocity_limit_exceeded"]) for row in substeps]
                    ),
                    axis=0,
                ).astype(np.int8),
                "mujoco_velocity_guard_active": np.max(
                    np.stack(
                        [_to_numpy(row["mujoco_velocity_guard_active"]) for row in substeps]
                    ),
                    axis=0,
                ).astype(np.int8),
                "base_com_position_engine_world_post_step_pre_reset": _to_numpy(
                    terminal_state["base_com_position_engine_world"]
                ),
                "base_com_position_diag_post_step_pre_reset": _to_numpy(
                    terminal_state["base_com_position_diag"]
                ),
                "base_orientation_control_wxyz_post_step_pre_reset": _to_numpy(
                    terminal_state["base_orientation_control_wxyz"]
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
                "native_terminated_int8": np.int8(terminated_bool),
                "native_truncated_int8": np.int8(truncated_bool),
                "native_termination_reason_code_int16": np.int16(reason_code),
                "common_diagnostic_flags_int8": diagnostic_flags,
                "next_actor_obs_policy_returned": _to_numpy(env._debug_next_obs),
                "next_obs_is_reset_int8": np.int8(_to_numpy(env._debug_next_obs_is_reset)),
            }

            for quantity in (
                "pd_torque_unclipped",
                "pd_torque_effort_clipped",
                "isaac_host_pd_torque_estimate",
                "mujoco_commanded_torque",
            ):
                for frame in ("canonical", "engine_native"):
                    values = np.stack(
                        [_to_numpy(row[f"{quantity}_{frame}"]) for row in substeps]
                    )
                    _aggregate_torque(
                        control_row,
                        f"{quantity}_{frame}",
                        values,
                        env.physics_dt,
                    )

            contact_active = np.stack(
                [_to_numpy(row["wheel_contact_active"]) for row in substeps]
            )
            contact_force = np.stack(
                [_to_numpy(row["wheel_normal_force_n"]) for row in substeps]
            )
            contact_impulse = np.stack(
                [_to_numpy(row["wheel_normal_impulse_ns"]) for row in substeps]
            )
            normal_force_world = np.stack(
                [_to_numpy(row["wheel_normal_force_world"]) for row in substeps]
            )
            friction_force_world = np.stack(
                [_to_numpy(row["wheel_friction_force_world"]) for row in substeps]
            )
            control_row["wheel_contact_active"] = np.max(contact_active, axis=0).astype(np.int8)
            if np.isnan(contact_force).all():
                control_row["wheel_normal_force_n_mean"] = np.full(2, np.nan)
                control_row["wheel_normal_force_n_peak"] = np.full(2, np.nan)
                control_row["wheel_normal_impulse_ns"] = np.full(2, np.nan)
                control_row["wheel_normal_force_world_mean"] = np.full((2, 3), np.nan)
                control_row["wheel_friction_force_world_mean"] = np.full((2, 3), np.nan)
            else:
                control_row["wheel_normal_force_n_mean"] = np.mean(contact_force, axis=0)
                control_row["wheel_normal_force_n_peak"] = np.max(contact_force, axis=0)
                control_row["wheel_normal_impulse_ns"] = np.sum(contact_impulse, axis=0)
                control_row["wheel_normal_force_world_mean"] = np.mean(
                    normal_force_world, axis=0
                )
                control_row["wheel_friction_force_world_mean"] = (
                    np.full((2, 3), np.nan)
                    if np.isnan(friction_force_world).all()
                    else np.nanmean(friction_force_world, axis=0)
                )

            control_rows.append(control_row)
            observation = next_observations["policy"]
            if terminated_bool or truncated_bool:
                stopped_reason = reason_code
                break

        control_trace = _rows_to_arrays(control_rows)
        substep_trace = _rows_to_arrays(substep_rows)
        validate_control_trace(control_trace)
        validate_substep_trace(
            substep_trace,
            physics_steps_per_action=env.cfg.decimation,
            continuity_atol=1.0e-6,
        )
        save_npz(output / "control_trace.npz", control_trace)
        save_npz(output / "substep_trace.npz", substep_trace)

        articulation_props = env.cfg.robot_cfg.spawn.articulation_props
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "collection_id": uuid.uuid4().hex,
            "engine": "isaac_sim",
            "engine_version": _distribution_version("isaacsim"),
            "isaac_lab_version": dependency_manifest["isaac_lab"]["tag"].removeprefix("v"),
            "isaac_lab_extension_version": _distribution_version("isaaclab"),
            "isaac_lab_commit": dependency_manifest["isaac_lab"]["commit"],
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "inference_device": str(env.device),
            "inference_dtype": "float32",
            "simulation_device": str(env.cfg.sim.device),
            "physics_pipeline": "gpu" if str(env.cfg.sim.device).startswith("cuda") else "cpu",
            "scenario_hash": scenario.payload_hash,
            "frozen_files": frozen_files,
            "scenario_variant": args.mode,
            "base_pitch_torque_pulse": pitch_torque_pulse.payload,
            "action_sequence_identity": action_sequence_identity,
            "command": list(scenario.command),
            "random_seed": scenario.random_seed,
            "requested_control_ticks": args.control_ticks,
            "completed_control_ticks": len(control_rows),
            "configured_episode_length_s": env.cfg.episode_length_s,
            "control_dt_s": env.step_dt,
            "physics_dt_s": env.physics_dt,
            "physics_steps_per_action": env.cfg.decimation,
            "substep_continuity_atol": 1.0e-6,
            "canonical_joint_order": list(policy_manifest["action"]["canonical_joint_order"]),
            "canonical_from_engine_native": [1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            "all_hinge_order": all_hinge_names,
            "r_diag_from_engine_world": policy_manifest["frames"]["r_control_from_usd"],
            "actor_sha256": sha256_file(actor_path),
            "policy_manifest_sha256": sha256_file(policy_manifest_path),
            "model_manifest_sha256": sha256_file(model_manifest_path),
            "model_xml_sha256": model_manifest["model_xml"]["sha256"],
            "solver_position_iterations": articulation_props.solver_position_iteration_count,
            "solver_velocity_iterations": articulation_props.solver_velocity_iteration_count,
            "clone_in_fabric": env.cfg.scene.clone_in_fabric,
            "use_fabric": env.cfg.sim.use_fabric,
            "render_mode": "headless" if args.headless else "interactive",
            "observer_mode": "debug_step_subclass",
            "contact_observation_mode": (
                "instrumented_unverified" if args.contact_sensors else "disabled"
            ),
            "contact_processing_state": (
                "contact_report_api_enabled_for_debug_sensors"
                if args.contact_sensors
                else "baseline_without_contact_report_observer"
            ),
            "contact_active_force_threshold_n": 1.0,
            "contact_force_semantics": (
                "ContactSensor.net_forces_w is the unfiltered net normal-force vector; "
                "its norm is the normal-force magnitude and it excludes friction"
            ),
            "termination_reason_codes": {
                "0": "none",
                "1": "invalid",
                "2": "height",
                "3": "tilt",
                "4": "root_linear_velocity",
                "5": "root_angular_velocity",
                "6": "joint_velocity",
                "7": "timeout",
            },
            "unavailable_fields": {
                "mujoco_commanded_torque": "not available in Isaac Sim",
                "physx_internal_drive_torque": "PhysX public API unavailable",
                "wheel_friction_force_world": (
                    "GPU PhysX does not support the required global-ground filter for "
                    "ContactSensor friction tracking"
                ),
                "unexpected_contact": (
                    "not collected in the baseline without contact sensors"
                    if not args.contact_sensors
                    else "unfiltered wheel sensors cannot classify unexpected contact; "
                    "the frozen task relies on its wheel-only collision contract"
                ),
            },
            "stopped_reason_code": stopped_reason,
        }
        metadata["stopped_reason"] = metadata["termination_reason_codes"][str(stopped_reason)]
        formal_task_root = root / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat"
        schema_root = root / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas"
        formal_sources = {
            "formal_environment": formal_task_root / "env.py",
            "formal_environment_config": formal_task_root / "env_cfg.py",
            "formal_observation_adapter": formal_task_root / "observations.py",
            "formal_action_adapter": formal_task_root / "control.py",
            "formal_action_schema": schema_root / "action.py",
            "formal_frame_schema": schema_root / "frames.py",
            "formal_normalization_schema": schema_root / "normalization.py",
        }
        metadata["formal_source_sha256"] = build_file_hashes(formal_sources)
        write_json(output / "metadata.json", metadata)
        source_paths = {
            "collector": Path(__file__),
            "debug_environment": Path(__file__).with_name("isaac_debug_env.py"),
            "trace_schema": Path(__file__).with_name("trace_schema.py"),
            "comparison": Path(__file__).with_name("compare_traces.py"),
            "stand_scenario": Path(__file__).with_name("stand_scenario.py"),
            "pitch_torque_pulse": Path(__file__).with_name("pitch_torque_pulse.py"),
            **formal_sources,
            "metadata": output / "metadata.json",
            "reset_snapshot": output / "reset_snapshot.json",
            "control_trace": output / "control_trace.npz",
            "substep_trace": output / "substep_trace.npz",
        }
        if embedded_action_path is not None:
            source_paths["input_action_sequence"] = embedded_action_path
        write_json(output / "file_hashes.json", build_file_hashes(source_paths))
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
