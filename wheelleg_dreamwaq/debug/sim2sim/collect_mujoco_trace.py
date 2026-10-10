from __future__ import annotations

import argparse
import json
import platform
import uuid
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from wheelleg_mujoco.contract import load_policy_contract
from wheelleg_mujoco.observation import build_actor_observation, collect_kinematic_state
from wheelleg_mujoco.runner import WheelLegMujocoRuntime

from .pitch_torque_pulse import BasePitchTorquePulse
from .stand_scenario import StandScenarioV1, verify_frozen_files
from .trace_schema import (
    SCHEMA_VERSION,
    build_file_hashes,
    closed_loop_action_identity,
    engine_native_to_canonical,
    generated_action_sequence_identity,
    load_action_sequence,
    save_npz,
    sha256_file,
    validate_control_trace,
    validate_substep_trace,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTOR_RELATIVE = Path("artifacts/phase1_v4/formal-4x1000-20261005/exports/run-01/actor.ts")
POLICY_MANIFEST_RELATIVE = Path(
    "artifacts/phase1_v4/formal-4x1000-20261005/exports/run-01/policy_manifest.json"
)
MODEL_RELATIVE = Path("sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml")
MODEL_MANIFEST_RELATIVE = Path("sim2sim/mujoco/model_manifest.json")
ALLOWED_CONTACTS = {
    frozenset(("floor", "left_wheel_proxy")),
    frozenset(("floor", "right_wheel_proxy")),
}
TERMINATION_REASON_CODES = {
    0: "none",
    1: "height",
    2: "tilt",
    3: "root_linear_velocity",
    4: "root_angular_velocity",
    5: "joint_velocity",
    6: "loop_closure",
    7: "virtual_leg_length",
    8: "unexpected_contact",
    9: "non_finite",
}


def _required_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model is missing {object_type.name} {name!r}")
    return int(object_id)


def _object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name is not None else f"unnamed_{object_id}"


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=np.float64).reshape(9))
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def _physical_observation(
    state: dict[str, Any], command: np.ndarray, previous_action: np.ndarray, q_nominal: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        (
            state["base_angular_velocity_control"],
            state["projected_gravity"],
            command,
            state["active_joint_position_canonical"][:4] - q_nominal,
            state["active_joint_velocity_canonical"],
            previous_action,
        )
    ).astype(np.float64)


def _resolve_all_hinges(model: mujoco.MjModel, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if len(names) != 26 or len(set(names)) != 26:
        raise ValueError(f"Expected 26 unique hinge names, got {len(names)}")
    joint_ids = np.asarray([_required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
    if np.any(model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Every shared joint-order entry must resolve to a MuJoCo hinge")
    return np.asarray(model.jnt_qposadr[joint_ids]), np.asarray(model.jnt_dofadr[joint_ids])


def _virtual_leg_state(
    model: mujoco.MjModel, data: mujoco.MjData, base_body_id: int
) -> tuple[np.ndarray, np.ndarray]:
    rotation_world_from_body = np.asarray(data.xmat[base_body_id]).reshape(3, 3)
    lengths: list[float] = []
    angles: list[float] = []
    for hip_name, wheel_name in (("jIJ", "jwheel_left"), ("jAB", "jwheel_right")):
        hip_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, hip_name)
        wheel_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, wheel_name)
        vector_body = rotation_world_from_body.T @ (data.xanchor[wheel_id] - data.xanchor[hip_id])
        s_w, d_w = float(vector_body[1]), float(-vector_body[2])
        lengths.append(float(np.hypot(s_w, d_w)))
        angles.append(float(np.arctan2(d_w, s_w)))
    return np.asarray(lengths), np.asarray(angles)


def _loop_closure_error(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    side_pairs = (
        (
            ("site_kn_op_a1", "site_kn_op_b1"),
            ("site_kn_op_a2", "site_kn_op_b2"),
            ("site_mk_io_a1", "site_mk_io_b1"),
            ("site_mk_io_a2", "site_mk_io_b2"),
        ),
        (
            ("site_cf_gh_a1", "site_cf_gh_b1"),
            ("site_cf_gh_a2", "site_cf_gh_b2"),
            ("site_ec_ag_a1", "site_ec_ag_b1"),
            ("site_ec_ag_a2", "site_ec_ag_b2"),
        ),
    )
    result = []
    for pairs in side_pairs:
        errors = []
        for first_name, second_name in pairs:
            first = _required_id(model, mujoco.mjtObj.mjOBJ_SITE, first_name)
            second = _required_id(model, mujoco.mjtObj.mjOBJ_SITE, second_name)
            errors.append(float(np.linalg.norm(data.site_xpos[first] - data.site_xpos[second])))
        result.append(max(errors))
    return np.asarray(result)


def _contact_state(model: mujoco.MjModel, data: mujoco.MjData, physics_dt: float) -> dict[str, Any]:
    normal_force = np.zeros(2)
    normal_force_world = np.zeros((2, 3))
    friction_force_world = np.zeros((2, 3))
    unexpected = 0
    wheel_index = {"left_wheel_proxy": 0, "right_wheel_proxy": 1}
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        first = _object_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        second = _object_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        if frozenset((first, second)) not in ALLOWED_CONTACTS:
            unexpected = 1
            continue
        wheel_name = first if first in wheel_index else second
        mujoco.mj_contactForce(model, data, index, force)
        wheel = wheel_index[wheel_name]
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        sign = -1.0 if wheel_name == first else 1.0
        normal_world = sign * (frame.T @ np.asarray((force[0], 0.0, 0.0)))
        friction_world = sign * (frame.T @ np.asarray((0.0, force[1], force[2])))
        normal_force[wheel] += max(0.0, float(force[0]))
        normal_force_world[wheel] += normal_world
        friction_force_world[wheel] += friction_world
    return {
        "wheel_contact_active": (normal_force > 1.0).astype(np.int8),
        "wheel_normal_force_n": normal_force,
        "wheel_normal_impulse_ns": normal_force * physics_dt,
        "wheel_normal_force_world": normal_force_world,
        "wheel_friction_force_world": friction_force_world,
        "unexpected_contact": np.int8(unexpected),
    }


def _capture_state(
    runtime: WheelLegMujocoRuntime,
    all_hinge_qpos: np.ndarray,
    all_hinge_dof: np.ndarray,
    initial_com_world: np.ndarray,
) -> dict[str, Any]:
    model, data, contract = runtime.model, runtime.data, runtime.contract
    kinematic = collect_kinematic_state(model, data, runtime.model_map, contract)
    rotation_world_from_body = np.asarray(data.xmat[runtime.model_map.base_body_id]).reshape(3, 3)
    rotation_control_world_from_body = (
        contract.r_control_from_mujoco @ rotation_world_from_body @ contract.r_control_from_mujoco.T
    )
    com_world = np.asarray(data.xipos[runtime.model_map.base_body_id]).copy()
    position_diag = contract.r_control_from_mujoco @ (com_world - initial_com_world)
    position_diag[2] = kinematic.base_height_m
    lengths, phi0 = _virtual_leg_state(model, data, runtime.model_map.base_body_id)
    return {
        "active_joint_position_engine_native": runtime.model_map.joint_position_mujoco(data).copy(),
        "active_joint_velocity_engine_native": runtime.model_map.joint_velocity_mujoco(data).copy(),
        "active_joint_position_canonical": runtime.model_map.joint_position_control(data).copy(),
        "active_joint_velocity_canonical": runtime.model_map.joint_velocity_control(data).copy(),
        "all_hinge_position_named": np.asarray(data.qpos[all_hinge_qpos]).copy(),
        "all_hinge_velocity_named": np.asarray(data.qvel[all_hinge_dof]).copy(),
        "base_com_position_engine_world": com_world,
        "base_com_position_diag": position_diag,
        "base_orientation_control_wxyz": _matrix_to_quaternion_wxyz(rotation_control_world_from_body),
        "base_linear_velocity_control": kinematic.com_linear_velocity_control.copy(),
        "base_angular_velocity_control": kinematic.angular_velocity_control.copy(),
        "projected_gravity": kinematic.projected_gravity_control.copy(),
        "base_height": np.float64(kinematic.base_height_m),
        "virtual_leg_length": lengths,
        "virtual_leg_phi0": phi0,
        "loop_closure_error": _loop_closure_error(model, data),
        **_contact_state(model, data, contract.physics_dt_s),
        "kinematic": kinematic,
    }


def _diagnostic_flags(state: dict[str, Any]) -> np.ndarray:
    tilt = float(np.arccos(np.clip(-state["projected_gravity"][2], -1.0, 1.0)))
    return np.asarray(
        (
            state["base_height"] < 0.10 or state["base_height"] > 0.40,
            tilt > 0.80,
            np.linalg.norm(state["base_linear_velocity_control"]) > 20.0,
            np.linalg.norm(state["base_angular_velocity_control"]) > 35.0,
            np.max(np.abs(state["all_hinge_velocity_named"])) > 80.0,
            np.max(state["loop_closure_error"]) > 5.0e-3,
            np.min(state["virtual_leg_length"]) <= 0.05,
            state["unexpected_contact"] == 1,
        ),
        dtype=np.int8,
    )


def _native_termination_reason(state: dict[str, Any]) -> int:
    numeric_values = [
        np.asarray(value)
        for name, value in state.items()
        if name != "kinematic" and np.asarray(value).dtype.kind in "biufc"
    ]
    if not all(np.isfinite(value).all() for value in numeric_values):
        return 9
    height = float(state["base_height"])
    tilt = float(np.arccos(np.clip(-state["projected_gravity"][2], -1.0, 1.0)))
    if not 0.10 <= height <= 0.40:
        return 1
    if tilt > 0.80:
        return 2
    if np.linalg.norm(state["base_linear_velocity_control"]) > 20.0:
        return 3
    if np.linalg.norm(state["base_angular_velocity_control"]) > 35.0:
        return 4
    if np.max(np.abs(state["all_hinge_velocity_named"])) > 80.0:
        return 5
    if np.max(state["loop_closure_error"]) > 5.0e-3:
        return 6
    if np.min(state["virtual_leg_length"]) <= 0.05:
        return 7
    if state["unexpected_contact"] == 1:
        return 8
    return 0


def _serializable_state(
    state: dict[str, Any], *, q_nominal: np.ndarray, command: np.ndarray, previous_action: np.ndarray
) -> dict[str, Any]:
    payload = {
        key: (value.tolist() if isinstance(value, np.ndarray) else float(value))
        for key, value in state.items()
        if key != "kinematic"
    }
    payload["actor_obs_physical_rebuilt"] = _physical_observation(
        state, command, previous_action, q_nominal
    ).tolist()
    return payload


def _rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Cannot serialize an empty trace")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows[1:]):
        raise ValueError("Trace rows do not share one fixed schema")
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        values = [np.asarray(row[key]) for row in rows]
        arrays[key] = np.stack(values) if values[0].ndim else np.asarray(values)
    return arrays


def _torque_reference(
    runtime: WheelLegMujocoRuntime, pre: dict[str, Any], targets: Any
) -> tuple[np.ndarray, np.ndarray]:
    position = pre["active_joint_position_engine_native"]
    velocity = pre["active_joint_velocity_engine_native"]
    torque = np.empty(6)
    torque[:4] = runtime.contract.leg_kp * (targets.leg_position_target_mujoco - position[:4])
    torque[:4] -= runtime.contract.leg_kd * velocity[:4]
    torque[4:6] = runtime.contract.wheel_kd * (targets.wheel_velocity_target_mujoco - velocity[4:6])
    return torque, np.clip(torque, -runtime.contract.effort_limits, runtime.contract.effort_limits)


def _aggregate_torque(
    control_row: dict[str, Any], prefix: str, native_values: list[np.ndarray], dt: float
) -> None:
    native = np.stack(native_values)
    canonical = engine_native_to_canonical(native)
    for frame, values in (("engine_native", native), ("canonical", canonical)):
        control_row[f"{prefix}_{frame}_last_substep"] = values[-1]
        control_row[f"{prefix}_{frame}_mean"] = values.mean(axis=0)
        control_row[f"{prefix}_{frame}_peak_abs"] = np.max(np.abs(values), axis=0)
        control_row[f"{prefix}_{frame}_impulse"] = values.sum(axis=0) * dt


def _select_action(
    mode: str, tick: int, policy_action: np.ndarray, action_sequence: np.ndarray | None
) -> np.ndarray:
    if mode == "closed_loop":
        return policy_action
    if action_sequence is None or tick >= len(action_sequence):
        raise ValueError(f"Mode {mode!r} has no action for tick {tick}")
    return np.asarray(action_sequence[tick], dtype=np.float64)


def _apply_base_pitch_torque(
    runtime: WheelLegMujocoRuntime,
    pulse: BasePitchTorquePulse,
    tick: int,
) -> tuple[np.ndarray, np.ndarray]:
    control_torque = pulse.control_vector(tick)
    native_body_torque = runtime.contract.r_control_from_mujoco.T @ control_torque
    rotation_world_from_body = np.asarray(
        runtime.data.xmat[runtime.model_map.base_body_id], dtype=np.float64
    ).reshape(3, 3)
    world_torque = rotation_world_from_body @ native_body_torque
    applied = runtime.data.xfrc_applied[runtime.model_map.base_body_id]
    applied[:] = 0.0
    applied[3:6] = world_torque
    return control_torque, world_torque


def _substep_row(
    *,
    tick: int,
    substep: int,
    time_s: float,
    pre: dict[str, Any],
    post: dict[str, Any],
    target_canonical: np.ndarray,
    target_native: np.ndarray,
    unclipped_native: np.ndarray,
    clipped_native: np.ndarray,
    commanded_native: np.ndarray,
    effort_event: np.ndarray,
    velocity_event: np.ndarray,
    guard_event: np.ndarray,
    external_torque_control: np.ndarray,
    external_torque_world: np.ndarray,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "control_tick": np.int64(tick),
        "substep_index": np.int16(substep),
        "physics_time_s": np.float64(time_s),
        "target_command_canonical": target_canonical,
        "target_command_engine_native": target_native,
        "pd_torque_unclipped_canonical": engine_native_to_canonical(unclipped_native),
        "pd_torque_unclipped_engine_native": unclipped_native,
        "pd_torque_effort_clipped_canonical": engine_native_to_canonical(clipped_native),
        "pd_torque_effort_clipped_engine_native": clipped_native,
        "isaac_host_pd_torque_estimate_canonical": np.full(6, np.nan),
        "isaac_host_pd_torque_estimate_engine_native": np.full(6, np.nan),
        "mujoco_commanded_torque_canonical": engine_native_to_canonical(commanded_native),
        "mujoco_commanded_torque_engine_native": commanded_native,
        "effort_limit_event": effort_event,
        "joint_velocity_limit_exceeded": velocity_event,
        "mujoco_velocity_guard_active": guard_event,
        "external_base_torque_control": external_torque_control,
        "external_base_torque_engine_world": external_torque_world,
    }
    for suffix, state in (("pre_step", pre), ("post_step", post)):
        for name in (
            "active_joint_position_canonical",
            "active_joint_velocity_canonical",
            "active_joint_position_engine_native",
            "active_joint_velocity_engine_native",
            "all_hinge_position_named",
            "all_hinge_velocity_named",
            "base_com_position_engine_world",
            "base_com_position_diag",
            "base_orientation_control_wxyz",
            "base_linear_velocity_control",
            "base_angular_velocity_control",
        ):
            row[f"{name}_{suffix}"] = state[name]
    for name in (
        "projected_gravity",
        "virtual_leg_length",
        "virtual_leg_phi0",
        "loop_closure_error",
        "wheel_contact_active",
        "wheel_normal_force_n",
        "wheel_normal_impulse_ns",
        "wheel_normal_force_world",
        "wheel_friction_force_world",
    ):
        row[f"{name}_post_step" if name.startswith(("projected", "virtual", "loop")) else name] = post[name]
    return row


def collect_mujoco_trace(
    *,
    project_root: str | Path,
    output_directory: str | Path,
    control_ticks: int,
    mode: str,
    action_sequence: np.ndarray | None = None,
    action_sequence_identity: dict[str, Any] | None = None,
    pitch_torque_pulse: BasePitchTorquePulse | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scenario = StandScenarioV1()
    frozen_files = verify_frozen_files(root)
    if not 1 <= control_ticks <= scenario.control_ticks:
        raise ValueError(f"control_ticks must be in [1, {scenario.control_ticks}]")
    pitch_torque_pulse = pitch_torque_pulse or BasePitchTorquePulse.disabled()
    pitch_torque_pulse.validate_control_ticks(control_ticks)
    if mode not in {"closed_loop", "zero_action", "channel_pulse", "isaac_policy_replay"}:
        raise ValueError(f"Unsupported collection mode: {mode}")
    if mode == "closed_loop" and action_sequence is not None:
        raise ValueError("closed_loop mode cannot consume an external action sequence")
    if mode == "zero_action" and action_sequence is None:
        action_sequence = np.zeros((control_ticks, 6), dtype=np.float32)
        action_sequence_identity = generated_action_sequence_identity(
            action_sequence,
            source="generated_zero",
            key="zero_action",
        )
    if action_sequence is not None:
        action_sequence = np.asarray(action_sequence)
        if action_sequence.ndim != 2 or action_sequence.shape[1] != 6:
            raise ValueError(f"Action sequence must have shape (N, 6), got {action_sequence.shape}")
        if action_sequence.dtype != np.float32:
            raise TypeError(f"Action sequence must use float32, got {action_sequence.dtype}")
        if len(action_sequence) < control_ticks:
            raise ValueError(
                f"Action sequence has {len(action_sequence)} rows for {control_ticks} requested ticks"
            )
        if not np.isfinite(action_sequence).all():
            raise ValueError("Action sequence contains NaN or Inf")
        if action_sequence_identity is None:
            action_sequence_identity = generated_action_sequence_identity(
                action_sequence,
                source="in_memory",
                key=mode,
            )
    elif action_sequence_identity is None:
        action_sequence_identity = closed_loop_action_identity()
    embedded_action_path = None
    if action_sequence is not None:
        embedded_action_path = save_npz(
            output / "input_action_sequence.npz",
            {"action_sequence": np.asarray(action_sequence)},
        )

    actor_path = root / ACTOR_RELATIVE
    policy_manifest_path = root / POLICY_MANIFEST_RELATIVE
    model_path = root / MODEL_RELATIVE
    model_manifest_path = root / MODEL_MANIFEST_RELATIVE
    contract, _, model_manifest = load_policy_contract(
        policy_manifest_path, actor_path, model_manifest_path
    )
    if sha256_file(model_path) != model_manifest["model_xml"]["sha256"]:
        raise ValueError("MuJoCo XML hash differs from the frozen model manifest")
    policy = torch.jit.load(str(actor_path), map_location="cpu").eval()
    runtime = WheelLegMujocoRuntime(model_path, contract, model_manifest)
    all_hinge_names = list(model_manifest["joint_order"])
    all_hinge_qpos, all_hinge_dof = _resolve_all_hinges(runtime.model, all_hinge_names)
    command = np.asarray(scenario.command)

    mujoco.mj_resetDataKeyframe(runtime.model, runtime.data, runtime.model_map.reset_keyframe_id)
    runtime.data.ctrl[:] = 0.0
    runtime.data.xfrc_applied[:] = 0.0
    runtime.previous_action.fill(0.0)
    reset_pre = {
        "root_qpos_engine_native": runtime.data.qpos[:7].copy().tolist(),
        "root_qvel_engine_native": runtime.data.qvel[:6].copy().tolist(),
        "all_hinge_position_named": runtime.data.qpos[all_hinge_qpos].copy().tolist(),
        "all_hinge_velocity_named": runtime.data.qvel[all_hinge_dof].copy().tolist(),
        "available_fields": [
            "root_qpos_engine_native",
            "root_qvel_engine_native",
            "all_hinge_position_named",
            "all_hinge_velocity_named",
        ],
    }
    mujoco.mj_forward(runtime.model, runtime.data)
    initial_com_world = np.asarray(runtime.data.xipos[runtime.model_map.base_body_id]).copy()
    reset_post_state = _capture_state(runtime, all_hinge_qpos, all_hinge_dof, initial_com_world)
    reset_returned = build_actor_observation(
        reset_post_state["kinematic"], command, runtime.previous_action, contract
    )
    reset_post = _serializable_state(
        reset_post_state,
        q_nominal=contract.q_nominal,
        command=command,
        previous_action=runtime.previous_action,
    )
    reset_post["actor_obs_policy_rebuilt"] = reset_returned.tolist()
    write_json(
        output / "reset_snapshot.json",
        {
            "schema_version": SCHEMA_VERSION,
            "reset_written_pre_forward": reset_pre,
            "reset_forwarded_post_forward": reset_post,
            "reset_returned_to_policy": {
                "actor_obs_policy_returned": reset_returned.tolist(),
                "previous_action": runtime.previous_action.tolist(),
            },
        },
    )

    observation = reset_returned
    control_rows: list[dict[str, Any]] = []
    substep_rows: list[dict[str, Any]] = []
    stopped_reason = 0
    for tick in range(control_ticks):
        previous_action = runtime.previous_action.copy()
        pre_control = _capture_state(runtime, all_hinge_qpos, all_hinge_dof, initial_com_world)
        physical_observation = _physical_observation(pre_control, command, previous_action, contract.q_nominal)
        rebuilt_observation = build_actor_observation(
            pre_control["kinematic"], command, previous_action, contract
        )
        if not np.allclose(observation, rebuilt_observation, rtol=0.0, atol=1.0e-7):
            raise ValueError("Actual policy observation differs from rebuilt pre-step observation")
        with torch.inference_mode():
            output_tensor = policy(torch.from_numpy(observation).unsqueeze(0))
        policy_action = output_tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)
        if policy_action.shape != (6,) or not np.isfinite(policy_action).all():
            raise FloatingPointError("TorchScript Actor returned an invalid action")
        selected_action = _select_action(mode, tick, policy_action, action_sequence)
        targets = runtime.controller.prepare(selected_action)
        target_native = np.concatenate(
            (targets.leg_position_target_mujoco, targets.wheel_velocity_target_mujoco)
        )
        target_canonical = engine_native_to_canonical(target_native)

        torque_rows: dict[str, list[np.ndarray]] = {
            "pd_torque_unclipped": [],
            "pd_torque_effort_clipped": [],
            "mujoco_commanded_torque": [],
        }
        effort_events: list[np.ndarray] = []
        velocity_events: list[np.ndarray] = []
        guard_events: list[np.ndarray] = []
        wheel_force_rows: list[np.ndarray] = []
        wheel_impulse_rows: list[np.ndarray] = []
        wheel_active_rows: list[np.ndarray] = []
        wheel_normal_world_rows: list[np.ndarray] = []
        wheel_friction_world_rows: list[np.ndarray] = []
        external_torque_control_rows: list[np.ndarray] = []
        external_torque_world_rows: list[np.ndarray] = []
        final_post: dict[str, Any] | None = None
        for substep in range(contract.physics_steps_per_action):
            pre = _capture_state(runtime, all_hinge_qpos, all_hinge_dof, initial_com_world)
            unclipped_native, clipped_native = _torque_reference(runtime, pre, targets)
            commanded_native = runtime.controller.compute_torque(runtime.data, targets).copy()
            expected_command = clipped_native.copy()
            if not np.allclose(expected_command, commanded_native, rtol=0.0, atol=1.0e-12):
                raise ValueError("MuJoCo controller torque differs from the frozen debug reference")
            effort_event = (np.abs(unclipped_native) > contract.effort_limits).astype(np.int8)
            # PhysicsV5 has no joint-speed limiter; retain explicit inactive trace fields.
            velocity_event = np.zeros(6, dtype=np.int8)
            guard_event = np.zeros(6, dtype=np.int8)
            runtime.controller.apply_torque(runtime.data, commanded_native)
            external_torque_control, external_torque_world = _apply_base_pitch_torque(
                runtime, pitch_torque_pulse, tick
            )
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
                    commanded_native=commanded_native,
                    effort_event=effort_event,
                    velocity_event=velocity_event,
                    guard_event=guard_event,
                    external_torque_control=external_torque_control,
                    external_torque_world=external_torque_world,
                )
            )
            torque_rows["pd_torque_unclipped"].append(unclipped_native)
            torque_rows["pd_torque_effort_clipped"].append(clipped_native)
            torque_rows["mujoco_commanded_torque"].append(commanded_native)
            effort_events.append(effort_event)
            velocity_events.append(velocity_event)
            guard_events.append(guard_event)
            wheel_force_rows.append(post["wheel_normal_force_n"])
            wheel_impulse_rows.append(post["wheel_normal_impulse_ns"])
            wheel_active_rows.append(post["wheel_contact_active"])
            wheel_normal_world_rows.append(post["wheel_normal_force_world"])
            wheel_friction_world_rows.append(post["wheel_friction_force_world"])
            external_torque_control_rows.append(external_torque_control)
            external_torque_world_rows.append(external_torque_world)

        if final_post is None:
            raise RuntimeError("MuJoCo did not execute any physics substeps")
        runtime.previous_action = targets.clipped_action.copy()
        next_observation = build_actor_observation(
            final_post["kinematic"], command, runtime.previous_action, contract
        )
        diagnostic_flags = _diagnostic_flags(final_post)
        reason_code = _native_termination_reason(final_post)

        control_row: dict[str, Any] = {
            "control_tick": np.int64(tick),
            "control_time_s": np.float64(runtime.data.time),
            "command": command,
            "actor_obs_physical_pre_step": physical_observation,
            "actor_obs_policy_pre_step": observation,
            "previous_action_before_inference": previous_action,
            "actor_output_raw": policy_action,
            "action_clipped": targets.clipped_action,
            "target_command_canonical": target_canonical,
            "target_command_engine_native": target_native,
            "active_joint_position_canonical_post_step_pre_reset": final_post[
                "active_joint_position_canonical"
            ],
            "active_joint_velocity_canonical_post_step_pre_reset": final_post[
                "active_joint_velocity_canonical"
            ],
            "active_joint_position_engine_native_post_step_pre_reset": final_post[
                "active_joint_position_engine_native"
            ],
            "active_joint_velocity_engine_native_post_step_pre_reset": final_post[
                "active_joint_velocity_engine_native"
            ],
            "all_hinge_position_named_post_step_pre_reset": final_post[
                "all_hinge_position_named"
            ],
            "all_hinge_velocity_named_post_step_pre_reset": final_post[
                "all_hinge_velocity_named"
            ],
            "effort_limit_event": np.max(np.stack(effort_events), axis=0).astype(np.int8),
            "joint_velocity_limit_exceeded": np.max(np.stack(velocity_events), axis=0).astype(
                np.int8
            ),
            "mujoco_velocity_guard_active": np.max(np.stack(guard_events), axis=0).astype(np.int8),
            "base_com_position_engine_world_post_step_pre_reset": final_post[
                "base_com_position_engine_world"
            ],
            "base_com_position_diag_post_step_pre_reset": final_post["base_com_position_diag"],
            "base_orientation_control_wxyz_post_step_pre_reset": final_post[
                "base_orientation_control_wxyz"
            ],
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
            "wheel_contact_active": np.max(np.stack(wheel_active_rows), axis=0).astype(np.int8),
            "wheel_normal_force_n_mean": np.stack(wheel_force_rows).mean(axis=0),
            "wheel_normal_force_n_peak": np.stack(wheel_force_rows).max(axis=0),
            "wheel_normal_impulse_ns": np.stack(wheel_impulse_rows).sum(axis=0),
            "wheel_normal_force_world_mean": np.stack(wheel_normal_world_rows).mean(axis=0),
            "wheel_friction_force_world_mean": np.stack(
                wheel_friction_world_rows
            ).mean(axis=0),
            "external_base_torque_control": external_torque_control_rows[-1],
            "external_base_torque_engine_world_mean": np.stack(
                external_torque_world_rows
            ).mean(axis=0),
            "native_terminated_int8": np.int8(reason_code != 0),
            "native_truncated_int8": np.int8(0),
            "native_termination_reason_code_int16": np.int16(reason_code),
            "common_diagnostic_flags_int8": diagnostic_flags,
            "next_actor_obs_policy_returned": next_observation,
            "next_obs_is_reset_int8": np.int8(0),
        }
        for prefix, values in torque_rows.items():
            _aggregate_torque(control_row, prefix, values, contract.physics_dt_s)
        nan_rows = [np.full(6, np.nan) for _ in range(contract.physics_steps_per_action)]
        _aggregate_torque(
            control_row, "isaac_host_pd_torque_estimate", nan_rows, contract.physics_dt_s
        )
        control_rows.append(control_row)
        observation = next_observation
        if reason_code:
            stopped_reason = reason_code
            break

    control_trace = _rows_to_arrays(control_rows)
    substep_trace = _rows_to_arrays(substep_rows)
    validate_control_trace(control_trace)
    validate_substep_trace(
        substep_trace,
        physics_steps_per_action=contract.physics_steps_per_action,
        continuity_atol=1.0e-12,
    )
    save_npz(output / "control_trace.npz", control_trace)
    save_npz(output / "substep_trace.npz", substep_trace)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "collection_id": uuid.uuid4().hex,
        "engine": "mujoco",
        "engine_version": mujoco.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "inference_device": "cpu",
        "inference_dtype": "float32",
        "simulation_device": "cpu",
        "physics_pipeline": "cpu",
        "scenario_hash": scenario.payload_hash,
        "frozen_files": frozen_files,
        "scenario_variant": mode,
        "base_pitch_torque_pulse": pitch_torque_pulse.payload,
        "action_sequence_identity": action_sequence_identity,
        "command": list(scenario.command),
        "random_seed": scenario.random_seed,
        "requested_control_ticks": control_ticks,
        "completed_control_ticks": len(control_rows),
        "control_dt_s": contract.control_dt_s,
        "physics_dt_s": contract.physics_dt_s,
        "physics_steps_per_action": contract.physics_steps_per_action,
        "substep_continuity_atol": 1.0e-12,
        "canonical_joint_order": list(runtime.model_map.joint_names),
        "canonical_from_engine_native": [1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
        "all_hinge_order": all_hinge_names,
        "r_diag_from_engine_world": contract.r_control_from_mujoco.tolist(),
        "actor_sha256": sha256_file(actor_path),
        "policy_manifest_sha256": sha256_file(policy_manifest_path),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "model_xml_sha256": sha256_file(model_path),
        "render_mode": "headless",
        "observer_mode": "explicit_mujoco_runtime",
        "mujoco_solver": mujoco.mjtSolver(runtime.model.opt.solver).name,
        "mujoco_integrator": mujoco.mjtIntegrator(runtime.model.opt.integrator).name,
        "mujoco_iterations": int(runtime.model.opt.iterations),
        "mujoco_line_search_iterations": int(runtime.model.opt.ls_iterations),
        "mujoco_noslip_iterations": int(runtime.model.opt.noslip_iterations),
        "mujoco_ccd_iterations": int(runtime.model.opt.ccd_iterations),
        "mujoco_tolerance": float(runtime.model.opt.tolerance),
        "mujoco_line_search_tolerance": float(runtime.model.opt.ls_tolerance),
        "mujoco_noslip_tolerance": float(runtime.model.opt.noslip_tolerance),
        "mujoco_ccd_tolerance": float(runtime.model.opt.ccd_tolerance),
        "contact_observation_mode": "native",
        "contact_processing_state": "native_mujoco_contacts",
        "contact_active_force_threshold_n": 1.0,
        "termination_reason_codes": {str(key): value for key, value in TERMINATION_REASON_CODES.items()},
        "unavailable_fields": {
            "isaac_host_pd_torque_estimate": "not available in MuJoCo",
            "physx_internal_drive_torque": "PhysX public API unavailable",
        },
        "stopped_reason_code": stopped_reason,
        "stopped_reason": TERMINATION_REASON_CODES[stopped_reason],
    }
    mujoco_sources = {
        "mujoco_observation_adapter": root / "sim2sim/mujoco/wheelleg_mujoco/observation.py",
        "mujoco_action_adapter": root / "sim2sim/mujoco/wheelleg_mujoco/control.py",
        "mujoco_contract": root / "sim2sim/mujoco/wheelleg_mujoco/contract.py",
        "mujoco_runtime": root / "sim2sim/mujoco/wheelleg_mujoco/runner.py",
    }
    metadata["mujoco_source_sha256"] = build_file_hashes(mujoco_sources)
    write_json(output / "metadata.json", metadata)
    source_paths = {
        "collector": Path(__file__),
        "trace_schema": Path(__file__).with_name("trace_schema.py"),
        "comparison": Path(__file__).with_name("compare_traces.py"),
        "stand_scenario": Path(__file__).with_name("stand_scenario.py"),
        "pitch_torque_pulse": Path(__file__).with_name("pitch_torque_pulse.py"),
        **mujoco_sources,
        "metadata": output / "metadata.json",
        "reset_snapshot": output / "reset_snapshot.json",
        "control_trace": output / "control_trace.npz",
        "substep_trace": output / "substep_trace.npz",
    }
    if embedded_action_path is not None:
        source_paths["input_action_sequence"] = embedded_action_path
    write_json(output / "file_hashes.json", build_file_hashes(source_paths))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a deterministic WheelLeg MuJoCo debug trace.")
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
    parser.add_argument("--base-pitch-torque-nm", type=float, default=0.0)
    parser.add_argument("--pitch-torque-start-tick", type=int, default=0)
    parser.add_argument("--pitch-torque-end-tick", type=int, default=0)
    args = parser.parse_args()
    pitch_torque_pulse = BasePitchTorquePulse(
        magnitude_nm=args.base_pitch_torque_nm,
        start_tick=args.pitch_torque_start_tick,
        end_tick=args.pitch_torque_end_tick,
    )
    sequence = None
    action_sequence_identity = None
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
    metadata = collect_mujoco_trace(
        project_root=args.project_root,
        output_directory=args.output,
        control_ticks=args.control_ticks,
        mode=args.mode,
        action_sequence=sequence,
        action_sequence_identity=action_sequence_identity,
        pitch_torque_pulse=pitch_torque_pulse,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
