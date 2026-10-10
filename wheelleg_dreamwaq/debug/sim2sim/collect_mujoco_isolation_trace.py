from __future__ import annotations

import argparse
import json
import math
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from wheelleg_mujoco.contract import load_policy_contract
from wheelleg_mujoco.control import MixedActionController
from wheelleg_mujoco.metrics import LOOP_SITE_PAIRS, VIRTUAL_LEG_JOINTS
from wheelleg_mujoco.model_map import ACTUATOR_NAMES, CANONICAL_JOINT_ORDER, CONTROLLED_JOINT_SIGNS
from wheelleg_mujoco.observation import collect_kinematic_state

from .isolation_scenarios import IsolationScenario
from .stand_scenario import FROZEN_FILES, verify_frozen_files
from .trace_schema import (
    build_file_hashes,
    engine_native_to_canonical,
    save_npz,
    sha256_file,
    stable_payload_hash,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "MujocoIsolationTraceV1"
CONTACT_FORCE_THRESHOLD_N = 1.0


def _required_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model is missing required {object_type.name}: {name}")
    return object_id


@dataclass(frozen=True)
class DebugModelMap:
    joint_names: tuple[str, ...]
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    actuator_ids: np.ndarray
    base_body_id: int
    free_joint_id: int | None
    reset_keyframe_id: int

    def joint_position_mujoco(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.qpos[self.qpos_addresses], dtype=np.float64)

    def joint_velocity_mujoco(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.qvel[self.dof_addresses], dtype=np.float64)

    def joint_position_control(self, data: mujoco.MjData) -> np.ndarray:
        return self.joint_position_mujoco(data) * CONTROLLED_JOINT_SIGNS

    def joint_velocity_control(self, data: mujoco.MjData) -> np.ndarray:
        return self.joint_velocity_mujoco(data) * CONTROLLED_JOINT_SIGNS


def build_debug_model_map(model: mujoco.MjModel) -> DebugModelMap:
    joint_ids = np.asarray(
        [_required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in CANONICAL_JOINT_ORDER],
        dtype=np.int32,
    )
    if any(model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE for joint_id in joint_ids):
        raise ValueError("Every ActionV1 joint must be a MuJoCo hinge")
    actuator_ids = np.asarray(
        [_required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES],
        dtype=np.int32,
    )
    if not np.array_equal(model.actuator_trnid[actuator_ids, 0], joint_ids):
        raise ValueError("Debug actuator-to-joint mapping does not match ActionV1")
    free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    if free_joint_id >= 0 and model.jnt_type[free_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("base_free exists but is not a free joint")
    return DebugModelMap(
        joint_names=CANONICAL_JOINT_ORDER,
        joint_ids=joint_ids,
        qpos_addresses=np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32),
        dof_addresses=np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32),
        actuator_ids=actuator_ids,
        base_body_id=_required_id(model, mujoco.mjtObj.mjOBJ_BODY, "base"),
        free_joint_id=None if free_joint_id < 0 else int(free_joint_id),
        reset_keyframe_id=_required_id(model, mujoco.mjtObj.mjOBJ_KEY, "contract_v4_reset"),
    )


def _validate_scenario_model(
    model: mujoco.MjModel,
    model_map: DebugModelMap,
    scenario: IsolationScenario,
) -> None:
    expected_free_joint = scenario.kind != "fixed_base"
    if (model_map.free_joint_id is not None) != expected_free_joint:
        raise ValueError(f"{scenario.kind} has an unexpected base_free configuration")

    expected_ground_contact = scenario.kind == "grounded_pitch"
    if scenario.has_ground_contact != expected_ground_contact:
        raise ValueError(f"{scenario.kind} has an inconsistent ground contact declaration")
    expected_gravity = np.asarray((0.0, 0.0, -9.81) if expected_ground_contact else (0.0, 0.0, 0.0))
    if not np.allclose(model.opt.gravity, expected_gravity, rtol=0.0, atol=0.0):
        raise ValueError(f"{scenario.kind} has an unexpected gravity vector")

    pair_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, pair_id)
        for pair_id in range(model.npair)
    }
    required_ground_pairs = {"floor_left_wheel", "floor_right_wheel"}
    if expected_ground_contact:
        if pair_names != required_ground_pairs:
            raise ValueError("grounded_pitch must have exactly two wheel ground contact pairs")
    elif pair_names:
        raise ValueError(f"{scenario.kind} must not have ground contact pairs")

    for geom_name in ("left_wheel_proxy", "right_wheel_proxy"):
        geom_id = _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_SPHERE:
            raise ValueError(f"{geom_name} must remain a sphere in isolation experiments")


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"Rotation must be 3x3, got {rotation.shape}")
    quaternion = np.empty(4, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion[:] = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion[:] = (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion[:] = (
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion[:] = (
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def _control_rotation_from_engine(model: mujoco.MjModel, data: mujoco.MjData, model_map: DebugModelMap, transform: np.ndarray) -> np.ndarray:
    engine_rotation = np.asarray(data.xmat[model_map.base_body_id], dtype=np.float64).reshape(3, 3)
    return transform @ engine_rotation @ transform.T


def _roll_pitch_yaw(rotation: np.ndarray) -> np.ndarray:
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.asarray((roll, pitch, yaw), dtype=np.float64)


def _apply_control_pitch(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    model_map: DebugModelMap,
    transform: np.ndarray,
    pitch_deg: float,
) -> None:
    if model_map.free_joint_id is None:
        raise ValueError("A grounded pitch perturbation requires base_free")
    pitch = math.radians(pitch_deg)
    cosine = math.cos(pitch)
    sine = math.sin(pitch)
    control_rotation = np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )
    engine_rotation = transform.T @ control_rotation @ transform
    quaternion = _matrix_to_quaternion_wxyz(engine_rotation)
    qpos_address = int(model.jnt_qposadr[model_map.free_joint_id])
    data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    wheel_geom_ids = np.asarray(
        [
            _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_proxy"),
            _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel_proxy"),
        ],
        dtype=np.int32,
    )
    wheel_bottom = data.geom_xpos[wheel_geom_ids, 2] - model.geom_size[wheel_geom_ids, 0]
    data.qpos[qpos_address + 2] -= float(np.min(wheel_bottom))
    mujoco.mj_forward(model, data)


def _loop_closure_errors(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    errors = []
    for first, second in LOOP_SITE_PAIRS:
        first_id = _required_id(model, mujoco.mjtObj.mjOBJ_SITE, first)
        second_id = _required_id(model, mujoco.mjtObj.mjOBJ_SITE, second)
        errors.append(np.linalg.norm(data.site_xpos[first_id] - data.site_xpos[second_id]))
    return np.asarray(errors, dtype=np.float64)


def _virtual_leg_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    model_map: DebugModelMap,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_world_from_body = np.asarray(data.xmat[model_map.base_body_id], dtype=np.float64).reshape(3, 3)
    lengths = []
    angles = []
    for hip_name, wheel_name in VIRTUAL_LEG_JOINTS:
        hip_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, hip_name)
        wheel_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, wheel_name)
        vector_body = rotation_world_from_body.T @ (data.xanchor[wheel_id] - data.xanchor[hip_id])
        lateral = float(vector_body[1])
        downward = float(-vector_body[2])
        lengths.append(math.hypot(lateral, downward))
        angles.append(math.atan2(downward, lateral))
    return np.asarray(lengths, dtype=np.float64), np.asarray(angles, dtype=np.float64)


def _wheel_contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    wheel_ids = {
        _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_proxy"): 0,
        _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel_proxy"): 1,
    }
    force = np.zeros(2, dtype=np.float64)
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        wheel_index = wheel_ids.get(int(contact.geom1))
        if wheel_index is None:
            wheel_index = wheel_ids.get(int(contact.geom2))
        if wheel_index is None:
            continue
        mujoco.mj_contactForce(model, data, contact_index, contact_force)
        force[wheel_index] += max(0.0, float(contact_force[0]))
    return force


def _snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    model_map: DebugModelMap,
    contract: Any,
) -> dict[str, np.ndarray | float]:
    state = collect_kinematic_state(model, data, model_map, contract)
    rotation = _control_rotation_from_engine(model, data, model_map, contract.r_control_from_mujoco)
    lengths, angles = _virtual_leg_state(model, data, model_map)
    return {
        "active_joint_position_canonical": model_map.joint_position_control(data).copy(),
        "active_joint_velocity_canonical": model_map.joint_velocity_control(data).copy(),
        "base_com_position_world": np.asarray(data.xipos[model_map.base_body_id], dtype=np.float64).copy(),
        "base_rpy_control": _roll_pitch_yaw(rotation),
        "base_linear_velocity_control": state.com_linear_velocity_control.copy(),
        "base_angular_velocity_control": state.angular_velocity_control.copy(),
        "projected_gravity_control": state.projected_gravity_control.copy(),
        "base_height_m": float(state.base_height_m),
        "virtual_leg_length_m": lengths,
        "virtual_leg_phi0_rad": angles,
        "loop_closure_error_m": _loop_closure_errors(model, data),
    }


def _rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Refusing to write an empty isolation trace")
    return {name: np.asarray([row[name] for row in rows]) for name in rows[0]}


def _collect_isolation_trace_into_directory(
    project_root: str | Path,
    output_directory: str | Path,
    scenario: IsolationScenario,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen_files = verify_frozen_files(root)
    actor_path = root / FROZEN_FILES["actor"][0]
    policy_manifest_path = root / FROZEN_FILES["policy_manifest"][0]
    model_manifest_path = root / FROZEN_FILES["model_manifest"][0]
    contract, _, _ = load_policy_contract(policy_manifest_path, actor_path, model_manifest_path)

    model_path = root / scenario.model_relative
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if abs(float(model.opt.timestep) - contract.physics_dt_s) > 1.0e-15:
        raise ValueError("Isolation model timestep differs from the frozen policy contract")
    data = mujoco.MjData(model)
    model_map = build_debug_model_map(model)
    _validate_scenario_model(model, model_map, scenario)
    controller = MixedActionController(model_map, contract)
    action_sequence = scenario.action_sequence()

    mujoco.mj_resetDataKeyframe(model, data, model_map.reset_keyframe_id)
    data.ctrl[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    if scenario.kind == "grounded_pitch":
        _apply_control_pitch(
            model,
            data,
            model_map,
            contract.r_control_from_mujoco,
            scenario.initial_pitch_deg,
        )
    initial = _snapshot(model, data, model_map, contract)

    rows: list[dict[str, Any]] = []
    stopped_reason = "completed"
    for tick, action in enumerate(action_sequence):
        targets = controller.prepare(action)
        target_native = np.concatenate(
            (targets.leg_position_target_mujoco, targets.wheel_velocity_target_mujoco)
        )
        torque_native_rows = []
        torque_canonical_rows = []
        effort_limit_rows = []
        wheel_force_rows = []
        for _ in range(contract.physics_steps_per_action):
            torque_native = controller.compute_torque(data, targets).copy()
            controller.apply_torque(data, torque_native)
            mujoco.mj_step(model, data)
            torque_native_rows.append(torque_native)
            torque_canonical_rows.append(engine_native_to_canonical(torque_native))
            effort_limit_rows.append(
                np.isclose(np.abs(torque_native), controller.effort_limits, rtol=0.0, atol=1.0e-12)
            )
            wheel_force_rows.append(_wheel_contact_force(model, data))
        post = _snapshot(model, data, model_map, contract)
        torque_native_array = np.asarray(torque_native_rows)
        torque_canonical_array = np.asarray(torque_canonical_rows)
        wheel_force = np.asarray(wheel_force_rows)
        rows.append(
            {
                "control_tick": np.int64(tick),
                "control_time_s": np.float64(data.time),
                "action_canonical": targets.clipped_action.copy(),
                "target_canonical": engine_native_to_canonical(target_native),
                "target_engine_native": target_native.copy(),
                "commanded_torque_canonical_mean": torque_canonical_array.mean(axis=0),
                "commanded_torque_canonical_peak_abs": np.max(
                    np.abs(torque_canonical_array), axis=0
                ),
                "commanded_torque_canonical_impulse_nms": (
                    torque_canonical_array.sum(axis=0) * contract.physics_dt_s
                ),
                "commanded_torque_engine_native_mean": torque_native_array.mean(axis=0),
                "commanded_torque_engine_native_peak_abs": np.max(
                    np.abs(torque_native_array), axis=0
                ),
                "commanded_torque_engine_native_impulse_nms": (
                    torque_native_array.sum(axis=0) * contract.physics_dt_s
                ),
                "effort_limit_fraction": np.asarray(effort_limit_rows).mean(axis=0),
                "velocity_limit_fraction": np.zeros(6, dtype=np.float64),  # PhysicsV5: disabled.
                "active_joint_position_canonical": post["active_joint_position_canonical"],
                "active_joint_velocity_canonical": post["active_joint_velocity_canonical"],
                "active_joint_position_engine_native": model_map.joint_position_mujoco(data).copy(),
                "active_joint_velocity_engine_native": model_map.joint_velocity_mujoco(data).copy(),
                "base_com_position_world": post["base_com_position_world"],
                "base_rpy_control": post["base_rpy_control"],
                "base_linear_velocity_control": post["base_linear_velocity_control"],
                "base_angular_velocity_control": post["base_angular_velocity_control"],
                "projected_gravity_control": post["projected_gravity_control"],
                "base_height_m": np.float64(post["base_height_m"]),
                "virtual_leg_length_m": post["virtual_leg_length_m"],
                "virtual_leg_phi0_rad": post["virtual_leg_phi0_rad"],
                "loop_closure_error_m": post["loop_closure_error_m"],
                "wheel_contact_active": np.max(
                    wheel_force > CONTACT_FORCE_THRESHOLD_N, axis=0
                ).astype(np.int8),
                "wheel_normal_force_n_mean": wheel_force.mean(axis=0),
                "wheel_normal_force_n_peak": wheel_force.max(axis=0),
            }
        )
        if not all(np.isfinite(np.asarray(value)).all() for value in rows[-1].values()):
            stopped_reason = "non_finite"
            break
        if scenario.kind == "grounded_pitch" and abs(float(post["base_rpy_control"][1])) > 0.8:
            stopped_reason = "tilt"
            break

    trace = _rows_to_arrays(rows)
    trace_path = save_npz(output / "isolation_trace.npz", trace)
    initial_pitch_deg = math.degrees(float(initial["base_rpy_control"][1]))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario.payload,
        "scenario_hash": stable_payload_hash(scenario.payload),
        "requested_control_ticks": scenario.control_ticks,
        "completed_control_ticks": len(rows),
        "stopped_reason": stopped_reason,
        "physics_dt_s": contract.physics_dt_s,
        "physics_steps_per_action": contract.physics_steps_per_action,
        "control_dt_s": contract.control_dt_s,
        "model_path": scenario.model_relative.replace("\\", "/"),
        "model_sha256": sha256_file(model_path),
        "model_has_free_joint": model_map.free_joint_id is not None,
        "contact_force_threshold_n": CONTACT_FORCE_THRESHOLD_N,
        "initial_state": {
            "pitch_deg": initial_pitch_deg,
            "base_rpy_control_rad": np.asarray(initial["base_rpy_control"]).tolist(),
            "base_com_position_world": np.asarray(initial["base_com_position_world"]).tolist(),
            "active_joint_position_canonical": np.asarray(
                initial["active_joint_position_canonical"]
            ).tolist(),
            "loop_closure_error_m": np.asarray(initial["loop_closure_error_m"]).tolist(),
        },
        "frozen_files": frozen_files,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
    }
    metadata_path = write_json(output / "metadata.json", metadata)
    write_json(
        output / "file_hashes.json",
        build_file_hashes(
            {
                "model_xml": model_path,
                "metadata": metadata_path,
                "isolation_trace": trace_path,
                "collector": Path(__file__),
                "scenario_contract": Path(__file__).with_name("isolation_scenarios.py"),
            }
        ),
    )
    return metadata


def collect_isolation_trace(
    project_root: str | Path,
    output_directory: str | Path,
    scenario: IsolationScenario,
) -> dict[str, Any]:
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        metadata = _collect_isolation_trace_into_directory(project_root, temporary, scenario)
        temporary.rename(output)
        return metadata
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _scenario_from_args(args: argparse.Namespace) -> IsolationScenario:
    if args.scenario == "grounded_pitch":
        pulse_arguments = {
            "channel": args.channel,
            "wheel_pulse_mode": args.wheel_pulse_mode,
            "amplitude": args.amplitude,
            "pulse_start_tick": args.pulse_start_tick,
            "pulse_end_tick": args.pulse_end_tick,
        }
        supplied = sorted(name for name, value in pulse_arguments.items() if value is not None)
        if supplied:
            raise ValueError(f"grounded_pitch does not accept pulse arguments: {supplied}")
        return IsolationScenario.grounded_pitch(
            control_ticks=args.control_ticks,
            initial_pitch_deg=5.0 if args.initial_pitch_deg is None else args.initial_pitch_deg,
            wheel_common_action=0.2 if args.wheel_action is None else args.wheel_action,
        )
    grounded_arguments = {
        "initial_pitch_deg": args.initial_pitch_deg,
        "wheel_action": args.wheel_action,
    }
    supplied = sorted(name for name, value in grounded_arguments.items() if value is not None)
    if supplied:
        raise ValueError(f"{args.scenario} does not accept grounded arguments: {supplied}")
    if args.scenario == "suspended" and args.wheel_pulse_mode is not None:
        if args.channel is not None:
            raise ValueError("suspended accepts either --channel or --wheel-pulse-mode, not both")
        return IsolationScenario.suspended_wheel_pulse(
            control_ticks=args.control_ticks,
            mode=args.wheel_pulse_mode,
            amplitude=0.1 if args.amplitude is None else args.amplitude,
            pulse_start_tick=5 if args.pulse_start_tick is None else args.pulse_start_tick,
            pulse_end_tick=20 if args.pulse_end_tick is None else args.pulse_end_tick,
        )
    if args.scenario == "fixed_base" and args.wheel_pulse_mode is not None:
        raise ValueError("fixed_base does not accept --wheel-pulse-mode")
    factory = {
        "fixed_base": IsolationScenario.fixed_base_pulse,
        "suspended": IsolationScenario.suspended_pulse,
    }[args.scenario]
    return factory(
        control_ticks=args.control_ticks,
        channel=4 if args.channel is None else args.channel,
        amplitude=0.1 if args.amplitude is None else args.amplitude,
        pulse_start_tick=5 if args.pulse_start_tick is None else args.pulse_start_tick,
        pulse_end_tick=20 if args.pulse_end_tick is None else args.pulse_end_tick,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a debug-only MuJoCo isolation trace.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=("fixed_base", "suspended", "grounded_pitch"),
        required=True,
    )
    parser.add_argument("--control-ticks", type=int, default=50)
    parser.add_argument("--channel", type=int)
    parser.add_argument("--wheel-pulse-mode", choices=("common", "differential"))
    parser.add_argument("--amplitude", type=float)
    parser.add_argument("--pulse-start-tick", type=int)
    parser.add_argument("--pulse-end-tick", type=int)
    parser.add_argument("--initial-pitch-deg", type=float)
    parser.add_argument(
        "--wheel-action",
        type=float,
        help="Normalized canonical ActionV1 value; 0.2 maps to a 5 rad/s wheel target.",
    )
    args = parser.parse_args()
    metadata = collect_isolation_trace(
        args.project_root,
        args.output,
        _scenario_from_args(args),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
