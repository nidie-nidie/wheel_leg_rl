from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from wheelleg_mujoco.contract import AdapterContract
from wheelleg_mujoco.control import MixedActionController
from wheelleg_mujoco.metrics import collect_metrics
from wheelleg_mujoco.model_map import CANONICAL_JOINT_ORDER, build_model_map
from wheelleg_mujoco.observation import build_actor_observation, collect_kinematic_state
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parents[1]))  # Shared debug modules, including isolated test runs.
MODEL_PATH = PROJECT_ROOT / "models" / "wheel_leg_urdf4_v1.xml"
Q_NOMINAL = np.array((-0.33367134, 0.33367134, -0.33367134, 0.33367134))


def _contract() -> AdapterContract:
    return AdapterContract(
        q_nominal=Q_NOMINAL.copy(),
        action_clip=1.0,
        leg_action_scale=0.35,
        wheel_action_scale=25.0,
        leg_target_lower=np.full(4, -1.0),
        leg_target_upper=np.full(4, 1.0),
        wheel_signs=np.array((1.0, -1.0)),
        r_control_from_mujoco=np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        normalization={
            "angular_velocity_scale": 0.25,
            "angular_velocity_clip": 5.0,
            "projected_gravity_clip": 1.0,
            "vx_max_abs": 1.5,
            "yaw_rate_max_abs": 1.0,
            "nominal_base_height": 0.20,
            "height_command_span": 0.04,
            "leg_position_scale": 1.0,
            "leg_position_clip": 1.5,
            "joint_velocity_scale": 0.05,
            "joint_velocity_clip": 5.0,
        },
        leg_kp=120.0,
        leg_kd=4.0,
        wheel_kd=0.6,
        effort_limits=np.array((18.0, 18.0, 18.0, 18.0, 9.0, 9.0)),
        physics_dt_s=0.001,
        physics_steps_per_action=20,
        control_dt_s=0.02,
    )


def _model_data():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "contract_v4_reset")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return model, data, build_model_map(model)


def test_model_map_uses_canonical_order_and_control_signs() -> None:
    model, data, model_map = _model_data()
    assert model_map.joint_names == CANONICAL_JOINT_ORDER
    source_positions = np.array((-0.2, 0.3, -0.4, 0.5, 1.25, 2.5))
    source_velocities = np.array((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    data.qpos[model_map.qpos_addresses] = source_positions
    data.qvel[model_map.dof_addresses] = source_velocities
    np.testing.assert_allclose(model_map.joint_position_control(data), (-0.2, 0.3, -0.4, 0.5, 1.25, -2.5))
    np.testing.assert_allclose(model_map.joint_velocity_control(data), (1.0, 2.0, 3.0, 4.0, 5.0, -6.0))
    assert model_map.actuator_ids.shape == (6,)
    assert model.nu == 6


def test_action_adapter_clips_scales_signs_and_saturates_torque() -> None:
    _, data, model_map = _model_data()
    controller = MixedActionController(model_map, _contract())
    targets = controller.prepare(np.array((2.0, -2.0, 0.5, -0.5, 0.4, 0.4)))
    np.testing.assert_allclose(targets.clipped_action, (1.0, -1.0, 0.5, -0.5, 0.4, 0.4))
    np.testing.assert_allclose(
        targets.leg_position_target_mujoco,
        Q_NOMINAL + 0.35 * np.array((1.0, -1.0, 0.5, -0.5)),
    )
    np.testing.assert_allclose(targets.wheel_velocity_target_mujoco, (10.0, -10.0))

    data.qpos[model_map.qpos_addresses[:4]] = 10.0
    data.qvel[model_map.dof_addresses] = 100.0
    torque = controller.compute_torque(data, targets)
    np.testing.assert_allclose(np.abs(torque[:4]), 18.0)
    np.testing.assert_allclose(np.abs(torque[4:]), 9.0)
    controller.apply_torque(data, torque)
    np.testing.assert_allclose(data.ctrl[model_map.actuator_ids], torque)

    data.qpos[model_map.qpos_addresses[:4]] = -10.0
    data.qvel[model_map.dof_addresses] = 50.0
    targets = controller.prepare(np.ones(6))
    torque = controller.compute_torque(data, targets)
    # Above the former 45 rad/s limit, only PD and motor effort clipping apply.
    np.testing.assert_array_equal(torque[:4], np.full(4, 18.0))
    np.testing.assert_array_equal(torque[4:], np.full(2, -9.0))


def test_root_cause_target_probe_matches_formal_motor_torque_above_old_cap() -> None:
    from debug.sim2sim.root_cause_suite.mujoco_worker import _formal_target_torque

    model, data, model_map = _model_data()
    contract = _contract()
    data.qpos[model_map.qpos_addresses[:4]] = -10.0
    data.qvel[model_map.dof_addresses] = 50.0
    action = np.ones(6)
    controller = MixedActionController(model_map, contract)
    reference = controller.compute_torque(data, controller.prepare(action))
    np.testing.assert_array_equal(_formal_target_torque(model, data, action, contract), reference)
    np.testing.assert_array_equal(reference[:4], np.full(4, 18.0))


def test_debug_timing_step_retains_motor_control_and_reports_no_speed_guard() -> None:
    from debug.sim2sim.dreamwaq_debug_contract import TIMING_PROFILES
    from debug.sim2sim.evaluate_debug_mujoco import step_with_debug_timing

    runtime = WheelLegMujocoRuntime(MODEL_PATH, _contract())
    action = np.full(6, 0.1)
    command = np.array((0.0, 0.0, 0.20))
    result = step_with_debug_timing(runtime, TIMING_PROFILES["formal_1ms"], action, command)
    assert result.physics_steps == 20
    assert result.metrics["velocity_limit_event_fraction"] == 0.0
    assert np.isfinite(result.observation).all()
    assert np.isfinite(result.applied_torque).all()


def test_debug_joint_speed_failure_threshold_remains_diagnostic_only() -> None:
    from debug.sim2sim.evaluate_debug_mujoco import _failure_reason

    runtime = WheelLegMujocoRuntime(MODEL_PATH, _contract())
    metrics = {
        "base_height_m": 0.20, "tilt_rad": 0.0, "root_linear_speed_mps": 0.0,
        "root_angular_speed_rad_s": 0.0, "max_hinge_speed_rad_s": 79.0,
        "max_loop_closure_error_m": 0.0, "l0_left_m": 0.20, "l0_right_m": 0.20,
    }
    assert _failure_reason(runtime, metrics) is None
    before = runtime.data.qvel.copy()
    metrics["max_hinge_speed_rad_s"] = 81.0
    assert _failure_reason(runtime, metrics) == "joint_velocity"
    np.testing.assert_array_equal(runtime.data.qvel, before)


@pytest.mark.parametrize("with_external_wrench", (False, True))
def test_runtime_adds_no_speed_limit_wrench_and_matches_direct_physics(with_external_wrench: bool) -> None:
    runtime = WheelLegMujocoRuntime(MODEL_PATH, _contract())
    command = np.array((0.0, 0.0, 0.20))
    runtime.reset(command)
    # Remove ground contact; exceed both former rigid-body speed thresholds.
    runtime.data.qpos[2] += 100.0
    runtime.data.qvel[:6] = (200.0, 0.0, 0.0, 10.0, 0.0, 0.0)
    mujoco.mj_forward(runtime.model, runtime.data)
    if with_external_wrench:
        runtime.data.xfrc_applied[runtime.model_map.base_body_id] = (1., 2., 3., 4., 5., 6.)
    expected_wrench = runtime.data.xfrc_applied.copy()
    reference = mujoco.MjData(runtime.model)
    mujoco.mj_copyData(reference, runtime.model, runtime.data)
    action = np.full(6, 0.2)
    targets = runtime.controller.prepare(action)
    for _ in range(runtime.contract.physics_steps_per_action):
        torque = runtime.controller.compute_torque(reference, targets)
        runtime.controller.apply_torque(reference, torque)
        mujoco.mj_step(runtime.model, reference)
    result = runtime.step(action, command)
    np.testing.assert_array_equal(runtime.data.xfrc_applied, expected_wrench)
    np.testing.assert_array_equal(runtime.data.qfrc_applied, np.zeros(runtime.model.nv))
    np.testing.assert_array_equal(runtime.data.qpos, reference.qpos)
    np.testing.assert_array_equal(runtime.data.qvel, reference.qvel)
    assert result.physics_steps == 20
    assert np.isfinite(result.observation).all()


def test_observation_uses_com_height_and_frozen_normalization() -> None:
    model, data, model_map = _model_data()
    command = np.array((1.5, -1.0, 0.24))
    previous_action = np.array((-2.0, -0.5, 0.0, 0.5, 1.0, 2.0))
    contract = _contract()
    state = collect_kinematic_state(model, data, model_map, contract)
    observation = build_actor_observation(state, command, previous_action, contract)

    assert observation.shape == (25,)
    assert observation.dtype == np.float32
    np.testing.assert_allclose(observation[3:6], (0.0, 0.0, -1.0), atol=1.0e-7)
    np.testing.assert_allclose(observation[6:9], (1.0, -1.0, 1.0), atol=1.0e-7)
    np.testing.assert_allclose(observation[19:25], (-1.0, -0.5, 0.0, 0.5, 1.0, 1.0))
    assert not np.isclose(data.qpos[2], data.xipos[model_map.base_body_id, 2])
    assert state.base_height_m == data.xipos[model_map.base_body_id, 2]


def test_frame_transform_and_nonzero_pose_are_finite() -> None:
    model, data, model_map = _model_data()
    contract = _contract()
    np.testing.assert_allclose(
        contract.r_control_from_mujoco @ np.array((1.0, 2.0, 3.0)),
        (-2.0, 1.0, 3.0),
    )
    angle = np.pi / 2.0
    data.qpos[3:7] = (np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0))
    data.qvel[:6] = (0.5, -0.25, 0.1, 0.2, -0.3, 0.4)
    mujoco.mj_forward(model, data)
    state = collect_kinematic_state(model, data, model_map, contract)
    observation = build_actor_observation(
        state,
        np.array((0.0, 0.0, 0.20)),
        np.zeros(6),
        contract,
    )
    metrics = collect_metrics(model, data, model_map, contract, state)
    assert np.isfinite(observation).all()
    assert np.isfinite(np.asarray(list(metrics.values()), dtype=np.float64)).all()
    assert abs(abs(metrics["yaw_rad"]) - np.pi / 2.0) < 1.0e-6


def test_runtime_executes_exactly_twenty_physics_steps_per_action() -> None:
    runtime = WheelLegMujocoRuntime(MODEL_PATH, _contract())
    observation = runtime.reset(np.array((0.0, 0.0, 0.20)))
    assert observation.shape == (25,)
    start_time = runtime.data.time
    result = runtime.step(np.zeros(6), np.array((0.0, 0.0, 0.20)))
    assert abs(runtime.data.time - start_time - 0.02) < 1.0e-12
    assert result.physics_steps == 20
    assert result.observation.shape == (25,)
    assert np.isfinite(result.observation).all()
    assert result.metrics["max_loop_closure_error_m"] < 5.0e-3
    assert result.metrics["base_height_m"] > 0.05


def test_dreamwaq_runtime_owns_frame_major_history_without_changing_physics() -> None:
    contract = replace(
        _contract(),
        policy_input_dimension=125,
        history_length=5,
        history_layout="frame_major",
    )
    runtime = WheelLegMujocoRuntime(MODEL_PATH, contract)
    initial = runtime.reset(np.array((0.0, 0.0, 0.20)))
    assert initial.shape == (125,)
    initial_frames = initial.reshape(5, 25)
    np.testing.assert_array_equal(initial_frames, np.repeat(initial_frames[:1], 5, axis=0))

    start_time = runtime.data.time
    result = runtime.step(np.zeros(6), np.array((0.0, 0.0, 0.20)))
    assert abs(runtime.data.time - start_time - 0.02) < 1.0e-12
    frames = result.observation.reshape(5, 25)
    np.testing.assert_array_equal(frames[:4], initial_frames[1:])
    assert result.observation.dtype == np.float32
    assert np.isfinite(result.observation).all()


def test_repeated_reset_discards_old_motion_and_repeats_current_history() -> None:
    contract = replace(_contract(), policy_input_dimension=125, history_length=5, history_layout="frame_major")
    runtime = WheelLegMujocoRuntime(MODEL_PATH, contract)
    command = np.array((0.0, 0.0, 0.20))
    expected = runtime.reset(command).copy()
    expected_qpos = runtime.data.qpos.copy()
    expected_qvel = runtime.data.qvel.copy()
    expected_time = float(runtime.data.time)
    for _ in range(4):
        runtime.data.qvel[3:6] = (0.4, -0.6, 0.8)
        mujoco.mj_forward(runtime.model, runtime.data)
        moving = collect_kinematic_state(runtime.model, runtime.data, runtime.model_map, contract)
        assert np.linalg.norm(moving.angular_velocity_control) > 0.1
        advanced = runtime.step(np.full(6, 0.2), command)
        assert not np.array_equal(advanced.observation, expected)
        runtime.previous_action.fill(0.7)
        actual = runtime.reset(command)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(runtime.data.qpos, expected_qpos)
        np.testing.assert_array_equal(runtime.data.qvel, expected_qvel)
        assert float(runtime.data.time) == expected_time
        frames = actual.reshape(5, 25)
        np.testing.assert_array_equal(frames, np.repeat(frames[:1], 5, axis=0))
