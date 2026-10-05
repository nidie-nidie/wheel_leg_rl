from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from wheelleg_mujoco.contract import AdapterContract
from wheelleg_mujoco.control import MixedActionController
from wheelleg_mujoco.metrics import collect_metrics
from wheelleg_mujoco.model_map import CANONICAL_JOINT_ORDER, build_model_map
from wheelleg_mujoco.observation import build_actor_observation, collect_kinematic_state
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        velocity_limits=np.full(6, 45.0),
        passive_velocity_limit=80.0,
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
    assert np.all(torque <= 0.0)
    assert np.all(controller.last_velocity_limit_mask[:4])
    assert not np.any(controller.last_velocity_limit_mask[4:])


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
