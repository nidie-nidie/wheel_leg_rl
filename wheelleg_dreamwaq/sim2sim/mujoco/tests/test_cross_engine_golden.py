from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT_PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_PROJECT / "source" / "wheelleg_dreamwaq"))

from wheelleg_dreamwaq.schemas.normalization import NormalizationV2  # noqa: E402
from wheelleg_dreamwaq.schemas.observation import assemble_actor_obs  # noqa: E402
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import (  # noqa: E402
    ControlLimits,
    compute_action_targets,
)
from wheelleg_mujoco.contract import AdapterContract  # noqa: E402
from wheelleg_mujoco.control import MixedActionController  # noqa: E402
from wheelleg_mujoco.model_map import build_model_map  # noqa: E402
from wheelleg_mujoco.observation import KinematicState, build_actor_observation  # noqa: E402

import mujoco  # noqa: E402


MODEL_PATH = ROOT_PROJECT / "sim2sim" / "mujoco" / "models" / "wheel_leg_urdf4_v1.xml"
Q_NOMINAL = np.array((-0.33367134, 0.33367134, -0.33367134, 0.33367134))


def _contract() -> AdapterContract:
    normalization = NormalizationV2()
    return AdapterContract(
        q_nominal=Q_NOMINAL.copy(),
        action_clip=1.0,
        leg_action_scale=0.35,
        wheel_action_scale=25.0,
        leg_target_lower=np.full(4, -1.0),
        leg_target_upper=np.full(4, 1.0),
        wheel_signs=np.array((1.0, -1.0)),
        r_control_from_mujoco=np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        normalization={name: float(value) for name, value in vars(normalization).items()},
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


def test_actor_observation_matches_training_schema_for_same_control_frame_state() -> None:
    contract = _contract()
    normalization = NormalizationV2()
    state = KinematicState(
        angular_velocity_control=np.array((24.0, -24.0, 1.25)),
        projected_gravity_control=np.array((0.25, -0.5, -0.9)),
        joint_position_control=np.array((-0.10, 0.05, -0.90, 0.95, 2.0, -3.0)),
        joint_velocity_control=np.array((10.0, -20.0, 120.0, -120.0, 5.0, -5.0)),
        com_linear_velocity_control=np.array((0.2, -0.1, 0.0)),
        base_height_m=0.22,
    )
    command = np.array((1.2, -0.75, 0.23))
    previous_action = np.array((1.2, -1.2, 0.4, -0.4, 0.8, -0.8))
    actual = build_actor_observation(state, command, previous_action, contract)

    expected = assemble_actor_obs(
        angular_velocity=normalization.normalize_angular_velocity(
            torch.tensor(state.angular_velocity_control, dtype=torch.float64).unsqueeze(0)
        ),
        projected_gravity=normalization.normalize_projected_gravity(
            torch.tensor(state.projected_gravity_control, dtype=torch.float64).unsqueeze(0)
        ),
        command=normalization.normalize_command(torch.tensor(command, dtype=torch.float64).unsqueeze(0)),
        leg_position_error=normalization.normalize_leg_position_error(
            torch.tensor(state.joint_position_control[:4] - Q_NOMINAL, dtype=torch.float64).unsqueeze(0)
        ),
        joint_velocity=normalization.normalize_joint_velocity(
            torch.tensor(state.joint_velocity_control, dtype=torch.float64).unsqueeze(0)
        ),
        previous_action=normalization.normalize_previous_action(
            torch.tensor(previous_action, dtype=torch.float64).unsqueeze(0)
        ),
    ).squeeze(0)
    np.testing.assert_allclose(actual, expected.numpy().astype(np.float32), rtol=0.0, atol=1.0e-7)


def test_action_targets_match_training_contract_before_mujoco_wheel_sign_mapping() -> None:
    contract = _contract()
    action = np.array((2.0, -2.0, 0.25, -0.25, 0.4, -0.6))
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    controller = MixedActionController(build_model_map(model), contract)
    actual = controller.prepare(action)

    clipped, leg_targets, wheel_targets_usd = compute_action_targets(
        torch.tensor(action, dtype=torch.float64).unsqueeze(0),
        torch.tensor(Q_NOMINAL, dtype=torch.float64),
        ControlLimits(),
    )
    np.testing.assert_allclose(actual.clipped_action, clipped.squeeze(0).numpy(), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        actual.leg_position_target_mujoco,
        leg_targets.squeeze(0).numpy(),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        actual.wheel_velocity_target_mujoco,
        wheel_targets_usd.squeeze(0).numpy(),
        rtol=0.0,
        atol=1.0e-12,
    )
