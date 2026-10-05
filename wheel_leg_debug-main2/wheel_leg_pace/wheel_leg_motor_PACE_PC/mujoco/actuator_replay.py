from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Mapping, Sequence, Union

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fit.model_manifest import load_model_manifest, verify_against_firmware_manifest
from pace_raw.manifest import load_firmware_manifest


class _DelayLine:
    def __init__(self, delay_steps: int):
        if delay_steps < 0:
            raise ValueError("delay_steps cannot be negative")
        self.delay_steps = int(delay_steps)
        self._values = collections.deque(
            [0.0] * (self.delay_steps + 1), maxlen=self.delay_steps + 1
        )

    def reset(self) -> None:
        self._values.clear()
        self._values.extend([0.0] * (self.delay_steps + 1))

    def push(self, value: float) -> float:
        self._values.append(float(value))
        return float(self._values[0])


def _friction_torque(parameters: Mapping[str, float], velocity: float) -> float:
    transition = float(parameters.get("friction_transition_velocity_rad_s", 0.05))
    static = float(parameters["static_friction"])
    dynamic = float(parameters["dynamic_friction"])
    speed_ratio = abs(float(velocity)) / max(transition, 1.0e-6)
    magnitude = dynamic + (static - dynamic) * np.exp(-(speed_ratio * speed_ratio))
    return float(magnitude * np.tanh(float(velocity) / max(transition, 1.0e-6)))


class ActuatorReplay:
    """Stateful six-channel actuator model in the firmware canonical order."""

    def __init__(self, value: Union[pathlib.Path, str, Mapping[str, object]]):
        self.manifest = load_model_manifest(value)
        verify_against_firmware_manifest(self.manifest, load_firmware_manifest())
        self.motors = self.manifest["motors"]
        self.dm_delay = [
            _DelayLine(int(self.motors[index]["model"]["delay_steps"]))
            for index in range(4)
        ]
        self.wheel_delay = []
        self.wheel_mode = [None, None]
        for index in range(4, 6):
            model = self.motors[index]["model"]
            maximum = max(
                int(model["torque_mode"]["delay_steps"]),
                int(model["velocity_mode"]["delay_steps"]),
            )
            self.wheel_delay.append(_DelayLine(maximum))

    @property
    def canonical_order(self) -> tuple:
        return tuple(motor["name"] for motor in self.motors)

    def reset(self) -> None:
        for line in self.dm_delay + self.wheel_delay:
            line.reset()
        self.wheel_mode = [None, None]

    def compute_dm_effort(
        self,
        target_position: Sequence[float],
        position: Sequence[float],
        velocity: Sequence[float],
        target_velocity: Sequence[float] = None,
        torque_feedforward: Sequence[float] = None,
    ) -> np.ndarray:
        target_position = np.asarray(target_position, dtype=np.float64)
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        target_velocity = (
            np.zeros(4, dtype=np.float64)
            if target_velocity is None
            else np.asarray(target_velocity, dtype=np.float64)
        )
        torque_feedforward = (
            np.zeros(4, dtype=np.float64)
            if torque_feedforward is None
            else np.asarray(torque_feedforward, dtype=np.float64)
        )
        for name, values in (
            ("target_position", target_position),
            ("position", position),
            ("velocity", velocity),
            ("target_velocity", target_velocity),
            ("torque_feedforward", torque_feedforward),
        ):
            if values.shape != (4,):
                raise ValueError(f"{name} must contain four DM values")
        output = np.zeros(4, dtype=np.float64)
        for index in range(4):
            fitted = self.motors[index]["model"]
            controller = fitted["controller"]
            encoder_position = position[index] - float(fitted["encoder_bias_rad"])
            effort = (
                float(controller["kp"])
                * (target_position[index] - encoder_position)
                + float(controller["kd"])
                * (target_velocity[index] - velocity[index])
                + torque_feedforward[index]
            )
            delayed_effort = self.dm_delay[index].push(effort)
            limit = float(fitted["effort_limit_nm"])
            output[index] = np.clip(delayed_effort, -limit, limit)
        return output

    def compute_wheel_effort(
        self,
        command: Sequence[float],
        velocity: Sequence[float],
        mode: str,
    ) -> np.ndarray:
        if mode not in ("torque", "velocity"):
            raise ValueError("wheel mode must be torque or velocity")
        command = np.asarray(command, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        if command.shape != (2,) or velocity.shape != (2,):
            raise ValueError("wheel command and velocity must contain two values")
        output = np.zeros(2, dtype=np.float64)
        for local_index, motor_index in enumerate((4, 5)):
            fitted = self.motors[motor_index]["model"]
            if self.wheel_mode[local_index] != mode:
                selected_delay = int(fitted[mode + "_mode"]["delay_steps"])
                self.wheel_delay[local_index] = _DelayLine(selected_delay)
                self.wheel_mode[local_index] = mode
            if mode == "torque":
                effort = command[local_index]
            else:
                torque_model = fitted["torque_mode"]
                velocity_model = fitted["velocity_mode"]
                acceleration = (
                    float(velocity_model["command_gain"]) * command[local_index]
                    - velocity[local_index]
                ) / float(velocity_model["time_constant_s"])
                effort = (
                    float(torque_model["armature"]) * acceleration
                    + float(torque_model["viscous_friction"]) * velocity[local_index]
                    + _friction_torque(torque_model, velocity[local_index])
                )
            delayed_effort = self.wheel_delay[local_index].push(effort)
            limit = float(fitted["effort_limit_nm"])
            output[local_index] = np.clip(delayed_effort, -limit, limit)
        return output

    def compute_effort(
        self,
        dm_target_position: Sequence[float],
        joint_position: Sequence[float],
        joint_velocity: Sequence[float],
        wheel_command: Sequence[float],
        wheel_mode: str = "torque",
    ) -> np.ndarray:
        position = np.asarray(joint_position, dtype=np.float64)
        velocity = np.asarray(joint_velocity, dtype=np.float64)
        if position.shape != (6,) or velocity.shape != (6,):
            raise ValueError("joint state must follow the six-channel canonical order")
        return np.concatenate(
            [
                self.compute_dm_effort(
                    dm_target_position, position[:4], velocity[:4]
                ),
                self.compute_wheel_effort(
                    wheel_command, velocity[4:], wheel_mode
                ),
            ]
        )

    def joint_parameter_patch(self) -> dict:
        armature = []
        damping = []
        frictionloss = []
        for index, motor in enumerate(self.motors):
            fitted = motor["model"] if index < 4 else motor["model"]["torque_mode"]
            armature.append(float(fitted["armature"]))
            damping.append(float(fitted["viscous_friction"]))
            frictionloss.append(float(fitted["dynamic_friction"]))
        return {
            "joint_order": [motor["mujoco_joint"] for motor in self.motors],
            "armature": armature,
            "damping": damping,
            "frictionloss": frictionloss,
        }

    def apply_to_mujoco_model(self, mj_model, dof_indices: Sequence[int]) -> None:
        indices = np.asarray(dof_indices, dtype=np.int64)
        if indices.shape != (6,):
            raise ValueError("dof_indices must follow the six-channel canonical order")
        patch = self.joint_parameter_patch()
        mj_model.dof_armature[indices] = patch["armature"]
        mj_model.dof_damping[indices] = patch["damping"]
        mj_model.dof_frictionloss[indices] = patch["frictionloss"]

    @staticmethod
    def write_ctrl(mj_data, actuator_indices: Sequence[int], effort: Sequence[float]) -> None:
        indices = np.asarray(actuator_indices, dtype=np.int64)
        values = np.asarray(effort, dtype=np.float64)
        if indices.shape != (6,) or values.shape != (6,):
            raise ValueError("actuator indices and effort must contain six canonical values")
        mj_data.ctrl[indices] = values


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the MuJoCo fitted joint parameter patch")
    parser.add_argument("manifest", type=pathlib.Path)
    args = parser.parse_args()
    replay = ActuatorReplay(args.manifest)
    print(json.dumps(replay.joint_parameter_patch(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
