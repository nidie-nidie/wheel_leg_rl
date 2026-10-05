from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .contract import AdapterContract
from .model_map import ModelMap


@dataclass(frozen=True)
class ActionTargets:
    clipped_action: np.ndarray
    leg_position_target_mujoco: np.ndarray
    wheel_velocity_target_mujoco: np.ndarray


class MixedActionController:
    def __init__(self, model_map: ModelMap, contract: AdapterContract) -> None:
        self.model_map = model_map
        self.contract = contract
        self.leg_kp = contract.leg_kp
        self.leg_kd = contract.leg_kd
        self.wheel_kd = contract.wheel_kd
        self.effort_limits = contract.effort_limits
        self.velocity_limits = contract.velocity_limits
        self.last_velocity_limit_mask = np.zeros(6, dtype=bool)

    def prepare(self, action: np.ndarray) -> ActionTargets:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,):
            raise ValueError(f"ActionV1 must have shape (6,), got {action.shape}")
        clipped = np.clip(action, -self.contract.action_clip, self.contract.action_clip)
        leg_target = np.clip(
            self.contract.q_nominal + self.contract.leg_action_scale * clipped[:4],
            self.contract.leg_target_lower,
            self.contract.leg_target_upper,
        )
        wheel_target_control = self.contract.wheel_action_scale * clipped[4:6]
        wheel_target_mujoco = wheel_target_control * self.contract.wheel_signs
        return ActionTargets(clipped, leg_target, wheel_target_mujoco)

    def compute_torque(self, data: mujoco.MjData, targets: ActionTargets) -> np.ndarray:
        position = self.model_map.joint_position_mujoco(data)
        velocity = self.model_map.joint_velocity_mujoco(data)
        torque = np.empty(6, dtype=np.float64)
        torque[:4] = self.leg_kp * (targets.leg_position_target_mujoco - position[:4]) - self.leg_kd * velocity[:4]
        torque[4:6] = self.wheel_kd * (targets.wheel_velocity_target_mujoco - velocity[4:6])
        torque = np.clip(torque, -self.effort_limits, self.effort_limits)
        self.last_velocity_limit_mask = (np.abs(velocity) >= self.velocity_limits) & (torque * velocity > 0.0)
        torque[self.last_velocity_limit_mask] = 0.0
        return torque

    def apply_torque(self, data: mujoco.MjData, torque: np.ndarray) -> None:
        torque = np.asarray(torque, dtype=np.float64)
        if torque.shape != (6,):
            raise ValueError(f"Canonical torque must have shape (6,), got {torque.shape}")
        data.ctrl[:] = 0.0
        data.ctrl[self.model_map.actuator_ids] = torque
