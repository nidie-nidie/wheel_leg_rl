from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .contract import AdapterContract
from .control import MixedActionController
from .metrics import collect_metrics
from .model_map import build_model_map
from .model_semantics import compiled_model_semantics, stable_hash
from .observation import build_actor_observation, collect_kinematic_state


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    clipped_action: np.ndarray
    applied_torque: np.ndarray
    metrics: dict[str, float]
    physics_steps: int


class WheelLegMujocoRuntime:
    def __init__(self, model_path: str | Path, contract: AdapterContract, model_manifest: dict | None = None) -> None:
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.contract = contract
        if abs(float(self.model.opt.timestep) - contract.physics_dt_s) > 1.0e-15:
            raise ValueError(f"MuJoCo timestep differs from policy contract: {self.model.opt.timestep}")
        if model_manifest is not None:
            actual_semantics = compiled_model_semantics(self.model)
            if stable_hash(actual_semantics) != model_manifest["dynamics_semantics_hash"]:
                raise ValueError("Compiled MuJoCo dynamics differ from the frozen model manifest")
        self.data = mujoco.MjData(self.model)
        self.model_map = build_model_map(self.model)
        self.controller = MixedActionController(self.model_map, contract)
        self.previous_action = np.zeros(6, dtype=np.float64)
        self.last_reset_metrics: dict[str, float] | None = None

    def reset(self, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (3,):
            raise ValueError(f"CommandV1 must have shape (3,), got {command.shape}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model_map.reset_keyframe_id)
        self.data.ctrl[:] = 0.0
        self.previous_action.fill(0.0)
        mujoco.mj_forward(self.model, self.data)
        state = collect_kinematic_state(self.model, self.data, self.model_map, self.contract)
        self.last_reset_metrics = collect_metrics(
            self.model,
            self.data,
            self.model_map,
            self.contract,
            state,
        )
        return build_actor_observation(state, command, self.previous_action, self.contract)

    def step(self, action: np.ndarray, command: np.ndarray) -> StepResult:
        command = np.asarray(command, dtype=np.float64)
        targets = self.controller.prepare(action)
        torque = np.zeros(6, dtype=np.float64)
        effort_saturation_count = 0
        velocity_limit_event_count = 0
        for _ in range(self.contract.physics_steps_per_action):
            torque = self.controller.compute_torque(self.data, targets)
            effort_saturation_count += int(np.isclose(np.abs(torque), self.controller.effort_limits).sum())
            velocity_limit_event_count += int(self.controller.last_velocity_limit_mask.sum())
            self.controller.apply_torque(self.data, torque)
            mujoco.mj_step(self.model, self.data)
        self.previous_action = targets.clipped_action.copy()
        state = collect_kinematic_state(self.model, self.data, self.model_map, self.contract)
        observation = build_actor_observation(state, command, self.previous_action, self.contract)
        metrics = collect_metrics(self.model, self.data, self.model_map, self.contract, state)
        metrics.update(
            {
                "action_saturation_fraction": float(np.mean(np.abs(np.asarray(action)) >= 1.0)),
                "effort_saturation_fraction": effort_saturation_count
                / (self.contract.physics_steps_per_action * 6),
                "velocity_limit_event_fraction": velocity_limit_event_count
                / (self.contract.physics_steps_per_action * 6),
            }
        )
        return StepResult(
            observation=observation,
            clipped_action=self.previous_action.copy(),
            applied_torque=torque.copy(),
            metrics=metrics,
            physics_steps=self.contract.physics_steps_per_action,
        )
