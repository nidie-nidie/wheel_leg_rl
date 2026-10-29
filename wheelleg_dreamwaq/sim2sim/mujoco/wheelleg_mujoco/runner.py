from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from wheelleg_dreamwaq.schemas.heading_control import HeadingHold, HeadingHoldConfig, yaw_from_rotation

from .contract import AdapterContract
from .control import MixedActionController
from .metrics import collect_metrics
from .model_map import build_model_map
from .model_semantics import compiled_model_semantics, stable_hash
from .observation import build_actor_observation, collect_kinematic_state, _normalize_command


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
        if contract.policy_input_dimension not in (25, 125):
            raise ValueError("Policy input dimension must be 25 for PPO or 125 for DreamWaQ")
        if contract.policy_input_dimension == 125 and (
            contract.history_length != 5 or contract.history_layout != "frame_major"
        ):
            raise ValueError("DreamWaQ MuJoCo history contract is invalid")
        self._history: np.ndarray | None = None
        self.last_reset_metrics: dict[str, float] | None = None
        self.heading = None
        if contract.heading_hold is not None:
            fields = {key: contract.heading_hold[key] for key in ("enabled", "kp", "kd", "max_yaw_rate", "control_dt_s")}
            self.heading = HeadingHold(HeadingHoldConfig(**fields), 1, dtype=torch.float64)
        self.source_command: np.ndarray | None = None
        self.effective_command: np.ndarray | None = None
        self._prepared_time: float | None = None

    def _heading_yaw(self) -> torch.Tensor:
        rotation = np.array(self.data.xmat[self.model_map.base_body_id], copy=True).reshape(1, 3, 3)
        return yaw_from_rotation(torch.from_numpy(rotation))

    def set_heading_enabled(self, enabled: bool) -> None:
        if self.heading is None:
            raise ValueError("当前 profile 没有航向保持配置")
        self.heading.toggle(enabled, self._heading_yaw())
        self._prepared_time = None

    def prepare_policy_observation(self, source_command: np.ndarray) -> np.ndarray:
        if self.heading is None or self._history is None:
            raise RuntimeError("当前 runtime 不支持航向策略输入准备")
        source = np.asarray(source_command, dtype=np.float64)
        if source.shape != (3,) or not np.isfinite(source).all():
            raise ValueError("源命令必须为有限三维向量")
        if self._prepared_time != self.data.time or not np.array_equal(source, self.source_command):
            state = collect_kinematic_state(self.model, self.data, self.model_map, self.contract)
            effective = self.heading.update(self._heading_yaw(),
                torch.tensor([state.angular_velocity_control[2]], dtype=torch.float64),
                torch.from_numpy(source.copy()).view(1, 3))
            self.source_command = source.copy()
            self.effective_command = effective.numpy()[0].copy()
            self._history[-1, 6:9] = _normalize_command(self.effective_command, self.contract)
            self._prepared_time = float(self.data.time)
        return self._history.reshape(-1).astype(np.float32, copy=True)

    def _reset_policy_observation(self, current: np.ndarray) -> np.ndarray:
        if self.contract.policy_input_dimension == 25:
            self._history = None
            return current
        self._history = np.repeat(current[None, :], self.contract.history_length, axis=0)
        return self._history.reshape(-1).astype(np.float32, copy=True)

    def _append_policy_observation(self, current: np.ndarray) -> np.ndarray:
        if self.contract.policy_input_dimension == 25:
            return current
        if self._history is None or self._history.shape != (self.contract.history_length, 25):
            raise RuntimeError("DreamWaQ MuJoCo history was not initialized by reset")
        self._history[:-1] = self._history[1:]
        self._history[-1] = current
        return self._history.reshape(-1).astype(np.float32, copy=True)

    def reset(self, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (3,) or not np.isfinite(command).all():
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
        self._prepared_time = None
        if self.heading is not None:
            self.heading.reset(self._heading_yaw())
            self.source_command = command.copy()
            self.effective_command = command.copy()
        current = build_actor_observation(state, command, self.previous_action, self.contract)
        return self._reset_policy_observation(current)

    def step(self, action: np.ndarray, command: np.ndarray) -> StepResult:
        command = np.asarray(command, dtype=np.float64)
        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError("动作必须为有限六维向量")
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("命令必须为有限三维向量")
        if self.heading is not None:
            if self._prepared_time != self.data.time or not np.array_equal(command, self.source_command):
                raise RuntimeError("必须先为本控制步准备源命令对应的策略观测")
            command = self.effective_command.copy()
        targets = self.controller.prepare(action)
        if not all(np.isfinite(values).all() for values in
                   (targets.clipped_action, targets.leg_position_target_mujoco, targets.wheel_velocity_target_mujoco)):
            raise FloatingPointError("电机目标包含非有限数值")
        torque = np.zeros(6, dtype=np.float64)
        effort_saturation_count = 0
        for _ in range(self.contract.physics_steps_per_action):
            torque = self.controller.compute_torque(self.data, targets)
            effort_saturation_count += int(np.isclose(np.abs(torque), self.controller.effort_limits).sum())
            self.controller.apply_torque(self.data, torque)
            mujoco.mj_step(self.model, self.data)
        self.previous_action = targets.clipped_action.copy()
        state = collect_kinematic_state(self.model, self.data, self.model_map, self.contract)
        current_observation = build_actor_observation(state, command, self.previous_action, self.contract)
        observation = self._append_policy_observation(current_observation)
        metrics = collect_metrics(self.model, self.data, self.model_map, self.contract, state)
        metrics.update(
            {
                "action_saturation_fraction": (0.0 if np.isinf(self.contract.action_clip)
                                               else float(np.mean(np.abs(np.asarray(action)) >= 1.0))),
                "effort_saturation_fraction": effort_saturation_count
                / (self.contract.physics_steps_per_action * 6),
            }
        )
        if np.isinf(self.contract.action_clip):
            metrics["action_outside_former_unit_interval_fraction"] = float(np.mean(np.abs(raw) > 1))
        if self.heading is not None:
            metrics.update({
                "source_vx_mps": float(self.source_command[0]),
                "source_yaw_rate_rad_s": float(self.source_command[1]),
                "effective_vx_mps": float(command[0]), "effective_yaw_rate_rad_s": float(command[1]),
                "heading_reference_rad": float(self.heading.reference[0]),
                "heading_error_rad": float(self.heading.error[0]),
                "heading_command_saturated": float(self.heading.saturated[0]),
                "wheel_differential_enabled": float(command[0] != 0 and command[1] == 0),
                "heading_enabled": float(self.heading.enabled),
            })
        return StepResult(
            observation=observation,
            clipped_action=self.previous_action.copy(),
            applied_torque=torque.copy(),
            metrics=metrics,
            physics_steps=self.contract.physics_steps_per_action,
        )
