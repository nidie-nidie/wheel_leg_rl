from __future__ import annotations

import torch

from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.schemas.frames import quat_rotate_inverse_wxyz

from debug.no_action_clip.experiment import raw_action_targets
from debug.no_action_clip.geometry import fk_diagnostic_metrics, verify_actual_geometry
from debug.no_action_clip.rewards import SplitStraightRewardWeights, wheel_differential_reward


class RawActionEnv(WheelLegFlatEnv):
    def _virtual_leg_state(self, root_quaternion, joint_position, *, enforce_guards=True):
        virtual_leg = super()._virtual_leg_state(root_quaternion, joint_position, enforce_guards=False)
        if enforce_guards:
            hip = self._world_anchors(self._virtual_leg_hip_body_ids, self._virtual_leg_hip_local_anchors)
            wheel = self._world_anchors(self._virtual_leg_wheel_body_ids, self._virtual_leg_wheel_local_anchors)
            orientation = root_quaternion.unsqueeze(1).expand(-1, 2, -1)
            vector_body = quat_rotate_inverse_wxyz(orientation, wheel - hip)
            verify_actual_geometry(virtual_leg, vector_body)
        return virtual_leg

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_previous_action.copy_(self._previous_action)
        self._previous_action.copy_(self._canonical_action)
        raw, leg_targets, wheel_targets = raw_action_targets(actions, self._q_reference, self.cfg.control)
        self._canonical_action.copy_(raw)
        self.actions = self._canonical_action
        self._leg_position_targets.copy_(leg_targets)
        self._wheel_velocity_targets.copy_(wheel_targets)
        if self._leg_reversal_reward is not None:
            self._leg_reversal_reward.step(self._leg_position_targets)

    def _get_rewards(self) -> torch.Tensor:
        differential = None
        if isinstance(self.cfg.reward_weights, SplitStraightRewardWeights):
            # 父环境最后会推进命令；本步差速成本必须提前计算。
            state = self._current_state()
            differential = wheel_differential_reward(
                state.last_applied_action, state.command, self.cfg.reward_weights,
                control_dt=self.step_dt,
            )
        reward = super()._get_rewards()
        log = self.extras["log"]
        if differential is not None:
            weighted, cost, enabled = differential
            reward = reward + weighted
            if not torch.isfinite(reward).all():
                raise FloatingPointError("加入差速项后的总奖励包含 NaN 或 Inf")
            self._reward_sums["wheel_differential"] += weighted
            log["Reward/wheel_differential"] = weighted.mean()
            log["Debug/wheel_differential_active_fraction"] = enabled.float().mean()
            log["Debug/wheel_differential_cost_mean"] = cost.mean()
        log.pop("Action/saturation_fraction")
        log["Debug/action_outside_former_unit_interval_fraction"] = (
            self._canonical_action.abs() > 1.0).float().mean()
        log["Debug/raw_action_abs_max"] = self._canonical_action.abs().max()
        log["Debug/leg_target_at_bound_fraction"] = (
            self._leg_position_targets.abs() >= 1.0 - 1.0e-6).float().mean()
        log.update(fk_diagnostic_metrics({
            "length_true": self._state.virtual_leg_length_true,
            "phi0_true": self._state.virtual_leg_phi0_true,
            "length_fk": self._state.virtual_leg_length_fk,
            "phi0_fk": self._state.virtual_leg_phi0_fk,
            "fk_valid": self._state.virtual_leg_fk_valid,
            "wheel_error": self._state.virtual_leg_wheel_error,
        }))
        return reward
