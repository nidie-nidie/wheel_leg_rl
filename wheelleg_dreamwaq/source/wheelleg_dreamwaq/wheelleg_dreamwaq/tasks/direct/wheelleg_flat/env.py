from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2
from wheelleg_dreamwaq.assets.asset_overrides import (
    assert_embedded_ground_absent,
    configure_wheel_only_collisions,
    is_wheel_collision_path,
)
from wheelleg_dreamwaq.schemas.action import CANONICAL_JOINT_ORDER, controlled_joint_feedback_usd_to_control
from wheelleg_dreamwaq.schemas.frames import (
    projected_gravity_from_quaternion,
    quat_rotate_inverse_wxyz,
    quat_rotate_wxyz,
    transform_usd_vector_to_control,
)
from wheelleg_dreamwaq.schemas.physics import validate_phase1_physics
from wheelleg_dreamwaq.kinematics.virtual_leg import (
    MAX_FK_LENGTH_ERROR_M,
    MAX_FK_PHI0_ERROR_RAD,
    MAX_FK_WHEEL_ERROR_M,
    MAX_LOOP_CLOSURE_ERROR_M,
    MIN_VIRTUAL_LEG_LENGTH_M,
    load_usd_anchor_audit,
    offset_leg_fk,
    offset_geometry_from_audit,
    virtual_leg_from_body_vector,
    wrap_to_pi,
)

from .commands import sample_commands
from .control import compute_action_targets
from .env_cfg import WheelLegFlatEnvCfg
from .observations import build_observations
from .rewards import compute_reward
from .state import WheelLegState
from .terminations import compute_dones


class WheelLegFlatEnv(DirectRLEnv):
    cfg: WheelLegFlatEnvCfg

    def __init__(self, cfg: WheelLegFlatEnvCfg, render_mode: str | None = None, **kwargs):
        articulation = cfg.robot_cfg.spawn.articulation_props
        validate_phase1_physics(
            sim_dt=cfg.sim.dt,
            decimation=cfg.decimation,
            solver_position_iterations=articulation.solver_position_iteration_count,
            solver_velocity_iterations=articulation.solver_velocity_iteration_count,
        )
        super().__init__(cfg, render_mode, **kwargs)

        joint_ids, joint_names = self.robot.find_joints(list(CANONICAL_JOINT_ORDER), preserve_order=True)
        if tuple(joint_names) != CANONICAL_JOINT_ORDER or len(set(joint_ids)) != len(CANONICAL_JOINT_ORDER):
            raise RuntimeError(f"Canonical controlled joints did not resolve uniquely: {joint_names} -> {joint_ids}")
        self._controlled_joint_ids = joint_ids
        self._leg_joint_ids = joint_ids[:4]
        self._wheel_joint_ids = joint_ids[4:]
        self._q_nominal = torch.tensor(self.cfg.q_nominal, dtype=torch.float32, device=self.device)

        self._canonical_action = torch.zeros((self.num_envs, 6), device=self.device)
        self._previous_action = torch.zeros_like(self._canonical_action)
        self._previous_previous_action = torch.zeros_like(self._canonical_action)
        self._leg_position_targets = self._q_nominal.repeat(self.num_envs, 1)
        self._wheel_velocity_targets = torch.zeros((self.num_envs, 2), device=self.device)
        self._commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_joint_velocity = controlled_joint_feedback_usd_to_control(
            self.robot.data.joint_vel[:, self._controlled_joint_ids]
        ).clone()
        self._state: WheelLegState | None = None
        self._reward_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in vars(self.cfg.reward_weights)
            if name not in {"tracking_sigma", "height_sigma", "soft_leg_limit"}
        }
        self._termination_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "invalid",
                "height_terminated",
                "tilt_terminated",
                "root_linear_terminated",
                "root_angular_terminated",
                "joint_velocity_terminated",
                "timeout",
            )
        }
        self._controlled_effort_limits = self.robot.data.joint_effort_limits[0, self._controlled_joint_ids].clone()
        self._resolve_virtual_leg_geometry()
        self._assert_runtime_asset_contract()

    def _body_id(self, name: str) -> int:
        body_ids, body_names = self.robot.find_bodies([name], preserve_order=True)
        if body_names != [name] or len(body_ids) != 1:
            raise RuntimeError(f"Virtual-leg body did not resolve uniquely: {name!r} -> {body_names}")
        return body_ids[0]

    def _resolve_virtual_leg_geometry(self) -> None:
        audit = load_usd_anchor_audit()
        if audit.get("asset_bundle_version") != ASSET_BUNDLE_V2.version:
            raise RuntimeError("Virtual-leg audit was not generated from AssetBundleV2")
        if audit.get("asset_bundle_hash") != ASSET_BUNDLE_V2.bundle_hash:
            raise RuntimeError("Virtual-leg audit asset hash does not match the selected AssetBundleV2")

        hip_body_ids: list[int] = []
        wheel_body_ids: list[int] = []
        hip_anchors: list[list[float]] = []
        wheel_anchors: list[list[float]] = []
        for side in ("left", "right"):
            leg = audit["legs"][side]
            hip_body_ids.append(self._body_id(leg["hip"]["body_name"]))
            wheel_body_ids.append(self._body_id(leg["wheel"]["body_name"]))
            hip_anchors.append(leg["hip"]["local_anchor_m"])
            wheel_anchors.append(leg["wheel"]["local_anchor_m"])
        self._virtual_leg_hip_body_ids = hip_body_ids
        self._virtual_leg_wheel_body_ids = wheel_body_ids
        self._virtual_leg_hip_local_anchors = torch.tensor(hip_anchors, device=self.device, dtype=torch.float32)
        self._virtual_leg_wheel_local_anchors = torch.tensor(wheel_anchors, device=self.device, dtype=torch.float32)
        self._virtual_leg_fk_geometries = (
            offset_geometry_from_audit(audit["legs"]["left"]),
            offset_geometry_from_audit(audit["legs"]["right"]),
        )

        loop_body_ids: list[int] = []
        loop_local_anchors: list[list[float]] = []
        for joint in audit["loop_joints"]:
            for side in joint["sides"]:
                loop_body_ids.append(self._body_id(side["body_name"]))
                loop_local_anchors.append(side["local_anchor_m"])
        self._loop_body_ids = loop_body_ids
        self._loop_local_anchors = torch.tensor(loop_local_anchors, device=self.device, dtype=torch.float32)

    def _world_anchors(self, body_ids: list[int], local_anchors: torch.Tensor) -> torch.Tensor:
        positions = self.robot.data.body_link_pos_w[:, body_ids]
        orientations = self.robot.data.body_link_quat_w[:, body_ids]
        local = local_anchors.unsqueeze(0).expand(self.num_envs, -1, -1)
        return positions + quat_rotate_wxyz(orientations, local)

    def _virtual_leg_state(self, root_quaternion: torch.Tensor, joint_position: torch.Tensor) -> dict[str, torch.Tensor]:
        hip_world = self._world_anchors(self._virtual_leg_hip_body_ids, self._virtual_leg_hip_local_anchors)
        wheel_world = self._world_anchors(self._virtual_leg_wheel_body_ids, self._virtual_leg_wheel_local_anchors)
        vector_world = wheel_world - hip_world
        root_orientation = root_quaternion.unsqueeze(1).expand(-1, 2, -1)
        vector_body = quat_rotate_inverse_wxyz(root_orientation, vector_world)
        true_length, true_phi0 = virtual_leg_from_body_vector(vector_body)

        left_fk = offset_leg_fk(joint_position[:, 0:2], self._virtual_leg_fk_geometries[0])
        right_fk = offset_leg_fk(joint_position[:, 2:4], self._virtual_leg_fk_geometries[1])
        fk_vector = torch.stack((left_fk.wheel_vector_body, right_fk.wheel_vector_body), dim=1)
        fk_length = torch.stack((left_fk.length, right_fk.length), dim=1)
        fk_phi0 = torch.stack((left_fk.phi0, right_fk.phi0), dim=1)
        fk_valid = torch.stack((left_fk.valid, right_fk.valid), dim=1)
        true_planar_vector = vector_body.clone()
        true_planar_vector[..., 0] = 0.0
        wheel_error = torch.linalg.vector_norm(fk_vector - true_planar_vector, dim=-1)

        loop_world = self._world_anchors(self._loop_body_ids, self._loop_local_anchors)
        loop_world = loop_world.reshape(self.num_envs, -1, 2, 3)
        loop_error = torch.linalg.vector_norm(loop_world[:, :, 0] - loop_world[:, :, 1], dim=-1)

        length_error = torch.abs(fk_length - true_length)
        phi0_error = torch.abs(wrap_to_pi(fk_phi0 - true_phi0))
        invalid = (
            ~torch.isfinite(vector_body).all(dim=-1)
            | (true_length <= MIN_VIRTUAL_LEG_LENGTH_M)
            | ~fk_valid
            | (wheel_error > MAX_FK_WHEEL_ERROR_M)
            | (length_error > MAX_FK_LENGTH_ERROR_M)
            | (phi0_error > MAX_FK_PHI0_ERROR_RAD)
        )
        if bool(invalid.any().item()):
            first = torch.nonzero(invalid, as_tuple=False)[0]
            env_index = int(first[0].item())
            side_index = int(first[1].item())
            diagnostic_names = ["jAB", "jAG", "jBE", "jEC", "jCF", "jGH"]
            diagnostic_ids, resolved_names = self.robot.find_joints(diagnostic_names, preserve_order=True)
            diagnostic_q = {
                name: float(self.robot.data.joint_pos[env_index, joint_id].item())
                for name, joint_id in zip(resolved_names, diagnostic_ids, strict=True)
            }
            raise RuntimeError(
                "Virtual-leg geometry guard failed: "
                f"wheel={float(wheel_error.max().item()):.6g} m, "
                f"length={float(length_error.max().item()):.6g} m, "
                f"phi0={float(torch.rad2deg(phi0_error.max()).item()):.6g} deg; "
                f"first env={env_index} side={side_index} "
                f"q={joint_position[env_index, side_index * 2 : side_index * 2 + 2].tolist()} "
                f"true_L0={float(true_length[env_index, side_index].item()):.9g} "
                f"fk_L0={float(fk_length[env_index, side_index].item()):.9g} "
                f"true_phi0={float(true_phi0[env_index, side_index].item()):.9g} "
                f"fk_phi0={float(fk_phi0[env_index, side_index].item()):.9g} "
                f"loop={loop_error[env_index].tolist()} right_q={diagnostic_q}"
            )
        if bool((loop_error > MAX_LOOP_CLOSURE_ERROR_M).any().item()):
            raise RuntimeError(
                f"Loop-closure anchor error exceeded 5 mm: {float(loop_error.max().item()):.6g} m"
            )
        return {
            "length_true": true_length,
            "phi0_true": true_phi0,
            "length_fk": fk_length,
            "phi0_fk": fk_phi0,
            "fk_valid": fk_valid,
            "wheel_error": wheel_error,
            "loop_error": loop_error,
        }

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        source_robot_path = "/World/envs/env_0/Robot"
        assert_embedded_ground_absent(self.sim.get_initial_stage(), source_robot_path)
        configure_wheel_only_collisions(self.sim.get_initial_stage(), source_robot_path)

        self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path,
            self.cfg.ground.spawn,
            translation=self.cfg.ground.translation,
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.ground.prim_path])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _assert_runtime_asset_contract(self) -> None:
        from pxr import Usd, UsdPhysics

        stage = self.sim.stage
        physics_scenes = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        if physics_scenes != ["/physicsScene"]:
            raise RuntimeError(f"Expected exactly one global PhysicsScene, found {physics_scenes}")
        indices = {0, self.num_envs - 1}
        for index in indices:
            robot_prim_path = f"/World/envs/env_{index}/Robot"
            assert_embedded_ground_absent(stage, robot_prim_path)
            robot_prim = stage.GetPrimAtPath(robot_prim_path)
            enabled_collision_paths = [
                str(candidate.GetPath())
                for candidate in Usd.PrimRange(robot_prim)
                if candidate.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(candidate).GetCollisionEnabledAttr().Get() is not False
            ]
            if len(enabled_collision_paths) != 2 or not all(
                is_wheel_collision_path(candidate, robot_prim_path) for candidate in enabled_collision_paths
            ):
                raise RuntimeError(
                    f"Expected wheel-only robot collisions in {robot_prim_path}, found {enabled_collision_paths}"
                )
        if ASSET_BUNDLE_V2.articulation_root.rsplit("/", 1)[-1] != "base_link":
            raise RuntimeError("Unexpected articulation root contract")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_previous_action.copy_(self._previous_action)
        self._previous_action.copy_(self._canonical_action)
        clipped, leg_targets, wheel_targets = compute_action_targets(actions, self._q_nominal, self.cfg.control)
        self._canonical_action.copy_(clipped)
        self.actions = self._canonical_action
        self._leg_position_targets.copy_(leg_targets)
        self._wheel_velocity_targets.copy_(wheel_targets)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._leg_position_targets, joint_ids=self._leg_joint_ids)
        self.robot.set_joint_velocity_target(self._wheel_velocity_targets, joint_ids=self._wheel_joint_ids)

    def _capture_state(self) -> WheelLegState:
        joint_position = controlled_joint_feedback_usd_to_control(
            self.robot.data.joint_pos[:, self._controlled_joint_ids]
        )
        joint_velocity = controlled_joint_feedback_usd_to_control(
            self.robot.data.joint_vel[:, self._controlled_joint_ids]
        )
        joint_acceleration = (joint_velocity - self._previous_joint_velocity) / self.step_dt
        self._previous_joint_velocity.copy_(joint_velocity)
        root_quaternion = self.robot.data.root_link_quat_w
        projected_gravity_usd = projected_gravity_from_quaternion(root_quaternion)
        virtual_leg = self._virtual_leg_state(root_quaternion, joint_position)
        state = WheelLegState(
            root_link_pos_w=self.robot.data.root_link_pos_w,
            root_link_quat_w=root_quaternion,
            root_com_pos_w=self.robot.data.root_com_pos_w,
            root_com_linear_velocity=transform_usd_vector_to_control(self.robot.data.root_com_lin_vel_b),
            root_angular_velocity=transform_usd_vector_to_control(self.robot.data.root_link_ang_vel_b),
            projected_gravity=transform_usd_vector_to_control(projected_gravity_usd),
            base_height=(self.robot.data.root_com_pos_w[:, 2] - self.scene.env_origins[:, 2]).unsqueeze(-1),
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            joint_acceleration=joint_acceleration,
            applied_torque=controlled_joint_feedback_usd_to_control(
                self.robot.data.applied_torque[:, self._controlled_joint_ids]
            ),
            last_applied_action=self._canonical_action,
            previous_applied_action=self._previous_action,
            previous_previous_applied_action=self._previous_previous_action,
            command=self._commands,
            virtual_leg_length_true=virtual_leg["length_true"],
            virtual_leg_phi0_true=virtual_leg["phi0_true"],
            virtual_leg_length_fk=virtual_leg["length_fk"],
            virtual_leg_phi0_fk=virtual_leg["phi0_fk"],
            virtual_leg_fk_valid=virtual_leg["fk_valid"],
            virtual_leg_wheel_error=virtual_leg["wheel_error"],
            loop_closure_position_error=virtual_leg["loop_error"],
        )
        state.assert_finite()
        self._state = state
        return state

    def _current_state(self) -> WheelLegState:
        return self._state if self._state is not None else self._capture_state()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        actor_obs, critic_obs = build_observations(
            self._current_state(), q_nominal=self._q_nominal, normalization=self.cfg.normalization
        )
        return {"policy": actor_obs, "critic": critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        reward, weighted_terms = compute_reward(
            self._current_state(),
            self.cfg.reward_weights,
            control_dt=self.step_dt,
            terminated=self.reset_terminated,
        )
        for name, value in weighted_terms.items():
            self._reward_sums[name] += value
        log = {f"Reward/{name}": value.mean() for name, value in weighted_terms.items()}
        log.update(
            {
                "State/base_height": self._state.base_height.mean(),
                "State/vx": self._state.root_com_linear_velocity[:, 0].mean(),
                "State/yaw_rate": self._state.root_angular_velocity[:, 2].mean(),
                "Command/vx": self._commands[:, 0].mean(),
                "Command/yaw_rate": self._commands[:, 1].mean(),
                "Command/base_height": self._commands[:, 2].mean(),
                "Action/saturation_fraction": (torch.abs(self._canonical_action) > 0.999).float().mean(),
                "Actuator/effort_saturation_fraction": (
                    torch.abs(self._state.applied_torque)
                    >= 0.99 * self._controlled_effort_limits.unsqueeze(0)
                ).float().mean(),
                "Joint/leg_soft_limit_fraction": (
                    torch.abs(self._state.joint_position[:, :4]) >= self.cfg.reward_weights.soft_leg_limit
                ).float().mean(),
                "Tracking/vx_abs_error": torch.abs(
                    self._commands[:, 0] - self._state.root_com_linear_velocity[:, 0]
                ).mean(),
                "Tracking/yaw_rate_abs_error": torch.abs(
                    self._commands[:, 1] - self._state.root_angular_velocity[:, 2]
                ).mean(),
                "Tracking/base_height_abs_error": torch.abs(
                    self._commands[:, 2] - self._state.base_height[:, 0]
                ).mean(),
                "Metric/phi0_delta_abs_rad": torch.abs(
                    wrap_to_pi(self._state.virtual_leg_phi0_true[:, 0] - self._state.virtual_leg_phi0_true[:, 1])
                ).mean(),
                "Metric/phi0_left_rad": self._state.virtual_leg_phi0_true[:, 0].mean(),
                "Metric/phi0_right_rad": self._state.virtual_leg_phi0_true[:, 1].mean(),
                "Metric/phi0_fk_vs_true_max_abs_rad": torch.abs(
                    wrap_to_pi(self._state.virtual_leg_phi0_fk - self._state.virtual_leg_phi0_true)
                ).max(),
                "Metric/virtual_leg_length_left_m": self._state.virtual_leg_length_true[:, 0].mean(),
                "Metric/virtual_leg_length_right_m": self._state.virtual_leg_length_true[:, 1].mean(),
                "Metric/virtual_leg_fk_wheel_error_max_m": self._state.virtual_leg_wheel_error.max(),
                "Metric/loop_closure_error_max_m": self._state.loop_closure_position_error.max(),
            }
        )
        for name in self._termination_sums:
            log[f"Termination/{name}"] = self._termination_sums[name].mean()
        self.extras["log"] = log
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._capture_state()
        terminated, truncated, diagnostics = compute_dones(
            self._state,
            self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            limits=self.cfg.termination,
        )
        self.extras["termination_diagnostics"] = diagnostics
        if hasattr(self, "_termination_sums"):
            for name in self._termination_sums:
                values = truncated if name == "timeout" else diagnostics[name]
                self._termination_sums[name] += values.float()
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        previous_joint_acceleration = None
        if hasattr(self, "_state") and self._state is not None:
            previous_joint_acceleration = self._state.joint_acceleration.clone()

        if hasattr(self, "_reward_sums"):
            episode_log = self.extras.setdefault("log", {})
            for name, values in self._reward_sums.items():
                episode_log[f"Episode/{name}"] = values[env_ids].mean()
                values[env_ids] = 0.0
            for name, values in self._termination_sums.items():
                episode_log[f"EpisodeTermination/{name}"] = values[env_ids].mean()
                values[env_ids] = 0.0

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        joint_position = self.robot.data.default_joint_pos[env_ids].clone()
        joint_velocity = self.robot.data.default_joint_vel[env_ids].clone()
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_position, joint_velocity, None, env_ids)

        if hasattr(self, "_canonical_action"):
            self._canonical_action[env_ids] = 0.0
            self._previous_action[env_ids] = 0.0
            self._previous_previous_action[env_ids] = 0.0
            self._leg_position_targets[env_ids] = self._q_nominal
            self._wheel_velocity_targets[env_ids] = 0.0
            self._commands[env_ids] = sample_commands(len(env_ids), device=self.device, ranges=self.cfg.commands)
            self._previous_joint_velocity[env_ids] = controlled_joint_feedback_usd_to_control(
                joint_velocity[:, self._controlled_joint_ids]
            )
            self.robot.set_joint_position_target(
                self._leg_position_targets[env_ids], joint_ids=self._leg_joint_ids, env_ids=env_ids
            )
            self.robot.set_joint_velocity_target(
                self._wheel_velocity_targets[env_ids], joint_ids=self._wheel_joint_ids, env_ids=env_ids
            )
            self._state = None
            refreshed_state = self._capture_state()
            if previous_joint_acceleration is not None and len(env_ids) < self.num_envs:
                active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                active[env_ids] = False
                refreshed_state.joint_acceleration[active] = previous_joint_acceleration[active]
            refreshed_state.joint_acceleration[env_ids] = 0.0
            refreshed_state.applied_torque[env_ids] = 0.0
        else:
            self._state = None
