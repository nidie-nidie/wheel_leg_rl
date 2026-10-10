from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import carb
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationContext

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2
from wheelleg_dreamwaq.assets.asset_overrides import (
    assert_embedded_ground_absent,
    configure_wheel_only_collisions,
    is_wheel_collision_path,
)
from wheelleg_dreamwaq.assets.wheelleg import PASSIVE_JOINT_NAMES
from wheelleg_dreamwaq.schemas.action import CANONICAL_JOINT_ORDER, controlled_joint_feedback_usd_to_control
from wheelleg_dreamwaq.schemas.frames import (
    projected_gravity_from_quaternion,
    quat_rotate_inverse_wxyz,
    quat_rotate_wxyz,
    transform_usd_vector_to_control,
)
from wheelleg_dreamwaq.schemas.physics import validate_phase1_physics
from wheelleg_dreamwaq.schemas.physics import PHYSICS_SCHEMA_VERSION
from wheelleg_dreamwaq.schemas.randomization import (
    CLOSED_CHAIN_MAX_ACTIVE_ERROR_RAD,
    CLOSED_CHAIN_MAX_LENGTH_ERROR_M,
    CLOSED_CHAIN_MAX_LOOP_ERROR_M,
    CLOSED_CHAIN_MAX_PASSIVE_RAW_DELTA_RAD,
    CLOSED_CHAIN_MAX_PASSIVE_WRAPPED_DELTA_RAD,
    CLOSED_CHAIN_MAX_PHI0_ERROR_RAD,
    CLOSED_CHAIN_MAX_WHEEL_ERROR_M,
    CLOSED_CHAIN_NOMINAL_BRANCH_SIGNATURE,
    CLOSED_CHAIN_PHYSICAL_PASSIVE_BRANCH_JOINTS,
    CLOSED_CHAIN_RELAXATION_STEPS,
    CLOSED_CHAIN_RELAXATION_VERSION,
    CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
    CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION,
    CLOSED_CHAIN_ROOT_LIFT_M,
    CLOSED_CHAIN_TRACE_STEPS,
    canonical_tensor_sha256,
    closed_chain_reset_contract_payload,
    compute_closed_chain_root_height_offset,
    profile_contract_hash,
    validate_closed_chain_reset_cache_artifact,
)
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

from .control import compute_action_targets
from .env_cfg import WheelLegFlatEnvCfg
from .observations import build_observations
from .randomization import WheelLegRandomizationRuntime
from .rewards import compute_reward
from .state import WheelLegState
from .terminations import compute_dones


class WheelLegFlatEnv(DirectRLEnv):
    cfg: WheelLegFlatEnvCfg

    def __init__(
        self,
        cfg: WheelLegFlatEnvCfg,
        render_mode: str | None = None,
        *,
        closed_chain_reset_cache: dict[str, Any] | None = None,
        **kwargs,
    ):
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
        self._passive_joint_ids, passive_names = self.robot.find_joints(PASSIVE_JOINT_NAMES, preserve_order=True)
        if passive_names != PASSIVE_JOINT_NAMES:
            raise RuntimeError(f"Passive joints did not resolve in frozen order: {passive_names}")
        self._physical_passive_branch_ids, branch_names = self.robot.find_joints(
            list(CLOSED_CHAIN_PHYSICAL_PASSIVE_BRANCH_JOINTS), preserve_order=True
        )
        if tuple(branch_names) != CLOSED_CHAIN_PHYSICAL_PASSIVE_BRANCH_JOINTS:
            raise RuntimeError(f"Physical passive branch joints did not resolve in frozen order: {branch_names}")
        self._q_nominal = torch.tensor(self.cfg.q_nominal, dtype=torch.float32, device=self.device)

        self._resolve_virtual_leg_geometry()
        self._assert_runtime_asset_contract()
        if self.cfg.seed is None:
            raise ValueError("WheelLeg randomization requires an explicit environment seed")
        self._randomization = WheelLegRandomizationRuntime(
            profile=self.cfg.randomization,
            master_seed=int(self.cfg.seed),
            device=self.device,
            num_envs=self.num_envs,
            robot=self.robot,
            controlled_joint_ids=self._controlled_joint_ids,
            leg_joint_ids=self._leg_joint_ids,
            wheel_body_ids=self._virtual_leg_wheel_body_ids,
            q_nominal=self._q_nominal,
            geometries=self._virtual_leg_fk_geometries,
            soft_leg_limit=self.cfg.reward_weights.soft_leg_limit,
        )
        self._q_reference = self._randomization.q_reference
        self._controlled_effort_limits = self._randomization.realized_effort_limit
        self._closed_chain_reset_cache_artifact = self._initialize_closed_chain_reset_cache(
            closed_chain_reset_cache
        )
        self._q_reset_projected_env = self._closed_chain_reset_cache_artifact[
            "q_reset_projected_env"
        ].to(device=self.device)
        self._root_height_offset_env = self._closed_chain_reset_cache_artifact[
            "root_height_offset_env"
        ].to(device=self.device)

        self._canonical_action = torch.zeros((self.num_envs, 6), device=self.device)
        self._previous_action = torch.zeros_like(self._canonical_action)
        self._previous_previous_action = torch.zeros_like(self._canonical_action)
        self._leg_position_targets = self._q_reference.clone()
        self._wheel_velocity_targets = torch.zeros((self.num_envs, 2), device=self.device)
        self._commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_joint_velocity = controlled_joint_feedback_usd_to_control(
            self.robot.data.joint_vel[:, self._controlled_joint_ids]
        ).clone()
        self._state: WheelLegState | None = None
        self._resume_sequence_capture_stage = "inactive"
        self._resume_sequence_trace_tensors: dict[str, torch.Tensor] = {}
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

    def _virtual_leg_state(
        self,
        root_quaternion: torch.Tensor,
        joint_position: torch.Tensor,
        *,
        enforce_guards: bool = True,
    ) -> dict[str, torch.Tensor]:
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
        if enforce_guards and bool(invalid.any().item()):
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
        if enforce_guards and bool((loop_error > MAX_LOOP_CLOSURE_ERROR_M).any().item()):
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

    @staticmethod
    def _gravity_tuple(value: Any) -> tuple[float, float, float]:
        return tuple(float(component) for component in value)

    def _closed_chain_cache_identity(self) -> dict[str, Any]:
        audit = self._randomization.audit
        return {
            "asset_bundle_version": ASSET_BUNDLE_V2.version,
            "asset_bundle_hash": ASSET_BUNDLE_V2.bundle_hash,
            "physics_schema_version": PHYSICS_SCHEMA_VERSION,
            "randomization_profile_hash": profile_contract_hash(self.cfg.randomization),
            "realized_plan_hash": audit["realized_plan_hash"],
            "actuator_plan_hash": self._randomization.actuator_plan_hash,
            "master_seed": int(self.cfg.seed),
            "num_envs": self.num_envs,
            "joint_names": list(self.robot.joint_names),
            "passive_joint_names": list(PASSIVE_JOINT_NAMES),
            "physical_passive_branch_joint_names": list(CLOSED_CHAIN_PHYSICAL_PASSIVE_BRANCH_JOINTS),
        }

    def _assert_passive_actuator_contract(self) -> None:
        actuator = self.robot.actuators.get("passive")
        if actuator is None:
            raise RuntimeError("Passive actuator group is missing")
        expected_stiffness = torch.zeros_like(actuator.stiffness)
        expected_damping = torch.full_like(actuator.damping, 0.05)
        if not torch.equal(actuator.stiffness, expected_stiffness):
            raise RuntimeError("Passive actuator stiffness must remain exactly zero")
        if not torch.allclose(actuator.damping, expected_damping, rtol=0.0, atol=1.0e-7):
            raise RuntimeError("Passive actuator damping must remain 0.05")
        passive_velocity_target = torch.zeros(
            (self.num_envs, len(self._passive_joint_ids)), device=self.device
        )
        self.robot.set_joint_velocity_target(passive_velocity_target, joint_ids=self._passive_joint_ids)
        if not torch.equal(
            self.robot.data.joint_vel_target[:, self._passive_joint_ids], passive_velocity_target
        ):
            raise RuntimeError("Passive joint velocity target must remain zero")

    def _closed_chain_root_state(self) -> torch.Tensor:
        root_state = self.robot.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins
        root_state[:, 2] += CLOSED_CHAIN_ROOT_LIFT_M
        root_state[:, 7:] = 0.0
        return root_state

    def _write_closed_chain_boundary_state(
        self,
        joint_position: torch.Tensor,
        root_state: torch.Tensor,
    ) -> None:
        env_ids = self.robot._ALL_INDICES
        joint_velocity = torch.zeros_like(joint_position)
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_position, joint_velocity, None, env_ids)
        self.robot.set_joint_position_target(self._q_reference, joint_ids=self._leg_joint_ids)
        self.robot.set_joint_velocity_target(
            torch.zeros((self.num_envs, 2), device=self.device), joint_ids=self._wheel_joint_ids
        )
        self.scene.write_data_to_sim()

    def _closed_chain_metrics(self) -> dict[str, Any]:
        joint_position = self.robot.data.joint_pos
        joint_velocity = self.robot.data.joint_vel
        controlled_position = controlled_joint_feedback_usd_to_control(
            joint_position[:, self._controlled_joint_ids]
        )
        virtual_leg = self._virtual_leg_state(
            self.robot.data.root_link_quat_w,
            controlled_position,
            enforce_guards=False,
        )
        nominal = self.robot.data.default_joint_pos
        passive_raw_delta = joint_position[:, self._passive_joint_ids] - nominal[:, self._passive_joint_ids]
        passive_wrapped_delta = torch.atan2(torch.sin(passive_raw_delta), torch.cos(passive_raw_delta))
        branch_signature = torch.sign(
            joint_position[:, self._physical_passive_branch_ids]
        ).to(dtype=torch.int8)
        nominal_branch = torch.tensor(
            CLOSED_CHAIN_NOMINAL_BRANCH_SIGNATURE,
            dtype=torch.int8,
            device=self.device,
        ).unsqueeze(0)
        limits = self.robot.data.joint_pos_limits
        below = joint_position < limits[..., 0]
        above = joint_position > limits[..., 1]
        finite = torch.isfinite(joint_position).all() & torch.isfinite(joint_velocity).all()
        per_joint_stats = {}
        for joint_id, name in enumerate(self.robot.joint_names):
            delta = joint_position[:, joint_id] - nominal[:, joint_id]
            per_joint_stats[name] = {
                "minimum_rad": float(joint_position[:, joint_id].min().item()),
                "maximum_rad": float(joint_position[:, joint_id].max().item()),
                "max_abs_nominal_delta_rad": float(torch.abs(delta).max().item()),
            }
        return {
            "finite": bool(finite.item()),
            "loop_error_max_m": float(virtual_leg["loop_error"].max().item()),
            "wheel_error_max_m": float(virtual_leg["wheel_error"].max().item()),
            "length_error_max_m": float(
                torch.abs(virtual_leg["length_fk"] - virtual_leg["length_true"]).max().item()
            ),
            "phi0_error_max_rad": float(
                torch.abs(wrap_to_pi(virtual_leg["phi0_fk"] - virtual_leg["phi0_true"])).max().item()
            ),
            "active_reference_error_max_rad": float(
                torch.abs(joint_position[:, self._leg_joint_ids] - self._q_reference).max().item()
            ),
            "passive_raw_delta_max_abs_rad": float(torch.abs(passive_raw_delta).max().item()),
            "passive_wrapped_delta_max_abs_rad": float(torch.abs(passive_wrapped_delta).max().item()),
            "hard_limit_violation_count": int((below | above).sum().item()),
            "branch_signature_mismatch_count": int((branch_signature != nominal_branch).sum().item()),
            "joint_speed_max_rad_s": float(torch.abs(joint_velocity).max().item()),
            "physical_passive_branch_signature": branch_signature.detach().cpu().tolist(),
            "per_joint_stats": per_joint_stats,
        }

    @staticmethod
    def _assert_closed_chain_metrics(metrics: dict[str, Any]) -> None:
        failures = []
        checks = (
            (metrics["finite"], "non-finite state"),
            (metrics["loop_error_max_m"] <= CLOSED_CHAIN_MAX_LOOP_ERROR_M, "loop closure"),
            (metrics["wheel_error_max_m"] <= CLOSED_CHAIN_MAX_WHEEL_ERROR_M, "wheel FK"),
            (metrics["length_error_max_m"] <= CLOSED_CHAIN_MAX_LENGTH_ERROR_M, "leg length FK"),
            (metrics["phi0_error_max_rad"] <= CLOSED_CHAIN_MAX_PHI0_ERROR_RAD, "phi0 FK"),
            (
                metrics["active_reference_error_max_rad"] <= CLOSED_CHAIN_MAX_ACTIVE_ERROR_RAD,
                "active reference",
            ),
            (
                metrics["passive_raw_delta_max_abs_rad"] < CLOSED_CHAIN_MAX_PASSIVE_RAW_DELTA_RAD,
                "passive raw delta",
            ),
            (
                metrics["passive_wrapped_delta_max_abs_rad"]
                <= CLOSED_CHAIN_MAX_PASSIVE_WRAPPED_DELTA_RAD,
                "passive wrapped delta",
            ),
            (metrics["hard_limit_violation_count"] == 0, "joint hard limit"),
            (metrics["branch_signature_mismatch_count"] == 0, "passive assembly branch"),
        )
        failures.extend(label for passed, label in checks if not passed)
        if failures:
            raise RuntimeError(
                "Closed-chain reset cache hard gate failed: "
                f"{failures}; metrics={metrics}"
            )

    def _apply_closed_chain_cache_position(self, joint_position: torch.Tensor) -> dict[str, Any]:
        root_state = self._closed_chain_root_state()
        self._write_closed_chain_boundary_state(joint_position, root_state)
        self.sim.forward()
        self.scene.update(self.physics_dt)
        metrics = self._closed_chain_metrics()
        self._assert_closed_chain_metrics(metrics)
        return metrics

    def _generate_closed_chain_reset_cache(self) -> dict[str, Any]:
        self._assert_passive_actuator_contract()
        root_state = self._closed_chain_root_state()
        joint_position = self.robot.data.default_joint_pos.clone()
        joint_position[:, self._leg_joint_ids] = self._q_reference
        wheel_nominal = joint_position[:, self._wheel_joint_ids].clone()
        counters_before = {
            "sim_step_counter": int(self._sim_step_counter),
            "common_step_counter": int(self.common_step_counter),
            "episode_length_max": int(self.episode_length_buf.max().item()),
        }
        physics_view = SimulationContext.instance().physics_sim_view
        original_gravity = self._gravity_tuple(physics_view.get_gravity())
        trace = []
        gravity_zero_readback = None
        try:
            physics_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))
            gravity_zero_readback = self._gravity_tuple(physics_view.get_gravity())
            if gravity_zero_readback != (0.0, 0.0, 0.0):
                raise RuntimeError(f"Failed to set zero gravity for closed-chain relaxation: {gravity_zero_readback}")
            for step in range(CLOSED_CHAIN_RELAXATION_STEPS + 1):
                if step > 0:
                    joint_position = self.robot.data.joint_pos.clone()
                    joint_position[:, self._leg_joint_ids] = self._q_reference
                    joint_position[:, self._wheel_joint_ids] = wheel_nominal
                self._write_closed_chain_boundary_state(joint_position, root_state)
                if step == 0:
                    self.sim.forward()
                else:
                    self.sim.step(render=False)
                self.scene.update(self.physics_dt)
                if step in CLOSED_CHAIN_TRACE_STEPS:
                    trace.append(
                        {
                            "step": step,
                            "joint_position_rad": self.robot.data.joint_pos.detach().cpu().contiguous(),
                            "joint_velocity_rad_s": self.robot.data.joint_vel.detach().cpu().contiguous(),
                            "metrics": self._closed_chain_metrics(),
                        }
                    )
            joint_position = self.robot.data.joint_pos.clone()
            joint_position[:, self._leg_joint_ids] = self._q_reference
            joint_position[:, self._wheel_joint_ids] = wheel_nominal
            root_state[:, 7:] = 0.0
            self._write_closed_chain_boundary_state(joint_position, root_state)
            self.sim.forward()
            self.scene.update(self.physics_dt)
            metrics = self._closed_chain_metrics()
            self._assert_closed_chain_metrics(metrics)
        finally:
            physics_view.set_gravity(carb.Float3(*original_gravity))
            gravity_restored = self._gravity_tuple(physics_view.get_gravity())
            if gravity_restored != original_gravity:
                raise RuntimeError(
                    f"Closed-chain relaxation did not restore gravity: {gravity_restored} != {original_gravity}"
                )
        counters_after = {
            "sim_step_counter": int(self._sim_step_counter),
            "common_step_counter": int(self.common_step_counter),
            "episode_length_max": int(self.episode_length_buf.max().item()),
        }
        if counters_after != counters_before:
            raise RuntimeError(
                f"Closed-chain relaxation advanced DirectRLEnv counters: {counters_before} -> {counters_after}"
            )
        q_reset = joint_position.detach().to(device="cpu", dtype=torch.float32).contiguous()
        root_height_offset = compute_closed_chain_root_height_offset(
            self._q_reference,
            q_nominal=self._q_nominal,
            geometries=self._virtual_leg_fk_geometries,
        )
        actuator_plan = self._randomization.actuator_plan
        artifact = {
            "schema_version": CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
            "algorithm_version": CLOSED_CHAIN_RELAXATION_VERSION,
            "root_height_algorithm_version": CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION,
            "contract": closed_chain_reset_contract_payload(sim_dt=self.physics_dt),
            "identity": self._closed_chain_cache_identity(),
            "q_reset_projected_env": q_reset,
            "root_height_offset_env": root_height_offset,
            "tensor_sha256": canonical_tensor_sha256(
                {
                    "q_reset_projected_env": q_reset,
                    "root_height_offset_env": root_height_offset,
                }
            ),
            "actuator_plan": actuator_plan,
            "metrics": {
                **metrics,
                "root_height_offset_m": {
                    "minimum": float(root_height_offset.min().item()),
                    "maximum": float(root_height_offset.max().item()),
                    "mean": float(root_height_offset.mean().item()),
                    "std": float(root_height_offset.std(unbiased=False).item()),
                },
                "gravity_before": list(original_gravity),
                "gravity_zero_readback": list(gravity_zero_readback),
                "gravity_restored_readback": list(original_gravity),
                "environment_counters_before": counters_before,
                "environment_counters_after": counters_after,
                "underlying_physics_time_advanced_s": CLOSED_CHAIN_RELAXATION_STEPS * self.physics_dt,
            },
            "trace": trace,
        }
        return validate_closed_chain_reset_cache_artifact(artifact)

    def _load_closed_chain_reset_cache(self, artifact: dict[str, Any]) -> dict[str, Any]:
        artifact = validate_closed_chain_reset_cache_artifact(artifact)
        expected_identity = self._closed_chain_cache_identity()
        if artifact["identity"] != expected_identity:
            raise RuntimeError(
                "Closed-chain reset cache identity mismatch: "
                f"saved={artifact['identity']}, expected={expected_identity}"
            )
        expected_contract = closed_chain_reset_contract_payload(sim_dt=self.physics_dt)
        if artifact["contract"] != expected_contract:
            raise RuntimeError("Closed-chain reset cache relaxation contract mismatch")
        expected_actuator_plan = self._randomization.actuator_plan
        for name, expected in expected_actuator_plan.items():
            if not torch.equal(artifact["actuator_plan"][name], expected):
                raise RuntimeError(f"Closed-chain reset cache actuator plan mismatch: {name}")
        expected_root_height_offset = compute_closed_chain_root_height_offset(
            self._q_reference,
            q_nominal=self._q_nominal,
            geometries=self._virtual_leg_fk_geometries,
        )
        if not torch.equal(artifact["root_height_offset_env"], expected_root_height_offset):
            raise RuntimeError("Closed-chain reset cache root-height alignment mismatch")
        self._assert_passive_actuator_contract()
        joint_position = artifact["q_reset_projected_env"].to(device=self.device)
        validation_metrics = self._apply_closed_chain_cache_position(joint_position)
        self._closed_chain_resume_validation_metrics = validation_metrics
        return artifact

    def _initialize_closed_chain_reset_cache(
        self,
        source_artifact: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if source_artifact is None:
            self._closed_chain_cache_origin = "generated"
            self._closed_chain_resume_validation_metrics = None
            return self._generate_closed_chain_reset_cache()
        self._closed_chain_cache_origin = "loaded"
        return self._load_closed_chain_reset_cache(source_artifact)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_previous_action.copy_(self._previous_action)
        self._previous_action.copy_(self._canonical_action)
        clipped, leg_targets, wheel_targets = compute_action_targets(actions, self._q_reference, self.cfg.control)
        self._canonical_action.copy_(clipped)
        self.actions = self._canonical_action
        self._leg_position_targets.copy_(leg_targets)
        self._wheel_velocity_targets.copy_(wheel_targets)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._leg_position_targets, joint_ids=self._leg_joint_ids)
        self.robot.set_joint_velocity_target(self._wheel_velocity_targets, joint_ids=self._wheel_joint_ids)

    def _read_state(self) -> WheelLegState:
        joint_position = controlled_joint_feedback_usd_to_control(
            self.robot.data.joint_pos[:, self._controlled_joint_ids]
        )
        joint_velocity = controlled_joint_feedback_usd_to_control(
            self.robot.data.joint_vel[:, self._controlled_joint_ids]
        )
        joint_acceleration = (joint_velocity - self._previous_joint_velocity) / self.step_dt
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
        return state

    def _capture_state(self) -> WheelLegState:
        state = self._read_state()
        self._previous_joint_velocity.copy_(state.joint_velocity)
        self._state = state
        return state

    def _current_state(self) -> WheelLegState:
        return self._state if self._state is not None else self._capture_state()

    @staticmethod
    def _canonical_resume_trace_tensor(values: torch.Tensor) -> torch.Tensor:
        return values.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()

    def begin_resume_sequence_capture(self) -> None:
        if self._resume_sequence_capture_stage not in {"inactive", "complete"}:
            raise RuntimeError(
                f"Resume-sequence capture is already active: {self._resume_sequence_capture_stage}"
            )
        self._resume_sequence_trace_tensors = {}
        self._resume_sequence_capture_stage = "awaiting_reset"

    def _capture_resume_reset_samples(
        self,
        *,
        command: torch.Tensor,
        root_velocity_world_usd: torch.Tensor,
    ) -> None:
        if self._resume_sequence_capture_stage != "awaiting_reset":
            return
        self._resume_sequence_trace_tensors = {
            "post_restore_command": self._canonical_resume_trace_tensor(command),
            "post_restore_root_velocity_world_usd": self._canonical_resume_trace_tensor(
                root_velocity_world_usd
            ),
        }
        self._resume_sequence_capture_stage = "awaiting_discarded_policy_obs"

    def _capture_resume_policy_observation(self, actor_obs: torch.Tensor) -> None:
        if self._resume_sequence_capture_stage == "awaiting_discarded_policy_obs":
            self._resume_sequence_trace_tensors["post_restore_reset_discarded_policy_obs"] = (
                self._canonical_resume_trace_tensor(actor_obs)
            )
            self._resume_sequence_capture_stage = "awaiting_first_policy_observation"
        elif self._resume_sequence_capture_stage == "awaiting_first_policy_observation":
            self._resume_sequence_trace_tensors["first_policy_observation"] = (
                self._canonical_resume_trace_tensor(actor_obs)
            )
            self._resume_sequence_capture_stage = "complete"

    def get_resume_sequence_trace_tensors(self) -> dict[str, torch.Tensor]:
        if self._resume_sequence_capture_stage != "complete":
            raise RuntimeError(
                "Resume-sequence capture is incomplete: "
                f"{self._resume_sequence_capture_stage}"
            )
        expected = {
            "post_restore_command",
            "post_restore_root_velocity_world_usd",
            "post_restore_reset_discarded_policy_obs",
            "first_policy_observation",
        }
        if set(self._resume_sequence_trace_tensors) != expected:
            raise RuntimeError("Resume-sequence capture fields are incomplete")
        return {
            name: values.clone()
            for name, values in self._resume_sequence_trace_tensors.items()
        }

    def _get_observations(self) -> dict[str, torch.Tensor]:
        actor_obs, critic_obs = build_observations(
            self._current_state(),
            q_reference=self._q_reference,
            normalization=self.cfg.normalization,
            actor_noise=self._randomization.sample_actor_noise(),
        )
        self._capture_resume_policy_observation(actor_obs)
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
        base_height_error = torch.abs(self._commands[:, 2] - self._state.base_height[:, 0])

        def masked_height_error(mask: torch.Tensor) -> torch.Tensor:
            return base_height_error[mask].mean() if bool(mask.any().item()) else base_height_error.new_zeros(())

        negative_offset = self._root_height_offset_env < -1.0e-3
        near_zero_offset = torch.abs(self._root_height_offset_env) <= 1.0e-3
        positive_offset = self._root_height_offset_env > 1.0e-3
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
                    >= 0.99 * self._controlled_effort_limits
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
                "Tracking/base_height_abs_error": base_height_error.mean(),
                "Tracking/base_height_abs_error_offset_negative": masked_height_error(negative_offset),
                "Tracking/base_height_abs_error_offset_near_zero": masked_height_error(near_zero_offset),
                "Tracking/base_height_abs_error_offset_positive": masked_height_error(positive_offset),
                "Randomization/root_height_offset_negative_fraction": negative_offset.float().mean(),
                "Randomization/root_height_offset_near_zero_fraction": near_zero_offset.float().mean(),
                "Randomization/root_height_offset_positive_fraction": positive_offset.float().mean(),
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
        if self._resume_sequence_capture_stage == "awaiting_reset":
            expected_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            if not torch.equal(env_ids, expected_env_ids):
                raise RuntimeError("Resume-sequence capture requires a full ordered environment reset")
        transition_state = self._state

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
        joint_position = self._q_reset_projected_env[env_ids].clone()
        joint_velocity = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] += self._root_height_offset_env[env_ids]
        root_state[:, 7:] = self._randomization.sample_root_velocity(len(env_ids))

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_position, joint_velocity, None, env_ids)

        if hasattr(self, "_canonical_action"):
            self._canonical_action[env_ids] = 0.0
            self._previous_action[env_ids] = 0.0
            self._previous_previous_action[env_ids] = 0.0
            self._leg_position_targets[env_ids] = self._q_reference[env_ids]
            self._wheel_velocity_targets[env_ids] = 0.0
            self._commands[env_ids] = self._randomization.sample_commands(len(env_ids), self.cfg.commands)
            self._capture_resume_reset_samples(
                command=self._commands[env_ids],
                root_velocity_world_usd=root_state[:, 7:],
            )
            self._previous_joint_velocity[env_ids] = controlled_joint_feedback_usd_to_control(
                joint_velocity[:, self._controlled_joint_ids]
            )
            self.robot.set_joint_position_target(
                self._leg_position_targets[env_ids], joint_ids=self._leg_joint_ids, env_ids=env_ids
            )
            self.robot.set_joint_velocity_target(
                self._wheel_velocity_targets[env_ids], joint_ids=self._wheel_joint_ids, env_ids=env_ids
            )
            self.robot.set_joint_velocity_target(
                torch.zeros((len(env_ids), len(self._passive_joint_ids)), device=self.device),
                joint_ids=self._passive_joint_ids,
                env_ids=env_ids,
            )
            # Isaac Lab 2.3.2 writers update the base pose/COM velocity but leave
            # these derived root caches valid at the unchanged reset timestamp.
            # Recompute them from this reset without advancing any clock/physics.
            self.robot.data._root_link_vel_w.timestamp = -1.0
            self.robot.data._root_com_pose_w.timestamp = -1.0
            reset_state = self._read_state()
            # The actuator diagnostic buffer still describes the preceding physics step.
            # No torque has been applied in the new episode yet, so reset rows are zero.
            reset_state.applied_torque[env_ids] = 0.0
            if transition_state is None or len(env_ids) == self.num_envs:
                self._state = reset_state
            else:
                self._state = transition_state.replace_rows(reset_state, env_ids)
        else:
            self._state = None

    @property
    def randomization_audit(self) -> dict:
        audit = self._randomization.audit
        artifact = self._closed_chain_reset_cache_artifact
        audit["closed_chain_reset_cache"] = {
            "origin": self._closed_chain_cache_origin,
            "schema_version": artifact["schema_version"],
            "algorithm_version": artifact["algorithm_version"],
            "root_height_algorithm_version": artifact["root_height_algorithm_version"],
            "tensor_sha256": artifact["tensor_sha256"],
            "identity": artifact["identity"],
            "metrics": artifact["metrics"],
            "resume_validation_metrics": self._closed_chain_resume_validation_metrics,
        }
        return audit

    @property
    def closed_chain_reset_cache_artifact(self) -> dict[str, Any]:
        return self._closed_chain_reset_cache_artifact

    def get_randomization_rng_state(self) -> dict:
        return self._randomization.get_rng_state()

    def set_randomization_rng_state(
        self,
        payload: dict,
        stream_names: Sequence[str] | None = None,
    ) -> None:
        self._randomization.set_rng_state(payload, stream_names=stream_names)
