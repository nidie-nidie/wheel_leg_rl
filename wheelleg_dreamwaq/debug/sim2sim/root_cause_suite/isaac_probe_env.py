from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils import math as math_utils

from debug.sim2sim.isaac_debug_env import WheelLegSim2SimDebugEnv, make_debug_env_cfg
from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2
from wheelleg_dreamwaq.assets.asset_overrides import (
    assert_embedded_ground_absent,
    configure_wheel_only_collisions,
)
from wheelleg_dreamwaq.assets.wheelleg import PASSIVE_JOINT_NAMES, WHEELLEG_CFG
from wheelleg_dreamwaq.schemas.action import controlled_joint_feedback_usd_to_control
from wheelleg_dreamwaq.schemas.randomization import (
    NOMINAL_EVALUATION_PROFILE_V1,
    validate_closed_chain_reset_cache_artifact,
)

from .compiled_properties import BodyProperty, composite_properties
from .contracts import (
    ISAAC_LOOP_CLOSURE_PRIM_PATHS,
    DeclaredInfinitySpec,
    declared_usd_infinity_spec,
    encode_declared_usd_value,
    validate_declared_infinity_payload,
)
from .variant_builders import IsaacVariantSpec, ROBOT_HINGES


LOOP_JOINT_RELATIVE_PATHS = (
    "jIO/jIO_loop_closure",
    "jKN/jKN_loop_closure",
    "jEC/jAG_loop_closure",
    "jCF/jCF_revolute_joint",
)


@dataclass(frozen=True)
class ProbeEnvironmentIdentity:
    variant: dict[str, Any]
    closure_joints: tuple[dict[str, Any], ...]
    drive: dict[str, Any]
    compiled: dict[str, Any]


def _usd_value(value: Any, *, prim_path: str, attribute: str) -> Any:
    def plain(item: Any) -> Any:
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        if hasattr(item, "pathString"):
            return str(item.pathString)
        if isinstance(item, (tuple, list)):
            return [plain(child) for child in item]
        try:
            return [plain(child) for child in item]
        except TypeError:
            return str(item)

    return encode_declared_usd_value(
        plain(value), prim_path=prim_path, attribute=attribute
    )


def _prim_semantics(prim: Any, *, excluded_attributes: set[str] | None = None) -> dict[str, Any]:
    excluded = set() if excluded_attributes is None else set(excluded_attributes)
    prim_path = str(prim.GetPath())
    attributes: dict[str, Any] = {}
    for attribute in prim.GetAttributes():
        name = attribute.GetName()
        if name in excluded:
            continue
        attributes[name] = {
            "type": str(attribute.GetTypeName()),
            "value": _usd_value(
                attribute.Get(), prim_path=prim_path, attribute=name
            ),
        }
    relationships = {
        relationship.GetName(): [str(path) for path in relationship.GetTargets()]
        for relationship in prim.GetRelationships()
    }
    return {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "attributes": dict(sorted(attributes.items())),
        "relationships": dict(sorted(relationships.items())),
    }


def make_probe_env_cfg(
    spec: IsaacVariantSpec,
    *,
    device: str,
    num_envs: int,
    enable_contact_sensors: bool = False,
    episode_length_s: float = 10.02,
) -> Any:
    if num_envs <= 0:
        raise ValueError("Probe environment count must be positive")
    cfg = make_debug_env_cfg(device=device, enable_contact_sensors=enable_contact_sensors)
    cfg.scene.num_envs = int(num_envs)
    cfg.scene.env_spacing = 1.5
    cfg.scene.replicate_physics = True
    cfg.scene.clone_in_fabric = False
    cfg.episode_length_s = float(episode_length_s)
    cfg.randomization = NOMINAL_EVALUATION_PROFILE_V1
    cfg.sim.gravity = (0.0, 0.0, -9.81) if spec.gravity_enabled else (0.0, 0.0, 0.0)

    articulation = WHEELLEG_CFG.spawn.articulation_props.replace(
        fix_root_link=bool(spec.fixed_base),
    )
    spawn = WHEELLEG_CFG.spawn.replace(
        activate_contact_sensors=enable_contact_sensors,
        articulation_props=articulation,
    )
    actuators = dict(WHEELLEG_CFG.actuators)
    if not spec.drive_enabled:
        actuators = {
            name: actuator.replace(stiffness=0.0, damping=0.0)
            for name, actuator in actuators.items()
        }
    cfg.robot_cfg = WHEELLEG_CFG.replace(spawn=spawn, actuators=actuators)
    return cfg


class RootCauseIsaacProbeEnv(WheelLegSim2SimDebugEnv):
    """Suite-local immutable diagnostic variant of the production debug environment."""

    def __init__(
        self,
        cfg: Any,
        *,
        variant: IsaacVariantSpec,
        direct_effort: bool,
        reset_artifact: dict[str, Any] | None = None,
        replicate_reset_row_zero: bool = False,
        enable_contact_sensors: bool = False,
        **kwargs: Any,
    ) -> None:
        self._root_cause_variant = variant
        self._root_cause_direct_effort = bool(direct_effort)
        self._root_cause_reset_artifact = reset_artifact
        self._root_cause_replicate_reset_row_zero = bool(replicate_reset_row_zero)
        self._root_cause_effort = torch.zeros((cfg.scene.num_envs, 6), device=cfg.sim.device)
        self._root_cause_closure_records: list[dict[str, Any]] = []
        self._root_cause_record_captures = False
        self._root_cause_capture_buffer: list[dict[str, torch.Tensor]] = []
        self._root_cause_substep_system: list[dict[str, torch.Tensor]] = []
        super().__init__(
            cfg,
            enable_contact_sensors=enable_contact_sensors,
            **kwargs,
        )

    def _setup_scene(self) -> None:
        from pxr import UsdPhysics

        self.robot = Articulation(self.cfg.robot_cfg)
        source_robot_path = "/World/envs/env_0/Robot"
        stage = self.sim.get_initial_stage()
        assert_embedded_ground_absent(stage, source_robot_path)
        configure_wheel_only_collisions(stage, source_robot_path)

        self._root_cause_closure_records = []
        for relative in LOOP_JOINT_RELATIVE_PATHS:
            path = f"{source_robot_path}/{relative}"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdPhysics.Joint):
                raise RuntimeError(f"Frozen loop joint is missing: {path}")
            joint = UsdPhysics.Joint(prim)
            enabled_before = joint.GetJointEnabledAttr().Get()
            if enabled_before is None:
                enabled_before = True
            if not self._root_cause_variant.closure_enabled:
                joint.CreateJointEnabledAttr(False).Set(False)
            enabled_after = joint.GetJointEnabledAttr().Get()
            if enabled_after is None:
                enabled_after = True
            self._root_cause_closure_records.append(
                {
                    "path": path,
                    "enabled_before": bool(enabled_before),
                    "enabled_after": bool(enabled_after),
                    "resolved_prim": _prim_semantics(
                        prim, excluded_attributes={"physics:jointEnabled"}
                    ),
                }
            )

        if self._root_cause_variant.ground_enabled:
            self.cfg.ground.spawn.func(
                self.cfg.ground.prim_path,
                self.cfg.ground.spawn,
                translation=self.cfg.ground.translation,
            )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu" and self._root_cause_variant.ground_enabled:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.ground.prim_path])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _initialize_closed_chain_reset_cache(
        self, source_artifact: dict[str, Any] | None
    ) -> dict[str, Any]:
        del source_artifact
        supplied = self._root_cause_reset_artifact
        if supplied is None:
            joint_position = self.robot.data.default_joint_pos.detach().clone()
            joint_position[:, self._leg_joint_ids] = self._q_reference
            root_height_offset = torch.zeros(self.num_envs, device=self.device)
            origin = "suite_nominal_direct"
            source_identity = None
        else:
            artifact = validate_closed_chain_reset_cache_artifact(supplied)
            source_joint = artifact["q_reset_projected_env"]
            source_height = artifact["root_height_offset_env"]
            if source_joint.shape[0] == self.num_envs:
                joint_position = source_joint.to(device=self.device).clone()
                root_height_offset = source_height.to(device=self.device).clone()
                selected_rows = list(range(self.num_envs))
            elif self.num_envs == 1:
                joint_position = source_joint[:1].to(device=self.device).clone()
                root_height_offset = source_height[:1].to(device=self.device).clone()
                selected_rows = [0]
            elif self._root_cause_replicate_reset_row_zero:
                joint_position = source_joint[:1].repeat(self.num_envs, 1).to(
                    device=self.device
                )
                root_height_offset = source_height[:1].repeat(self.num_envs).to(
                    device=self.device
                )
                selected_rows = [0] * self.num_envs
            else:
                raise ValueError(
                    "Frozen reset cache row count must equal num_envs, except for an explicit one-env row-0 probe"
                )
            origin = (
                "suite_frozen_row0_replicated"
                if self._root_cause_replicate_reset_row_zero and self.num_envs != 1
                else "suite_frozen_tensor"
            )
            source_identity = {
                "schema_version": artifact["schema_version"],
                "algorithm_version": artifact["algorithm_version"],
                "root_height_algorithm_version": artifact["root_height_algorithm_version"],
                "tensor_sha256": artifact["tensor_sha256"],
                "identity": artifact["identity"],
                "selected_environment_rows": selected_rows,
            }
        self._closed_chain_cache_origin = origin
        self._closed_chain_resume_validation_metrics = None
        return {
            "schema_version": "RootCauseSuiteResetTensorV1",
            "origin": origin,
            "source_identity": source_identity,
            "q_reset_projected_env": joint_position,
            "root_height_offset_env": root_height_offset,
        }

    def _assert_passive_actuator_contract(self) -> None:
        if self._root_cause_variant.drive_enabled:
            super()._assert_passive_actuator_contract()
            return
        actuator = self.robot.actuators.get("passive")
        if actuator is None:
            raise RuntimeError("Passive actuator group is missing")
        if not torch.equal(actuator.stiffness, torch.zeros_like(actuator.stiffness)):
            raise RuntimeError("Drive-off passive stiffness is non-zero")
        if not torch.equal(actuator.damping, torch.zeros_like(actuator.damping)):
            raise RuntimeError("Drive-off passive damping is non-zero")
        target = torch.zeros((self.num_envs, len(self._passive_joint_ids)), device=self.device)
        self.robot.set_joint_velocity_target(target, joint_ids=self._passive_joint_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if not self._root_cause_direct_effort:
            super()._pre_physics_step(actions)
            return
        values = actions.to(device=self.device, dtype=torch.float32)
        if values.shape != (self.num_envs, 6):
            raise ValueError(
                f"Direct effort must have shape ({self.num_envs}, 6), got {tuple(values.shape)}"
            )
        self._previous_previous_action.copy_(self._previous_action)
        self._previous_action.copy_(self._canonical_action)
        self._canonical_action.copy_(values)
        self.actions = self._canonical_action
        self._root_cause_effort.copy_(values)
        current = self.robot.data.joint_pos[:, self._controlled_joint_ids]
        self._leg_position_targets.copy_(current[:, :4])
        self._wheel_velocity_targets.zero_()

    def _apply_action(self) -> None:
        if not self._root_cause_direct_effort:
            super()._apply_action()
            return
        current = self.robot.data.joint_pos[:, self._controlled_joint_ids]
        self.robot.set_joint_position_target(current, joint_ids=self._controlled_joint_ids)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(current), joint_ids=self._controlled_joint_ids
        )
        native = controlled_joint_feedback_usd_to_control(self._root_cause_effort)
        self.robot.set_joint_effort_target(native, joint_ids=self._controlled_joint_ids)

    def _debug_target_and_reference(
        self, pre_state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._root_cause_direct_effort:
            return super()._debug_target_and_reference(pre_state)
        target_native = torch.cat((self._leg_position_targets, self._wheel_velocity_targets), dim=-1)
        target_canonical = controlled_joint_feedback_usd_to_control(target_native)
        unclipped = self._root_cause_effort
        clipped = torch.clamp(
            unclipped,
            -self._controlled_effort_limits.unsqueeze(0),
            self._controlled_effort_limits.unsqueeze(0),
        )
        return target_canonical, target_native, unclipped, clipped

    def _debug_capture_direct_state(self) -> dict[str, torch.Tensor]:
        state = super()._debug_capture_direct_state()
        if self._root_cause_record_captures:
            system = self.capture_system_state()
            self._root_cause_capture_buffer.append(
                {
                    name: value.detach().clone()
                    for name, value in system.items()
                }
            )
        return state

    def step(self, action: torch.Tensor):
        self._root_cause_capture_buffer = []
        self._root_cause_record_captures = True
        try:
            result = super().step(action)
        finally:
            self._root_cause_record_captures = False
        expected = 2 * int(self.cfg.decimation)
        if len(self._root_cause_capture_buffer) != expected:
            raise RuntimeError(
                "Root-cause system-state capture count differs from pre/post substep cadence: "
                f"{len(self._root_cause_capture_buffer)} != {expected}"
            )
        self._root_cause_substep_system = self._root_cause_capture_buffer[1::2]
        return result

    def validate_variant_identity(self) -> ProbeEnvironmentIdentity:
        hinge_ids, hinge_names = self.robot.find_joints(list(ROBOT_HINGES), preserve_order=True)
        if tuple(hinge_names) != ROBOT_HINGES or len(set(hinge_ids)) != len(ROBOT_HINGES):
            raise RuntimeError("Isaac compiled hinge identity differs from the frozen 26 names")
        stiffness = self.robot.root_physx_view.get_dof_stiffnesses().to(self.device)
        damping = self.robot.root_physx_view.get_dof_dampings().to(self.device)
        armature = self.robot.root_physx_view.get_dof_armatures().to(self.device)
        selected_stiffness = stiffness[:, hinge_ids]
        selected_damping = damping[:, hinge_ids]
        data_stiffness = self.robot.data.joint_stiffness[:, hinge_ids]
        data_damping = self.robot.data.joint_damping[:, hinge_ids]
        data_armature = self.robot.data.joint_armature[:, hinge_ids]
        joint_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        velocity_limits = self.robot.root_physx_view.get_dof_max_velocities().to(
            self.device
        )
        effort_limits = self.robot.root_physx_view.get_dof_max_forces().to(self.device)
        if not self._root_cause_variant.drive_enabled:
            if not torch.equal(selected_stiffness, torch.zeros_like(selected_stiffness)):
                raise RuntimeError("Drive-off PhysX stiffness is non-zero")
            if not torch.equal(selected_damping, torch.zeros_like(selected_damping)):
                raise RuntimeError("Drive-off PhysX damping is non-zero")
            if not torch.equal(data_stiffness, torch.zeros_like(data_stiffness)):
                raise RuntimeError("Drive-off ArticulationData stiffness is non-zero")
            if not torch.equal(data_damping, torch.zeros_like(data_damping)):
                raise RuntimeError("Drive-off ArticulationData damping is non-zero")
            for actuator in self.robot.actuators.values():
                if not torch.equal(actuator.stiffness, torch.zeros_like(actuator.stiffness)):
                    raise RuntimeError("Drive-off actuator cache stiffness is non-zero")
                if not torch.equal(actuator.damping, torch.zeros_like(actuator.damping)):
                    raise RuntimeError("Drive-off actuator cache damping is non-zero")
        closure_paths = tuple(
            str(record["path"]) for record in self._root_cause_closure_records
        )
        if len(closure_paths) != len(ISAAC_LOOP_CLOSURE_PRIM_PATHS) or set(
            closure_paths
        ) != set(ISAAC_LOOP_CLOSURE_PRIM_PATHS):
            raise RuntimeError(
                "Resolved closure prim paths differ from the frozen four-path identity"
            )
        expected_enabled = self._root_cause_variant.closure_enabled
        if any(record["enabled_after"] != expected_enabled for record in self._root_cause_closure_records):
            raise RuntimeError("Closure joint enabled state differs from the requested variant")
        robot_hinges = {
            name: {
                "stiffness": float(selected_stiffness[0, index].item()),
                "damping": float(selected_damping[0, index].item()),
                "armature": float(armature[0, hinge_ids[index]].item()),
                "position_limits": joint_limits[0, hinge_ids[index]]
                .detach()
                .cpu()
                .tolist(),
                "velocity_limit": float(
                    velocity_limits[0, hinge_ids[index]].item()
                ),
                "effort_limit": float(effort_limits[0, hinge_ids[index]].item()),
                "articulation_data_stiffness": float(
                    data_stiffness[0, index].item()
                ),
                "articulation_data_damping": float(data_damping[0, index].item()),
                "articulation_data_armature": float(
                    data_armature[0, index].item()
                ),
            }
            for index, name in enumerate(hinge_names)
        }
        body_properties = {
            record["name"]: {
                key: value for key, value in record.items() if key != "name"
            }
            for record in self.compiled_body_property_records(require_single_env=False)
        }
        stage = self.sim.get_initial_stage()
        physics_scene = stage.GetPrimAtPath(self.cfg.sim.physics_prim_path)
        if not physics_scene.IsValid():
            raise RuntimeError("Resolved PhysX scene prim is missing")
        articulation_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not articulation_prim.IsValid():
            raise RuntimeError("Resolved articulation prim is missing")
        closure = {
            record["path"]: {
                "enabled": record["enabled_after"],
                "resolved_prim": record["resolved_prim"],
            }
            for record in self._root_cause_closure_records
        }
        material_properties = (
            self.robot.root_physx_view.get_material_properties()[0]
            .detach()
            .cpu()
            .tolist()
        )
        identity = ProbeEnvironmentIdentity(
            variant={
                "ground_enabled": self._root_cause_variant.ground_enabled,
                "gravity_enabled": self._root_cause_variant.gravity_enabled,
                "closure_enabled": self._root_cause_variant.closure_enabled,
                "drive_enabled": self._root_cause_variant.drive_enabled,
                "fixed_base": self._root_cause_variant.fixed_base,
                "direct_effort": self._root_cause_direct_effort,
                "asset_bundle_version": ASSET_BUNDLE_V2.version,
                "asset_bundle_hash": ASSET_BUNDLE_V2.bundle_hash,
            },
            closure_joints=tuple(self._root_cause_closure_records),
            drive={
                "hinge_names": list(hinge_names),
                "stiffness": selected_stiffness[0].detach().cpu().tolist(),
                "damping": selected_damping[0].detach().cpu().tolist(),
                "armature": armature[:, hinge_ids][0].detach().cpu().tolist(),
                "data_stiffness": data_stiffness[0].detach().cpu().tolist(),
                "data_damping": data_damping[0].detach().cpu().tolist(),
                "data_armature": data_armature[0].detach().cpu().tolist(),
                "passive_joint_names": list(PASSIVE_JOINT_NAMES),
                "reset_origin": self._closed_chain_cache_origin,
            },
            compiled={
                "topology": {
                    "body_count": int(self.robot.num_bodies),
                    "joint_count": int(self.robot.num_joints),
                    "body_names": list(self.robot.body_names),
                    "joint_names": list(self.robot.joint_names),
                    "robot_hinge_names": list(hinge_names),
                    "actuator_group_names": sorted(self.robot.actuators),
                    "constraint_names": sorted(closure),
                },
                "physics_scene": _prim_semantics(physics_scene),
                "articulation": {
                    "resolved_prim": _prim_semantics(articulation_prim),
                    "enabled_self_collisions": self.cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions,
                    "solver_position_iteration_count": int(
                        self.cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count
                    ),
                    "solver_velocity_iteration_count": int(
                        self.cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count
                    ),
                    "fixed_base": bool(self._root_cause_variant.fixed_base),
                },
                "bodies": body_properties,
                "robot_hinges": robot_hinges,
                "closure_constraints": closure,
                "material_properties": material_properties,
                "actuator_configs": {
                    name: actuator.to_dict()
                    for name, actuator in sorted(self.cfg.robot_cfg.actuators.items())
                },
            },
        )
        identity_payload = {
            "variant": identity.variant,
            "closure_joints": list(identity.closure_joints),
            "drive": identity.drive,
            "compiled": identity.compiled,
        }

        def declaration_for_path(
            path: tuple[str | int, ...],
        ) -> DeclaredInfinitySpec | None:
            if path == (
                "compiled",
                "physics_scene",
                "attributes",
                "physxScene:maxBiasCoefficient",
                "value",
            ):
                return declared_usd_infinity_spec(
                    "/physicsScene", "physxScene:maxBiasCoefficient"
                )
            if (
                len(path) == 6
                and path[0] == "closure_joints"
                and isinstance(path[1], int)
                and path[2:4] == ("resolved_prim", "attributes")
                and path[5] == "value"
                and 0 <= path[1] < len(identity.closure_joints)
            ):
                attribute = str(path[4])
                closure_path = str(identity.closure_joints[path[1]]["path"])
                return declared_usd_infinity_spec(closure_path, attribute)
            if (
                len(path) == 7
                and path[:2] == ("compiled", "closure_constraints")
                and isinstance(path[2], str)
                and path[3:5] == ("resolved_prim", "attributes")
                and path[6] == "value"
            ):
                return declared_usd_infinity_spec(str(path[2]), str(path[5]))
            return None

        validate_declared_infinity_payload(
            identity_payload,
            declaration_for_path=declaration_for_path,
            label="Isaac ProbeEnvironmentIdentity",
        )
        return identity

    def compiled_body_property_records(
        self, *, require_single_env: bool = True
    ) -> list[dict[str, Any]]:
        if require_single_env and self.num_envs != 1:
            raise RuntimeError("Compiled body-property audit requires exactly one environment")
        masses = self.robot.root_physx_view.get_masses()[0]
        inertias = self.robot.root_physx_view.get_inertias()[0].reshape(self.robot.num_bodies, 3, 3)
        coms = self.robot.root_physx_view.get_coms()[0]
        com_quat_wxyz = math_utils.convert_quat(coms[:, 3:7], to="wxyz")
        return [
            {
                "name": name,
                "mass_kg": float(masses[index].item()),
                "center_of_mass_m": coms[index, :3].detach().cpu().tolist(),
                "principal_axes_wxyz": com_quat_wxyz[index].detach().cpu().tolist(),
                "inertia_com_body_kg_m2": inertias[index].detach().cpu().tolist(),
            }
            for index, name in enumerate(self.robot.body_names)
        ]

    def capture_system_state(self) -> dict[str, torch.Tensor]:
        data = self.robot.data
        masses = data.default_mass.to(self.device)
        positions = data.body_com_pos_w
        velocities = data.body_com_lin_vel_w
        angular_velocities = data.body_com_ang_vel_w
        rotations = math_utils.matrix_from_quat(data.body_link_quat_w)
        inertias_body = data.default_inertia.to(self.device).reshape(self.num_envs, self.robot.num_bodies, 3, 3)
        inertias_world = rotations @ inertias_body @ rotations.transpose(-1, -2)
        linear_momenta = masses.unsqueeze(-1) * velocities
        total_mass = masses.sum(dim=-1)
        system_com = (masses.unsqueeze(-1) * positions).sum(dim=1) / total_mass.unsqueeze(-1)
        angular_com_each = torch.matmul(inertias_world, angular_velocities.unsqueeze(-1)).squeeze(-1)
        offsets = positions - system_com.unsqueeze(1)
        angular_momentum = angular_com_each + torch.linalg.cross(offsets, linear_momenta, dim=-1)
        translational_ke = 0.5 * masses * torch.sum(velocities.square(), dim=-1)
        rotational_ke = 0.5 * torch.sum(angular_velocities * angular_com_each, dim=-1)
        return {
            "system_com_position_world": system_com,
            "linear_momentum_world": linear_momenta.sum(dim=1),
            "angular_momentum_com_world": angular_momentum.sum(dim=1),
            "kinetic_energy_j": (translational_ke + rotational_ke).sum(dim=1),
        }

    def compiled_body_properties(self) -> list[BodyProperty]:
        data = self.robot.data
        if self.num_envs != 1:
            raise RuntimeError("Compiled body-property audit requires exactly one environment")
        properties: list[BodyProperty] = []
        rotations = math_utils.matrix_from_quat(
            data.body_link_quat_w.to(torch.float64)
        )[0]
        inertias = data.default_inertia[0].reshape(self.robot.num_bodies, 3, 3)
        for index, name in enumerate(self.robot.body_names):
            properties.append(
                BodyProperty(
                    name=name,
                    mass=float(data.default_mass[0, index].item()),
                    com_position_world=data.body_com_pos_w[0, index].detach().cpu().numpy(),
                    inertia_com_body=inertias[index].detach().cpu().numpy(),
                    rotation_world_from_body=rotations[index].detach().cpu().numpy(),
                    com_linear_velocity_world=data.body_com_lin_vel_w[0, index].detach().cpu().numpy(),
                    angular_velocity_world=data.body_com_ang_vel_w[0, index].detach().cpu().numpy(),
                )
            )
        return properties

    def compiled_composite(self) -> Any:
        return composite_properties(self.compiled_body_properties())
