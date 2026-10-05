"""A fitted-motor A1 flat task isolated from the default locomotion tasks."""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .env_cfg import GogoA1FlatStudentEnvCfg, _apply_play_settings
from . import mdp as gogo_mdp
from .pace_actuator_cfg import load_fitted_a1_actuator

UPSIDE_DOWN_FIXED_AIR_POSITION = (0.0, 0.0, 1.0)
UPSIDE_DOWN_FIXED_AIR_ROTATION_WXYZ = (0.0, 1.0, 0.0, 0.0)
PAPER_FOOT_BODY_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def _required_environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must name the fitted A1 PACE input")
    return Path(value)


def _apply_local_a1_usd_override(cfg: GogoA1FlatStudentEnvCfg) -> None:
    value = os.environ.get("A1_LOCAL_USD_PATH")
    if not value:
        return
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"A1_LOCAL_USD_PATH does not exist: {path}")
    cfg.scene.robot.spawn.usd_path = str(path)


def _apply_local_terrain_usd_override(cfg: GogoA1FlatStudentEnvCfg) -> None:
    value = os.environ.get("A1_LOCAL_TERRAIN_USD_PATH")
    if not value:
        return
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"A1_LOCAL_TERRAIN_USD_PATH does not exist: {path}")
    cfg.scene.terrain.terrain_type = "usd"
    cfg.scene.terrain.usd_path = str(path)
    cfg.scene.terrain.terrain_generator = None


def _apply_upside_down_fixed_air_fixture(cfg: GogoA1FlatStudentEnvCfg) -> None:
    cfg.scene.robot.init_state.pos = UPSIDE_DOWN_FIXED_AIR_POSITION
    cfg.scene.robot.init_state.rot = UPSIDE_DOWN_FIXED_AIR_ROTATION_WXYZ
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.commands.base_velocity.rel_standing_envs = 1.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


def _apply_fitted_pace_actuator(cfg: GogoA1FlatStudentEnvCfg):
    actuator, evidence = load_fitted_a1_actuator(
        _required_environment_path("A1_PACE_FIT_MANIFEST"),
        pace_repository_root=_required_environment_path("A1_PACE_REPOSITORY_ROOT"),
    )
    cfg.scene.robot.actuators = {"base_legs": actuator}
    cfg.sim.dt = 0.002
    cfg.decimation = 10
    cfg.sim.render_interval = cfg.decimation
    cfg.pace_fit_manifest_path = str(evidence.fit_manifest_path)
    cfg.pace_fit_manifest_sha256 = evidence.fit_manifest_sha256
    cfg.pace_mean_path = str(evidence.mean_path)
    cfg.pace_mean_sha256 = evidence.mean_sha256


def _disable_gogo_flat_shaping_rewards(cfg: GogoA1FlatStudentEnvCfg) -> None:
    cfg.rewards.lin_vel_z_l2 = None
    cfg.rewards.ang_vel_xy_l2 = None
    cfg.rewards.flat_orientation_l2 = None
    cfg.rewards.dof_acc_l2 = None
    cfg.rewards.dof_torques_l2 = None
    cfg.rewards.action_rate_l2 = None
    cfg.rewards.feet_air_time = None
    cfg.rewards.base_height_l2 = None
    cfg.rewards.undesired_contacts = None
    cfg.rewards.dof_pos_limits = None
    cfg.rewards.trot_phase_contact = None
    cfg.rewards.swing_foot_clearance = None
    cfg.rewards.swing_foot_height = None
    cfg.rewards.swing_foot_drag = None
    cfg.rewards.stance_foot_slip = None
    cfg.rewards.trot_pair_timing = None
    cfg.rewards.stand_still_joint_deviation = None
    cfg.rewards.stand_still_action = None


def _apply_paper_flat_walk_settings(cfg: GogoA1FlatStudentEnvCfg) -> None:
    cfg.scene.robot.spawn.articulation_props.fix_root_link = False
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (10.0, 10.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    _disable_gogo_flat_shaping_rewards(cfg)
    cfg.rewards.track_lin_vel_xy_exp.weight = 0.2
    cfg.rewards.track_ang_vel_z_exp.weight = 0.2
    cfg.rewards.paper_equivalent_energy = RewTerm(
        func=gogo_mdp.paper_equivalent_energy_proxy,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "torque_limit": 33.5,
            "velocity_limit": 21.0,
            "torque_weight": 0.7,
            "positive_power_weight": 0.3,
            "clip_max": 2.0,
        },
    )
    cfg.rewards.paper_collision = RewTerm(
        func=gogo_mdp.paper_collision_indicator,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
            "threshold": 1.0,
        },
    )
    cfg.rewards.paper_ftd = RewTerm(
        func=gogo_mdp.paper_scheduled_foot_touchdown_velocity,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=PAPER_FOOT_BODY_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=PAPER_FOOT_BODY_NAMES, preserve_order=True),
            "history_length": 3,
            "half_life_iterations": 500.0,
            "ppo_steps_per_iteration": 24,
        },
    )


@configclass
class GogoA1PaceFlatEnvCfg(GogoA1FlatStudentEnvCfg):
    """Flat A1 locomotion with a PACE-fitted motor model and no extra delay patch."""

    pace_fixture_pose: str = "upside_down_fixed_air"
    pace_fit_manifest_path: str = ""
    pace_fit_manifest_sha256: str = ""
    pace_mean_path: str = ""
    pace_mean_sha256: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_local_a1_usd_override(self)
        _apply_local_terrain_usd_override(self)
        _apply_fitted_pace_actuator(self)
        _apply_upside_down_fixed_air_fixture(self)


@configclass
class GogoA1PaceFlatEnvCfg_PLAY(GogoA1PaceFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class GogoA1PacePaperFlatEnvCfg(GogoA1FlatStudentEnvCfg):
    """Upright flat walking with the fitted PACE actuator and paper reward scales."""

    pace_fixture_pose: str = "upright_flat_walk"
    pace_fit_manifest_path: str = ""
    pace_fit_manifest_sha256: str = ""
    pace_mean_path: str = ""
    pace_mean_sha256: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_local_a1_usd_override(self)
        _apply_local_terrain_usd_override(self)
        _apply_fitted_pace_actuator(self)
        _apply_paper_flat_walk_settings(self)


@configclass
class GogoA1PacePaperFlatEnvCfg_PLAY(GogoA1PacePaperFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
