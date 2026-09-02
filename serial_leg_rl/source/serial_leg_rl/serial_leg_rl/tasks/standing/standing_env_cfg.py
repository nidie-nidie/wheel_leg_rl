from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from serial_leg_rl.assets.local_terrain import prepare_local_flat_terrain_usd
from serial_leg_rl.assets.wheel_leg_robot import PASSIVE_JOINT_NAMES, WHEEL_LEG_ROBOT_CFG
from serial_leg_rl.tasks.standing.terrain_cfg import SERIAL_LEG_ROUGH_TERRAINS_CFG


@configclass
class StandingFlatSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=prepare_local_flat_terrain_usd().as_posix(),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot = WHEEL_LEG_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    left_wheel_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jwheel_left",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )
    right_wheel_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jwheel_right",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


@configclass
class StandingRoughSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SERIAL_LEG_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot = WHEEL_LEG_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    left_wheel_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jwheel_left",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )
    right_wheel_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jwheel_right",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


@configclass
class StandingEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 8.0
    action_space = 6

    history_length = 5
    single_observation_dim = 33
    observation_space = history_length * single_observation_dim
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: StandingFlatSceneCfg = StandingFlatSceneCfg(num_envs=2048, env_spacing=1.5, replicate_physics=True)

    robot_cfg = WHEEL_LEG_ROBOT_CFG
    leg_joint_names = ["jIJ", "jIO", "jAB", "jAG"]
    wheel_joint_names = ["jwheel_left", "jwheel_right"]
    passive_joint_names = PASSIVE_JOINT_NAMES

    target_base_height = 0.20
    min_base_height = 0.08
    max_base_height = 0.40
    max_tilt_rad = 0.65
    termination_grace_steps = 25

    action_scale_leg_pos = 0.45
    action_scale_wheel_vel = 25.0
    action_rate_limit = 0.35

    target_leg_length = 0.1973
    wheel_contact_force_threshold = 1.0
    reset_leg_joint_pos_noise = 0.0
    reset_leg_joint_vel_noise = 0.0
    reset_root_lin_vel_noise = 0.0
    reset_root_ang_vel_noise = 0.0

    reward_alive = 0.15
    reward_upright = 2.0
    reward_height = 1.0
    reward_leg_length = 0.8
    reward_wheel_contact = 0.5
    penalty_ang_vel_xy = 0.05
    penalty_joint_vel = 0.001
    penalty_wheel_vel = 0.0005
    penalty_leg_length_symmetry = 0.4
    penalty_leg_length_rate = 0.02
    penalty_wheel_air = 0.25
    penalty_action_rate = 0.02
    penalty_termination = 5.0

    max_obs_abs = 100.0
    max_reward_abs = 100.0
    max_root_ang_vel = 35.0
    max_joint_vel = 80.0
    max_root_lin_vel = 20.0
    max_ang_vel_penalty = 20.0
    max_joint_vel_penalty = 20.0
    max_wheel_vel_penalty = 20.0
    max_leg_length_rate_penalty = 20.0


@configclass
class StandingRoughEnvCfg(StandingEnvCfg):
    scene: StandingRoughSceneCfg = StandingRoughSceneCfg(num_envs=2048, env_spacing=1.5, replicate_physics=True)
