from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.rough_env_cfg import UnitreeA1RoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg

from . import mdp as gogo_mdp
from .stair_terrain import MeshForwardStairsTerrainCfg


def _enable_mjcf_importer_extension() -> None:
    try:
        import omni.kit.app
    except ModuleNotFoundError:
        return

    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled("isaacsim.asset.importer.mjcf"):
        manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)


_enable_mjcf_importer_extension()


REPO_ROOT = Path(__file__).resolve().parents[2]

A1_DEPLOY_ACTION_SCALE = {
    ".*_hip_joint": 0.25,
    ".*_thigh_joint": 0.45,
    ".*_calf_joint": 0.45,
}

A1_DREAMWAQ_ACTION_SCALE = 0.25

DOG_MJCF_PATH = REPO_ROOT / "assets/dog/dog.xml"
_MJCF_FILE_CFG = getattr(sim_utils, "MjcfFileCfg", None)
DOG_NOMINAL_TOTAL_MASS_KG = 0.754
DOG_ONE_KG_TOTAL_MASS_KG = 1.000
DOG_ONE_KG_BASE_MASS_DELTA = DOG_ONE_KG_TOTAL_MASS_KG - DOG_NOMINAL_TOTAL_MASS_KG
DOG_SERVO_TORQUE_LIMIT_NM = 0.686466
DOG_TORQUE_SAFE_TORQUE_LIMIT_NM = 0.60
DOG_SERVO_VELOCITY_LIMIT_RAD_S = 7.0
DOG_SERVO_STIFFNESS = 12.0
DOG_SERVO_DAMPING = 0.8
DOG_SERVO_FRICTION = 0.010
DOG_SERVO_DYNAMIC_FRICTION = 0.008
DOG_SERVO_VISCOUS_FRICTION = 0.002
DOG_SERVO_ARMATURE = 0.00002
DOG_JOINT_ORDER = [
    "fr_hip_abduction",
    "fr_hip_pitch",
    "fr_knee",
    "fl_hip_abduction",
    "fl_hip_pitch",
    "fl_knee",
    "rr_hip_abduction",
    "rr_hip_pitch",
    "rr_knee",
    "rl_hip_abduction",
    "rl_hip_pitch",
    "rl_knee",
]
DOG_FOOT_PHASE_BODY_ORDER = ["fl_foot", "rr_foot", "fr_foot", "rl_foot"]
DOG_TROT_PHASE_PARAMS = {
    "f_min": 0.9,
    "f_max": 2.2,
    "gain": 0.55,
}
DOG_STABLE_STEP_TROT_PHASE_PARAMS = {
    "f_min": 1.65,
    "f_max": 2.90,
    "gain": 1.15,
}
DOG_TORQUE_SAFE_TROT_PHASE_PARAMS = {
    "f_min": 1.35,
    "f_max": 2.35,
    "gain": 0.75,
}
DOG_TORQUE_SAFE_STAGE3_TROT_PHASE_PARAMS = {
    "f_min": 1.10,
    "f_max": 1.95,
    "gain": 0.45,
}
DOG_ACTION_SCALE = {
    ".*_hip_abduction": 0.14,
    ".*_hip_pitch": 0.30,
    ".*_knee": 0.48,
}
DOG_STABLE_STEP_ACTION_SCALE = {
    ".*_hip_abduction": 0.105,
    ".*_hip_pitch": 0.225,
    ".*_knee": 0.360,
}
DOG_TORQUE_SAFE_ACTION_SCALE = {
    ".*_hip_abduction": 0.085,
    ".*_hip_pitch": 0.180,
    ".*_knee": 0.260,
}
DOG_TORQUE_SAFE_STAGE2_ACTION_SCALE = {
    ".*_hip_abduction": 0.095,
    ".*_hip_pitch": 0.205,
    ".*_knee": 0.315,
}
DOG_TORQUE_SAFE_STAGE3_ACTION_SCALE = {
    ".*_hip_abduction": 0.075,
    ".*_hip_pitch": 0.160,
    ".*_knee": 0.235,
}
DOG_TORQUE_SAFE_STAGE3_STIFFNESS = 8.0
DOG_TORQUE_SAFE_STAGE3_DAMPING = 0.12
DOG_DEFAULT_JOINT_POS = {
    ".*_hip_abduction": 0.0,
    ".*_hip_pitch": 0.45,
    ".*_knee": -0.90,
}

DOG_CFG = ArticulationCfg(
    articulation_root_prim_path="/base/base",
    spawn=(
        _MJCF_FILE_CFG(
            asset_path=str(DOG_MJCF_PATH),
            fix_base=False,
            import_sites=True,
            self_collision=False,
            make_instanceable=True,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=50.0,
                max_angular_velocity=50.0,
                max_depenetration_velocity=0.35,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        )
        if _MJCF_FILE_CFG is not None
        else None
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.138),
        joint_pos=DOG_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "servos": ImplicitActuatorCfg(
            joint_names_expr=DOG_JOINT_ORDER,
            effort_limit_sim=DOG_SERVO_TORQUE_LIMIT_NM,
            velocity_limit_sim=DOG_SERVO_VELOCITY_LIMIT_RAD_S,
            stiffness=DOG_SERVO_STIFFNESS,
            damping=DOG_SERVO_DAMPING,
            armature=DOG_SERVO_ARMATURE,
            friction=DOG_SERVO_FRICTION,
            dynamic_friction=DOG_SERVO_DYNAMIC_FRICTION,
            viscous_friction=DOG_SERVO_VISCOUS_FRICTION,
        ),
    },
)
if _MJCF_FILE_CFG is None:
    DOG_CFG = None

STAIR_NUM_STEPS = 1
STAIR_STEP_HEIGHT_RANGE = (0.045, 0.32)
STAIR_STEP_DEPTH = 0.52
STAIR_APPROACH_LENGTH = 1.50
STAIR_TOP_PLATFORM_LENGTH = 1.35
STAIR_FIRST_RISER_X = 0.5 * STAIR_APPROACH_LENGTH
STAIR_TOP_PLATFORM_X = STAIR_FIRST_RISER_X + STAIR_NUM_STEPS * STAIR_STEP_DEPTH
STAIR_TARGET_X = STAIR_FIRST_RISER_X + 0.44 * STAIR_STEP_DEPTH
STAIR_TARGET_Z = STAIR_NUM_STEPS * STAIR_STEP_HEIGHT_RANGE[0] + 0.24
STAIR_TARGET_BASE_CLEARANCE = 0.24

FORWARD_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(3.8, 3.0),
    border_width=0.4,
    border_height=1.0,
    num_rows=8,
    num_cols=4,
    color_scheme="height",
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=None,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "forward_stairs": MeshForwardStairsTerrainCfg(
            proportion=1.0,
            step_height_range=STAIR_STEP_HEIGHT_RANGE,
            step_depth=STAIR_STEP_DEPTH,
            num_steps=STAIR_NUM_STEPS,
            approach_length=STAIR_APPROACH_LENGTH,
            top_platform_length=STAIR_TOP_PLATFORM_LENGTH,
            width=2.0,
        )
    },
)


def _apply_deploy_action_scale(cfg) -> None:
    cfg.actions.joint_pos.scale = A1_DEPLOY_ACTION_SCALE


def _apply_dreamwaq_action_scale(cfg) -> None:
    cfg.actions.joint_pos.scale = A1_DREAMWAQ_ACTION_SCALE


def _apply_public_a1_pd_gains(cfg) -> None:
    actuator = cfg.scene.robot.actuators.get("base_legs")
    if actuator is None:
        return
    actuator.stiffness = 20.0
    actuator.damping = 0.5


def _apply_public_hf_stairs(cfg) -> None:
    terrain_cfg = cfg.scene.terrain.terrain_generator
    if terrain_cfg is None:
        return

    terrains = terrain_cfg.sub_terrains
    if "pyramid_stairs" in terrains:
        terrains["pyramid_stairs"] = terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=terrains["pyramid_stairs"].proportion,
            step_height_range=(0.05, 0.23),
            step_width=0.31,
            platform_width=3.0,
            border_width=0.0,
            horizontal_scale=terrain_cfg.horizontal_scale,
            vertical_scale=terrain_cfg.vertical_scale,
            slope_threshold=terrain_cfg.slope_threshold,
        )
    if "pyramid_stairs_inv" in terrains:
        terrains["pyramid_stairs_inv"] = terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=terrains["pyramid_stairs_inv"].proportion,
            step_height_range=(0.05, 0.23),
            step_width=0.31,
            platform_width=3.0,
            border_width=0.0,
            horizontal_scale=terrain_cfg.horizontal_scale,
            vertical_scale=terrain_cfg.vertical_scale,
            slope_threshold=terrain_cfg.slope_threshold,
        )


def _apply_public_upstairs_biased_mix(cfg) -> None:
    terrain_cfg = cfg.scene.terrain.terrain_generator
    if terrain_cfg is None:
        return

    # Bridge the gap between the easy upstairs slice and high upstairs failures
    # without changing DreamWaQ rewards, observations, or terrain geometry.
    terrain_cfg.difficulty_range = (0.25, 0.70)
    cfg.scene.terrain.max_init_terrain_level = 5

    terrains = terrain_cfg.sub_terrains
    upstairs_biased_proportions = {
        "pyramid_stairs_inv": 0.55,
        "pyramid_stairs": 0.20,
        "boxes": 0.10,
        "random_rough": 0.10,
        "hf_pyramid_slope": 0.025,
        "hf_pyramid_slope_inv": 0.025,
    }
    for name, proportion in upstairs_biased_proportions.items():
        if name in terrains:
            terrains[name].proportion = proportion


def _apply_public_hf_stairs_low_up_mix(cfg) -> None:
    terrain_cfg = cfg.scene.terrain.terrain_generator
    if terrain_cfg is None:
        return

    # Gated curriculum probe: keep the public mixed terrain family, but expose
    # the policy to a narrower upstairs band before returning to high steps.
    terrain_cfg.difficulty_range = (0.25, 0.50)
    cfg.scene.terrain.max_init_terrain_level = 5

    terrains = terrain_cfg.sub_terrains
    low_up_proportions = {
        "pyramid_stairs_inv": 0.45,
        "pyramid_stairs": 0.25,
        "boxes": 0.10,
        "random_rough": 0.10,
        "hf_pyramid_slope": 0.05,
        "hf_pyramid_slope_inv": 0.05,
    }
    for name, proportion in low_up_proportions.items():
        if name in terrains:
            terrains[name].proportion = proportion


def _apply_play_settings(cfg) -> None:
    cfg.scene.num_envs = 50
    cfg.scene.env_spacing = 2.5
    cfg.commands.base_velocity.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    cfg.observations.critic.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


@configclass
class GogoA1ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase = ObsTerm(func=gogo_mdp.trot_phase, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class GogoA1StairObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_lin_acc = ObsTerm(func=gogo_mdp.base_lin_acc, noise=Unoise(n_min=-0.5, n_max=0.5))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        motor_power = ObsTerm(func=gogo_mdp.motor_power_percent, noise=Unoise(n_min=-0.02, n_max=0.02))
        actions = ObsTerm(func=gogo_mdp.last_action_with_reset_seed)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_lin_acc = ObsTerm(func=gogo_mdp.base_lin_acc, noise=Unoise(n_min=-0.5, n_max=0.5))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        motor_power = ObsTerm(func=gogo_mdp.motor_power_percent, noise=Unoise(n_min=-0.02, n_max=0.02))
        actions = ObsTerm(func=gogo_mdp.last_action_with_reset_seed)
        stair_goal = ObsTerm(
            func=gogo_mdp.stair_goal_state,
            params={
                "target_x": STAIR_TARGET_X,
                "target_z": STAIR_TARGET_Z,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "num_steps": STAIR_NUM_STEPS,
                "base_clearance": STAIR_TARGET_BASE_CLEARANCE,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class GogoA1StairApproachGoalObservationsCfg(GogoA1ObservationsCfg):
    @configclass
    class PolicyCfg(GogoA1ObservationsCfg.PolicyCfg):
        handoff_state = ObsTerm(
            func=gogo_mdp.stair_handoff_state,
            params={"target_x": 0.70, "x_scale": 1.0, "y_scale": 0.5, "yaw_scale": 1.0},
        )

    @configclass
    class CriticCfg(GogoA1ObservationsCfg.CriticCfg):
        handoff_state = ObsTerm(
            func=gogo_mdp.stair_handoff_state,
            params={"target_x": 0.70, "x_scale": 1.0, "y_scale": 0.5, "yaw_scale": 1.0},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class GogoA1DreamWaQObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class GogoA1DreamWaQPaperLikeFullStateObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        contact_forces = ObsTerm(
            func=gogo_mdp.normalized_contact_forces,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
                "force_range": (0.0, 50.0),
            },
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=5.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class GogoA1FlatStudentEnvCfg(UnitreeA1FlatEnvCfg):
    observations: GogoA1ObservationsCfg = GogoA1ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations = GogoA1ObservationsCfg()

        foot_names = ["FL_foot", "RR_foot", "FR_foot", "RL_foot"]

        def foot_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", body_names=foot_names, preserve_order=True)

        def foot_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names=foot_names, preserve_order=True)

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 1.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.55, 0.55)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        self.rewards.track_lin_vel_xy_exp.weight = 1.6
        self.rewards.track_ang_vel_z_exp.weight = 0.55
        self.rewards.action_rate_l2.weight = -0.008
        self.rewards.dof_acc_l2.weight = -5.0e-7
        self.rewards.dof_torques_l2.weight = -1.4e-4

        self.rewards.feet_air_time.params["threshold"] = 0.09
        self.rewards.feet_air_time.weight = 0.02

        self.rewards.trot_phase_contact = RewTerm(
            func=gogo_mdp.trot_phase_contact_reward,
            weight=0.2,
            params={"asset_cfg": foot_asset_cfg(), "sensor_cfg": foot_sensor_cfg(), "contact_threshold": 0.5},
        )
        self.rewards.swing_foot_clearance = RewTerm(
            func=gogo_mdp.swing_foot_clearance_reward,
            weight=0.08,
            params={"asset_cfg": foot_asset_cfg(), "target_height": 0.045, "std": 0.25},
        )
        self.rewards.swing_foot_height = RewTerm(
            func=gogo_mdp.swing_foot_height_penalty,
            weight=-0.35,
            params={"asset_cfg": foot_asset_cfg(), "max_height": 0.075},
        )
        self.rewards.swing_foot_drag = RewTerm(
            func=gogo_mdp.swing_foot_drag_penalty,
            weight=-0.25,
            params={
                "asset_cfg": foot_asset_cfg(),
                "sensor_cfg": foot_sensor_cfg(),
                "contact_threshold": 0.5,
                "clearance": 0.04,
            },
        )
        self.rewards.stance_foot_slip = RewTerm(
            func=gogo_mdp.stance_foot_slip_penalty,
            weight=-0.25,
            params={"asset_cfg": foot_asset_cfg(), "sensor_cfg": foot_sensor_cfg(), "contact_threshold": 0.5},
        )
        self.rewards.trot_pair_timing = RewTerm(
            func=gogo_mdp.trot_pair_timing_reward,
            weight=0.03,
            params={"sensor_cfg": foot_sensor_cfg(), "contact_threshold": 0.5},
        )
        self.rewards.stand_still_joint_deviation = RewTerm(
            func=gogo_mdp.stand_still_joint_deviation_penalty,
            weight=-0.6,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.stand_still_action = RewTerm(
            func=gogo_mdp.stand_still_action_penalty,
            weight=-0.05,
        )
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-1.0,
            params={"target_height": 0.405, "asset_cfg": SceneEntityCfg("robot")},
        )


class GogoA1FlatStudentEnvCfg_PLAY(GogoA1FlatStudentEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class GogoA1FlatFastElegantEnvCfg(GogoA1FlatStudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_deploy_action_scale(self)

        self.commands.base_velocity.rel_standing_envs = 0.08
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.55, 0.55)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        self.rewards.track_lin_vel_xy_exp.weight = 1.8
        self.rewards.action_rate_l2.weight = -0.008
        self.rewards.dof_torques_l2.weight = -1.0e-4

        self.rewards.trot_phase_contact.weight = 0.15
        self.rewards.swing_foot_clearance.params["target_height"] = 0.055
        self.rewards.swing_foot_height.weight = -0.2
        self.rewards.swing_foot_height.params["max_height"] = 0.09
        self.rewards.swing_foot_drag.params["clearance"] = 0.045


class GogoA1FlatFastElegantEnvCfg_PLAY(GogoA1FlatFastElegantEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class GogoA1FlatFastRobustEnvCfg(GogoA1FlatFastElegantEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.04
        self.commands.base_velocity.resampling_time_range = (2.5, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.25, 2.35)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.75, 0.75)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.15, 1.15)

        self.events.physics_material.params["static_friction_range"] = (0.65, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.45, 1.0)
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.5, 3.5)
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 10.0),
            params={"velocity_range": {"x": (-0.35, 0.35), "y": (-0.30, 0.30), "yaw": (-0.30, 0.30)}},
        )

        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 0.7
        self.rewards.flat_orientation_l2.weight = -3.5
        self.rewards.lin_vel_z_l2.weight = -2.6
        self.rewards.ang_vel_xy_l2.weight = -0.14
        self.rewards.base_height_l2.weight = -1.2
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.action_rate_l2.weight = -0.010
        self.rewards.dof_acc_l2.weight = -7.5e-7
        self.rewards.dof_torques_l2.weight = -1.2e-4
        self.rewards.trot_phase_contact.weight = 0.12
        self.rewards.swing_foot_clearance.weight = 0.07
        self.rewards.swing_foot_drag.weight = -0.30
        self.rewards.stance_foot_slip.weight = -0.35


class GogoA1FlatFastRobustEnvCfg_PLAY(GogoA1FlatFastRobustEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class GogoDogServoFlatFastElegantEnvCfg(GogoA1FlatFastElegantEnvCfg):
    """Flat fast/elegant A1 gait task retargeted to the 7 kgf*cm servo dog MJCF."""

    def __post_init__(self):
        super().__post_init__()

        if DOG_CFG is None:
            raise RuntimeError(
                "The current IsaacLab runtime has no MJCF spawn configuration; "
                "the servo-dog task requires a converted USD asset."
            )
        self.scene.robot = DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 1.25
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/.*",
            history_length=3,
            track_air_time=True,
        )
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        self.actions.joint_pos.joint_names = DOG_JOINT_ORDER
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = DOG_ACTION_SCALE
        self.actions.joint_pos.use_default_offset = True
        self.actions.joint_pos.clip = {
            ".*_hip_abduction": (-0.34, 0.34),
            ".*_hip_pitch": (0.00, 0.86),
            ".*_knee": (-1.60, -0.40),
        }

        def foot_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", body_names=DOG_FOOT_PHASE_BODY_ORDER, preserve_order=True)

        def foot_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names=DOG_FOOT_PHASE_BODY_ORDER, preserve_order=True)

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.22
        self.commands.base_velocity.resampling_time_range = (4.0, 7.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.16, 0.65)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.80, 0.80)

        self.observations.policy.gait_phase.params.update(DOG_TROT_PHASE_PARAMS)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.03, n_max=0.03)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.12, n_max=0.12)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.006, n_max=0.006)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.60, n_max=0.60)
        self.observations.critic.projected_gravity.noise = Unoise(n_min=-0.03, n_max=0.03)
        self.observations.critic.base_ang_vel.noise = Unoise(n_min=-0.12, n_max=0.12)
        self.observations.critic.joint_pos.noise = Unoise(n_min=-0.006, n_max=0.006)
        self.observations.critic.joint_vel.noise = Unoise(n_min=-0.60, n_max=0.60)

        self.events.physics_material.params["static_friction_range"] = (0.34, 0.58)
        self.events.physics_material.params["dynamic_friction_range"] = (0.30, 0.52)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.03, 0.06)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["force_range"] = (-0.25, 0.25)
        self.events.base_external_force_torque.params["torque_range"] = (-0.015, 0.015)
        self.events.reset_robot_joints.params["position_range"] = (0.985, 1.015)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.01, 0.01)
        self.events.push_robot = None
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_ORDER, preserve_order=True),
                "stiffness_distribution_params": (0.95, 1.08),
                "damping_distribution_params": (0.95, 1.20),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.joint_parameters = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_ORDER, preserve_order=True),
                "friction_distribution_params": (0.80, 1.25),
                "armature_distribution_params": (0.90, 1.20),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        self.rewards.is_alive = RewTerm(func=mdp.is_alive, weight=0.08)
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-4.0)
        self.rewards.track_lin_vel_xy_exp.weight = 1.65
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.24
        self.rewards.track_ang_vel_z_exp.weight = 0.65
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.28
        self.rewards.flat_orientation_l2.weight = -1.80
        self.rewards.pitch_abs_l1 = RewTerm(
            func=gogo_mdp.pitch_abs_l1,
            weight=-0.45,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.lin_vel_z_l2.weight = -1.1
        self.rewards.ang_vel_xy_l2.weight = -0.08
        self.rewards.base_height_l2.weight = -0.70
        self.rewards.base_height_l2.params["target_height"] = 0.138
        self.rewards.action_rate_l2.weight = -0.026
        self.rewards.dof_acc_l2.weight = -1.8e-6
        self.rewards.dof_torques_l2.weight = -0.0018
        self.rewards.joint_vel_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.0032,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_ORDER, preserve_order=True)},
        )
        self.rewards.action_smoothness_l2 = RewTerm(func=gogo_mdp.action_smoothness_l2, weight=-0.035)
        self.rewards.joint_deviation_l2 = RewTerm(
            func=gogo_mdp.joint_deviation_l2,
            weight=-0.24,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_ORDER, preserve_order=True),
                "moving_scale": 1.0,
                "standing_scale": 5.0,
                "joint_weights": (1.0, 1.6, 1.3, 1.0, 1.6, 1.3, 1.0, 1.15, 1.0, 1.0, 1.15, 1.0),
            },
        )
        self.rewards.backward_front_leg_deviation = RewTerm(
            func=gogo_mdp.commanded_backward_joint_deviation_penalty,
            weight=-0.45,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["fr_hip_pitch", "fr_knee", "fl_hip_pitch", "fl_knee"],
                    preserve_order=True,
                ),
                "min_backward_command": 0.08,
                "allowances": (0.18, 0.25, 0.18, 0.25),
            },
        )
        self.rewards.feet_air_time = None
        self.rewards.trot_phase_contact = RewTerm(
            func=gogo_mdp.trot_phase_contact_reward,
            weight=0.55,
            params={
                "asset_cfg": foot_asset_cfg(),
                "sensor_cfg": foot_sensor_cfg(),
                "contact_threshold": 0.40,
                **DOG_TROT_PHASE_PARAMS,
            },
        )
        self.rewards.feet_air_time = RewTerm(
            func=gogo_mdp.commanded_feet_air_time_reward,
            weight=0.28,
            params={
                "sensor_cfg": foot_sensor_cfg(),
                "command_name": "base_velocity",
                "command_threshold": 0.05,
                "threshold": 0.035,
                "max_air_time": 0.18,
            },
        )
        self.rewards.swing_foot_clearance.params["asset_cfg"] = foot_asset_cfg()
        self.rewards.swing_foot_clearance.params["target_height"] = 0.046
        self.rewards.swing_foot_clearance.params["std"] = 0.11
        self.rewards.swing_foot_clearance.params.update(DOG_TROT_PHASE_PARAMS)
        self.rewards.swing_foot_clearance.weight = 0.65
        self.rewards.swing_foot_lift = RewTerm(
            func=gogo_mdp.phase_swing_foot_lift_reward,
            weight=1.75,
            params={
                "asset_cfg": foot_asset_cfg(),
                "min_height": 0.026,
                "target_height": 0.058,
                "foot_radius": 0.012,
                "velocity_scale": 5.0,
                **DOG_TROT_PHASE_PARAMS,
            },
        )
        self.rewards.moving_foot_low_height = RewTerm(
            func=gogo_mdp.phase_swing_foot_low_height_penalty,
            weight=-1.40,
            params={
                "asset_cfg": foot_asset_cfg(),
                "foot_radius": 0.012,
                "clearance": 0.026,
                "velocity_scale": 7.0,
                **DOG_TROT_PHASE_PARAMS,
            },
        )
        self.rewards.swing_foot_low_pose = RewTerm(
            func=gogo_mdp.phase_swing_foot_low_height_pose_penalty,
            weight=-1.10,
            params={
                "asset_cfg": foot_asset_cfg(),
                "foot_radius": 0.012,
                "clearance": 0.024,
                **DOG_TROT_PHASE_PARAMS,
            },
        )
        self.rewards.global_moving_low_foot = RewTerm(
            func=gogo_mdp.moving_foot_low_height_penalty,
            weight=-0.75,
            params={
                "asset_cfg": foot_asset_cfg(),
                "foot_radius": 0.012,
                "clearance": 0.018,
                "velocity_scale": 8.0,
            },
        )
        self.rewards.yaw_backward_low_foot = RewTerm(
            func=gogo_mdp.yaw_or_backward_low_foot_penalty,
            weight=-2.20,
            params={
                "asset_cfg": foot_asset_cfg(),
                "min_backward_command": 0.04,
                "min_yaw_command_abs": 0.10,
                "foot_radius": 0.012,
                "clearance": 0.024,
                "velocity_scale": 8.0,
            },
        )
        self.rewards.yaw_backward_swing_lift = RewTerm(
            func=gogo_mdp.yaw_or_backward_swing_foot_lift_reward,
            weight=0.95,
            params={
                "asset_cfg": foot_asset_cfg(),
                "min_backward_command": 0.04,
                "min_yaw_command_abs": 0.10,
                "foot_radius": 0.012,
                "min_height": 0.026,
                "target_height": 0.060,
                **DOG_TROT_PHASE_PARAMS,
            },
        )
        self.rewards.yaw_backward_swing_low_pose = RewTerm(
            func=gogo_mdp.yaw_or_backward_swing_foot_low_height_penalty,
            weight=-2.80,
            params={
                "asset_cfg": foot_asset_cfg(),
                "min_backward_command": 0.04,
                "min_yaw_command_abs": 0.10,
                "foot_radius": 0.012,
                "clearance": 0.030,
                **DOG_TROT_PHASE_PARAMS,
            },
        )
        self.rewards.swing_foot_height.params["asset_cfg"] = foot_asset_cfg()
        self.rewards.swing_foot_height.params["max_height"] = 0.080
        self.rewards.swing_foot_height.params.update(DOG_TROT_PHASE_PARAMS)
        self.rewards.swing_foot_height.weight = -0.035
        self.rewards.swing_foot_drag = RewTerm(
            func=gogo_mdp.swing_foot_drag_penalty,
            weight=-1.95,
            params={
                "asset_cfg": foot_asset_cfg(),
                "sensor_cfg": foot_sensor_cfg(),
                "contact_threshold": 0.40,
                "clearance": 0.040,
                "velocity_scale": 7.0,
                **DOG_TROT_PHASE_PARAMS,
            },
        )
        self.rewards.stance_foot_slip = RewTerm(
            func=gogo_mdp.stance_foot_slip_penalty,
            weight=-0.62,
            params={
                "asset_cfg": foot_asset_cfg(),
                "sensor_cfg": foot_sensor_cfg(),
                "contact_threshold": 0.40,
                **DOG_TROT_PHASE_PARAMS,
            },
        )
        self.rewards.contact_foot_slide = RewTerm(
            func=gogo_mdp.contact_foot_slide_penalty,
            weight=-0.75,
            params={
                "asset_cfg": foot_asset_cfg(),
                "sensor_cfg": foot_sensor_cfg(),
                "contact_threshold": 0.40,
            },
        )
        self.rewards.trot_pair_timing = None
        self.rewards.stand_still_joint_deviation.weight = -2.10
        self.rewards.stand_still_action.weight = -1.00
        self.rewards.stand_still_joint_vel = RewTerm(
            func=gogo_mdp.stand_still_joint_vel_penalty,
            weight=-0.040,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_ORDER, preserve_order=True)},
        )
        self.rewards.stand_still_foot_vel = RewTerm(
            func=gogo_mdp.stand_still_foot_vel_penalty,
            weight=-0.65,
            params={"asset_cfg": foot_asset_cfg()},
        )
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.commanded_planar_velocity = RewTerm(
            func=gogo_mdp.gait_gated_commanded_planar_velocity_reward,
            weight=0.38,
            params={
                "asset_cfg": foot_asset_cfg(),
                "command_name": "base_velocity",
                "min_command_norm": 0.04,
                "max_speed": 0.65,
                "foot_radius": 0.012,
                "clearance": 0.024,
                "min_gate": 0.28,
                **DOG_TROT_PHASE_PARAMS,
            },
        )
        self.rewards.commanded_yaw_velocity = RewTerm(
            func=gogo_mdp.gait_gated_commanded_yaw_velocity_reward,
            weight=0.16,
            params={
                "asset_cfg": foot_asset_cfg(),
                "command_name": "base_velocity",
                "min_command_abs": 0.08,
                "max_yaw_rate": 0.85,
                "foot_radius": 0.012,
                "clearance": 0.024,
                "min_gate": 0.28,
                **DOG_TROT_PHASE_PARAMS,
            },
        )
        self.rewards.low_clearance_velocity_progress = RewTerm(
            func=gogo_mdp.low_clearance_velocity_progress_penalty,
            weight=-0.42,
            params={
                "asset_cfg": foot_asset_cfg(),
                "command_name": "base_velocity",
                "min_planar_command_norm": 0.04,
                "max_speed": 0.65,
                "min_yaw_command_abs": 0.08,
                "max_yaw_rate": 0.85,
                "foot_radius": 0.012,
                "clearance": 0.026,
                "velocity_scale": 8.0,
                "foot_weights": (1.0, 1.0, 1.1, 1.25),
                **DOG_TROT_PHASE_PARAMS,
            },
        )
        self.rewards.commanded_velocity_stuck = RewTerm(
            func=gogo_mdp.commanded_velocity_stuck_penalty,
            weight=-0.45,
            params={
                "command_name": "base_velocity",
                "min_planar_command_norm": 0.06,
                "min_planar_speed": 0.025,
                "min_yaw_command_abs": 0.12,
                "min_yaw_rate": 0.045,
            },
        )

        self.terminations.base_contact = None
        self.terminations.base_pose_fall = DoneTerm(
            func=gogo_mdp.base_pose_fall,
            params={
                "min_base_height": 0.070,
                "max_pitch_projected": 0.88,
                "max_roll_projected": 0.78,
                "grace_time_s": 0.25,
            },
        )


class GogoDogServoFlatFastElegantEnvCfg_PLAY(GogoDogServoFlatFastElegantEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.25
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GogoDogServoFlatFastAgileEnvCfg(GogoDogServoFlatFastElegantEnvCfg):
    """Main flat-ground servo-dog task tuned for fast, clean footed locomotion."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.08
        self.commands.base_velocity.resampling_time_range = (4.0, 7.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.06, 0.72)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.14, 0.14)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.45, 0.45)

        self.rewards.is_alive.weight = 0.16
        self.rewards.termination_penalty.weight = -12.0
        self.rewards.track_lin_vel_xy_exp.weight = 2.45
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.26
        self.rewards.track_ang_vel_z_exp.weight = 0.52
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.30
        self.rewards.flat_orientation_l2.weight = -3.10
        self.rewards.pitch_abs_l1.weight = -0.82
        self.rewards.lin_vel_z_l2.weight = -1.80
        self.rewards.ang_vel_xy_l2.weight = -0.26
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.022
        self.rewards.dof_acc_l2.weight = -1.6e-6
        self.rewards.dof_torques_l2.weight = -0.0017
        self.rewards.joint_vel_l2.weight = -0.0034
        self.rewards.action_smoothness_l2.weight = -0.026
        self.rewards.joint_deviation_l2.weight = -0.20
        self.rewards.joint_deviation_l2.params["moving_scale"] = 0.95
        self.rewards.backward_front_leg_deviation.weight = -0.28

        self.rewards.trot_phase_contact.weight = 0.38
        self.rewards.feet_air_time.weight = 0.30
        self.rewards.feet_air_time.params["threshold"] = 0.032
        self.rewards.swing_foot_clearance.weight = 0.72
        self.rewards.swing_foot_clearance.params["target_height"] = 0.048
        self.rewards.swing_foot_clearance.params["std"] = 0.12
        self.rewards.swing_foot_lift.weight = 2.25
        self.rewards.swing_foot_lift.params["min_height"] = 0.016
        self.rewards.swing_foot_lift.params["target_height"] = 0.054
        self.rewards.swing_foot_lift.params["velocity_scale"] = 2.2
        self.rewards.swing_foot_up_velocity = RewTerm(
            func=gogo_mdp.phase_swing_foot_up_velocity_reward,
            weight=0.95,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=DOG_FOOT_PHASE_BODY_ORDER, preserve_order=True),
                "target_up_speed": 0.16,
                "foot_radius": 0.012,
                "max_clearance": 0.052,
                **DOG_TROT_PHASE_PARAMS,
            },
        )
        self.rewards.moving_foot_low_height.weight = -1.05
        self.rewards.moving_foot_low_height.params["clearance"] = 0.026
        self.rewards.swing_foot_low_pose.weight = -0.75
        self.rewards.swing_foot_low_pose.params["clearance"] = 0.024
        self.rewards.global_moving_low_foot.weight = -0.70
        self.rewards.global_moving_low_foot.params["clearance"] = 0.018
        self.rewards.yaw_backward_low_foot.weight = -1.35
        self.rewards.yaw_backward_swing_lift.weight = 0.85
        self.rewards.yaw_backward_swing_low_pose.weight = -1.65
        self.rewards.swing_foot_drag.weight = -1.25
        self.rewards.swing_foot_drag.params["clearance"] = 0.034
        self.rewards.stance_foot_slip.weight = -0.58
        self.rewards.contact_foot_slide.weight = -0.70

        self.rewards.commanded_forward_velocity = RewTerm(
            func=gogo_mdp.commanded_forward_velocity_reward,
            weight=1.45,
            params={
                "command_name": "base_velocity",
                "min_command_x": 0.05,
                "max_speed": 0.72,
            },
        )
        self.rewards.commanded_planar_velocity.weight = 0.48
        self.rewards.commanded_planar_velocity.params["max_speed"] = 0.72
        self.rewards.commanded_planar_velocity.params["clearance"] = 0.024
        self.rewards.commanded_planar_velocity.params["min_gate"] = 0.48
        self.rewards.commanded_yaw_velocity.weight = 0.12
        self.rewards.commanded_yaw_velocity.params["max_yaw_rate"] = 0.55
        self.rewards.commanded_yaw_velocity.params["clearance"] = 0.026
        self.rewards.commanded_yaw_velocity.params["min_gate"] = 0.24
        self.rewards.low_clearance_velocity_progress.weight = -0.46
        self.rewards.low_clearance_velocity_progress.params["max_speed"] = 0.72
        self.rewards.low_clearance_velocity_progress.params["max_yaw_rate"] = 0.55
        self.rewards.low_clearance_velocity_progress.params["clearance"] = 0.027
        self.rewards.low_clearance_velocity_progress.params["foot_weights"] = (1.0, 1.0, 1.1, 1.2)
        self.rewards.commanded_velocity_stuck.weight = -0.45
        self.rewards.commanded_velocity_stuck.params["min_planar_speed"] = 0.035

        self.rewards.stand_still_joint_deviation.weight = -1.45
        self.rewards.stand_still_action.weight = -0.75
        self.rewards.stand_still_joint_vel.weight = -0.045
        self.rewards.stand_still_foot_vel.weight = -0.55

        self.events.base_external_force_torque.params["force_range"] = (-0.10, 0.10)
        self.events.base_external_force_torque.params["torque_range"] = (-0.006, 0.006)


class GogoDogServoFlatFastAgileEnvCfg_PLAY(GogoDogServoFlatFastAgileEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.25
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GogoDogServoFlatForwardWarmupEnvCfg(GogoDogServoFlatFastElegantEnvCfg):
    """Simplified forward-only warmup task for learning a clean servo-dog gait."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.015, n_max=0.015)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.05, n_max=0.05)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.003, n_max=0.003)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.25, n_max=0.25)
        self.observations.critic.projected_gravity.noise = Unoise(n_min=-0.015, n_max=0.015)
        self.observations.critic.base_ang_vel.noise = Unoise(n_min=-0.05, n_max=0.05)
        self.observations.critic.joint_pos.noise = Unoise(n_min=-0.003, n_max=0.003)
        self.observations.critic.joint_vel.noise = Unoise(n_min=-0.25, n_max=0.25)

        self.events.physics_material.params["static_friction_range"] = (0.40, 0.58)
        self.events.physics_material.params["dynamic_friction_range"] = (0.34, 0.52)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.01, 0.02)
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.98, 1.04)
        self.events.actuator_gains.params["damping_distribution_params"] = (0.98, 1.08)
        self.events.joint_parameters.params["friction_distribution_params"] = (0.92, 1.12)
        self.events.joint_parameters.params["armature_distribution_params"] = (0.95, 1.08)
        self.events.reset_robot_joints.params["position_range"] = (0.992, 1.008)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.004, 0.004)

        self.rewards.is_alive.weight = 0.12
        self.rewards.termination_penalty.weight = -8.0
        self.rewards.track_lin_vel_xy_exp.weight = 2.2
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.26
        self.rewards.track_ang_vel_z_exp.weight = 0.25
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.30
        self.rewards.flat_orientation_l2.weight = -1.45
        self.rewards.pitch_abs_l1.weight = -0.25
        self.rewards.lin_vel_z_l2.weight = -0.75
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.base_height_l2.weight = -0.25
        self.rewards.base_height_l2.params["target_height"] = 0.138
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.dof_acc_l2.weight = -8.0e-7
        self.rewards.dof_torques_l2.weight = -0.0011
        self.rewards.joint_vel_l2.weight = -0.0016
        self.rewards.action_smoothness_l2.weight = -0.012
        self.rewards.joint_deviation_l2.weight = -0.12
        self.rewards.joint_deviation_l2.params["moving_scale"] = 0.65
        self.rewards.joint_deviation_l2.params["standing_scale"] = 2.0

        self.rewards.backward_front_leg_deviation = None
        self.rewards.moving_foot_low_height = None
        self.rewards.swing_foot_low_pose = None
        self.rewards.global_moving_low_foot = None
        self.rewards.yaw_backward_low_foot = None
        self.rewards.yaw_backward_swing_lift = None
        self.rewards.yaw_backward_swing_low_pose = None
        self.rewards.commanded_planar_velocity = None
        self.rewards.commanded_yaw_velocity = None
        self.rewards.low_clearance_velocity_progress = None
        self.rewards.commanded_velocity_stuck = None
        self.rewards.stand_still_joint_deviation = None
        self.rewards.stand_still_action = None
        self.rewards.stand_still_joint_vel = None
        self.rewards.stand_still_foot_vel = None

        self.rewards.trot_phase_contact.weight = 0.22
        self.rewards.feet_air_time.weight = 0.12
        self.rewards.feet_air_time.params["threshold"] = 0.030
        self.rewards.swing_foot_clearance.weight = 0.36
        self.rewards.swing_foot_clearance.params["target_height"] = 0.044
        self.rewards.swing_foot_clearance.params["std"] = 0.13
        self.rewards.swing_foot_lift.weight = 0.85
        self.rewards.swing_foot_lift.params["min_height"] = 0.014
        self.rewards.swing_foot_lift.params["target_height"] = 0.050
        self.rewards.swing_foot_lift.params["velocity_scale"] = 2.0
        self.rewards.swing_foot_height.weight = -0.02
        self.rewards.swing_foot_height.params["max_height"] = 0.085
        self.rewards.swing_foot_drag.weight = -0.55
        self.rewards.swing_foot_drag.params["clearance"] = 0.032
        self.rewards.stance_foot_slip.weight = -0.25
        self.rewards.contact_foot_slide.weight = -0.25
        self.rewards.dof_pos_limits.weight = -0.60
        self.rewards.commanded_forward_velocity = RewTerm(
            func=gogo_mdp.commanded_forward_velocity_reward,
            weight=1.10,
            params={
                "command_name": "base_velocity",
                "min_command_x": 0.08,
                "max_speed": 0.45,
            },
        )


class GogoDogServoFlatForwardWarmupEnvCfg_PLAY(GogoDogServoFlatForwardWarmupEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.25
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GogoDogServoFlatStableStepEnvCfg(GogoDogServoFlatForwardWarmupEnvCfg):
    """Fine-tuning task that trades long stride for a steadier, hardware-friendlier gait."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.actions.joint_pos.scale = DOG_STABLE_STEP_ACTION_SCALE

        self.commands.base_velocity.resampling_time_range = (4.0, 7.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.16, 0.32)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.policy.gait_phase.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.010, n_max=0.010)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.035, n_max=0.035)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.18, n_max=0.18)
        self.observations.critic.projected_gravity.noise = Unoise(n_min=-0.010, n_max=0.010)
        self.observations.critic.base_ang_vel.noise = Unoise(n_min=-0.035, n_max=0.035)
        self.observations.critic.joint_vel.noise = Unoise(n_min=-0.18, n_max=0.18)

        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.99, 1.03)
        self.events.actuator_gains.params["damping_distribution_params"] = (1.00, 1.10)
        self.events.joint_parameters.params["friction_distribution_params"] = (0.96, 1.12)
        self.events.joint_parameters.params["armature_distribution_params"] = (0.98, 1.10)
        self.events.reset_robot_joints.params["position_range"] = (0.996, 1.004)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.003, 0.003)

        self.rewards.track_lin_vel_xy_exp.weight = 1.55
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.22
        self.rewards.track_ang_vel_z_exp.weight = 0.22
        self.rewards.flat_orientation_l2.weight = -2.35
        self.rewards.pitch_abs_l1.weight = -0.70
        self.rewards.lin_vel_z_l2.weight = -1.05
        self.rewards.ang_vel_xy_l2.weight = -0.16
        self.rewards.base_height_l2.weight = -0.35
        self.rewards.action_rate_l2.weight = -0.040
        self.rewards.dof_acc_l2.weight = -3.0e-6
        self.rewards.dof_torques_l2.weight = -0.0030
        self.rewards.joint_vel_l2.weight = -0.0060
        self.rewards.action_smoothness_l2.weight = -0.050
        self.rewards.joint_deviation_l2.weight = -0.32
        self.rewards.joint_deviation_l2.params["moving_scale"] = 1.20
        self.rewards.joint_deviation_l2.params["standing_scale"] = 2.50

        self.rewards.trot_phase_contact.weight = 0.34
        self.rewards.trot_phase_contact.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.rewards.feet_air_time.weight = 0.035
        self.rewards.feet_air_time.params["threshold"] = 0.020
        self.rewards.swing_foot_clearance.weight = 0.22
        self.rewards.swing_foot_clearance.params["target_height"] = 0.036
        self.rewards.swing_foot_clearance.params["std"] = 0.10
        self.rewards.swing_foot_clearance.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.rewards.swing_foot_lift.weight = 0.38
        self.rewards.swing_foot_lift.params["min_height"] = 0.014
        self.rewards.swing_foot_lift.params["target_height"] = 0.040
        self.rewards.swing_foot_lift.params["velocity_scale"] = 2.0
        self.rewards.swing_foot_lift.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.rewards.swing_foot_height.weight = -0.18
        self.rewards.swing_foot_height.params["max_height"] = 0.060
        self.rewards.swing_foot_height.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.rewards.swing_foot_drag.weight = -0.70
        self.rewards.swing_foot_drag.params["clearance"] = 0.024
        self.rewards.swing_foot_drag.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.rewards.stance_foot_slip.weight = -0.45
        self.rewards.stance_foot_slip.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.rewards.contact_foot_slide.weight = -0.42
        self.rewards.dof_pos_limits.weight = -0.75
        self.rewards.commanded_forward_velocity.weight = 0.48
        self.rewards.commanded_forward_velocity.params["min_command_x"] = 0.08
        self.rewards.commanded_forward_velocity.params["max_speed"] = 0.32


class GogoDogServoFlatStableStepEnvCfg_PLAY(GogoDogServoFlatStableStepEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.25
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GogoDogServoFlatTorqueSafeEnvCfg(GogoDogServoFlatStableStepEnvCfg):
    """Fine-tuning task that teaches the policy to stay below the servo torque budget."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot.actuators["servos"].effort_limit_sim = DOG_TORQUE_SAFE_TORQUE_LIMIT_NM
        self.actions.joint_pos.scale = DOG_STABLE_STEP_ACTION_SCALE

        self.commands.base_velocity.resampling_time_range = (5.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.14, 0.28)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.policy.gait_phase.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)

        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.985, 1.015)
        self.events.actuator_gains.params["damping_distribution_params"] = (1.02, 1.12)
        self.events.joint_parameters.params["friction_distribution_params"] = (0.98, 1.12)
        self.events.joint_parameters.params["armature_distribution_params"] = (1.00, 1.12)

        torque_asset_cfg = SceneEntityCfg("robot", joint_names=DOG_JOINT_ORDER, preserve_order=True)
        self.rewards.computed_torque_l2 = RewTerm(
            func=gogo_mdp.computed_torque_l2,
            weight=-0.018,
            params={
                "asset_cfg": torque_asset_cfg,
                "normalize_by_limit": DOG_TORQUE_SAFE_TORQUE_LIMIT_NM,
            },
        )
        self.rewards.computed_torque_limit = RewTerm(
            func=gogo_mdp.computed_torque_limit_penalty,
            weight=-0.45,
            params={
                "asset_cfg": torque_asset_cfg,
                "soft_limit": 0.55,
                "hard_limit": 0.90,
                "max_excess_ratio": 3.0,
                "joint_weights": (1.0, 1.25, 2.0, 1.0, 1.25, 2.0, 1.0, 1.25, 2.0, 1.0, 1.25, 2.0),
            },
        )

        self.rewards.track_lin_vel_xy_exp.weight = 1.35
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.24
        self.rewards.track_ang_vel_z_exp.weight = 0.18
        self.rewards.flat_orientation_l2.weight = -2.30
        self.rewards.pitch_abs_l1.weight = -0.72
        self.rewards.lin_vel_z_l2.weight = -1.05
        self.rewards.ang_vel_xy_l2.weight = -0.16
        self.rewards.base_height_l2.weight = -0.36
        self.rewards.action_rate_l2.weight = -0.048
        self.rewards.dof_acc_l2.weight = -3.8e-6
        self.rewards.dof_torques_l2.weight = -0.0038
        self.rewards.joint_vel_l2.weight = -0.0075
        self.rewards.action_smoothness_l2.weight = -0.060
        self.rewards.joint_deviation_l2.weight = -0.34
        self.rewards.joint_deviation_l2.params["moving_scale"] = 1.20
        self.rewards.joint_deviation_l2.params["standing_scale"] = 2.50

        self.rewards.trot_phase_contact.weight = 0.32
        self.rewards.trot_phase_contact.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.rewards.feet_air_time.weight = 0.015
        self.rewards.feet_air_time.params["threshold"] = 0.018
        self.rewards.swing_foot_clearance.weight = 0.18
        self.rewards.swing_foot_clearance.params["target_height"] = 0.034
        self.rewards.swing_foot_clearance.params["std"] = 0.12
        self.rewards.swing_foot_clearance.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.rewards.swing_foot_lift.weight = 0.30
        self.rewards.swing_foot_lift.params["min_height"] = 0.012
        self.rewards.swing_foot_lift.params["target_height"] = 0.038
        self.rewards.swing_foot_lift.params["velocity_scale"] = 1.8
        self.rewards.swing_foot_lift.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.rewards.swing_foot_height.weight = -0.20
        self.rewards.swing_foot_height.params["max_height"] = 0.056
        self.rewards.swing_foot_height.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.rewards.swing_foot_drag.weight = -0.65
        self.rewards.swing_foot_drag.params["clearance"] = 0.022
        self.rewards.swing_foot_drag.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.rewards.stance_foot_slip.weight = -0.48
        self.rewards.stance_foot_slip.params.update(DOG_STABLE_STEP_TROT_PHASE_PARAMS)
        self.rewards.contact_foot_slide.weight = -0.43
        self.rewards.dof_pos_limits.weight = -0.85
        self.rewards.commanded_forward_velocity.weight = 0.30
        self.rewards.commanded_forward_velocity.params["min_command_x"] = 0.07
        self.rewards.commanded_forward_velocity.params["max_speed"] = 0.28


class GogoDogServoFlatTorqueSafeEnvCfg_PLAY(GogoDogServoFlatTorqueSafeEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.25
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GogoDogServoFlatTorqueSafeStage2EnvCfg(GogoDogServoFlatTorqueSafeEnvCfg):
    """Second torque-safe stage that tightens computed torque after stable adaptation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.actions.joint_pos.scale = DOG_TORQUE_SAFE_STAGE2_ACTION_SCALE
        self.commands.base_velocity.ranges.lin_vel_x = (0.12, 0.25)

        self.rewards.computed_torque_l2.weight = -0.030
        self.rewards.computed_torque_limit.weight = -0.95
        self.rewards.computed_torque_limit.params["soft_limit"] = 0.48
        self.rewards.computed_torque_limit.params["hard_limit"] = DOG_TORQUE_SAFE_TORQUE_LIMIT_NM
        self.rewards.computed_torque_limit.params["max_excess_ratio"] = 3.5
        self.rewards.computed_torque_limit.params["joint_weights"] = (
            1.0,
            1.35,
            2.25,
            1.0,
            1.35,
            2.25,
            1.0,
            1.35,
            2.25,
            1.0,
            1.35,
            2.25,
        )

        self.rewards.track_lin_vel_xy_exp.weight = 1.18
        self.rewards.commanded_forward_velocity.weight = 0.22
        self.rewards.commanded_forward_velocity.params["max_speed"] = 0.25
        self.rewards.action_rate_l2.weight = -0.055
        self.rewards.dof_acc_l2.weight = -4.5e-6
        self.rewards.dof_torques_l2.weight = -0.0046
        self.rewards.joint_vel_l2.weight = -0.0095
        self.rewards.action_smoothness_l2.weight = -0.070


class GogoDogServoFlatTorqueSafeStage2EnvCfg_PLAY(GogoDogServoFlatTorqueSafeStage2EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.25
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GogoDogServoFlatTorqueSafeStage3EnvCfg(GogoDogServoFlatTorqueSafeStage2EnvCfg):
    """Torque-safe stage with lower PD gains so computed torque can fit a 0.6 Nm servo."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot.actuators["servos"].stiffness = DOG_TORQUE_SAFE_STAGE3_STIFFNESS
        self.scene.robot.actuators["servos"].damping = DOG_TORQUE_SAFE_STAGE3_DAMPING
        self.actions.joint_pos.scale = DOG_TORQUE_SAFE_STAGE3_ACTION_SCALE

        self.commands.base_velocity.resampling_time_range = (6.0, 9.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.20)
        self.observations.policy.gait_phase.params.update(DOG_TORQUE_SAFE_STAGE3_TROT_PHASE_PARAMS)

        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.98, 1.03)
        self.events.actuator_gains.params["damping_distribution_params"] = (0.95, 1.15)

        self.rewards.computed_torque_l2.weight = -0.045
        self.rewards.computed_torque_limit.weight = -1.40
        self.rewards.computed_torque_limit.params["soft_limit"] = 0.42
        self.rewards.computed_torque_limit.params["hard_limit"] = DOG_TORQUE_SAFE_TORQUE_LIMIT_NM
        self.rewards.computed_torque_limit.params["max_excess_ratio"] = 3.0

        self.rewards.track_lin_vel_xy_exp.weight = 0.95
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.26
        self.rewards.commanded_forward_velocity.weight = 0.12
        self.rewards.commanded_forward_velocity.params["max_speed"] = 0.20
        self.rewards.trot_phase_contact.weight = 0.24
        self.rewards.trot_phase_contact.params.update(DOG_TORQUE_SAFE_STAGE3_TROT_PHASE_PARAMS)
        self.rewards.swing_foot_clearance.weight = 0.10
        self.rewards.swing_foot_clearance.params["target_height"] = 0.026
        self.rewards.swing_foot_clearance.params.update(DOG_TORQUE_SAFE_STAGE3_TROT_PHASE_PARAMS)
        self.rewards.swing_foot_lift.weight = 0.14
        self.rewards.swing_foot_lift.params["target_height"] = 0.030
        self.rewards.swing_foot_lift.params["velocity_scale"] = 1.2
        self.rewards.swing_foot_lift.params.update(DOG_TORQUE_SAFE_STAGE3_TROT_PHASE_PARAMS)
        self.rewards.swing_foot_height.weight = -0.26
        self.rewards.swing_foot_height.params["max_height"] = 0.045
        self.rewards.swing_foot_height.params.update(DOG_TORQUE_SAFE_STAGE3_TROT_PHASE_PARAMS)
        self.rewards.swing_foot_drag.weight = -0.55
        self.rewards.swing_foot_drag.params["clearance"] = 0.018
        self.rewards.swing_foot_drag.params.update(DOG_TORQUE_SAFE_STAGE3_TROT_PHASE_PARAMS)
        self.rewards.stance_foot_slip.weight = -0.55
        self.rewards.stance_foot_slip.params.update(DOG_TORQUE_SAFE_STAGE3_TROT_PHASE_PARAMS)
        self.rewards.action_rate_l2.weight = -0.075
        self.rewards.dof_acc_l2.weight = -6.0e-6
        self.rewards.dof_torques_l2.weight = -0.0060
        self.rewards.joint_vel_l2.weight = -0.0180
        self.rewards.action_smoothness_l2.weight = -0.090


class GogoDogServoFlatTorqueSafeStage3EnvCfg_PLAY(GogoDogServoFlatTorqueSafeStage3EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.25
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GogoDogServoFlatTorqueSafeStage3OneKgEnvCfg(GogoDogServoFlatTorqueSafeStage3EnvCfg):
    """Stage 3 torque-safe task with total robot mass fixed to approximately 1 kg."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.add_base_mass.params["mass_distribution_params"] = (
            DOG_ONE_KG_BASE_MASS_DELTA,
            DOG_ONE_KG_BASE_MASS_DELTA,
        )
        self.events.add_base_mass.params["operation"] = "add"


class GogoDogServoFlatTorqueSafeStage3OneKgEnvCfg_PLAY(GogoDogServoFlatTorqueSafeStage3OneKgEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.25
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GogoDogServoFlatHardwareLimitOneKgEnvCfg(GogoDogServoFlatTorqueSafeStage2EnvCfg):
    """Hardware-limit stage: train angle-target policies under a 1 kg, 0.6 Nm clipped servo model."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot.actuators["servos"].effort_limit_sim = DOG_TORQUE_SAFE_TORQUE_LIMIT_NM
        self.scene.robot.actuators["servos"].stiffness = DOG_SERVO_STIFFNESS
        self.scene.robot.actuators["servos"].damping = 0.35
        self.actions.joint_pos.scale = DOG_TORQUE_SAFE_STAGE2_ACTION_SCALE

        self.commands.base_velocity.resampling_time_range = (6.0, 9.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.04, 0.14)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.add_base_mass.params["mass_distribution_params"] = (
            DOG_ONE_KG_BASE_MASS_DELTA,
            DOG_ONE_KG_BASE_MASS_DELTA,
        )
        self.events.add_base_mass.params["operation"] = "add"
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.96, 1.04)
        self.events.actuator_gains.params["damping_distribution_params"] = (0.90, 1.15)
        self.events.joint_parameters.params["friction_distribution_params"] = (0.92, 1.14)
        self.events.joint_parameters.params["armature_distribution_params"] = (1.00, 1.16)

        torque_asset_cfg = SceneEntityCfg("robot", joint_names=DOG_JOINT_ORDER, preserve_order=True)
        self.rewards.computed_torque_l2.weight = -0.004
        self.rewards.computed_torque_limit.weight = 0.0
        self.rewards.applied_torque_soft_limit = RewTerm(
            func=gogo_mdp.applied_torque_soft_limit_penalty,
            weight=-1.70,
            params={
                "asset_cfg": torque_asset_cfg,
                "soft_limit": 0.48,
                "hard_limit": DOG_TORQUE_SAFE_TORQUE_LIMIT_NM,
                "max_excess_ratio": 1.0,
                "joint_weights": (1.0, 1.25, 2.2, 1.0, 1.25, 2.2, 1.0, 1.25, 2.4, 1.0, 1.25, 2.4),
            },
        )
        self.rewards.applied_torque_near_limit_fraction = RewTerm(
            func=gogo_mdp.applied_torque_near_limit_fraction,
            weight=-0.75,
            params={"asset_cfg": torque_asset_cfg, "threshold": 0.54},
        )

        self.terminations.applied_torque_saturation = DoneTerm(
            func=gogo_mdp.applied_torque_saturation_termination,
            params={
                "asset_cfg": torque_asset_cfg,
                "threshold": 0.595,
                "max_saturated_joints": 7,
                "grace_time_s": 0.35,
            },
        )

        self.rewards.track_lin_vel_xy_exp.weight = 1.30
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.20
        self.rewards.track_ang_vel_z_exp.weight = 0.08
        self.rewards.commanded_forward_velocity.weight = 0.32
        self.rewards.commanded_forward_velocity.params["min_command_x"] = 0.035
        self.rewards.commanded_forward_velocity.params["max_speed"] = 0.16
        self.rewards.flat_orientation_l2.weight = -2.65
        self.rewards.pitch_abs_l1.weight = -0.90
        self.rewards.lin_vel_z_l2.weight = -1.25
        self.rewards.ang_vel_xy_l2.weight = -0.20
        self.rewards.base_height_l2.weight = -0.44
        self.rewards.dof_torques_l2.weight = -0.0040
        self.rewards.joint_vel_l2.weight = -0.0110
        self.rewards.dof_acc_l2.weight = -5.5e-6
        self.rewards.action_rate_l2.weight = -0.070
        self.rewards.action_smoothness_l2.weight = -0.105
        self.rewards.joint_deviation_l2.weight = -0.30

        self.rewards.trot_phase_contact.weight = 0.28
        self.rewards.trot_phase_contact.params.update(DOG_TORQUE_SAFE_TROT_PHASE_PARAMS)
        self.rewards.feet_air_time.weight = 0.010
        self.rewards.swing_foot_clearance.weight = 0.11
        self.rewards.swing_foot_clearance.params["target_height"] = 0.026
        self.rewards.swing_foot_lift.weight = 0.12
        self.rewards.swing_foot_lift.params["target_height"] = 0.030
        self.rewards.swing_foot_lift.params["velocity_scale"] = 1.1
        self.rewards.swing_foot_height.weight = -0.27
        self.rewards.swing_foot_height.params["max_height"] = 0.044
        self.rewards.swing_foot_drag.weight = -0.52
        self.rewards.swing_foot_drag.params["clearance"] = 0.018
        self.rewards.stance_foot_slip.weight = -0.58
        self.rewards.contact_foot_slide.weight = -0.45
        self.rewards.dof_pos_limits.weight = -0.90


class GogoDogServoFlatHardwareLimitOneKgEnvCfg_PLAY(GogoDogServoFlatHardwareLimitOneKgEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.25
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GogoDogServoFlatHardwareLimitOneKgStageAEnvCfg(GogoDogServoFlatHardwareLimitOneKgEnvCfg):
    """Conservative first hardware-limit stage: prioritize low saturation over speed."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.actions.joint_pos.scale = {
            ".*_hip_abduction": 0.060,
            ".*_hip_pitch": 0.125,
            ".*_knee": 0.185,
        }
        self.commands.base_velocity.rel_standing_envs = 0.65
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.04)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.gait_phase.params.update(DOG_TORQUE_SAFE_STAGE3_TROT_PHASE_PARAMS)

        self.rewards.track_lin_vel_xy_exp.weight = 0.16
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.12
        self.rewards.commanded_forward_velocity.weight = 0.02
        self.rewards.commanded_forward_velocity.params["min_command_x"] = 0.010
        self.rewards.commanded_forward_velocity.params["max_speed"] = 0.055
        self.rewards.applied_torque_soft_limit.weight = -5.00
        self.rewards.applied_torque_soft_limit.params["soft_limit"] = 0.42
        self.rewards.applied_torque_soft_limit.params["joint_weights"] = (
            1.0,
            1.35,
            2.8,
            1.0,
            1.35,
            2.8,
            1.0,
            1.35,
            3.2,
            1.0,
            1.35,
            3.2,
        )
        self.rewards.applied_torque_near_limit_fraction.weight = -2.20
        self.rewards.applied_torque_near_limit_fraction.params["threshold"] = 0.50
        self.rewards.dof_torques_l2.weight = -0.0070
        self.rewards.joint_vel_l2.weight = -0.0300
        self.rewards.dof_acc_l2.weight = -1.0e-5
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-0.050)
        self.rewards.action_rate_l2.weight = -0.150
        self.rewards.action_smoothness_l2.weight = -0.220
        self.rewards.flat_orientation_l2.weight = -7.00
        self.rewards.pitch_abs_l1.weight = -2.20
        self.rewards.lin_vel_z_l2.weight = -2.00
        self.rewards.ang_vel_xy_l2.weight = -0.55
        self.rewards.joint_deviation_l2.weight = -0.70
        self.rewards.joint_deviation_l2.params["standing_scale"] = 3.20

        self.rewards.trot_phase_contact.weight = 0.02
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.swing_foot_clearance.weight = 0.0
        self.rewards.swing_foot_clearance.params["target_height"] = 0.018
        self.rewards.swing_foot_lift.weight = 0.0
        self.rewards.swing_foot_lift.params["target_height"] = 0.020
        self.rewards.swing_foot_lift.params["velocity_scale"] = 0.7
        self.rewards.swing_foot_height.weight = -0.50
        self.rewards.swing_foot_height.params["max_height"] = 0.034
        self.rewards.swing_foot_drag.weight = -0.35
        self.rewards.stance_foot_slip.weight = -0.80
        self.rewards.contact_foot_slide.weight = -0.60

        self.terminations.applied_torque_saturation = None


class GogoDogServoFlatHardwareLimitOneKgStageAEnvCfg_PLAY(GogoDogServoFlatHardwareLimitOneKgStageAEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.25
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GogoDogServoFlatBackYawLiftEnvCfg(GogoDogServoFlatFastElegantEnvCfg):
    """Short fine-tuning stage that targets backing-up, yaw, and foot lift quality."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.22, 0.18)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.18, 0.18)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.95, 0.95)

        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.08, n_max=0.08)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.35, n_max=0.35)
        self.observations.critic.base_ang_vel.noise = Unoise(n_min=-0.08, n_max=0.08)
        self.observations.critic.joint_vel.noise = Unoise(n_min=-0.35, n_max=0.35)

        self.events.base_external_force_torque = None
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.98, 1.04)
        self.events.actuator_gains.params["damping_distribution_params"] = (0.98, 1.08)
        self.events.joint_parameters.params["friction_distribution_params"] = (0.90, 1.12)
        self.events.reset_robot_joints.params["position_range"] = (0.992, 1.008)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.004, 0.004)

        self.rewards.track_lin_vel_xy_exp.weight = 1.45
        self.rewards.track_ang_vel_z_exp.weight = 0.88
        self.rewards.flat_orientation_l2.weight = -2.05
        self.rewards.pitch_abs_l1.weight = -0.65
        self.rewards.action_rate_l2.weight = -0.034
        self.rewards.dof_acc_l2.weight = -2.2e-6
        self.rewards.joint_vel_l2.weight = -0.0042
        self.rewards.action_smoothness_l2.weight = -0.046
        self.rewards.joint_deviation_l2.weight = -0.27
        self.rewards.joint_deviation_l2.params["moving_scale"] = 1.18
        self.rewards.backward_front_leg_deviation.weight = -0.75
        self.rewards.backward_front_leg_deviation.params["allowances"] = (0.15, 0.22, 0.15, 0.22)

        self.rewards.trot_phase_contact.weight = 0.45
        self.rewards.feet_air_time.weight = 0.42
        self.rewards.feet_air_time.params["threshold"] = 0.030
        self.rewards.swing_foot_clearance.weight = 0.85
        self.rewards.swing_foot_clearance.params["target_height"] = 0.050
        self.rewards.swing_foot_lift.weight = 2.05
        self.rewards.swing_foot_lift.params["min_height"] = 0.026
        self.rewards.swing_foot_lift.params["target_height"] = 0.060
        self.rewards.moving_foot_low_height.weight = -1.55
        self.rewards.swing_foot_low_pose.weight = -1.15
        self.rewards.global_moving_low_foot.weight = -0.85
        self.rewards.yaw_backward_low_foot.weight = -2.55
        self.rewards.yaw_backward_swing_lift.weight = 1.45
        self.rewards.yaw_backward_swing_lift.params["target_height"] = 0.062
        self.rewards.yaw_backward_swing_low_pose.weight = -3.15
        self.rewards.yaw_backward_swing_low_pose.params["clearance"] = 0.030
        self.rewards.swing_foot_drag.weight = -2.05
        self.rewards.stance_foot_slip.weight = -0.70
        self.rewards.contact_foot_slide.weight = -0.90
        self.rewards.low_clearance_velocity_progress.weight = -0.52
        self.rewards.low_clearance_velocity_progress.params["clearance"] = 0.028
        self.rewards.low_clearance_velocity_progress.params["foot_weights"] = (1.0, 1.0, 1.15, 1.55)
        self.rewards.commanded_planar_velocity.weight = 0.32
        self.rewards.commanded_planar_velocity.params["clearance"] = 0.028
        self.rewards.commanded_planar_velocity.params["min_gate"] = 0.32
        self.rewards.commanded_yaw_velocity.weight = 0.18
        self.rewards.commanded_yaw_velocity.params["clearance"] = 0.028
        self.rewards.commanded_yaw_velocity.params["min_gate"] = 0.32

        self.rewards.stand_still_joint_deviation.weight = -2.80
        self.rewards.stand_still_action.weight = -1.45
        self.rewards.stand_still_joint_vel.weight = -0.090
        self.rewards.stand_still_foot_vel.weight = -1.10


class GogoDogServoFlatBackYawLiftEnvCfg_PLAY(GogoDogServoFlatBackYawLiftEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.25
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class GogoDogServoFlatMLPEnvCfg(GogoDogServoFlatFastElegantEnvCfg):
    """Backward-compatible alias for the flat fast/elegant servo-dog MLP task."""


class GogoDogServoFlatMLPEnvCfg_PLAY(GogoDogServoFlatFastElegantEnvCfg_PLAY):
    pass


@configclass
class GogoA1DreamWaQRoughEnvCfg(UnitreeA1RoughEnvCfg):
    observations: GogoA1DreamWaQObservationsCfg = GogoA1DreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations = GogoA1DreamWaQObservationsCfg()
        _apply_dreamwaq_action_scale(self)

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.resampling_time_range = (5.0, 10.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 20
            self.scene.terrain.terrain_generator.curriculum = True
            terrains = self.scene.terrain.terrain_generator.sub_terrains
            if "boxes" in terrains:
                terrains["boxes"].grid_height_range = (0.025, 0.12)
            if "random_rough" in terrains:
                terrains["random_rough"].noise_range = (0.01, 0.08)
                terrains["random_rough"].noise_step = 0.01
            if "hf_pyramid_slope" in terrains:
                terrains["hf_pyramid_slope"].slope_range = (0.0, 0.40)
            if "hf_pyramid_slope_inv" in terrains:
                terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.40)

        self.events.physics_material.params["static_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["force_range"] = (-60.0, 60.0)
        self.events.base_external_force_torque.params["torque_range"] = (-8.0, 8.0)
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.15, 0.15)}},
        )
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=0.0,
            params={
                "target_height": 0.36,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=0.0)
        self.rewards.joint_power = RewTerm(func=gogo_mdp.joint_power_l1, weight=-2.0e-5)
        self.rewards.action_smoothness = RewTerm(func=gogo_mdp.action_smoothness_l2, weight=-0.01)
        self.rewards.power_distribution = RewTerm(func=gogo_mdp.power_distribution_variance, weight=-1.0e-5)


@configclass
class GogoA1DreamWaQWarmupEnvCfg(GogoA1DreamWaQRoughEnvCfg):
    """Low-difficulty DreamWaQ stage used before resuming into full rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        _apply_dreamwaq_action_scale(self)

        self.commands.base_velocity.rel_standing_envs = 0.20
        self.commands.base_velocity.resampling_time_range = (6.0, 10.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.25, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.20, 0.20)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.35, 0.35)

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.actuator_gains = None
        self.events.physics_material.params["static_friction_range"] = (0.7, 1.1)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.0)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.25, 0.75)

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 8
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.max_init_terrain_level = 0
            terrains = self.scene.terrain.terrain_generator.sub_terrains
            if "pyramid_stairs" in terrains:
                terrains["pyramid_stairs"].step_height_range = (0.0, 0.02)
            if "pyramid_stairs_inv" in terrains:
                terrains["pyramid_stairs_inv"].step_height_range = (0.0, 0.02)
            if "boxes" in terrains:
                terrains["boxes"].grid_height_range = (0.0, 0.02)
            if "random_rough" in terrains:
                terrains["random_rough"].noise_range = (0.0, 0.012)
                terrains["random_rough"].noise_step = 0.005
            if "hf_pyramid_slope" in terrains:
                terrains["hf_pyramid_slope"].slope_range = (0.0, 0.06)
            if "hf_pyramid_slope_inv" in terrains:
                terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.06)

        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.base_height_l2.params["target_height"] = 0.36
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.termination_penalty.weight = 0.0
        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.action_smoothness.weight = -0.01
        self.rewards.power_distribution.weight = -1.0e-5


class GogoA1DreamWaQWarmupEnvCfg_PLAY(GogoA1DreamWaQWarmupEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1DreamWaQForwardEnvCfg(GogoA1DreamWaQWarmupEnvCfg):
    """Forward locomotion stage that prevents the warm-up policy from converging to standing."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.28, 0.55)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.08, 0.08)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.12, 0.12)

        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness.weight = -0.01
        self.rewards.forward_velocity = RewTerm(
            func=gogo_mdp.forward_velocity_reward,
            weight=0.0,
            params={"command_name": "base_velocity", "min_command_x": 0.10, "max_speed": 0.70},
        )
        self.rewards.commanded_stuck = RewTerm(
            func=gogo_mdp.commanded_stuck_penalty,
            weight=0.0,
            params={"command_name": "base_velocity", "min_command_x": 0.18, "min_speed_x": 0.06},
        )


class GogoA1DreamWaQForwardEnvCfg_PLAY(GogoA1DreamWaQForwardEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1DreamWaQMidRoughEnvCfg(GogoA1DreamWaQForwardEnvCfg):
    """Intermediate terrain stage between Forward/EasyRough and full Rough."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (4.0, 7.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.20, 0.20)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.35, 0.35)

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.actuator_gains = None
        self.events.physics_material.params["static_friction_range"] = (0.55, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.45, 1.05)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 1.0)

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 12
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.max_init_terrain_level = 1
            terrains = self.scene.terrain.terrain_generator.sub_terrains
            if "pyramid_stairs" in terrains:
                terrains["pyramid_stairs"].step_height_range = (0.0, 0.04)
            if "pyramid_stairs_inv" in terrains:
                terrains["pyramid_stairs_inv"].step_height_range = (0.0, 0.04)
            if "boxes" in terrains:
                terrains["boxes"].grid_height_range = (0.0, 0.05)
            if "random_rough" in terrains:
                terrains["random_rough"].noise_range = (0.0, 0.03)
                terrains["random_rough"].noise_step = 0.005
            if "hf_pyramid_slope" in terrains:
                terrains["hf_pyramid_slope"].slope_range = (0.0, 0.16)
            if "hf_pyramid_slope_inv" in terrains:
                terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.16)

        self.rewards.feet_air_time.weight = 0.1
        self.rewards.termination_penalty.weight = 0.0
        self.rewards.forward_velocity.weight = 0.0
        self.rewards.commanded_stuck.weight = 0.0


class GogoA1DreamWaQMidRoughEnvCfg_PLAY(GogoA1DreamWaQMidRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 4
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1DreamWaQHardRoughEnvCfg(GogoA1DreamWaQForwardEnvCfg):
    """High-geometry curriculum stage before enabling full rough perturbations."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.85)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.30, 0.30)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.50, 0.50)

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.actuator_gains = None
        self.events.physics_material.params["static_friction_range"] = (0.45, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.35, 1.10)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 1.50)

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 16
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.85)
            self.scene.terrain.max_init_terrain_level = 2
            terrains = self.scene.terrain.terrain_generator.sub_terrains
            if "pyramid_stairs" in terrains:
                terrains["pyramid_stairs"].step_height_range = (0.0, 0.12)
            if "pyramid_stairs_inv" in terrains:
                terrains["pyramid_stairs_inv"].step_height_range = (0.0, 0.12)
            if "boxes" in terrains:
                terrains["boxes"].grid_height_range = (0.0, 0.12)
            if "random_rough" in terrains:
                terrains["random_rough"].noise_range = (0.0, 0.06)
                terrains["random_rough"].noise_step = 0.01
            if "hf_pyramid_slope" in terrains:
                terrains["hf_pyramid_slope"].slope_range = (0.0, 0.30)
            if "hf_pyramid_slope_inv" in terrains:
                terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.30)

        self.rewards.feet_air_time.weight = 0.1
        self.rewards.termination_penalty.weight = 0.0
        self.rewards.forward_velocity.weight = 0.0
        self.rewards.commanded_stuck.weight = 0.0


class GogoA1DreamWaQHardRoughEnvCfg_PLAY(GogoA1DreamWaQHardRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1DreamWaQRoughGeoEnvCfg(GogoA1DreamWaQRoughEnvCfg):
    """Full rough geometry curriculum before enabling push/force/actuator randomization."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.85)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.30, 0.30)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.50, 0.50)

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.actuator_gains = None
        self.events.physics_material.params["static_friction_range"] = (0.45, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.35, 1.10)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 1.50)

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 20
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
            self.scene.terrain.max_init_terrain_level = 1

        self.rewards.feet_air_time.weight = 0.1
        self.rewards.termination_penalty.weight = 0.0


@configclass
class GogoA1DreamWaQRoughGeoTableIHeightEnvCfg(GogoA1DreamWaQRoughGeoEnvCfg):
    """DreamWaQ Table-I body-height ablation with terrain-relative height."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["target_height"] = 0.36
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner")


@configclass
class GogoA1DreamWaQRoughGeoTableIHeightA1TargetEnvCfg(GogoA1DreamWaQRoughGeoEnvCfg):
    """DreamWaQ Table-I body-height branch using the public A1 target height."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["target_height"] = 0.25
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner")


@configclass
class GogoA1DreamWaQRoughGeoPositiveRewardEnvCfg(GogoA1DreamWaQRoughGeoEnvCfg):
    """DreamWaQ rough geometry branch with public-code non-negative reward clipping."""


@configclass
class GogoA1DreamWaQRoughGeoPositiveRewardA1HeightEnvCfg(GogoA1DreamWaQRoughGeoTableIHeightA1TargetEnvCfg):
    """Positive reward clipping plus public A1 terrain-relative body-height target."""


@configclass
class GogoA1DreamWaQRoughGeoPublicTerrainLevel3EnvCfg(GogoA1DreamWaQRoughGeoEnvCfg):
    """RoughGeo curriculum with public DreamWaQ-like terrain proportions and level-3 initialization."""

    def __post_init__(self):
        super().__post_init__()

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.max_init_terrain_level = 3
            terrains = self.scene.terrain.terrain_generator.sub_terrains
            public_like_proportions = {
                "pyramid_stairs": 0.35,
                "pyramid_stairs_inv": 0.35,
                "boxes": 0.10,
                "random_rough": 0.10,
                "hf_pyramid_slope": 0.05,
                "hf_pyramid_slope_inv": 0.05,
            }
            for name, proportion in public_like_proportions.items():
                if name in terrains:
                    terrains[name].proportion = proportion


@configclass
class GogoA1DreamWaQRoughGeoPublicTerrainLevel5EnvCfg(GogoA1DreamWaQRoughGeoPublicTerrainLevel3EnvCfg):
    """Public-proportion curriculum with public DreamWaQ max initial terrain level."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.max_init_terrain_level = 5


@configclass
class GogoA1DreamWaQFromScratchPaperLikeEnvCfg(GogoA1DreamWaQRoughEnvCfg):
    """From-scratch DreamWaQ branch aligned with the paper and public A1 config."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 20
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
            self.scene.terrain.max_init_terrain_level = 5
            terrains = self.scene.terrain.terrain_generator.sub_terrains
            public_like_proportions = {
                "pyramid_stairs": 0.35,
                "pyramid_stairs_inv": 0.35,
                "boxes": 0.10,
                "random_rough": 0.10,
                "hf_pyramid_slope": 0.05,
                "hf_pyramid_slope_inv": 0.05,
            }
            for name, proportion in public_like_proportions.items():
                if name in terrains:
                    terrains[name].proportion = proportion
        self.curriculum.terrain_levels = CurrTerm(func=gogo_mdp.terrain_levels_vel_public_init_guard)

        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["target_height"] = 0.25
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.rewards.termination_penalty.weight = 0.0
        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.action_smoothness.weight = -0.01
        self.rewards.power_distribution.weight = -1.0e-5


@configclass
class GogoA1DreamWaQFromScratchPaperLikeStage0EnvCfg(GogoA1DreamWaQFromScratchPaperLikeEnvCfg):
    """Paper-like DreamWaQ branch with terrain curriculum initialized at level 0 only."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.max_init_terrain_level = 0


@configclass
class GogoA1DreamWaQFromScratchPaperLikeStage1EnvCfg(GogoA1DreamWaQFromScratchPaperLikeEnvCfg):
    """Paper-like DreamWaQ branch with low initial terrain levels."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.max_init_terrain_level = 1


@configclass
class GogoA1DreamWaQFromScratchPaperLikeStage3EnvCfg(GogoA1DreamWaQFromScratchPaperLikeEnvCfg):
    """Paper-like DreamWaQ branch with mid initial terrain levels."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.max_init_terrain_level = 3


@configclass
class GogoA1DreamWaQFromScratchPaperLikeStage5EnvCfg(GogoA1DreamWaQFromScratchPaperLikeEnvCfg):
    """Paper-like DreamWaQ branch with public DreamWaQ initial terrain levels."""


@configclass
class GogoA1DreamWaQFromScratchPaperLikeFullStateEnvCfg(GogoA1DreamWaQFromScratchPaperLikeEnvCfg):
    """Paper-like DreamWaQ branch with public critic full-state observations."""

    observations: GogoA1DreamWaQPaperLikeFullStateObservationsCfg = GogoA1DreamWaQPaperLikeFullStateObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.observations = GogoA1DreamWaQPaperLikeFullStateObservationsCfg()


@configclass
class GogoA1DreamWaQFromScratchPaperLikeFullStateStage0EnvCfg(
    GogoA1DreamWaQFromScratchPaperLikeFullStateEnvCfg
):
    """Full-state paper-like branch with terrain curriculum initialized at level 0 only."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.max_init_terrain_level = 0


@configclass
class GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20EnvCfg(
    GogoA1DreamWaQFromScratchPaperLikeFullStateEnvCfg
):
    """Full-state paper-like branch with public DreamWaQ A1 PD gains."""

    def __post_init__(self):
        super().__post_init__()

        _apply_public_a1_pd_gains(self)


@configclass
class GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20EnvCfg(
    GogoA1DreamWaQFromScratchPaperLikeFullStateStage0EnvCfg
):
    """Stage0 full-state branch with public DreamWaQ A1 PD gains."""

    def __post_init__(self):
        super().__post_init__()

        _apply_public_a1_pd_gains(self)


@configclass
class GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitEnvCfg(
    GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20EnvCfg
):
    """Full-state A1PD20 branch with public A1 joint-limit penalty."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.dof_pos_limits.weight = -10.0


@configclass
class GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitEnvCfg(
    GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20EnvCfg
):
    """Stage0 A1PD20 branch with public A1 joint-limit penalty."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.dof_pos_limits.weight = -10.0


@configclass
class GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg(
    GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitEnvCfg
):
    """A1PD20 DofLimit branch with public legged_gym-style heightfield stairs."""

    def __post_init__(self):
        super().__post_init__()

        _apply_public_hf_stairs(self)


@configclass
class GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsUpBiasEnvCfg(
    GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg
):
    """Public HfStairs branch with mixed terrain biased toward physical +x upstairs."""

    def __post_init__(self):
        super().__post_init__()

        _apply_public_upstairs_biased_mix(self)


@configclass
class GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsLowUpEnvCfg(
    GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg
):
    """Public HfStairs branch with a gated low-upstairs mixed curriculum."""

    def __post_init__(self):
        super().__post_init__()

        _apply_public_hf_stairs_low_up_mix(self)


@configclass
class GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsEnvCfg(
    GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitEnvCfg
):
    """Stage0 A1PD20 DofLimit branch with public legged_gym-style heightfield stairs."""

    def __post_init__(self):
        super().__post_init__()

        _apply_public_hf_stairs(self)


class GogoA1DreamWaQFromScratchPaperLikeEnvCfg_PLAY(GogoA1DreamWaQFromScratchPaperLikeEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeStage0EnvCfg_PLAY(GogoA1DreamWaQFromScratchPaperLikeStage0EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeStage1EnvCfg_PLAY(GogoA1DreamWaQFromScratchPaperLikeStage1EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeStage3EnvCfg_PLAY(GogoA1DreamWaQFromScratchPaperLikeStage3EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeStage5EnvCfg_PLAY(GogoA1DreamWaQFromScratchPaperLikeStage5EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeFullStateEnvCfg_PLAY(
    GogoA1DreamWaQFromScratchPaperLikeFullStateEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeFullStateStage0EnvCfg_PLAY(
    GogoA1DreamWaQFromScratchPaperLikeFullStateStage0EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20EnvCfg_PLAY(
    GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20EnvCfg_PLAY(
    GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitEnvCfg_PLAY(
    GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitEnvCfg_PLAY(
    GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg_PLAY(
    GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsUpBiasEnvCfg_PLAY(
    GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsUpBiasEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsLowUpEnvCfg_PLAY(
    GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsLowUpEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsEnvCfg_PLAY(
    GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1DreamWaQRoughGeoTableIFootClearanceEnvCfg(GogoA1DreamWaQRoughGeoEnvCfg):
    """DreamWaQ Table-I foot-clearance ablation."""

    def __post_init__(self):
        super().__post_init__()

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.rewards.dreamwaq_foot_clearance = RewTerm(
            func=gogo_mdp.dreamwaq_foot_clearance_l2,
            weight=-0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=foot_names, preserve_order=True),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "desired_clearance": 0.08,
            },
        )


@configclass
class GogoA1DreamWaQRoughGeoTableIFullEnvCfg(GogoA1DreamWaQRoughGeoTableIFootClearanceEnvCfg):
    """DreamWaQ Table-I body-height plus foot-clearance ablation."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["target_height"] = 0.36
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner")


class GogoA1DreamWaQRoughGeoEnvCfg_PLAY(GogoA1DreamWaQRoughGeoEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1DreamWaQRoughGeoHardEnvCfg(GogoA1DreamWaQRoughGeoEnvCfg):
    """High-difficulty full rough geometry stage without dynamics perturbations."""

    def __post_init__(self):
        super().__post_init__()

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.35, 1.0)
            self.scene.terrain.max_init_terrain_level = 1


@configclass
class GogoA1DreamWaQRoughGeoMidHighEnvCfg(GogoA1DreamWaQRoughGeoEnvCfg):
    """Middle-high full rough geometry stage before the hardest terrain tail."""

    def __post_init__(self):
        super().__post_init__()

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.35, 0.70)
            self.scene.terrain.max_init_terrain_level = 1


@configclass
class GogoA1DreamWaQRoughGeoMidLowEnvCfg(GogoA1DreamWaQRoughGeoEnvCfg):
    """First narrow high-geometry curriculum bin after RoughGeo-init1."""

    def __post_init__(self):
        super().__post_init__()

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.35, 0.55)
            self.scene.terrain.max_init_terrain_level = 1


@configclass
class GogoA1DreamWaQRoughGeoMidEnvCfg(GogoA1DreamWaQRoughGeoEnvCfg):
    """Second narrow high-geometry curriculum bin after RoughGeoMidLow."""

    def __post_init__(self):
        super().__post_init__()

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.55, 0.70)
            self.scene.terrain.max_init_terrain_level = 1


class GogoA1DreamWaQRoughGeoMidLowEnvCfg_PLAY(GogoA1DreamWaQRoughGeoMidLowEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQRoughGeoMidEnvCfg_PLAY(GogoA1DreamWaQRoughGeoMidEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1DreamWaQRoughGeoDownStairsEnvCfg(GogoA1DreamWaQRoughGeoEnvCfg):
    """Single-terrain curriculum for inverted pyramid stairs, the current dominant failure mode."""

    def __post_init__(self):
        super().__post_init__()

        if self.scene.terrain.terrain_generator is not None:
            terrains = self.scene.terrain.terrain_generator.sub_terrains
            self.scene.terrain.terrain_generator.sub_terrains = {
                "pyramid_stairs_inv": terrains["pyramid_stairs_inv"],
            }
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
            self.scene.terrain.max_init_terrain_level = 1


class GogoA1DreamWaQRoughGeoDownStairsEnvCfg_PLAY(GogoA1DreamWaQRoughGeoDownStairsEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1DreamWaQRoughGeoDownStairsSlowEnvCfg(GogoA1DreamWaQRoughGeoDownStairsEnvCfg):
    """Low-speed inverted stairs curriculum after speed-sensitivity diagnosis."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (5.0, 9.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.40)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.08, 0.08)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.15, 0.15)


class GogoA1DreamWaQRoughGeoDownStairsSlowEnvCfg_PLAY(GogoA1DreamWaQRoughGeoDownStairsSlowEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1DreamWaQRoughGeoDownStairsMidGeomEnvCfg(GogoA1DreamWaQRoughGeoDownStairsEnvCfg):
    """Mid-geometry inverted stairs curriculum before exposing the high step-height tail."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.55)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.10, 0.10)

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.25, 0.50)


class GogoA1DreamWaQRoughGeoDownStairsMidGeomEnvCfg_PLAY(GogoA1DreamWaQRoughGeoDownStairsMidGeomEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1DreamWaQRoughGeoInvStairsMixEnvCfg(GogoA1DreamWaQRoughGeoEnvCfg):
    """Mixed rough curriculum with extra medium inverted-stairs exposure."""

    def __post_init__(self):
        super().__post_init__()

        if self.scene.terrain.terrain_generator is not None:
            terrains = self.scene.terrain.terrain_generator.sub_terrains
            if "pyramid_stairs_inv" in terrains:
                terrains["pyramid_stairs_inv"].step_height_range = (0.095, 0.14)
                original_proportions = {name: terrain.proportion for name, terrain in terrains.items()}
                inv_proportion = 0.35
                remaining_total = sum(
                    proportion for name, proportion in original_proportions.items() if name != "pyramid_stairs_inv"
                )
                for name, terrain in terrains.items():
                    if name == "pyramid_stairs_inv":
                        terrain.proportion = inv_proportion
                    else:
                        terrain.proportion = (1.0 - inv_proportion) * original_proportions[name] / remaining_total


@configclass
class GogoA1DreamWaQRoughGeoTableIHeightInvStairsMixEnvCfg(GogoA1DreamWaQRoughGeoTableIHeightEnvCfg):
    """Table-I terrain-relative body height with extra medium inverted-stairs exposure."""

    def __post_init__(self):
        super().__post_init__()

        if self.scene.terrain.terrain_generator is not None:
            terrains = self.scene.terrain.terrain_generator.sub_terrains
            if "pyramid_stairs_inv" in terrains:
                terrains["pyramid_stairs_inv"].step_height_range = (0.095, 0.14)
                original_proportions = {name: terrain.proportion for name, terrain in terrains.items()}
                inv_proportion = 0.35
                remaining_total = sum(
                    proportion for name, proportion in original_proportions.items() if name != "pyramid_stairs_inv"
                )
                for name, terrain in terrains.items():
                    if name == "pyramid_stairs_inv":
                        terrain.proportion = inv_proportion
                    else:
                        terrain.proportion = (1.0 - inv_proportion) * original_proportions[name] / remaining_total


class GogoA1DreamWaQRoughGeoInvStairsMixEnvCfg_PLAY(GogoA1DreamWaQRoughGeoInvStairsMixEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQRoughGeoMidHighEnvCfg_PLAY(GogoA1DreamWaQRoughGeoMidHighEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1DreamWaQRoughGeoHardEnvCfg_PLAY(GogoA1DreamWaQRoughGeoHardEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1DreamWaQFlatStartEnvCfg(GogoA1DreamWaQForwardEnvCfg):
    """Flat-ground locomotion start stage before terrain adaptation."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        _apply_dreamwaq_action_scale(self)

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (6.0, 10.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.65)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.10, 0.10)

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.actuator_gains = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.1, 0.3)

        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.base_height_l2.params["target_height"] = 0.36
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.termination_penalty.weight = 0.0
        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.action_smoothness.weight = -0.01
        self.rewards.power_distribution.weight = -1.0e-5
        self.rewards.forward_velocity.weight = 0.0
        self.rewards.forward_velocity.params["max_speed"] = 0.65
        self.rewards.commanded_stuck.weight = 0.0
        self.rewards.commanded_stuck.params["min_speed_x"] = 0.06


class GogoA1DreamWaQFlatStartEnvCfg_PLAY(GogoA1DreamWaQFlatStartEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None


class GogoA1DreamWaQRoughEnvCfg_PLAY(GogoA1DreamWaQRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairApproachEnvCfg(GogoA1FlatFastElegantEnvCfg):
    observations: GogoA1ObservationsCfg = GogoA1ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations = GogoA1ObservationsCfg()

        def trunk_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names="trunk")

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FORWARD_STAIRS_TERRAINS_CFG.copy()
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.env_spacing = 3.0
        self.scene.height_scanner = None

        self.curriculum.terrain_levels = CurrTerm(
            func=gogo_mdp.stair_climb_terrain_levels,
            params={
                "target_x": STAIR_TARGET_X,
                "target_z": STAIR_TARGET_Z,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "num_steps": STAIR_NUM_STEPS,
                "target_base_clearance": STAIR_TARGET_BASE_CLEARANCE,
            },
        )

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.26, 0.34)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.events.reset_base = EventTerm(
            func=gogo_mdp.reset_stair_climb_root_state,
            mode="reset",
            params={
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "approach_probability": 1.0,
                "approach_x_range": (-0.20, 0.14),
                "near_step_x_range": (0.58, 0.70),
                "y_range": (-0.05, 0.05),
                "yaw_range": (-0.08, 0.08),
                "base_height": 0.31,
                "step_base_clearance": 0.29,
                "velocity_range": {
                    "x": (0.0, 0.05),
                    "y": (-0.02, 0.02),
                    "z": (-0.02, 0.02),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.04, 0.04),
                    "yaw": (-0.05, 0.05),
                },
                "joint_position_scale_range": (0.98, 1.02),
                "joint_velocity_range": (-0.05, 0.05),
            },
        )
        self.events.reset_robot_joints = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        handoff_x = 0.70
        max_handoff_x = STAIR_FIRST_RISER_X + 0.02
        self.episode_length_s = 5.0

        self.rewards.track_lin_vel_xy_exp.weight = 0.65
        self.rewards.track_ang_vel_z_exp.weight = 0.08
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.trot_phase_contact.weight = 0.08
        self.rewards.swing_foot_clearance.weight = 0.06
        self.rewards.swing_foot_clearance.params["target_height"] = 0.065
        self.rewards.swing_foot_height.weight = -0.12
        self.rewards.swing_foot_height.params["max_height"] = 0.10
        self.rewards.swing_foot_drag.weight = -0.12
        self.rewards.swing_foot_drag.params["clearance"] = 0.045
        self.rewards.stance_foot_slip.weight = -0.18
        self.rewards.trot_pair_timing.weight = 0.02
        self.rewards.stand_still_joint_deviation.weight = 0.0
        self.rewards.stand_still_action.weight = 0.0
        self.rewards.base_height_l2.weight = -0.25
        self.rewards.base_height_l2.params["target_height"] = 0.31
        self.rewards.flat_orientation_l2.weight = -0.25
        self.rewards.lin_vel_z_l2.weight = -0.8
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.dof_acc_l2.weight = -8.0e-7
        self.rewards.dof_torques_l2.weight = -1.8e-4

        self.rewards.handoff_success = RewTerm(
            func=gogo_mdp.stair_handoff_success_reward,
            weight=95.0,
            params={
                "target_x": handoff_x,
                "max_x": max_handoff_x,
                "min_base_height": 0.255,
                "max_abs_y": 0.18,
                "max_pitch_projected": 0.30,
                "max_roll_projected": 0.24,
                "max_yaw": 0.25,
                "min_steps": 12,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
            },
        )
        self.rewards.handoff_pose = RewTerm(
            func=gogo_mdp.stair_handoff_pose_reward,
            weight=13.0,
            params={
                "target_x": handoff_x,
                "target_base_height": 0.31,
                "x_std": 0.22,
                "y_std": 0.15,
                "height_std": 0.10,
                "yaw_std": 0.20,
                "pitch_std": 0.24,
                "roll_std": 0.20,
                "max_speed": 0.38,
            },
        )
        self.rewards.handoff_progress = RewTerm(
            func=gogo_mdp.stair_handoff_progress_reward,
            weight=4.5,
            params={
                "start_x": -0.20,
                "target_x": handoff_x,
                "max_speed": 0.38,
                "yaw_allowance": 0.10,
                "roll_allowance": 0.22,
            },
        )
        self.rewards.handoff_lateral_drift = RewTerm(
            func=gogo_mdp.stair_lateral_drift_penalty,
            weight=-2.2,
        )
        self.rewards.handoff_yaw_drift = RewTerm(
            func=gogo_mdp.stair_yaw_drift_penalty,
            weight=-6.0,
            params={"yaw_allowance": 0.08},
        )
        self.rewards.handoff_tilt = RewTerm(
            func=gogo_mdp.stair_tilt_penalty,
            weight=-8.0,
            params={"pitch_allowance": 0.22},
        )
        self.rewards.handoff_stuck = RewTerm(
            func=gogo_mdp.stair_stuck_penalty,
            weight=-0.9,
            params={"before_goal_x": handoff_x, "min_speed_x": 0.04},
        )
        self.rewards.handoff_overshoot = RewTerm(
            func=gogo_mdp.stair_handoff_overshoot_penalty,
            weight=-10.0,
            params={"first_riser_x": STAIR_FIRST_RISER_X, "margin": -0.03},
        )
        self.rewards.trunk_contact = RewTerm(
            func=gogo_mdp.trunk_contact_penalty,
            weight=-8.0,
            params={"sensor_cfg": trunk_sensor_cfg(), "threshold": 1.0},
        )
        self.rewards.fall_failure = RewTerm(
            func=gogo_mdp.stair_fall,
            weight=-40.0,
            params={
                "min_base_height": 0.15,
                "max_lateral_offset": 0.75,
                "max_pitch_projected": 0.78,
                "max_roll_projected": 0.62,
            },
        )
        self.rewards.bad_yaw_failure = RewTerm(
            func=gogo_mdp.stair_bad_yaw,
            weight=-30.0,
            params={"max_yaw": 0.75, "gate_x": 0.42, "grace_time_s": 0.25},
        )

        self.terminations.success = DoneTerm(
            func=gogo_mdp.stair_handoff_success,
            params={
                "target_x": handoff_x,
                "max_x": max_handoff_x,
                "min_base_height": 0.255,
                "max_abs_y": 0.18,
                "max_pitch_projected": 0.30,
                "max_roll_projected": 0.24,
                "max_yaw": 0.25,
                "min_steps": 12,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
            },
        )
        self.terminations.fall = DoneTerm(
            func=gogo_mdp.stair_fall,
            params={
                "min_base_height": 0.12,
                "max_lateral_offset": 0.85,
                "max_pitch_projected": 0.84,
                "max_roll_projected": 0.68,
            },
        )
        self.terminations.bad_yaw = DoneTerm(
            func=gogo_mdp.stair_bad_yaw,
            params={"max_yaw": 0.85, "gate_x": 0.42, "grace_time_s": 0.25},
        )
        self.terminations.overshoot = DoneTerm(
            func=gogo_mdp.stair_handoff_overshoot,
            params={"overshoot_x": STAIR_FIRST_RISER_X + 0.08, "grace_time_s": 0.6},
        )
        self.terminations.trunk_contact = DoneTerm(
            func=gogo_mdp.trunk_contact_failure,
            params={"sensor_cfg": trunk_sensor_cfg(), "threshold": 8.0, "grace_time_s": 0.25},
        )
        self.terminations.base_contact = None


class GogoA1StairApproachEnvCfg_PLAY(GogoA1StairApproachEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2


@configclass
class GogoA1StairApproachGoalEnvCfg(GogoA1StairApproachEnvCfg):
    observations: GogoA1StairApproachGoalObservationsCfg = GogoA1StairApproachGoalObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations = GogoA1StairApproachGoalObservationsCfg()


class GogoA1StairApproachGoalEnvCfg_PLAY(GogoA1StairApproachGoalEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2


@configclass
class GogoA1StairClimbEnvCfg(GogoA1FlatStudentEnvCfg):
    observations: GogoA1StairObservationsCfg = GogoA1StairObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations = GogoA1StairObservationsCfg()
        _apply_deploy_action_scale(self)

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        def foot_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", body_names=foot_names, preserve_order=True)

        def trunk_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names="trunk")

        def foot_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names=foot_names, preserve_order=True)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FORWARD_STAIRS_TERRAINS_CFG.copy()
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.env_spacing = 3.0
        self.scene.height_scanner = None

        self.curriculum.terrain_levels = CurrTerm(
            func=gogo_mdp.stair_climb_terrain_levels,
            params={
                "target_x": STAIR_TARGET_X,
                "target_z": STAIR_TARGET_Z,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "num_steps": STAIR_NUM_STEPS,
                "target_base_clearance": STAIR_TARGET_BASE_CLEARANCE,
            },
        )

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (8.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.28)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.events.reset_base = EventTerm(
            func=gogo_mdp.reset_stair_climb_mixed_switch_and_near_step,
            mode="reset",
            params={
                "dataset_path": "outputs/flat_switch_states_x058_cmd025.pt",
                "switch_probability": 0.75,
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "near_step_x_range": (0.70, 0.90),
                "y_range": (-0.06, 0.06),
                "yaw_range": (-0.08, 0.08),
                "step_base_clearance": 0.29,
                "x_jitter": (-0.03, 0.04),
                "y_jitter": (-0.03, 0.03),
                "yaw_jitter": (-0.08, 0.08),
                "velocity_scale_range": (0.85, 1.15),
            },
        )
        self.events.reset_robot_joints = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.episode_length_s = 8.0

        self.rewards.track_lin_vel_xy_exp.weight = 0.18
        self.rewards.track_ang_vel_z_exp.weight = 0.05
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.trot_phase_contact.weight = 0.05
        self.rewards.trot_phase_contact.params["sensor_cfg"] = foot_sensor_cfg()
        self.rewards.trot_phase_contact.params["contact_threshold"] = 0.5
        self.rewards.swing_foot_clearance.weight = 0.08
        self.rewards.swing_foot_clearance.params["target_height"] = 0.09
        self.rewards.swing_foot_clearance.params["std"] = 0.35
        self.rewards.swing_foot_height.weight = -0.08
        self.rewards.swing_foot_height.params["max_height"] = 0.16
        self.rewards.swing_foot_drag.weight = -0.08
        self.rewards.swing_foot_drag.params["sensor_cfg"] = foot_sensor_cfg()
        self.rewards.swing_foot_drag.params["contact_threshold"] = 0.5
        self.rewards.swing_foot_drag.params["clearance"] = 0.055
        self.rewards.stance_foot_slip.weight = -0.12
        self.rewards.stance_foot_slip.params["sensor_cfg"] = foot_sensor_cfg()
        self.rewards.stance_foot_slip.params["contact_threshold"] = 0.5
        self.rewards.trot_pair_timing.weight = 0.0
        self.rewards.stand_still_joint_deviation.weight = 0.0
        self.rewards.stand_still_action.weight = 0.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -0.35
        self.rewards.ang_vel_xy_l2.weight = -0.14
        self.rewards.action_rate_l2.weight = -0.04
        self.rewards.dof_acc_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.weight = -2.6e-4

        self.rewards.stair_forward_progress = RewTerm(
            func=gogo_mdp.stair_forward_progress_reward,
            weight=0.55,
            params={"target_x": STAIR_TARGET_X},
        )
        self.rewards.stair_forward_velocity = RewTerm(
            func=gogo_mdp.stair_forward_velocity_reward,
            weight=0.25,
            params={"target_x": STAIR_TARGET_X, "max_speed": 0.45},
        )
        self.rewards.stair_height_progress = RewTerm(
            func=gogo_mdp.stair_height_progress_reward,
            weight=2.2,
            params={
                "target_z": STAIR_TARGET_Z,
                "gate_x": STAIR_FIRST_RISER_X - 0.20,
                "base_z": 0.28,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "num_steps": STAIR_NUM_STEPS,
                "target_base_clearance": STAIR_TARGET_BASE_CLEARANCE,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
                "max_pitch_projected": 0.70,
                "max_roll_projected": 0.38,
            },
        )
        self.rewards.stair_base_height_first = RewTerm(
            func=gogo_mdp.stair_base_height_over_step_reward,
            weight=2.2,
            params={
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_index": 1,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "base_clearance": 0.23,
                "gate_margin": -0.20,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
                "max_pitch_projected": 0.72,
                "max_roll_projected": 0.42,
            },
        )
        self.rewards.stair_front_feet_on_step = RewTerm(
            func=gogo_mdp.stair_foot_placement_reward,
            weight=1.5,
            params={
                "asset_cfg": foot_asset_cfg(),
                "foot_indices": (0, 1),
                "edge_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "step_index": 1,
                "x_margin": 0.02,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
            },
        )
        self.rewards.stair_base_on_first_step = RewTerm(
            func=gogo_mdp.stair_base_on_step_reward,
            weight=9.0,
            params={
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_depth": STAIR_STEP_DEPTH,
                "step_index": 1,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "base_clearance": 0.23,
                "x_fraction": 0.34,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
                "max_pitch_projected": 0.72,
                "max_roll_projected": 0.48,
            },
        )
        self.rewards.stair_front_feet_on_second_step = RewTerm(
            func=gogo_mdp.stair_foot_placement_reward,
            weight=0.0,
            params={
                "asset_cfg": foot_asset_cfg(),
                "foot_indices": (0, 1),
                "edge_x": STAIR_FIRST_RISER_X + STAIR_STEP_DEPTH,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "step_index": 2,
                "x_margin": 0.02,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
            },
        )
        self.rewards.stair_base_on_second_step = RewTerm(
            func=gogo_mdp.stair_base_on_step_reward,
            weight=0.0,
            params={
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_depth": STAIR_STEP_DEPTH,
                "step_index": 2,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "base_clearance": 0.22,
                "x_fraction": 0.55,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
                "max_pitch_projected": 0.72,
                "max_roll_projected": 0.42,
            },
        )
        self.rewards.stair_rear_feet_on_step = RewTerm(
            func=gogo_mdp.stair_foot_placement_reward,
            weight=9.0,
            params={
                "asset_cfg": foot_asset_cfg(),
                "foot_indices": (2, 3),
                "edge_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "step_index": 1,
                "x_margin": 0.02,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
            },
        )
        self.rewards.stair_rear_feet_contact_on_step = RewTerm(
            func=gogo_mdp.stair_foot_contact_on_step_reward,
            weight=11.0,
            params={
                "asset_cfg": foot_asset_cfg(),
                "contact_sensor_cfg": foot_sensor_cfg(),
                "foot_indices": (2, 3),
                "edge_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "step_index": 1,
                "x_margin": 0.02,
                "contact_threshold": 0.5,
                "trunk_sensor_cfg": trunk_sensor_cfg(),
                "trunk_contact_threshold": 1.0,
            },
        )
        self.rewards.stair_base_rear_feet_on_first_step = RewTerm(
            func=gogo_mdp.stair_base_with_feet_on_step_reward,
            weight=8.0,
            params={
                "asset_cfg": foot_asset_cfg(),
                "contact_sensor_cfg": foot_sensor_cfg(),
                "foot_indices": (2, 3),
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_depth": STAIR_STEP_DEPTH,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "step_index": 1,
                "base_clearance": 0.22,
                "base_x_fraction": 0.32,
                "foot_x_margin": -0.02,
                "contact_threshold": 0.5,
                "trunk_sensor_cfg": trunk_sensor_cfg(),
                "trunk_contact_threshold": 1.0,
            },
        )
        self.rewards.stair_base_rear_feet_pose_on_first_step = RewTerm(
            func=gogo_mdp.stair_base_with_feet_pose_on_step_reward,
            weight=10.0,
            params={
                "asset_cfg": foot_asset_cfg(),
                "foot_indices": (2, 3),
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_depth": STAIR_STEP_DEPTH,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "step_index": 1,
                "base_clearance": 0.22,
                "base_x_fraction": 0.30,
                "foot_x_margin": -0.06,
                "trunk_sensor_cfg": trunk_sensor_cfg(),
                "trunk_contact_threshold": 1.0,
            },
        )
        self.rewards.stair_rear_feet_forward = RewTerm(
            func=gogo_mdp.stair_rear_feet_forward_reward,
            weight=4.0,
            params={
                "asset_cfg": foot_asset_cfg(),
                "edge_x": STAIR_FIRST_RISER_X,
                "foot_indices": (2, 3),
                "x_margin": -0.10,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
                "max_pitch_projected": 0.80,
                "max_roll_projected": 0.50,
            },
        )
        self.rewards.stair_rear_feet_min_x_progress = None
        self.rewards.stair_goal = RewTerm(
            func=gogo_mdp.stair_goal_reward,
            weight=6.0,
            params={
                "target_x": STAIR_TARGET_X,
                "target_z": STAIR_TARGET_Z,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "num_steps": STAIR_NUM_STEPS,
                "target_base_clearance": STAIR_TARGET_BASE_CLEARANCE,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
                "max_pitch_projected": 0.70,
                "max_roll_projected": 0.48,
            },
        )
        self.rewards.stair_success = RewTerm(
            func=gogo_mdp.stair_success_with_rear_feet_reward,
            weight=140.0,
            params={
                "target_x": STAIR_TARGET_X,
                "target_z": STAIR_TARGET_Z,
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "num_steps": STAIR_NUM_STEPS,
                "target_base_clearance": STAIR_TARGET_BASE_CLEARANCE,
                "foot_asset_cfg": foot_asset_cfg(),
                "rear_foot_indices": (2, 3),
                "foot_contact_sensor_cfg": foot_sensor_cfg(),
                "rear_contact_threshold": 0.5,
                "min_rear_contacts": 2,
                "foot_x_margin": -0.02,
                "foot_z_margin": 0.07,
                "foot_z_upper_margin": 0.17,
                "trunk_sensor_cfg": trunk_sensor_cfg(),
                "trunk_contact_threshold": 1.0,
                "max_pitch_projected": 0.58,
                "max_roll_projected": 0.42,
                "max_yaw": 0.45,
            },
        )
        self.rewards.stair_lateral_drift = RewTerm(
            func=gogo_mdp.stair_lateral_drift_penalty,
            weight=-0.6,
        )
        self.rewards.stair_yaw_drift = RewTerm(
            func=gogo_mdp.stair_yaw_drift_penalty,
            weight=-2.0,
            params={"yaw_allowance": 0.18},
        )
        self.rewards.stair_near_goal_heading = RewTerm(
            func=gogo_mdp.stair_near_goal_heading_reward,
            weight=2.5,
            params={
                "target_x": STAIR_TARGET_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "target_base_clearance": STAIR_TARGET_BASE_CLEARANCE,
                "num_steps": STAIR_NUM_STEPS,
                "x_fraction": 0.52,
                "min_height_fraction": 0.50,
                "yaw_std": 0.22,
            },
        )
        self.rewards.stair_tilt = RewTerm(
            func=gogo_mdp.stair_tilt_penalty,
            weight=-6.0,
            params={"pitch_allowance": 0.38},
        )
        self.rewards.stair_foot_clearance = RewTerm(
            func=gogo_mdp.swing_foot_clearance_reward,
            weight=0.15,
            params={"asset_cfg": foot_asset_cfg(), "target_height": 0.11, "std": 0.45},
        )
        self.rewards.stair_stuck = RewTerm(
            func=gogo_mdp.stair_stuck_penalty,
            weight=-0.35,
            params={"before_goal_x": STAIR_TARGET_X, "min_speed_x": 0.04},
        )
        self.rewards.stair_near_goal_timeout = RewTerm(
            func=gogo_mdp.stair_near_goal_timeout_penalty,
            weight=-1.2,
            params={
                "target_x": STAIR_TARGET_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "target_base_clearance": STAIR_TARGET_BASE_CLEARANCE,
                "num_steps": STAIR_NUM_STEPS,
                "min_progress_fraction": 0.58,
                "min_height_fraction": 0.50,
                "grace_time_s": 3.0,
            },
        )
        self.rewards.stair_final_approach = RewTerm(
            func=gogo_mdp.stair_final_approach_reward,
            weight=5.0,
            params={
                "target_x": STAIR_TARGET_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "target_base_clearance": STAIR_TARGET_BASE_CLEARANCE,
                "num_steps": STAIR_NUM_STEPS,
                "start_fraction": 0.72,
                "min_height_fraction": 0.55,
            },
        )
        self.rewards.stair_low_base_at_riser = RewTerm(
            func=gogo_mdp.stair_low_base_at_riser_penalty,
            weight=-1.5,
            params={
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "min_base_clearance": 0.21,
                "x_window": (-0.05, 0.50),
            },
        )
        self.rewards.trunk_contact = RewTerm(
            func=gogo_mdp.trunk_contact_penalty,
            weight=-4.0,
            params={"sensor_cfg": trunk_sensor_cfg(), "threshold": 1.0},
        )
        self.rewards.fall_failure = RewTerm(
            func=gogo_mdp.stair_fall,
            weight=-24.0,
            params={
                "min_base_height": 0.22,
                "max_lateral_offset": 1.20,
                "max_pitch_projected": 0.82,
                "max_roll_projected": 0.62,
            },
        )
        self.rewards.trunk_contact_failure = RewTerm(
            func=gogo_mdp.trunk_contact_failure,
            weight=-8.0,
            params={"sensor_cfg": trunk_sensor_cfg(), "threshold": 5.0, "grace_time_s": 0.5},
        )
        self.rewards.bad_yaw_failure = RewTerm(
            func=gogo_mdp.stair_bad_yaw,
            weight=-12.0,
            params={"max_yaw": 1.20, "gate_x": STAIR_FIRST_RISER_X - 0.15, "grace_time_s": 0.9},
        )

        self.terminations.success = DoneTerm(
            func=gogo_mdp.stair_climb_success_with_rear_feet,
            params={
                "target_x": STAIR_TARGET_X,
                "target_z": STAIR_TARGET_Z,
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "num_steps": STAIR_NUM_STEPS,
                "target_base_clearance": STAIR_TARGET_BASE_CLEARANCE,
                "foot_asset_cfg": foot_asset_cfg(),
                "rear_foot_indices": (2, 3),
                "foot_contact_sensor_cfg": foot_sensor_cfg(),
                "rear_contact_threshold": 0.5,
                "min_rear_contacts": 2,
                "foot_x_margin": -0.02,
                "foot_z_margin": 0.07,
                "foot_z_upper_margin": 0.17,
                "trunk_sensor_cfg": trunk_sensor_cfg(),
                "trunk_contact_threshold": 1.0,
                "max_pitch_projected": 0.58,
                "max_roll_projected": 0.42,
                "max_yaw": 0.45,
            },
        )
        self.terminations.fall = DoneTerm(
            func=gogo_mdp.stair_fall,
            params={
                "min_base_height": 0.16,
                "max_lateral_offset": 1.10,
                "max_pitch_projected": 0.90,
                "max_roll_projected": 0.70,
            },
        )
        self.terminations.bad_yaw = DoneTerm(
            func=gogo_mdp.stair_bad_yaw,
            params={"max_yaw": 1.55, "gate_x": STAIR_FIRST_RISER_X - 0.10, "grace_time_s": 1.0},
        )
        self.terminations.trunk_contact = DoneTerm(
            func=gogo_mdp.trunk_contact_failure,
            params={"sensor_cfg": trunk_sensor_cfg(), "threshold": 6.0, "grace_time_s": 0.35},
        )
        self.terminations.base_contact = None


class GogoA1StairClimbEnvCfg_PLAY(GogoA1StairClimbEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairBridgeEnvCfg(GogoA1StairClimbEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        def foot_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", body_names=foot_names, preserve_order=True)

        def trunk_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names="trunk")

        self.events.reset_base = EventTerm(
            func=gogo_mdp.reset_stair_climb_mixed_switch_and_near_step,
            mode="reset",
            params={
                "dataset_path": "outputs/flat_switch_states_x058_cmd025.pt",
                "switch_probability": 0.90,
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "near_step_x_range": (0.62, 0.70),
                "y_range": (-0.05, 0.05),
                "yaw_range": (-0.05, 0.05),
                "step_base_clearance": 0.29,
                "x_jitter": (-0.015, 0.025),
                "y_jitter": (-0.02, 0.02),
                "yaw_jitter": (-0.04, 0.04),
                "velocity_scale_range": (0.90, 1.10),
            },
        )

        bridge_start_x = 0.58
        bridge_target_x = 0.70
        bridge_target_height = STAIR_STEP_HEIGHT_RANGE[0] + 0.27
        self.episode_length_s = 4.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.16, 0.32)

        self.rewards.stair_success = RewTerm(
            func=gogo_mdp.stair_bridge_success_reward,
            weight=80.0,
            params={
                "target_x": bridge_target_x,
                "min_base_height": 0.27,
                "max_pitch_projected": 0.30,
                "max_roll_projected": 0.24,
                "max_yaw": 0.25,
                "min_steps": 8,
            },
        )
        self.rewards.stair_goal = RewTerm(
            func=gogo_mdp.stair_bridge_handoff_reward,
            weight=18.0,
            params={
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "x_std": 0.18,
                "height_std": 0.12,
                "yaw_std": 0.22,
                "pitch_std": 0.24,
                "roll_std": 0.20,
                "max_speed": 0.42,
            },
        )
        self.rewards.stair_forward_progress = RewTerm(
            func=gogo_mdp.stair_bridge_forward_band_reward,
            weight=7.0,
            params={
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "max_speed": 0.42,
                "yaw_allowance": 0.18,
                "roll_allowance": 0.24,
            },
        )
        self.rewards.stair_forward_velocity.weight = 0.1
        self.rewards.stair_height_progress.weight = 0.4
        self.rewards.stair_base_height_first.weight = 0.6
        self.rewards.stair_front_feet_on_step.weight = 0.4
        self.rewards.stair_base_on_first_step.weight = 0.8
        self.rewards.stair_rear_feet_on_step.weight = 0.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 0.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 0.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 0.0
        self.rewards.stair_rear_feet_forward.weight = 0.0
        self.rewards.stair_final_approach.weight = 0.0
        self.rewards.stair_near_goal_timeout.weight = -0.4
        self.rewards.stair_yaw_drift.weight = -7.0
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.08
        self.rewards.stair_lateral_drift.weight = -1.6
        self.rewards.stair_tilt.weight = -8.0
        self.rewards.action_rate_l2.weight = -0.07
        self.rewards.dof_torques_l2.weight = -3.2e-4
        self.rewards.trunk_contact.weight = -8.0
        self.rewards.fall_failure.weight = -35.0
        self.rewards.bad_yaw_failure.weight = -28.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 0.75,
            "gate_x": STAIR_FIRST_RISER_X - 0.18,
            "grace_time_s": 0.25,
        }

        self.terminations.success = DoneTerm(
            func=gogo_mdp.stair_bridge_success,
            params={
                "target_x": bridge_target_x,
                "min_base_height": 0.27,
                "max_pitch_projected": 0.30,
                "max_roll_projected": 0.24,
                "max_yaw": 0.25,
                "min_steps": 8,
            },
        )
        self.terminations.fall = DoneTerm(
            func=gogo_mdp.stair_fall,
            params={
                "min_base_height": 0.12,
                "max_lateral_offset": 0.75,
                "max_pitch_projected": 0.82,
                "max_roll_projected": 0.62,
            },
        )
        self.terminations.bad_yaw = DoneTerm(
            func=gogo_mdp.stair_bad_yaw,
            params={
                "max_yaw": 0.75,
                "gate_x": STAIR_FIRST_RISER_X - 0.18,
                "grace_time_s": 0.25,
            },
        )
        self.terminations.trunk_contact = DoneTerm(
            func=gogo_mdp.trunk_contact_failure,
            params={"sensor_cfg": trunk_sensor_cfg(), "threshold": 8.0, "grace_time_s": 0.35},
        )


class GogoA1StairBridgeEnvCfg_PLAY(GogoA1StairBridgeEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairBridgeX058ToX070RealEnvCfg(GogoA1StairBridgeEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.events.reset_base.params.update(
            {
                "dataset_path": "outputs/flat_switch_states_x058_cmd025_bridge_big.pt",
                "switch_probability": 0.94,
                "near_step_x_range": (0.56, 0.66),
                "x_jitter": (-0.004, 0.010),
                "y_jitter": (-0.018, 0.018),
                "yaw_jitter": (-0.018, 0.018),
                "velocity_scale_range": (0.92, 1.08),
            }
        )

        bridge_start_x = 0.58
        bridge_target_x = 0.70
        bridge_target_height = STAIR_STEP_HEIGHT_RANGE[0] + 0.27
        self.episode_length_s = 4.8
        self.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.20)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.rewards.stair_success.weight = 130.0
        self.rewards.stair_success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": 0.29,
                "max_pitch_projected": 0.24,
                "max_roll_projected": 0.18,
                "max_yaw": 0.18,
                "min_steps": 10,
            }
        )
        self.rewards.stair_goal.weight = 24.0
        self.rewards.stair_goal.params.update(
            {
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "x_std": 0.12,
                "height_std": 0.10,
                "yaw_std": 0.14,
                "pitch_std": 0.18,
                "roll_std": 0.14,
                "max_speed": 0.28,
            }
        )
        self.rewards.stair_forward_progress.weight = 8.0
        self.rewards.stair_forward_progress.params.update(
            {
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "max_speed": 0.34,
                "yaw_allowance": 0.12,
                "roll_allowance": 0.18,
            }
        )
        self.rewards.stair_forward_velocity.weight = 0.02
        self.rewards.stair_height_progress.weight = 0.25
        self.rewards.stair_base_height_first.weight = 0.35
        self.rewards.stair_front_feet_on_step.weight = 0.18
        self.rewards.stair_base_on_first_step.weight = 0.30
        self.rewards.stair_yaw_drift.weight = -9.0
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.05
        self.rewards.stair_lateral_drift.weight = -2.4
        self.rewards.stair_tilt.weight = -10.0
        self.rewards.action_rate_l2.weight = -0.08
        self.rewards.dof_torques_l2.weight = -3.4e-4
        self.rewards.trunk_contact.weight = -18.0
        self.rewards.fall_failure.weight = -45.0
        self.rewards.trunk_contact_failure.weight = -36.0
        self.rewards.bad_yaw_failure.weight = -36.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 0.45,
            "gate_x": STAIR_FIRST_RISER_X - 0.22,
            "grace_time_s": 0.20,
        }
        self.rewards.stair_near_goal_timeout.weight = -1.2
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 1.5
        self.rewards.stair_stuck.weight = -0.35
        self.rewards.stair_stuck.params["min_speed_x"] = 0.035

        self.terminations.success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": 0.29,
                "max_pitch_projected": 0.24,
                "max_roll_projected": 0.18,
                "max_yaw": 0.18,
                "min_steps": 10,
            }
        )
        self.terminations.fall.params = {
            "min_base_height": 0.12,
            "max_lateral_offset": 0.55,
            "max_pitch_projected": 0.72,
            "max_roll_projected": 0.52,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 0.62,
            "gate_x": STAIR_FIRST_RISER_X - 0.20,
            "grace_time_s": 0.30,
        }
        self.terminations.trunk_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
            "threshold": 6.0,
            "grace_time_s": 0.20,
        }


class GogoA1StairBridgeX058ToX070RealEnvCfg_PLAY(GogoA1StairBridgeX058ToX070RealEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairBridgeX058ToX068CenteredEnvCfg(GogoA1StairBridgeX058ToX070RealEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        bridge_start_x = 0.58
        bridge_target_x = 0.68
        bridge_target_height = STAIR_STEP_HEIGHT_RANGE[0] + 0.245
        self.episode_length_s = 4.4
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.18)

        self.events.reset_base.params.update(
            {
                "switch_probability": 0.98,
                "near_step_x_range": (0.56, 0.64),
                "x_jitter": (-0.003, 0.008),
                "y_jitter": (-0.010, 0.010),
                "yaw_jitter": (-0.012, 0.012),
                "velocity_scale_range": (0.88, 1.02),
            }
        )

        self.rewards.stair_success.weight = 160.0
        self.rewards.stair_success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": 0.265,
                "max_abs_y": 0.045,
                "max_pitch_projected": 0.28,
                "max_roll_projected": 0.20,
                "max_yaw": 0.16,
                "min_steps": 10,
            }
        )
        self.rewards.stair_goal.weight = 30.0
        self.rewards.stair_goal.params.update(
            {
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "x_std": 0.11,
                "y_std": 0.055,
                "height_std": 0.105,
                "yaw_std": 0.12,
                "pitch_std": 0.20,
                "roll_std": 0.15,
                "max_speed": 0.24,
            }
        )
        self.rewards.stair_forward_progress.weight = 7.0
        self.rewards.stair_forward_progress.params.update(
            {
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "max_speed": 0.30,
                "yaw_allowance": 0.10,
                "roll_allowance": 0.16,
            }
        )
        self.rewards.stair_lateral_drift.weight = -18.0
        self.rewards.stair_yaw_drift.weight = -10.0
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.035
        self.rewards.stair_tilt.weight = -9.0
        self.rewards.action_rate_l2.weight = -0.08
        self.rewards.dof_torques_l2.weight = -3.2e-4
        self.rewards.stair_near_goal_timeout.weight = -0.8
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 1.8
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 0.44,
            "gate_x": STAIR_FIRST_RISER_X - 0.22,
            "grace_time_s": 0.25,
        }

        self.terminations.success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": 0.265,
                "max_abs_y": 0.045,
                "max_pitch_projected": 0.28,
                "max_roll_projected": 0.20,
                "max_yaw": 0.16,
                "min_steps": 10,
            }
        )
        self.terminations.fall.params = {
            "min_base_height": 0.12,
            "max_lateral_offset": 0.45,
            "max_pitch_projected": 0.72,
            "max_roll_projected": 0.52,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 0.58,
            "gate_x": STAIR_FIRST_RISER_X - 0.20,
            "grace_time_s": 0.35,
        }


class GogoA1StairBridgeX058ToX068CenteredEnvCfg_PLAY(GogoA1StairBridgeX058ToX068CenteredEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairBridgeX058ToHighX080CenteredEnvCfg(GogoA1StairBridgeX058ToX068CenteredEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        bridge_start_x = 0.58
        bridge_target_x = 0.80
        bridge_target_height = STAIR_STEP_HEIGHT_RANGE[0] + 0.335
        self.episode_length_s = 5.4
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.16)

        self.events.reset_base.params.update(
            {
                "switch_probability": 0.98,
                "near_step_x_range": (0.56, 0.64),
                "x_jitter": (-0.003, 0.008),
                "y_jitter": (-0.010, 0.010),
                "yaw_jitter": (-0.012, 0.012),
                "velocity_scale_range": (0.88, 1.02),
            }
        )

        self.rewards.stair_success.weight = 220.0
        self.rewards.stair_success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": 0.355,
                "max_abs_y": 0.070,
                "max_pitch_projected": 0.42,
                "max_roll_projected": 0.28,
                "max_yaw": 0.20,
                "min_steps": 14,
            }
        )
        self.rewards.stair_goal.weight = 24.0
        self.rewards.stair_goal.params.update(
            {
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "x_std": 0.15,
                "y_std": 0.070,
                "height_std": 0.110,
                "yaw_std": 0.14,
                "pitch_std": 0.28,
                "roll_std": 0.18,
                "max_speed": 0.26,
            }
        )
        self.rewards.stair_forward_progress.weight = 5.5
        self.rewards.stair_forward_progress.params.update(
            {
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "max_speed": 0.28,
                "yaw_allowance": 0.12,
                "roll_allowance": 0.22,
            }
        )

        self.rewards.stair_height_progress.weight = 4.0
        self.rewards.stair_height_progress.params.update(
            {
                "base_z": 0.255,
                "target_base_clearance": 0.335,
                "max_pitch_projected": 0.64,
                "max_roll_projected": 0.38,
            }
        )
        self.rewards.stair_base_height_first.weight = 5.0
        self.rewards.stair_base_height_first.params.update(
            {
                "base_clearance": 0.315,
                "gate_margin": -0.10,
                "max_pitch_projected": 0.64,
                "max_roll_projected": 0.38,
            }
        )
        self.rewards.stair_front_feet_on_step.weight = 3.5
        self.rewards.stair_front_feet_on_step.params.update(
            {
                "x_margin": 0.00,
                "z_margin": 0.05,
                "max_pitch_projected": 0.72,
                "max_roll_projected": 0.42,
            }
        )
        self.rewards.stair_base_on_first_step.weight = 7.0
        self.rewards.stair_base_on_first_step.params.update(
            {
                "base_clearance": 0.315,
                "x_fraction": 0.08,
                "lateral_std": 0.24,
                "max_pitch_projected": 0.66,
                "max_roll_projected": 0.40,
            }
        )
        self.rewards.stair_rear_feet_forward.weight = 1.8
        self.rewards.stair_lateral_drift.weight = -10.0
        self.rewards.stair_yaw_drift.weight = -8.0
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.055
        self.rewards.stair_tilt.weight = -8.0
        self.rewards.stair_near_goal_timeout.weight = -0.7
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 2.4
        self.rewards.stair_stuck.weight = -0.45
        self.rewards.stair_stuck.params["min_speed_x"] = 0.025
        self.rewards.trunk_contact.weight = -22.0
        self.rewards.fall_failure.weight = -48.0
        self.rewards.trunk_contact_failure.weight = -42.0
        self.rewards.bad_yaw_failure.weight = -42.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 0.50,
            "gate_x": STAIR_FIRST_RISER_X - 0.18,
            "grace_time_s": 0.35,
        }

        self.terminations.success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": 0.355,
                "max_abs_y": 0.070,
                "max_pitch_projected": 0.42,
                "max_roll_projected": 0.28,
                "max_yaw": 0.20,
                "min_steps": 14,
            }
        )
        self.terminations.fall.params = {
            "min_base_height": 0.12,
            "max_lateral_offset": 0.42,
            "max_pitch_projected": 0.78,
            "max_roll_projected": 0.54,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 0.62,
            "gate_x": STAIR_FIRST_RISER_X - 0.16,
            "grace_time_s": 0.45,
        }
        self.terminations.trunk_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
            "threshold": 4.0,
            "grace_time_s": 0.18,
        }


class GogoA1StairBridgeX058ToHighX080CenteredEnvCfg_PLAY(GogoA1StairBridgeX058ToHighX080CenteredEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairBridgeX058ToMidX068HighEnvCfg(GogoA1StairBridgeX058ToHighX080CenteredEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        bridge_start_x = 0.58
        bridge_target_x = 0.68
        bridge_target_height = 0.305
        self.episode_length_s = 4.8
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.15)

        self.rewards.stair_forward_progress.weight = 2.0
        self.rewards.stair_forward_progress.params.update(
            {
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "max_speed": 0.22,
                "yaw_allowance": 0.10,
                "roll_allowance": 0.20,
            }
        )
        self.rewards.stair_xz_progress = RewTerm(
            func=gogo_mdp.stair_bridge_xz_progress_reward,
            weight=18.0,
            params={
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "base_z": 0.270,
                "y_std": 0.090,
                "yaw_std": 0.13,
                "pitch_allowance": 0.48,
                "roll_allowance": 0.24,
            },
        )
        self.rewards.stair_goal.weight = 34.0
        self.rewards.stair_goal.params.update(
            {
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "x_std": 0.095,
                "y_std": 0.080,
                "height_std": 0.050,
                "yaw_std": 0.13,
                "pitch_std": 0.30,
                "roll_std": 0.20,
                "max_speed": 0.22,
            }
        )
        self.rewards.stair_success.weight = 180.0
        self.rewards.stair_success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": bridge_target_height,
                "max_abs_y": 0.090,
                "max_pitch_projected": 0.48,
                "max_roll_projected": 0.30,
                "max_yaw": 0.24,
                "min_steps": 12,
            }
        )
        self.rewards.stair_height_progress.weight = 8.0
        self.rewards.stair_height_progress.params.update(
            {
                "base_z": 0.265,
                "target_z": bridge_target_height,
                "step_height_range": None,
                "target_base_clearance": 0.305,
                "gate_x": STAIR_FIRST_RISER_X - 0.18,
                "max_pitch_projected": 0.66,
                "max_roll_projected": 0.42,
            }
        )
        self.rewards.stair_base_height_first.weight = 8.0
        self.rewards.stair_base_height_first.params.update(
            {
                "base_clearance": 0.265,
                "gate_margin": -0.12,
                "max_pitch_projected": 0.66,
                "max_roll_projected": 0.42,
            }
        )
        self.rewards.stair_base_on_first_step.weight = 6.0
        self.rewards.stair_base_on_first_step.params.update(
            {
                "base_clearance": 0.265,
                "x_fraction": -0.10,
                "lateral_std": 0.22,
                "max_pitch_projected": 0.66,
                "max_roll_projected": 0.42,
            }
        )
        self.rewards.stair_front_feet_on_step.weight = 3.0
        self.rewards.stair_rear_feet_forward.weight = 1.2
        self.rewards.stair_low_base_at_riser.weight = -4.0
        self.rewards.stair_low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.245,
                "x_window": (-0.18, 0.18),
            }
        )
        self.rewards.stair_near_goal_timeout.weight = -2.4
        self.rewards.stair_near_goal_timeout.params.update(
            {
                "target_x": bridge_target_x,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "target_base_clearance": 0.265,
                "num_steps": 1,
                "min_progress_fraction": 0.82,
                "min_height_fraction": 0.78,
                "grace_time_s": 1.8,
            }
        )
        self.rewards.stair_lateral_drift.weight = -11.0
        self.rewards.stair_yaw_drift.weight = -9.0
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.045
        self.rewards.stair_tilt.weight = -8.0
        self.rewards.stair_stuck.weight = -0.70
        self.rewards.stair_stuck.params["min_speed_x"] = 0.020
        self.rewards.trunk_contact.weight = -26.0
        self.rewards.fall_failure.weight = -52.0
        self.rewards.trunk_contact_failure.weight = -48.0
        self.rewards.bad_yaw_failure.weight = -46.0

        self.terminations.success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": bridge_target_height,
                "max_abs_y": 0.090,
                "max_pitch_projected": 0.48,
                "max_roll_projected": 0.30,
                "max_yaw": 0.24,
                "min_steps": 12,
            }
        )
        self.terminations.fall.params = {
            "min_base_height": 0.12,
            "max_lateral_offset": 0.40,
            "max_pitch_projected": 0.80,
            "max_roll_projected": 0.56,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 0.58,
            "gate_x": STAIR_FIRST_RISER_X - 0.18,
            "grace_time_s": 0.45,
        }


class GogoA1StairBridgeX058ToMidX068HighEnvCfg_PLAY(GogoA1StairBridgeX058ToMidX068HighEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairBridgeX058ToMidX072HoldEnvCfg(GogoA1StairBridgeX058ToMidX068HighEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        bridge_start_x = 0.58
        bridge_hold_x = 0.64
        bridge_target_x = 0.72
        bridge_target_height = 0.315
        self.episode_length_s = 5.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.15)

        self.rewards.stair_forward_progress.weight = 1.2
        self.rewards.stair_forward_progress.params.update(
            {
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "max_speed": 0.22,
                "yaw_allowance": 0.10,
                "roll_allowance": 0.22,
            }
        )
        self.rewards.stair_xz_progress.weight = 14.0
        self.rewards.stair_xz_progress.params.update(
            {
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "base_z": 0.270,
                "y_std": 0.090,
                "yaw_std": 0.14,
                "pitch_allowance": 0.52,
                "roll_allowance": 0.26,
            }
        )
        self.rewards.stair_height_hold = RewTerm(
            func=gogo_mdp.stair_bridge_height_hold_reward,
            weight=22.0,
            params={
                "x_gate": bridge_hold_x,
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "height_std": 0.050,
                "y_std": 0.090,
                "yaw_std": 0.15,
                "pitch_allowance": 0.52,
                "roll_allowance": 0.27,
            },
        )
        self.rewards.stair_goal.weight = 30.0
        self.rewards.stair_goal.params.update(
            {
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "x_std": 0.110,
                "y_std": 0.085,
                "height_std": 0.060,
                "yaw_std": 0.15,
                "pitch_std": 0.32,
                "roll_std": 0.22,
                "max_speed": 0.24,
            }
        )
        self.rewards.stair_success.weight = 210.0
        self.rewards.stair_success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": bridge_target_height,
                "max_abs_y": 0.095,
                "max_pitch_projected": 0.50,
                "max_roll_projected": 0.32,
                "max_yaw": 0.26,
                "min_steps": 14,
            }
        )
        self.rewards.stair_height_progress.weight = 6.5
        self.rewards.stair_height_progress.params.update(
            {
                "base_z": 0.265,
                "target_z": bridge_target_height,
                "step_height_range": None,
                "target_base_clearance": 0.315,
                "gate_x": STAIR_FIRST_RISER_X - 0.18,
                "max_pitch_projected": 0.68,
                "max_roll_projected": 0.44,
            }
        )
        self.rewards.stair_base_height_first.weight = 7.5
        self.rewards.stair_base_height_first.params.update(
            {
                "base_clearance": 0.275,
                "gate_margin": -0.12,
                "max_pitch_projected": 0.68,
                "max_roll_projected": 0.44,
            }
        )
        self.rewards.stair_base_on_first_step.weight = 7.0
        self.rewards.stair_base_on_first_step.params.update(
            {
                "base_clearance": 0.275,
                "x_fraction": -0.04,
                "lateral_std": 0.22,
                "max_pitch_projected": 0.68,
                "max_roll_projected": 0.44,
            }
        )
        self.rewards.stair_near_goal_timeout.weight = -3.0
        self.rewards.stair_near_goal_timeout.params.update(
            {
                "target_x": bridge_target_x,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "target_base_clearance": 0.275,
                "num_steps": 1,
                "min_progress_fraction": 0.86,
                "min_height_fraction": 0.76,
                "grace_time_s": 1.8,
            }
        )
        self.rewards.stair_low_base_at_riser.weight = -5.0
        self.rewards.stair_low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.255,
                "x_window": (-0.16, 0.24),
            }
        )
        self.rewards.stair_lateral_drift.weight = -11.0
        self.rewards.stair_yaw_drift.weight = -9.0
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.050
        self.rewards.stair_stuck.weight = -0.85
        self.rewards.stair_stuck.params["min_speed_x"] = 0.020
        self.rewards.trunk_contact.weight = -28.0
        self.rewards.fall_failure.weight = -54.0
        self.rewards.trunk_contact_failure.weight = -50.0
        self.rewards.bad_yaw_failure.weight = -48.0

        self.terminations.success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": bridge_target_height,
                "max_abs_y": 0.095,
                "max_pitch_projected": 0.50,
                "max_roll_projected": 0.32,
                "max_yaw": 0.26,
                "min_steps": 14,
            }
        )
        self.terminations.bad_yaw.params = {
            "max_yaw": 0.60,
            "gate_x": STAIR_FIRST_RISER_X - 0.18,
            "grace_time_s": 0.45,
        }


class GogoA1StairBridgeX058ToMidX072HoldEnvCfg_PLAY(GogoA1StairBridgeX058ToMidX072HoldEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairBridgeX058ToMidX070HoldEnvCfg(GogoA1StairBridgeX058ToMidX068HighEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        bridge_start_x = 0.58
        bridge_hold_x = 0.63
        bridge_target_x = 0.70
        bridge_target_height = 0.310
        self.episode_length_s = 4.8
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.15)

        self.rewards.stair_forward_progress.weight = 1.4
        self.rewards.stair_forward_progress.params.update(
            {
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "max_speed": 0.22,
                "yaw_allowance": 0.12,
                "roll_allowance": 0.22,
            }
        )
        self.rewards.stair_xz_progress.weight = 17.0
        self.rewards.stair_xz_progress.params.update(
            {
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "base_z": 0.270,
                "y_std": 0.095,
                "yaw_std": 0.16,
                "pitch_allowance": 0.52,
                "roll_allowance": 0.28,
            }
        )
        self.rewards.stair_height_hold = RewTerm(
            func=gogo_mdp.stair_bridge_height_hold_reward,
            weight=18.0,
            params={
                "x_gate": bridge_hold_x,
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "height_std": 0.055,
                "y_std": 0.095,
                "yaw_std": 0.17,
                "pitch_allowance": 0.54,
                "roll_allowance": 0.30,
            },
        )
        self.rewards.stair_goal.weight = 32.0
        self.rewards.stair_goal.params.update(
            {
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "x_std": 0.110,
                "y_std": 0.090,
                "height_std": 0.065,
                "yaw_std": 0.17,
                "pitch_std": 0.34,
                "roll_std": 0.24,
                "max_speed": 0.24,
            }
        )
        self.rewards.stair_success.weight = 210.0
        self.rewards.stair_success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": bridge_target_height,
                "max_abs_y": 0.100,
                "max_pitch_projected": 0.52,
                "max_roll_projected": 0.34,
                "max_yaw": 0.30,
                "min_steps": 12,
            }
        )
        self.rewards.stair_height_progress.weight = 7.0
        self.rewards.stair_height_progress.params.update(
            {
                "base_z": 0.265,
                "target_z": bridge_target_height,
                "step_height_range": None,
                "target_base_clearance": 0.310,
                "gate_x": STAIR_FIRST_RISER_X - 0.18,
                "max_pitch_projected": 0.70,
                "max_roll_projected": 0.46,
            }
        )
        self.rewards.stair_base_height_first.weight = 7.5
        self.rewards.stair_base_height_first.params.update(
            {
                "base_clearance": 0.270,
                "gate_margin": -0.12,
                "max_pitch_projected": 0.70,
                "max_roll_projected": 0.46,
            }
        )
        self.rewards.stair_base_on_first_step.weight = 7.0
        self.rewards.stair_base_on_first_step.params.update(
            {
                "base_clearance": 0.270,
                "x_fraction": -0.06,
                "lateral_std": 0.24,
                "max_pitch_projected": 0.70,
                "max_roll_projected": 0.46,
            }
        )
        self.rewards.stair_near_goal_timeout.weight = -2.6
        self.rewards.stair_near_goal_timeout.params.update(
            {
                "target_x": bridge_target_x,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "target_base_clearance": 0.270,
                "num_steps": 1,
                "min_progress_fraction": 0.84,
                "min_height_fraction": 0.74,
                "grace_time_s": 1.8,
            }
        )
        self.rewards.stair_low_base_at_riser.weight = -4.4
        self.rewards.stair_low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.250,
                "x_window": (-0.16, 0.22),
            }
        )
        self.rewards.stair_lateral_drift.weight = -10.0
        self.rewards.stair_yaw_drift.weight = -8.5
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.055
        self.rewards.stair_stuck.weight = -0.75
        self.rewards.stair_stuck.params["min_speed_x"] = 0.020
        self.rewards.trunk_contact.weight = -26.0
        self.rewards.fall_failure.weight = -52.0
        self.rewards.trunk_contact_failure.weight = -48.0
        self.rewards.bad_yaw_failure.weight = -46.0

        self.terminations.success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": bridge_target_height,
                "max_abs_y": 0.100,
                "max_pitch_projected": 0.52,
                "max_roll_projected": 0.34,
                "max_yaw": 0.30,
                "min_steps": 12,
            }
        )
        self.terminations.bad_yaw.params = {
            "max_yaw": 0.62,
            "gate_x": STAIR_FIRST_RISER_X - 0.18,
            "grace_time_s": 0.45,
        }


class GogoA1StairBridgeX058ToMidX070HoldEnvCfg_PLAY(GogoA1StairBridgeX058ToMidX070HoldEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairBridgeX058ToMidX070HoldResetMixEnvCfg(GogoA1StairBridgeX058ToMidX070HoldEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        bridge_start_x = 0.58
        bridge_hold_x = 0.625
        bridge_target_x = 0.70
        bridge_target_height = 0.310
        self.episode_length_s = 4.8
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.16)

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "dataset_path": "outputs/flat_switch_states_x058_cmd025_bridge_big.pt",
                "approach_probability": 0.10,
                "switch_probability": 0.35,
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "approach_x_range": (-0.08, 0.18),
                "near_step_x_range": (0.60, 0.66),
                "y_range": (-0.040, 0.040),
                "yaw_range": (-0.045, 0.045),
                "base_height": 0.31,
                "step_base_clearance": 0.330,
                "x_jitter": (-0.004, 0.010),
                "y_jitter": (-0.010, 0.010),
                "yaw_jitter": (-0.014, 0.014),
                "velocity_scale_range": (0.90, 1.04),
                "approach_velocity_range": {
                    "x": (0.0, 0.045),
                    "y": (-0.010, 0.010),
                    "z": (-0.010, 0.010),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.025, 0.025),
                    "yaw": (-0.025, 0.025),
                },
                "near_velocity_range": {
                    "x": (0.0, 0.040),
                    "y": (-0.008, 0.008),
                    "z": (-0.008, 0.008),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.018, 0.018),
                    "yaw": (-0.018, 0.018),
                },
                "joint_position_scale_range": (0.99, 1.01),
                "joint_velocity_range": (-0.030, 0.030),
            }
        )
        self.events.reset_base = EventTerm(
            func=gogo_mdp.reset_stair_climb_mixed_approach_switch_and_near_step,
            mode="reset",
            params=reset_params,
        )

        self.rewards.stair_forward_progress.weight = 1.6
        self.rewards.stair_forward_progress.params.update(
            {
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "max_speed": 0.24,
                "yaw_allowance": 0.12,
                "roll_allowance": 0.24,
            }
        )
        self.rewards.stair_xz_progress.weight = 20.0
        self.rewards.stair_xz_progress.params.update(
            {
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "base_z": 0.270,
                "y_std": 0.090,
                "yaw_std": 0.150,
                "pitch_allowance": 0.54,
                "roll_allowance": 0.30,
            }
        )
        self.rewards.stair_height_hold.weight = 24.0
        self.rewards.stair_height_hold.params.update(
            {
                "x_gate": bridge_hold_x,
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "height_std": 0.060,
                "y_std": 0.090,
                "yaw_std": 0.155,
                "pitch_allowance": 0.55,
                "roll_allowance": 0.31,
            }
        )
        self.rewards.stair_goal.weight = 36.0
        self.rewards.stair_goal.params.update(
            {
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "x_std": 0.115,
                "y_std": 0.085,
                "height_std": 0.070,
                "yaw_std": 0.160,
                "pitch_std": 0.34,
                "roll_std": 0.24,
                "max_speed": 0.25,
            }
        )
        self.rewards.stair_success.weight = 240.0
        self.rewards.stair_success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": bridge_target_height,
                "max_abs_y": 0.095,
                "max_pitch_projected": 0.54,
                "max_roll_projected": 0.36,
                "max_yaw": 0.28,
                "min_steps": 12,
            }
        )
        self.rewards.stair_height_progress.weight = 9.0
        self.rewards.stair_height_progress.params.update(
            {
                "base_z": 0.265,
                "target_z": bridge_target_height,
                "step_height_range": None,
                "target_base_clearance": 0.310,
                "gate_x": STAIR_FIRST_RISER_X - 0.18,
                "max_pitch_projected": 0.72,
                "max_roll_projected": 0.48,
            }
        )
        self.rewards.stair_base_height_first.weight = 8.5
        self.rewards.stair_base_height_first.params.update(
            {
                "base_clearance": 0.280,
                "gate_margin": -0.14,
                "max_pitch_projected": 0.72,
                "max_roll_projected": 0.48,
            }
        )
        self.rewards.stair_base_on_first_step.weight = 8.0
        self.rewards.stair_base_on_first_step.params.update(
            {
                "base_clearance": 0.280,
                "x_fraction": -0.08,
                "lateral_std": 0.22,
                "max_pitch_projected": 0.72,
                "max_roll_projected": 0.48,
            }
        )
        self.rewards.stair_near_goal_timeout.weight = -3.4
        self.rewards.stair_near_goal_timeout.params.update(
            {
                "target_x": bridge_target_x,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "target_base_clearance": 0.280,
                "num_steps": 1,
                "min_progress_fraction": 0.82,
                "min_height_fraction": 0.74,
                "grace_time_s": 1.5,
            }
        )
        self.rewards.stair_low_base_at_riser.weight = -6.0
        self.rewards.stair_low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.258,
                "x_window": (-0.18, 0.24),
            }
        )
        self.rewards.low_base_at_riser_failure = RewTerm(
            func=gogo_mdp.stair_low_base_at_riser_failure,
            weight=-42.0,
            params={
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "min_base_clearance": 0.238,
                "x_window": (-0.12, 0.26),
                "grace_time_s": 0.55,
                "max_speed_x": 0.06,
            },
        )
        self.rewards.stair_lateral_drift.weight = -11.5
        self.rewards.stair_yaw_drift.weight = -9.5
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.050
        self.rewards.stair_stuck.weight = -1.00
        self.rewards.stair_stuck.params["min_speed_x"] = 0.024
        self.rewards.trunk_contact.weight = -30.0
        self.rewards.fall_failure.weight = -58.0
        self.rewards.trunk_contact_failure.weight = -54.0
        self.rewards.bad_yaw_failure.weight = -52.0

        self.terminations.success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": bridge_target_height,
                "max_abs_y": 0.095,
                "max_pitch_projected": 0.54,
                "max_roll_projected": 0.36,
                "max_yaw": 0.28,
                "min_steps": 12,
            }
        )
        self.terminations.fall.params = {
            "min_base_height": 0.13,
            "max_lateral_offset": 0.42,
            "max_pitch_projected": 0.82,
            "max_roll_projected": 0.58,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 0.62,
            "gate_x": STAIR_FIRST_RISER_X - 0.18,
            "grace_time_s": 0.45,
        }
        self.terminations.trunk_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
            "threshold": 4.0,
            "grace_time_s": 0.12,
        }
        self.terminations.low_base_at_riser = DoneTerm(
            func=gogo_mdp.stair_low_base_at_riser_failure,
            params={
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "min_base_clearance": 0.228,
                "x_window": (-0.12, 0.26),
                "grace_time_s": 0.65,
                "max_speed_x": 0.04,
            },
        )


class GogoA1StairBridgeX058ToMidX070HoldResetMixEnvCfg_PLAY(GogoA1StairBridgeX058ToMidX070HoldResetMixEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairBridgeX058ToMidX066LiftEnvCfg(GogoA1StairBridgeX058ToMidX070HoldResetMixEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        bridge_start_x = 0.58
        bridge_hold_x = 0.615
        bridge_target_x = 0.66
        bridge_target_height = 0.300
        self.episode_length_s = 4.4
        self.commands.base_velocity.ranges.lin_vel_x = (0.07, 0.15)

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "approach_probability": 0.00,
                "switch_probability": 0.72,
                "near_step_x_range": (0.595, 0.635),
                "y_range": (-0.045, 0.045),
                "yaw_range": (-0.050, 0.050),
                "step_base_clearance": 0.315,
                "x_jitter": (-0.004, 0.010),
                "y_jitter": (-0.010, 0.010),
                "yaw_jitter": (-0.014, 0.014),
                "velocity_scale_range": (0.92, 1.05),
                "near_velocity_range": {
                    "x": (0.0, 0.045),
                    "y": (-0.008, 0.008),
                    "z": (-0.008, 0.008),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.018, 0.018),
                    "yaw": (-0.018, 0.018),
                },
            }
        )
        self.events.reset_base.params = reset_params

        self.rewards.stair_forward_progress.weight = 2.2
        self.rewards.stair_forward_progress.params.update(
            {
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "max_speed": 0.20,
                "yaw_allowance": 0.13,
                "roll_allowance": 0.25,
            }
        )
        self.rewards.stair_xz_progress.weight = 26.0
        self.rewards.stair_xz_progress.params.update(
            {
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "base_z": 0.270,
                "y_std": 0.095,
                "yaw_std": 0.165,
                "pitch_allowance": 0.56,
                "roll_allowance": 0.32,
            }
        )
        self.rewards.stair_height_hold.weight = 18.0
        self.rewards.stair_height_hold.params.update(
            {
                "x_gate": bridge_hold_x,
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "height_std": 0.065,
                "y_std": 0.095,
                "yaw_std": 0.170,
                "pitch_allowance": 0.56,
                "roll_allowance": 0.32,
            }
        )
        self.rewards.stair_goal.weight = 40.0
        self.rewards.stair_goal.params.update(
            {
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "x_std": 0.090,
                "y_std": 0.090,
                "height_std": 0.070,
                "yaw_std": 0.170,
                "pitch_std": 0.36,
                "roll_std": 0.26,
                "max_speed": 0.25,
            }
        )
        self.rewards.stair_success.weight = 260.0
        self.rewards.stair_success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": bridge_target_height,
                "max_abs_y": 0.105,
                "max_pitch_projected": 0.56,
                "max_roll_projected": 0.38,
                "max_yaw": 0.32,
                "min_steps": 12,
            }
        )
        self.rewards.stair_height_progress.weight = 11.0
        self.rewards.stair_height_progress.params.update(
            {
                "base_z": 0.265,
                "target_z": bridge_target_height,
                "step_height_range": None,
                "target_base_clearance": 0.300,
                "gate_x": STAIR_FIRST_RISER_X - 0.18,
                "max_pitch_projected": 0.74,
                "max_roll_projected": 0.50,
            }
        )
        self.rewards.stair_base_height_first.weight = 9.5
        self.rewards.stair_base_height_first.params.update(
            {
                "base_clearance": 0.270,
                "gate_margin": -0.16,
                "max_pitch_projected": 0.74,
                "max_roll_projected": 0.50,
            }
        )
        self.rewards.stair_base_on_first_step.weight = 7.0
        self.rewards.stair_base_on_first_step.params.update(
            {
                "base_clearance": 0.270,
                "x_fraction": -0.12,
                "lateral_std": 0.24,
                "max_pitch_projected": 0.74,
                "max_roll_projected": 0.50,
            }
        )
        self.rewards.stair_near_goal_timeout.weight = -2.2
        self.rewards.stair_near_goal_timeout.params.update(
            {
                "target_x": bridge_target_x,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "target_base_clearance": 0.270,
                "num_steps": 1,
                "min_progress_fraction": 0.78,
                "min_height_fraction": 0.70,
                "grace_time_s": 1.9,
            }
        )
        self.rewards.stair_low_base_at_riser.weight = -4.0
        self.rewards.stair_low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.242,
                "x_window": (-0.18, 0.22),
            }
        )
        self.rewards.low_base_at_riser_failure.weight = -20.0
        self.rewards.low_base_at_riser_failure.params.update(
            {
                "min_base_clearance": 0.218,
                "x_window": (-0.10, 0.24),
                "grace_time_s": 1.05,
                "max_speed_x": 0.020,
            }
        )
        self.rewards.stair_lateral_drift.weight = -10.0
        self.rewards.stair_yaw_drift.weight = -8.0
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.060
        self.rewards.stair_stuck.weight = -0.70
        self.rewards.stair_stuck.params["min_speed_x"] = 0.018
        self.rewards.trunk_contact.weight = -22.0
        self.rewards.fall_failure.weight = -44.0
        self.rewards.trunk_contact_failure.weight = -34.0
        self.rewards.bad_yaw_failure.weight = -38.0

        self.terminations.success.params.update(
            {
                "target_x": bridge_target_x,
                "min_base_height": bridge_target_height,
                "max_abs_y": 0.105,
                "max_pitch_projected": 0.56,
                "max_roll_projected": 0.38,
                "max_yaw": 0.32,
                "min_steps": 12,
            }
        )
        self.terminations.fall.params = {
            "min_base_height": 0.12,
            "max_lateral_offset": 0.48,
            "max_pitch_projected": 0.84,
            "max_roll_projected": 0.60,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 0.68,
            "gate_x": STAIR_FIRST_RISER_X - 0.18,
            "grace_time_s": 0.55,
        }
        self.terminations.trunk_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
            "threshold": 5.0,
            "grace_time_s": 0.18,
        }
        self.terminations.low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.205,
                "x_window": (-0.10, 0.24),
                "grace_time_s": 1.15,
                "max_speed_x": 0.012,
            }
        )


class GogoA1StairBridgeX058ToMidX066LiftEnvCfg_PLAY(GogoA1StairBridgeX058ToMidX066LiftEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairFinisherX070EnvCfg(GogoA1StairClimbEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.events.reset_base = EventTerm(
            func=gogo_mdp.reset_stair_climb_from_switch_state_dataset,
            mode="reset",
            params={
                "dataset_path": "outputs/flat_switch_states_x070_cmd030_strict_true.pt",
                "x_jitter": (-0.02, 0.03),
                "y_jitter": (-0.02, 0.02),
                "yaw_jitter": (-0.035, 0.035),
                "velocity_scale_range": (0.90, 1.08),
            },
        )

        self.episode_length_s = 6.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.22, 0.34)

        self.rewards.stair_forward_progress.weight = 0.35
        self.rewards.stair_forward_velocity.weight = 0.55
        self.rewards.stair_forward_velocity.params["max_speed"] = 0.38
        self.rewards.stair_height_progress.weight = 1.0
        self.rewards.stair_base_height_first.weight = 0.8
        self.rewards.stair_front_feet_on_step.weight = 0.6
        self.rewards.stair_base_on_first_step.weight = 2.0
        self.rewards.stair_rear_feet_on_step.weight = 8.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 10.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 7.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 8.0
        self.rewards.stair_rear_feet_forward.weight = 5.5
        self.rewards.stair_goal.weight = 2.2
        self.rewards.stair_success.weight = 220.0
        self.rewards.stair_lateral_drift.weight = -2.0
        self.rewards.stair_yaw_drift.weight = -5.0
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.10
        self.rewards.stair_near_goal_heading.weight = 1.6
        self.rewards.stair_near_goal_heading.params["yaw_std"] = 0.16
        self.rewards.stair_tilt.weight = -12.0
        self.rewards.stair_tilt.params["pitch_allowance"] = 0.30
        self.rewards.stair_stuck.weight = -2.2
        self.rewards.stair_stuck.params["min_speed_x"] = 0.08
        self.rewards.stair_near_goal_timeout.weight = -6.0
        self.rewards.stair_near_goal_timeout.params["min_progress_fraction"] = 0.62
        self.rewards.stair_near_goal_timeout.params["min_height_fraction"] = 0.42
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 1.8
        self.rewards.stair_final_approach.weight = 4.0
        self.rewards.stair_final_approach.params["start_fraction"] = 0.62
        self.rewards.stair_low_base_at_riser.weight = -3.0
        self.rewards.trunk_contact.weight = -8.0
        self.rewards.fall_failure.weight = -45.0
        self.rewards.bad_yaw_failure.weight = -35.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 0.85,
            "gate_x": STAIR_FIRST_RISER_X - 0.12,
            "grace_time_s": 0.35,
        }
        self.rewards.track_lin_vel_xy_exp.weight = 0.08
        self.rewards.track_ang_vel_z_exp.weight = 0.03
        self.rewards.action_rate_l2.weight = -0.07
        self.rewards.dof_torques_l2.weight = -3.5e-4
        self.rewards.ang_vel_xy_l2.weight = -0.22

        self.terminations.fall.params = {
            "min_base_height": 0.12,
            "max_lateral_offset": 0.85,
            "max_pitch_projected": 0.88,
            "max_roll_projected": 0.70,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 0.95,
            "gate_x": STAIR_FIRST_RISER_X - 0.12,
            "grace_time_s": 0.35,
        }


class GogoA1StairFinisherX070EnvCfg_PLAY(GogoA1StairFinisherX070EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairFinisherX070MediumResetActionEnvCfg(GogoA1StairFinisherX070EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.events.reset_base.params = {
            "dataset_path": "outputs/flat_switch_states_x070_cmd030_medium_true_resetaction.pt",
            "x_jitter": (-0.015, 0.025),
            "y_jitter": (-0.025, 0.025),
            "yaw_jitter": (-0.025, 0.025),
            "velocity_scale_range": (0.94, 1.06),
        }

        self.episode_length_s = 5.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.28, 0.34)

        self.rewards.track_lin_vel_xy_exp.weight = 0.14
        self.rewards.stair_forward_progress.weight = 0.50
        self.rewards.stair_forward_velocity.weight = 0.75
        self.rewards.stair_forward_velocity.params["max_speed"] = 0.44
        self.rewards.stair_height_progress.weight = 1.2
        self.rewards.stair_base_height_first.weight = 1.0
        self.rewards.stair_base_on_first_step.weight = 2.5
        self.rewards.stair_rear_feet_on_step.weight = 9.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 11.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 8.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 9.0
        self.rewards.stair_rear_feet_forward.weight = 6.0
        self.rewards.stair_goal.weight = 2.6
        self.rewards.stair_success.weight = 260.0
        self.rewards.stair_lateral_drift.weight = -2.6
        self.rewards.stair_yaw_drift.weight = -7.0
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.08
        self.rewards.stair_near_goal_heading.weight = 2.0
        self.rewards.stair_near_goal_heading.params["yaw_std"] = 0.14
        self.rewards.stair_tilt.weight = -10.0
        self.rewards.stair_tilt.params["pitch_allowance"] = 0.30
        self.rewards.stair_stuck.weight = -3.0
        self.rewards.stair_stuck.params["min_speed_x"] = 0.09
        self.rewards.stair_near_goal_timeout.weight = -8.0
        self.rewards.stair_near_goal_timeout.params["min_progress_fraction"] = 0.58
        self.rewards.stair_near_goal_timeout.params["min_height_fraction"] = 0.42
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 1.2
        self.rewards.stair_final_approach.weight = 5.0
        self.rewards.stair_final_approach.params["start_fraction"] = 0.58
        self.rewards.trunk_contact.weight = -10.0
        self.rewards.fall_failure.weight = -50.0
        self.rewards.trunk_contact_failure.weight = -18.0
        self.rewards.bad_yaw_failure.weight = -42.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 0.85,
            "gate_x": STAIR_FIRST_RISER_X - 0.12,
            "grace_time_s": 0.30,
        }
        self.rewards.action_rate_l2.weight = -0.06
        self.rewards.dof_torques_l2.weight = -3.0e-4
        self.rewards.ang_vel_xy_l2.weight = -0.20

        self.terminations.fall.params = {
            "min_base_height": 0.14,
            "max_lateral_offset": 0.85,
            "max_pitch_projected": 0.86,
            "max_roll_projected": 0.68,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 0.95,
            "gate_x": STAIR_FIRST_RISER_X - 0.12,
            "grace_time_s": 0.35,
        }
        self.terminations.trunk_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
            "threshold": 5.0,
            "grace_time_s": 0.30,
        }


class GogoA1StairFinisherX070MediumResetActionEnvCfg_PLAY(GogoA1StairFinisherX070MediumResetActionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairFinisherX068CenteredResetActionEnvCfg(GogoA1StairFinisherX070EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.events.reset_base.params = {
            "dataset_path": "outputs/flat_switch_states_x070_cmd030_centered_relaxed_resetaction.pt",
            "x_jitter": (-0.012, 0.030),
            "y_jitter": (-0.018, 0.018),
            "yaw_jitter": (-0.020, 0.020),
            "velocity_scale_range": (0.95, 1.08),
        }

        self.episode_length_s = 5.5
        self.commands.base_velocity.ranges.lin_vel_x = (0.28, 0.34)

        self.rewards.track_lin_vel_xy_exp.weight = 0.12
        self.rewards.stair_forward_progress.weight = 0.52
        self.rewards.stair_forward_velocity.weight = 0.62
        self.rewards.stair_forward_velocity.params["max_speed"] = 0.42
        self.rewards.stair_height_progress.weight = 1.25
        self.rewards.stair_base_height_first.weight = 1.0
        self.rewards.stair_base_on_first_step.weight = 2.8
        self.rewards.stair_rear_feet_on_step.weight = 9.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 11.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 8.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 9.5
        self.rewards.stair_rear_feet_forward.weight = 6.0
        self.rewards.stair_goal.weight = 3.0
        self.rewards.stair_success.weight = 230.0
        self.rewards.stair_lateral_drift.weight = -1.8
        self.rewards.stair_yaw_drift.weight = -4.2
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.10
        self.rewards.stair_near_goal_heading.weight = 2.0
        self.rewards.stair_near_goal_heading.params["yaw_std"] = 0.16
        self.rewards.stair_tilt.weight = -8.0
        self.rewards.stair_tilt.params["pitch_allowance"] = 0.32
        self.rewards.stair_stuck.weight = -1.4
        self.rewards.stair_stuck.params["min_speed_x"] = 0.07
        self.rewards.stair_near_goal_timeout.weight = -4.2
        self.rewards.stair_near_goal_timeout.params["min_progress_fraction"] = 0.58
        self.rewards.stair_near_goal_timeout.params["min_height_fraction"] = 0.40
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 1.6
        self.rewards.stair_final_approach.weight = 5.2
        self.rewards.stair_final_approach.params["start_fraction"] = 0.58
        self.rewards.stair_low_base_at_riser.weight = -2.2
        self.rewards.trunk_contact.weight = -6.0
        self.rewards.fall_failure.weight = -40.0
        self.rewards.trunk_contact_failure.weight = -12.0
        self.rewards.bad_yaw_failure.weight = -28.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 0.90,
            "gate_x": STAIR_FIRST_RISER_X - 0.12,
            "grace_time_s": 0.35,
        }
        self.rewards.action_rate_l2.weight = -0.055
        self.rewards.dof_torques_l2.weight = -3.0e-4
        self.rewards.ang_vel_xy_l2.weight = -0.18

        self.terminations.fall.params = {
            "min_base_height": 0.13,
            "max_lateral_offset": 0.90,
            "max_pitch_projected": 0.88,
            "max_roll_projected": 0.70,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 1.00,
            "gate_x": STAIR_FIRST_RISER_X - 0.12,
            "grace_time_s": 0.35,
        }
        self.terminations.trunk_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
            "threshold": 6.0,
            "grace_time_s": 0.32,
        }


class GogoA1StairFinisherX068CenteredResetActionEnvCfg_PLAY(GogoA1StairFinisherX068CenteredResetActionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbMixedApproachX068EnvCfg(GogoA1StairClimbEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.events.reset_base = EventTerm(
            func=gogo_mdp.reset_stair_climb_mixed_approach_switch_and_near_step,
            mode="reset",
            params={
                "dataset_path": "outputs/flat_switch_states_x070_cmd030_centered_relaxed_resetaction.pt",
                "approach_probability": 0.30,
                "switch_probability": 0.45,
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "approach_x_range": (-0.20, 0.18),
                "near_step_x_range": (0.66, 0.86),
                "y_range": (-0.05, 0.05),
                "yaw_range": (-0.06, 0.06),
                "base_height": 0.31,
                "step_base_clearance": 0.29,
                "x_jitter": (-0.012, 0.030),
                "y_jitter": (-0.018, 0.018),
                "yaw_jitter": (-0.020, 0.020),
                "velocity_scale_range": (0.95, 1.08),
                "approach_velocity_range": {
                    "x": (0.0, 0.05),
                    "y": (-0.02, 0.02),
                    "z": (-0.02, 0.02),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.04, 0.04),
                    "yaw": (-0.04, 0.04),
                },
                "near_velocity_range": {
                    "x": (0.0, 0.03),
                    "y": (-0.01, 0.01),
                    "z": (-0.01, 0.01),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.02, 0.02),
                    "yaw": (-0.02, 0.02),
                },
                "joint_position_scale_range": (0.99, 1.01),
                "joint_velocity_range": (-0.03, 0.03),
            },
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.24, 0.34)
        self.rewards.stair_success.weight = 155.0
        self.rewards.stair_rear_feet_forward.weight = 4.5
        self.rewards.stair_goal.weight = 6.5
        self.rewards.stair_final_approach.weight = 5.6
        self.rewards.stair_near_goal_timeout.weight = -1.6
        self.rewards.stair_stuck.weight = -0.45
        self.rewards.stair_yaw_drift.weight = -2.2
        self.rewards.stair_lateral_drift.weight = -0.8
        self.rewards.trunk_contact.weight = -4.5
        self.rewards.fall_failure.weight = -26.0
        self.rewards.bad_yaw_failure.weight = -14.0


class GogoA1StairClimbMixedApproachX068EnvCfg_PLAY(GogoA1StairClimbMixedApproachX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbMixedApproachX068YawEnvCfg(GogoA1StairClimbMixedApproachX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.stair_yaw_drift.weight = -3.6
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.12
        self.rewards.stair_near_goal_heading.weight = 3.2
        self.rewards.stair_near_goal_heading.params["yaw_std"] = 0.16
        self.rewards.stair_lateral_drift.weight = -1.0
        self.rewards.bad_yaw_failure.weight = -24.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 0.95,
            "gate_x": STAIR_FIRST_RISER_X - 0.14,
            "grace_time_s": 0.55,
        }

        self.terminations.bad_yaw.params = {
            "max_yaw": 1.35,
            "gate_x": STAIR_FIRST_RISER_X - 0.10,
            "grace_time_s": 0.75,
        }


class GogoA1StairClimbMixedApproachX068YawEnvCfg_PLAY(GogoA1StairClimbMixedApproachX068YawEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbCurriculumX068EnvCfg(GogoA1StairClimbMixedApproachX068YawEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 4
        self.curriculum.terrain_levels = CurrTerm(
            func=gogo_mdp.stair_climb_terrain_levels,
            params={
                "target_x": STAIR_TARGET_X,
                "target_z": STAIR_TARGET_Z,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "num_steps": STAIR_NUM_STEPS,
                "target_base_clearance": STAIR_TARGET_BASE_CLEARANCE,
            },
        )

        self.events.reset_base.params["approach_probability"] = 0.45
        self.events.reset_base.params["switch_probability"] = 0.20
        self.events.reset_base.params["near_step_x_range"] = (0.64, 0.86)
        self.events.reset_base.params["step_base_clearance"] = 0.31
        self.events.reset_base.params["x_jitter"] = (-0.010, 0.020)
        self.events.reset_base.params["y_jitter"] = (-0.015, 0.015)
        self.events.reset_base.params["yaw_jitter"] = (-0.015, 0.015)
        self.events.reset_base.params["velocity_scale_range"] = (0.90, 1.05)

        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.32)
        self.rewards.stair_success.weight = 175.0
        self.rewards.stair_goal.weight = 7.5
        self.rewards.stair_height_progress.weight = 2.8
        self.rewards.stair_base_height_first.weight = 2.8
        self.rewards.stair_base_on_first_step.weight = 10.0
        self.rewards.stair_rear_feet_on_step.weight = 10.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 12.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 11.0
        self.rewards.stair_final_approach.weight = 6.5
        self.rewards.stair_near_goal_timeout.weight = -2.0
        self.rewards.stair_low_base_at_riser.weight = -2.5
        self.rewards.trunk_contact.weight = -6.0
        self.rewards.fall_failure.weight = -30.0
        self.rewards.trunk_contact_failure.weight = -12.0
        self.rewards.bad_yaw_failure.weight = -26.0


class GogoA1StairClimbCurriculumX068EnvCfg_PLAY(GogoA1StairClimbCurriculumX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbRandomLowMidX068EnvCfg(GogoA1StairClimbMixedApproachX068YawEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.terrain.max_init_terrain_level = 2
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 4
        self.curriculum.terrain_levels = None

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "min_level": 0,
                "max_level": 2,
                "approach_probability": 0.50,
                "switch_probability": 0.12,
                "near_step_x_range": (0.62, 0.82),
                "step_base_clearance": 0.33,
                "x_jitter": (-0.008, 0.016),
                "y_jitter": (-0.012, 0.012),
                "yaw_jitter": (-0.012, 0.012),
                "velocity_scale_range": (0.90, 1.03),
            }
        )
        self.events.reset_base = EventTerm(
            func=gogo_mdp.reset_random_terrain_level_then_stair_climb_mixed_approach,
            mode="reset",
            params=reset_params,
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.18, 0.28)
        self.rewards.stair_success.weight = 185.0
        self.rewards.stair_forward_progress.weight = 0.30
        self.rewards.stair_forward_velocity.weight = 0.12
        self.rewards.stair_goal.weight = 5.0
        self.rewards.stair_height_progress.weight = 2.2
        self.rewards.stair_base_height_first.weight = 2.2
        self.rewards.stair_base_on_first_step.weight = 8.0
        self.rewards.stair_rear_feet_on_step.weight = 9.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 11.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 9.0
        self.rewards.stair_final_approach.weight = 4.0
        self.rewards.stair_near_goal_timeout.weight = -2.4
        self.rewards.stair_low_base_at_riser.weight = -4.0
        self.rewards.trunk_contact.weight = -14.0
        self.rewards.fall_failure.weight = -34.0
        self.rewards.trunk_contact_failure.weight = -28.0
        self.rewards.bad_yaw_failure.weight = -22.0
        self.terminations.trunk_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
            "threshold": 2.0,
            "grace_time_s": 0.08,
        }


class GogoA1StairClimbRandomLowMidX068EnvCfg_PLAY(GogoA1StairClimbRandomLowMidX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbLowStepStableX068EnvCfg(GogoA1StairClimbMixedApproachX068YawEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 4
        self.curriculum.terrain_levels = None

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "min_level": 0,
                "max_level": 0,
                "approach_probability": 0.58,
                "switch_probability": 0.06,
                "near_step_x_range": (0.60, 0.80),
                "step_base_clearance": 0.34,
                "x_jitter": (-0.006, 0.014),
                "y_jitter": (-0.010, 0.010),
                "yaw_jitter": (-0.010, 0.010),
                "velocity_scale_range": (0.88, 1.02),
                "approach_velocity_range": {
                    "x": (0.0, 0.04),
                    "y": (-0.01, 0.01),
                    "z": (-0.01, 0.01),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.025, 0.025),
                    "yaw": (-0.025, 0.025),
                },
                "near_velocity_range": {
                    "x": (0.0, 0.025),
                    "y": (-0.008, 0.008),
                    "z": (-0.008, 0.008),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.018, 0.018),
                    "yaw": (-0.018, 0.018),
                },
            }
        )
        self.events.reset_base = EventTerm(
            func=gogo_mdp.reset_random_terrain_level_then_stair_climb_mixed_approach,
            mode="reset",
            params=reset_params,
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.16, 0.26)
        self.rewards.track_lin_vel_xy_exp.weight = 0.12
        self.rewards.stair_forward_progress.weight = 0.12
        self.rewards.stair_forward_velocity.weight = 0.06
        self.rewards.stair_goal.weight = 3.8
        self.rewards.stair_height_progress.weight = 1.8
        self.rewards.stair_base_height_first.weight = 1.5
        self.rewards.stair_base_on_first_step.weight = 5.0
        self.rewards.stair_rear_feet_on_step.weight = 10.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 13.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 7.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 11.0
        self.rewards.stair_rear_feet_forward.weight = 5.0
        self.rewards.stair_success.weight = 230.0
        self.rewards.stair_success.params["foot_x_margin"] = 0.00
        self.rewards.stair_success.params["max_yaw"] = 0.36
        self.rewards.stair_success.params["max_pitch_projected"] = 0.54
        self.rewards.stair_success.params["max_roll_projected"] = 0.38
        self.terminations.success.params["foot_x_margin"] = 0.00
        self.terminations.success.params["max_yaw"] = 0.36
        self.terminations.success.params["max_pitch_projected"] = 0.54
        self.terminations.success.params["max_roll_projected"] = 0.38
        self.rewards.stair_final_approach.weight = 2.4
        self.rewards.stair_near_goal_timeout.weight = -3.2
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 2.2
        self.rewards.stair_stuck.weight = -0.8
        self.rewards.stair_stuck.params["min_speed_x"] = 0.055
        self.rewards.stair_low_base_at_riser.weight = -7.0
        self.rewards.stair_low_base_at_riser.params["min_base_clearance"] = 0.25
        self.rewards.stair_yaw_drift.weight = -5.0
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.10
        self.rewards.stair_near_goal_heading.weight = 4.0
        self.rewards.stair_near_goal_heading.params["yaw_std"] = 0.14
        self.rewards.stair_lateral_drift.weight = -1.2
        self.rewards.stair_tilt.weight = -8.0
        self.rewards.trunk_contact.weight = -26.0
        self.rewards.trunk_contact.params["threshold"] = 0.8
        self.rewards.fall_failure.weight = -40.0
        self.rewards.trunk_contact_failure.weight = -48.0
        self.rewards.trunk_contact_failure.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
            "threshold": 1.2,
            "grace_time_s": 0.02,
        }
        self.rewards.bad_yaw_failure.weight = -38.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 0.78,
            "gate_x": STAIR_FIRST_RISER_X - 0.16,
            "grace_time_s": 0.42,
        }
        self.rewards.low_base_at_riser_failure = RewTerm(
            func=gogo_mdp.stair_low_base_at_riser_failure,
            weight=-36.0,
            params={
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "min_base_clearance": 0.23,
                "x_window": (-0.05, 0.42),
                "grace_time_s": 0.62,
                "max_speed_x": 0.09,
            },
        )

        self.terminations.bad_yaw.params = {
            "max_yaw": 1.05,
            "gate_x": STAIR_FIRST_RISER_X - 0.12,
            "grace_time_s": 0.55,
        }
        self.terminations.trunk_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
            "threshold": 1.5,
            "grace_time_s": 0.04,
        }
        self.terminations.low_base_at_riser = DoneTerm(
            func=gogo_mdp.stair_low_base_at_riser_failure,
            params={
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "min_base_clearance": 0.22,
                "x_window": (-0.05, 0.40),
                "grace_time_s": 0.78,
                "max_speed_x": 0.06,
            },
        )


class GogoA1StairClimbLowStepStableX068EnvCfg_PLAY(GogoA1StairClimbLowStepStableX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbSwitchNearLowX068EnvCfg(GogoA1StairClimbLowStepStableX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "approach_probability": 0.08,
                "switch_probability": 0.52,
                "near_step_x_range": (0.58, 0.74),
                "step_base_clearance": 0.335,
                "x_jitter": (-0.010, 0.018),
                "y_jitter": (-0.012, 0.012),
                "yaw_jitter": (-0.018, 0.018),
                "velocity_scale_range": (0.92, 1.08),
                "approach_velocity_range": {
                    "x": (0.0, 0.03),
                    "y": (-0.008, 0.008),
                    "z": (-0.008, 0.008),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.020, 0.020),
                    "yaw": (-0.020, 0.020),
                },
                "near_velocity_range": {
                    "x": (0.0, 0.035),
                    "y": (-0.008, 0.008),
                    "z": (-0.008, 0.008),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.015, 0.015),
                    "yaw": (-0.015, 0.015),
                },
            }
        )
        self.events.reset_base.params = reset_params

        self.commands.base_velocity.ranges.lin_vel_x = (0.18, 0.28)
        self.rewards.stair_forward_progress.weight = 0.08
        self.rewards.stair_forward_velocity.weight = 0.04
        self.rewards.stair_goal.weight = 4.2
        self.rewards.stair_height_progress.weight = 2.2
        self.rewards.stair_base_height_first.weight = 2.0
        self.rewards.stair_base_on_first_step.weight = 6.0
        self.rewards.stair_rear_feet_on_step.weight = 12.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 15.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 8.5
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 13.0
        self.rewards.stair_success.weight = 260.0
        self.rewards.stair_success.params["max_yaw"] = 0.42
        self.terminations.success.params["max_yaw"] = 0.42
        self.rewards.stair_final_approach.weight = 3.0
        self.rewards.stair_near_goal_timeout.weight = -2.5
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 2.8
        self.rewards.stair_low_base_at_riser.weight = -6.0
        self.rewards.stair_yaw_drift.weight = -3.2
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.16
        self.rewards.stair_near_goal_heading.weight = 3.4
        self.rewards.stair_near_goal_heading.params["yaw_std"] = 0.20
        self.rewards.trunk_contact.weight = -24.0
        self.rewards.trunk_contact_failure.weight = -44.0
        self.rewards.bad_yaw_failure.weight = -14.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 1.10,
            "gate_x": STAIR_FIRST_RISER_X - 0.12,
            "grace_time_s": 1.15,
        }
        self.rewards.low_base_at_riser_failure.weight = -30.0
        self.terminations.bad_yaw.params = {
            "max_yaw": 1.55,
            "gate_x": STAIR_FIRST_RISER_X - 0.06,
            "grace_time_s": 1.6,
        }
        self.terminations.trunk_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
            "threshold": 1.8,
            "grace_time_s": 0.08,
        }
        self.terminations.low_base_at_riser.params = {
            "first_riser_x": STAIR_FIRST_RISER_X,
            "step_height_range": STAIR_STEP_HEIGHT_RANGE,
            "min_base_clearance": 0.215,
            "x_window": (-0.05, 0.40),
            "grace_time_s": 1.0,
            "max_speed_x": 0.05,
        }


class GogoA1StairClimbSwitchNearLowX068EnvCfg_PLAY(GogoA1StairClimbSwitchNearLowX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbNearFinisherLowX068EnvCfg(GogoA1StairClimbSwitchNearLowX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "approach_probability": 0.0,
                "switch_probability": 0.0,
                "near_step_x_range": (0.61, 0.77),
                "y_range": (-0.035, 0.035),
                "yaw_range": (-0.035, 0.035),
                "step_base_clearance": 0.345,
                "near_velocity_range": {
                    "x": (0.0, 0.045),
                    "y": (-0.006, 0.006),
                    "z": (-0.006, 0.006),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.012, 0.012),
                    "yaw": (-0.012, 0.012),
                },
            }
        )
        self.events.reset_base.params = reset_params

        self.commands.base_velocity.ranges.lin_vel_x = (0.16, 0.24)
        self.rewards.stair_success.weight = 310.0
        self.rewards.stair_success.params["max_yaw"] = 0.50
        self.terminations.success.params["max_yaw"] = 0.50
        self.rewards.stair_rear_feet_on_step.weight = 15.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 18.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 11.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 16.0
        self.rewards.stair_final_approach.weight = 4.0
        self.rewards.stair_goal.weight = 5.0
        self.rewards.stair_near_goal_timeout.weight = -1.2
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 3.6
        self.rewards.stair_low_base_at_riser.weight = -3.5
        self.rewards.low_base_at_riser_failure.weight = -18.0
        self.rewards.stair_yaw_drift.weight = -2.0
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.22
        self.rewards.bad_yaw_failure.weight = -8.0
        self.rewards.trunk_contact.weight = -22.0
        self.rewards.trunk_contact_failure.weight = -42.0
        self.terminations.low_base_at_riser.params = {
            "first_riser_x": STAIR_FIRST_RISER_X,
            "step_height_range": STAIR_STEP_HEIGHT_RANGE,
            "min_base_clearance": 0.205,
            "x_window": (-0.05, 0.40),
            "grace_time_s": 1.4,
            "max_speed_x": 0.035,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 1.70,
            "gate_x": STAIR_FIRST_RISER_X - 0.04,
            "grace_time_s": 1.8,
        }


class GogoA1StairClimbNearFinisherLowX068EnvCfg_PLAY(GogoA1StairClimbNearFinisherLowX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbNearFinisherRelaxedX068EnvCfg(GogoA1StairClimbNearFinisherLowX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        target_base_clearance = 0.22
        for term_name in (
            "stair_goal",
            "stair_success",
            "stair_near_goal_heading",
            "stair_near_goal_timeout",
            "stair_final_approach",
        ):
            term = getattr(self.rewards, term_name, None)
            if term is not None and "target_base_clearance" in term.params:
                term.params["target_base_clearance"] = target_base_clearance

        self.rewards.stair_success.weight = 240.0
        self.rewards.stair_success.params.update(
            {
                "target_base_clearance": target_base_clearance,
                "min_rear_contacts": 1,
                "foot_x_margin": 0.0,
                "max_pitch_projected": 0.60,
                "max_roll_projected": 0.45,
                "max_yaw": 0.55,
            }
        )
        self.terminations.success.params.update(
            {
                "target_base_clearance": target_base_clearance,
                "min_rear_contacts": 1,
                "foot_x_margin": 0.0,
                "max_pitch_projected": 0.60,
                "max_roll_projected": 0.45,
                "max_yaw": 0.55,
            }
        )

        self.rewards.stair_goal.weight = 4.6
        self.rewards.stair_final_approach.weight = 4.4
        self.rewards.stair_near_goal_heading.weight = 3.2
        self.rewards.stair_near_goal_heading.params["yaw_std"] = 0.22
        self.rewards.stair_near_goal_timeout.weight = -0.6
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 4.2
        self.rewards.stair_rear_feet_on_step.weight = 15.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 22.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 12.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 18.0
        self.rewards.stair_low_base_at_riser.weight = -1.5
        self.rewards.stair_low_base_at_riser.params["min_base_clearance"] = 0.205
        self.rewards.low_base_at_riser_failure.weight = -8.0
        self.terminations.low_base_at_riser.params = {
            "first_riser_x": STAIR_FIRST_RISER_X,
            "step_height_range": STAIR_STEP_HEIGHT_RANGE,
            "min_base_clearance": 0.195,
            "x_window": (-0.05, 0.40),
            "grace_time_s": 1.8,
            "max_speed_x": 0.02,
        }


class GogoA1StairClimbNearFinisherRelaxedX068EnvCfg_PLAY(GogoA1StairClimbNearFinisherRelaxedX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbNearFinisherEdgeX068EnvCfg(GogoA1StairClimbNearFinisherRelaxedX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.stair_success.weight = 220.0
        self.rewards.stair_success.params.update(
            {
                "foot_x_margin": -0.04,
                "target_base_clearance": 0.22,
                "min_rear_contacts": 1,
                "max_pitch_projected": 0.58,
                "max_roll_projected": 0.42,
                "max_yaw": 0.50,
            }
        )
        self.terminations.success.params.update(
            {
                "foot_x_margin": -0.04,
                "target_base_clearance": 0.22,
                "min_rear_contacts": 1,
                "max_pitch_projected": 0.58,
                "max_roll_projected": 0.42,
                "max_yaw": 0.50,
            }
        )
        self.rewards.stair_goal.weight = 5.0
        self.rewards.stair_final_approach.weight = 5.0
        self.rewards.stair_near_goal_timeout.weight = -1.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 20.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 18.0


class GogoA1StairClimbNearFinisherEdgeX068EnvCfg_PLAY(GogoA1StairClimbNearFinisherEdgeX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbSwitchNearEdgeX068EnvCfg(GogoA1StairClimbSwitchNearLowX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "approach_probability": 0.10,
                "switch_probability": 0.56,
                "near_step_x_range": (0.56, 0.74),
                "step_base_clearance": 0.335,
                "x_jitter": (-0.012, 0.018),
                "y_jitter": (-0.018, 0.018),
                "yaw_jitter": (-0.030, 0.030),
                "velocity_scale_range": (0.90, 1.10),
                "approach_velocity_range": {
                    "x": (0.0, 0.05),
                    "y": (-0.010, 0.010),
                    "z": (-0.010, 0.010),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.024, 0.024),
                    "yaw": (-0.024, 0.024),
                },
                "near_velocity_range": {
                    "x": (0.0, 0.045),
                    "y": (-0.010, 0.010),
                    "z": (-0.010, 0.010),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.018, 0.018),
                    "yaw": (-0.018, 0.018),
                },
            }
        )
        self.events.reset_base.params = reset_params

        target_base_clearance = 0.22
        for term_name in (
            "stair_goal",
            "stair_success",
            "stair_near_goal_heading",
            "stair_near_goal_timeout",
            "stair_final_approach",
        ):
            term = getattr(self.rewards, term_name, None)
            if term is not None and "target_base_clearance" in term.params:
                term.params["target_base_clearance"] = target_base_clearance

        self.commands.base_velocity.ranges.lin_vel_x = (0.17, 0.26)
        self.rewards.stair_success.weight = 220.0
        self.rewards.stair_success.params.update(
            {
                "target_base_clearance": target_base_clearance,
                "min_rear_contacts": 1,
                "foot_x_margin": -0.04,
                "max_pitch_projected": 0.58,
                "max_roll_projected": 0.42,
                "max_yaw": 0.50,
            }
        )
        self.terminations.success.params.update(
            {
                "target_base_clearance": target_base_clearance,
                "min_rear_contacts": 1,
                "foot_x_margin": -0.04,
                "max_pitch_projected": 0.58,
                "max_roll_projected": 0.42,
                "max_yaw": 0.50,
            }
        )
        self.rewards.stair_goal.weight = 4.8
        self.rewards.stair_height_progress.weight = 2.0
        self.rewards.stair_base_on_first_step.weight = 6.0
        self.rewards.stair_rear_feet_on_step.weight = 12.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 18.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 16.0
        self.rewards.stair_final_approach.weight = 4.8
        self.rewards.stair_near_goal_timeout.weight = -1.4
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 3.4
        self.rewards.stair_low_base_at_riser.weight = -2.8
        self.rewards.stair_low_base_at_riser.params["min_base_clearance"] = 0.215
        self.rewards.stair_yaw_drift.weight = -2.4
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.18
        self.rewards.trunk_contact.weight = -22.0
        self.rewards.trunk_contact_failure.weight = -42.0
        self.rewards.bad_yaw_failure.weight = -10.0
        self.rewards.low_base_at_riser_failure.weight = -16.0
        self.terminations.low_base_at_riser.params = {
            "first_riser_x": STAIR_FIRST_RISER_X,
            "step_height_range": STAIR_STEP_HEIGHT_RANGE,
            "min_base_clearance": 0.205,
            "x_window": (-0.05, 0.40),
            "grace_time_s": 1.25,
            "max_speed_x": 0.035,
        }


class GogoA1StairClimbSwitchNearEdgeX068EnvCfg_PLAY(GogoA1StairClimbSwitchNearEdgeX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbSwitchX058EdgeX068EnvCfg(GogoA1StairClimbSwitchNearEdgeX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "dataset_path": "outputs/flat_switch_states_x058_cmd025.pt",
                "approach_probability": 0.02,
                "switch_probability": 0.88,
                "near_step_x_range": (0.58, 0.74),
                "x_jitter": (-0.006, 0.012),
                "y_jitter": (-0.020, 0.020),
                "yaw_jitter": (-0.020, 0.020),
                "velocity_scale_range": (0.90, 1.04),
            }
        )
        self.events.reset_base.params = reset_params

        self.commands.base_velocity.ranges.lin_vel_x = (0.22, 0.30)
        self.rewards.track_lin_vel_xy_exp.weight = 0.10
        self.rewards.stair_forward_progress.weight = 0.10
        self.rewards.stair_forward_velocity.weight = 0.08
        self.rewards.stair_yaw_drift.weight = -3.4
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.12
        self.rewards.stair_near_goal_heading.weight = 4.2
        self.rewards.stair_near_goal_heading.params["yaw_std"] = 0.16
        self.rewards.bad_yaw_failure.weight = -18.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 1.00,
            "gate_x": STAIR_FIRST_RISER_X - 0.14,
            "grace_time_s": 0.75,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 1.35,
            "gate_x": STAIR_FIRST_RISER_X - 0.06,
            "grace_time_s": 1.15,
        }
        self.rewards.stair_stuck.weight = -1.0
        self.rewards.stair_stuck.params["min_speed_x"] = 0.06
        self.rewards.stair_near_goal_timeout.weight = -1.8
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 2.8


class GogoA1StairClimbSwitchX058EdgeX068EnvCfg_PLAY(GogoA1StairClimbSwitchX058EdgeX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbSwitchX070EdgeX068EnvCfg(GogoA1StairClimbSwitchNearEdgeX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "dataset_path": "outputs/flat_switch_states_x070_cmd025_edge_big.pt",
                "approach_probability": 0.00,
                "switch_probability": 0.82,
                "near_step_x_range": (0.68, 0.78),
                "x_jitter": (-0.006, 0.012),
                "y_jitter": (-0.018, 0.018),
                "yaw_jitter": (-0.020, 0.020),
                "velocity_scale_range": (0.92, 1.06),
                "near_velocity_range": {
                    "x": (0.0, 0.035),
                    "y": (-0.008, 0.008),
                    "z": (-0.008, 0.008),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.016, 0.016),
                    "yaw": (-0.016, 0.016),
                },
            }
        )
        self.events.reset_base.params = reset_params

        self.commands.base_velocity.ranges.lin_vel_x = (0.18, 0.26)
        self.rewards.track_lin_vel_xy_exp.weight = 0.10
        self.rewards.stair_forward_progress.weight = 0.08
        self.rewards.stair_forward_velocity.weight = 0.05
        self.rewards.stair_goal.weight = 5.2
        self.rewards.stair_final_approach.weight = 5.2
        self.rewards.stair_near_goal_heading.weight = 4.2
        self.rewards.stair_near_goal_heading.params["yaw_std"] = 0.15
        self.rewards.stair_yaw_drift.weight = -3.2
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.12
        self.rewards.stair_lateral_drift.weight = -1.4
        self.rewards.stair_rear_feet_on_step.weight = 14.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 20.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 18.0
        self.rewards.stair_success.weight = 240.0
        self.rewards.stair_success.params.update(
            {
                "foot_x_margin": -0.04,
                "target_base_clearance": 0.22,
                "min_rear_contacts": 1,
                "max_pitch_projected": 0.56,
                "max_roll_projected": 0.40,
                "max_yaw": 0.45,
            }
        )
        self.terminations.success.params.update(
            {
                "foot_x_margin": -0.04,
                "target_base_clearance": 0.22,
                "min_rear_contacts": 1,
                "max_pitch_projected": 0.56,
                "max_roll_projected": 0.40,
                "max_yaw": 0.45,
            }
        )
        self.rewards.trunk_contact.weight = -24.0
        self.rewards.trunk_contact_failure.weight = -46.0
        self.rewards.bad_yaw_failure.weight = -20.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 0.75,
            "gate_x": STAIR_FIRST_RISER_X - 0.08,
            "grace_time_s": 0.70,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 1.05,
            "gate_x": STAIR_FIRST_RISER_X - 0.04,
            "grace_time_s": 1.10,
        }
        self.rewards.stair_stuck.weight = -0.7
        self.rewards.stair_stuck.params["min_speed_x"] = 0.045
        self.rewards.stair_near_goal_timeout.weight = -1.4
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 3.0
        self.rewards.stair_low_base_at_riser.weight = -2.5
        self.rewards.low_base_at_riser_failure.weight = -16.0


class GogoA1StairClimbSwitchX070EdgeX068EnvCfg_PLAY(GogoA1StairClimbSwitchX070EdgeX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbBridgeHandoffEdgeX068EnvCfg(GogoA1StairClimbSwitchX070EdgeX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.events.reset_base = EventTerm(
            func=gogo_mdp.reset_stair_climb_from_switch_state_dataset,
            mode="reset",
            params={
                "dataset_path": "outputs/bridge_handoff_states_x068_v47_330.pt",
                "x_jitter": (-0.010, 0.018),
                "y_jitter": (-0.018, 0.018),
                "yaw_jitter": (-0.025, 0.025),
                "velocity_scale_range": (0.90, 1.08),
            },
        )

        self.episode_length_s = 7.5
        self.commands.base_velocity.ranges.lin_vel_x = (0.16, 0.25)

        self.rewards.track_lin_vel_xy_exp.weight = 0.08
        self.rewards.stair_forward_progress.weight = 0.06
        self.rewards.stair_forward_velocity.weight = 0.04
        self.rewards.stair_height_progress.weight = 2.4
        self.rewards.stair_base_height_first.weight = 2.4
        self.rewards.stair_base_on_first_step.weight = 7.0
        self.rewards.stair_front_feet_on_step.weight = 1.4
        self.rewards.stair_goal.weight = 5.6
        self.rewards.stair_final_approach.weight = 5.6
        self.rewards.stair_near_goal_heading.weight = 4.8
        self.rewards.stair_near_goal_heading.params["yaw_std"] = 0.18
        self.rewards.stair_rear_feet_on_step.weight = 16.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 22.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 10.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 20.0
        self.rewards.stair_rear_feet_forward.weight = 5.5
        self.rewards.stair_success.weight = 280.0
        self.rewards.stair_success.params.update(
            {
                "foot_x_margin": -0.05,
                "target_base_clearance": 0.22,
                "min_rear_contacts": 1,
                "max_pitch_projected": 0.58,
                "max_roll_projected": 0.42,
                "max_yaw": 0.50,
            }
        )
        self.terminations.success.params.update(
            {
                "foot_x_margin": -0.05,
                "target_base_clearance": 0.22,
                "min_rear_contacts": 1,
                "max_pitch_projected": 0.58,
                "max_roll_projected": 0.42,
                "max_yaw": 0.50,
            }
        )
        self.rewards.stair_lateral_drift.weight = -1.2
        self.rewards.stair_yaw_drift.weight = -2.4
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.16
        self.rewards.stair_tilt.weight = -7.0
        self.rewards.stair_stuck.weight = -0.9
        self.rewards.stair_stuck.params["min_speed_x"] = 0.035
        self.rewards.stair_near_goal_timeout.weight = -1.1
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 3.4
        self.rewards.stair_low_base_at_riser.weight = -2.0
        self.rewards.low_base_at_riser_failure.weight = -10.0
        self.rewards.trunk_contact.weight = -24.0
        self.rewards.trunk_contact_failure.weight = -46.0
        self.rewards.bad_yaw_failure.weight = -12.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 0.95,
            "gate_x": STAIR_FIRST_RISER_X - 0.04,
            "grace_time_s": 1.0,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 1.35,
            "gate_x": STAIR_FIRST_RISER_X - 0.02,
            "grace_time_s": 1.45,
        }
        self.terminations.fall.params = {
            "min_base_height": 0.13,
            "max_lateral_offset": 0.90,
            "max_pitch_projected": 0.88,
            "max_roll_projected": 0.68,
        }
        self.terminations.low_base_at_riser.params = {
            "first_riser_x": STAIR_FIRST_RISER_X,
            "step_height_range": STAIR_STEP_HEIGHT_RANGE,
            "min_base_clearance": 0.19,
            "x_window": (-0.05, 0.42),
            "grace_time_s": 1.6,
            "max_speed_x": 0.02,
        }


class GogoA1StairClimbBridgeHandoffEdgeX068EnvCfg_PLAY(GogoA1StairClimbBridgeHandoffEdgeX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbBridgeHandoffCenteredX068EnvCfg(GogoA1StairClimbBridgeHandoffEdgeX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.events.reset_base.params.update(
            {
                "dataset_path": "outputs/bridge_handoff_states_x068_centered_v49_180_y009.pt",
                "x_jitter": (-0.006, 0.012),
                "y_jitter": (-0.010, 0.010),
                "yaw_jitter": (-0.018, 0.018),
                "velocity_scale_range": (0.94, 1.04),
            }
        )

        self.episode_length_s = 7.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.23)
        self.rewards.stair_lateral_drift.weight = -2.8
        self.rewards.stair_yaw_drift.weight = -4.0
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.11
        self.rewards.stair_near_goal_heading.weight = 5.2
        self.rewards.stair_near_goal_heading.params["yaw_std"] = 0.14
        self.rewards.bad_yaw_failure.weight = -22.0
        self.rewards.bad_yaw_failure.params = {
            "max_yaw": 0.72,
            "gate_x": STAIR_FIRST_RISER_X - 0.04,
            "grace_time_s": 0.75,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 0.95,
            "gate_x": STAIR_FIRST_RISER_X - 0.02,
            "grace_time_s": 1.05,
        }
        self.terminations.fall.params = {
            "min_base_height": 0.13,
            "max_lateral_offset": 0.52,
            "max_pitch_projected": 0.82,
            "max_roll_projected": 0.58,
        }
        self.rewards.stair_success.params.update(
            {
                "foot_x_margin": -0.045,
                "target_base_clearance": 0.22,
                "min_rear_contacts": 1,
                "max_pitch_projected": 0.56,
                "max_roll_projected": 0.38,
                "max_yaw": 0.42,
            }
        )
        self.terminations.success.params.update(
            {
                "foot_x_margin": -0.045,
                "target_base_clearance": 0.22,
                "min_rear_contacts": 1,
                "max_pitch_projected": 0.56,
                "max_roll_projected": 0.38,
                "max_yaw": 0.42,
            }
        )


class GogoA1StairClimbBridgeHandoffCenteredX068EnvCfg_PLAY(GogoA1StairClimbBridgeHandoffCenteredX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbSwitchX058EdgeBridgeAssistX068EnvCfg(GogoA1StairClimbSwitchX058EdgeX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        def foot_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", body_names=foot_names, preserve_order=True)

        def trunk_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names="trunk")

        bridge_start_x = 0.58
        bridge_hold_x = 0.62
        bridge_target_x = STAIR_TARGET_X
        bridge_target_height = 0.305

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "approach_probability": 0.00,
                "switch_probability": 0.86,
                "near_step_x_range": (0.60, 0.72),
                "step_base_clearance": 0.325,
                "x_jitter": (-0.004, 0.010),
                "y_jitter": (-0.014, 0.014),
                "yaw_jitter": (-0.018, 0.018),
                "velocity_scale_range": (0.92, 1.05),
                "near_velocity_range": {
                    "x": (0.0, 0.040),
                    "y": (-0.008, 0.008),
                    "z": (-0.008, 0.008),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.018, 0.018),
                    "yaw": (-0.018, 0.018),
                },
            }
        )
        self.events.reset_base.params = reset_params

        self.episode_length_s = 7.2
        self.commands.base_velocity.ranges.lin_vel_x = (0.16, 0.25)

        self.rewards.stair_forward_progress = RewTerm(
            func=gogo_mdp.stair_bridge_forward_band_reward,
            weight=0.75,
            params={
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "max_speed": 0.28,
                "yaw_allowance": 0.14,
                "roll_allowance": 0.26,
            },
        )
        self.rewards.stair_xz_progress = RewTerm(
            func=gogo_mdp.stair_bridge_xz_progress_reward,
            weight=13.0,
            params={
                "start_x": bridge_start_x,
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "base_z": 0.270,
                "y_std": 0.110,
                "yaw_std": 0.18,
                "pitch_allowance": 0.58,
                "roll_allowance": 0.34,
            },
        )
        self.rewards.stair_height_hold = RewTerm(
            func=gogo_mdp.stair_bridge_height_hold_reward,
            weight=9.0,
            params={
                "x_gate": bridge_hold_x,
                "target_x": bridge_target_x,
                "target_base_height": bridge_target_height,
                "height_std": 0.070,
                "y_std": 0.110,
                "yaw_std": 0.18,
                "pitch_allowance": 0.58,
                "roll_allowance": 0.34,
            },
        )
        self.rewards.stair_height_progress.weight = 4.0
        self.rewards.stair_height_progress.params.update(
            {
                "target_z": bridge_target_height,
                "step_height_range": None,
                "target_base_clearance": 0.300,
                "gate_x": STAIR_FIRST_RISER_X - 0.18,
                "max_pitch_projected": 0.74,
                "max_roll_projected": 0.50,
            }
        )
        self.rewards.stair_base_height_first.weight = 4.5
        self.rewards.stair_base_height_first.params.update(
            {
                "base_clearance": 0.270,
                "gate_margin": -0.16,
                "max_pitch_projected": 0.74,
                "max_roll_projected": 0.50,
            }
        )
        self.rewards.stair_base_on_first_step.weight = 7.5
        self.rewards.stair_base_on_first_step.params.update(
            {
                "base_clearance": 0.255,
                "x_fraction": 0.10,
                "lateral_std": 0.26,
                "max_pitch_projected": 0.72,
                "max_roll_projected": 0.48,
            }
        )
        self.rewards.stair_rear_feet_min_x_progress = RewTerm(
            func=gogo_mdp.stair_rear_feet_min_x_progress_reward,
            weight=9.0,
            params={
                "asset_cfg": foot_asset_cfg(),
                "first_riser_x": STAIR_FIRST_RISER_X,
                "foot_indices": (2, 3),
                "start_margin": -0.34,
                "target_margin": -0.04,
                "body_gate_margin": -0.12,
                "lateral_std": 0.40,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
                "max_pitch_projected": 0.78,
                "max_roll_projected": 0.52,
            },
        )
        self.rewards.stair_rear_feet_forward.weight = 6.5
        self.rewards.stair_rear_feet_on_step.weight = 14.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 22.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 11.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 20.0
        self.rewards.stair_goal.weight = 5.4
        self.rewards.stair_final_approach.weight = 5.4
        self.rewards.stair_success.weight = 280.0
        self.rewards.stair_success.params.update(
            {
                "target_base_clearance": 0.22,
                "min_rear_contacts": 1,
                "foot_x_margin": -0.045,
                "max_pitch_projected": 0.58,
                "max_roll_projected": 0.42,
                "max_yaw": 0.48,
            }
        )
        self.terminations.success.params.update(
            {
                "target_base_clearance": 0.22,
                "min_rear_contacts": 1,
                "foot_x_margin": -0.045,
                "max_pitch_projected": 0.58,
                "max_roll_projected": 0.42,
                "max_yaw": 0.48,
            }
        )
        self.rewards.stair_stuck.weight = -1.2
        self.rewards.stair_stuck.params["min_speed_x"] = 0.045
        self.rewards.stair_near_goal_timeout.weight = -1.8
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 2.8
        self.rewards.stair_low_base_at_riser.weight = -3.0
        self.rewards.stair_low_base_at_riser.params["min_base_clearance"] = 0.215
        self.rewards.low_base_at_riser_failure.weight = -18.0
        self.rewards.trunk_contact.weight = -24.0
        self.rewards.trunk_contact_failure.weight = -44.0
        self.rewards.bad_yaw_failure.weight = -18.0
        self.terminations.low_base_at_riser.params = {
            "first_riser_x": STAIR_FIRST_RISER_X,
            "step_height_range": STAIR_STEP_HEIGHT_RANGE,
            "min_base_clearance": 0.205,
            "x_window": (-0.08, 0.34),
            "grace_time_s": 1.2,
            "max_speed_x": 0.025,
        }
        self.terminations.trunk_contact.params = {
            "sensor_cfg": trunk_sensor_cfg(),
            "threshold": 5.0,
            "grace_time_s": 0.16,
        }
        self.terminations.fall.params = {
            "min_base_height": 0.13,
            "max_lateral_offset": 0.70,
            "max_pitch_projected": 0.88,
            "max_roll_projected": 0.64,
        }
        self.terminations.bad_yaw.params = {
            "max_yaw": 1.05,
            "gate_x": STAIR_FIRST_RISER_X - 0.06,
            "grace_time_s": 1.1,
        }


class GogoA1StairClimbSwitchX058EdgeBridgeAssistX068EnvCfg_PLAY(
    GogoA1StairClimbSwitchX058EdgeBridgeAssistX068EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1StairClimbSwitchX058EdgeBridgeAssistTightX068EnvCfg(
    GogoA1StairClimbSwitchX058EdgeBridgeAssistX068EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()

        tight_success = {
            "target_base_clearance": 0.22,
            "min_rear_contacts": 1,
            "foot_x_margin": -0.025,
            "foot_z_margin": 0.010,
            "foot_z_upper_margin": 0.125,
            "foot_lateral_tolerance": 0.34,
            "max_pitch_projected": 0.58,
            "max_roll_projected": 0.42,
            "max_yaw": 0.42,
            "max_abs_y": 0.12,
        }
        self.rewards.stair_success.params.update(tight_success)
        self.terminations.success.params.update(tight_success)

        self.rewards.stair_rear_feet_min_x_progress.weight = 11.0
        self.rewards.stair_rear_feet_min_x_progress.params.update(
            {
                "target_margin": -0.025,
                "lateral_std": 0.28,
                "body_gate_margin": -0.10,
            }
        )
        self.rewards.stair_rear_feet_on_step.weight = 18.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 30.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 13.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 26.0
        self.rewards.stair_lateral_drift.weight = -4.0
        self.rewards.stair_xz_progress.params.update({"y_std": 0.085, "yaw_std": 0.16})
        self.rewards.stair_height_hold.params.update({"y_std": 0.085, "yaw_std": 0.16})
        self.rewards.stair_success.weight = 340.0
        self.rewards.stair_near_goal_timeout.weight = -2.2
        self.terminations.fall.params.update({"max_lateral_offset": 0.50})


class GogoA1StairClimbSwitchX058EdgeBridgeAssistTightX068EnvCfg_PLAY(
    GogoA1StairClimbSwitchX058EdgeBridgeAssistTightX068EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1StairClimbSwitchX058RearCatchX068EnvCfg(
    GogoA1StairClimbSwitchX058EdgeBridgeAssistTightX068EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        def foot_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", body_names=foot_names, preserve_order=True)

        def foot_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names=foot_names, preserve_order=True)

        def trunk_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names="trunk")

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "approach_probability": 0.0,
                "switch_probability": 0.42,
                "near_step_x_range": (0.72, 0.86),
                "step_base_clearance": 0.335,
                "x_jitter": (-0.004, 0.010),
                "y_jitter": (-0.010, 0.010),
                "yaw_jitter": (-0.014, 0.014),
                "velocity_scale_range": (0.90, 1.02),
                "near_velocity_range": {
                    "x": (0.0, 0.020),
                    "y": (-0.006, 0.006),
                    "z": (-0.006, 0.006),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.012, 0.012),
                    "yaw": (-0.012, 0.012),
                },
            }
        )
        self.events.reset_base.params = reset_params

        self.episode_length_s = 6.6
        self.commands.base_velocity.ranges.lin_vel_x = (0.12, 0.21)
        self.rewards.track_lin_vel_xy_exp.weight = 0.06
        self.rewards.stair_forward_progress.weight = 0.35
        self.rewards.stair_forward_velocity.weight = 0.03
        self.rewards.stair_height_progress.weight = 3.2
        self.rewards.stair_base_height_first.weight = 3.6
        self.rewards.stair_base_on_first_step.weight = 8.5
        self.rewards.stair_xz_progress.weight = 8.0
        self.rewards.stair_height_hold.weight = 7.5
        self.rewards.stair_xz_progress.params.update(
            {
                "start_x": STAIR_FIRST_RISER_X - 0.02,
                "target_x": STAIR_TARGET_X,
                "target_base_height": 0.310,
                "base_z": 0.285,
                "y_std": 0.075,
                "yaw_std": 0.14,
            }
        )
        self.rewards.stair_height_hold.params.update(
            {
                "x_gate": STAIR_FIRST_RISER_X - 0.02,
                "target_x": STAIR_TARGET_X,
                "target_base_height": 0.315,
                "height_std": 0.060,
                "y_std": 0.075,
                "yaw_std": 0.14,
            }
        )

        self.rewards.stair_rear_feet_xz_contact = RewTerm(
            func=gogo_mdp.stair_rear_feet_xz_contact_reward,
            weight=44.0,
            params={
                "asset_cfg": foot_asset_cfg(),
                "contact_sensor_cfg": foot_sensor_cfg(),
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "foot_indices": (2, 3),
                "step_index": 1,
                "foot_x_margin": -0.018,
                "foot_z_margin": 0.004,
                "x_std": 0.085,
                "z_std": 0.050,
                "lateral_std": 0.24,
                "contact_threshold": 0.5,
                "min_contacts_for_bonus": 1,
                "body_gate_margin": -0.08,
                "trunk_sensor_cfg": trunk_sensor_cfg(),
                "trunk_contact_threshold": 1.0,
                "max_pitch_projected": 0.74,
                "max_roll_projected": 0.48,
            },
        )
        self.rewards.stair_rear_feet_min_x_progress.weight = 8.0
        self.rewards.stair_rear_feet_min_x_progress.params.update(
            {
                "start_margin": -0.30,
                "target_margin": -0.018,
                "body_gate_margin": -0.06,
                "lateral_std": 0.24,
            }
        )
        self.rewards.stair_rear_feet_forward.weight = 4.0
        self.rewards.stair_rear_feet_forward.params.update({"x_margin": -0.02, "lateral_std": 0.24})
        self.rewards.stair_rear_feet_on_step.weight = 24.0
        self.rewards.stair_rear_feet_on_step.params.update({"x_margin": -0.010})
        self.rewards.stair_rear_feet_contact_on_step.weight = 42.0
        self.rewards.stair_rear_feet_contact_on_step.params.update({"x_margin": -0.010})
        self.rewards.stair_base_rear_feet_on_first_step.weight = 18.0
        self.rewards.stair_base_rear_feet_on_first_step.params.update(
            {
                "base_clearance": 0.215,
                "base_x_fraction": 0.24,
                "foot_x_margin": -0.018,
                "foot_z_margin": 0.012,
            }
        )
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 34.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.params.update(
            {
                "base_clearance": 0.215,
                "base_x_fraction": 0.24,
                "foot_x_margin": -0.018,
                "foot_z_margin": 0.012,
            }
        )
        self.rewards.stair_goal.weight = 3.5
        self.rewards.stair_final_approach.weight = 3.2
        self.rewards.stair_success.weight = 420.0
        self.rewards.stair_success.params.update(
            {
                "target_base_clearance": 0.21,
                "min_rear_contacts": 1,
                "foot_x_margin": -0.020,
                "foot_z_margin": 0.006,
                "foot_z_upper_margin": 0.115,
                "foot_lateral_tolerance": 0.30,
                "max_pitch_projected": 0.58,
                "max_roll_projected": 0.40,
                "max_yaw": 0.40,
                "max_abs_y": 0.10,
            }
        )
        self.terminations.success.params.update(self.rewards.stair_success.params)
        self.rewards.stair_lateral_drift.weight = -5.5
        self.rewards.stair_yaw_drift.weight = -3.8
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.09
        self.rewards.stair_stuck.weight = -0.8
        self.rewards.stair_stuck.params["min_speed_x"] = 0.030
        self.rewards.stair_near_goal_timeout.weight = -1.4
        self.rewards.stair_near_goal_timeout.params["grace_time_s"] = 3.2
        self.rewards.stair_low_base_at_riser.weight = -1.5
        self.rewards.trunk_contact.weight = -28.0
        self.rewards.trunk_contact_failure.weight = -50.0
        self.rewards.low_base_at_riser_failure.weight = -10.0
        self.terminations.fall.params.update(
            {
                "min_base_height": 0.13,
                "max_lateral_offset": 0.42,
                "max_pitch_projected": 0.86,
                "max_roll_projected": 0.62,
            }
        )
        self.terminations.low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.19,
                "x_window": (-0.06, 0.38),
                "grace_time_s": 1.5,
                "max_speed_x": 0.018,
            }
        )


class GogoA1StairClimbSwitchX058RearCatchX068EnvCfg_PLAY(
    GogoA1StairClimbSwitchX058RearCatchX068EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


class GogoA1StairClimbSwitchX058RearCatchStageX068EnvCfg(
    GogoA1StairClimbSwitchX058RearCatchX068EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()

        stage_target_x = STAIR_FIRST_RISER_X + 0.24 * STAIR_STEP_DEPTH

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "approach_probability": 0.0,
                "switch_probability": 0.0,
                "near_step_x_range": (0.78, 0.90),
                "step_base_clearance": 0.340,
                "x_jitter": (-0.003, 0.008),
                "y_jitter": (-0.008, 0.008),
                "yaw_jitter": (-0.010, 0.010),
                "near_velocity_range": {
                    "x": (0.0, 0.012),
                    "y": (-0.004, 0.004),
                    "z": (-0.004, 0.004),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.010, 0.010),
                    "yaw": (-0.010, 0.010),
                },
            }
        )
        self.events.reset_base.params = reset_params

        self.episode_length_s = 4.8
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.16)
        self.rewards.track_lin_vel_xy_exp.weight = 0.035
        self.rewards.stair_forward_progress.weight = 0.12
        self.rewards.stair_forward_velocity.weight = 0.015
        self.rewards.stair_goal.weight = 1.4
        self.rewards.stair_final_approach.weight = 1.2
        self.rewards.stair_height_progress.weight = 2.4
        self.rewards.stair_base_height_first.weight = 2.4
        self.rewards.stair_base_on_first_step.weight = 4.5

        self.rewards.stair_xz_progress.params.update(
            {
                "start_x": STAIR_FIRST_RISER_X - 0.01,
                "target_x": stage_target_x,
                "target_base_height": 0.320,
                "base_z": 0.290,
                "y_std": 0.065,
                "yaw_std": 0.12,
            }
        )
        self.rewards.stair_height_hold.params.update(
            {
                "x_gate": STAIR_FIRST_RISER_X - 0.01,
                "target_x": stage_target_x,
                "target_base_height": 0.320,
                "height_std": 0.052,
                "y_std": 0.065,
                "yaw_std": 0.12,
            }
        )
        self.rewards.stair_rear_feet_xz_contact.weight = 72.0
        self.rewards.stair_rear_feet_xz_contact.params.update(
            {
                "foot_x_margin": -0.010,
                "foot_z_margin": 0.002,
                "x_std": 0.055,
                "z_std": 0.040,
                "lateral_std": 0.20,
                "min_contacts_for_bonus": 2,
                "body_gate_margin": -0.02,
                "max_pitch_projected": 0.66,
                "max_roll_projected": 0.44,
                "aggregation": "min",
            }
        )
        self.rewards.stair_rear_feet_min_x_progress.weight = 16.0
        self.rewards.stair_rear_feet_min_x_progress.params.update(
            {
                "start_margin": -0.24,
                "target_margin": -0.006,
                "body_gate_margin": -0.02,
                "lateral_std": 0.20,
                "max_pitch_projected": 0.70,
                "max_roll_projected": 0.46,
            }
        )
        self.rewards.stair_rear_feet_forward.weight = 8.0
        self.rewards.stair_rear_feet_forward.params.update({"x_margin": -0.006, "lateral_std": 0.20})
        self.rewards.stair_rear_feet_on_step.weight = 44.0
        self.rewards.stair_rear_feet_on_step.params.update({"x_margin": -0.006})
        self.rewards.stair_rear_feet_contact_on_step.weight = 76.0
        self.rewards.stair_rear_feet_contact_on_step.params.update({"x_margin": -0.006})
        self.rewards.stair_base_rear_feet_on_first_step.weight = 32.0
        self.rewards.stair_base_rear_feet_on_first_step.params.update(
            {
                "base_clearance": 0.205,
                "base_x_fraction": 0.20,
                "foot_x_margin": -0.006,
                "foot_z_margin": 0.006,
                "max_pitch_projected": 0.64,
                "max_roll_projected": 0.42,
            }
        )
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 58.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.params.update(
            {
                "base_clearance": 0.205,
                "base_x_fraction": 0.20,
                "foot_x_margin": -0.006,
                "foot_z_margin": 0.006,
                "max_pitch_projected": 0.64,
                "max_roll_projected": 0.42,
            }
        )

        stage_success = {
            "target_x": stage_target_x,
            "target_base_clearance": 0.195,
            "min_rear_contacts": 2,
            "foot_x_margin": -0.006,
            "foot_z_margin": 0.004,
            "foot_z_upper_margin": 0.105,
            "foot_lateral_tolerance": 0.26,
            "max_pitch_projected": 0.60,
            "max_roll_projected": 0.40,
            "max_yaw": 0.34,
            "max_abs_y": 0.09,
        }
        self.rewards.stair_success.weight = 520.0
        self.rewards.stair_success.params.update(stage_success)
        self.terminations.success.params.update(stage_success)

        self.rewards.stair_lateral_drift.weight = -7.0
        self.rewards.stair_yaw_drift.weight = -4.6
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.06
        self.rewards.stair_tilt.weight = -9.5
        self.rewards.stair_stuck.weight = -0.5
        self.rewards.stair_stuck.params.update({"min_speed_x": 0.020, "before_goal_x": stage_target_x + 0.08})
        self.rewards.stair_near_goal_timeout.weight = -0.8
        self.rewards.stair_near_goal_timeout.params.update(
            {
                "target_x": stage_target_x,
                "target_base_clearance": 0.195,
                "min_progress_fraction": 0.78,
                "min_height_fraction": 0.70,
                "grace_time_s": 2.6,
            }
        )
        self.rewards.stair_low_base_at_riser.weight = -1.0
        self.rewards.trunk_contact.weight = -34.0
        self.rewards.trunk_contact_failure.weight = -58.0
        self.rewards.bad_yaw_failure.weight = -22.0
        self.terminations.fall.params.update(
            {
                "min_base_height": 0.13,
                "max_lateral_offset": 0.32,
                "max_pitch_projected": 0.78,
                "max_roll_projected": 0.54,
            }
        )
        self.terminations.bad_yaw.params.update(
            {
                "max_yaw": 0.82,
                "gate_x": STAIR_FIRST_RISER_X - 0.02,
                "grace_time_s": 0.8,
            }
        )
        self.terminations.low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.18,
                "x_window": (-0.04, 0.30),
                "grace_time_s": 1.2,
                "max_speed_x": 0.014,
            }
        )


class GogoA1StairClimbSwitchX058RearCatchStageX068EnvCfg_PLAY(
    GogoA1StairClimbSwitchX058RearCatchStageX068EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbFlatStartX058RearStrictX068EnvCfg(GogoA1StairClimbEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        def foot_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", body_names=foot_names, preserve_order=True)

        def foot_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names=foot_names, preserve_order=True)

        def trunk_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names="trunk")

        self.events.reset_base = EventTerm(
            func=gogo_mdp.reset_stair_climb_root_state,
            mode="reset",
            params={
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "approach_probability": 1.0,
                "approach_x_range": (-0.20, 0.24),
                "near_step_x_range": (0.40, 0.52),
                "y_range": (-0.035, 0.035),
                "yaw_range": (-0.035, 0.035),
                "base_height": 0.31,
                "step_base_clearance": 0.29,
                "velocity_range": {
                    "x": (0.0, 0.035),
                    "y": (-0.006, 0.006),
                    "z": (-0.004, 0.004),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.010, 0.010),
                    "yaw": (-0.010, 0.010),
                },
                "joint_position_scale_range": (0.99, 1.01),
                "joint_velocity_range": (-0.025, 0.025),
            },
        )

        self.episode_length_s = 8.8
        self.commands.base_velocity.ranges.lin_vel_x = (0.16, 0.26)
        self.rewards.track_lin_vel_xy_exp.weight = 0.16
        self.rewards.track_ang_vel_z_exp.weight = 0.03
        self.rewards.stair_forward_progress.weight = 0.55
        self.rewards.stair_forward_velocity.weight = 0.18

        self.rewards.stair_approach_progress = RewTerm(
            func=gogo_mdp.stair_handoff_progress_reward,
            weight=2.6,
            params={
                "start_x": -0.20,
                "target_x": STAIR_FIRST_RISER_X - 0.16,
                "max_speed": 0.35,
                "yaw_allowance": 0.10,
                "roll_allowance": 0.24,
            },
        )
        self.rewards.stair_approach_handoff = RewTerm(
            func=gogo_mdp.stair_bridge_handoff_reward,
            weight=3.2,
            params={
                "target_x": STAIR_FIRST_RISER_X - 0.14,
                "target_base_height": 0.31,
                "x_std": 0.18,
                "y_std": 0.10,
                "height_std": 0.08,
                "yaw_std": 0.16,
                "pitch_std": 0.28,
                "roll_std": 0.22,
                "max_speed": 0.36,
            },
        )

        self.rewards.stair_height_progress.weight = 2.8
        self.rewards.stair_height_progress.params.update(
            {
                "gate_x": STAIR_FIRST_RISER_X - 0.16,
                "base_z": 0.285,
                "target_base_clearance": 0.225,
                "max_pitch_projected": 0.74,
                "max_roll_projected": 0.50,
            }
        )
        self.rewards.stair_base_height_first.weight = 2.4
        self.rewards.stair_base_height_first.params.update(
            {
                "base_clearance": 0.215,
                "gate_margin": -0.14,
                "max_pitch_projected": 0.74,
                "max_roll_projected": 0.50,
            }
        )
        self.rewards.stair_front_feet_on_step.weight = 4.0
        self.rewards.stair_front_feet_on_step.params.update({"x_margin": -0.010, "z_margin": 0.020})
        self.rewards.stair_base_on_first_step.weight = 7.5
        self.rewards.stair_base_on_first_step.params.update(
            {
                "base_clearance": 0.215,
                "x_fraction": 0.26,
                "lateral_std": 0.24,
                "max_pitch_projected": 0.70,
                "max_roll_projected": 0.46,
            }
        )

        self.rewards.stair_rear_feet_min_x_progress = RewTerm(
            func=gogo_mdp.stair_rear_feet_min_x_progress_reward,
            weight=12.0,
            params={
                "asset_cfg": foot_asset_cfg(),
                "first_riser_x": STAIR_FIRST_RISER_X,
                "foot_indices": (2, 3),
                "start_margin": -0.38,
                "target_margin": -0.012,
                "body_gate_margin": -0.20,
                "lateral_std": 0.24,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
                "max_pitch_projected": 0.78,
                "max_roll_projected": 0.52,
            },
        )
        self.rewards.stair_rear_feet_forward.weight = 7.0
        self.rewards.stair_rear_feet_forward.params.update(
            {"x_margin": -0.012, "lateral_std": 0.24, "max_pitch_projected": 0.78, "max_roll_projected": 0.52}
        )
        self.rewards.stair_rear_feet_on_step.weight = 26.0
        self.rewards.stair_rear_feet_on_step.params.update({"x_margin": -0.006, "z_margin": 0.020})
        self.rewards.stair_rear_feet_contact_on_step.weight = 36.0
        self.rewards.stair_rear_feet_contact_on_step.params.update(
            {
                "x_margin": -0.006,
                "z_margin": 0.008,
                "z_upper_margin": 0.115,
                "lateral_std": 0.24,
            }
        )
        self.rewards.stair_rear_feet_xz_contact = RewTerm(
            func=gogo_mdp.stair_rear_feet_xz_contact_reward,
            weight=42.0,
            params={
                "asset_cfg": foot_asset_cfg(),
                "contact_sensor_cfg": foot_sensor_cfg(),
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "foot_indices": (2, 3),
                "step_index": 1,
                "foot_x_margin": -0.006,
                "foot_z_margin": 0.000,
                "x_std": 0.045,
                "z_std": 0.026,
                "lateral_std": 0.22,
                "contact_threshold": 0.5,
                "min_contacts_for_bonus": 2,
                "body_gate_margin": -0.04,
                "trunk_sensor_cfg": trunk_sensor_cfg(),
                "trunk_contact_threshold": 1.0,
                "max_pitch_projected": 0.70,
                "max_roll_projected": 0.48,
                "aggregation": "min",
            },
        )
        self.rewards.stair_base_rear_feet_on_first_step.weight = 20.0
        self.rewards.stair_base_rear_feet_on_first_step.params.update(
            {
                "base_clearance": 0.205,
                "base_x_fraction": 0.22,
                "foot_x_margin": -0.004,
                "foot_z_margin": 0.000,
                "max_pitch_projected": 0.68,
                "max_roll_projected": 0.46,
            }
        )
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 32.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.params.update(
            {
                "base_clearance": 0.205,
                "base_x_fraction": 0.22,
                "foot_x_margin": -0.004,
                "foot_z_margin": 0.000,
                "max_pitch_projected": 0.68,
                "max_roll_projected": 0.46,
            }
        )
        self.rewards.stair_rear_feet_low_after_base = RewTerm(
            func=gogo_mdp.stair_rear_feet_low_after_base_penalty,
            weight=-48.0,
            params={
                "asset_cfg": foot_asset_cfg(),
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "foot_indices": (2, 3),
                "base_x_margin": 0.06,
                "foot_x_margin": -0.018,
                "foot_z_margin": 0.018,
                "x_std": 0.055,
                "z_std": 0.030,
                "lateral_std": 0.24,
                "max_pitch_projected": 0.74,
                "max_roll_projected": 0.50,
            },
        )

        self.rewards.stair_goal.weight = 5.0
        self.rewards.stair_final_approach.weight = 4.5
        self.rewards.stair_success.weight = 360.0
        strict_success = {
            "target_x": STAIR_TARGET_X,
            "target_base_clearance": 0.220,
            "min_rear_contacts": 2,
            "foot_x_margin": -0.006,
            "foot_z_margin": 0.012,
            "foot_z_upper_margin": 0.120,
            "foot_lateral_tolerance": 0.30,
            "max_pitch_projected": 0.60,
            "max_roll_projected": 0.42,
            "max_yaw": 0.38,
            "max_abs_y": 0.11,
        }
        self.rewards.stair_success.params.update(strict_success)
        self.terminations.success.params.update(strict_success)

        self.rewards.stair_lateral_drift.weight = -4.5
        self.rewards.stair_yaw_drift.weight = -4.5
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.08
        self.rewards.stair_tilt.weight = -8.0
        self.rewards.stair_stuck.weight = -1.4
        self.rewards.stair_stuck.params.update(
            {"min_command_x": 0.12, "min_speed_x": 0.030, "before_goal_x": STAIR_TARGET_X + 0.08}
        )
        self.rewards.stair_near_goal_timeout.weight = -2.0
        self.rewards.stair_near_goal_timeout.params.update(
            {
                "min_progress_fraction": 0.66,
                "min_height_fraction": 0.55,
                "grace_time_s": 3.2,
            }
        )
        self.rewards.stair_low_base_at_riser.weight = -5.0
        self.rewards.stair_low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.205,
                "x_window": (-0.12, 0.36),
            }
        )
        self.rewards.low_base_at_riser_failure = RewTerm(
            func=gogo_mdp.stair_low_base_at_riser_failure,
            weight=-24.0,
            params={
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "min_base_clearance": 0.190,
                "x_window": (-0.10, 0.32),
                "grace_time_s": 1.1,
                "max_speed_x": 0.020,
            },
        )
        self.rewards.trunk_contact.weight = -30.0
        self.rewards.trunk_contact_failure.weight = -50.0
        self.rewards.bad_yaw_failure.weight = -28.0
        self.rewards.fall_failure.weight = -48.0

        self.terminations.fall.params.update(
            {
                "min_base_height": 0.135,
                "max_lateral_offset": 0.55,
                "max_pitch_projected": 0.82,
                "max_roll_projected": 0.60,
            }
        )
        self.terminations.bad_yaw.params.update(
            {
                "max_yaw": 0.95,
                "gate_x": STAIR_FIRST_RISER_X - 0.10,
                "grace_time_s": 0.7,
            }
        )
        self.terminations.low_base_at_riser = DoneTerm(
            func=gogo_mdp.stair_low_base_at_riser_failure,
            params={
                "first_riser_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "min_base_clearance": 0.190,
                "x_window": (-0.10, 0.32),
                "grace_time_s": 1.1,
                "max_speed_x": 0.020,
            },
        )
        self.terminations.trunk_contact.params.update(
            {
                "sensor_cfg": trunk_sensor_cfg(),
                "threshold": 5.0,
                "grace_time_s": 0.18,
            }
        )


class GogoA1StairClimbFlatStartX058RearStrictX068EnvCfg_PLAY(
    GogoA1StairClimbFlatStartX058RearStrictX068EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbFlatEdgeCommitX058X068EnvCfg(GogoA1StairClimbFlatStartX058RearStrictX068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        def foot_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", body_names=foot_names, preserve_order=True)

        def foot_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names=foot_names, preserve_order=True)

        def trunk_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names="trunk")

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "approach_probability": 1.0,
                "approach_x_range": (0.36, 0.58),
                "near_step_x_range": (0.50, 0.62),
                "y_range": (-0.025, 0.025),
                "yaw_range": (-0.025, 0.025),
                "base_height": 0.31,
                "velocity_range": {
                    "x": (0.02, 0.07),
                    "y": (-0.004, 0.004),
                    "z": (-0.004, 0.004),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.008, 0.008),
                    "yaw": (-0.008, 0.008),
                },
                "joint_position_scale_range": (0.995, 1.005),
                "joint_velocity_range": (-0.018, 0.018),
            }
        )
        self.events.reset_base.params = reset_params

        edge_target_x = STAIR_FIRST_RISER_X + 0.16
        self.episode_length_s = 5.8
        self.commands.base_velocity.ranges.lin_vel_x = (0.14, 0.23)
        self.rewards.track_lin_vel_xy_exp.weight = 0.10
        self.rewards.stair_forward_progress.weight = 1.0
        self.rewards.stair_forward_velocity.weight = 0.35
        self.rewards.stair_forward_progress.params["target_x"] = edge_target_x
        self.rewards.stair_forward_velocity.params.update({"target_x": edge_target_x, "max_speed": 0.40})

        self.rewards.stair_approach_progress.weight = 4.2
        self.rewards.stair_approach_progress.params.update(
            {
                "start_x": 0.36,
                "target_x": STAIR_FIRST_RISER_X - 0.02,
                "max_speed": 0.36,
                "yaw_allowance": 0.08,
                "roll_allowance": 0.24,
            }
        )
        self.rewards.stair_approach_handoff.weight = 4.4
        self.rewards.stair_approach_handoff.params.update(
            {
                "target_x": STAIR_FIRST_RISER_X - 0.03,
                "target_base_height": 0.31,
                "x_std": 0.13,
                "y_std": 0.08,
                "yaw_std": 0.13,
                "max_speed": 0.34,
            }
        )

        self.rewards.stair_height_progress.weight = 4.6
        self.rewards.stair_height_progress.params.update(
            {
                "gate_x": STAIR_FIRST_RISER_X - 0.08,
                "target_base_clearance": 0.205,
                "max_pitch_projected": 0.76,
                "max_roll_projected": 0.52,
            }
        )
        self.rewards.stair_base_height_first.weight = 4.0
        self.rewards.stair_base_height_first.params.update(
            {
                "base_clearance": 0.195,
                "gate_margin": -0.06,
                "max_pitch_projected": 0.76,
                "max_roll_projected": 0.52,
            }
        )
        self.rewards.stair_front_feet_on_step.weight = 9.5
        self.rewards.stair_front_feet_on_step.params.update({"x_margin": -0.018, "z_margin": 0.025})
        self.rewards.stair_front_feet_contact_on_step = RewTerm(
            func=gogo_mdp.stair_foot_contact_on_step_reward,
            weight=18.0,
            params={
                "asset_cfg": foot_asset_cfg(),
                "contact_sensor_cfg": foot_sensor_cfg(),
                "foot_indices": (0, 1),
                "edge_x": STAIR_FIRST_RISER_X,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "step_index": 1,
                "x_margin": -0.018,
                "z_margin": 0.008,
                "z_upper_margin": 0.120,
                "lateral_std": 0.24,
                "contact_threshold": 0.5,
                "trunk_sensor_cfg": trunk_sensor_cfg(),
                "trunk_contact_threshold": 1.0,
                "max_pitch_projected": 0.76,
                "max_roll_projected": 0.52,
            },
        )
        self.rewards.stair_base_on_first_step.weight = 13.0
        self.rewards.stair_base_on_first_step.params.update(
            {
                "base_clearance": 0.195,
                "x_fraction": 0.12,
                "lateral_std": 0.22,
                "max_pitch_projected": 0.76,
                "max_roll_projected": 0.52,
            }
        )

        self.rewards.stair_rear_feet_forward.weight = 5.0
        self.rewards.stair_rear_feet_on_step.weight = 10.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 12.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 6.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 10.0
        self.rewards.stair_rear_feet_xz_contact.weight = 8.0
        self.rewards.stair_rear_feet_low_after_base.weight = -8.0

        self.rewards.stair_goal = RewTerm(
            func=gogo_mdp.stair_goal_reward,
            weight=14.0,
            params={
                "target_x": edge_target_x,
                "target_z": STAIR_TARGET_Z,
                "xy_std": 0.36,
                "z_std": 0.24,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "num_steps": STAIR_NUM_STEPS,
                "target_base_clearance": 0.205,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
                "max_pitch_projected": 0.72,
                "max_roll_projected": 0.50,
            },
        )
        self.rewards.stair_success = RewTerm(
            func=gogo_mdp.stair_success_reward,
            weight=240.0,
            params={
                "target_x": edge_target_x,
                "target_z": STAIR_TARGET_Z,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "num_steps": STAIR_NUM_STEPS,
                "target_base_clearance": 0.195,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
                "max_pitch_projected": 0.66,
                "max_roll_projected": 0.46,
                "max_yaw": 0.36,
                "max_abs_y": 0.10,
            },
        )
        self.terminations.success = DoneTerm(
            func=gogo_mdp.stair_climb_success,
            params={
                "target_x": edge_target_x,
                "target_z": STAIR_TARGET_Z,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "num_steps": STAIR_NUM_STEPS,
                "target_base_clearance": 0.195,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
                "max_pitch_projected": 0.66,
                "max_roll_projected": 0.46,
                "max_yaw": 0.36,
                "max_abs_y": 0.10,
            },
        )

        self.rewards.stair_low_base_at_riser.weight = -0.8
        self.rewards.stair_low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.175,
                "x_window": (0.00, 0.34),
            }
        )
        self.rewards.low_base_at_riser_failure.weight = -5.0
        self.rewards.low_base_at_riser_failure.params.update(
            {
                "min_base_clearance": 0.160,
                "x_window": (0.00, 0.32),
                "grace_time_s": 1.8,
                "max_speed_x": 0.010,
            }
        )
        self.terminations.low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.155,
                "x_window": (0.02, 0.32),
                "grace_time_s": 2.2,
                "max_speed_x": 0.006,
            }
        )
        self.rewards.stair_near_goal_timeout.weight = -1.0
        self.rewards.stair_near_goal_timeout.params.update(
            {
                "target_x": edge_target_x,
                "target_base_clearance": 0.195,
                "min_progress_fraction": 0.78,
                "min_height_fraction": 0.52,
                "grace_time_s": 3.2,
            }
        )
        self.rewards.stair_stuck.weight = -0.6
        self.rewards.stair_stuck.params.update({"before_goal_x": edge_target_x + 0.05, "min_speed_x": 0.020})
        self.rewards.stair_lateral_drift.weight = -3.8
        self.rewards.stair_yaw_drift.weight = -3.8
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.07
        self.rewards.trunk_contact.weight = -34.0
        self.rewards.trunk_contact_failure.weight = -58.0
        self.terminations.fall.params.update(
            {
                "min_base_height": 0.12,
                "max_lateral_offset": 0.45,
                "max_pitch_projected": 0.86,
                "max_roll_projected": 0.64,
            }
        )
        self.terminations.bad_yaw.params.update(
            {
                "max_yaw": 0.82,
                "gate_x": STAIR_FIRST_RISER_X - 0.04,
                "grace_time_s": 0.65,
            }
        )


class GogoA1StairClimbFlatEdgeCommitX058X068EnvCfg_PLAY(GogoA1StairClimbFlatEdgeCommitX058X068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbFlatEdgeFrontContactX058X068EnvCfg(GogoA1StairClimbFlatEdgeCommitX058X068EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        def foot_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", body_names=foot_names, preserve_order=True)

        def foot_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names=foot_names, preserve_order=True)

        def trunk_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names="trunk")

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "approach_probability": 1.0,
                "approach_x_range": (0.38, 0.50),
                "near_step_x_range": (0.46, 0.54),
                "y_range": (-0.020, 0.020),
                "yaw_range": (-0.020, 0.020),
                "base_height": 0.31,
                "velocity_range": {
                    "x": (0.035, 0.095),
                    "y": (-0.003, 0.003),
                    "z": (-0.004, 0.004),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.006, 0.006),
                    "yaw": (-0.006, 0.006),
                },
                "joint_position_scale_range": (0.995, 1.005),
                "joint_velocity_range": (-0.015, 0.015),
            }
        )
        self.events.reset_base.params = reset_params

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = 1
        self.curriculum.terrain_levels = None

        front_target_x = STAIR_FIRST_RISER_X + 0.010
        self.episode_length_s = 4.8
        self.commands.base_velocity.ranges.lin_vel_x = (0.18, 0.30)
        self.rewards.track_lin_vel_xy_exp.weight = 0.08
        self.rewards.stair_forward_progress.weight = 1.5
        self.rewards.stair_forward_progress.params["target_x"] = front_target_x
        self.rewards.stair_forward_velocity.weight = 0.62
        self.rewards.stair_forward_velocity.params.update({"target_x": front_target_x, "max_speed": 0.46})

        self.rewards.stair_approach_progress.weight = 5.0
        self.rewards.stair_approach_progress.params.update(
            {
                "start_x": 0.38,
                "target_x": STAIR_FIRST_RISER_X - 0.010,
                "max_speed": 0.42,
                "yaw_allowance": 0.06,
                "roll_allowance": 0.24,
            }
        )
        self.rewards.stair_approach_handoff.weight = 1.2
        self.rewards.stair_approach_handoff.params.update(
            {
                "target_x": STAIR_FIRST_RISER_X - 0.015,
                "target_base_height": 0.31,
                "x_std": 0.08,
                "y_std": 0.07,
                "yaw_std": 0.10,
                "max_speed": 0.38,
            }
        )

        self.rewards.stair_height_progress.weight = 2.2
        self.rewards.stair_height_progress.params.update(
            {
                "gate_x": STAIR_FIRST_RISER_X - 0.035,
                "target_base_clearance": 0.150,
                "max_pitch_projected": 0.80,
                "max_roll_projected": 0.56,
            }
        )
        self.rewards.stair_base_height_first.weight = 1.4
        self.rewards.stair_base_height_first.params.update(
            {
                "base_clearance": 0.145,
                "gate_margin": -0.025,
                "max_pitch_projected": 0.80,
                "max_roll_projected": 0.56,
            }
        )

        self.rewards.stair_front_feet_on_step.weight = 18.0
        self.rewards.stair_front_feet_on_step.params.update({"x_margin": -0.030, "z_margin": 0.012})
        self.rewards.stair_front_feet_contact_on_step.weight = 48.0
        self.rewards.stair_front_feet_contact_on_step.params.update(
            {
                "x_margin": -0.030,
                "z_margin": 0.006,
                "z_upper_margin": 0.145,
                "lateral_std": 0.22,
                "contact_threshold": 0.5,
                "max_pitch_projected": 0.80,
                "max_roll_projected": 0.56,
            }
        )
        self.rewards.stair_base_on_first_step.weight = 4.0
        self.rewards.stair_base_on_first_step.params.update(
            {
                "base_clearance": 0.150,
                "x_fraction": 0.015,
                "lateral_std": 0.20,
                "max_pitch_projected": 0.80,
                "max_roll_projected": 0.56,
            }
        )

        self.rewards.stair_rear_feet_forward.weight = 0.6
        self.rewards.stair_rear_feet_on_step.weight = 0.0
        self.rewards.stair_rear_feet_contact_on_step.weight = 0.0
        self.rewards.stair_base_rear_feet_on_first_step.weight = 0.0
        self.rewards.stair_base_rear_feet_pose_on_first_step.weight = 0.0
        self.rewards.stair_rear_feet_xz_contact.weight = 0.0
        self.rewards.stair_rear_feet_low_after_base.weight = 0.0

        self.rewards.stair_goal = RewTerm(
            func=gogo_mdp.stair_goal_reward,
            weight=5.0,
            params={
                "target_x": front_target_x,
                "target_z": STAIR_TARGET_Z,
                "xy_std": 0.16,
                "z_std": 0.20,
                "step_height_range": STAIR_STEP_HEIGHT_RANGE,
                "num_steps": STAIR_NUM_STEPS,
                "target_base_clearance": 0.150,
                "sensor_cfg": trunk_sensor_cfg(),
                "contact_threshold": 1.0,
                "max_pitch_projected": 0.76,
                "max_roll_projected": 0.52,
            },
        )
        front_success_params = {
            "target_x": front_target_x,
            "first_riser_x": STAIR_FIRST_RISER_X,
            "step_height_range": STAIR_STEP_HEIGHT_RANGE,
            "step_index": 1,
            "target_base_clearance": 0.135,
            "foot_asset_cfg": foot_asset_cfg(),
            "front_foot_indices": (0, 1),
            "foot_contact_sensor_cfg": foot_sensor_cfg(),
            "front_contact_threshold": 0.5,
            "min_front_feet": 1,
            "min_front_contacts": 1,
            "foot_x_margin": -0.028,
            "foot_z_margin": 0.012,
            "foot_z_upper_margin": 0.150,
            "foot_lateral_tolerance": 0.30,
            "trunk_sensor_cfg": trunk_sensor_cfg(),
            "trunk_contact_threshold": 1.0,
            "max_pitch_projected": 0.78,
            "max_roll_projected": 0.54,
            "max_yaw": 0.30,
            "max_abs_y": 0.075,
            "min_steps": 8,
        }
        self.rewards.stair_success = RewTerm(
            func=gogo_mdp.stair_front_feet_on_step_success_reward,
            weight=320.0,
            params=front_success_params,
        )
        self.terminations.success = DoneTerm(
            func=gogo_mdp.stair_front_feet_on_step_success,
            params=front_success_params,
        )

        self.rewards.stair_low_base_at_riser.weight = -0.15
        self.rewards.stair_low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.125,
                "x_window": (-0.010, 0.16),
            }
        )
        self.rewards.low_base_at_riser_failure.weight = 0.0
        self.terminations.low_base_at_riser.params.update(
            {
                "min_base_clearance": 0.100,
                "x_window": (0.02, 0.20),
                "grace_time_s": 3.0,
                "max_speed_x": -1.0,
            }
        )
        self.rewards.stair_near_goal_timeout.weight = -2.5
        self.rewards.stair_near_goal_timeout.params.update(
            {
                "target_x": front_target_x,
                "target_base_clearance": 0.135,
                "min_progress_fraction": 0.88,
                "min_height_fraction": 0.54,
                "grace_time_s": 2.2,
            }
        )
        self.rewards.stair_stuck.weight = -1.1
        self.rewards.stair_stuck.params.update({"before_goal_x": front_target_x + 0.03, "min_speed_x": 0.026})
        self.rewards.stair_lateral_drift.weight = -5.0
        self.rewards.stair_yaw_drift.weight = -5.5
        self.rewards.stair_yaw_drift.params["yaw_allowance"] = 0.045
        self.rewards.stair_tilt.weight = -5.5
        self.rewards.trunk_contact.weight = -42.0
        self.rewards.trunk_contact_failure.weight = -72.0
        self.rewards.fall_failure.weight = -54.0
        self.rewards.bad_yaw_failure.weight = -32.0

        self.terminations.fall.params.update(
            {
                "min_base_height": 0.115,
                "max_lateral_offset": 0.28,
                "max_pitch_projected": 0.90,
                "max_roll_projected": 0.66,
            }
        )
        self.terminations.bad_yaw.params.update(
            {
                "max_yaw": 0.55,
                "gate_x": STAIR_FIRST_RISER_X - 0.06,
                "grace_time_s": 0.45,
            }
        )
        self.terminations.trunk_contact.params.update(
            {
                "sensor_cfg": trunk_sensor_cfg(),
                "threshold": 5.0,
                "grace_time_s": 0.16,
            }
        )


class GogoA1StairClimbFlatEdgeFrontContactX058X068EnvCfg_PLAY(
    GogoA1StairClimbFlatEdgeFrontContactX058X068EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class GogoA1StairClimbFlatEdgeFrontFootOnlyX058X068EnvCfg(
    GogoA1StairClimbFlatEdgeFrontContactX058X068EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        def foot_asset_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", body_names=foot_names, preserve_order=True)

        def foot_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names=foot_names, preserve_order=True)

        def trunk_sensor_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("contact_forces", body_names="trunk")

        reset_params = dict(self.events.reset_base.params)
        reset_params.update(
            {
                "approach_x_range": (0.46, 0.54),
                "near_step_x_range": (0.50, 0.58),
                "velocity_range": {
                    "x": (0.06, 0.13),
                    "y": (-0.003, 0.003),
                    "z": (-0.004, 0.004),
                    "roll": (0.0, 0.0),
                    "pitch": (-0.006, 0.006),
                    "yaw": (-0.006, 0.006),
                },
            }
        )
        self.events.reset_base.params = reset_params

        front_foot_target_x = STAIR_FIRST_RISER_X - 0.012
        self.episode_length_s = 5.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.18, 0.32)
        self.rewards.track_lin_vel_xy_exp.weight = 0.06
        self.rewards.stair_forward_progress.weight = 1.0
        self.rewards.stair_forward_progress.params["target_x"] = STAIR_FIRST_RISER_X - 0.03
        self.rewards.stair_forward_velocity.weight = 0.75
        self.rewards.stair_forward_velocity.params.update({"target_x": STAIR_FIRST_RISER_X - 0.01, "max_speed": 0.50})

        self.rewards.stair_approach_progress.weight = 7.0
        self.rewards.stair_approach_progress.params.update(
            {
                "start_x": 0.46,
                "target_x": STAIR_FIRST_RISER_X - 0.035,
                "max_speed": 0.46,
                "yaw_allowance": 0.08,
                "roll_allowance": 0.28,
            }
        )
        self.rewards.stair_approach_handoff.weight = 0.4

        self.rewards.stair_height_progress.weight = 0.7
        self.rewards.stair_base_height_first.weight = 0.4
        self.rewards.stair_base_on_first_step.weight = 0.4

        self.rewards.stair_front_feet_on_step.weight = 32.0
        self.rewards.stair_front_feet_on_step.params.update({"x_margin": -0.045, "z_margin": 0.006})
        self.rewards.stair_front_feet_contact_on_step.weight = 92.0
        self.rewards.stair_front_feet_contact_on_step.params.update(
            {
                "x_margin": -0.045,
                "z_margin": 0.012,
                "z_upper_margin": 0.165,
                "lateral_std": 0.26,
                "contact_threshold": 0.4,
                "max_pitch_projected": 0.92,
                "max_roll_projected": 0.68,
            }
        )

        front_success_params = {
            "target_x": front_foot_target_x,
            "first_riser_x": STAIR_FIRST_RISER_X,
            "step_height_range": STAIR_STEP_HEIGHT_RANGE,
            "step_index": 1,
            "target_base_clearance": 0.055,
            "foot_asset_cfg": foot_asset_cfg(),
            "front_foot_indices": (0, 1),
            "foot_contact_sensor_cfg": foot_sensor_cfg(),
            "front_contact_threshold": 0.4,
            "min_front_feet": 1,
            "min_front_contacts": 1,
            "foot_x_margin": -0.045,
            "foot_z_margin": 0.012,
            "foot_z_upper_margin": 0.165,
            "foot_lateral_tolerance": 0.36,
            "trunk_sensor_cfg": trunk_sensor_cfg(),
            "trunk_contact_threshold": 120.0,
            "max_pitch_projected": 0.92,
            "max_roll_projected": 0.68,
            "max_yaw": 0.55,
            "max_abs_y": 0.14,
            "min_steps": 5,
        }
        self.rewards.stair_success = RewTerm(
            func=gogo_mdp.stair_front_feet_on_step_success_reward,
            weight=220.0,
            params=front_success_params,
        )
        self.terminations.success = DoneTerm(
            func=gogo_mdp.stair_front_feet_on_step_success,
            params=front_success_params,
        )

        self.rewards.stair_goal.weight = 1.5
        self.rewards.stair_near_goal_timeout.weight = -0.8
        self.rewards.stair_stuck.weight = -0.5
        self.rewards.stair_lateral_drift.weight = -4.0
        self.rewards.stair_yaw_drift.weight = -3.5
        self.rewards.stair_tilt.weight = -3.2
        self.rewards.trunk_contact.weight = -95.0
        self.rewards.trunk_contact_failure.weight = -5.0
        self.terminations.trunk_contact.params.update(
            {
                "sensor_cfg": trunk_sensor_cfg(),
                "threshold": 120.0,
                "grace_time_s": 5.5,
            }
        )
        self.terminations.fall.params.update(
            {
                "min_base_height": 0.105,
                "max_lateral_offset": 0.40,
                "max_pitch_projected": 1.00,
                "max_roll_projected": 0.76,
            }
        )
        self.terminations.bad_yaw.params.update(
            {
                "max_yaw": 0.82,
                "gate_x": STAIR_FIRST_RISER_X - 0.08,
                "grace_time_s": 0.80,
            }
        )


class GogoA1StairClimbFlatEdgeFrontFootOnlyX058X068EnvCfg_PLAY(
    GogoA1StairClimbFlatEdgeFrontFootOnlyX058X068EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_settings(self)
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
