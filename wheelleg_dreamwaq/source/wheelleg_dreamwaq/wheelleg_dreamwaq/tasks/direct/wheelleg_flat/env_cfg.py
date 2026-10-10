from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG
from wheelleg_dreamwaq.schemas.normalization import NormalizationV2
from wheelleg_dreamwaq.schemas.physics import DECIMATION, SIM_DT_S
from wheelleg_dreamwaq.schemas.randomization import NOMINAL_TRAINING_PROFILE_V1

from .commands import CommandRanges
from .control import ControlLimits
from .rewards import RewardWeights
from .terminations import TerminationLimits


@configclass
class GroundPlaneCfg:
    prim_path = "/World/Ground"
    translation: tuple[float, float, float] = (0.0, 0.0, -0.05)
    spawn: sim_utils.CuboidCfg = sim_utils.CuboidCfg(
        size=(100.0, 100.0, 0.10),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.22, 0.25)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )


@configclass
class WheelLegFlatEnvCfg(DirectRLEnvCfg):
    seed = 42
    decimation = DECIMATION
    episode_length_s = 10.0
    action_space = 6
    observation_space = 25
    state_space = 41

    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT_S,
        render_interval=decimation,
        device="cuda:0",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    robot_cfg = WHEELLEG_CFG

    q_nominal = (-0.33367134, 0.33367134, -0.33367134, 0.33367134)
    control = ControlLimits()
    ground = GroundPlaneCfg()
    commands = CommandRanges()
    normalization = NormalizationV2()
    randomization = NOMINAL_TRAINING_PROFILE_V1
    reward_weights = RewardWeights()
    termination = TerminationLimits()
