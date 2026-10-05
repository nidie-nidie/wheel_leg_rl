"""PACE system-identification task configuration for Unitree A1."""

import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg
from pace_sim2real.utils import PaceDCMotorCfg


A1_PACE_JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)


A1_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[
        ".*_hip_joint",
        ".*_thigh_joint",
        ".*_calf_joint",
    ],
    saturation_effort=33.5,
    effort_limit=33.5,
    velocity_limit=21.0,
    stiffness={".*": 25.0},
    damping={".*": 2.0},
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=10,
)


@configclass
class A1PaceCfg(PaceCfg):
    """Optimizer and data contract for Unitree A1."""

    robot_name: str = "a1"
    data_dir: str = "a1/chirp_data.pt"
    bounds_params: torch.Tensor = torch.zeros((49, 2))
    joint_order: list[str] = list(A1_PACE_JOINT_ORDER)

    def __post_init__(self) -> None:
        self.bounds_params[:12, 0] = 1.0e-5
        self.bounds_params[:12, 1] = 1.0
        self.bounds_params[12:24, 1] = 7.0
        self.bounds_params[24:36, 1] = 0.5
        self.bounds_params[36:48, 0] = -0.1
        self.bounds_params[36:48, 1] = 0.1
        self.bounds_params[48, 1] = 10.0
        self.cmaes.save_optimization_process = True


@configclass
class A1PaceSceneCfg(PaceSim2realSceneCfg):
    """Fixed-base, no-contact A1 scene used by PACE."""

    robot: ArticulationCfg = UNITREE_A1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(0.0, 1.0, 0.0, 0.0),
            joint_pos=dict(UNITREE_A1_CFG.init_state.joint_pos),
            joint_vel=dict(UNITREE_A1_CFG.init_state.joint_vel),
        ),
        actuators={"base_legs": A1_PACE_ACTUATOR_CFG},
    )
    feet_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        history_length=1,
        track_air_time=False,
    )


@configclass
class A1PaceEnvCfg(PaceSim2realEnvCfg):
    """PACE environment running A1 simulation and control at 500 Hz."""

    scene: A1PaceSceneCfg = A1PaceSceneCfg()
    sim2real: PaceCfg = A1PaceCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.dt = 0.002
        self.decimation = 1


@configclass
class A1PaceSyntheticEnvCfg(A1PaceEnvCfg):
    """Synthetic proof task using the unchanged official optimizer schedule."""

    def __post_init__(self) -> None:
        super().__post_init__()


@configclass
class A1PaceSmokeEnvCfg(A1PaceEnvCfg):
    """One-generation A1 fit task for real-data smoke checks."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim2real.robot_name = "a1_smoke"
        self.sim2real.cmaes.max_iteration = 1
        self.sim2real.cmaes.save_interval = 1
