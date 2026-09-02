from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from serial_leg_rl.assets.sim_urdf import PROJECT_ROOT


CLOSED_CHAIN_USD_PATH = PROJECT_ROOT / "wheel_leg_urdf4_usd (1)" / "wheel_leg_urdf4" / "wheel_leg_urdf4.usd"


LEG_JOINT_NAMES = ["jIJ", "jIO", "jAB", "jAG"]
WHEEL_JOINT_NAMES = ["jwheel_left", "jwheel_right"]
PASSIVE_JOINT_NAMES = [
    "jOP",
    "jIO_dummy_child_link1",
    "jIO_dummy_child_link2",
    "jGH",
    "jAG_dummy_child_link1",
    "jAG_dummy_child_link2",
    "jJM",
    "jMK",
    "jKN",
    "jKN_dummy_child_link1",
    "jKN_dummy_child_link2",
    "jMK_dummy_child1",
    "jMK_dummy_child2",
    "jBE",
    "jEC",
    "jCF",
    "jCF_dummy_child_link1",
    "jCF_dummy_child_link2",
    "jEC_dummy_child_link1",
    "jEC_dummy_child_link2",
]


WHEEL_LEG_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=CLOSED_CHAIN_USD_PATH.as_posix(),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.20),
        joint_pos={
            "jIJ": -0.30,
            "jIO": 0.35,
            "jAB": -0.30,
            "jAG": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=LEG_JOINT_NAMES,
            effort_limit=20.0,
            effort_limit_sim=20.0,
            velocity_limit=45.0,
            velocity_limit_sim=80.0,
            stiffness=80.0,
            damping=3.0,
            armature=0.02,
            friction=0.02,
            min_delay=0,
            max_delay=1,
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=WHEEL_JOINT_NAMES,
            effort_limit=2.42,
            effort_limit_sim=2.42,
            velocity_limit=51.31,
            velocity_limit_sim=51.31,
            stiffness=0.0,
            damping=0.35,
            armature=0.01,
            friction=0.01,
            min_delay=0,
            max_delay=1,
        ),
        "passive_joints": DelayedPDActuatorCfg(
            joint_names_expr=PASSIVE_JOINT_NAMES,
            effort_limit=20.0,
            effort_limit_sim=20.0,
            velocity_limit=40.0,
            velocity_limit_sim=80.0,
            stiffness=60.0,
            damping=2.5,
            armature=0.005,
            friction=0.02,
            min_delay=0,
            max_delay=0,
        ),
    },
)
