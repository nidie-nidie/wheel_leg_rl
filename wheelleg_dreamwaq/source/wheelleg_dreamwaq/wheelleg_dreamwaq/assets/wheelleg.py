from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from wheelleg_dreamwaq.schemas.physics import SOLVER_POSITION_ITERATIONS, SOLVER_VELOCITY_ITERATIONS

from .asset_contract import ASSET_BUNDLE_V2
from .paths import asset_root_v2


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


WHEELLEG_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    articulation_root_prim_path="/base_link",
    spawn=sim_utils.UsdFileCfg(
        usd_path=(asset_root_v2() / ASSET_BUNDLE_V2.entry_file).as_posix(),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=SOLVER_POSITION_ITERATIONS,
            solver_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.20003),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "jIO": 0.33367157,
            "jAG": 0.33367112,
            "jIJ": -0.33367151,
            "jAB": -0.33367088,
            "jOP": -0.37150037,
            "jIO_dummy_child_link1": -0.01975457,
            "jIO_dummy_child_link2": -0.01975455,
            "jGH": 0.37148619,
            "jAG_dummy_child_link1": -0.00834354,
            "jAG_dummy_child_link2": -0.00834353,
            "jJM": -0.37146848,
            "jBE": 0.37146658,
            "jwheel_left": 0.0,
            "jwheel_right": 0.0,
            "jMK": 0.07562718,
            "jEC": 0.07562435,
            "jKN": -0.37150618,
            "jMK_dummy_child1": -0.06897040,
            "jMK_dummy_child2": -0.06897411,
            "jCF": 0.37150583,
            "jEC_dummy_child_link1": 0.00279344,
            "jEC_dummy_child_link2": 0.00279370,
            "jKN_dummy_child_link1": -0.06563117,
            "jKN_dummy_child_link2": -0.06613248,
            "jCF_dummy_child_link1": 0.01561109,
            "jCF_dummy_child_link2": 0.01608925,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=LEG_JOINT_NAMES,
            effort_limit_sim=18.0,
            velocity_limit_sim=45.0,
            stiffness=120.0,
            damping=4.0,
            armature=0.05,
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINT_NAMES,
            effort_limit_sim=9.0,
            velocity_limit_sim=45.0,
            stiffness=0.0,
            damping=0.6,
            armature=0.05,
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
        ),
        "passive": ImplicitActuatorCfg(
            joint_names_expr=PASSIVE_JOINT_NAMES,
            effort_limit_sim=18.0,
            velocity_limit_sim=80.0,
            stiffness=0.0,
            damping=0.05,
            armature=0.005,
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
        ),
    },
)
