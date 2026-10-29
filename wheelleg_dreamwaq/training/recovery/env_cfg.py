from __future__ import annotations

from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import GroundPlaneCfg

from .contracts import RecoveryContractV2


@configclass
class RecoveryEnvCfg(DirectRLEnvCfg):
    seed = 42
    decimation = 4
    episode_length_s = 8.0
    is_finite_horizon = True
    action_space = 6
    observation_space = 110
    state_space = 34
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4, device="cuda:0")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8, env_spacing=2.0, replicate_physics=True, clone_in_fabric=False
    )
    robot_cfg = WHEELLEG_CFG.copy()
    robot_cfg.spawn.activate_contact_sensors = True
    robot_cfg.spawn.articulation_props.enabled_self_collisions = False
    ground = GroundPlaneCfg()
    resolved_config: dict = {}


def build_env_cfg(resolved_config: dict) -> RecoveryEnvCfg:
    environment = resolved_config["environment"]
    contract = RecoveryContractV2(**resolved_config["contract"])
    cfg = RecoveryEnvCfg()
    if resolved_config["schema_version"] == "RecoveryResolvedConfigV3":
        cfg.observation_space = resolved_config["training_contract"]["actor_input_dim"]
        cfg.state_space = resolved_config["training_contract"]["critic_input_dim"]
    cfg.resolved_config = deepcopy(resolved_config)
    cfg.seed = int(environment["seed"])
    cfg.scene.num_envs = int(environment["num_envs"])
    cfg.scene.env_spacing = float(environment["env_spacing_m"])
    cfg.ground.spawn.size = (*environment["ground_size_xy_m"], 0.10)
    if cfg.scene.num_envs <= 0:
        raise ValueError("起身并行环境数必须为正整数")
    cfg.sim.device = str(environment["device"])
    cfg.sim.dt = contract.isaac_physics_dt_s
    cfg.decimation = contract.isaac_decimation
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = contract.max_recovery_steps * contract.control_dt_s
    cfg.log_dir = str(environment["run_dir"])
    for name, value in environment["physx_buffers"].items():
        if name not in {"gpu_max_rigid_contact_count", "gpu_max_rigid_patch_count", "gpu_found_lost_pairs_capacity",
                        "gpu_found_lost_aggregate_pairs_capacity", "gpu_total_aggregate_pairs_capacity"}:
            raise ValueError(f"未知 PhysX 接触容量字段: {name}")
        if not isinstance(value, int) or value <= 0:
            raise ValueError("PhysX 接触容量必须为正整数")
        setattr(cfg.sim.physx, name, value)
    cfg.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )
    return cfg
