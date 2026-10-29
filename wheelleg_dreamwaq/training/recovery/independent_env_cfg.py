from __future__ import annotations

from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .env_cfg import RecoveryEnvCfg


@configclass
class IndependentStandingEnvCfg(RecoveryEnvCfg):
    episode_length_s = 10.0
    observation_space = 110
    state_space = 41


def build_independent_env_cfg(resolved_config: dict) -> IndependentStandingEnvCfg:
    if resolved_config["schema_version"] != "RecoveryIndependentStandV1":
        raise ValueError("独立站立环境要求完整新契约")
    environment = resolved_config["environment"]
    contract = resolved_config["contract"]
    cfg = IndependentStandingEnvCfg()
    cfg.resolved_config = deepcopy(resolved_config)
    cfg.seed = int(environment["seed"])
    cfg.scene.num_envs = int(environment["num_envs"])
    cfg.scene.env_spacing = float(environment["env_spacing_m"])
    cfg.ground.spawn.size = (*environment["ground_size_xy_m"], 0.10)
    cfg.sim.device = str(environment["device"])
    cfg.sim.dt = float(contract["isaac_physics_dt_s"])
    cfg.decimation = int(contract["isaac_decimation"])
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = float(contract["max_recovery_steps"] * contract["control_dt_s"])
    cfg.log_dir = str(environment["run_dir"])
    cfg.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1.,
        restitution=0., friction_combine_mode="multiply", restitution_combine_mode="multiply")
    if cfg.scene.num_envs <= 0 or cfg.decimation != 4 or cfg.sim.dt != 0.005:
        raise ValueError("独立站立物理时序或环境数量无效")
    allowed = {"gpu_max_rigid_contact_count", "gpu_max_rigid_patch_count", "gpu_found_lost_pairs_capacity",
               "gpu_found_lost_aggregate_pairs_capacity", "gpu_total_aggregate_pairs_capacity"}
    for name, value in environment["physx_buffers"].items():
        if name not in allowed or type(value) is not int or value <= 0:
            raise ValueError("PhysX 接触容量字段无效")
        setattr(cfg.sim.physx, name, value)
    return cfg
