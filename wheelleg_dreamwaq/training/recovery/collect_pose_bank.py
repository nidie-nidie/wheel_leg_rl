from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

from .contact_model import file_sha256, payload_sha256
from .contracts import write_json
from .mujoco_runtime import MujocoRecoveryRuntime
from .pose_bank import PoseBank


@dataclass(frozen=True)
class CollectionConfig:
    seed: int
    trajectories: int
    steps_per_trajectory: int
    record_stride: int
    train_fraction: float
    frozen_walker_fraction: float
    drop_clearance_m: float
    fall_tilt_min_rad: float
    near_upright_max_rad: float
    supine_min_rad: float
    dominance_margin: float
    static_linear_speed_m_s: float
    static_angular_speed_rad_s: float
    static_joint_speed_rad_s: float
    maximum_contact_penetration_m: float
    minimum_clearance_m: float
    maximum_initial_angular_speed_rad_s: float
    allowed_body_pairs: list[list[str]]

    def __post_init__(self):
        if self.seed < 0 or self.trajectories < 2 or self.steps_per_trajectory < 5 or self.record_stride < 1:
            raise ValueError("姿态库生成数量、种子或步数无效")
        if not 0 < self.train_fraction < 1 or not 0.3 <= self.frozen_walker_fraction <= 1:
            raise ValueError("姿态库需要独立测试轨迹与至少30%的行走失稳采集轨迹")
        values = asdict(self)
        numeric = [value for value in values.values() if isinstance(value, (int, float))]
        if not np.isfinite(numeric).all():
            raise ValueError("采集配置包含非有限数值")
        if not 0 <= self.near_upright_max_rad < self.fall_tilt_min_rad < self.supine_min_rad <= np.pi:
            raise ValueError("姿态倾角分类范围无效")
        if self.drop_clearance_m < 0 or self.maximum_contact_penetration_m < 0 or self.dominance_margin < 0:
            raise ValueError("姿态采集几何门限无效")
        if min(self.static_linear_speed_m_s, self.static_angular_speed_rad_s, self.static_joint_speed_rad_s,
               self.maximum_initial_angular_speed_rad_s) <= 0:
            raise ValueError("运动分类速度门限必须为正数")


def classify_pose(state, config: CollectionConfig) -> str | None:
    gravity = state.projected_gravity_control
    tilt = float(np.arccos(np.clip(-gravity[2], -1, 1)))
    if tilt <= config.near_upright_max_rad:
        return "near_upright"
    if tilt >= config.supine_min_rad:
        return "supine"
    if tilt < config.fall_tilt_min_rad or abs(abs(gravity[0]) - abs(gravity[1])) < config.dominance_margin:
        return None
    if abs(gravity[0]) > abs(gravity[1]):
        return "front" if gravity[0] > 0 else "back"
    return "left" if gravity[1] > 0 else "right"


def minimum_mesh_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    minimum = np.inf
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] == 0 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[start:start + count]
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        minimum = min(minimum, float(world[:, 2].min()))
    if not np.isfinite(minimum):
        raise ValueError("模型缺少真实网格")
    return float(minimum)


def orient_for_drop(runtime: MujocoRecoveryRuntime, category: str, generator: np.random.Generator,
                    clearance_m: float, angular_speed: float = 0.0) -> None:
    runtime.reset_nominal()
    body = runtime.model_map.base_body_id
    center = runtime.data.xipos[body].copy()
    axes = {"front": (1, 1.4), "back": (1, -1.4), "left": (0, 1.4),
            "right": (0, -1.4), "supine": (1, np.pi), "near_upright": (1, 0.0)}
    index, angle = axes[category]
    angle += float(generator.uniform(-0.12, 0.12))
    axis_control = np.eye(3)[index]
    axis_usd = runtime.walker_contract.r_control_from_mujoco.T @ axis_control
    rotation = np.zeros(4)
    mujoco.mju_axisAngle2Quat(rotation, axis_usd, angle)
    free = runtime.model_map.free_joint_id
    qadr, vadr = int(runtime.model.jnt_qposadr[free]), int(runtime.model.jnt_dofadr[free])
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, rotation, runtime.data.qpos[qadr + 3:qadr + 7].copy())
    runtime.data.qpos[qadr + 3:qadr + 7] = result
    mujoco.mj_forward(runtime.model, runtime.data)
    runtime.data.qpos[qadr:qadr + 3] += center - runtime.data.xipos[body]
    mujoco.mj_forward(runtime.model, runtime.data)
    runtime.data.qpos[qadr + 2] += clearance_m - minimum_mesh_height(runtime.model, runtime.data)
    if angular_speed:
        direction = generator.normal(size=3)
        direction /= np.linalg.norm(direction)
        runtime.data.qvel[vadr + 3:vadr + 6] = direction * angular_speed
    mujoco.mj_forward(runtime.model, runtime.data)
    runtime.history[:] = runtime._frame()
    runtime.clean_history[:] = runtime._physical_frame()
    runtime.walker_history[:] = runtime._walker_frame(np.array([0.0, 0.0, 0.20]))


def collect(runtime: MujocoRecoveryRuntime, config: CollectionConfig, *, audit_path: Path | None = None) -> PoseBank:
    generator = np.random.default_rng(config.seed)
    categories = ("front", "back", "left", "right", "supine", "near_upright")
    records = []
    audit = []
    for trajectory in range(config.trajectories):
        expert = trajectory < int(np.ceil(config.frozen_walker_fraction * config.trajectories))
        requested = categories[trajectory % len(categories)]
        split = "train" if generator.random() < config.train_fraction else "test"
        orient_for_drop(runtime, "near_upright" if expert else requested, generator, config.drop_clearance_m,
                        float(generator.uniform(0, config.maximum_initial_angular_speed_rad_s)) if expert else 0.0)
        trajectory_id = f"mujoco-seed{config.seed}-trajectory{trajectory}"
        accepted = 0
        source_fell = False
        for step in range(config.steps_per_trajectory):
            action = runtime.walker_action() if expert else np.zeros(6)
            result = runtime.step(action, locomotion=expert)
            state = runtime.state()
            category = classify_pose(state, config)
            if expert and category in categories[:5]:
                source_fell = True
            valid = all(substep["loop_error_m"] < runtime.contract.max_loop_error_m and
                        substep["contact"]["maximum_penetration_m"] <= config.maximum_contact_penetration_m and
                        substep["clearance"]["minimum_nonallowed_clearance_m"] >= config.minimum_clearance_m
                        for substep in result.substeps)
            if not valid:
                audit.append({"trajectory_id": trajectory_id, "step": step + 1, "reason": "physical_validation_failed"})
                break
            if step < 4 or (step + 1) % config.record_stride or category is None:
                continue
            if expert and not source_fell:
                continue
            static = (np.linalg.norm(state.com_linear_velocity_control) <= config.static_linear_speed_m_s and
                      np.linalg.norm(state.angular_velocity_control) <= config.static_angular_speed_rad_s and
                      np.max(np.abs(runtime.data.qvel[6:])) <= config.static_joint_speed_rad_s)
            record = runtime.pose_record(record_id=f"{trajectory_id}-step{step + 1}", trajectory_id=trajectory_id,
                                         split=split, category=category, motion_kind="static" if static else "dynamic",
                                         metadata={"origin": "frozen_walker_fall" if expert else "rigid_drop",
                                                   "collection_config_sha256": payload_sha256(asdict(config)),
                                                   "requested_category": requested,
                                                   "physical_validation_passed": True})
            records.append(record)
            accepted += 1
        audit.append({"trajectory_id": trajectory_id, "accepted_records": accepted, "frozen_walker_fall": source_fell})
    metadata = {"schema_version": "RecoveryPoseCollectionV1", "collection_config": asdict(config),
                "collection_config_sha256": payload_sha256(asdict(config)), "audit": audit,
                "contact_manifest_sha256": file_sha256(runtime.contact_manifest_path),
                "parameter_plan_sha256": runtime.parameter_sha256,
                "conversion_validation": "required", "formal_validation": "required"}
    if audit_path is not None:
        write_json(audit_path, {**metadata, "record_count": len(records)})
    if not records:
        raise RuntimeError("真实物理采集没有产生合法姿态，采集审计记录保留全部拒绝原因")
    return PoseBank(records=records, metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--walker-export", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--actual-parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = CollectionConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    parameters = json.loads(args.actual_parameters.read_text(encoding="utf-8"))
    runtime = MujocoRecoveryRuntime(args.contact_manifest, args.walker_export,
                                    allowed_body_pairs=config.allowed_body_pairs, actual_parameters=parameters)
    bank = collect(runtime, config, audit_path=args.output.with_suffix(".audit.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bank.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
