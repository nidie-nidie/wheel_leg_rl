from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

from .collect_pose_bank import classify_pose, orient_for_drop
from .contracts import CATEGORIES, file_sha256, stable_hash, write_json
from .mujoco_runtime import MujocoRecoveryRuntime
from .pose_bank import PoseBank


@dataclass(frozen=True)
class BootstrapClassification:
    fall_tilt_min_rad: float = 0.90
    near_upright_max_rad: float = 0.25
    supine_min_rad: float = 2.65
    dominance_margin: float = 0.20
    static_linear_speed_m_s: float = 0.03
    static_angular_speed_rad_s: float = 0.10
    static_joint_speed_rad_s: float = 0.15


@dataclass(frozen=True)
class BootstrapCollectionConfig:
    seed: int = 42001
    training_trajectories_per_category: int = 8
    testing_trajectories_per_category: int = 2
    maximum_attempts_per_record: int = 20
    minimum_record_control_steps: int = 5
    maximum_record_control_steps: int = 25
    drop_clearance_m: float = 0.001
    maximum_initial_angular_speed_rad_s: float = 0.35
    maximum_ground_penetration_m: float = 0.001


class BootstrapMechanicalAudit:
    def __init__(self, runtime, limits):
        self.base_audit = runtime.geometry_audit
        self.runtime = runtime
        self.limits = limits
        self.failure = None
        self.checked_substeps = 0
        self.last_time_s = None
        self.maximum_native_joint_speed = 0.0
        self.joints = []
        for index in range(runtime.model.njnt):
            if runtime.model.jnt_type[index] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = mujoco.mj_id2name(runtime.model, mujoco.mjtObj.mjOBJ_JOINT, index)
            if name not in limits:
                raise ValueError(f'缺少具名关节机械范围来源: {name}')
            low, high = limits[name]
            if low is not None and (high is None or not np.isfinite([low,high]).all() or low >= high):
                raise ValueError(f'机械范围无效: {name}')
            self.joints.append((name,int(runtime.model.jnt_qposadr[index]),int(runtime.model.jnt_dofadr[index]),low,high))

    def inspect(self, data):
        if self.last_time_s is None or float(data.time) != self.last_time_s:
            self.checked_substeps += 1
            self.last_time_s = float(data.time)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            self.failure = 'nonfinite_native_physics'
        for name,qadr,vadr,low,high in self.joints:
            self.maximum_native_joint_speed = max(self.maximum_native_joint_speed,abs(float(data.qvel[vadr])))
            if low is not None and not low <= data.qpos[qadr] <= high:
                self.failure = f'joint_mechanical_range/{name}'
        return self.base_audit.inspect(data)


def collect_bootstrap(runtime, config, classification, mechanical_limits, *, output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('bootstrap姿态库必须写入新版本目录')
    output.mkdir(parents=True)
    classification_payload = asdict(classification)
    config_payload = asdict(config)
    classification_hash = stable_hash(classification_payload)
    write_json(output/'classification.json',classification_payload)
    write_json(output/'collection_config.json',config_payload)
    write_json(output/'mechanical_limits.json',mechanical_limits)
    if not 5 <= config.minimum_record_control_steps <= config.maximum_record_control_steps:
        raise ValueError('每条记录至少需要五个真实物理控制步')
    generator = np.random.default_rng(config.seed)
    audit = BootstrapMechanicalAudit(runtime,mechanical_limits['joint_position_limits_native_rad'])
    runtime.geometry_audit = audit
    records,trajectories = [],[]
    serial = 0
    for category in CATEGORIES:
        for split,count in (('train',config.training_trajectories_per_category),('test',config.testing_trajectories_per_category)):
            accepted = 0
            for attempt in range(count*config.maximum_attempts_per_record):
                if accepted == count:
                    break
                trajectory_id = f'bootstrap-seed{config.seed}-{category}-{split}-trajectory{serial}'
                serial += 1
                # 绕control-x的正转对应重力gy<0；保存类别始终以实际IMU重力判定。
                orientation_request = {'left':'right','right':'left'}.get(category,category)
                orient_for_drop(runtime,orientation_request,generator,config.drop_clearance_m,
                    float(generator.uniform(0,config.maximum_initial_angular_speed_rad_s)))
                audit.failure = None
                audit.last_time_s = 0.0
                start_checks = audit.checked_substeps
                initial_category = classify_pose(runtime.state(),classification)
                info = {'trajectory_id':trajectory_id,'requested_category':category,'initial_actual_category':initial_category,
                        'split':split,'accepted':False,'failure_reason':None,'control_steps':0}
                wanted_step = int(generator.integers(config.minimum_record_control_steps,config.maximum_record_control_steps+1))
                maximum_loop = maximum_penetration = 0.0
                minimum_convex = np.inf
                for step_index in range(config.maximum_record_control_steps):
                    action = runtime.walker_action() if category == 'near_upright' else np.zeros(6)
                    result = runtime.step(action, locomotion=category == 'near_upright')
                    info['control_steps'] = step_index+1
                    for substep in result.substeps:
                        maximum_loop = max(maximum_loop,substep['loop_error_m'])
                        maximum_penetration = max(maximum_penetration,substep['contact']['maximum_penetration_m'])
                        minimum_convex = min(minimum_convex,substep['clearance']['minimum_nonallowed_clearance_m'])
                    reason = (audit.failure or
                        ('closed_loop_error' if maximum_loop >= runtime.contract.max_loop_error_m else None) or
                        ('excessive_ground_penetration' if maximum_penetration > config.maximum_ground_penetration_m else None))
                    if reason:
                        info['failure_reason'] = reason
                        break
                    actual_category = classify_pose(runtime.state(),classification)
                    if step_index+1 < wanted_step or actual_category != category:
                        continue
                    state = runtime.state()
                    native_joint_speed = max(abs(float(runtime.data.qvel[vadr])) for _,_,vadr,_,_ in audit.joints)
                    static = (np.linalg.norm(state.com_linear_velocity_control) <= classification.static_linear_speed_m_s and
                              np.linalg.norm(state.angular_velocity_control) <= classification.static_angular_speed_rad_s and
                              native_joint_speed <= classification.static_joint_speed_rad_s)
                    metadata = {'origin':'bootstrap_frozen_walker' if category=='near_upright' else 'bootstrap_rigid_drop',
                        'stage':'bootstrap_diagnostic','real_nonconvex_validation_pending':True,
                        'classification_sha256':classification_hash,'collection_config_sha256':stable_hash(config_payload),
                        'requested_category':category,'actual_category':actual_category,
                        'initial_actual_category':initial_category,'orientation_request':orientation_request,
                        'projected_gravity_control':state.projected_gravity_control.tolist(),
                        'maximum_substep_loop_error_m':maximum_loop,'maximum_substep_ground_penetration_m':maximum_penetration,
                        'minimum_diagnostic_convex_clearance_m':minimum_convex,
                        'diagnostic_internal_convex_overlap':minimum_convex<0,
                        'mechanical_limits_sha256':stable_hash(mechanical_limits),
                        'mechanical_physical_range_calibrated':mechanical_limits['mechanical_physical_range_calibrated'],
                        'physical_control_steps':step_index+1,'physical_substeps_checked':audit.checked_substeps-start_checks,
                        'static_thresholds_satisfied':bool(static),'source_category_semantics':'gravity_control_gx_positive_front_gy_positive_left'}
                    record=runtime.pose_record(record_id=f'{trajectory_id}-step{step_index+1}',trajectory_id=trajectory_id,
                        split=split,category=actual_category,motion_kind='static' if static else 'dynamic',metadata=metadata)
                    records.append(record)
                    info['accepted']=True
                    accepted += 1
                    break
                if not info['accepted'] and info['failure_reason'] is None:
                    info['failure_reason']='actual_category_not_available_after_five_real_frames'
                info.update({'maximum_loop_error_m':maximum_loop,'maximum_ground_penetration_m':maximum_penetration,
                             'minimum_diagnostic_convex_clearance_m':float(minimum_convex)})
                trajectories.append(info)
            if accepted != count:
                write_json(output/'collection_audit.json',{'stage':'bootstrap_diagnostic','trajectories':trajectories,
                    'failure_category':category,'failure_split':split,'accepted_records':len(records)})
                raise RuntimeError(f'未达到bootstrap类别轨迹数量: {category}/{split}, accepted={accepted}/{count}')
    bank=PoseBank(records,{'stage':'bootstrap_diagnostic','real_nonconvex_validation_pending':True,
        'classification':classification_payload,'classification_sha256':classification_hash,'collection_config':config_payload,
        'collection_config_sha256':stable_hash(config_payload),'mechanical_limits':mechanical_limits,
        'mechanical_limits_sha256':stable_hash(mechanical_limits),'contact_manifest_sha256':file_sha256(runtime.contact_manifest_path),
        'mechanical_physical_range_calibrated':mechanical_limits['mechanical_physical_range_calibrated'],
        'parameter_plan_sha256':runtime.parameter_sha256,'formal_validation':'required','conversion_validation':'required'})
    bank.save(output/'pose_bank.json')
    coverage={category:{split:sum(r.initial_category==category and r.split==split for r in records) for split in ('train','test')} for category in CATEGORIES}
    write_json(output/'collection_audit.json',{'stage':'bootstrap_diagnostic','trajectories':trajectories,
        'coverage':coverage,'pose_bank_sha256':bank.sha256,'records':len(records),'physical_substeps_checked':audit.checked_substeps,
        'real_nonconvex_validation_pending':True,'formal_validation':'required'})
    print(json.dumps({'output':str(output/'pose_bank.json'),'coverage':coverage,'records':len(records),
        'parameter_plan_sha256':runtime.parameter_sha256,'pose_bank_sha256':bank.sha256},indent=2))
    return bank


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--contact-manifest',type=Path,required=True)
    parser.add_argument('--walker-export',type=Path,required=True)
    parser.add_argument('--actual-parameters',type=Path,required=True)
    parser.add_argument('--mechanical-limits',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    runtime=MujocoRecoveryRuntime(args.contact_manifest,args.walker_export,
        actual_parameters=json.loads(args.actual_parameters.read_text(encoding='utf-8')))
    collect_bootstrap(runtime,BootstrapCollectionConfig(),BootstrapClassification(),
        json.loads(args.mechanical_limits.read_text(encoding='utf-8')),output_dir=args.output_dir)


if __name__=='__main__':
    main()
