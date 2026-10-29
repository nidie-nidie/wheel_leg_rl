from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from isaaclab.app import AppLauncher


def _arguments():
    parser = argparse.ArgumentParser(description="验证世界竖直辅助力、COM 作用点及瞬时施力清除")
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _run(arguments):
    import torch
    from pxr import Gf, UsdPhysics

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz
    from isaaclab.utils.wrench_composer import WrenchComposer

    from recovery.contracts import file_sha256, source_sha256
    from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2, verify_asset_bundle
    from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG

    dt = 0.005
    force_n = 20.0
    probe_mass_kg = 2.0
    probe_com_offset_m = (1.27, 0.0, 0.0)
    orientations = (
        ("upright", 0.0, 0.0, 0.0),
        ("left", torch.pi / 2, 0.0, 0.0),
        ("right", -torch.pi / 2, 0.0, 0.0),
        ("front", 0.0, torch.pi / 2, 0.0),
        ("back", 0.0, -torch.pi / 2, 0.0),
        ("inverted", torch.pi, 0.0, 0.0),
    )
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=dt, gravity=(0.0, 0.0, -9.81), device=arguments.device)
    )
    asset_report = verify_asset_bundle(bundle=ASSET_BUNDLE_V2)

    # 真实偏置刚体独立隔离作用点语义；不使用地面或接触约束。
    for index in range(len(orientations) * 2):
        sim_utils.create_prim(f"/World/ProbeEnv_{index}", "Xform")
    probe = RigidObject(
        RigidObjectCfg(
            prim_path="/World/ProbeEnv_.*/Body",
            spawn=sim_utils.CuboidCfg(
                size=(0.20, 0.18, 0.16),
                mass_props=sim_utils.MassPropertiesCfg(mass=probe_mass_kg),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=100.0,
                    max_angular_velocity=100.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 20.0, 10.0)),
        )
    )
    for index in range(len(orientations) * 2):
        mass_api = UsdPhysics.MassAPI.Apply(sim.stage.GetPrimAtPath(f"/World/ProbeEnv_{index}/Body"))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*probe_com_offset_m))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.20, 0.30, 0.40))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))

    # 实际机器人保留源资产、闭链和执行器参数，仅将两个副本放到无接触高空。
    for index in range(2):
        sim_utils.create_prim(f"/World/RobotEnv_{index}", "Xform")
    robot_cfg = WHEELLEG_CFG.copy()
    robot_cfg.prim_path = "/World/RobotEnv_.*/Robot"
    robot_cfg.init_state.pos = (-20.0, 0.0, 10.0)
    robot = Articulation(robot_cfg)
    sim.reset()

    device = sim.device
    probe_ids = torch.arange(probe.num_instances, device=device)
    forced_probe_ids = probe_ids[1::2]
    eulers = torch.tensor([item[1:] for item in orientations], device=device)
    quaternions = quat_from_euler_xyz(eulers[:, 0], eulers[:, 1], eulers[:, 2]).repeat_interleave(2, 0)
    probe_pose = probe.data.default_root_state[:, :7].clone()
    probe_pose[:, 0] = probe_ids.to(torch.float32) * 5.0
    probe_pose[:, 1] = 20.0
    probe_pose[:, 2] = 10.0
    probe_pose[:, 3:] = quaternions
    zero_probe_velocity = torch.zeros(probe.num_instances, 6, device=device)
    robot_pose = robot.data.default_root_state[:, :7].clone()
    robot_pose[:, 0] = torch.tensor([-20.0, -15.0], device=device)
    robot_pose[:, 1] = 0.0
    robot_pose[:, 2] = 10.0
    robot_pose[:, 3:] = quaternions[2]
    zero_robot_velocity = torch.zeros(robot.num_instances, 6, device=device)
    robot_joint_position = robot.data.default_joint_pos.clone()
    robot_joint_velocity = torch.zeros_like(robot_joint_position)
    base_ids, base_names = robot.find_bodies(["base_link"], preserve_order=True)
    if len(base_ids) != 1 or base_names != ["base_link"]:
        raise RuntimeError("真实资产的 base_link 必须唯一解析")
    base_id = base_ids[0]
    robot_masses = robot.root_physx_view.get_masses().to(device)
    total_mass = robot_masses.sum(dim=1)
    expected_mass = torch.full_like(total_mass, ASSET_BUNDLE_V2.total_mass_kg)
    torch.testing.assert_close(total_mass, expected_mass, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(
        probe.data.body_com_pos_b[:, 0],
        torch.tensor(probe_com_offset_m, device=device).expand(probe.num_instances, 3),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        probe.data.default_mass.to(device).reshape(-1),
        torch.full((probe.num_instances,), probe_mass_kg, device=device),
        rtol=0.0,
        atol=1e-6,
    )

    def reset_states():
        probe.reset()
        probe.write_root_link_pose_to_sim(probe_pose)
        probe.write_root_com_velocity_to_sim(zero_probe_velocity)
        robot.reset()
        robot.write_root_link_pose_to_sim(robot_pose)
        robot.write_root_com_velocity_to_sim(zero_robot_velocity)
        robot.write_joint_state_to_sim(robot_joint_position, robot_joint_velocity)
        robot.set_joint_position_target(robot_joint_position)
        robot.set_joint_velocity_target(robot_joint_velocity)
        sim.forward()
        probe.update(dt)
        robot.update(dt)

    def step():
        probe.write_data_to_sim()
        robot.write_data_to_sim()
        sim.step(render=False)
        probe.update(dt)
        robot.update(dt)

    def robot_momentum():
        return (robot_masses[:, :, None] * robot.data.body_com_lin_vel_w).sum(dim=1)

    reset_states()
    probe_before = probe.data.root_com_vel_w.clone()
    robot_before = robot_momentum().clone()
    actual_base_offset = (robot.data.body_com_pos_w[:, base_id] - robot.data.body_link_pos_w[:, base_id]).clone()
    robot_force = torch.tensor([[[0.0, 0.0, force_n]]], device=device)
    probe_force = robot_force.expand(len(orientations), 1, 3).contiguous()
    probe_torque = torch.zeros_like(probe_force)
    probe.instantaneous_wrench_composer.set_forces_and_torques(
        forces=probe_force,
        torques=probe_torque,
        positions=None,
        body_ids=[0],
        env_ids=forced_probe_ids,
        is_global=True,
    )
    robot.instantaneous_wrench_composer.set_forces_and_torques(
        forces=robot_force,
        torques=torch.zeros_like(robot_force),
        positions=None,
        body_ids=base_ids,
        env_ids=[1],
        is_global=True,
    )
    probe_local = probe.instantaneous_wrench_composer.composed_force_as_torch[:, 0].clone()
    probe_world = quat_apply(probe.data.root_link_quat_w, probe_local)
    robot_local = robot.instantaneous_wrench_composer.composed_force_as_torch.clone()
    robot_world = quat_apply(robot.data.body_link_quat_w.reshape(-1, 4), robot_local.reshape(-1, 3)).reshape_as(robot_local)
    expected_probe_world = torch.zeros_like(probe_world)
    expected_probe_world[forced_probe_ids, 2] = force_n
    expected_robot_world = torch.zeros_like(robot_world)
    expected_robot_world[1, base_id, 2] = force_n
    torch.testing.assert_close(probe_world, expected_probe_world, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(robot_world, expected_robot_world, rtol=0.0, atol=1e-5)
    for entity in (probe, robot):
        torch.testing.assert_close(
            entity.instantaneous_wrench_composer.composed_torque_as_torch,
            torch.zeros_like(entity.instantaneous_wrench_composer.composed_torque_as_torch),
            rtol=0.0,
            atol=0.0,
        )
    step()
    first_probe_velocity = probe.data.root_com_vel_w.clone()
    first_robot_momentum = robot_momentum().clone()
    for entity in (probe, robot):
        if entity.instantaneous_wrench_composer.active:
            raise RuntimeError("instantaneous wrench 在提交后没有清除")
    probe_delta = first_probe_velocity - probe_before
    probe_difference = probe_delta[1::2] - probe_delta[0::2]
    expected_delta_velocity = torch.tensor([0.0, 0.0, force_n * dt / probe_mass_kg], device=device)
    torch.testing.assert_close(
        probe_difference[:, :3], expected_delta_velocity.expand(len(orientations), 3), rtol=0.0, atol=2e-5
    )
    torch.testing.assert_close(probe_difference[:, 3:], torch.zeros_like(probe_difference[:, 3:]), rtol=0.0, atol=2e-5)
    robot_delta_momentum = first_robot_momentum - robot_before
    robot_difference = robot_delta_momentum[1] - robot_delta_momentum[0]
    expected_impulse = torch.tensor([0.0, 0.0, force_n * dt], device=device)
    torch.testing.assert_close(robot_difference, expected_impulse, rtol=0.01, atol=1e-3)

    # 下一步不再提交外力，两个单刚体的速度差应保持不变。
    step()
    second_probe_difference = probe.data.root_com_vel_w[1::2] - probe.data.root_com_vel_w[0::2]
    torch.testing.assert_close(second_probe_difference, first_probe_velocity[1::2] - first_probe_velocity[0::2], rtol=0.0, atol=2e-5)

    # 显式 COM 力矩提供正对照，证明引擎和读取能够检测额外角冲量。
    reset_states()
    control_torque = torch.tensor([[[0.0, 12.0, 0.0]]], device=device).expand(len(orientations), 1, 3).contiguous()
    probe.instantaneous_wrench_composer.set_forces_and_torques(
        forces=torch.zeros_like(probe_force),
        torques=control_torque,
        positions=None,
        body_ids=[0],
        env_ids=forced_probe_ids,
        is_global=True,
    )
    inertia_body = probe.root_physx_view.get_inertias().to(device).reshape(-1, 3, 3)[forced_probe_ids]
    torque_local = quat_apply_inverse(quaternions[forced_probe_ids], control_torque[:, 0])
    expected_control_angular = quat_apply(
        quaternions[forced_probe_ids], torch.linalg.solve(inertia_body, torque_local.unsqueeze(-1)).squeeze(-1) * dt
    )
    step()
    control_angular_difference = probe.data.root_com_ang_vel_w[1::2] - probe.data.root_com_ang_vel_w[0::2]
    torch.testing.assert_close(control_angular_difference, expected_control_angular, rtol=0.002, atol=2e-5)
    if not bool((control_angular_difference.norm(dim=1) > 0.10).all()):
        raise RuntimeError("已知额外 COM 力矩没有产生可辨别的角响应")

    origin_lever = -quat_apply(quaternions[forced_probe_ids], torch.tensor(probe_com_offset_m, device=device).expand(len(orientations), 3))
    origin_torque = torch.linalg.cross(origin_lever, probe_force[:, 0])
    incorrectly_origin_angular = quat_apply(
        quaternions[forced_probe_ids],
        torch.linalg.solve(
            inertia_body,
            quat_apply_inverse(quaternions[forced_probe_ids], origin_torque).unsqueeze(-1),
        ).squeeze(-1) * dt,
    )
    if float(incorrectly_origin_angular.norm(dim=1).max()) < 0.20:
        raise RuntimeError("偏置模型不能区分链接原点施力与 COM 施力")

    reports = []
    for index, (name, *_) in enumerate(orientations):
        reports.append({
            "orientation": name,
            "quaternion_wxyz": quaternions[2 * index].tolist(),
            "force_impulse_delta_linear_velocity_m_s": probe_difference[index, :3].tolist(),
            "force_delta_angular_velocity_rad_s": probe_difference[index, 3:].tolist(),
            "world_z_velocity_error_m_s": float((probe_difference[index, :3] - expected_delta_velocity).abs().max()),
            "com_torque_angular_velocity_error_rad_s": float(probe_difference[index, 3:].abs().max()),
            "hypothetical_link_origin_angular_velocity_rad_s": incorrectly_origin_angular[index].tolist(),
            "positive_torque_control_rad_s": control_angular_difference[index].tolist(),
            "positive_torque_control_error_rad_s": float((control_angular_difference[index] - expected_control_angular[index]).abs().max()),
        })
    source_paths = {
        "wrench_composer": Path(inspect.getfile(WrenchComposer)),
        "articulation": Path(inspect.getfile(Articulation)),
        "rigid_object": Path(inspect.getfile(RigidObject)),
        "verification_script": Path(__file__),
    }
    return {
        "schema_version": "RecoveryGuidancePhysicsVerificationV1",
        "passed": True,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": subprocess.list2cmdline([sys.executable, "-m", "recovery.tests.verify_guidance_physics", *sys.argv[1:]]),
        "device": str(device),
        "physics_dt_s": dt,
        "gravity_world_m_s2": [0.0, 0.0, -9.81],
        "source_sha256": source_sha256(),
        "source_files": {key: {"path": str(path.resolve()), "sha256": file_sha256(path)} for key, path in source_paths.items()},
        "submitted_force_world_n": [0.0, 0.0, force_n],
        "submitted_torque_world_nm": [0.0, 0.0, 0.0],
        "positions": None,
        "is_global": True,
        "single_rigid_body_probe": {
            "physical_model": "真实 PhysX 单刚体，无接触，显式 USD MassAPI 偏置与惯量",
            "mass_kg": probe_mass_kg,
            "com_offset_body_m": list(probe_com_offset_m),
            "inertia_diagonal_kg_m2": [0.20, 0.30, 0.40],
            "expected_delta_linear_velocity_m_s": expected_delta_velocity.tolist(),
            "position_none_com_application_verified": True,
            "instantaneous_force_cleared_after_one_step": True,
            "reports": reports,
        },
        "actual_robot": {
            "asset_bundle_sha256": asset_report.bundle_hash,
            "base_link_body_id": base_id,
            "base_link_unique": True,
            "body_names": robot.body_names,
            "actual_total_mass_kg": total_mass.tolist(),
            "base_link_world_com_minus_link_origin_m": actual_base_offset.tolist(),
            "base_link_com_offset_norm_m": actual_base_offset.norm(dim=1).tolist(),
            "expected_total_linear_impulse_n_s": expected_impulse.tolist(),
            "observed_total_linear_impulse_n_s": robot_difference.tolist(),
            "maximum_impulse_error_n_s": float((robot_difference - expected_impulse).abs().max()),
            "composed_force_only_at_base_link": True,
            "composed_force_world_max_abs_error_n": float((robot_world - expected_robot_world).abs().max()),
            "angular_response_scope": "整机包含关节、闭链和 armature；COM 作用点由偏置单刚体真实响应隔离验证，不要求整机角速度为零",
        },
        "tolerances": {
            "single_body_linear_velocity_atol_m_s": 2e-5,
            "single_body_zero_torque_angular_velocity_atol_rad_s": 2e-5,
            "single_body_positive_torque_rtol": 0.002,
            "actual_robot_impulse_atol_n_s": 0.001,
            "actual_robot_impulse_rtol": 0.01,
        },
    }


def main():
    arguments = _arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        launcher = AppLauncher(arguments)
        result = _run(arguments)
        with arguments.output.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"passed": True, "evidence": str(arguments.output.resolve())}, ensure_ascii=False), flush=True)
    except BaseException:
        # Kit 的关闭流程不能延迟真实异常及失败证据的输出。
        failure = traceback.format_exc()
        sys.__stderr__.write(failure)
        sys.__stderr__.flush()
        with arguments.output.open("w", encoding="utf-8") as stream:
            json.dump({"schema_version": "RecoveryGuidancePhysicsVerificationV1", "passed": False,
                       "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                       "command": subprocess.list2cmdline([sys.executable, "-m", "recovery.tests.verify_guidance_physics", *sys.argv[1:]]),
                       "traceback": failure}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os._exit(1)
    launcher.app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
