# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# 版权所有 (c) 2022-2025, Isaac Lab 项目开发者。
# All rights reserved.
# 保留所有权利。

# SPDX-License-Identifier: BSD-3-Clause
# SPDX 许可证标识符: BSD-3-Clause

from __future__ import annotations

import json
import csv
import os
import time
from pathlib import Path

import torch
from collections.abc import Sequence

from isaaclab_assets.robots.wheel_leg_robot import WHEELLEGROBOT_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: 

from isaaclab_tasks.direct.WheelLegRobot.APP.INS_task import INS

@configclass
class EventCfg:
    """Randomization events applied during training resets."""

    randomize_wheel_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("wheellegrobot", body_names=["L_lun", "R_lun"]),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

@configclass
class WheelLegRobotEnvCfg(DirectRLEnvCfg):
    # 环境配置
    num_envs = 4096
    num_obs = 186
    num_slice_obs = 31
    history_length = 5    
    israndcmd = True # 是否使用训练命令课程。当前课程最后阶段会加入前进/后退、左右转向和跳跃。
    decimation = 2  # 控制渲染间隔
    episode_length_s = 16.0  # 每个回合的最大时长（秒）
    action_scale_leg_angle = 0.30  # 腿部目标摆角缩放 [rad]
    action_scale_leg_length = 0.06  # 腿长目标相对命令的最大偏置 [m]
    action_scale_wheel_velocity = 30.0  # 轮子目标速度缩放 [rad/s]
    action_rate_limiter_enabled = True  # 限制相邻控制步的动作跳变，压制策略输出的高频满幅震荡。
    action_rate_limit_leg_angle = 0.08  # 腿角动作每个env step最大变化量，归一化action单位。
    action_rate_limit_leg_length = 0.25  # 腿长动作每个env step最大变化量，固定腿长阶段只用于诊断/兼容。
    action_rate_limit_wheel = 0.35  # 轮速动作每个env step最大变化量，保留前进/转向响应。
    leg_length_min = 0.06  # 腿长目标下限 [m]
    leg_length_max = 0.24  # 腿长目标上限 [m]。机构几何下0.30m不可达，先用保守可达范围。
    command_x_vel_range = (0.0, 0.0)  # 关闭自动课程时使用的x速度范围。
    command_yaw_vel_range = (0.0, 0.0)  # 关闭自动课程时使用的yaw速度范围。
    command_leg_length = 0.12  # 跳跃训练默认腿长命令
    command_leg_length_range = (0.08, 0.20)  # 训练时随机腿长命令，键盘播放可用完整0.06~0.24范围
    fixed_leg_length = True  # 跳跃键直接控制腿长目标，策略主要学习姿态/轮速配合。
    action_space = 6  # 动作空间的维度
    observation_space = num_obs  # 观测空间的维度
    state_space = 0  # 状态空间的维度
    dt = 1 / 100  # 时间步长 [s]    



    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # events
    events: EventCfg = EventCfg()
    # 机器人配置
    robot_cfg: ArticulationCfg = WHEELLEGROBOT_CFG.replace(prim_path="/World/envs/env_.*/robot")
    L_F_Joint1 = "L_F_Joint1"
    L_H_Joint1 = "L_H_Joint1"
    R_F_Joint1 = "R_F_Joint1"
    R_H_Joint1 = "R_H_Joint1"
    L_F_Joint2 = "L_F_Joint2"
    L_H_Joint2 = "L_H_Joint2"
    R_F_Joint2 = "R_F_Joint2"
    R_H_Joint2 = "R_H_Joint2"
    L_lun_Joint = "L_lun_Joint"
    R_lun_Joint = "R_lun_Joint"

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=1.5, replicate_physics=True)

    # 重置配置

    # 奖励权重
    rew_scale_alive = 0
    rew_scale_terminated = -100
    rew_scale_xyvel = 4.0
    rew_scale_tracking_ang_vel = 4.0
    rew_scale_zvel = -0.2
    rew_scale_torques = -0.0001
    rew_scale_action_rate = -0.006
    rew_scale_leg_length_reward = 2.0
    rew_scale_error_dof_reward = -0.1
    rew_scale_projected_gravity = 8.0
    rew_scale_ang_vel_xy = -0.05
    rew_scale_similar_legged = 0.7
    rew_scale_legged_theta = 0.8
    rew_scale_yaw_error = -1.5
    rew_scale_leg_length_action = -0.02
    rew_scale_leg_joint_vel = -0.002
    rew_scale_leg_joint_accel = -0.004
    rew_scale_stand_still = -0.1
    rew_scale_jump_height = 5.0
    rew_scale_jump_up_vel = 2.0
    jump_target_base_height = 0.35
    jump_command_period_s = 2.0
    jump_command_crouch_s = 0.12
    jump_command_extend_s = 0.30
    jump_command_recover_s = 0.70
    jump_crouch_leg_length = 0.06
    jump_extend_leg_length = 0.24
    jump_recover_leg_length = 0.12
    auto_curriculum = True
    curriculum_initial_stage = 0
    curriculum_reward_window = 4000
    curriculum_min_episodes_per_stage = 12000
    curriculum_reward_margin = 800.0
    curriculum_min_episode_length_fraction = 0.98
    curriculum_max_stage = 3
    curriculum_required_stable_windows = 5

    obs_scales = {
        "lin_vel": 1.5,
        "ang_vel": 0.5,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "height_measurements": 5.0,
        "yaw_error": 1.0,
    }

    # terrain 地形配置
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # 地面物体路径
        terrain_type="generator",  # 使用生成器生成地形
        terrain_generator=ROUGH_TERRAINS_CFG,  # 使用预定义的粗糙地形生成配置
        max_init_terrain_level=9,  # 最大初始地形难度
        collision_group=-1,  # 碰撞组
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 摩擦合成方式
            restitution_combine_mode="multiply",  # 反弹合成方式
            static_friction=1.0,  # 静态摩擦系数
            dynamic_friction=1.0,  # 动态摩擦系数
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",  # 视觉材质路径
            project_uvw=True,  # 是否应用UV映射
        ),
        debug_vis=False,  # 是否启用调试可视化
    )


class WheelLegRobotEnv(DirectRLEnv):
    cfg: WheelLegRobotEnvCfg

    def __init__(self, cfg: WheelLegRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.israndcmd = self.cfg.israndcmd
        self.dt = self.cfg.dt
        self.ins = INS(self.num_envs,self.dt,self.device)
        self.data_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # 获取机器人关节的索引
        self._L_F_Joint1_idx, _ = self.wheellegrobot.find_joints(self.cfg.L_F_Joint1)  # 左前腿关节1
        self._L_H_Joint1_idx, _ = self.wheellegrobot.find_joints(self.cfg.L_H_Joint1)  # 左后腿关节1
        self._R_F_Joint1_idx, _ = self.wheellegrobot.find_joints(self.cfg.R_F_Joint1)  # 右前腿关节1
        self._R_H_Joint1_idx, _ = self.wheellegrobot.find_joints(self.cfg.R_H_Joint1)  # 右后腿关节1
        self._L_F_Joint2_idx, _ = self.wheellegrobot.find_joints(self.cfg.L_F_Joint2)  # 左前腿关节2
        self._L_H_Joint2_idx, _ = self.wheellegrobot.find_joints(self.cfg.L_H_Joint2)  # 左后腿关节2
        self._R_F_Joint2_idx, _ = self.wheellegrobot.find_joints(self.cfg.R_F_Joint2)  # 右前腿关节2
        self._R_H_Joint2_idx, _ = self.wheellegrobot.find_joints(self.cfg.R_H_Joint2)  # 右后腿关节2
        self._L_lun_Joint_idx, _ = self.wheellegrobot.find_joints(self.cfg.L_lun_Joint)  # 左臀部关节
        self._R_lun_Joint_idx, _ = self.wheellegrobot.find_joints(self.cfg.R_lun_Joint)  # 右臀部关节
        self.base_link_idx, _ = self.wheellegrobot.find_bodies("base_link")
        self.action_scale_leg_angle = self.cfg.action_scale_leg_angle
        self.action_scale_leg_length = self.cfg.action_scale_leg_length
        self.action_scale_wheel_velocity = self.cfg.action_scale_wheel_velocity

        # 目标命令 (4维: 线速度x,线速度y,角速度yaw,腿长)
        self.commands = torch.tensor([0,0,0,self.cfg.command_leg_length]).repeat(self.num_envs,1).to(self.device) # (num_envs,4)
        self._keyboard_command_path = Path(__file__).resolve().parent / "APP" / "keyboard_commands.json"
        self._keyboard_command_timeout_s = 0.25
        self._keyboard_default_command = torch.tensor(
            [0.0, 0.0, 0.0, self.cfg.command_leg_length], dtype=self.commands.dtype, device=self.device
        )
        self._enable_external_keyboard = self.num_envs == 1

        # 初始化数据数组
        self.nan_obs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.history_length = self.cfg.history_length
        self.slice_obs_buf = torch.zeros((self.num_envs, self.cfg.num_slice_obs), device=self.device)
        self.history_obs_buf = torch.zeros((self.num_envs, self.cfg.history_length, self.cfg.num_slice_obs), device=self.device)
        self.obs_buf = torch.zeros((self.num_envs, self.cfg.num_obs), device=self.device)
        self.history_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.com_position = torch.zeros(self.num_envs,3, device=self.device)
        self.raw_actions = torch.zeros(self.num_envs,6, device=self.device)
        self.last_raw_actions = torch.zeros(self.num_envs,6, device=self.device)
        self.last_actions = torch.zeros(self.num_envs,6, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.actions, device=self.device)
        self._episode_reward_sums = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.curriculum_stage = self.cfg.curriculum_initial_stage
        self._curriculum_window_rewards = torch.zeros(self.cfg.curriculum_reward_window, dtype=torch.float, device=self.device)
        self._curriculum_window_lengths = torch.zeros(self.cfg.curriculum_reward_window, dtype=torch.float, device=self.device)
        self._curriculum_window_idx = 0
        self._curriculum_window_count = 0
        self._curriculum_stage_episode_count = 0
        self._curriculum_stage_baseline_reward: float | None = None
        self._curriculum_reward_mean = 0.0
        self._curriculum_length_mean = 0.0
        self._curriculum_reward_threshold = 0.0
        self._curriculum_stable_window_count = 0
        self._leg_joint_ids = (
            self._L_F_Joint1_idx
            + self._L_H_Joint1_idx
            + self._R_F_Joint1_idx
            + self._R_H_Joint1_idx
        )
        self._wheel_joint_ids = self._L_lun_Joint_idx + self._R_lun_Joint_idx
        self._leg_position_targets = torch.zeros((self.num_envs, 4), device=self.device)
        self._wheel_velocity_targets = torch.zeros((self.num_envs, 2), device=self.device)
        self._last_leg_position_targets = torch.zeros_like(self._leg_position_targets)
        self._last_wheel_velocity_targets = torch.zeros_like(self._wheel_velocity_targets)
        self._action_rate_limits = torch.tensor(
            [
                self.cfg.action_rate_limit_leg_angle,
                self.cfg.action_rate_limit_leg_angle,
                self.cfg.action_rate_limit_leg_length,
                self.cfg.action_rate_limit_leg_length,
                self.cfg.action_rate_limit_wheel,
                self.cfg.action_rate_limit_wheel,
            ],
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)
        self.x_filter = torch.zeros(self.num_envs,1,device=self.device)
        self.v_filter = torch.zeros(self.num_envs,1,device=self.device)
        self.target_yaw = torch.zeros(self.num_envs,device=self.device)
        # 初始化腿部的各个参数
        self.l5 = 0.088  # AE长度，单位为米
        self.l1 = 0.105  # 腿部长度，单位为米
        self.l2 = 0.16   # 腿部长度，单位为米
        self.l3 = 0.16   # 腿部长度，单位为米
        self.l4 = 0.105  # 腿部长度，单位为米
        # 初始化腿部的各个参数
        self.csv_filename = "vmcr_data.csv"
        file_exists = os.path.isfile(self.csv_filename)
        self.csv_file = open(self.csv_filename, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            self.csv_writer.writerow(["time",'theta', 'v_filter', 'myPithR'])
            self.csv_file.flush()

    def _update_commands_from_keyboard_file(self):
        """Read commands written by the standalone keyboard helper."""
        if not self._enable_external_keyboard:
            return

        try:
            with self._keyboard_command_path.open("r", encoding="utf-8") as command_file:
                payload = json.load(command_file)
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return

        commands = payload.get("commands")
        timestamp = payload.get("timestamp")
        if not isinstance(commands, list) or len(commands) != 4:
            return
        if not isinstance(timestamp, (int, float)):
            return

        if time.time() - float(timestamp) > self._keyboard_command_timeout_s:
            command_tensor = self._keyboard_default_command
        else:
            try:
                command_tensor = torch.tensor(commands, dtype=self.commands.dtype, device=self.device)
            except (TypeError, ValueError):
                return

        command_tensor[3] = torch.clamp(command_tensor[3], self.cfg.leg_length_min, self.cfg.leg_length_max)
        self.commands[:] = command_tensor.unsqueeze(0).repeat(self.num_envs, 1)


    def _setup_scene(self):
        # 初始化机器人
        self.wheellegrobot = Articulation(self.cfg.robot_cfg)
        # 添加地面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # 克隆并复制环境
        self.scene.clone_environments(copy_from_source=False)
        # 将机器人添加到场景中
        self.scene.articulations["wheellegrobot"] = self.wheellegrobot
        # 添加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # 添加地形
        # self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        # self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 动作含义:
        # 0: 左腿目标摆角, 1: 右腿目标摆角,
        # 2: 左腿目标腿长偏置, 3: 右腿目标腿长偏置,
        # 4: 左轮目标速度, 5: 右轮目标速度。
        self.last_raw_actions[:] = self.raw_actions[:]
        self.last_actions[:] = self.actions[:]
        self._last_leg_position_targets[:] = self._leg_position_targets[:]
        self._last_wheel_velocity_targets[:] = self._wheel_velocity_targets[:]
        self.raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        if self.cfg.action_rate_limiter_enabled:
            action_delta = torch.clamp(
                self.raw_actions - self.actions,
                min=-self._action_rate_limits,
                max=self._action_rate_limits,
            )
            self.actions[:] = torch.clamp(self.actions + action_delta, -1.0, 1.0)
        else:
            self.actions[:] = self.raw_actions

        theta_L_target = self.action_scale_leg_angle * self.actions[:, 0].unsqueeze(1)
        theta_R_target = self.action_scale_leg_angle * self.actions[:, 1].unsqueeze(1)
        commanded_leg_length = torch.clamp(
            self.commands[:, 3].unsqueeze(1),
            self.cfg.leg_length_min,
            self.cfg.leg_length_max,
        )
        if self.cfg.fixed_leg_length:
            leg_length_L_target = commanded_leg_length
            leg_length_R_target = commanded_leg_length
        else:
            leg_length_L_target = torch.clamp(
                commanded_leg_length + self.action_scale_leg_length * self.actions[:, 2].unsqueeze(1),
                self.cfg.leg_length_min,
                self.cfg.leg_length_max,
            )
            leg_length_R_target = torch.clamp(
                commanded_leg_length + self.action_scale_leg_length * self.actions[:, 3].unsqueeze(1),
                self.cfg.leg_length_min,
                self.cfg.leg_length_max,
            )

        phi1_L_target, phi4_L_target = self.inv_dofpos_from_length_angle(leg_length_L_target, theta_L_target)
        phi1_R_target, phi4_R_target = self.inv_dofpos_from_length_angle(leg_length_R_target, theta_R_target)

        self._leg_position_targets[:, 0] = (phi1_L_target - 3.0 * torch.pi / 4.0).squeeze(1)
        self._leg_position_targets[:, 1] = (phi4_L_target - torch.pi / 4.0).squeeze(1)
        self._leg_position_targets[:, 2] = (phi4_R_target - torch.pi / 4.0).squeeze(1)
        self._leg_position_targets[:, 3] = (phi1_R_target - 3.0 * torch.pi / 4.0).squeeze(1)
        self._wheel_velocity_targets[:] = self.action_scale_wheel_velocity * self.actions[:, 4:6]

    def _apply_action(self) -> None:
        # 腿部使用位置目标，轮子使用速度目标。
        self.wheellegrobot.set_joint_position_target(self._leg_position_targets, joint_ids=self._leg_joint_ids)
        self.wheellegrobot.set_joint_velocity_target(self._wheel_velocity_targets, joint_ids=self._wheel_joint_ids)
        # print(self.commands[0,:])

    def _apply_training_jump_command_profile(self) -> None:
        if not self.israndcmd or self._enable_external_keyboard:
            return

        phase = torch.remainder(self.data_time, self.cfg.jump_command_period_s)
        leg_length = torch.full((self.num_envs,), self.cfg.jump_recover_leg_length, device=self.device)
        leg_length = torch.where(
            phase < self.cfg.jump_command_crouch_s,
            torch.full_like(leg_length, self.cfg.jump_crouch_leg_length),
            leg_length,
        )
        leg_length = torch.where(
            (phase >= self.cfg.jump_command_crouch_s) & (phase < self.cfg.jump_command_extend_s),
            torch.full_like(leg_length, self.cfg.jump_extend_leg_length),
            leg_length,
        )
        leg_length = torch.where(
            (phase >= self.cfg.jump_command_extend_s) & (phase < self.cfg.jump_command_recover_s),
            torch.full_like(leg_length, self.cfg.jump_recover_leg_length),
            leg_length,
        )
        self.commands[:, 3] = leg_length

    def _get_curriculum_x_vel_range(self) -> tuple[float, float]:
        if not self.cfg.auto_curriculum:
            return self.cfg.command_x_vel_range
        if self.curriculum_stage <= 0:
            return 0.0, 0.0
        if self.curriculum_stage == 1:
            return -0.4, 0.4
        return -0.8, 0.8

    def _get_curriculum_yaw_vel_range(self) -> tuple[float, float]:
        if not self.cfg.auto_curriculum:
            return self.cfg.command_yaw_vel_range
        if self.curriculum_stage < 3:
            return 0.0, 0.0
        return -1.0, 1.0

    def _record_curriculum_episodes(self, env_ids: torch.Tensor) -> None:
        if not self.cfg.auto_curriculum or not hasattr(self, "_episode_reward_sums"):
            return

        completed_mask = self.episode_length_buf[env_ids] > 0
        if not torch.any(completed_mask):
            return

        completed_ids = env_ids[completed_mask]
        rewards = self._episode_reward_sums[completed_ids].detach()
        lengths = self.episode_length_buf[completed_ids].float().detach()
        num_rewards = rewards.numel()
        if num_rewards == 0:
            return

        if num_rewards > self.cfg.curriculum_reward_window:
            rewards = rewards[-self.cfg.curriculum_reward_window :]
            lengths = lengths[-self.cfg.curriculum_reward_window :]
            num_rewards = rewards.numel()

        window_ids = (
            torch.arange(num_rewards, device=self.device, dtype=torch.long) + self._curriculum_window_idx
        ) % self.cfg.curriculum_reward_window
        self._curriculum_window_rewards[window_ids] = rewards
        self._curriculum_window_lengths[window_ids] = lengths
        self._curriculum_window_idx = int((self._curriculum_window_idx + num_rewards) % self.cfg.curriculum_reward_window)
        self._curriculum_window_count = min(
            self.cfg.curriculum_reward_window, self._curriculum_window_count + num_rewards
        )
        self._curriculum_stage_episode_count += int(num_rewards)

        valid_rewards = self._curriculum_window_rewards[: self._curriculum_window_count]
        valid_lengths = self._curriculum_window_lengths[: self._curriculum_window_count]
        self._curriculum_reward_mean = float(torch.mean(valid_rewards).item())
        self._curriculum_length_mean = float(torch.mean(valid_lengths).item())

        if (
            self._curriculum_stage_baseline_reward is None
            and self._curriculum_window_count >= self.cfg.curriculum_reward_window
            and self._curriculum_stage_episode_count >= self.cfg.curriculum_min_episodes_per_stage
        ):
            self._curriculum_stage_baseline_reward = self._curriculum_reward_mean
            self._curriculum_reward_threshold = (
                self._curriculum_stage_baseline_reward + self.cfg.curriculum_reward_margin
            )

        length_threshold = self.max_episode_length * self.cfg.curriculum_min_episode_length_fraction
        window_passed = (
            self._curriculum_stage_baseline_reward is not None
            and self.curriculum_stage < self.cfg.curriculum_max_stage
            and self._curriculum_stage_episode_count >= self.cfg.curriculum_min_episodes_per_stage
            and self._curriculum_reward_mean >= self._curriculum_reward_threshold
            and self._curriculum_length_mean >= length_threshold
        )
        if window_passed:
            self._curriculum_stable_window_count += 1
        else:
            self._curriculum_stable_window_count = 0

        can_advance = self._curriculum_stable_window_count >= self.cfg.curriculum_required_stable_windows
        if can_advance:
            self.curriculum_stage += 1
            self._curriculum_window_rewards.zero_()
            self._curriculum_window_lengths.zero_()
            self._curriculum_window_idx = 0
            self._curriculum_window_count = 0
            self._curriculum_stage_episode_count = 0
            self._curriculum_stage_baseline_reward = None
            self._curriculum_reward_threshold = 0.0
            self._curriculum_stable_window_count = 0

    def _get_observations(self) -> dict:
        # 更新仿真时间
        self.data_time += self.dt
        if self.data_time[0] > 3 and not self.israndcmd and not self._enable_external_keyboard:
            self.commands = torch.tensor([0,0,0,self.cfg.command_leg_length]).repeat(self.num_envs,1).to(self.device)
        self._apply_training_jump_command_profile()
        # 更新键盘输入指令
        # self.commands = self.listener.commands
        # 获取角速度    
        self._update_commands_from_keyboard_file()
        self.Gyro = self.wheellegrobot.data.root_ang_vel_b
        # 机器人的四元数
        q = self.wheellegrobot.data.body_quat_w[:,self.base_link_idx,:]
        self.ins.update(q)
        # 计算 Pitch  形状为(env_num,1)
        Pitch = self.ins.Pitch
        Yaw = self.ins.Yaw
        self.target_yaw += self.commands[:,2] * self.dt   
        yaw_error = (Yaw - self.target_yaw.unsqueeze(1))  # 适当缩放
        xvel_error = (self.wheellegrobot.data.root_lin_vel_b[:,0] - self.commands[:,0]).unsqueeze(1)

        # print("yaw::",self.ins.YawTotalAngle[0],"  target_yaw::",self.target_yaw[0])
        # print("xvel::",self.wheellegrobot.data.root_lin_vel_b[0],"  target_vel::",self.commands[0])
        #phi值 形状为(env_num,1)
        vmcr_phi1 = torch.pi / 2.0 + (self.wheellegrobot.data.joint_pos[:,self.wheellegrobot.find_joints(self.cfg.R_H_Joint1)[0]] + torch.pi / 4)
        vmcr_phi4 = torch.pi / 2.0 + (self.wheellegrobot.data.joint_pos[:,self.wheellegrobot.find_joints(self.cfg.R_F_Joint1)[0]] - torch.pi / 4)
        vmcl_phi1 = torch.pi / 2.0 + (self.wheellegrobot.data.joint_pos[:,self.wheellegrobot.find_joints(self.cfg.L_F_Joint1)[0]] + torch.pi / 4)
        vmcl_phi4 = torch.pi / 2.0 + (self.wheellegrobot.data.joint_pos[:,self.wheellegrobot.find_joints(self.cfg.L_H_Joint1)[0]] - torch.pi / 4)
        
        self.vmcr_phi1 = vmcr_phi1.squeeze(1)
        self.vmcr_phi4 = vmcr_phi4.squeeze(1)
        self.vmcl_phi1 = vmcl_phi1.squeeze(1)
        self.vmcl_phi4 = vmcl_phi4.squeeze(1)

        Right_VmcData = self.VMC_calc_Leg(Pitch,vmcr_phi1,vmcr_phi4,isright=True)
        Left_VmcData = self.VMC_calc_Leg(Pitch,vmcl_phi1,vmcl_phi4,isright=False)
        # 右腿
        self.theta_R = Right_VmcData["theta"]
        self.leg_length_R = Right_VmcData["L0"]
        default_dof_vmcr_phi1,default_dof_vmcr_phi4 = self.inv_dofpos(self.leg_length_R)
        # 右腿关节位置误差
        error_dof_vmcr_phi1 = vmcr_phi1- default_dof_vmcr_phi1
        error_dof_vmcr_phi4 = vmcr_phi4- default_dof_vmcr_phi4

        # 右腿
        self.theta_L = Left_VmcData["theta"]
        self.leg_length_L = Left_VmcData["L0"]
        default_dof_vmcl_phi1,default_dof_vmcl_phi4 = self.inv_dofpos(self.leg_length_L)
        # 关节位置误差
        error_dof_vmcl_phi1 = vmcl_phi1- default_dof_vmcl_phi1
        error_dof_vmcl_phi4 = vmcl_phi4- default_dof_vmcl_phi4

        self.estimate_vel_and_pos(self.wheellegrobot.data.joint_vel[:,self._R_lun_Joint_idx], self.wheellegrobot.data.joint_vel[:,self._L_lun_Joint_idx])

        # print("速度：：",self.wheellegrobot.data.root_lin_vel_b[0])
        # 获取观测值(确保维数相同)
        self.slice_obs_buf = torch.cat(
            (
            # 机器人的线速度
            self.wheellegrobot.data.root_lin_vel_b * self.cfg.obs_scales["lin_vel"],         # 索引 0: 机器人的线速度 (形状：[env_num,3])
            self.Gyro * self.cfg.obs_scales["ang_vel"],                                     # 索引 3: 机器人的角速度 (形状：[env_num,3])
            # 原有的观测分量
            self.wheellegrobot.data.projected_gravity_b,    # 索引 6: 重力在机器人坐标系投影 (形状：[env_num,3])

            # 来自命令的输入
            self.commands[:,0].unsqueeze(1).to(self.device) * self.cfg.obs_scales["lin_vel"],  # 索引 9: 线速度x (来自命令) (形状：[env_num, 1])
            self.commands[:,1].unsqueeze(1).to(self.device) * self.cfg.obs_scales["ang_vel"],  # 索引 10: 线速度y (来自命令) (形状：[env_num, 1])
            self.commands[:,2].unsqueeze(1).to(self.device) * self.cfg.obs_scales["dof_pos"],  # 索引 11: 角速度yaw (来自命令) (形状：[env_num, 1])
            self.commands[:,3].unsqueeze(1).to(self.device) * self.cfg.obs_scales["height_measurements"],  # 索引 12: 腿长L0 (来自命令) (形状：[env_num, 1])

            # 关节位置误差
            error_dof_vmcl_phi1.to(self.device) * self.cfg.obs_scales["dof_pos"],           # 索引 13: 左phi1关节误差 (形状：[env_num, 1])
            error_dof_vmcl_phi4.to(self.device) * self.cfg.obs_scales["dof_pos"],           # 索引 14: 左phi4关节误差 (形状：[env_num, 1])
            error_dof_vmcr_phi1.to(self.device) * self.cfg.obs_scales["dof_pos"],           # 索引 15: 右phi1关节误差 (形状：[env_num, 1])
            error_dof_vmcr_phi4.to(self.device) * self.cfg.obs_scales["dof_pos"],           # 索引 16: 右phi4关节误差 (形状：[env_num, 1])

            # 关节速度
            self.wheellegrobot.data.joint_vel[:,self._L_F_Joint1_idx] * self.cfg.obs_scales["dof_vel"],  # 索引 17: L_F大腿关节速度 (形状：[env_num, 1])
            self.wheellegrobot.data.joint_vel[:,self._L_H_Joint1_idx] * self.cfg.obs_scales["dof_vel"],  # 索引 18: L_H大腿关节速度 (形状：[env_num, 1])
            self.wheellegrobot.data.joint_vel[:,self._R_F_Joint1_idx] * self.cfg.obs_scales["dof_vel"],  # 索引 19: R_F大腿关节速度 (形状：[env_num, 1])
            self.wheellegrobot.data.joint_vel[:,self._R_H_Joint1_idx] * self.cfg.obs_scales["dof_vel"],  # 索引 20: R_H大腿关节速度 (形状：[env_num, 1])
            self.wheellegrobot.data.joint_vel[:,self._L_lun_Joint_idx] * self.cfg.obs_scales["dof_vel"],  # 索引 21: L轮子速度 (形状：[env_num, 1])
            self.wheellegrobot.data.joint_vel[:,self._R_lun_Joint_idx] * self.cfg.obs_scales["dof_vel"],  # 索引 22: R轮子速度 (形状：[env_num, 1])
            
            # 动作（关节控制信号）
            self.actions.to(self.device),                   # 索引 23: 动作控制信号 (形状：[env_num, 6])

            # x方向线速度误差
            xvel_error * self.cfg.obs_scales["lin_vel"],   # 索引 29: x线速度误差 (形状：[env_num, 1])
            # 航向角误差
            yaw_error * self.cfg.obs_scales["yaw_error"],   # 索引 30: 航向角误差 (形状：[env_num, 1])
            # 总共31维度

            ),
            dim=-1,
        )
        # 更新历史数据
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:,0] = self.wheellegrobot.data.joint_vel[:,self._L_F_Joint1_idx].squeeze()
        self.last_dof_vel[:,1] = self.wheellegrobot.data.joint_vel[:,self._L_H_Joint1_idx].squeeze()
        self.last_dof_vel[:,2] = self.wheellegrobot.data.joint_vel[:,self._R_F_Joint1_idx].squeeze()
        self.last_dof_vel[:,3] = self.wheellegrobot.data.joint_vel[:,self._R_H_Joint1_idx].squeeze()
        self.last_dof_vel[:,4] = self.wheellegrobot.data.joint_vel[:,self._L_lun_Joint_idx].squeeze()
        self.last_dof_vel[:,5] = self.wheellegrobot.data.joint_vel[:,self._R_lun_Joint_idx].squeeze()

        self.nan_obs = torch.isnan(self.slice_obs_buf).any(dim=-1)  # 检查每个环境的观测值中是否有 NaN，有Nan的环境为True
        # 将 NaN 值替换为零
        # Expand nan_obs to match the shape of obs
        expanded_nan_obs = self.nan_obs.unsqueeze(1).expand_as(self.slice_obs_buf)  # (num_envs, obs_dim)
        self.slice_obs_buf = torch.where(expanded_nan_obs, torch.zeros_like(self.slice_obs_buf), self.slice_obs_buf)

        self.obs_buf = torch.cat([self.history_obs_buf, self.slice_obs_buf.unsqueeze(1)], dim=1).view(self.num_envs, -1)
        # Update history buffer
        if self.history_length > 1:
            self.history_obs_buf[:, :-1, :] = self.history_obs_buf[:, 1:, :].clone() # 移位操作
        self.history_obs_buf[:, -1, :] = self.slice_obs_buf 

        # 写入数据到CSV
        self.csv_writer.writerow([
            self.data_time[0].detach().item(),      # 直接提取CUDA张量的数值
            self.theta_R[0][0].detach().item(),
            self.slice_obs_buf[0][0].detach().item(),
            Pitch[0][0].detach().item()
        ])
        self.csv_file.flush()


        observations = {"policy": self.obs_buf}
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        # 计算奖励
        (
            total_reward,
            rew_alive,
            rew_termination,
            rew_xyvel,
            rew_tracking_ang_vel,
            rew_zvel,
            rew_torques,
            rew_action_rate,
            rew_leg_length_action,
            rew_leg_joint_vel,
            rew_leg_joint_accel,
            rew_stand_still,
            rew_jump_height,
            rew_jump_up_vel,
            rew_leg_length_reward,
            rew_error_dof,
            rew_projected_gravity,
            rew_ang_vel_xy,
            rew_similar_legged,
            rew_legged_theta,
            rew_yaw_error,
        ) = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_xyvel,
            self.cfg.rew_scale_tracking_ang_vel,
            self.cfg.rew_scale_zvel,
            self.cfg.rew_scale_torques,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_leg_length_reward,
            self.cfg.rew_scale_error_dof_reward,
            self.cfg.rew_scale_projected_gravity,
            self.cfg.rew_scale_ang_vel_xy,
            self.cfg.rew_scale_similar_legged,
            self.cfg.rew_scale_legged_theta,
            self.cfg.rew_scale_yaw_error,
            self.cfg.rew_scale_leg_length_action,
            self.cfg.rew_scale_leg_joint_vel,
            self.cfg.rew_scale_leg_joint_accel,
            self.cfg.rew_scale_stand_still,
            self.cfg.rew_scale_jump_height,
            self.cfg.rew_scale_jump_up_vel,
            self.cfg.jump_target_base_height,
            self.leg_length_R,
            self.leg_length_L,
            self.last_actions,
            self.actions,
            self.slice_obs_buf,
            self.vmcl_phi1,
            self.vmcl_phi4,
            self.vmcr_phi1,
            self.vmcr_phi4,
            self.theta_R,
            self.theta_L,
            self.ins.YawTotalAngle,
            self.target_yaw,
            self.wheellegrobot.data.root_lin_vel_b,
            self.wheellegrobot.data.root_lin_vel_w,
            self.wheellegrobot.data.root_pos_w[:, 2],
            self.Gyro,
            self.commands,
            torch.cat(
                [
                    self.wheellegrobot.data.joint_vel[:, self._L_F_Joint1_idx],
                    self.wheellegrobot.data.joint_vel[:, self._L_H_Joint1_idx],
                    self.wheellegrobot.data.joint_vel[:, self._R_F_Joint1_idx],
                    self.wheellegrobot.data.joint_vel[:, self._R_H_Joint1_idx],
                ],
                dim=1,
            ),
            self.last_dof_vel[:, 0:4],
            torch.cat(
                [
                    self.wheellegrobot.data.joint_vel[:, self._L_lun_Joint_idx],
                    self.wheellegrobot.data.joint_vel[:, self._R_lun_Joint_idx],
                ],
                dim=1,
            ),
            self.reset_terminated,
        )
        self._episode_reward_sums += total_reward
        self.extras["log"] = {
            "Curriculum/stage": float(self.curriculum_stage),
            "Curriculum/reward_mean_episodes": self._curriculum_reward_mean,
            "Curriculum/reward_threshold": self._curriculum_reward_threshold,
            "Curriculum/episode_length_mean": self._curriculum_length_mean,
            "Curriculum/stable_window_count": float(self._curriculum_stable_window_count),
            "Reward/total": torch.mean(total_reward),
            "Reward/alive": torch.mean(rew_alive),
            "Reward/termination": torch.mean(rew_termination),
            "Reward/x_vel_tracking": torch.mean(rew_xyvel),
            "Reward/yaw_vel_tracking": torch.mean(rew_tracking_ang_vel),
            "Reward/z_vel_penalty": torch.mean(rew_zvel),
            "Reward/action_magnitude": torch.mean(rew_torques),
            "Reward/action_rate": torch.mean(rew_action_rate),
            "Reward/leg_length_action": torch.mean(rew_leg_length_action),
            "Reward/leg_joint_vel": torch.mean(rew_leg_joint_vel),
            "Reward/leg_joint_accel": torch.mean(rew_leg_joint_accel),
            "Reward/stand_still": torch.mean(rew_stand_still),
            "Reward/jump_height": torch.mean(rew_jump_height),
            "Reward/jump_up_vel": torch.mean(rew_jump_up_vel),
            "Reward/leg_length_tracking": torch.mean(rew_leg_length_reward),
            "Reward/dof_error": torch.mean(rew_error_dof),
            "Reward/upright": torch.mean(rew_projected_gravity),
            "Reward/ang_vel_xy": torch.mean(rew_ang_vel_xy),
            "Reward/leg_symmetry": torch.mean(rew_similar_legged),
            "Reward/leg_theta": torch.mean(rew_legged_theta),
            "Reward/yaw_error": torch.mean(rew_yaw_error),
            "Metrics/command_x_vel": torch.mean(self.commands[:, 0]),
            "Metrics/command_x_vel_abs": torch.mean(torch.abs(self.commands[:, 0])),
            "Metrics/actual_x_vel": torch.mean(self.wheellegrobot.data.root_lin_vel_b[:, 0]),
            "Metrics/actual_x_vel_abs": torch.mean(torch.abs(self.wheellegrobot.data.root_lin_vel_b[:, 0])),
            "Metrics/command_yaw_vel": torch.mean(self.commands[:, 2]),
            "Metrics/command_yaw_vel_abs": torch.mean(torch.abs(self.commands[:, 2])),
            "Metrics/actual_yaw_vel": torch.mean(self.Gyro[:, 2]),
            "Metrics/actual_yaw_vel_abs": torch.mean(torch.abs(self.Gyro[:, 2])),
            "Metrics/command_leg_length": torch.mean(self.commands[:, 3]),
            "Metrics/left_leg_length": torch.mean(self.leg_length_L),
            "Metrics/right_leg_length": torch.mean(self.leg_length_R),
            "Metrics/base_height": torch.mean(self.wheellegrobot.data.root_pos_w[:, 2]),
        }
        self._debug_control_log = self._get_debug_control_log()
        self.extras["log"].update(self._debug_control_log)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 判断是否超时或超出边界
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        pitch = self.ins.Pitch.squeeze(1)
        roll = self.ins.Roll.squeeze(1)
        YawTotalAngle = self.ins.YawTotalAngle.squeeze(1)
        # 获取重心位置
        com_position = self.wheellegrobot.data.body_pos_w[:, self.base_link_idx, :].squeeze(1)  # 重心位置
        com_position_z = com_position[:, 2]

        self.com_position = com_position
        # print(com_position_z[0])
        # 检查 NaN 或 Inf 值
        nan_or_inf_condition = torch.isnan(pitch) | torch.isinf(pitch) | \
                            torch.isnan(roll) | torch.isinf(roll) | \
                            torch.isnan(YawTotalAngle) | torch.isinf(YawTotalAngle) | \
                            torch.isnan(com_position_z) | torch.isinf(com_position_z)
        
        # 俯仰角、滚转角、偏航角范围检查
        angle_condition = torch.logical_or(pitch < -0.35, pitch > 0.35)
        angle_condition = torch.logical_or(angle_condition, roll < -0.35)
        angle_condition = torch.logical_or(angle_condition, roll > 0.35)


        #对theta角范围检查
        # theta_condition = torch.logical_or(angle_condition,self.theta_L.squeeze(1) < -0.2)
        # theta_condition = torch.logical_or(theta_condition,self.theta_L.squeeze(1) > 0.2)
        # theta_condition = torch.logical_or(theta_condition,self.theta_R.squeeze(1) < -0.2)
        # theta_condition = torch.logical_or(theta_condition,self.theta_R.squeeze(1) > 0.2)
        # # 重心位置检查：超出合适范围（0.07 到 2.5）会触发终止
        # com_condition = torch.logical_or(com_position_z < 0.9, com_position_z > 3)

        # # 合并所有终止条件：NaN 检查、俯仰角、滚转角、偏航角、重心位置    
        # out_of_bounds = torch.logical_or(angle_condition, com_condition)
        # invalid_values = torch.logical_or(theta_condition, nan_or_inf_condition)
        invalid_values = torch.logical_or(angle_condition, nan_or_inf_condition)
        invalid_values = torch.logical_or(invalid_values, self.nan_obs)
        # 返回检测结果
        return invalid_values, time_out



    def _reset_idx(self, env_ids: Sequence[int] | None):
        # 重置指定环境
        if env_ids is None:
            env_ids = self.wheellegrobot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        self._record_curriculum_episodes(env_ids)
        super()._reset_idx(env_ids)
        '''
        随机指令
        ''' 
        if self.israndcmd and not self._enable_external_keyboard:
            min_x_vel, max_x_vel = self._get_curriculum_x_vel_range()
            min_yaw_vel, max_yaw_vel = self._get_curriculum_yaw_vel_range()
            self.commands[env_ids,0] = torch.rand(len(env_ids), device=self.device) * (max_x_vel - min_x_vel) + min_x_vel
            self.commands[env_ids,1] = torch.zeros(len(env_ids), device=self.device)
            self.commands[env_ids,2] = (
                torch.rand(len(env_ids), device=self.device) * (max_yaw_vel - min_yaw_vel) + min_yaw_vel
            )
            self.commands[env_ids,3] = self.cfg.command_leg_length
            # else:
            #     num_list = [-1,1]
            #     # 生成随机索引
            #     rand_num = torch.randint(low=0, high=len(num_list), size=(1,)).item()
            #     self.commands[:,0] = rand_num * torch.ones(self.num_envs) * 2  # 后期阶段：高速指令
            #     self.commands[:,1] = rand_num * torch.ones(self.num_envs) * 2
            #     self.commands[:,2] = rand_num * torch.ones(self.num_envs) * 2
            #     self.commands[:,3] = torch.rand(self.num_envs) * 0.09 + 0.06

        # if self.israndcmd:
        #     # 每个环境独立生成指令
        #     rand_nums = torch.randint(0, 4, (self.num_envs,), device=self.device)
        #     leg_lengths = torch.rand(self.num_envs, device=self.device)*0.06+0.06
            
        #     # 创建指令矩阵
        #     command_matrix = torch.tensor([
        #         [0,0,2,0],   # 左转
        #         [0,0,-2,0],  # 右转
        #         [2,0,0,0],   # 前进
        #         [-2,0,0,0]   # 后退
        #     ], device=self.device)
            
        #     # 批量选择指令
        #     self.commands = command_matrix[rand_nums]
        #     self.commands[:,3] = leg_lengths  # 设置独立腿长
        # 重置关节位置和速度
        # 获取需要重置的环境数量
        '''
        随机关节
        '''    
        joint_pos = self.wheellegrobot.data.default_joint_pos[env_ids]
        joint_vel = self.wheellegrobot.data.default_joint_vel[env_ids]
        # 重置根状态
        default_root_state = self.wheellegrobot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # 仅重置指定环境的历史索引
        self.data_time[env_ids] = 0.0  # 重置时间
        self.history_idx[env_ids] = 0  # 使用标量广播机制
        self.last_actions[env_ids] = torch.zeros(len(env_ids), 6, device=self.device)
        self.raw_actions[env_ids] = torch.zeros(len(env_ids), 6, device=self.device)
        self.last_raw_actions[env_ids] = torch.zeros(len(env_ids), 6, device=self.device)
        if hasattr(self, "_episode_reward_sums"):
            self._episode_reward_sums[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0
        self.x_filter[env_ids] = torch.zeros(len(env_ids), 1, device=self.device)
        self.v_filter[env_ids] = torch.zeros(len(env_ids), 1, device=self.device)
        self.target_yaw[env_ids] = torch.zeros(len(env_ids), device=self.device)
        self.ins.reset(env_ids)

        # 将重置后的状态写入仿真
        self.wheellegrobot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.wheellegrobot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.wheellegrobot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.wheellegrobot.set_joint_position_target(
            joint_pos[:, self._leg_joint_ids], joint_ids=self._leg_joint_ids, env_ids=env_ids
        )
        self.wheellegrobot.set_joint_velocity_target(
            torch.zeros((len(env_ids), 2), device=self.device), joint_ids=self._wheel_joint_ids, env_ids=env_ids
        )
        self._leg_position_targets[env_ids] = joint_pos[:, self._leg_joint_ids]
        self._last_leg_position_targets[env_ids] = joint_pos[:, self._leg_joint_ids]
        self._wheel_velocity_targets[env_ids] = torch.zeros((len(env_ids), 2), device=self.device)
        self._last_wheel_velocity_targets[env_ids] = torch.zeros((len(env_ids), 2), device=self.device)

    def _get_debug_control_log(self) -> dict[str, torch.Tensor]:
        leg_joint_pos = self.wheellegrobot.data.joint_pos[:, self._leg_joint_ids]
        leg_joint_vel = self.wheellegrobot.data.joint_vel[:, self._leg_joint_ids]
        wheel_joint_vel = self.wheellegrobot.data.joint_vel[:, self._wheel_joint_ids]

        raw_action_delta = torch.abs(self.raw_actions - self.last_raw_actions)
        action_delta = torch.abs(self.actions - self.last_actions)
        action_limiter_error = torch.abs(self.raw_actions - self.actions)
        leg_target_delta = torch.abs(self._leg_position_targets - self._last_leg_position_targets)
        wheel_target_delta = torch.abs(self._wheel_velocity_targets - self._last_wheel_velocity_targets)
        leg_target_error = torch.abs(self._leg_position_targets - leg_joint_pos)
        wheel_target_error = torch.abs(self._wheel_velocity_targets - wheel_joint_vel)

        stand_mask = (
            (torch.abs(self.commands[:, 0]) < 0.05)
            & (torch.abs(self.commands[:, 2]) < 0.05)
            & (torch.abs(self.commands[:, 3] - self.cfg.command_leg_length) < 0.02)
        )
        if torch.any(stand_mask):
            stand_raw_action_delta = torch.mean(raw_action_delta[stand_mask])
            stand_action_delta = torch.mean(action_delta[stand_mask])
            stand_action_limiter_error = torch.mean(action_limiter_error[stand_mask])
            stand_leg_target_delta = torch.mean(leg_target_delta[stand_mask])
            stand_leg_target_error = torch.mean(leg_target_error[stand_mask])
            stand_leg_joint_vel = torch.mean(torch.abs(leg_joint_vel[stand_mask]))
            stand_wheel_joint_vel = torch.mean(torch.abs(wheel_joint_vel[stand_mask]))
        else:
            zero = torch.zeros((), device=self.device)
            stand_raw_action_delta = zero
            stand_action_delta = zero
            stand_action_limiter_error = zero
            stand_leg_target_delta = zero
            stand_leg_target_error = zero
            stand_leg_joint_vel = zero
            stand_wheel_joint_vel = zero

        return {
            "Debug/raw_action_delta_all": torch.mean(raw_action_delta),
            "Debug/raw_action_delta_leg_angle": torch.mean(raw_action_delta[:, 0:2]),
            "Debug/raw_action_delta_leg_length": torch.mean(raw_action_delta[:, 2:4]),
            "Debug/raw_action_delta_wheel": torch.mean(raw_action_delta[:, 4:6]),
            "Debug/action_delta_all": torch.mean(action_delta),
            "Debug/action_delta_leg_angle": torch.mean(action_delta[:, 0:2]),
            "Debug/action_delta_leg_length": torch.mean(action_delta[:, 2:4]),
            "Debug/action_delta_wheel": torch.mean(action_delta[:, 4:6]),
            "Debug/action_limiter_error": torch.mean(action_limiter_error),
            "Debug/leg_target_delta": torch.mean(leg_target_delta),
            "Debug/wheel_target_delta": torch.mean(wheel_target_delta),
            "Debug/leg_target_error": torch.mean(leg_target_error),
            "Debug/wheel_target_error": torch.mean(wheel_target_error),
            "Debug/leg_joint_vel_abs": torch.mean(torch.abs(leg_joint_vel)),
            "Debug/wheel_joint_vel_abs": torch.mean(torch.abs(wheel_joint_vel)),
            "Debug/stand_raw_action_delta": stand_raw_action_delta,
            "Debug/stand_action_delta": stand_action_delta,
            "Debug/stand_action_limiter_error": stand_action_limiter_error,
            "Debug/stand_leg_target_delta": stand_leg_target_delta,
            "Debug/stand_leg_target_error": stand_leg_target_error,
            "Debug/stand_leg_joint_vel_abs": stand_leg_joint_vel,
            "Debug/stand_wheel_joint_vel_abs": stand_wheel_joint_vel,
        }

    def inv_dofpos(self,L0):
        """
        该函数输入腿长L0，计算出理想状态下phi1和phi4作为关节目标值
        """
        A0 = torch.sqrt( (self.l5/2)**2 + L0**2 )
        alpha0 = torch.arcsin(L0/A0)
        alpha1 = torch.arccos( (A0**2 + self.l1**2 - self.l2**2)/ ( 2* A0*self.l1) )
        phi4 = torch.pi - alpha1 - alpha0
        phi1 = alpha1 + alpha0
        return phi1,phi4

    def inv_dofpos_from_length_angle(self, L0, theta):
        """
        输入目标腿长L0和虚拟腿摆角theta，反解五连杆两侧驱动关节角phi1/phi4。
        theta=0时，虚拟腿沿机体竖直方向；正负theta对应前后摆动。
        """
        phi0 = theta + torch.pi / 2.0
        xc = self.l5 / 2.0 + L0 * torch.cos(phi0)
        yc = L0 * torch.sin(phi0)

        left_dist = torch.sqrt(xc**2 + yc**2).clamp_min(1.0e-6)
        left_cos = (left_dist**2 + self.l1**2 - self.l2**2) / (2.0 * left_dist * self.l1)
        left_cos = torch.clamp(left_cos, -0.999, 0.999)
        left_beta = torch.atan2(yc, xc)
        left_alpha = torch.acos(left_cos)
        phi1 = left_beta + left_alpha

        right_x = xc - self.l5
        right_dist = torch.sqrt(right_x**2 + yc**2).clamp_min(1.0e-6)
        right_cos = (right_dist**2 + self.l4**2 - self.l3**2) / (2.0 * right_dist * self.l4)
        right_cos = torch.clamp(right_cos, -0.999, 0.999)
        right_beta = torch.atan2(yc, right_x)
        right_alpha = torch.acos(right_cos)
        phi4 = right_beta - right_alpha

        return phi1, phi4
    
    def VMC_calc_Leg(self,Pitch,phi1,phi4,isright):
        """
        计算任意一条腿的VMC的相关参数，避免修改类成员变量，返回计算结果
        """
        if isright:
            PitchLeg = -Pitch  # 获取腿部的俯仰角
        else:
            PitchLeg = Pitch  # 获取腿部的俯仰角

        # 计算坐标
        YD = self.l4 * torch.sin(phi4)
        YB = self.l1 * torch.sin(phi1)
        XD = self.l5 + self.l4 * torch.cos(phi4)
        XB = self.l1 * torch.cos(phi1)

        # 计算BD两点之间的距离
        lBD = torch.sqrt((XD - XB) ** 2 + (YD - YB) ** 2)

        # 计算中间变量
        A0 = 2 * self.l2 * (XD - XB)
        B0 = 2 * self.l2 * (YD - YB)
        C0 = self.l2 ** 2 + lBD ** 2 - self.l3 ** 2

        # 计算phi2和phi3角度
        phi2 = 2 * torch.arctan2((B0 + torch.sqrt(A0 ** 2 + B0 ** 2 - C0 ** 2)), A0 + C0)
        phi3 = torch.arctan2(YB - YD + self.l2 * torch.sin(phi2), XB - XD + self.l2 * torch.cos(phi2))

        # 计算C点的坐标
        XC = self.l1 * torch.cos(phi1) + self.l2 * torch.cos(phi2)
        YC = self.l1 * torch.sin(phi1) + self.l2 * torch.sin(phi2)

        # 计算腿长L0和极坐标角phi0
        L0 = torch.sqrt((XC - self.l5 / 2.0) ** 2 + YC ** 2)
        phi0 = torch.arctan2(YC, (XC - self.l5 / 2.0))
        alpha = phi0 - torch.pi / 2.0

        # 计算theta和其变化率
        theta = phi0 - torch.pi / 2.0 + PitchLeg   
        return {
            'phi0': phi0,
            'L0': L0,
            'alpha': alpha,
            'phi1': phi1,
            'phi4': phi4,     
            'phi2': phi2,
            'phi3': phi3,
            'theta': theta,
            'alpha': alpha,
        }
    
    def estimate_vel_and_pos(self,L_lun_vel,R_lun_vel):
        aver_v = (R_lun_vel*0.625 - L_lun_vel*0.625) / 2.0  # 计算平均速度
        self.v_filter.to(self.device)
        self.x_filter.to(self.device)        
        self.v_filter = aver_v
        # print(self.v_filter[0])
        # 提取滤波后的速度和位移
        self.x_filter += self.v_filter * self.dt # 通过速度估算位移
        
    
@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_xyvel: float,
    rew_scale_tracking_ang_vel: float,
    rew_scale_zvel: float,
    rew_scale_torques: float,
    rew_scale_action_rate: float,
    rew_scale_leg_length_reward: float,
    rew_scale_error_dof_reward: float,
    rew_scale_projected_gravity: float,
    rew_scale_ang_vel_xy: float,
    rew_scale_similar_legged: float,
    rew_scale_legged_theta: float,
    rew_scale_yaw_error: float,
    rew_scale_leg_length_action: float,
    rew_scale_leg_joint_vel: float,
    rew_scale_leg_joint_accel: float,
    rew_scale_stand_still: float,
    rew_scale_jump_height: float,
    rew_scale_jump_up_vel: float,
    jump_target_base_height: float,
    leg_length_R: torch.Tensor,
    leg_length_L: torch.Tensor,
    last_actions: torch.Tensor,
    actions: torch.Tensor,
    obs: torch.Tensor,
    vmcl_phi1: torch.Tensor,
    vmcl_phi4: torch.Tensor,
    vmcr_phi1: torch.Tensor,
    vmcr_phi4: torch.Tensor,
    theta_R: torch.Tensor,
    theta_L: torch.Tensor,
    yaw: torch.Tensor,
    target_yaw: torch.Tensor,
    lin_vel: torch.Tensor,
    lin_vel_w: torch.Tensor,
    base_height: torch.Tensor,
    ang_vel: torch.Tensor,
    commands: torch.Tensor,
    leg_joint_vel: torch.Tensor,
    last_leg_joint_vel: torch.Tensor,
    wheel_joint_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):

    # 计算总奖励
    # 存活奖励
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float()) 
    # 终止奖励
    rew_termination = rew_scale_terminated * reset_terminated.float() 
    # 线速度跟踪奖励。原先的0.005过尖，策略容易只学方向而不是稳定速度大小。
    x_vel_error = commands[:, 0] - lin_vel[:, 0]
    y_vel_error = commands[:, 1] - lin_vel[:, 1]
    rew_xyvel = rew_scale_xyvel * torch.exp(-(x_vel_error**2) / 0.08)
    rew_xyvel += 0.5 * rew_scale_xyvel * torch.exp(-(y_vel_error**2) / 0.08)

    # yaw角速度跟踪奖励。转向课程需要更宽的容差，否则稍有误差奖励就接近0。
    rew_tracking_ang_vel = rew_scale_tracking_ang_vel * torch.exp(-((commands[:,2] - ang_vel[:,2]) ** 2) / 0.05)

    # z轴线速度惩罚
    rew_zvel = rew_scale_zvel * obs[:,2]**2

    # 动作幅值惩罚。当前位置/速度控制模式下，actions表示腿角、腿长偏置和轮速目标的归一化命令。
    rew_torques = rew_scale_torques * torch.sum(torch.square(actions), dim=-1) 
    # 动作变化惩罚
    rew_action_rate = rew_scale_action_rate * torch.sum(torch.square(actions - last_actions), dim=-1)
    # 固定腿长训练阶段，腿长动作不参与执行，用轻惩罚让这两个输出逐渐收敛到0。
    rew_leg_length_action = rew_scale_leg_length_action * torch.sum(torch.square(actions[:, 2:4]), dim=-1)
    # 腿部高频抖动惩罚：关节速度和关节速度变化越大，惩罚越大。
    rew_leg_joint_vel = rew_scale_leg_joint_vel * torch.sum(torch.square(leg_joint_vel), dim=-1)
    rew_leg_joint_accel = rew_scale_leg_joint_accel * torch.sum(torch.square(leg_joint_vel - last_leg_joint_vel), dim=-1)
    # 静止命令下额外压腿和轮子的微小来回动作，不影响正常行驶/转向命令。
    stand_command_weight = torch.exp(-(torch.square(commands[:, 0]) / 0.02 + torch.square(commands[:, 2]) / 0.02))
    stand_motion = torch.sum(torch.square(leg_joint_vel), dim=-1) + 0.01 * torch.sum(torch.square(wheel_joint_vel), dim=-1)
    stand_motion = torch.clamp(stand_motion, max=20.0)
    rew_stand_still = rew_scale_stand_still * stand_command_weight * stand_motion
    # 跳跃第一版奖励：鼓励车体抬高和向上速度。姿态奖励/终止条件会约束不要靠翻车刷高度。
    rew_jump_height = rew_scale_jump_height * torch.exp(-torch.square(base_height - jump_target_base_height) / 0.01)
    rew_jump_up_vel = rew_scale_jump_up_vel * torch.clamp(lin_vel_w[:, 2], min=0.0, max=2.0)

    # 腿长奖励
    rew_leg_length_reward = rew_scale_leg_length_reward * torch.exp(-((commands[:,3] - leg_length_R.squeeze(1)) ** 2 + (commands[:,3] - leg_length_L.squeeze(1)) ** 2)/ 0.003)

    # 关节位置误差惩罚
    rew_error_dof = rew_scale_error_dof_reward * (torch.abs(obs[:,13]) + torch.abs(obs[:,14]) + torch.abs(obs[:,15]) + torch.abs(obs[:,16]))

    # 重力投影奖励
    projected_gravity_error = 1 + obs[:, 8] #[0, 0.2]
    projected_gravity_error = torch.square(projected_gravity_error)
    gravity_xy_error = torch.sum(torch.square(obs[:, 6:8]), dim=1)
    rew_projected_gravity = rew_scale_projected_gravity * torch.exp(-(projected_gravity_error + gravity_xy_error) / 0.05)
    
    # ​XY 轴角速度惩罚
    rew_ang_vel_xy = rew_scale_ang_vel_xy * torch.sum(torch.square(obs[:, 3:5]), dim=1)

    # 防劈叉奖励
    vmc_error_1 = (torch.pi-vmcl_phi4) - vmcr_phi1
    vmc_error_2 = (torch.pi-vmcr_phi4) - vmcl_phi1
    rew_similar_legged = rew_scale_similar_legged * torch.exp(-(vmc_error_1 + vmc_error_2)**2 / 0.01)

    # 腿部姿态矫正奖励
    rew_legged_theta = rew_scale_legged_theta * torch.exp(-(torch.abs(theta_L.squeeze(1)) + torch.abs(theta_R.squeeze(1)))**2 / 0.1)

    # yaw矫正惩罚
    # rew_yaw_error = rew_scale_yaw_error * (torch.exp(-(yaw.squeeze(1)-target_yaw)**2 / 0.001))
    rew_yaw_error = rew_scale_yaw_error * torch.abs(yaw.squeeze(1)-target_yaw)

    total_reward = rew_alive + rew_alive + rew_xyvel + rew_termination + rew_tracking_ang_vel + rew_zvel + rew_torques + rew_action_rate \
          + rew_leg_length_action + rew_leg_joint_vel + rew_leg_joint_accel + rew_stand_still + rew_jump_height + rew_jump_up_vel + rew_leg_length_reward + rew_error_dof + rew_projected_gravity + rew_ang_vel_xy + rew_similar_legged + rew_legged_theta + rew_yaw_error
    
    return (
        total_reward,
        rew_alive,
        rew_termination,
        rew_xyvel,
        rew_tracking_ang_vel,
        rew_zvel,
        rew_torques,
        rew_action_rate,
        rew_leg_length_action,
        rew_leg_joint_vel,
        rew_leg_joint_accel,
        rew_stand_still,
        rew_jump_height,
        rew_jump_up_vel,
        rew_leg_length_reward,
        rew_error_dof,
        rew_projected_gravity,
        rew_ang_vel_xy,
        rew_similar_legged,
        rew_legged_theta,
        rew_yaw_error,
    )
