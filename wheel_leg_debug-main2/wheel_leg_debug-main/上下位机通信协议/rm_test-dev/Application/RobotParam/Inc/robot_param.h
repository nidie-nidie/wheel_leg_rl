/**
  ****************************(C) COPYRIGHT 2024 Polarbear****************************
  * @file       robot_param.h
  * @brief      这里是机器人参数配置文件，包括底盘参数，物理参数等
  * @history
  *  Version    Date            Author          Modification
  *  V1.0.0     Mar-31-2024     Penguin         1. done
  *  V1.0.1     Apr-16-2024     Penguin         1. 添加云台和发射机构类型
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  ****************************(C) COPYRIGHT 2024 Polarbear****************************
  */

#ifndef ROBOT_PARAM_H
#define ROBOT_PARAM_H

#include "robot_typedef.h"
#include "struct_typedef.h"

// 机器人速度限制参数
#define MAX_SPEED_VECTOR_VX (3.5f)
#define MAX_SPEED_VECTOR_VY (3.5f)
#define MAX_SPEED_VECTOR_WZ (6.0f)

#define MIN_SPEED_VECTOR_VX (-MAX_SPEED_VECTOR_VX)
#define MIN_SPEED_VECTOR_VY (-MAX_SPEED_VECTOR_VY)
#define MIN_SPEED_VECTOR_WZ (-MAX_SPEED_VECTOR_WZ)

// 关节电机相关参数
#define MAX_TORQUE_PROTECT (25.0f) // (Nm)最大扭矩保护
// DM控制参数
#define CALIBRATE_VEL_KD (4.0f)  // 校准MIT速度控制KD
#define ZERO_FORCE_VEL_KD (4.0f) // 无力MIT速度控制KD
#define NORMAL_POS_KP (20.0f)    // 正常MIT位置控制KP
#define NORMAL_POS_KD (1.0f)     // 正常MIT位置控制KD
#define DEBUG_POS_KP (8.0f)      // 调试MIT位置控制KP
#define DEBUG_POS_KD (0.8f)      // 调试MIT位置控制KD
// DM电机限位
#define MIN_J0_ANGLE (-0.6f) // (rad)关节角度下限
#define MIN_J1_ANGLE (-1.8f) // (rad)关节角度下限
#define MIN_J2_ANGLE (-1.8f) // (rad)关节角度下限
#define MIN_J3_ANGLE (0.0f)  // (rad)关节角度下限
#define MAX_J0_ANGLE (1.8f)  // (rad)关节角度上限
#define MAX_J1_ANGLE (0.0f)  // (rad)关节角度上限
#define MAX_J2_ANGLE (0.6f)  // (rad)关节角度上限
#define MAX_J3_ANGLE (1.8f)  // (rad)关节角度上限

#define MAX_JOINT_TORQUE (7.0f)                        // (Nm)关节最大扭矩
#define MAX_JOINT_TORQUE_JUMP (20.0f)                  // (Nm)跳跃时的关节最大扭矩
#define MIN_JOINT_TORQUE (-MAX_JOINT_TORQUE)           // (Nm)关节最小扭矩
#define MIN_JOINT_TORQUE_JUMP (-MAX_JOINT_TORQUE_JUMP) // (Nm)跳跃时的关节最小扭矩
// 遥控器控制机器人相关参数
#define MAX_ROLL (0.3f)        // (rad)遥控器控制的最大滚转角，超过这个角度底盘不再增加滚转角以保护底盘
#define MAX_LEG_LENGTH (0.35f) // (m)遥控器控制的最大腿长，超过这个长度底盘不再增加腿长以保护底盘
#define MIN_ROLL (-MAX_ROLL)
#define MIN_LEG_LENGTH (0.11f)

#define INIT_LEG_LENGTH (0.2f) // (m)底盘初始腿长, 测试0.12-0.14比较合适
// #define INIT_LEG_LENGTH (0.20f) // (m)底盘初始腿长
#define INIT_ROLL (0.0f) // (rad)底盘初始滚转角

// physical parameters ---------------------
#define LEG_L1 (0.215f) // (m)腿1长度
#define LEG_L2 (0.258f) // (m)腿2长度
#define LEG_L3 (LEG_L2) // (m)腿3长度
#define LEG_L4 (LEG_L1) // (m)腿4长度
#define LEG_L5 (0.0f)   // (m)关节间距

#define BODY_MASS (8.5f) // (kg)机身重量
// DM电机初始角度与水平线的关系
#define J0_ANGLE_OFFSET (-0.19163715f)       // (rad)关节0角度偏移量(电机0点到水平线的夹角)
#define J1_ANGLE_OFFSET (0.19163715f + M_PI) // (rad)关节1角度偏移量(电机0点到水平线的夹角)
#define J2_ANGLE_OFFSET (0.19163715f + M_PI) // (rad)关节2角度偏移量(电机0点到水平线的夹角)
#define J3_ANGLE_OFFSET (-0.19163715f)       // (rad)关节3角度偏移量(电机0点到水平线的夹角)
// 电机旋转方向定义
#define J0_DIRECTION (1)
#define J1_DIRECTION (1)
#define J2_DIRECTION (1)
#define J3_DIRECTION (1)

#define W0_DIRECTION (1)
#define W1_DIRECTION (-1)

// 轮子相关参数
#define WHEEL_MASS (0.65f)     // (kg)轮子重量
#define WHEEL_RADIUS (0.0625f) // (m)轮子半径
#define WHEEL_BASE (0.51175f)  // (m)驱动轮轴距，即左右轮之间的默认距离

// 机器人物理参数
#define BODY_MASS (8.5f)                   // (kg)机身重量
#define BODY_GRAVITY (BODY_MASS * GRAVITY) // (N)机身重力

// 底盘校准相关参数
#define ZERO_POS_THRESHOLD 0.001f     // 关节位置小于该阈值时认为已经校准到位
#define CALIBRATE_STOP_VELOCITY 0.05f // 关节速度小于该阈值时认为已经停止, rad/s
#define CALIBRATE_STOP_TIME 200       // 校准停止状态持续超过该时间时认为已经稳定, ms
#define CALIBRATE_VELOCITY 2.0f       // 校准时的关节速度, rad/s
// IMU校准相关参数
#define TEMP_CALI_THRESHOLD 40.0f // IMU校准时温度上限
// 电池低压保护相关参数
#define VBAT_LOW_WARNING_THRESHOLD 100 // 电池电压计数高于该值时自动断电

// 底盘错误代码定义
#define JOINT_ERROR_OFFSET ((uint8_t)1 << 0) // 关节电机错误偏移量
#define WHEEL_ERROR_OFFSET ((uint8_t)1 << 1) // 驱动轮电机错误偏移量
#define DBUS_ERROR_OFFSET ((uint8_t)1 << 2)  // dbus错误偏移量
#define FLOATING_OFFSET ((uint8_t)1 << 3)    // 悬空状态偏移量

// 起立用的pid
#define KP_CHASSIS_STAND_UP (2000.0f)
#define KI_CHASSIS_STAND_UP (0.0f)
#define KD_CHASSIS_STAND_UP (10.0f)
#define MAX_IOUT_CHASSIS_STAND_UP (0.0f)
#define MAX_OUT_CHASSIS_STAND_UP (2000.0f)
// 腿长跟踪长度环PID参数
#define KP_CHASSIS_LEG_LENGTH_LENGTH (150.0f)
#define KI_CHASSIS_LEG_LENGTH_LENGTH (0.0f)
#define KD_CHASSIS_LEG_LENGTH_LENGTH (1500.0f)
#define MAX_IOUT_CHASSIS_LEG_LENGTH_LENGTH (0.0f)
#define MAX_OUT_CHASSIS_LEG_LENGTH_LENGTH (40.0f)
#define ALPHA_LEG_LENGTH_LENGTH (0.1f)
// roll轴跟踪角度环PID参数
#define KP_CHASSIS_ROLL_ANGLE (0.3f)
#define KI_CHASSIS_ROLL_ANGLE (0.0f)
#define KD_CHASSIS_ROLL_ANGLE (0.2f)
#define MAX_IOUT_CHASSIS_ROLL_ANGLE (0.0f)
#define MAX_OUT_CHASSIS_ROLL_ANGLE (0.1f)
// 防劈叉补偿PID参数
#define KP_CHASSIS_TP 30.0f
#define KI_CHASSIS_TP 0.0f
#define KD_CHASSIS_TP 1.0f
#define MAX_IOUT_CHASSIS_TP 0.0f
#define MAX_OUT_CHASSIS_TP 2.0f
// 偏航角补偿PID参数
#define KP_CHASSIS_TURN 2.5f
#define KI_CHASSIS_TURN 0.0f
#define KD_CHASSIS_TURN 0.3f
#define MAX_IOUT_CHASSIS_TURN 0.0f
#define MAX_OUT_CHASSIS_TURN 2.41f // 轮毂电机的额定扭矩(这里是翎控电机的额定扭矩)

// 离地检测相关参数
#define TAKE_OFF_FN_THRESHOLD (3.0f) // 支持力阈值，当支持力小于这个值时认为离地

// offset parameters ---------------------
#define X0_OFFSET (0.0f)   // 目标theta偏移量
#define X1_OFFSET (0.0f)   // 目标theta_dot偏移量
#define X2_OFFSET (-0.09f) // 目标x偏移量
#define X3_OFFSET (0.0f)   // 目标x_dot偏移量
#define X4_OFFSET (0.0f)   // 目标phi偏移量
#define X5_OFFSET (0.0f)   // 目标phi_dot偏移量

#endif /* ROBOT_PARAM_H */
