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
#define NORMAL_POS_KP (25.0f)     // 正常MIT位置控制KP
#define NORMAL_POS_KD (1.0f)     // 正常MIT位置控制KD
// Joint mapping test: pure VMC Tp feedforward plus light MIT damping.
#define JOINT_MAP_TEST_TP_NM (2.00f)
#define JOINT_MAP_TEST_MIT_KP (0.0f)
#define JOINT_MAP_TEST_MIT_KD (0.8f)
#define JOINT_MAP_TEST_MOTOR_TORQUE_LIMIT (2.00f)
#define LEG_HARDSTOP_REF_TEST_TORQUE (2.50f)          // Nm, explicit MIT torque feedforward
#define LEG_HARDSTOP_REF_TEST_HOLD_TORQUE (0.40f)     // Nm, light contact hold after one joint reaches
#define LEG_HARDSTOP_REF_TEST_VEL_KD (4.0f)           // viscous damping; no pure velocity command
#define LEG_HARDSTOP_REF_TEST_MIN_TRAVEL (0.10f)      // rad, reject an initial load/friction stall
#define LEG_HARDSTOP_REF_TEST_STOP_VELOCITY (0.05f)   // rad/s
#define LEG_HARDSTOP_REF_TEST_MIN_STOP_TORQUE (0.80f) // Nm
#define LEG_HARDSTOP_REF_TEST_MAX_TORQUE (3.00f)      // Nm, abort above this
#define LEG_HARDSTOP_REF_TEST_STOP_TIME_MS (400U)
#define LEG_HARDSTOP_REF_TEST_START_GRACE_MS (600U)
#define LEG_HARDSTOP_REF_TEST_TIMEOUT_MS (8000U)
#define LEG_HARDSTOP_REF_TEST_MAX_FDB_AGE_MS (50U)
#define SAFE_DEBUG_MIT_POS_KP (20.0f) // SAFE debug: soften MIT position loop so VMC/LQR Tp can act
#define SAFE_DEBUG_MIT_POS_KD (0.6f)  // SAFE debug: MIT velocity damping
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

#define MAX_JOINT_TORQUE (20.0f)                       // (Nm)关节最大扭矩
#define MAX_JOINT_TORQUE_JUMP (20.0f)                  // (Nm)跳跃时的关节最大扭矩
#define MIN_JOINT_TORQUE (-MAX_JOINT_TORQUE)           // (Nm)关节最小扭矩
#define MIN_JOINT_TORQUE_JUMP (-MAX_JOINT_TORQUE_JUMP) // (Nm)跳跃时的关节最小扭矩
// 遥控器控制机器人相关参数
#define MAX_ROLL (0.3f)        // (rad)遥控器控制的最大滚转角，超过这个角度底盘不再增加滚转角以保护底盘
#define MAX_LEG_LENGTH (0.35f) // (m)遥控器控制的最大腿长，超过这个长度底盘不再增加腿长以保护底盘
#define MIN_ROLL (-MAX_ROLL)
#define MIN_LEG_LENGTH (0.11f)

#define INIT_LEG_LENGTH (0.20f) // (m)SAFE和起立的目标腿长
// #define INIT_LEG_LENGTH (0.25f) // (m)previous SAFE debug leg length
#define INIT_ROLL (0.0f) // (rad)底盘初始滚转角

// SAFE debug leg length target ramp. This prevents leg_set from jumping from
// the current measured leg length directly to INIT_LEG_LENGTH at SAFE entry.
#define SAFE_DEBUG_LEG_SET_RAMP_ENABLE (1U)
#define SAFE_DEBUG_LEG_SET_RAMP_RATE (0.10f) // m/s

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
#define KP_CHASSIS_LEG_LENGTH_LENGTH (200.0f)
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

// Original project LQR state offsets ---------------------
#define X0_OFFSET (0.0f)   // 目标theta偏移量
#define X1_OFFSET (0.0f)   // 目标theta_dot偏移量
#define X2_OFFSET (-0.09f) // 目标x偏移量
#define X3_OFFSET (0.0f)   // 目标x_dot偏移量
#define X4_OFFSET (-0.0382f) // 目标phi偏移量，对应原始Pitch目标+0.0382 rad
#define X5_OFFSET (0.0f)   // 目标phi_dot偏移量

// Chassis controller selection ---------------------------
// 1: SAFE uses the complete LQR/VMC output path (no SAFE-only theta trim,
//    wheel/Tp scaling, wheel damping, or explicit pitch-Tp override).
// 0: retain the staged SAFE debug path below for bench tests.
#define CHASSIS_FULL_LQR_ENABLE (1U)

// SAFE debug-only parameters -----------------------------
// SAFE debug compatibility marker. Current clean(2)-style LQR feeds x/v
// directly into CalcLQR, so this macro is not used by the active control path.
#define SAFE_LQR_XV_SCALE (1.0f)
// Software wheel torque clamp before the LK motor's own torque/current clamp.
#define SAFE_WHEEL_T_LIMIT (1.50f) // Nm
// SAFE mode: restore LQR wheel torque gradually, then raise toward 1.0f.
#define SAFE_LQR_WHEEL_SCALE (0.05f)
// SAFE debug wheel damping. In SAFE, wheel torque uses damping instead of LQR T.
#define SAFE_WHEEL_DAMPING_KD (0.0f) // SAFE debug: pure wheel LQR test, no velocity damping
// SAFE debug wheel pitch assist. This keeps LQR wheel output disabled, but lets
// the wheel visibly react to pitch for sign and authority testing.
#define SAFE_DEBUG_WHEEL_PITCH_ASSIST_ENABLE (0U)
#define SAFE_DEBUG_WHEEL_PITCH_SIGN (1.0f)       // flip if pitch response is reversed
#define SAFE_DEBUG_WHEEL_PITCH_KP (0.00f)        // Nm/rad
#define SAFE_DEBUG_WHEEL_PITCH_KD (0.08f)        // Nm/(rad/s)
#define SAFE_DEBUG_WHEEL_PITCH_LIMIT (0.80f)     // Nm before wheel direction mapping
// SAFE debug leg support feedforward scale. Original project logic is 1.0f.
#define SAFE_LEG_GRAVITY_FF_SCALE (0.80f)

// SAFE static phi0 equilibrium trim search. This is a debug-only mode used to
// find a better phi0_eq before restoring full LQR.
#define SAFE_STATIC_PHI0_TRIM_ENABLE (1U)
#define SAFE_STATIC_COMMON_WHEEL_ENABLE (1U)
#define SAFE_STATIC_LQR_WHEEL_SCALE (0.45f)
#define SAFE_STATIC_LQR_TP_SCALE (2.8f)
#define SAFE_DEBUG_LQR_TP_SIGN (1.0f)     // SAFE debug: set 1.0f to restore LQR Tp sign
#define SAFE_PHI0_EQ_TRIM (-0.3491f)      // rad, fixed equilibrium offset: left 70.0deg / right 110.0deg
#define FULL_LQR_THETA_ZERO_OFFSET_TEST_TRIM (-0.2749f) // rad: left theta +15.75deg, right theta -15.75deg
#define SAFE_DEBUG_COMMON_WHEEL_TEST_ENABLE (0U) // SAFE debug: fixed common wheel torque for direction test
#define SAFE_DEBUG_COMMON_WHEEL_TEST_T (0.08f)   // Nm, keep small and test with robot lifted or held
#define SAFE_STATIC_PITCH_PHI0_TRIM_ENABLE (0U)  // SAFE debug: live pitch feedback into phi0_cmd
#define SAFE_STATIC_PITCH_PHI0_TRIM_KP (0.30f) // rad/rad, uses SAFE_DEBUG_PITCH_TO_PHI0_SIGN
#define SAFE_STATIC_PITCH_PHI0_TRIM_LIMIT (0.12f) // rad
#define SAFE_STATIC_PITCH_PHI0_TRIM_RATE_LIMIT (0.60f) // rad/s, slow phi0 target near balance
#define SAFE_PHI0_TRIM_SIGN (1.0f)        // flip if trim moves pitch farther from target
#define SAFE_PHI0_TRIM_KI (0.0f)          // 1/s, set >0 only when searching phi0 equilibrium
#define SAFE_PHI0_TRIM_LIMIT (0.80f)      // rad
#define SAFE_PHI0_TRIM_MAX_ABS_V (0.25f)  // m/s, freeze trim if chassis is moving too fast
#define SAFE_PHI0_TRIM_MAX_ABS_PITCH_RATE (1.2f) // rad/s, freeze trim during fast pitch motion
#define SAFE_PHI0_TRIM_MIN_L0 (0.18f)     // m, freeze trim if leg length is unreasonable
#define SAFE_PHI0_TRIM_MAX_L0 (0.32f)     // m
#define SAFE_PHI0_TRIM_TORQUE_MARGIN (0.0f) // Nm, freeze trim only at joint torque clamp
#define SAFE_PHI0_TRIM_WHEEL_T_MARGIN (0.03f) // Nm, freeze trim near wheel torque clamp

// SAFE debug pitch posture loop. Set SAFE_USE_EXPLICIT_PITCH_TP to 0 to restore
// the original LQR Tp path in SAFE.
#define SAFE_USE_EXPLICIT_PITCH_TP (0U)
#define SAFE_PITCH_TARGET (0.0f)    // rad, body pitch target for SAFE debug
#define SAFE_PITCH_TP_SIGN (1.0f)   // flip this first if the pitch response is reversed
#define SAFE_PITCH_TP_KP (2.5f)     // Nm/rad
#define SAFE_PITCH_TP_KD (0.12f)    // Nm/(rad/s)
#define SAFE_PITCH_TP_LIMIT (1.2f)  // Nm
#define SAFE_DEBUG_TP_CMD_SCALE (1.0f)
// SAFE debug: isolate pitch-to-phi0 position target by scaling the VMC Tp path.
#define SAFE_LQR_TP_SCALE (4.0f)   // SAFE mode: scale LQR leg Tp
// SAFE debug pitch-to-leg-position bias. Keep gain at 0.0f while validating
// the LQR Tp path, so phi0 position target stays neutral.
#define SAFE_DEBUG_PITCH_TO_PHI0_SIGN (1.0f)
#define SAFE_DEBUG_PITCH_TO_PHI0_GAIN (0.00f)  // rad/rad
#define SAFE_DEBUG_PITCH_TO_PHI0_LIMIT (0.35f) // rad
#define SAFE_DEBUG_LEFT_PHI0_BIAS_SIGN (1.0f)
#define SAFE_DEBUG_RIGHT_PHI0_BIAS_SIGN (-1.0f)
// SAFE theta Tp scaffold. Keep zero while isolating pitch posture response.
#define SAFE_THETA_TP_KP (1.0f)     // Nm/rad
#define SAFE_THETA_TP_KD (0.0f)     // Nm/(rad/s)
#define SAFE_THETA_TP_LIMIT (0.3f)  // Nm

#endif /* ROBOT_PARAM_H */
