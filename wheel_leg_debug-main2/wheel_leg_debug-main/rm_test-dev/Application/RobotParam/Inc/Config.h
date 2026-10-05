/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : Config.c
 * @brief          : Configuare the Robot Functions
 * @author         : Yan Yuanbin
 * @date           : 2023/05/21
 * @version        : v1.0
 ******************************************************************************
 * @attention      : To be perfected
 ******************************************************************************
 */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef ROBOT_CONFIG_H
#define ROBOT_CONFIG_H

/* Includes ------------------------------------------------------------------*/
#include "stdint.h"
#include "stdbool.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

/* General physics and mathematics constants ---------------------------------*/

/**
 * @brief the value of local gravity acceleration
 */

#define VAL_LIMIT(x, min, max) \
	do                         \
	{                          \
		if ((x) > (max))       \
		{                      \
			(x) = (max);       \
		}                      \
		else if ((x) < (min))  \
		{                      \
			(x) = (min);       \
		}                      \
	} while (0U)

#define GravityAccel 9.718f

#define Angle_to_rad 0.01745329f

#define Rad_to_angle 57.2957732f

/**
 * @brief Euler's Number
 */
#define Euler_Number 2.718281828459045f

/**
 * @brief radian system rotation degrees system , 180.f/PI
 */
#define RadiansToDegrees 57.295779513f

/**
 * @brief degrees system rotation radian system , PI/180.f
 */
#define DegreesToRadians 0.01745329251f

/* Vision reslove constants -------------------------------------------------*/

/**
 * @brief  Decision Marking mode
 *         0: select the minimum yaw armor
 *         1: select the minimum distance armor
 */
#define Yaw_Distance_Decision 0

/**
 * @brief ballistic coefficient
 * @note  17mm: 0.038
 *        42mm: 0.019
 */
#define Bullet_Coefficient 0.038f

/**
 * @brief the half width of little armor
 */
#define LittleArmor_HalfWidth 0.07f

/**
 * @brief the half width of Large armor
 */
#define LargeArmor_HalfWidth 0.1175f

// /* IMU reslove constants ---------------------------------------------------*/
// /**
//  * @brief the flag of Bmi088 Calibration
//  *        0: DISABLE
//  *        1: ENABLE
//  */
// #define IMU_Calibration_ENABLE 0U

/**
 * @brief the index of pitch angle update
 */
#define IMU_ANGLE_INDEX_PITCH 2U
/**
 * @brief the index of yaw angle update
 */
#define IMU_ANGLE_INDEX_YAW 0U
/**
 * @brief the index of roll angle update
 */
#define IMU_ANGLE_INDEX_ROLL 1U

/**
 * @brief the index of pitch gyro update
 */
#define IMU_GYRO_INDEX_PITCH 0U
/**
 * @brief the index of yaw gyro update
 */
#define IMU_GYRO_INDEX_YAW 2U
/**
 * @brief the index of roll gyro update
 */
#define IMU_GYRO_INDEX_ROLL 1U

/**
 * @brief the index of pitch accel update
 */
#define IMU_ACCEL_INDEX_PITCH 0U
/**
 * @brief the index of yaw accel update
 */
#define IMU_ACCEL_INDEX_YAW 2U
/**
 * @brief the index of roll accel update
 */
#define IMU_ACCEL_INDEX_ROLL 1U

/* Remote reslove constants -----------------------------------------------*/
/**
 * @brief the flag of remote control receive frame data
 * @note  0: CAN
 *        1: USART
 */
#define REMOTE_FRAME_USART_CAN 0U

/* reslove constants ---------------------------------------------------*/
// 启用DT7遥控器
#define ENABLE_ALARM_RC_OFFLINE false
// 启用ps2遥控器
#define ENABLE_ALARM_PS2_OFFLINE true
// 启用电机离线报警
#define ENABLE_ALARM_MOTOR_OFFLINE false
// 启用裁判系统离线检测
#define ENABLE_CHECK_REFEREE_OFFLINE false
// 启用电池电压过低报警
#define ENABLE_ALARM_VBAT_LOW false

// Dedicated joint/VMC mapping test. Set to 0U to restore the complete LQR
// controller and its normal POLARITY_LQR telemetry profile in one step.
#define JOINT_MAP_TEST_ENABLE 0U

// Repeatable mechanical-hardstop reference test for checking the leg angle
// coordinate chain. 1U: START runs only this test; 0U: restore normal full LQR.
// This diagnostic never sends the DM "save zero position" command.
#define LEG_HARDSTOP_REF_TEST_ENABLE 0U

// Full-LQR theta zero-offset experiment. 1U maps the measured symmetric
// raw-leg equilibrium (+left/-right) to zero LQR state without changing K.
// Set to 0U to restore the previous full-LQR theta_target behavior.
#define FULL_LQR_THETA_ZERO_OFFSET_TEST_ENABLE 1U

#if (JOINT_MAP_TEST_ENABLE && LEG_HARDSTOP_REF_TEST_ENABLE)
#error "Enable only one dedicated chassis test at a time"
#endif

// UART7 CSV telemetry profiles (921600, 8N1).
// THETA_CAL is a compact DMA profile for calibrating the physical vertical-leg
// pose against phi0/theta at several leg lengths without blocking control tasks.
#define UART7_TELEMETRY_PROFILE_LEGACY_SAFE 0U
#define UART7_TELEMETRY_PROFILE_POLARITY_LQR 1U
#define UART7_TELEMETRY_PROFILE_THETA_CAL 2U
#if (JOINT_MAP_TEST_ENABLE)
#define UART7_TELEMETRY_PROFILE UART7_TELEMETRY_PROFILE_LEGACY_SAFE
#else
#define UART7_TELEMETRY_PROFILE UART7_TELEMETRY_PROFILE_THETA_CAL
#endif
#define ENABLE_UART7_TELEMETRY 1U
#if ((UART7_TELEMETRY_PROFILE == UART7_TELEMETRY_PROFILE_POLARITY_LQR) || \
     (UART7_TELEMETRY_PROFILE == UART7_TELEMETRY_PROFILE_THETA_CAL))
#define UART7_TELEMETRY_PERIOD_MS 20U
#else
#define UART7_TELEMETRY_PERIOD_MS 50U
#endif

// 是否启用底盘校准功能
#define ENABLE_CHASSIS_CALIBRATE false

#endif // ROBOT_CONFIG_H
