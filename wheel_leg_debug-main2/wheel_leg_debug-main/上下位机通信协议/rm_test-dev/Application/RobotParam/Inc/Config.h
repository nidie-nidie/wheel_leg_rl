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
// Sim2Real bring-up: keep only the sensors, CAN motor bus, and protocol path enabled.
// Re-enable these modules when using the original remote-control/debug workflow.
#define ENABLE_TASK_DETECT false
#define ENABLE_TASK_CHASSIS_CONTROL false
#define ENABLE_TASK_OBSERVE false
#define ENABLE_TASK_MUSIC false
#define ENABLE_TASK_LED_FLOW true
#define ENABLE_UART5_SBUS_RX false
#define ENABLE_USART1_IMAGE_TRANS_RX false
#define ENABLE_USART1_REFEREE_RX false
#define ENABLE_SIM2REAL_PROTOCOL true
#define SIM2REAL_STATE_TX_PERIOD_MS 20U
#define ENABLE_SIM2REAL_MOTOR_OUTPUT true
#define ENABLE_SIM2REAL_RUN_OUTPUT true
#define ENABLE_SIM2REAL_IDLE_CAN_KEEPALIVE false
#define SIM2REAL_MOTOR_TX_PERIOD_MS 5U
#define SIM2REAL_ENABLE_TX_PERIOD_MS 1U
#define SIM2REAL_ENABLE_REPEAT_COUNT 5U
#define SIM2REAL_ESTOP_HOLD_MS 2000U
#define SIM2REAL_DAMPING_DM_KD 1.0f
#define SIM2REAL_HARD_LEG_TAU_LIMIT_NM 5.0f
#define SIM2REAL_HARD_WHEEL_TAU_LIMIT_NM 1.0f
#define SIM2REAL_HARD_LEG_DQ_LIMIT_RAD_S 10.0f
#define SIM2REAL_HARD_WHEEL_DQ_LIMIT_RAD_S 10.0f
#define ENABLE_USB_IMU_EULER_TX false
#define USB_IMU_EULER_TX_PERIOD_MS 50U

// 启用DT7遥控器
#define ENABLE_ALARM_RC_OFFLINE false
// 启用ps2遥控器
#define ENABLE_ALARM_PS2_OFFLINE false
// 启用电机离线报警
#define ENABLE_ALARM_MOTOR_OFFLINE false
// 启用裁判系统离线检测
#define ENABLE_CHECK_REFEREE_OFFLINE false
// 启用电池电压过低报警
#define ENABLE_ALARM_VBAT_LOW false

// 是否启用底盘校准功能
#define ENABLE_CHASSIS_CALIBRATE false

#endif // ROBOT_CONFIG_H
