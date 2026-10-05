/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : Motor.h
 * @brief          : The header file of Motor.h
 * @author         : GrassFan Wang
 * @date           : 2025/01/2
 * @version        : v1.0
 ******************************************************************************
 * @attention      : None
 ******************************************************************************
 */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef DEVICE_MOTOR_H
#define DEVICE_MOTOR_H

/* Includes ------------------------------------------------------------------*/
#include "Config.h"
#include "struct_typedef.h"
#include "stm32h723xx.h"
#include "bsp_can.h"

#define MOTOR_STABLE_RUNNING_TIME 10 // (ms)电机稳定运行时间

// 达妙电机控制模式
#define DM_MODE_MIT 0x000
#define DM_MODE_POS 0x100
#define DM_MODE_SPEED 0x200
// 达妙电机参数范围，见达妙电机上位机
#define DM_P_MIN -12.5f
#define DM_P_MAX 12.5f
#define DM_V_MIN -45.0f
#define DM_V_MAX 45.0f
#define DM_T_MIN -54.0f
#define DM_T_MAX 54.0f
#define DM_KP_MIN 0.0f
#define DM_KP_MAX 500.0f
#define DM_KD_MIN 0.0f
#define DM_KD_MAX 5.0f

/**
 * @brief typedef structure that contains the information for the motor FDCAN transmit and recieved .
 */
typedef struct
{
	uint32_t TxIdentifier; /*!< Specifies FDCAN transmit identifier */
	// only for DJI motor
	uint32_t RxIdentifier; /*!< Specifies FDCAN recieved identifier */

} Motor_CANFrameInfo_typedef;

/*********************** typedefs for DM Motor Control ***********************/

/**
 * @brief  typedef enum that control mode the type of DM_Motor.
 */
typedef enum
{
	DM_MIT,
	DM_POSITION_VELOCITY,
	DM_VELOCITY,
} DM_Motor_Control_Mode_Type_e;

/**
 * @brief  typedef enum that CMD of DM_Motor .
 */
typedef enum
{
	DM_Motor_Enable,
	DM_Motor_Disable,
	DM_Motor_Save_Zero_Position,
	DM_Motor_CMD_Type_Num,
} DM_Motor_CMD_Type_e;

/**
 * @brief typedef structure that contains the param range for the DM_Motor .
 */
typedef struct
{
	float P_MAX;
	float P_MIN;
	float V_MAX;
	float V_MIN;
	float T_MAX;
	float T_MIN;
} DM_Motor_Param_Range_Typedef;

/**
 * @brief typedef structure that contains the data for the DJI Motor Device.
 */
typedef struct
{
	uint8_t State;			 /*!< Motor Message */
	uint16_t P_int;			 /*!< Motor Positon  uint16 */
	uint16_t V_int;			 /*!< Motor Velocity uint16 */
	uint16_t T_int;			 /*!< Motor Torque   uint16 */
	float Position;			 /*!< Motor Positon  */
	float Velocity;			 /*!< Motor Velocity */
	float Torque;			 /*!< Motor Torque   */
	float Temperature_MOS;	 /*!< Motor Temperature_MOS   */
	float Temperature_Rotor; /*!< Motor Temperature_Rotor */
	float Angle;

} DM_Motor_Data_Typedef;

/**
 * @brief typedef structure that contains the information for the DJI Motor Device.
 */
typedef struct
{

	DM_Motor_Control_Mode_Type_e Control_Mode;
	Motor_CANFrameInfo_typedef FDCANFrame;
	DM_Motor_Param_Range_Typedef Param_Range;
	DM_Motor_Data_Typedef Data;

	uint32_t last_fdb_time; // 上次反馈时间
	bool offline;			// 电机是否离线
} DM_Motor_Info_Typedef;

/*********************** typedefs for DM Motor Control ***********************/

/**
 * @brief typedef enum that contains the type of DJI Motor Device.
 */
typedef enum
{
	DJI_GM6020,
	DJI_M3508,
	DJI_M2006,
	DJI_MOTOR_TYPE_NUM,
} DJI_Motor_Type_e;

/**
 * @brief typedef structure that contains the data for the Motor Device.
 */
typedef struct
{
	bool Initlized;		  /*!< init flag */
	int16_t Current;	  /*!< Motor electric current */
	int16_t Velocity;	  /*!< Motor rotate velocity (RPM)*/
	int16_t Encoder;	  /*!< Motor encoder angle */
	int16_t Last_Encoder; /*!< previous Motor encoder angle */
	float Angle;		  /*!< Motor angle in degree */
	uint8_t Temperature;  /*!< Motor Temperature */

} DJI_Motor_Data_Typedef;

/**
 * @brief typedef structure that contains the information for the Damiao Motor Device.
 */
typedef struct
{
	DJI_Motor_Type_e Type;				   /*!< Type of Motor */
	Motor_CANFrameInfo_typedef FDCANFrame; /*!< information for the CAN Transfer */
	DJI_Motor_Data_Typedef Data;		   /*!< information for the Motor Device */
} DJI_Motor_Info_Typedef;

/*********************** typedefs for LK Motor Control ***********************/

#define LK_STDID_OFFESET ((uint16_t)0x140)

#define LK_TORQUE_COEFFICIENT 0.32f					// (N*m/A)转矩系数
#define LK_CURRENT_TO_MF_CONTROL 124.1212121212121f // (2048/16.5)(1/A)电流转换为控制量

#define LK_MAX_MF_CONTROL_IQ 2048
#define LK_MIN_MF_CONTROL_IQ -2048
#define LK_MAX_MF_CONTROL_CURRENT 16.5f
#define LK_MIN_MF_CONTROL_CURRENT -16.5f

#define LK_MAX_MF_TORQUE 2.41f
#define LK_MIN_MF_TORQUE -2.41f

#define LK_MF_CONTROL_TO_CURRENT 0.008056640625f // (16.5/2048)(A)控制量转换为电流
#define LK_RAD_TO_DEGREE 57.2957795131f			 // (180/pi) (rad)->(degree)
#define LK_DEGREE_TO_RAD 0.0174532925f			 // (pi/180) (degree)->(rad)
/**
 * @brief  typedef enum that control mode the type of LK_Motor.
 */
typedef enum
{
	LK_TORQUE,
	LK_VELOCITY,
} LK_Motor_Control_Mode_Type_e;

/**
 * @brief  typedef enum that CMD of LK_Motor .
 */
typedef enum
{
	LK_Motor_Enable,
	LK_Motor_Disable,
	LK_Motor_Stop,
	LK_Motor_CMD_Type_Num,
} LK_Motor_CMD_Type_e;

/**
 * @brief typedef structure that contains the data for the LK Motor Device.
 */
typedef struct
{
	float Position;	   /*!< Motor Positon  */
	float Velocity;	   /*!< Motor Velocity */
	float Current;	   /*!< Motor Current   */
	float Temperature; /*!< Motor Temperature   */

	int8_t temprature; /*!< Motor temprature   */
	int16_t iq;		   /*< Motor current control value  */
	int16_t speed;	   /*< Motor speed control value  */
	uint16_t encoder;  /*< Motor encoder angle  */

} LK_Motor_Data_Typedef;

/**
 * @brief typedef structure that contains the information for the Damiao Motor Device.
 */
typedef struct
{
	LK_Motor_Control_Mode_Type_e Control_Mode;
	Motor_CANFrameInfo_typedef FDCANFrame; /*!< information for the CAN Transfer */
	LK_Motor_Data_Typedef Data;			   /*!< information for the Motor Device */

	uint32_t last_fdb_time; // 上次反馈时间
	bool offline;			// 电机是否离线
} LK_Motor_Info_Typedef;

/* Externs ------------------------------------------------------------------*/
extern LK_Motor_Info_Typedef LK_9025_Motor[2]; // 2个轮毂电机9025

extern DM_Motor_Info_Typedef DM_8009_Motor[4]; // 4个关节电机8009p

extern void DJI_Motor_Info_Update(uint32_t *Identifier, uint8_t *Rx_Buf, DJI_Motor_Info_Typedef *DJI_Motor);

extern void DM_Motor_Info_Update(uint32_t *Identifier, uint8_t *Rx_Buf, DM_Motor_Info_Typedef *DM_Motor);

extern void DM_Motor_Command(FDCAN_TxFrame_TypeDef *FDCAN_TxFrame, DM_Motor_Info_Typedef *DM_Motor, uint8_t CMD);

extern void DM_Motor_CAN_TxMessage(FDCAN_TxFrame_TypeDef *FDCAN_TxFrame, DM_Motor_Info_Typedef *DM_Motor, float Postion, float Velocity, float KP, float KD, float Torque);

extern void LK_Motor_Info_Update(uint32_t *Identifier, uint8_t *Rx_Buf, LK_Motor_Info_Typedef *LK_Motor);

extern void LK_Motor_Command(FDCAN_TxFrame_TypeDef *FDCAN_TxFrame, LK_Motor_Info_Typedef *LK_Motor, uint8_t CMD);

extern void LK_Motor_CAN_TxMessage(FDCAN_TxFrame_TypeDef *FDCAN_TxFrame, LK_Motor_Info_Typedef *LK_Motor, float Velocity, float Torque);

extern bool ScanOfflineMotor(void);

extern fp32 fp32_constrain(fp32 Value, fp32 minValue, fp32 maxValue);

#endif // DEVICE_MOTOR_H
