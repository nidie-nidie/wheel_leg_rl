#ifndef __CHASSIS_TASK_H
#define __CHASSIS_TASK_H

#include "main.h"
#include "PID.h"
#include "INS_task.h"
#include "Motor.h"
#include "robot_param.h"
#include "VMC_Calc.h"
#include "User_Lib.h"

typedef enum
{
	CHASSIS_OFF,			   // 底盘关闭
	CHASSIS_SAFE,			   // 底盘无力，所有控制量置0
	CHASSIS_STAND_UP,		   // 底盘起立，从倒地状态到站立状态的中间过程
	CHASSIS_CALIBRATE,		   // 底盘校准
	CHASSIS_FOLLOW_GIMBAL_YAW, // 底盘跟随云台（运动方向为云台坐标系方向，需进行坐标转换）
	CHASSIS_OFF_HOOK,		   // 底盘脱困模式
} ChassisMode_e;

typedef struct
{
	DM_Motor_Info_Typedef *joint_motor[4];
	LK_Motor_Info_Typedef *wheel_motor[2];

	ChassisMode_e mode; // 底盘模式
	uint8_t error_code; // 底盘错误代码

	float v_set;	// 一阶滤波后的期望速度，单位是m/s
	float target_v; // 遥控器控制的期望速度，单位是m/s
	float x_set;	// 期望位置，单位是m

	float turn_set;	   // 期望yaw轴弧度
	float roll_set;	   // 一阶滤波后的期望roll轴弧度
	float roll_target; // 遥控器控制的期望roll轴弧度
	// float roll_x;
	// float phi_set;
	// float theta_set;

	float leg_set; // 期望腿长，单位是m
	float last_leg_set;

	float v_filter; // 滤波后的车体速度，单位是m/s
	float x_filter; // 滤波后的车体位置，单位是m

	float myPithR;
	float myPithGyroR;
	float myPithL;
	float myPithGyroL;
	float roll;
	float total_yaw;
	float theta_err; // 两腿夹角误差

	float turn_T;  // yaw轴补偿
	float roll_f0; // roll轴补偿

	float leg_tp; // 防劈叉补偿

	uint8_t start_flag; // 启动标志

	uint8_t jump_flag;	// 右腿跳跃标志
	uint8_t jump_flag2; // 左腿跳跃标志

	uint8_t prejump_flag; // 预跳跃标志
	uint8_t recover_flag; // 一种情况下的倒地自起标志

} chassis_t;

typedef struct Calibrate
{
	float velocity[4];	   // 关节电机速度
	uint32_t stop_time[4]; // 停止时间
	bool reached[4];	   // 是否到达限位

	bool left_reached;  // 左边是否完成校准
	bool right_reached; // 右边是否完成校准


	bool calibrated;	   // 完成校准
	bool toggle;		   // 切换校准状态
} Calibrate_s;

extern chassis_t chassis_move;
extern vmc_leg_t left;
extern vmc_leg_t right;

extern uint8_t left_flag;
extern uint8_t right_flag;

extern uint32_t jump_time_r;
extern uint32_t jump_time_l;

extern uint32_t CHASS_TIME;

extern Calibrate_s CALIBRATE;

extern PID_Info_TypeDef stand_up_pid;
extern PID_Info_TypeDef legl_pid;
extern PID_Info_TypeDef legr_pid;

extern PID_Info_TypeDef roll_pid;
extern PID_Info_TypeDef tp_pid;
extern PID_Info_TypeDef turn_pid;

extern uint32_t CHASS_FSM_TIME;
extern bool chass_is_calibrated;

extern void mySaturate(float *in, float min, float max);

extern void UpdateCalibrateStatus(void);
extern void ChassisHandleException(void);
extern void ChassisSetMode(void);
extern void ChassisConsole(void);

extern void ConsoleCalibrate(void);
extern void ConsoleStandUp(void);

#endif