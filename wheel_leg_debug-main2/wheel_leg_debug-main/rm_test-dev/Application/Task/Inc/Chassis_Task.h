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

typedef struct
{
	volatile uint8_t active;
	volatile uint8_t done;
	volatile uint8_t fault; // 0=none, 1=timeout, 2=feedback stale, 3=over-torque
	volatile uint8_t reached_mask;        // bit0..3 correspond to J0..J3
	volatile uint8_t snapshot_valid_mask; // bit0=left, bit1=right
	float velocity_cmd[4];
	float torque_cmd[4];
	float start_joint_pos[4];
	uint32_t start_tick;
	uint32_t stop_candidate_tick[4];
	float snapshot_pitch[2];
	float snapshot_phi0[2];
	float snapshot_theta[2];
	float snapshot_L0[2];
	float snapshot_joint_pos[4];
} LegHardstopRefTest_s;

extern chassis_t chassis_move;
extern vmc_leg_t left;
extern vmc_leg_t right;

extern uint8_t left_flag;
extern uint8_t right_flag;

extern uint32_t jump_time_r;
extern uint32_t jump_time_l;

extern uint32_t CHASS_TIME;

extern Calibrate_s CALIBRATE;
extern LegHardstopRefTest_s leg_hardstop_ref_test;

extern PID_Info_TypeDef stand_up_pid;
extern PID_Info_TypeDef legl_pid;
extern PID_Info_TypeDef legr_pid;

extern PID_Info_TypeDef roll_pid;
extern PID_Info_TypeDef tp_pid;
extern PID_Info_TypeDef turn_pid;

extern float safe_debug_pitch_target;
extern float safe_debug_pitch_err;
extern float safe_debug_pitch_tp;
extern float safe_debug_left_theta_tp;
extern float safe_debug_right_theta_tp;
extern float safe_debug_left_tp_cmd;
extern float safe_debug_right_tp_cmd;
extern float safe_debug_phi0_bias;
extern float safe_debug_left_phi0_cmd;
extern float safe_debug_right_phi0_cmd;
extern float safe_debug_wheel_pitch_assist;
extern float safe_debug_phi0_trim;
extern float safe_debug_phi0_trim_gate;
extern float safe_debug_phi0_trim_enable;
extern float safe_debug_left_theta_eq;
extern float safe_debug_right_theta_eq;
extern float safe_debug_left_wheel_lqr_body;
extern float safe_debug_right_wheel_lqr_body;
extern float safe_debug_common_wheel_lqr_body;
extern float safe_debug_pitch_phi0_trim;
extern float safe_debug_leg_set_ramp;

// Joint mapping test command shared by the PS2 task, chassis tasks and UART7.
// side: 0=none, 1=left, 2=right. tp_cmd is signed virtual-leg Tp in N*m.
extern volatile uint8_t joint_map_test_side;
extern volatile float joint_map_test_tp_cmd;

extern uint32_t CHASS_FSM_TIME;
extern bool chass_is_calibrated;

extern void mySaturate(float *in, float min, float max);
extern void ChassisApplySafePitchTp(vmc_leg_t *leg, int8_t side_sign);
extern float ChassisCalcSafeWheelTorque(float wheel_velocity, float wheel_direction);

extern void UpdateCalibrateStatus(void);
extern void LegHardstopRefTest_Update(void);
extern void ChassisHandleException(void);
extern void ChassisSetMode(void);
extern void ChassisConsole(void);

extern void ConsoleCalibrate(void);
extern void ConsoleStandUp(void);

#endif
