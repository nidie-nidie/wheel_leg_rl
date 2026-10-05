#ifndef __GIMBAL_TASK_H__
#define __GIMBAL_TASK_H__

#include "struct_typedef.h"
#include "CAN_receive.h"
#include "pid.h"
#include "remote_control.h"

void gimbal_task(void const *pvParameters);

typedef struct
{
  const MIT_Motor_t *gimbal_motor_measure;
  float set_Tp;
} gimbal_motor_t;

typedef struct
{
  gimbal_motor_t gimbal_motor;
  gimbal_PID_t gimbal_motor_absolute_angle_pid;		//陀螺角做反馈的双环角度PID
  gimbal_PID_t gimbal_motor_auto_angle_pid;				//自瞄双环角度PID
  gimbal_PID_t gimbal_motor_relative_angle_pid;		//电机角做反馈的双环角度PID
  pid_type_def gimbal_motor_relative_speed_pid;		//电机角做反馈的双环速度PID
  pid_type_def gimbal_motor_absolute_speed_pid;		//陀螺角做反馈的双环速度PID
  pid_type_def gimbal_motor_auto_speed_pid;				//自瞄双环速度PID
	
  float offset_angle;				// rad
  float max_relative_angle; // rad
  float min_relative_angle; // rad
	
  float relative_angle;     // rad
  float relative_angle_set; // rad
  float absolute_angle;     // rad
  float absolute_angle_set; // rad
  float relative_speed;     // rad/s
  float relative_speed_set; // rad/s
  float absolute_speed;     // rad/s
  float absolute_speed_set; // rad/s
} gimbal_control_t;

typedef struct
{
	float yaw;							// rad
	float pitch;            // rad
	float roll;             // rad
	float wx;								// rad/s
	float wy;               // rad/s
	float wz;               // rad/s
	float ax;               // m2/s
	float ay;               // m2/s
	float az;								// m2/s
	const float *INS_Angle;	//传参指针
	const float *INS_Gyro;	//传参指针
	const float *INS_Accel;	//传参指针
}ins_data_t;

typedef enum
{
  GIMBAL_ZERO_FORCE = 0,        //云台无力				
  GIMBAL_ABSOLUTE_ANGLE,				//云台由绝对角控制（陀螺仪）
  GIMBAL_AUTO,									//云台由上位机控制
  GIMBAL_RETURN,								//云台回正
	GIMBAL_OPERATION,							//云台操作手模式
} gimbal_behaviour_e;

typedef struct
{
	//上发下
	float vx_set;										//1
	float vy_set;										//1
	uint8_t mode;										//1，跳跃flag、爬坡flag等
	float roll_set;									//1
	uint8_t legL_mode;							//1
	float relative_angle_yaw;				//3
	
	uint16_t buffer_energy_chassis;	//2,底盘缓冲能量
	uint16_t chassis_power_limit;		//2,底盘功率上限
	float LeftPhi0Set;							//1
	float LeftLengthSet;						//1
	float RightPhi0Set;							//1
	float RightLengthSet;						//1
	//下发上
	float LeftPhi0;									//1
	float LeftLength;								//1
	float RightPhi0;								//1
	float RightLength;							//1
	uint8_t ChassisState;						//1
	uint8_t last_mode;		
}TwoBoardControlGimbal_t;

typedef struct
{
	float LegAngleLeft;
	float LegAngleRight;
	uint32_t LeftX1;
	uint32_t LeftY1;
	uint32_t LeftX2;
	uint32_t LeftY2;
	uint32_t RightX1;
	uint32_t RightY1;
	uint32_t RightX2;
	uint32_t RightY2;
}dynamicUI_t;


typedef struct
{
  gimbal_control_t gimbal_yaw_motor;
  gimbal_control_t gimbal_pitch_motor;
	ins_data_t ins_data;
	gimbal_behaviour_e gimbal_behaviour;
	gimbal_behaviour_e last_gimbal_behaviour;
	TwoBoardControlGimbal_t TwoBoardControlGimbal;//双板通信，控制底盘只是设置一个值，所以挂靠在了云台线程上
	dynamicUI_t dynamicUI;
} gimbal_move_t;

#endif
