#ifndef CAN_RECEIVE_H
#define CAN_RECEIVE_H

#include "struct_typedef.h"


#define DM_4310_PMAX				3.14f
#define DM_4310_PMIN				-3.14f
#define DM_4310_VMAX				30.0f
#define DM_4310_VMIN				-30.0f
#define DM_4310_TMAX				10.0f
#define DM_4310_TMIN				-10.0f

typedef struct
{
  /*2006和3508协议*/
  int16_t ecd;
  int16_t speed_rpm;
  int16_t given_current;
  uint8_t temperate;
  int16_t last_ecd;
	/*减速比后的扭矩与转速*/
	float Torque;
	float speed_rads;
	/*记录旋转角度（减速比后）*/
	float angle; //int32型数据，存放累加ecd值用
	int32_t round;
	float last_angle;
} motor_measure_t;

typedef struct
{
  uint8_t Status;//状态
	
  float Position;//位置(rad)
  float Speed;//速度(rad/s)
  float Torque;//转矩(N·m)
}MIT_Motor_t;

typedef struct
{
	uint8_t errorCode;          //错误码
	float chassisPower;         //底盘当前功率
	uint16_t chassisPowerLimit; //返回的功率上限
	uint8_t capEnergy;          //电容能量
}SuperCap_t;



extern void CAN_cmd_yaw_init(void);
extern void CAN_cmd_yaw_clean(void);
extern void CAN_cmd_gimbal_yaw(float Tor);
extern void CAN_cmd_pitch_init(void);
extern void CAN_cmd_pitch_clean(void);
extern void CAN_cmd_gimbal_pitch(float Tor);
extern void CAN_cmd_shoot(int16_t shoot);
extern void CAN_cmd_fricion(int16_t left_friction, int16_t rigit_friction);
extern void CAN_cmd_chassis(void);
extern void CAN_cmd_chassis2(void);
extern void CAN_cmd_SuperPower(void);

const motor_measure_t *trigger_measure_point(void);
const motor_measure_t *left_motor_measure_point(void);
const motor_measure_t *right_motor_measure_point(void);
const MIT_Motor_t *yaw_motor_measure_point(void);
const MIT_Motor_t *pitch_motor_measure_point(void);

#endif
