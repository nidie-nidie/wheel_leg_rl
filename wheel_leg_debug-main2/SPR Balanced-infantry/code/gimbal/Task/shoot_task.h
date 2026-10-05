#ifndef __shoot_TASK_H__
#define __shoot_TASK_H__

#include "struct_typedef.h"

#include "CAN_receive.h"
#include "gimbal_task.h"
#include "remote_control.h"
#include "user_lib.h"
#include "USART_receive.h"

#define LEFT_FRICTION_SPEED -630
#define RIGHT_FRICTION_SPEED 630
#define TRIGGER_SPEED 7*2													//注意拨盘旋转方向，方向看机械设计，反了就给个负号

#define BLOCK_TRIGGER_SPEED         -0.1						//这个极性和拨盘保持一致，改方向时需要对应的判断留意一下，应该也要改（有点屎，需要优化）
#define BLOCK_TIME                  150 			//3* 150 = 450ms,控制周期0.001s
#define REVERSE_TIME                150+150*2

typedef enum
{
	SHOOT_STOP = 0,				//停下
	SHOOT_READY,					//开摩擦轮
	SHOOT_BULLET,					//连发
	SHOOT_SINGLE_BULLET,	//单发
	SHOOT_AUTO,						//自瞄发射
} shoot_mode_e;

typedef struct
{
  const motor_measure_t *shoot_motor_measure;
  float set_Tp;
	int16_t set_current;
} shoot_motor_t;

typedef struct
{
  shoot_motor_t shoot_motor;
  pid_type_def shoot_motor_absolute_speed_pid;		//速度环PID
	float speed_set;
	float angle_set;
	float speed;
	float angle;
} shoot_control_t;

typedef struct
{
	shoot_control_t left_fricion_motor;
	shoot_control_t right_fricion_motor;
	shoot_control_t trigger_motor;
	shoot_mode_e shoot_mode;
	shoot_mode_e last_shoot_mode;
	uint16_t block_time;
	uint8_t single_fire;
	uint8_t single_fire_time;
	
	int16_t heat_detect_fire;
	uint8_t heat_detect_fire_time;
	uint16_t heat_timeout;
	int16_t current_heat;
	
	uint8_t FrictionSwitch;
	uint8_t LastFn1;
	uint16_t count;
	float bullet_speed;
	uint8_t BulletSpeedFlag;
} shoot_move_t;

void fire_task(void const *pvParameters);

#endif
