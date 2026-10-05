#ifndef ARM_H
#define ARM_H

#include "pid.h"
#include "fdcan.h"
#include "DamiaoMotors.h"
#include "bsp_fdcan.h"
#include "M2006motors.h"
#include "M3508motors.h"
#include "robot_def.h"
#include "bsp_dwt.h"
#include "RC_protocol.h"


//extern dmmotor_t max_motor,min_motor,finesse_motor,pitch_motor;

extern M2006motor_t roll,trans;//roll和传送带

extern double height,roll_real;


void ARM_INIT(void);

void ARM_TASK(void);

//离线检测
void offline_check(void);

//挂起所有标志位
void set_all_start_flag(void);

void update_motor_timeout(void);

void Height_Calculation(void);//计算当前高度

//微调上升速度
void lift_speed_control_delta(ARM_CMD_data_t *arm_cmd,double delta_speed);

//高度控制的另外一种形式
void lift_height_cmd(double height_ref,volatile float *target_lift_speed);


 void lift_speed_control(double lift_speed_ref);
void Height_Calculation(void);
extern uint16_t rec_id1;
extern uint8_t rx_data1[8];
//测试任务
void TEST_TASK(void);
#define ROLL_OFFSET 35.07f//变速比
extern M2006motor_t roll;//抬升用的3508
extern float temp_roll;  // 保留变量值在每次进入此模式期间

extern M3508motor_t lift_motor;//抬升用的3508

void recieve_all_data(void);
#endif // ARM_H

