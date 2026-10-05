#ifndef CMD_H
#define CMD_H
#include "robot_def.h"
#include "main.h"
#include "usart.h"
#include "RC_protocol.h"
#include "pid.h"
#include "kf_imu.h"
#include "usart.h"
#include "minipc_protocol.h"
#include "arm.h"
#include "scara_kinematics.h"
#include "chassis.h"
#include "bsp_fdcan.h"
#include "referee.h"
#include "cm_device.h"
#include "usbd_cdc_if.h"
#include "UI_task.h"

// uint16_t 类型的按键标志


// uint8_t 类型的鼠标按键标志

//#include "arm_math.h"
extern uint8_t rx_data2[8];
extern uint16_t rec_id2;
#define PI               3.14159265358979f
extern ARM_CMD_data_t ARM_CMD_data;
void ROBOT_CMD_INIT(void);

void ROBOT_CMD_TASK(void);

#endif // CMD_H
