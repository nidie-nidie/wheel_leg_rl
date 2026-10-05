#ifndef VIDEO_H
#define VIDEO_H


#include "usart.h"

#include "robot_def.h"
#include "bsp_dwt.h"
#include "RC_protocol.h"
#include "robot_def.h"
#include "robot_cmd.h"
#include "tim.h"

// 用于遥控器数据读取,遥控器数据是一个大小为2的数组
#define LAST 1
#define TEMP 0

#define RE_RX_BUFFER_SIZE 255u 

extern uint8_t uart7_rx_buff[RE_RX_BUFFER_SIZE];
extern uint8_t uart10_rx_buff[255];
void VIDEO_INIT(void);


void VIDEO_TASK(void);//控制舵机
#endif // ARM_H
