/**
 ******************************************************************************
 * @file           : CAN_Task.c
 * @brief          : CAN task
 * @author         : GrassFam Wang
 * @date           : 2025/1/22
 * @version        : v1.1
 ******************************************************************************
 * @attention      : None
 ******************************************************************************
 */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "cmsis_os.h"
#include "CAN_Task.h"
#include "Control_Task.h"
#include "INS_Task.h"
#include "Motor.h"
#include "bsp_can.h"
#include "Remote_Control.h"
#include "Control_Task.h"

extern INS_t INS;

// debug variables

/* USER CODE BEGIN Header_CAN_Task */
/**
 * @brief Function implementing the StartCANTask thread.
 * @param argument: Not used
 * @retval None
 */
/* USER CODE END Header_CAN_Task */

void CAN_Task(void)
{
	TickType_t CAN_Task_SysTick = 0;
	while (INS.ins_flag == 0)
	{
		osDelay(1);
	}
	

	for(int j = 0; j < 10; j++)
	{
		DM_Motor_Command(&FDCAN3_TxFrame, &DM_8009_Motor[0], DM_Motor_Enable);
		osDelay(1);
	}
	for(int j = 0; j < 10; j++)
	{
		DM_Motor_Command(&FDCAN3_TxFrame, &DM_8009_Motor[1], DM_Motor_Enable);
		osDelay(1);
	}
	for(int j = 0; j < 10; j++)
	{
		DM_Motor_Command(&FDCAN3_TxFrame, &DM_8009_Motor[2], DM_Motor_Enable);
		osDelay(1);
	}
	for(int j = 0; j < 10; j++)
	{
		DM_Motor_Command(&FDCAN3_TxFrame, &DM_8009_Motor[3], DM_Motor_Enable);
		osDelay(1);
	}
	for(int j = 0; j < 10; j++)
	{
		LK_Motor_Command(&FDCAN3_TxFrame, &LK_9025_Motor[0], LK_Motor_Enable);
		osDelay(1);
	}
	for(int j = 0; j < 10; j++)
	{
		LK_Motor_Command(&FDCAN3_TxFrame, &LK_9025_Motor[1], LK_Motor_Enable);
		osDelay(1);
	}
	for (;;)
	{
		// CAN-FD
		DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, &DM_8009_Motor[0], 0, 0, 0, 0, 0);
		DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, &DM_8009_Motor[1], 0, 0, 0, 0, 0);
		DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, &DM_8009_Motor[2], 0, 0, 0, 0, 0);
		DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, &DM_8009_Motor[3], 0, 0, 0, 0, 0);

		LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, &LK_9025_Motor[0], 0, 1);
		LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, &LK_9025_Motor[1], 0, 1);

		osDelay(1);
	}
}
