/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * File Name          : freertos.c
 * Description        : Code for freertos applications
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2024 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "main.h"
#include "cmsis_os.h"

#include "Config.h"

#include "Music_Task.h"
#include "Led_Flow_Task.h"
#include "Detect_Task.h"
#include "ChassisR_Task.h"
#include "ChassisL_Task.h"
#include "INS_Task.h"
#include "Observe_Task.h"
#include "PS2_Task.h"
#include "Remote_Task.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "SEGGER_SYSVIEW_FreeRTOS.h"
#include "usart.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN Variables */

/* USER CODE END Variables */
// ins task
osThreadId Start_INS_TaskHandle;
uint32_t Start_INS_TaskBuffer[512];
osStaticThreadDef_t Start_INS_TaskControlBlock;
// chassis R task
osThreadId Start_ChassisR_TaskHandle;
uint32_t Start_ChassisR_TaskBuffer[512];
osStaticThreadDef_t Start_ChassisR_TaskControlBlock;
// chassis L task
osThreadId Start_ChassisL_TaskHandle;
uint32_t Start_ChassisL_TaskBuffer[512];
osStaticThreadDef_t Start_ChassisL_TaskControlBlock;
// observe task
osThreadId Start_Observe_TaskHandle;
uint32_t Start_Observe_TaskBuffer[512];
osStaticThreadDef_t Start_Observe_TaskControlBlock;
// detect task
osThreadId Start_Detect_TaskHandle;
uint32_t Start_Detect_TaskBuffer[128];
osStaticThreadDef_t Start_Detect_TaskControlBlock;
// music task
osThreadId Start_Music_TaskHandle;
uint32_t Start_Music_TaskBuffer[128];
osStaticThreadDef_t Start_Music_TaskControlBlock;
// LED RGB flow task
osThreadId Start_Led_RGB_Flow_TaskHandle;
uint32_t Start_Led_RGB_Flow_TaskBuffer[128];
osStaticThreadDef_t Start_Led_RGB_Flow_TaskControlBlock;

#if (ENABLE_UART7_TELEMETRY)
// UART7 compact CSV telemetry task
osThreadId Start_UART7_Telemetry_TaskHandle;
uint32_t Start_UART7_Telemetry_TaskBuffer[512];
osStaticThreadDef_t Start_UART7_Telemetry_TaskControlBlock;
#endif

#if (ENABLE_ALARM_PS2_OFFLINE)
osThreadId Start_PS2_TaskHandle;
uint32_t Start_PS2_TaskBuffer[128];
osStaticThreadDef_t Start_PS2_TaskControlBlock;
#endif

#if (ENABLE_ALARM_RC_OFFLINE)
osThreadId Start_Remote_TaskHandle;
uint32_t Start_Remote_TaskBuffer[128];
osStaticThreadDef_t Start_Remote_TaskControlBlock;
#endif

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN FunctionPrototypes */

/* USER CODE END FunctionPrototypes */

void INS_TASK(void const *argument);
void ChassisR_TASK(void const *argument);
void ChassisL_TASK(void const *argument);
void Detect_TASK(void const *argument);
void Music_TASK(void const *argument);
void Observe_TASK(void const *argument);
void Led_RGB_Flow_TASK(void const *argument);
void PS2_TASK(void const *argument);
void Remote_TASK(void const *argument);
#if (ENABLE_UART7_TELEMETRY)
void UART7_Telemetry_TASK(void const *argument);
static int32_t UART7_Telemetry_Scale10000(float value);
static int32_t UART7_Telemetry_Abs32(int32_t value);
static int UART7_Telemetry_AppendFormat(char *buffer, size_t buffer_size, int offset, const char *format, ...);
static int UART7_Telemetry_AppendScaled(char *buffer, size_t buffer_size, int offset, int32_t value);
#endif

extern void MX_USB_DEVICE_Init(void);
void MX_FREERTOS_Init(void); /* (MISRA C 2004 rule 8.1) */

/* GetIdleTaskMemory prototype (linked to static allocation support) */
void vApplicationGetIdleTaskMemory(StaticTask_t **ppxIdleTaskTCBBuffer, StackType_t **ppxIdleTaskStackBuffer, uint32_t *pulIdleTaskStackSize);

/* USER CODE BEGIN GET_IDLE_TASK_MEMORY */
static StaticTask_t xIdleTaskTCBBuffer;
static StackType_t xIdleStack[configMINIMAL_STACK_SIZE];

void vApplicationGetIdleTaskMemory(StaticTask_t **ppxIdleTaskTCBBuffer, StackType_t **ppxIdleTaskStackBuffer, uint32_t *pulIdleTaskStackSize)
{
	*ppxIdleTaskTCBBuffer = &xIdleTaskTCBBuffer;
	*ppxIdleTaskStackBuffer = &xIdleStack[0];
	*pulIdleTaskStackSize = configMINIMAL_STACK_SIZE;
	/* place for user code */
}
/* USER CODE END GET_IDLE_TASK_MEMORY */

/**
 * @brief  FreeRTOS initialization
 * @param  None
 * @retval None
 */
void MX_FREERTOS_Init(void)
{
	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* USER CODE BEGIN RTOS_MUTEX */
	/* add mutexes, ... */
	/* USER CODE END RTOS_MUTEX */

	/* USER CODE BEGIN RTOS_SEMAPHORES */
	/* add semaphores, ... */
	/* USER CODE END RTOS_SEMAPHORES */

	/* USER CODE BEGIN RTOS_TIMERS */
	/* start timers, add new ones, ... */
	/* USER CODE END RTOS_TIMERS */

	/* USER CODE BEGIN RTOS_QUEUES */
	/* add queues, ... */
	/* USER CODE END RTOS_QUEUES */

	/* Create the thread(s) */
	/* definition and creation of Start_Detect_Task */
	osThreadStaticDef(Start_Detect_Task, Detect_TASK, osPriorityLow, 0, 128, Start_Detect_TaskBuffer, &Start_Detect_TaskControlBlock);
	Start_Detect_TaskHandle = osThreadCreate(osThread(Start_Detect_Task), NULL);

	/* definition and creation of Start_INS_Task */
	osThreadStaticDef(Start_INS_Task, INS_TASK, osPriorityRealtime, 0, 512, Start_INS_TaskBuffer, &Start_INS_TaskControlBlock);
	Start_INS_TaskHandle = osThreadCreate(osThread(Start_INS_Task), NULL);

	/* definition and creation of Start_ChassisR_Task */
	osThreadStaticDef(Start_ChassisR_Task, ChassisR_TASK, osPriorityAboveNormal, 0, 512, Start_ChassisR_TaskBuffer, &Start_ChassisR_TaskControlBlock);
	Start_ChassisR_TaskHandle = osThreadCreate(osThread(Start_ChassisR_Task), NULL);

	/* definition and creation of Start_ChassisL_Task */
	osThreadStaticDef(Start_ChassisL_Task, ChassisL_TASK, osPriorityAboveNormal, 0, 512, Start_ChassisL_TaskBuffer, &Start_ChassisL_TaskControlBlock);
	Start_ChassisL_TaskHandle = osThreadCreate(osThread(Start_ChassisL_Task), NULL);

	/* definition and creation of Start_Observe_Task */
	osThreadStaticDef(Start_Observe_Task, Observe_TASK, osPriorityHigh, 0, 512, Start_Observe_TaskBuffer, &Start_Observe_TaskControlBlock);
	Start_Observe_TaskHandle = osThreadCreate(osThread(Start_Observe_Task), NULL);

	/* definition and creation of Start_Music_Task */
	osThreadStaticDef(Start_Music_Task, Music_TASK, osPriorityBelowNormal, 0, 128, Start_Music_TaskBuffer, &Start_Music_TaskControlBlock);
	Start_Music_TaskHandle = osThreadCreate(osThread(Start_Music_Task), NULL);

	/* definition and creation of Start_Led_RGB_Flow_Task */
	osThreadStaticDef(Start_Led_RGB_Flow_Task, Led_RGB_Flow_TASK, osPriorityLow, 0, 128, Start_Led_RGB_Flow_TaskBuffer, &Start_Led_RGB_Flow_TaskControlBlock);
	Start_Led_RGB_Flow_TaskHandle = osThreadCreate(osThread(Start_Led_RGB_Flow_Task), NULL);

#if (ENABLE_UART7_TELEMETRY)
	/* definition and creation of Start_UART7_Telemetry_Task */
	osThreadStaticDef(Start_UART7_Telemetry_Task, UART7_Telemetry_TASK, osPriorityLow, 0, 512, Start_UART7_Telemetry_TaskBuffer, &Start_UART7_Telemetry_TaskControlBlock);
	Start_UART7_Telemetry_TaskHandle = osThreadCreate(osThread(Start_UART7_Telemetry_Task), NULL);
#endif

#if (ENABLE_ALARM_PS2_OFFLINE)
	/* definition and creation of Start_PS2_Task */
	osThreadStaticDef(Start_PS2_Task, PS2_TASK, osPriorityAboveNormal, 0, 128, Start_PS2_TaskBuffer, &Start_PS2_TaskControlBlock);
	Start_PS2_TaskHandle = osThreadCreate(osThread(Start_PS2_Task), NULL);
#endif

#if (ENABLE_ALARM_RC_OFFLINE)
	/* definition and creation of Start_Remote_Task */
	osThreadStaticDef(Start_Remote_Task, Remote_TASK, osPriorityAboveNormal, 0, 128, Start_Remote_TaskBuffer, &Start_Remote_TaskControlBlock);
	Start_Remote_TaskHandle = osThreadCreate(osThread(Start_Remote_Task), NULL);
#endif

	/* USER CODE BEGIN RTOS_THREADS */
	/* add threads, ... */
	/* USER CODE END RTOS_THREADS */
}

/* USER CODE BEGIN Header_INS_Task */
/**
 * @brief  Function implementing the StartINS thread.
 * @param  argument: Not used
 * @retval None
 */
/* USER CODE END Header_INS_Task */
void INS_TASK(void const *argument)
{
	/* init code for USB_DEVICE */
	MX_USB_DEVICE_Init();
	/* USER CODE BEGIN INS_Task */
	/* Infinite loop */
	for (;;)
	{
		INS_task();
	}
	/* USER CODE END INS_Task */
}

/* USER CODE BEGIN Header_ChassisR_Task */
/**
 * @brief  Function implementing the Start_ChassisR_Task thread.
 * @param  argument: Not used
 * @retval None
 */
/* USER CODE END Header_ChassisR_Task */
void ChassisR_TASK(void const *argument)
{
	/* USER CODE BEGIN ChassisR_Task */
	/* Infinite loop */
	for (;;)
	{
		ChassisR_task();
	}
	/* USER CODE END ChassisR_Task */
}

/* USER CODE BEGIN Header_ChassisL_Task */
/**
 * @brief  Function implementing the Start_ChassisL_Task thread.
 * @param  argument: Not used
 * @retval None
 */
/* USER CODE END Header_ChassisL_Task */
void ChassisL_TASK(void const *argument)
{
	/* USER CODE BEGIN ChassisL_Task */
	/* Infinite loop */
	for (;;)
	{
		ChassisL_task();
	}
	/* USER CODE END ChassisL_Task */
}

/* USER CODE BEGIN Header_Detect_Task */
/**
 * @brief Function implementing the Start_Detect_Task thread.
 * @param argument: Not used
 * @retval None
 */
/* USER CODE END Header_Detect_Task */
void Detect_TASK(void const *argument)
{
	/* USER CODE BEGIN Detect_Task */
	/* Infinite loop */
	for (;;)
	{
		Detect_Task();
	}
	/* USER CODE END Detect_Task */
}

/* USER CODE BEGIN Header_Music_Task */
/**
 * @brief Function implementing the Start_Music_Task thread.
 * @param argument: Not used
 * @retval None
 */
void Music_TASK(void const *argument)
{
	/* USER CODE BEGIN Music_Task */
	/* Infinite loop */
	for (;;)
	{
		Music_Task();
	}
	/* USER CODE END Music_Task */
}

/* USER CODE BEGIN Header_Led_RGB_Flow_Task */
/**
 * @brief Function implementing the Led_RGB_Flow_Task thread.
 * @param argument: Not used
 * @retval None
 */
void Led_RGB_Flow_TASK(void const *argument)
{
	/* USER CODE BEGIN Led_RGB_Flow_Task */
	/* Infinite loop */
	for (;;)
	{
		led_RGB_flow_task();
	}
	/* USER CODE END Led_RGB_Flow_Task */
}

/* USER CODE BEGIN Header_Observe_TASK */
/**
 * @brief Function implementing the Observe_TASK thread.
 * @param argument: Not used
 * @retval None
 */
/* USER CODE END Header_Observe_TASK */
void Observe_TASK(void const *argument)
{
	/* USER CODE BEGIN Observe_TASK */
	/* Infinite loop */
	for (;;)
	{
		Observe_task();
	}
	/* USER CODE END Observe_TASK */
}

#if (ENABLE_UART7_TELEMETRY)
static int32_t UART7_Telemetry_Scale10000(float value)
{
	if (value >= 0.0f)
	{
		return (int32_t)(value * 10000.0f + 0.5f);
	}

	return (int32_t)(value * 10000.0f - 0.5f);
}

static int32_t UART7_Telemetry_Abs32(int32_t value)
{
	return (value < 0) ? -value : value;
}

static int UART7_Telemetry_AppendFormat(char *buffer, size_t buffer_size, int offset, const char *format, ...)
{
	int written = 0;
	va_list args;

	if ((offset < 0) || ((size_t)offset >= buffer_size))
	{
		return offset;
	}

	va_start(args, format);
	written = vsnprintf(&buffer[offset], buffer_size - (size_t)offset, format, args);
	va_end(args);

	if (written < 0)
	{
		return written;
	}

	return offset + written;
}

static int UART7_Telemetry_AppendScaled(char *buffer, size_t buffer_size, int offset, int32_t value)
{
	return UART7_Telemetry_AppendFormat(buffer, buffer_size, offset,
										",%s%ld.%04ld",
										(value < 0) ? "-" : "",
										(long)(UART7_Telemetry_Abs32(value) / 10000),
										(long)(UART7_Telemetry_Abs32(value) % 10000));
}

void UART7_Telemetry_TASK(void const *argument)
{
#if (UART7_TELEMETRY_PROFILE == UART7_TELEMETRY_PROFILE_POLARITY_LQR)
	static const char csv_header[] =
		"tick_ms,mode,start_flag,vbus_v,"
		"wheel_l_speed_dps_raw,wheel_r_speed_dps_raw,wheel_l_vel_radps,wheel_r_vel_radps,"
		"wheel_l_iq,wheel_r_iq,wheel_l_age_ms,wheel_r_age_ms,"
		"obs_wr_radps,obs_wl_radps,obs_vrb_mps,obs_vlb_mps,obs_aver_v_mps,"
		"imu_acc_x_mps2,kf_v_mps,kf_a_mps2,x_filter_m,obs_age_ms,"
		"kf_k00,kf_k01,kf_k10,kf_k11,v_set_mps,x_set_m,pitch_rad,pitch_rate_radps,"
		"x_l0,x_l1,x_l2,x_l3,x_l4,x_l5,x_r0,x_r1,x_r2,x_r3,x_r4,x_r5,"
		"lqr_l_wheel_raw_nm,lqr_l_leg_raw_nm,lqr_r_wheel_raw_nm,lqr_r_leg_raw_nm,"
		"turn_T_nm,left_wheel_T_limited_nm,right_wheel_T_limited_nm,uart7_dma_drop_count\r\n";
	__attribute__((section(".AXI_SRAM"))) static char tx_line[768];
	static uint32_t uart7_dma_drop_count = 0U;

	(void)argument;
	osDelay(1000U);
	HAL_UART_Transmit(&huart7, (uint8_t *)csv_header, (uint16_t)(sizeof(csv_header) - 1U), 100U);

	for (;;)
	{
		uint32_t now = HAL_GetTick();
		uint32_t wheel_l_age_ms = now;
		uint32_t wheel_r_age_ms = now;
		uint32_t obs_age_ms = now - observe_diag.update_tick_ms;
		int16_t wheel_l_speed_dps_raw = 0;
		int16_t wheel_r_speed_dps_raw = 0;
		int16_t wheel_l_iq = 0;
		int16_t wheel_r_iq = 0;
		float wheel_l_vel_radps = 0.0f;
		float wheel_r_vel_radps = 0.0f;
		uint8_t i = 0U;
		int tx_len = 0;

		// Never overwrite the DMA buffer while UART7 is still transmitting.
		if (huart7.gState != HAL_UART_STATE_READY)
		{
			uart7_dma_drop_count++;
			osDelay(UART7_TELEMETRY_PERIOD_MS);
			continue;
		}

		if (chassis_move.wheel_motor[0] != 0)
		{
			wheel_l_speed_dps_raw = chassis_move.wheel_motor[0]->Data.speed;
			wheel_l_vel_radps = chassis_move.wheel_motor[0]->Data.Velocity;
			wheel_l_iq = chassis_move.wheel_motor[0]->Data.iq;
			wheel_l_age_ms = now - chassis_move.wheel_motor[0]->last_fdb_time;
		}
		if (chassis_move.wheel_motor[1] != 0)
		{
			wheel_r_speed_dps_raw = chassis_move.wheel_motor[1]->Data.speed;
			wheel_r_vel_radps = chassis_move.wheel_motor[1]->Data.Velocity;
			wheel_r_iq = chassis_move.wheel_motor[1]->Data.iq;
			wheel_r_age_ms = now - chassis_move.wheel_motor[1]->last_fdb_time;
		}

		tx_len = snprintf(tx_line, sizeof(tx_line), "%lu,%u,%u",
						  (unsigned long)now,
						  (unsigned int)chassis_move.mode,
						  (unsigned int)chassis_move.start_flag);
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(vbus));
		tx_len = UART7_Telemetry_AppendFormat(tx_line, sizeof(tx_line), tx_len, ",%d,%d",
											(int)wheel_l_speed_dps_raw, (int)wheel_r_speed_dps_raw);
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(wheel_l_vel_radps));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(wheel_r_vel_radps));
		tx_len = UART7_Telemetry_AppendFormat(tx_line, sizeof(tx_line), tx_len, ",%d,%d,%lu,%lu",
											(int)wheel_l_iq, (int)wheel_r_iq,
											(unsigned long)wheel_l_age_ms, (unsigned long)wheel_r_age_ms);

		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(observe_diag.wr_radps));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(observe_diag.wl_radps));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(observe_diag.vrb_mps));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(observe_diag.vlb_mps));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(observe_diag.aver_v_mps));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(observe_diag.accel_measure_mps2));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(observe_diag.filtered_v_mps));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(observe_diag.filtered_a_mps2));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(chassis_move.x_filter));
		tx_len = UART7_Telemetry_AppendFormat(tx_line, sizeof(tx_line), tx_len, ",%lu", (unsigned long)obs_age_ms);

		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len,
											  UART7_Telemetry_Scale10000(observe_diag.kalman_k[i]));
		}
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(chassis_move.v_set));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(chassis_move.x_set));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(INS.Pitch));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(INS.Gyro[1]));

		for (i = 0U; i < 6U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(x_l[i]));
		}
		for (i = 0U; i < 6U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(x_r[i]));
		}

		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(T_Tp_l[0]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(T_Tp_l[1]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(T_Tp_r[0]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(T_Tp_r[1]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(chassis_move.turn_T));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.wheel_T));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.wheel_T));
		tx_len = UART7_Telemetry_AppendFormat(tx_line, sizeof(tx_line), tx_len, ",%lu\r\n",
											(unsigned long)uart7_dma_drop_count);

		if ((tx_len > 0) && (tx_len < (int)sizeof(tx_line)))
		{
			if (HAL_UART_Transmit_DMA(&huart7, (uint8_t *)tx_line, (uint16_t)tx_len) != HAL_OK)
			{
				uart7_dma_drop_count++;
			}
		}
		else
		{
			uart7_dma_drop_count++;
		}

		osDelay(UART7_TELEMETRY_PERIOD_MS);
	}
#elif (UART7_TELEMETRY_PROFILE == UART7_TELEMETRY_PROFILE_THETA_CAL)
	static const char csv_header[] =
		"tick_ms,mode,start_flag,pitch_rad,pitch_rate_radps,"
		"left_phi0_rad,right_phi0_rad,left_phi0_cmd_rad,right_phi0_cmd_rad,"
		"left_theta_rad,right_theta_rad,left_dtheta_radps,right_dtheta_radps,"
		"left_L0_m,right_L0_m,leg_set_m,"
		"j0_pos_rad,j1_pos_rad,j2_pos_rad,j3_pos_rad,"
		"j0_vel_radps,j1_vel_radps,j2_vel_radps,j3_vel_radps,"
		"j0_torque_fdb_nm,j1_torque_fdb_nm,j2_torque_fdb_nm,j3_torque_fdb_nm,"
		"j0_pos_set_rad,j1_pos_set_rad,j2_pos_set_rad,j3_pos_set_rad,"
		"j0_pos_err_rad,j1_pos_err_rad,j2_pos_err_rad,j3_pos_err_rad,"
		"left_F0_N,right_F0_N,left_Tp_nm,right_Tp_nm,"
		"lqr_l_wheel_raw_nm,lqr_l_leg_raw_nm,lqr_r_wheel_raw_nm,lqr_r_leg_raw_nm,"
		"left_wheel_T_limited_nm,right_wheel_T_limited_nm,"
		"hardstop_test_enable,hardstop_active,hardstop_done,hardstop_fault,"
		"hardstop_reached_mask,hardstop_snapshot_valid_mask,"
		"hardstop_vcmd_j0,hardstop_vcmd_j1,hardstop_vcmd_j2,hardstop_vcmd_j3,"
		"hardstop_tcmd_j0_nm,hardstop_tcmd_j1_nm,hardstop_tcmd_j2_nm,hardstop_tcmd_j3_nm,"
		"hardstop_travel_j0_rad,hardstop_travel_j1_rad,hardstop_travel_j2_rad,hardstop_travel_j3_rad,"
		"hardstop_snap_pitch_l,hardstop_snap_pitch_r,"
		"hardstop_snap_phi0_l,hardstop_snap_phi0_r,"
		"hardstop_snap_theta_l,hardstop_snap_theta_r,"
		"hardstop_snap_L0_l,hardstop_snap_L0_r,"
		"hardstop_snap_j0,hardstop_snap_j1,hardstop_snap_j2,hardstop_snap_j3,"
		"joint_max_age_ms,uart7_dma_drop_count\r\n";
	__attribute__((section(".AXI_SRAM"))) static char tx_line[1280];
	static uint32_t uart7_dma_drop_count = 0U;

	(void)argument;
	osDelay(1000U);
	HAL_UART_Transmit(&huart7, (uint8_t *)csv_header, (uint16_t)(sizeof(csv_header) - 1U), 100U);

	for (;;)
	{
		uint32_t now = HAL_GetTick();
		uint32_t joint_max_age_ms = 0U;
		float joint_pos_rad[4] = {0.0f};
		float joint_vel_radps[4] = {0.0f};
		float joint_torque_nm[4] = {0.0f};
		float joint_pos_set_rad[4] = {
			left.position_set[0], left.position_set[1],
			right.position_set[0], right.position_set[1]};
		uint8_t i = 0U;
		int tx_len = 0;

		// DMA owns tx_line until the previous transfer completes. Dropping a
		// telemetry sample is safer than delaying any controller task.
		if (huart7.gState != HAL_UART_STATE_READY)
		{
			uart7_dma_drop_count++;
			osDelay(UART7_TELEMETRY_PERIOD_MS);
			continue;
		}

		for (i = 0U; i < 4U; i++)
		{
			if (chassis_move.joint_motor[i] != 0)
			{
				uint32_t age_ms = now - chassis_move.joint_motor[i]->last_fdb_time;
				joint_pos_rad[i] = chassis_move.joint_motor[i]->Data.Position;
				joint_vel_radps[i] = chassis_move.joint_motor[i]->Data.Velocity;
				joint_torque_nm[i] = chassis_move.joint_motor[i]->Data.Torque;
				if (age_ms > joint_max_age_ms)
				{
					joint_max_age_ms = age_ms;
				}
			}
			else
			{
				joint_max_age_ms = now;
			}
		}

		tx_len = snprintf(tx_line, sizeof(tx_line), "%lu,%u,%u",
					  (unsigned long)now,
					  (unsigned int)chassis_move.mode,
					  (unsigned int)chassis_move.start_flag);
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(INS.Pitch));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(INS.Gyro[1]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.phi0));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.phi0));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_left_phi0_cmd));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_right_phi0_cmd));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.theta));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.theta));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.d_theta));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.d_theta));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.L0));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.L0));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(chassis_move.leg_set));

		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(joint_pos_rad[i]));
		}
		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(joint_vel_radps[i]));
		}
		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(joint_torque_nm[i]));
		}
		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(joint_pos_set_rad[i]));
		}
		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len,
										  UART7_Telemetry_Scale10000(joint_pos_set_rad[i] - joint_pos_rad[i]));
		}

		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.F0));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.F0));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.Tp));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.Tp));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(T_Tp_l[0]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(T_Tp_l[1]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(T_Tp_r[0]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(T_Tp_r[1]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.wheel_T));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.wheel_T));
		tx_len = UART7_Telemetry_AppendFormat(tx_line, sizeof(tx_line), tx_len, ",%u,%u,%u,%u,%u,%u",
										(unsigned int)LEG_HARDSTOP_REF_TEST_ENABLE,
										(unsigned int)leg_hardstop_ref_test.active,
										(unsigned int)leg_hardstop_ref_test.done,
										(unsigned int)leg_hardstop_ref_test.fault,
										(unsigned int)leg_hardstop_ref_test.reached_mask,
										(unsigned int)leg_hardstop_ref_test.snapshot_valid_mask);
		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len,
										  UART7_Telemetry_Scale10000(leg_hardstop_ref_test.velocity_cmd[i]));
		}
		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len,
										  UART7_Telemetry_Scale10000(leg_hardstop_ref_test.torque_cmd[i]));
		}
		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len,
										  UART7_Telemetry_Scale10000(
											  joint_pos_rad[i] - leg_hardstop_ref_test.start_joint_pos[i]));
		}
		for (i = 0U; i < 2U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len,
										  UART7_Telemetry_Scale10000(leg_hardstop_ref_test.snapshot_pitch[i]));
		}
		for (i = 0U; i < 2U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len,
										  UART7_Telemetry_Scale10000(leg_hardstop_ref_test.snapshot_phi0[i]));
		}
		for (i = 0U; i < 2U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len,
										  UART7_Telemetry_Scale10000(leg_hardstop_ref_test.snapshot_theta[i]));
		}
		for (i = 0U; i < 2U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len,
										  UART7_Telemetry_Scale10000(leg_hardstop_ref_test.snapshot_L0[i]));
		}
		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len,
										  UART7_Telemetry_Scale10000(leg_hardstop_ref_test.snapshot_joint_pos[i]));
		}
		tx_len = UART7_Telemetry_AppendFormat(tx_line, sizeof(tx_line), tx_len, ",%lu,%lu\r\n",
										(unsigned long)joint_max_age_ms,
										(unsigned long)uart7_dma_drop_count);

		if ((tx_len > 0) && (tx_len < (int)sizeof(tx_line)))
		{
			if (HAL_UART_Transmit_DMA(&huart7, (uint8_t *)tx_line, (uint16_t)tx_len) != HAL_OK)
			{
				uart7_dma_drop_count++;
			}
		}
		else
		{
			uart7_dma_drop_count++;
		}

		osDelay(UART7_TELEMETRY_PERIOD_MS);
	}
#else
	static const char csv_header[] =
		"tick_ms,mode,start_flag,vbus_v,pitch_rad,pitch_rate_radps,x_filter_m,v_filter_mps,"
		"j0_pos_rad,j1_pos_rad,j2_pos_rad,j3_pos_rad,"
		"j0_vel_radps,j1_vel_radps,j2_vel_radps,j3_vel_radps,"
		"j0_torque_fdb_nm,j1_torque_fdb_nm,j2_torque_fdb_nm,j3_torque_fdb_nm,joint_max_age_ms,"
		"wheel_l_vel_radps,wheel_r_vel_radps,wheel_l_iq,wheel_r_iq,wheel_max_age_ms,"
		"left_L0_m,right_L0_m,left_theta_rad,right_theta_rad,left_dtheta_radps,right_dtheta_radps,"
		"left_phi0_rad,right_phi0_rad,left_torque_set0_nm,left_torque_set1_nm,"
		"right_torque_set0_nm,right_torque_set1_nm,left_wheel_T_nm,right_wheel_T_nm,"
		"safe_wheel_pitch_assist_nm,"
		"safe_pitch_target_rad,safe_pitch_err_rad,safe_pitch_tp_nm,"
		"safe_left_theta_tp_nm,safe_right_theta_tp_nm,safe_left_tp_cmd_nm,safe_right_tp_cmd_nm,"
		"j0_pos_set_rad,j1_pos_set_rad,j2_pos_set_rad,j3_pos_set_rad,"
		"j0_pos_err_rad,j1_pos_err_rad,j2_pos_err_rad,j3_pos_err_rad,"
		"safe_phi0_bias_rad,safe_left_phi0_cmd_rad,safe_right_phi0_cmd_rad,"
		"safe_phi0_trim_rad,safe_phi0_trim_gate,safe_phi0_trim_enable,"
		"safe_left_theta_eq_rad,safe_right_theta_eq_rad,"
		"safe_left_wheel_lqr_body_nm,safe_right_wheel_lqr_body_nm,safe_common_wheel_lqr_body_nm,"
		"safe_pitch_phi0_trim_rad,safe_leg_set_ramp_m,"
		"joint_map_test_enable,joint_map_test_side,joint_map_test_tp_cmd_nm,"
		"left_F0_N,right_F0_N,left_Tp_nm,right_Tp_nm,"
		"j0_tx_torque_nm,j1_tx_torque_nm,j2_tx_torque_nm,j3_tx_torque_nm,leg_set_m\r\n";
	static char tx_line[1280];

	(void)argument;
	osDelay(1000U);
	HAL_UART_Transmit(&huart7, (uint8_t *)csv_header, (uint16_t)(sizeof(csv_header) - 1U), 100U);

	for (;;)
	{
		uint32_t now = HAL_GetTick();
		uint8_t i = 0U;
		int32_t joint_pos_rad[4] = {0};
		int32_t joint_vel_radps[4] = {0};
		int32_t joint_torque_nm[4] = {0};
		int32_t joint_pos_set_rad[4] = {0};
		int32_t joint_pos_err_rad[4] = {0};
		uint32_t joint_age_ms[4] = {now, now, now, now};
		uint32_t joint_max_age_ms = 0U;
		int32_t wheel_l_radps = 0;
		int32_t wheel_r_radps = 0;
		int16_t wheel_l_iq = 0;
		int16_t wheel_r_iq = 0;
		uint32_t wheel_l_age_ms = now;
		uint32_t wheel_r_age_ms = now;
		uint32_t wheel_max_age_ms = 0U;
		int tx_len = 0;
		uint16_t send_len = 0U;

		if (chassis_move.wheel_motor[0] != 0)
		{
			wheel_l_radps = UART7_Telemetry_Scale10000(chassis_move.wheel_motor[0]->Data.Velocity);
			wheel_l_iq = chassis_move.wheel_motor[0]->Data.iq;
			wheel_l_age_ms = now - chassis_move.wheel_motor[0]->last_fdb_time;
		}
		if (chassis_move.wheel_motor[1] != 0)
		{
			wheel_r_radps = UART7_Telemetry_Scale10000(chassis_move.wheel_motor[1]->Data.Velocity);
			wheel_r_iq = chassis_move.wheel_motor[1]->Data.iq;
			wheel_r_age_ms = now - chassis_move.wheel_motor[1]->last_fdb_time;
		}

		for (i = 0U; i < 4U; i++)
		{
			if (chassis_move.joint_motor[i] != 0)
			{
				joint_pos_rad[i] = UART7_Telemetry_Scale10000(chassis_move.joint_motor[i]->Data.Position);
				joint_vel_radps[i] = UART7_Telemetry_Scale10000(chassis_move.joint_motor[i]->Data.Velocity);
				joint_torque_nm[i] = UART7_Telemetry_Scale10000(chassis_move.joint_motor[i]->Data.Torque);
				joint_age_ms[i] = now - chassis_move.joint_motor[i]->last_fdb_time;
			}
			if (joint_age_ms[i] > joint_max_age_ms)
			{
				joint_max_age_ms = joint_age_ms[i];
			}
		}
		joint_pos_set_rad[0] = UART7_Telemetry_Scale10000(left.position_set[0]);
		joint_pos_set_rad[1] = UART7_Telemetry_Scale10000(left.position_set[1]);
		joint_pos_set_rad[2] = UART7_Telemetry_Scale10000(right.position_set[0]);
		joint_pos_set_rad[3] = UART7_Telemetry_Scale10000(right.position_set[1]);
		for (i = 0U; i < 4U; i++)
		{
			if (chassis_move.joint_motor[i] != 0)
			{
				joint_pos_err_rad[i] =
					UART7_Telemetry_Scale10000(((float)joint_pos_set_rad[i] / 10000.0f) - chassis_move.joint_motor[i]->Data.Position);
			}
		}
		wheel_max_age_ms = (wheel_l_age_ms > wheel_r_age_ms) ? wheel_l_age_ms : wheel_r_age_ms;

		tx_len = snprintf(tx_line, sizeof(tx_line), "%lu,%u,%u",
						  (unsigned long)now,
						  (unsigned int)chassis_move.mode,
						  (unsigned int)chassis_move.start_flag);
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(vbus));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(INS.Pitch));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(INS.Gyro[1]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(chassis_move.x_filter));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(chassis_move.v_filter));

		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, joint_pos_rad[i]);
		}
		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, joint_vel_radps[i]);
		}
		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, joint_torque_nm[i]);
		}
		tx_len = UART7_Telemetry_AppendFormat(tx_line, sizeof(tx_line), tx_len, ",%lu",
											(unsigned long)joint_max_age_ms);
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, wheel_l_radps);
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, wheel_r_radps);
		tx_len = UART7_Telemetry_AppendFormat(tx_line, sizeof(tx_line), tx_len, ",%d,%d,%lu",
											(int)wheel_l_iq,
											(int)wheel_r_iq,
											(unsigned long)wheel_max_age_ms);
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.L0));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.L0));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.theta));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.theta));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.d_theta));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.d_theta));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.phi0));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.phi0));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.torque_set[0]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.torque_set[1]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.torque_set[0]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.torque_set[1]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.wheel_T));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.wheel_T));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_wheel_pitch_assist));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_pitch_target));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_pitch_err));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_pitch_tp));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_left_theta_tp));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_right_theta_tp));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_left_tp_cmd));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_right_tp_cmd));
		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, joint_pos_set_rad[i]);
		}
		for (i = 0U; i < 4U; i++)
		{
			tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, joint_pos_err_rad[i]);
		}
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_phi0_bias));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_left_phi0_cmd));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_right_phi0_cmd));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_phi0_trim));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_phi0_trim_gate));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_phi0_trim_enable));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_left_theta_eq));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_right_theta_eq));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_left_wheel_lqr_body));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_right_wheel_lqr_body));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_common_wheel_lqr_body));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_pitch_phi0_trim));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(safe_debug_leg_set_ramp));
		tx_len = UART7_Telemetry_AppendFormat(tx_line, sizeof(tx_line), tx_len, ",%u,%u",
										(unsigned int)JOINT_MAP_TEST_ENABLE,
										(unsigned int)joint_map_test_side);
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len,
										UART7_Telemetry_Scale10000(joint_map_test_tp_cmd));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.F0));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.F0));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(left.Tp));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(right.Tp));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(-left.torque_set[1]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(-left.torque_set[0]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(-right.torque_set[0]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(-right.torque_set[1]));
		tx_len = UART7_Telemetry_AppendScaled(tx_line, sizeof(tx_line), tx_len, UART7_Telemetry_Scale10000(chassis_move.leg_set));
		tx_len = UART7_Telemetry_AppendFormat(tx_line, sizeof(tx_line), tx_len, "\r\n");

		if (tx_len > 0)
		{
			send_len = (tx_len < (int)sizeof(tx_line)) ? (uint16_t)tx_len : (uint16_t)(sizeof(tx_line) - 1U);
			HAL_UART_Transmit(&huart7, (uint8_t *)tx_line, send_len, 100U);
		}

		osDelay(UART7_TELEMETRY_PERIOD_MS);
	}
#endif
}
#endif

/* USER CODE BEGIN Header_PS2_Task */
/**
 * @brief Function implementing the PS2_TASK thread.
 * @param argument: Not used
 * @retval None
 */
/* USER CODE END Header_PS2_Task */
void PS2_TASK(void const *argument)
{
	/* USER CODE BEGIN PS2_Task */
	/* Infinite loop */
	for (;;)
	{
		pstwo_task();
	}
	/* USER CODE END PS2_Task */
}

/* USER CODE BEGIN Header_Remote_Task */
/**
 * @brief Function implementing the Remote_TASK thread.
 * @param argument: Not used
 * @retval None
 */
/* USER CODE END Header_Remote_Task */
void Remote_TASK(void const *argument)
{
	/* USER CODE BEGIN Remote_Task */
	/* Infinite loop */
	for (;;)
	{
		Remote_task();
	}
	/* USER CODE END Remote_Task */
}

/* Private application code --------------------------------------------------*/
/* USER CODE BEGIN Application */

/* USER CODE END Application */
