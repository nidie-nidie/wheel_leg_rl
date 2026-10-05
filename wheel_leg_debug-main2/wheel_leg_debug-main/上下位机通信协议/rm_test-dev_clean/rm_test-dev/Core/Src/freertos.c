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
