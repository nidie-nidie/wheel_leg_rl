/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : Detect_Task.h
 * @brief          : Detect task
 * @author         : GrassFan Wang
 * @date           : 2025/01/22
 * @version        : v1.0
 ******************************************************************************
 * @attention      : None
 ******************************************************************************
 */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef DETECT_TASK_H
#define DETECT_TASK_H

/* Includes ------------------------------------------------------------------*/
#include "stdint.h"
#include <stdbool.h>

#define VBAT_WARNNING_VAL_6S 22.0f
#define VBAT_LOW_VAL_6S 21.0f

extern float VBAT_WARNNING_VAL;
extern float VBAT_LOW_VAL;
extern float vbus;
extern bool vbus_low_warning;

extern void Detect_Task(void);

#endif // DETECT_TASK_H
