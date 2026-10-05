/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : bsp_gpio.c
 * @brief          : bsp gpio functions
 * @author         : GrassFan Wan
 * @date           : 2025/1/22
 * @version        : v1.0
 ******************************************************************************
 * @attention      : None
 ******************************************************************************
 */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "bsp_gpio.h"
#include "main.h"

/**
 * @brief  Configures the GPIO.
 * @param  None
 * @retval None
 */
void BSP_GPIO_Init(void)
{

    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_14, GPIO_PIN_SET); // Power_OUT2_ON
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_SET); // Power_OUT1_ON
}
