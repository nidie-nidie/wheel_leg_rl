/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : bsp_mcu.c
 * @brief          : MCU peripheral initialization functions
 * @author         : GrassFan Wang
 * @date           : 2025/01/22
 * @version        : v1.0
 ******************************************************************************
 * @attention      : none
 ******************************************************************************
 */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "bsp_mcu.h"
#include "bsp_gpio.h"
#include "bsp_can.h"
#include "bsp_pwm.h"
#include "bsp_uart.h"
#include "bsp_adc.h"
#include "BMI088driver.h"
#include "usb_device.h"

#include "spi.h"

/**
 * @brief Initializes the MCU.
 */
void MCU_Init(void)
{
	/* ----------------------- BSP Init ----------------------- */
	BSP_PWM_Init();
	BSP_GPIO_Init();
	BSP_FDCAN_Init();
	BSP_USART_Init();
	BSP_ADC_Init(); // 检测电源电压用
	MX_USB_DEVICE_Init();
	/* ----------------------- Device Init ----------------------- */
	/* BMI088初始化 */
	while (BMI088_init(&hspi2, 0) != BMI088_NO_ERROR)
	{
		;
	}
}
//------------------------------------------------------------------------------
