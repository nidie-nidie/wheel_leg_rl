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

/* Includes ------------------------------------------------------------------*/
#include "cmsis_os.h"
#include "Detect_Task.h"
#include "Config.h"
#include "Remote_Control.h"
#include "PS2_Task.h"
#include "bsp_adc.h"

float VBAT_WARNNING_VAL;
float VBAT_LOW_VAL;
float vbus;
bool vbus_low_warning;

uint32_t DETECT_TIME = 10; // 检测任务运行周期10ms

/**
 * @note turn on:  800
 *       turn off: 4150
 */

/* USER CODE BEGIN Header_Detect_Task */
/**
 * @brief Function implementing the StartDetectTask thread.
 * @param argument: Not used
 * @retval None
 */
/* USER CODE END Header_Detect_Task */
void Detect_Task(void)
{
    /* USER CODE BEGIN Detect_Task */
    //  TickType_t systick = 0;
    VBAT_WARNNING_VAL = VBAT_WARNNING_VAL_6S;
    VBAT_LOW_VAL = VBAT_LOW_VAL_6S;
    vbus_low_warning = false;
    /* Infinite loop */
    for (;;)
    {
        // 监测电压
        vbus = USER_ADC_Voltage_Update();
        if (vbus < VBAT_WARNNING_VAL)
        {

            vbus_low_warning = true;
        }
        else
        {
            vbus_low_warning = false;
        }
        if (ENABLE_ALARM_RC_OFFLINE)
        {
            Remote_Message_Moniter(&remote_ctrl); // 监测遥控器在线状态
        }

        if (ENABLE_ALARM_PS2_OFFLINE)
        {
            if (GetPs2IdleTime() > PS2_IDLE_TIMEOUT)
            {
                ps2_lost = false; // ps2手柄离线
                // ps2_lost = true; // ps2手柄离线
            }
        }

        osDelay(DETECT_TIME);
    }
    /* USER CODE END Detect_Task */
}
