#include "pace_time.h"

#include "tim.h"

extern TIM_HandleTypeDef htim2;

static volatile uint32_t pace_tick_count_value;
static pace_time_tick_callback_t pace_tick_callback;

bool pace_time_init(void)
{
    pace_tick_count_value = 0U;
    if (HAL_TIM_Base_Start(&htim5) != HAL_OK)
    {
        return false;
    }
    if (HAL_TIM_Base_Start_IT(&htim6) != HAL_OK)
    {
        return false;
    }
    return true;
}

uint32_t pace_time_now_us(void)
{
    return __HAL_TIM_GET_COUNTER(&htim5);
}

uint32_t pace_time_expand_fdcan_timestamp(uint16_t raw_timestamp, uint32_t observed_now_us)
{
    /* Compatibility entry point: FDCAN has an independent epoch. V2 uses
     * TIM5 ISR service time, not a fabricated hardware receive timestamp. */
    (void)raw_timestamp;
    return observed_now_us;
}

void pace_time_set_tick_callback(pace_time_tick_callback_t callback)
{
    pace_tick_callback = callback;
}

uint32_t pace_time_tick_count(void)
{
    return pace_tick_count_value;
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM2)
    {
        HAL_IncTick();
    }
    else if (htim->Instance == TIM6)
    {
        uint32_t now_us;
        pace_tick_count_value++;
        now_us = pace_time_now_us();
        if (pace_tick_callback != 0)
        {
            pace_tick_callback(now_us);
        }
    }
}
