#include "tim.h"

TIM_HandleTypeDef htim5;
TIM_HandleTypeDef htim6;

static uint32_t pace_apb1_timer_prescaler_for_1mhz(void)
{
    RCC_ClkInitTypeDef clock_config = {0};
    uint32_t flash_latency;
    uint32_t timer_clock;
    HAL_RCC_GetClockConfig(&clock_config, &flash_latency);
    timer_clock = HAL_RCC_GetPCLK1Freq();
    if (clock_config.APB1CLKDivider != RCC_HCLK_DIV1)
    {
        timer_clock *= 2U;
    }
    return (timer_clock / 1000000U) - 1U;
}

void MX_TIM5_Init(void)
{
    htim5.Instance = TIM5;
    htim5.Init.Prescaler = pace_apb1_timer_prescaler_for_1mhz();
    htim5.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim5.Init.Period = 0xFFFFFFFFUL;
    htim5.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim5.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    if (HAL_TIM_Base_Init(&htim5) != HAL_OK)
    {
        Error_Handler();
    }
}

void MX_TIM6_Init(void)
{
    TIM_MasterConfigTypeDef master = {0};
    htim6.Instance = TIM6;
    htim6.Init.Prescaler = pace_apb1_timer_prescaler_for_1mhz();
    htim6.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim6.Init.Period = 1999U;
    htim6.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    if (HAL_TIM_Base_Init(&htim6) != HAL_OK)
    {
        Error_Handler();
    }
    master.MasterOutputTrigger = TIM_TRGO_RESET;
    master.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
    if (HAL_TIMEx_MasterConfigSynchronization(&htim6, &master) != HAL_OK)
    {
        Error_Handler();
    }
}

void HAL_TIM_Base_MspInit(TIM_HandleTypeDef *tim_handle)
{
    if (tim_handle->Instance == TIM5)
    {
        __HAL_RCC_TIM5_CLK_ENABLE();
    }
    else if (tim_handle->Instance == TIM6)
    {
        __HAL_RCC_TIM6_CLK_ENABLE();
        HAL_NVIC_SetPriority(TIM6_DAC_IRQn, 5U, 0U);
        HAL_NVIC_EnableIRQ(TIM6_DAC_IRQn);
    }
}

void HAL_TIM_Base_MspDeInit(TIM_HandleTypeDef *tim_handle)
{
    if (tim_handle->Instance == TIM5)
    {
        __HAL_RCC_TIM5_CLK_DISABLE();
    }
    else if (tim_handle->Instance == TIM6)
    {
        __HAL_RCC_TIM6_CLK_DISABLE();
        HAL_NVIC_DisableIRQ(TIM6_DAC_IRQn);
    }
}
