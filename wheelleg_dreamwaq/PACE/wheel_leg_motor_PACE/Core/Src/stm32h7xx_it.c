#include "main.h"
#include "stm32h7xx_it.h"

#include "fdcan.h"
#include "tim.h"
#include "usart.h"

extern TIM_HandleTypeDef htim2;

void NMI_Handler(void)
{
    while (1)
    {
    }
}

void HardFault_Handler(void)
{
    while (1)
    {
    }
}

void MemManage_Handler(void)
{
    while (1)
    {
    }
}

void BusFault_Handler(void)
{
    while (1)
    {
    }
}

void UsageFault_Handler(void)
{
    while (1)
    {
    }
}

void DebugMon_Handler(void)
{
}

void DMA1_Stream3_IRQHandler(void)
{
    HAL_DMA_IRQHandler(&hdma_uart7_rx);
}

void DMA1_Stream4_IRQHandler(void)
{
    HAL_DMA_IRQHandler(&hdma_uart7_tx);
}

void FDCAN3_IT0_IRQHandler(void)
{
    HAL_FDCAN_IRQHandler(&hfdcan3);
}

void FDCAN3_IT1_IRQHandler(void)
{
    HAL_FDCAN_IRQHandler(&hfdcan3);
}

void TIM2_IRQHandler(void)
{
    HAL_TIM_IRQHandler(&htim2);
}

void TIM6_DAC_IRQHandler(void)
{
    HAL_TIM_IRQHandler(&htim6);
}

void UART7_IRQHandler(void)
{
    HAL_UART_IRQHandler(&huart7);
}
