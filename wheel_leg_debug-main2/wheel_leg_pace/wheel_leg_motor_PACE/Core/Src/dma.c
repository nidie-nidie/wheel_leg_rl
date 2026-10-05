#include "dma.h"

void MX_DMA_Init(void)
{
    __HAL_RCC_DMA1_CLK_ENABLE();
    HAL_NVIC_SetPriority(DMA1_Stream3_IRQn, 5U, 0U);
    HAL_NVIC_EnableIRQ(DMA1_Stream3_IRQn);
    HAL_NVIC_SetPriority(DMA1_Stream4_IRQn, 5U, 0U);
    HAL_NVIC_EnableIRQ(DMA1_Stream4_IRQn);
}
