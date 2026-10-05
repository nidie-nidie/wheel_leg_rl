#include "fdcan.h"

FDCAN_HandleTypeDef hfdcan3;

void MX_FDCAN3_Init(void)
{
    hfdcan3.Instance = FDCAN3;
    hfdcan3.Init.FrameFormat = FDCAN_FRAME_CLASSIC;
    hfdcan3.Init.Mode = FDCAN_MODE_NORMAL;
    hfdcan3.Init.AutoRetransmission = ENABLE;
    hfdcan3.Init.TransmitPause = DISABLE;
    hfdcan3.Init.ProtocolException = ENABLE;
    hfdcan3.Init.NominalPrescaler = 5U;
    hfdcan3.Init.NominalSyncJumpWidth = 5U;
    hfdcan3.Init.NominalTimeSeg1 = 14U;
    hfdcan3.Init.NominalTimeSeg2 = 5U;
    hfdcan3.Init.DataPrescaler = 5U;
    hfdcan3.Init.DataSyncJumpWidth = 5U;
    hfdcan3.Init.DataTimeSeg1 = 14U;
    hfdcan3.Init.DataTimeSeg2 = 5U;
    hfdcan3.Init.MessageRAMOffset = 0U;
    hfdcan3.Init.StdFiltersNbr = 1U;
    hfdcan3.Init.ExtFiltersNbr = 0U;
    hfdcan3.Init.RxFifo0ElmtsNbr = 16U;
    hfdcan3.Init.RxFifo0ElmtSize = FDCAN_DATA_BYTES_8;
    hfdcan3.Init.RxFifo1ElmtsNbr = 0U;
    hfdcan3.Init.RxFifo1ElmtSize = FDCAN_DATA_BYTES_8;
    hfdcan3.Init.RxBuffersNbr = 0U;
    hfdcan3.Init.RxBufferSize = FDCAN_DATA_BYTES_8;
    hfdcan3.Init.TxEventsNbr = 8U;
    hfdcan3.Init.TxBuffersNbr = 0U;
    hfdcan3.Init.TxFifoQueueElmtsNbr = 8U;
    hfdcan3.Init.TxFifoQueueMode = FDCAN_TX_FIFO_OPERATION;
    hfdcan3.Init.TxElmtSize = FDCAN_DATA_BYTES_8;
    if (HAL_FDCAN_Init(&hfdcan3) != HAL_OK)
    {
        Error_Handler();
    }
}

void HAL_FDCAN_MspInit(FDCAN_HandleTypeDef *fdcan_handle)
{
    GPIO_InitTypeDef gpio = {0};
    if (fdcan_handle->Instance != FDCAN3)
    {
        return;
    }

    __HAL_RCC_FDCAN_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    gpio.Pin = GPIO_PIN_12 | GPIO_PIN_13;
    gpio.Mode = GPIO_MODE_AF_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    gpio.Alternate = GPIO_AF5_FDCAN3;
    HAL_GPIO_Init(GPIOD, &gpio);

    HAL_NVIC_SetPriority(FDCAN3_IT0_IRQn, 5U, 0U);
    HAL_NVIC_EnableIRQ(FDCAN3_IT0_IRQn);
    HAL_NVIC_SetPriority(FDCAN3_IT1_IRQn, 5U, 0U);
    HAL_NVIC_EnableIRQ(FDCAN3_IT1_IRQn);
}

void HAL_FDCAN_MspDeInit(FDCAN_HandleTypeDef *fdcan_handle)
{
    if (fdcan_handle->Instance != FDCAN3)
    {
        return;
    }
    __HAL_RCC_FDCAN_CLK_DISABLE();
    HAL_GPIO_DeInit(GPIOD, GPIO_PIN_12 | GPIO_PIN_13);
    HAL_NVIC_DisableIRQ(FDCAN3_IT0_IRQn);
    HAL_NVIC_DisableIRQ(FDCAN3_IT1_IRQn);
}
