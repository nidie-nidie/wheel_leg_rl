#include "usart.h"

UART_HandleTypeDef huart7;
DMA_HandleTypeDef hdma_uart7_rx;
DMA_HandleTypeDef hdma_uart7_tx;

void MX_UART7_Init(void)
{
    huart7.Instance = UART7;
    huart7.Init.BaudRate = 921600U;
    huart7.Init.WordLength = UART_WORDLENGTH_8B;
    huart7.Init.StopBits = UART_STOPBITS_1;
    huart7.Init.Parity = UART_PARITY_NONE;
    huart7.Init.Mode = UART_MODE_TX_RX;
    huart7.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart7.Init.OverSampling = UART_OVERSAMPLING_16;
    huart7.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart7.Init.ClockPrescaler = UART_PRESCALER_DIV1;
    huart7.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    if (HAL_UART_Init(&huart7) != HAL_OK)
    {
        Error_Handler();
    }
    if ((HAL_UARTEx_SetTxFifoThreshold(&huart7, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK) ||
        (HAL_UARTEx_SetRxFifoThreshold(&huart7, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK) ||
        (HAL_UARTEx_DisableFifoMode(&huart7) != HAL_OK))
    {
        Error_Handler();
    }
}

void HAL_UART_MspInit(UART_HandleTypeDef *uart_handle)
{
    GPIO_InitTypeDef gpio = {0};
    RCC_PeriphCLKInitTypeDef peripheral_clock = {0};

    if (uart_handle->Instance != UART7)
    {
        return;
    }
    peripheral_clock.PeriphClockSelection = RCC_PERIPHCLK_UART7;
    peripheral_clock.Usart234578ClockSelection = RCC_USART234578CLKSOURCE_D2PCLK1;
    if (HAL_RCCEx_PeriphCLKConfig(&peripheral_clock) != HAL_OK)
    {
        Error_Handler();
    }

    __HAL_RCC_UART7_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    gpio.Pin = GPIO_PIN_7 | GPIO_PIN_8;
    gpio.Mode = GPIO_MODE_AF_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    gpio.Alternate = GPIO_AF7_UART7;
    HAL_GPIO_Init(GPIOE, &gpio);

    hdma_uart7_rx.Instance = DMA1_Stream3;
    hdma_uart7_rx.Init.Request = DMA_REQUEST_UART7_RX;
    hdma_uart7_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_uart7_rx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_uart7_rx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_uart7_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_uart7_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_uart7_rx.Init.Mode = DMA_CIRCULAR;
    hdma_uart7_rx.Init.Priority = DMA_PRIORITY_HIGH;
    hdma_uart7_rx.Init.FIFOMode = DMA_FIFOMODE_DISABLE;
    if (HAL_DMA_Init(&hdma_uart7_rx) != HAL_OK)
    {
        Error_Handler();
    }
    __HAL_LINKDMA(uart_handle, hdmarx, hdma_uart7_rx);

    hdma_uart7_tx.Instance = DMA1_Stream4;
    hdma_uart7_tx.Init.Request = DMA_REQUEST_UART7_TX;
    hdma_uart7_tx.Init.Direction = DMA_MEMORY_TO_PERIPH;
    hdma_uart7_tx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_uart7_tx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_uart7_tx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_uart7_tx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_uart7_tx.Init.Mode = DMA_NORMAL;
    hdma_uart7_tx.Init.Priority = DMA_PRIORITY_VERY_HIGH;
    hdma_uart7_tx.Init.FIFOMode = DMA_FIFOMODE_DISABLE;
    if (HAL_DMA_Init(&hdma_uart7_tx) != HAL_OK)
    {
        Error_Handler();
    }
    __HAL_LINKDMA(uart_handle, hdmatx, hdma_uart7_tx);

    HAL_NVIC_SetPriority(UART7_IRQn, 5U, 0U);
    HAL_NVIC_EnableIRQ(UART7_IRQn);
}

void HAL_UART_MspDeInit(UART_HandleTypeDef *uart_handle)
{
    if (uart_handle->Instance != UART7)
    {
        return;
    }
    __HAL_RCC_UART7_CLK_DISABLE();
    HAL_GPIO_DeInit(GPIOE, GPIO_PIN_7 | GPIO_PIN_8);
    HAL_DMA_DeInit(uart_handle->hdmarx);
    HAL_DMA_DeInit(uart_handle->hdmatx);
    HAL_NVIC_DisableIRQ(UART7_IRQn);
}
