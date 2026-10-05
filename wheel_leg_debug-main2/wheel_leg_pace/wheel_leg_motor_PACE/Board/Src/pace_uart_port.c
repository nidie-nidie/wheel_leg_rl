#include "pace_uart_port.h"

#include <string.h>

#include "usart.h"

static pace_uart_port_stats_t pace_uart_stats;
static pace_uart_rx_callback_t pace_uart_rx_callback;
static pace_uart_tx_callback_t pace_uart_tx_callback;
static uint8_t *pace_uart_rx_buffer;
static uint16_t pace_uart_rx_capacity;
static uint16_t pace_uart_rx_last_position;

void pace_uart_port_init(void)
{
    memset(&pace_uart_stats, 0, sizeof(pace_uart_stats));
    pace_uart_rx_callback = 0;
    pace_uart_tx_callback = 0;
    pace_uart_rx_buffer = 0;
    pace_uart_rx_capacity = 0U;
    pace_uart_rx_last_position = 0U;
}

void pace_uart_port_set_tx_callback(pace_uart_tx_callback_t callback)
{
    pace_uart_tx_callback = callback;
}

bool pace_uart_port_start_rx(uint8_t *buffer,
                             uint16_t capacity,
                             pace_uart_rx_callback_t callback)
{
    if ((buffer == 0) || (capacity == 0U))
    {
        return false;
    }
    pace_uart_rx_buffer = buffer;
    pace_uart_rx_capacity = capacity;
    pace_uart_rx_last_position = 0U;
    pace_uart_rx_callback = callback;
    if (HAL_UARTEx_ReceiveToIdle_DMA(&huart7, buffer, capacity) != HAL_OK)
    {
        pace_uart_stats.rx_error_count++;
        return false;
    }
    __HAL_DMA_DISABLE_IT(huart7.hdmarx, DMA_IT_HT);
    return true;
}

bool pace_uart_port_send_dma(const uint8_t *data, uint16_t length)
{
    if ((data == 0) || (length == 0U) || pace_uart_stats.tx_busy)
    {
        return false;
    }
    pace_uart_stats.tx_busy = true;
    pace_uart_stats.tx_error = false;
    if (HAL_UART_Transmit_DMA(&huart7, data, length) != HAL_OK)
    {
        pace_uart_stats.tx_busy = false;
        pace_uart_stats.tx_error = true;
        pace_uart_stats.tx_error_count++;
        return false;
    }
    return true;
}

bool pace_uart_port_is_busy(void)
{
    return pace_uart_stats.tx_busy;
}

const pace_uart_port_stats_t *pace_uart_port_stats(void)
{
    return &pace_uart_stats;
}

void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart == &huart7)
    {
        pace_uart_stats.tx_busy = false;
        pace_uart_stats.tx_complete_count++;
        if (pace_uart_tx_callback != 0)
        {
            pace_uart_tx_callback(true);
        }
    }
}

void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t size)
{
    if (huart == &huart7)
    {
        pace_uart_stats.rx_event_count++;
        if ((pace_uart_rx_callback != 0) && (pace_uart_rx_buffer != 0) &&
            (pace_uart_rx_capacity > 0U))
        {
            if (size > pace_uart_rx_last_position)
            {
                pace_uart_rx_callback(&pace_uart_rx_buffer[pace_uart_rx_last_position],
                                      (uint16_t)(size - pace_uart_rx_last_position));
            }
            else if (size < pace_uart_rx_last_position)
            {
                pace_uart_rx_callback(&pace_uart_rx_buffer[pace_uart_rx_last_position],
                                      (uint16_t)(pace_uart_rx_capacity - pace_uart_rx_last_position));
                if (size > 0U)
                {
                    pace_uart_rx_callback(pace_uart_rx_buffer, size);
                }
            }
            pace_uart_rx_last_position = (size == pace_uart_rx_capacity) ? 0U : size;
        }
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart == &huart7)
    {
        pace_uart_stats.tx_busy = false;
        pace_uart_stats.tx_error = true;
        pace_uart_stats.tx_error_count++;
        pace_uart_stats.rx_error_count++;
        if (pace_uart_tx_callback != 0)
        {
            pace_uart_tx_callback(false);
        }
    }
}
