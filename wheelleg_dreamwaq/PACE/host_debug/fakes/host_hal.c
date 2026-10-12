#include "host_hal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cmsis_os.h"
#include "pace_motor_manifest.h"

#define HOST_UART_CAPACITY (16U * 1024U * 1024U)
#define HOST_CAN_CAPACITY 8U

static FDCAN_TypeDef host_fdcan;
static DMA_HandleTypeDef host_dma_rx;
uint8_t host_tim2_instance, host_tim5_instance, host_tim6_instance;
FDCAN_HandleTypeDef hfdcan3 = {&host_fdcan, 0U};
UART_HandleTypeDef huart7 = {0, &host_dma_rx, 0U};
TIM_HandleTypeDef htim2 = {TIM2}, htim5 = {TIM5}, htim6 = {TIM6};
static uint32_t host_clock, host_primask, host_fdcan_lag;
static host_can_message_t host_tx_queue[HOST_CAN_CAPACITY];
static size_t host_tx_count;
static FDCAN_TxEventFifoTypeDef host_tx_event;
static bool host_tx_event_ready;
static FDCAN_RxHeaderTypeDef host_rx_header;
static uint8_t host_rx_data[64];
/* Exact mapping used by this repository's HAL_FDCAN_GetRxMessage. */
static const uint8_t host_dlc_bytes[16] = {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 12U, 16U, 20U, 24U, 32U, 48U, 64U};
static bool host_rx_ready;
static FDCAN_ProtocolStatusTypeDef host_protocol;
static uint32_t host_notification_bits;
static uint8_t host_uart_output[HOST_UART_CAPACITY];
static size_t host_uart_length;
static uint32_t host_tx_failures, host_rx_start_count, host_tx_call_count;
static uint8_t *host_rx_dma;
static uint16_t host_rx_capacity, host_rx_position;

void host_reset(void)
{
    host_clock = host_primask = host_fdcan_lag = 0U;
    host_tx_count = 0U;
    host_tx_event_ready = host_rx_ready = false;
    memset(&host_fdcan, 0, sizeof(host_fdcan));
    memset(&host_protocol, 0, sizeof(host_protocol));
    host_notification_bits = 0U;
    host_uart_length = 0U;
    host_tx_failures = host_rx_start_count = host_tx_call_count = 0U;
    host_rx_dma = 0;
    host_rx_capacity = host_rx_position = 0U;
    huart7.ErrorCode = 0U;
}
void host_set_time(uint32_t time_us) { host_clock = time_us; }
uint32_t host_now_us(void) { return host_clock; }
void host_tick(void) { HAL_TIM_PeriodElapsedCallback(&htim6); }
void host_set_fdcan_epoch_lag(uint32_t lag_us) { host_fdcan_lag = lag_us; }
uint32_t __get_PRIMASK(void) { return host_primask; }
void __disable_irq(void) { host_primask = 1U; }
void __enable_irq(void) { host_primask = 0U; }
void __DMB(void) { }
void Error_Handler(void) { fputs("host Error_Handler\n", stderr); abort(); }
void HAL_IncTick(void) { }
uint32_t HAL_GetTick(void) { return host_clock / 1000U; }
osStatus osDelay(uint32_t millisec) { host_clock += millisec * 1000U; return osOK; }
HAL_StatusTypeDef HAL_TIM_Base_Start(TIM_HandleTypeDef *timer) { (void)timer; return HAL_OK; }
HAL_StatusTypeDef HAL_TIM_Base_Start_IT(TIM_HandleTypeDef *timer) { (void)timer; return HAL_OK; }
HAL_StatusTypeDef HAL_FDCAN_ConfigFilter(FDCAN_HandleTypeDef *can, const FDCAN_FilterTypeDef *filter)
{ (void)can; (void)filter; return HAL_OK; }
HAL_StatusTypeDef HAL_FDCAN_ConfigGlobalFilter(FDCAN_HandleTypeDef *can, uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{ (void)can; (void)a; (void)b; (void)c; (void)d; return HAL_OK; }
HAL_StatusTypeDef HAL_FDCAN_ConfigTimestampCounter(FDCAN_HandleTypeDef *can, uint32_t prescaler)
{ (void)can; (void)prescaler; return HAL_OK; }
HAL_StatusTypeDef HAL_FDCAN_EnableTimestampCounter(FDCAN_HandleTypeDef *can, uint32_t operation)
{ (void)can; (void)operation; return HAL_OK; }
HAL_StatusTypeDef HAL_FDCAN_ActivateNotification(FDCAN_HandleTypeDef *can, uint32_t bits, uint32_t buffer)
{ (void)can; (void)buffer; host_notification_bits = bits; return HAL_OK; }
HAL_StatusTypeDef HAL_FDCAN_Start(FDCAN_HandleTypeDef *can) { (void)can; return HAL_OK; }
uint32_t HAL_FDCAN_GetTxFifoFreeLevel(FDCAN_HandleTypeDef *can)
{ (void)can; return (uint32_t)(HOST_CAN_CAPACITY - host_tx_count); }
HAL_StatusTypeDef HAL_FDCAN_AddMessageToTxFifoQ(FDCAN_HandleTypeDef *can,
                                               const FDCAN_TxHeaderTypeDef *header,
                                               const uint8_t *data)
{
    host_can_message_t *message;
    (void)can;
    if (host_tx_count == HOST_CAN_CAPACITY) return HAL_BUSY;
    message = &host_tx_queue[host_tx_count++];
    message->header = *header;
    memcpy(message->data, data, 8U);
    message->enqueue_time_us = host_clock;
    return HAL_OK;
}
HAL_StatusTypeDef HAL_FDCAN_GetProtocolStatus(FDCAN_HandleTypeDef *can, FDCAN_ProtocolStatusTypeDef *status)
{ (void)can; *status = host_protocol; return HAL_OK; }
bool host_can_peek(host_can_message_t *message)
{ if (!host_tx_count) return false; *message = host_tx_queue[0]; return true; }
size_t host_can_queued(void) { return host_tx_count; }
bool host_can_confirm_next(void)
{
    if (!host_tx_count) return false;
    host_tx_event.MessageMarker = host_tx_queue[0].header.MessageMarker;
    host_tx_event.TxTimestamp = (uint16_t)(host_clock - host_fdcan_lag);
    host_tx_event.EventType = 1U;
    memmove(host_tx_queue, &host_tx_queue[1], (host_tx_count - 1U) * sizeof(host_tx_queue[0]));
    --host_tx_count;
    host_tx_event_ready = true;
    host_fdcan.TXEFS = 1U;
    HAL_FDCAN_TxEventFifoCallback(&hfdcan3, FDCAN_IT_TX_EVT_FIFO_NEW_DATA);
    return true;
}
void host_can_confirm_all(void) { while (host_can_confirm_next()) { } }
HAL_StatusTypeDef HAL_FDCAN_GetTxEvent(FDCAN_HandleTypeDef *can, FDCAN_TxEventFifoTypeDef *event)
{
    (void)can;
    if (!host_tx_event_ready) return HAL_ERROR;
    *event = host_tx_event;
    host_tx_event_ready = false;
    host_fdcan.TXEFS = 0U;
    return HAL_OK;
}
void host_can_feedback(const FDCAN_RxHeaderTypeDef *header, const uint8_t *data)
{
    host_rx_header = *header;
    if (header->DataLength > 15U) abort();
    memcpy(host_rx_data, data, host_dlc_bytes[header->DataLength]);
    host_rx_ready = true;
    HAL_FDCAN_RxFifo0Callback(&hfdcan3, FDCAN_IT_RX_FIFO0_NEW_MESSAGE);
}
uint32_t HAL_FDCAN_GetRxFifoFillLevel(FDCAN_HandleTypeDef *can, uint32_t fifo)
{ (void)can; (void)fifo; return host_rx_ready ? 1U : 0U; }
HAL_StatusTypeDef HAL_FDCAN_GetRxMessage(FDCAN_HandleTypeDef *can, uint32_t fifo,
                                       FDCAN_RxHeaderTypeDef *header, uint8_t *data)
{
    uint32_t size;
    (void)can; (void)fifo;
    if (!host_rx_ready) return HAL_ERROR;
    *header = host_rx_header;
    size = host_dlc_bytes[host_rx_header.DataLength];
    memcpy(data, host_rx_data, size);
    host_rx_ready = false;
    return HAL_OK;
}
void host_can_motor_feedback(uint8_t motor, uint16_t position_raw)
{
    FDCAN_RxHeaderTypeDef header = {0};
    uint8_t data[8] = {0};
    header.Identifier = pace_motor_manifest[motor].feedback_can_id;
    header.IdType = FDCAN_STANDARD_ID;
    header.RxFrameType = FDCAN_DATA_FRAME;
    header.FDFormat = FDCAN_CLASSIC_CAN;
    header.DataLength = FDCAN_DLC_BYTES_8;
    header.RxTimestamp = (uint16_t)(host_clock - host_fdcan_lag);
    if (motor < 4U) {
        data[0] = (uint8_t)(0x10U | pace_motor_manifest[motor].device_id);
        data[1] = (uint8_t)(position_raw >> 8); data[2] = (uint8_t)position_raw;
        data[3] = 0x7FU; data[4] = 0xF7U; data[5] = 0xFFU;
        data[6] = 25U; data[7] = 26U;
    } else {
        data[0] = 0xA2U; data[1] = 25U;
        data[6] = (uint8_t)position_raw; data[7] = (uint8_t)(position_raw >> 8);
    }
    host_can_feedback(&header, data);
}
void host_can_feedback_all(void)
{ uint8_t motor; for (motor = 0U; motor < 6U; ++motor) host_can_motor_feedback(motor, 32767U); }
void host_can_rx_lost(void)
{ HAL_FDCAN_RxFifo0Callback(&hfdcan3, FDCAN_IT_RX_FIFO0_MESSAGE_LOST | FDCAN_IT_RX_FIFO0_FULL); }
void host_can_set_bus_off(bool bus_off)
{
    host_protocol.BusOff = bus_off ? 1U : 0U;
    HAL_FDCAN_ErrorStatusCallback(&hfdcan3, FDCAN_IT_BUS_OFF);
}
uint32_t host_can_notifications(void) { return host_notification_bits; }
HAL_StatusTypeDef HAL_UARTEx_ReceiveToIdle_DMA(UART_HandleTypeDef *uart, uint8_t *buffer, uint16_t capacity)
{
    (void)uart; host_rx_dma = buffer; host_rx_capacity = capacity; host_rx_position = 0U;
    ++host_rx_start_count; return HAL_OK;
}
HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *uart, const uint8_t *bytes, uint16_t length, uint32_t timeout)
{
    (void)uart; (void)timeout; ++host_tx_call_count;
    if (host_tx_failures) { --host_tx_failures; return HAL_TIMEOUT; }
    if (host_uart_length + length > HOST_UART_CAPACITY) return HAL_ERROR;
    memcpy(&host_uart_output[host_uart_length], bytes, length); host_uart_length += length;
    return HAL_OK;
}
HAL_StatusTypeDef HAL_UART_Transmit_DMA(UART_HandleTypeDef *uart, const uint8_t *bytes, uint16_t length)
{
    HAL_StatusTypeDef result = HAL_UART_Transmit(uart, bytes, length, 0U);
    if (result == HAL_OK) HAL_UART_TxCpltCallback(uart);
    return result;
}
HAL_StatusTypeDef HAL_UART_AbortTransmit(UART_HandleTypeDef *uart) { (void)uart; return HAL_OK; }
HAL_StatusTypeDef HAL_UART_AbortReceive(UART_HandleTypeDef *uart)
{ (void)uart; host_rx_dma = 0; return HAL_OK; }
void host_uart_receive(const uint8_t *bytes, size_t length)
{
    size_t index;
    if (!host_rx_dma || !host_rx_capacity) return;
    for (index = 0U; index < length; ++index) {
        host_rx_dma[host_rx_position++] = bytes[index];
        if (host_rx_position == host_rx_capacity) {
            HAL_UARTEx_RxEventCallback(&huart7, host_rx_capacity);
            host_rx_position = 0U;
        }
    }
    if (host_rx_position) HAL_UARTEx_RxEventCallback(&huart7, host_rx_position);
}
void host_uart_receive_error(void)
{ host_rx_dma = 0; huart7.ErrorCode = HAL_UART_ERROR_ORE; HAL_UART_ErrorCallback(&huart7); }
void host_uart_fail_next(uint32_t count) { host_tx_failures = count; }
const uint8_t *host_uart_bytes(void) { return host_uart_output; }
size_t host_uart_size(void) { return host_uart_length; }
void host_uart_clear(void) { host_uart_length = 0U; }
uint32_t host_uart_rx_starts(void) { return host_rx_start_count; }
uint32_t host_uart_tx_calls(void) { return host_tx_call_count; }
bool host_uart_write_file(const char *path)
{
    FILE *output = fopen(path, "wb");
    bool ok;
    if (!output) return false;
    ok = fwrite(host_uart_output, 1U, host_uart_length, output) == host_uart_length;
    if (fclose(output) != 0) ok = false;
    return ok;
}
