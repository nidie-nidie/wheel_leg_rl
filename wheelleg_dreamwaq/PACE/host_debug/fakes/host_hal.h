#ifndef PACE_HOST_HAL_H
#define PACE_HOST_HAL_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef enum { HAL_OK = 0, HAL_ERROR = 1, HAL_BUSY = 2, HAL_TIMEOUT = 3 } HAL_StatusTypeDef;
typedef struct { uint32_t TXEFS; } FDCAN_TypeDef;
typedef struct { FDCAN_TypeDef *Instance; uint32_t ErrorCode; } FDCAN_HandleTypeDef;
typedef struct { uint32_t Identifier, IdType, TxFrameType, DataLength, ErrorStateIndicator, BitRateSwitch, FDFormat, TxEventFifoControl, MessageMarker; } FDCAN_TxHeaderTypeDef;
typedef struct { uint32_t Identifier, IdType, RxFrameType, DataLength, ErrorStateIndicator, BitRateSwitch, FDFormat, RxTimestamp, FilterIndex, IsFilterMatchingFrame; } FDCAN_RxHeaderTypeDef;
typedef struct { uint32_t MessageMarker, TxTimestamp, EventType; } FDCAN_TxEventFifoTypeDef;
typedef struct { uint32_t IdType, FilterIndex, FilterType, FilterConfig, FilterID1, FilterID2; } FDCAN_FilterTypeDef;
typedef struct { uint32_t Warning, ErrorPassive, BusOff; } FDCAN_ProtocolStatusTypeDef;
typedef struct { uint32_t unused; } DMA_HandleTypeDef;
typedef struct { void *Instance; DMA_HandleTypeDef *hdmarx; uint32_t ErrorCode; } UART_HandleTypeDef;
typedef struct { void *Instance; } TIM_HandleTypeDef;
#define FDCAN_STANDARD_ID 0U
#define FDCAN_EXTENDED_ID 0x40000000U
#define FDCAN_DATA_FRAME 0U
#define FDCAN_REMOTE_FRAME 0x20000000U
#define FDCAN_DLC_BYTES_8 8U
#define FDCAN_DLC_BYTES_64 15U
#define FDCAN_CLASSIC_CAN 0U
#define FDCAN_FD_CAN 0x00200000U
#define FDCAN_ESI_ACTIVE 0U
#define FDCAN_BRS_OFF 0U
#define FDCAN_STORE_TX_EVENTS 0x00800000U
#define FDCAN_FILTER_MASK 2U
#define FDCAN_FILTER_TO_RXFIFO0 1U
#define FDCAN_REJECT 2U
#define FDCAN_FILTER_REMOTE 0U
#define FDCAN_TIMESTAMP_PRESC_1 0U
#define FDCAN_TIMESTAMP_INTERNAL 1U
#define FDCAN_RX_FIFO0 0U
#define FDCAN_TXEFS_EFFL 0x3FU
#define FDCAN_IT_RX_FIFO0_NEW_MESSAGE (1U << 0)
#define FDCAN_IT_TX_EVT_FIFO_NEW_DATA (1U << 12)
#define FDCAN_IT_TX_EVT_FIFO_FULL (1U << 14)
#define FDCAN_IT_TX_EVT_FIFO_ELT_LOST (1U << 15)
#define FDCAN_IT_ERROR_PASSIVE (1U << 23)
#define FDCAN_IT_ERROR_WARNING (1U << 24)
#define FDCAN_IT_BUS_OFF (1U << 25)
#define FDCAN_IT_ARB_PROTOCOL_ERROR (1U << 27)
#define FDCAN_IT_DATA_PROTOCOL_ERROR (1U << 28)
#define FDCAN_IT_RX_FIFO0_FULL (1U << 2)
#define FDCAN_IT_RX_FIFO0_MESSAGE_LOST (1U << 3)
#define DMA_IT_HT 1U
#define HAL_UART_ERROR_NONE 0U
#define HAL_UART_ERROR_ORE 1U
extern FDCAN_HandleTypeDef hfdcan3;
extern UART_HandleTypeDef huart7;
extern TIM_HandleTypeDef htim2, htim5, htim6;
extern uint8_t host_tim2_instance, host_tim5_instance, host_tim6_instance;
#define TIM2 ((void *)&host_tim2_instance)
#define TIM5 ((void *)&host_tim5_instance)
#define TIM6 ((void *)&host_tim6_instance)
#define __HAL_TIM_GET_COUNTER(handle) host_now_us()
#define __HAL_DMA_DISABLE_IT(handle, flag) ((void)(handle), (void)(flag))
uint32_t __get_PRIMASK(void);
void __disable_irq(void);
void __enable_irq(void);
void __DMB(void);
void Error_Handler(void);
void HAL_IncTick(void);
uint32_t HAL_GetTick(void);
HAL_StatusTypeDef HAL_TIM_Base_Start(TIM_HandleTypeDef *);
HAL_StatusTypeDef HAL_TIM_Base_Start_IT(TIM_HandleTypeDef *);
HAL_StatusTypeDef HAL_FDCAN_ConfigFilter(FDCAN_HandleTypeDef *, const FDCAN_FilterTypeDef *);
HAL_StatusTypeDef HAL_FDCAN_ConfigGlobalFilter(FDCAN_HandleTypeDef *, uint32_t, uint32_t, uint32_t, uint32_t);
HAL_StatusTypeDef HAL_FDCAN_ConfigTimestampCounter(FDCAN_HandleTypeDef *, uint32_t);
HAL_StatusTypeDef HAL_FDCAN_EnableTimestampCounter(FDCAN_HandleTypeDef *, uint32_t);
HAL_StatusTypeDef HAL_FDCAN_ActivateNotification(FDCAN_HandleTypeDef *, uint32_t, uint32_t);
HAL_StatusTypeDef HAL_FDCAN_Start(FDCAN_HandleTypeDef *);
uint32_t HAL_FDCAN_GetTxFifoFreeLevel(FDCAN_HandleTypeDef *);
HAL_StatusTypeDef HAL_FDCAN_AddMessageToTxFifoQ(FDCAN_HandleTypeDef *, const FDCAN_TxHeaderTypeDef *, const uint8_t *);
HAL_StatusTypeDef HAL_FDCAN_GetProtocolStatus(FDCAN_HandleTypeDef *, FDCAN_ProtocolStatusTypeDef *);
HAL_StatusTypeDef HAL_FDCAN_GetTxEvent(FDCAN_HandleTypeDef *, FDCAN_TxEventFifoTypeDef *);
uint32_t HAL_FDCAN_GetRxFifoFillLevel(FDCAN_HandleTypeDef *, uint32_t);
HAL_StatusTypeDef HAL_FDCAN_GetRxMessage(FDCAN_HandleTypeDef *, uint32_t, FDCAN_RxHeaderTypeDef *, uint8_t *);
HAL_StatusTypeDef HAL_UARTEx_ReceiveToIdle_DMA(UART_HandleTypeDef *, uint8_t *, uint16_t);
HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *, const uint8_t *, uint16_t, uint32_t);
HAL_StatusTypeDef HAL_UART_Transmit_DMA(UART_HandleTypeDef *, const uint8_t *, uint16_t);
HAL_StatusTypeDef HAL_UART_AbortTransmit(UART_HandleTypeDef *);
HAL_StatusTypeDef HAL_UART_AbortReceive(UART_HandleTypeDef *);
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *);
void HAL_FDCAN_TxEventFifoCallback(FDCAN_HandleTypeDef *, uint32_t);
void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *, uint32_t);
void HAL_FDCAN_ErrorStatusCallback(FDCAN_HandleTypeDef *, uint32_t);
void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *, uint16_t);
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *);
void HAL_UART_ErrorCallback(UART_HandleTypeDef *);
typedef struct { FDCAN_TxHeaderTypeDef header; uint8_t data[8]; uint32_t enqueue_time_us; } host_can_message_t;
void host_reset(void);
void host_set_time(uint32_t);
uint32_t host_now_us(void);
void host_tick(void);
void host_set_fdcan_epoch_lag(uint32_t);
bool host_can_peek(host_can_message_t *);
size_t host_can_queued(void);
bool host_can_confirm_next(void);
void host_can_confirm_all(void);
/* Caller supplies the full payload length encoded by DLC, up to 64 bytes. */
void host_can_feedback(const FDCAN_RxHeaderTypeDef *, const uint8_t *data);
void host_can_motor_feedback(uint8_t motor, uint16_t position_raw);
void host_can_feedback_all(void);
void host_can_rx_lost(void);
void host_can_set_bus_off(bool);
uint32_t host_can_notifications(void);
void host_uart_receive(const uint8_t *, size_t);
void host_uart_receive_error(void);
void host_uart_fail_next(uint32_t);
const uint8_t *host_uart_bytes(void);
size_t host_uart_size(void);
void host_uart_clear(void);
uint32_t host_uart_rx_starts(void);
uint32_t host_uart_tx_calls(void);
bool host_uart_write_file(const char *);
#endif
