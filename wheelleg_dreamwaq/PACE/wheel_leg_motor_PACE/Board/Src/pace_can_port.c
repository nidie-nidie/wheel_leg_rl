#include "pace_can_port.h"

#include <string.h>

#include "fdcan.h"
#include "pace_motor_manifest.h"
#include "pace_time.h"

typedef struct
{
    bool used;
    uint8_t marker;
    uint32_t enqueue_timestamp_us;
    pace_encoded_command_t encoded;
} pace_can_pending_t;

static pace_motor_registry_t *pace_registry;
static pace_can_port_stats_t pace_can_stats;
static pace_can_pending_t pace_pending[PACE_CAN_PENDING_CAPACITY];
static uint8_t pace_next_marker;
static uint32_t pace_last_confirmed_timestamp[PACE_MOTOR_COUNT];
static bool pace_last_confirmed_valid[PACE_MOTOR_COUNT];

static int32_t pace_pending_find_free(void)
{
    uint8_t index;
    for (index = 0U; index < PACE_CAN_PENDING_CAPACITY; ++index)
    {
        if (!pace_pending[index].used)
        {
            return index;
        }
    }
    return -1;
}

static int32_t pace_pending_find_marker(uint8_t marker)
{
    uint8_t index;
    for (index = 0U; index < PACE_CAN_PENDING_CAPACITY; ++index)
    {
        if (pace_pending[index].used && (pace_pending[index].marker == marker))
        {
            return index;
        }
    }
    return -1;
}

static bool pace_marker_allocate(uint8_t *marker)
{
    uint16_t attempt;
    for (attempt = 0U; attempt < 256U; ++attempt)
    {
        uint8_t candidate = pace_next_marker++;
        if (pace_pending_find_marker(candidate) < 0)
        {
            *marker = candidate;
            return true;
        }
    }
    return false;
}

void pace_can_port_init(pace_motor_registry_t *registry)
{
    pace_registry = registry;
    pace_next_marker = 0U;
    memset(&pace_can_stats, 0, sizeof(pace_can_stats));
    memset(pace_pending, 0, sizeof(pace_pending));
    memset(pace_last_confirmed_timestamp, 0, sizeof(pace_last_confirmed_timestamp));
    memset(pace_last_confirmed_valid, 0, sizeof(pace_last_confirmed_valid));
}

bool pace_can_port_start(void)
{
    FDCAN_FilterTypeDef filter = {0};
    uint32_t notifications;

    filter.IdType = FDCAN_STANDARD_ID;
    filter.FilterIndex = 0U;
    filter.FilterType = FDCAN_FILTER_MASK;
    filter.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
    filter.FilterID1 = 0U;
    filter.FilterID2 = 0U;
    if (HAL_FDCAN_ConfigFilter(&hfdcan3, &filter) != HAL_OK)
    {
        return false;
    }
    if (HAL_FDCAN_ConfigGlobalFilter(&hfdcan3, FDCAN_REJECT, FDCAN_REJECT,
                                     FDCAN_FILTER_REMOTE, FDCAN_FILTER_REMOTE) != HAL_OK)
    {
        return false;
    }
    if ((HAL_FDCAN_ConfigTimestampCounter(&hfdcan3, FDCAN_TIMESTAMP_PRESC_1) != HAL_OK) ||
        (HAL_FDCAN_EnableTimestampCounter(&hfdcan3, FDCAN_TIMESTAMP_INTERNAL) != HAL_OK))
    {
        return false;
    }
    notifications = FDCAN_IT_RX_FIFO0_NEW_MESSAGE |
                    FDCAN_IT_TX_EVT_FIFO_NEW_DATA |
                    FDCAN_IT_TX_EVT_FIFO_FULL |
                    FDCAN_IT_TX_EVT_FIFO_ELT_LOST |
                    FDCAN_IT_ERROR_PASSIVE |
                    FDCAN_IT_ERROR_WARNING |
                    FDCAN_IT_BUS_OFF |
                    FDCAN_IT_ARB_PROTOCOL_ERROR |
                    FDCAN_IT_DATA_PROTOCOL_ERROR;
    if (HAL_FDCAN_ActivateNotification(&hfdcan3, notifications, 0U) != HAL_OK)
    {
        return false;
    }
    return HAL_FDCAN_Start(&hfdcan3) == HAL_OK;
}

pace_can_send_result_t pace_can_port_send(const pace_encoded_command_t *encoded)
{
    FDCAN_TxHeaderTypeDef header = {0};
    int32_t pending_index;
    uint8_t marker;
    uint32_t primask;
    HAL_StatusTypeDef status;
    uint8_t fill_level;

    if ((encoded == 0) || (encoded->canonical_index >= PACE_MOTOR_COUNT) ||
        (encoded->dlc != 8U) || (pace_registry == 0))
    {
        return PACE_CAN_SEND_INVALID;
    }
    if (HAL_FDCAN_GetTxFifoFreeLevel(&hfdcan3) == 0U)
    {
        pace_can_stats.queue_full_count++;
        pace_can_stats.enqueue_fail_count[encoded->canonical_index]++;
        return PACE_CAN_SEND_QUEUE_FULL;
    }
    pending_index = pace_pending_find_free();
    if ((pending_index < 0) || !pace_marker_allocate(&marker))
    {
        pace_can_stats.enqueue_fail_count[encoded->canonical_index]++;
        return PACE_CAN_SEND_PENDING_FULL;
    }

    header.Identifier = encoded->can_id;
    header.IdType = FDCAN_STANDARD_ID;
    header.TxFrameType = FDCAN_DATA_FRAME;
    header.DataLength = FDCAN_DLC_BYTES_8;
    header.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
    header.BitRateSwitch = FDCAN_BRS_OFF;
    header.FDFormat = FDCAN_CLASSIC_CAN;
    header.TxEventFifoControl = FDCAN_STORE_TX_EVENTS;
    header.MessageMarker = marker;

    primask = __get_PRIMASK();
    __disable_irq();
    pace_pending[pending_index].used = true;
    pace_pending[pending_index].marker = marker;
    pace_pending[pending_index].enqueue_timestamp_us = pace_time_now_us();
    pace_pending[pending_index].encoded = *encoded;
    status = HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan3, &header, encoded->data);
    if (status != HAL_OK)
    {
        pace_pending[pending_index].used = false;
    }
    if (primask == 0U)
    {
        __enable_irq();
    }

    if (status != HAL_OK)
    {
        pace_can_stats.hal_error_count++;
        pace_can_stats.enqueue_fail_count[encoded->canonical_index]++;
        return PACE_CAN_SEND_HAL_ERROR;
    }
    pace_can_stats.queued_count[encoded->canonical_index]++;
    fill_level = (uint8_t)(PACE_CAN_PENDING_CAPACITY - HAL_FDCAN_GetTxFifoFreeLevel(&hfdcan3));
    if (fill_level > pace_can_stats.tx_fifo_high_water)
    {
        pace_can_stats.tx_fifo_high_water = fill_level;
    }
    return PACE_CAN_SEND_OK;
}

void pace_can_port_refresh_protocol_state(void)
{
    FDCAN_ProtocolStatusTypeDef protocol = {0};
    uint8_t state = PACE_CAN_STATE_ACTIVE;
    if (HAL_FDCAN_GetProtocolStatus(&hfdcan3, &protocol) != HAL_OK)
    {
        pace_can_stats.hal_error_count++;
        return;
    }
    if (protocol.Warning != 0U)
    {
        state |= PACE_CAN_STATE_WARNING;
    }
    if (protocol.ErrorPassive != 0U)
    {
        state |= PACE_CAN_STATE_PASSIVE;
    }
    if (protocol.BusOff != 0U)
    {
        state |= PACE_CAN_STATE_BUS_OFF;
    }
    pace_can_stats.can_state = state;
}

const pace_can_port_stats_t *pace_can_port_stats(void)
{
    return &pace_can_stats;
}

uint8_t pace_can_port_pending_count(void)
{
    uint8_t count = 0U;
    uint8_t index;
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    for (index = 0U; index < PACE_CAN_PENDING_CAPACITY; ++index)
    {
        if (pace_pending[index].used)
        {
            count++;
        }
    }
    if (primask == 0U)
    {
        __enable_irq();
    }
    return count;
}

void HAL_FDCAN_TxEventFifoCallback(FDCAN_HandleTypeDef *hfdcan, uint32_t tx_event_fifo_its)
{
    if (hfdcan != &hfdcan3)
    {
        return;
    }
    if ((tx_event_fifo_its & FDCAN_IT_TX_EVT_FIFO_ELT_LOST) != 0U)
    {
        pace_can_stats.tx_event_lost_count++;
    }
    while ((hfdcan3.Instance->TXEFS & FDCAN_TXEFS_EFFL) != 0U)
    {
        FDCAN_TxEventFifoTypeDef event = {0};
        int32_t pending_index;
        uint32_t observed_now_us;
        uint32_t tx_timestamp_us;
        uint32_t latency_us;
        uint8_t motor_index;

        if (HAL_FDCAN_GetTxEvent(&hfdcan3, &event) != HAL_OK)
        {
            pace_can_stats.hal_error_count++;
            break;
        }
        pending_index = pace_pending_find_marker((uint8_t)event.MessageMarker);
        if (pending_index < 0)
        {
            pace_can_stats.unmatched_tx_event_count++;
            continue;
        }
        observed_now_us = pace_time_now_us();
        tx_timestamp_us = pace_time_expand_fdcan_timestamp((uint16_t)event.TxTimestamp,
                                                           observed_now_us);
        latency_us = tx_timestamp_us - pace_pending[pending_index].enqueue_timestamp_us;
        pace_can_stats.last_enqueue_to_tx_us = latency_us;
        if (latency_us > pace_can_stats.max_enqueue_to_tx_us)
        {
            pace_can_stats.max_enqueue_to_tx_us = latency_us;
        }
        motor_index = pace_pending[pending_index].encoded.canonical_index;
        if (pace_last_confirmed_valid[motor_index])
        {
            const uint32_t period_us = tx_timestamp_us - pace_last_confirmed_timestamp[motor_index];
            pace_can_stats.last_command_period_us[motor_index] = period_us;
            if (period_us > pace_can_stats.max_command_period_us[motor_index])
            {
                pace_can_stats.max_command_period_us[motor_index] = period_us;
            }
        }
        pace_last_confirmed_timestamp[motor_index] = tx_timestamp_us;
        pace_last_confirmed_valid[motor_index] = true;
        if (pace_motor_registry_confirm_tx(pace_registry,
                                           &pace_pending[pending_index].encoded,
                                           tx_timestamp_us))
        {
            pace_can_stats.confirmed_count[motor_index]++;
        }
        pace_pending[pending_index].used = false;
    }
}

void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *hfdcan, uint32_t rx_fifo0_its)
{
    (void)rx_fifo0_its;
    if ((hfdcan != &hfdcan3) || (pace_registry == 0))
    {
        return;
    }
    while (HAL_FDCAN_GetRxFifoFillLevel(&hfdcan3, FDCAN_RX_FIFO0) > 0U)
    {
        FDCAN_RxHeaderTypeDef header = {0};
        uint8_t data[8];
        uint8_t motor_index;
        uint32_t timestamp_us;
        if (HAL_FDCAN_GetRxMessage(&hfdcan3, FDCAN_RX_FIFO0, &header, data) != HAL_OK)
        {
            pace_can_stats.hal_error_count++;
            break;
        }
        motor_index = pace_motor_index_from_feedback_id((uint16_t)header.Identifier);
        timestamp_us = pace_time_expand_fdcan_timestamp((uint16_t)header.RxTimestamp,
                                                        pace_time_now_us());
        if ((motor_index != PACE_MOTOR_INDEX_INVALID) &&
            pace_motor_registry_apply_feedback(pace_registry,
                                               (uint16_t)header.Identifier,
                                               data,
                                               8U,
                                               timestamp_us))
        {
            pace_can_stats.rx_count[motor_index]++;
        }
    }
}

void HAL_FDCAN_ErrorStatusCallback(FDCAN_HandleTypeDef *hfdcan, uint32_t error_status_its)
{
    if (hfdcan == &hfdcan3)
    {
        (void)error_status_its;
        pace_can_stats.can_error_count++;
        pace_can_port_refresh_protocol_state();
    }
}

void HAL_FDCAN_ErrorCallback(FDCAN_HandleTypeDef *hfdcan)
{
    if (hfdcan == &hfdcan3)
    {
        pace_can_stats.can_error_count++;
        pace_can_stats.hal_error_count++;
        pace_can_port_refresh_protocol_state();
    }
}
