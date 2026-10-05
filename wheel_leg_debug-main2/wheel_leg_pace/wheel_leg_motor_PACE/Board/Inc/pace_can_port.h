#ifndef PACE_CAN_PORT_H
#define PACE_CAN_PORT_H

#include <stdbool.h>
#include <stdint.h>

#include "pace_motor_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PACE_CAN_PENDING_CAPACITY 8U

typedef enum
{
    PACE_CAN_SEND_OK = 0,
    PACE_CAN_SEND_INVALID,
    PACE_CAN_SEND_QUEUE_FULL,
    PACE_CAN_SEND_PENDING_FULL,
    PACE_CAN_SEND_HAL_ERROR
} pace_can_send_result_t;

typedef enum
{
    PACE_CAN_STATE_ACTIVE = 0,
    PACE_CAN_STATE_WARNING = 1U << 0,
    PACE_CAN_STATE_PASSIVE = 1U << 1,
    PACE_CAN_STATE_BUS_OFF = 1U << 2
} pace_can_state_flags_t;

typedef struct
{
    uint32_t queued_count[PACE_MOTOR_COUNT];
    uint32_t confirmed_count[PACE_MOTOR_COUNT];
    uint32_t rx_count[PACE_MOTOR_COUNT];
    uint32_t enqueue_fail_count[PACE_MOTOR_COUNT];
    uint32_t queue_full_count;
    uint32_t hal_error_count;
    uint32_t tx_event_lost_count;
    uint32_t unmatched_tx_event_count;
    uint32_t can_error_count;
    uint32_t last_enqueue_to_tx_us;
    uint32_t max_enqueue_to_tx_us;
    uint32_t last_command_period_us[PACE_MOTOR_COUNT];
    uint32_t max_command_period_us[PACE_MOTOR_COUNT];
    uint8_t tx_fifo_high_water;
    uint8_t can_state;
} pace_can_port_stats_t;

void pace_can_port_init(pace_motor_registry_t *registry);
bool pace_can_port_start(void);
pace_can_send_result_t pace_can_port_send(const pace_encoded_command_t *encoded);
void pace_can_port_refresh_protocol_state(void);
const pace_can_port_stats_t *pace_can_port_stats(void);
uint8_t pace_can_port_pending_count(void);

#ifdef __cplusplus
}
#endif

#endif
