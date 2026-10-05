#ifndef PACE_MOTOR_REGISTRY_H
#define PACE_MOTOR_REGISTRY_H

#include <stdbool.h>
#include <stdint.h>

#include "pace_motor_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    pace_motor_state_t motors[PACE_MOTOR_COUNT];
    bool lk_encoder_initialized[PACE_LK_MOTOR_COUNT];
    uint16_t lk_last_encoder[PACE_LK_MOTOR_COUNT];
} pace_motor_registry_t;

void pace_motor_registry_init(pace_motor_registry_t *registry);
bool pace_motor_registry_apply_feedback(pace_motor_registry_t *registry,
                                        uint16_t can_id,
                                        const uint8_t data[PACE_CAN_CLASSIC_MAX_DATA],
                                        uint8_t dlc,
                                        uint32_t rx_timestamp_us);
bool pace_motor_registry_confirm_tx(pace_motor_registry_t *registry,
                                    const pace_encoded_command_t *encoded,
                                    uint32_t tx_timestamp_us);
void pace_motor_registry_refresh_ages(pace_motor_registry_t *registry,
                                      uint32_t now_us,
                                      uint32_t offline_timeout_us);
const pace_motor_state_t *pace_motor_registry_get(const pace_motor_registry_t *registry,
                                                  uint8_t canonical_index);

#ifdef __cplusplus
}
#endif

#endif
