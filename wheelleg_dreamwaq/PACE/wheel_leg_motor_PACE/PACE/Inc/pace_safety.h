#ifndef PACE_SAFETY_H
#define PACE_SAFETY_H

#include <stdbool.h>
#include <stdint.h>

#include "pace_motor_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    PACE_SAFETY_NONE = 0,
    PACE_SAFETY_POSITION = 1U << 0,
    PACE_SAFETY_DM_VELOCITY = 1U << 1,
    PACE_SAFETY_DM_TORQUE = 1U << 2,
    PACE_SAFETY_DM_TEMPERATURE = 1U << 3,
    PACE_SAFETY_LK_VELOCITY = 1U << 4,
    PACE_SAFETY_LK_CURRENT = 1U << 5,
    PACE_SAFETY_LK_TEMPERATURE = 1U << 6,
    PACE_SAFETY_FEEDBACK_TIMEOUT = 1U << 7,
    PACE_SAFETY_CAN = 1U << 8,
    PACE_SAFETY_UART_OVERFLOW = 1U << 9,
    PACE_SAFETY_DURATION = 1U << 10
} pace_safety_flags_t;

typedef struct
{
    bool position_limit_enabled;
    float dm_position_min_rad[PACE_DM_MOTOR_COUNT];
    float dm_position_max_rad[PACE_DM_MOTOR_COUNT];
    float dm_max_velocity_rad_s;
    float dm_max_torque_nm;
    uint8_t dm_max_temperature_c;
    float lk_max_velocity_rad_s;
    float lk_max_current_a;
    int8_t lk_max_temperature_c;
    uint32_t feedback_timeout_us;
    uint32_t session_timeout_us;
} pace_safety_limits_t;

void pace_safety_default_limits(pace_safety_limits_t *limits);
uint16_t pace_safety_check(const pace_motor_registry_t *registry,
                           const pace_safety_limits_t *limits,
                           uint32_t now_us,
                           uint32_t session_start_us,
                           bool can_fault,
                           bool uart_overflow);

#ifdef __cplusplus
}
#endif

#endif
