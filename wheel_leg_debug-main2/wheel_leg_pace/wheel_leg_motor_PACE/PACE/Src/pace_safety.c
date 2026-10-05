#include "pace_safety.h"

#include <math.h>
#include <string.h>

void pace_safety_default_limits(pace_safety_limits_t *limits)
{
    uint8_t index;
    if (limits == 0)
    {
        return;
    }
    memset(limits, 0, sizeof(*limits));
    limits->position_limit_enabled = false;
    for (index = 0U; index < PACE_DM_MOTOR_COUNT; ++index)
    {
        limits->dm_position_min_rad[index] = -12.5f;
        limits->dm_position_max_rad[index] = 12.5f;
    }
    limits->dm_max_velocity_rad_s = 12.0f;
    limits->dm_max_torque_nm = 30.0f;
    limits->dm_max_temperature_c = 80U;
    limits->lk_max_velocity_rad_s = 30.0f;
    limits->lk_max_current_a = 12.0f;
    limits->lk_max_temperature_c = 80;
    limits->feedback_timeout_us = 30000U;
    limits->session_timeout_us = 3600000000UL;
}

uint16_t pace_safety_check(const pace_motor_registry_t *registry,
                           const pace_safety_limits_t *limits,
                           uint32_t now_us,
                           uint32_t session_start_us,
                           bool can_fault,
                           bool uart_overflow)
{
    uint16_t flags = PACE_SAFETY_NONE;
    uint8_t index;
    if ((registry == 0) || (limits == 0))
    {
        return PACE_SAFETY_FEEDBACK_TIMEOUT;
    }
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        const pace_motor_state_t *state = &registry->motors[index];
        if (!state->rx_valid || (state->rx_age_us > limits->feedback_timeout_us))
        {
            flags |= PACE_SAFETY_FEEDBACK_TIMEOUT;
            continue;
        }
        if (state->motor_type == PACE_MOTOR_TYPE_DM8009)
        {
            if (limits->position_limit_enabled &&
                ((state->feedback.dm.position_rad < limits->dm_position_min_rad[index]) ||
                 (state->feedback.dm.position_rad > limits->dm_position_max_rad[index])))
            {
                flags |= PACE_SAFETY_POSITION;
            }
            if (fabsf(state->feedback.dm.velocity_rad_s) > limits->dm_max_velocity_rad_s)
            {
                flags |= PACE_SAFETY_DM_VELOCITY;
            }
            if (fabsf(state->feedback.dm.torque_nm) > limits->dm_max_torque_nm)
            {
                flags |= PACE_SAFETY_DM_TORQUE;
            }
            if ((state->feedback.dm.temperature_mos_c > limits->dm_max_temperature_c) ||
                (state->feedback.dm.temperature_rotor_c > limits->dm_max_temperature_c))
            {
                flags |= PACE_SAFETY_DM_TEMPERATURE;
            }
        }
        else
        {
            if (fabsf(state->feedback.lk.velocity_rad_s) > limits->lk_max_velocity_rad_s)
            {
                flags |= PACE_SAFETY_LK_VELOCITY;
            }
            if (fabsf(state->feedback.lk.current_a) > limits->lk_max_current_a)
            {
                flags |= PACE_SAFETY_LK_CURRENT;
            }
            if (state->feedback.lk.temperature_c > limits->lk_max_temperature_c)
            {
                flags |= PACE_SAFETY_LK_TEMPERATURE;
            }
        }
    }
    if (can_fault)
    {
        flags |= PACE_SAFETY_CAN;
    }
    if (uart_overflow)
    {
        flags |= PACE_SAFETY_UART_OVERFLOW;
    }
    if ((now_us - session_start_us) > limits->session_timeout_us)
    {
        flags |= PACE_SAFETY_DURATION;
    }
    return flags;
}
