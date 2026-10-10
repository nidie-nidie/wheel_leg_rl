#include "pace_motor_registry.h"

#include <string.h>

#include "dm_motor.h"
#include "lk_motor.h"
#include "pace_motor_manifest.h"

#define PACE_TWO_PI_F (6.28318530717958647692f)

void pace_motor_registry_init(pace_motor_registry_t *registry)
{
    uint8_t index;
    if (registry == 0)
    {
        return;
    }
    memset(registry, 0, sizeof(*registry));
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        registry->motors[index].motor_type = pace_motor_manifest[index].motor_type;
    }
}

static void pace_motor_registry_unwrap_lk(pace_motor_registry_t *registry,
                                          uint8_t canonical_index,
                                          pace_lk_feedback_t *feedback)
{
    const pace_motor_manifest_entry_t *entry = &pace_motor_manifest[canonical_index];
    const uint8_t slot = entry->hardware_slot;

    if (!registry->lk_encoder_initialized[slot])
    {
        registry->lk_last_encoder[slot] = feedback->encoder_raw;
        registry->lk_encoder_initialized[slot] = true;
    }
    else
    {
        const int32_t delta = (int32_t)feedback->encoder_raw - (int32_t)registry->lk_last_encoder[slot];
        if (delta > 32767)
        {
            registry->motors[canonical_index].feedback.lk.turn_count--;
        }
        else if (delta < -32768)
        {
            registry->motors[canonical_index].feedback.lk.turn_count++;
        }
        registry->lk_last_encoder[slot] = feedback->encoder_raw;
    }
    feedback->turn_count = registry->motors[canonical_index].feedback.lk.turn_count;
    feedback->position_rad += (float)feedback->turn_count * PACE_TWO_PI_F;
}

bool pace_motor_registry_apply_feedback(pace_motor_registry_t *registry,
                                        uint16_t can_id,
                                        const uint8_t data[PACE_CAN_CLASSIC_MAX_DATA],
                                        uint8_t dlc,
                                        uint32_t rx_timestamp_us)
{
    uint8_t index;
    pace_motor_state_t *state;
    if ((registry == 0) || (data == 0))
    {
        return false;
    }
    index = pace_motor_index_from_feedback_id(can_id);
    if (index == PACE_MOTOR_INDEX_INVALID)
    {
        return false;
    }
    state = &registry->motors[index];
    if (state->motor_type == PACE_MOTOR_TYPE_DM8009)
    {
        if (!pace_dm_decode_feedback(data, dlc, &state->feedback.dm))
        {
            return false;
        }
    }
    else if (state->motor_type == PACE_MOTOR_TYPE_LK9025)
    {
        pace_lk_feedback_t decoded;
        if (!pace_lk_decode_feedback(data, dlc, &decoded))
        {
            return false;
        }
        pace_motor_registry_unwrap_lk(registry, index, &decoded);
        state->feedback.lk = decoded;
    }
    else
    {
        return false;
    }
    state->last_rx_timestamp_us = rx_timestamp_us;
    state->rx_age_us = 0U;
    state->rx_valid = true;
    state->online = true;
    state->rx_count++;
    return true;
}

bool pace_motor_registry_confirm_tx(pace_motor_registry_t *registry,
                                    const pace_encoded_command_t *encoded,
                                    uint32_t tx_timestamp_us)
{
    pace_motor_state_t *state;
    if ((registry == 0) || (encoded == 0) || (encoded->canonical_index >= PACE_MOTOR_COUNT))
    {
        return false;
    }
    state = &registry->motors[encoded->canonical_index];
    if (state->motor_type != encoded->motor_type)
    {
        return false;
    }
    state->last_confirmed_command = *encoded;
    state->last_tx_timestamp_us = tx_timestamp_us;
    state->tx_age_us = 0U;
    state->tx_valid = true;
    state->tx_count++;
    return true;
}

void pace_motor_registry_refresh_ages(pace_motor_registry_t *registry,
                                      uint32_t now_us,
                                      uint32_t offline_timeout_us)
{
    uint8_t index;
    if (registry == 0)
    {
        return;
    }
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        pace_motor_state_t *state = &registry->motors[index];
        if (state->rx_valid)
        {
            state->rx_age_us = now_us - state->last_rx_timestamp_us;
            state->online = state->rx_age_us <= offline_timeout_us;
        }
        if (state->tx_valid)
        {
            state->tx_age_us = now_us - state->last_tx_timestamp_us;
        }
    }
}

const pace_motor_state_t *pace_motor_registry_get(const pace_motor_registry_t *registry,
                                                  uint8_t canonical_index)
{
    if ((registry == 0) || (canonical_index >= PACE_MOTOR_COUNT))
    {
        return 0;
    }
    return &registry->motors[canonical_index];
}
