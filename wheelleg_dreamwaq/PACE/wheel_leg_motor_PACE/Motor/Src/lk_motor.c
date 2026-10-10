#include "lk_motor.h"

#include <limits.h>
#include <math.h>
#include <string.h>

#include "pace_motor_manifest.h"

#define PACE_PI_F (3.14159265358979323846f)

static float pace_lk_clampf(float value, float minimum, float maximum, bool *saturated)
{
    if (value < minimum)
    {
        *saturated = true;
        return minimum;
    }
    if (value > maximum)
    {
        *saturated = true;
        return maximum;
    }
    return value;
}

static bool pace_lk_prepare(uint8_t canonical_index,
                            pace_command_mode_t command_mode,
                            pace_encoded_command_t *encoded)
{
    const pace_motor_manifest_entry_t *entry;
    if (encoded == 0)
    {
        return false;
    }
    entry = pace_motor_manifest_get(canonical_index);
    if ((entry == 0) || (entry->motor_type != PACE_MOTOR_TYPE_LK9025))
    {
        return false;
    }
    memset(encoded, 0, sizeof(*encoded));
    encoded->canonical_index = canonical_index;
    encoded->motor_type = PACE_MOTOR_TYPE_LK9025;
    encoded->command_mode = command_mode;
    encoded->can_id = entry->command_can_id;
    encoded->dlc = 8U;
    return true;
}

bool pace_lk_pack_torque(uint8_t canonical_index,
                         float torque_nm,
                         pace_encoded_command_t *encoded)
{
    bool saturated = false;
    float clamped;
    int16_t iq_control;

    if (!isfinite(torque_nm) || !pace_lk_prepare(canonical_index, PACE_COMMAND_LK_TORQUE, encoded))
    {
        return false;
    }
    clamped = pace_lk_clampf(torque_nm, PACE_LK_TORQUE_MIN_NM, PACE_LK_TORQUE_MAX_NM, &saturated);
    iq_control = (int16_t)((clamped / PACE_LK_TORQUE_CONSTANT_NM_PER_A) * PACE_LK_CURRENT_COMMAND_PER_A);
    encoded->data[0] = 0xA1U;
    encoded->data[4] = (uint8_t)((uint16_t)iq_control & 0xFFU);
    encoded->data[5] = (uint8_t)(((uint16_t)iq_control >> 8) & 0xFFU);
    encoded->raw.lk.primary = iq_control;
    encoded->saturated = saturated;
    return true;
}

bool pace_lk_pack_velocity(uint8_t canonical_index,
                           float velocity_rad_s,
                           pace_encoded_command_t *encoded)
{
    double raw;
    int32_t speed_control;
    bool saturated = false;

    if (!isfinite(velocity_rad_s) || !pace_lk_prepare(canonical_index, PACE_COMMAND_LK_VELOCITY, encoded))
    {
        return false;
    }
    raw = (double)velocity_rad_s * (double)PACE_LK_RAD_TO_DEG * 100.0;
    if (raw > (double)INT32_MAX)
    {
        raw = (double)INT32_MAX;
        saturated = true;
    }
    else if (raw < (double)INT32_MIN)
    {
        raw = (double)INT32_MIN;
        saturated = true;
    }
    speed_control = (int32_t)raw;
    encoded->data[0] = 0xA2U;
    encoded->data[4] = (uint8_t)((uint32_t)speed_control & 0xFFU);
    encoded->data[5] = (uint8_t)(((uint32_t)speed_control >> 8) & 0xFFU);
    encoded->data[6] = (uint8_t)(((uint32_t)speed_control >> 16) & 0xFFU);
    encoded->data[7] = (uint8_t)(((uint32_t)speed_control >> 24) & 0xFFU);
    encoded->raw.lk.primary = speed_control;
    encoded->saturated = saturated;
    return true;
}

bool pace_lk_pack_special(uint8_t canonical_index,
                          pace_command_mode_t command_mode,
                          pace_encoded_command_t *encoded)
{
    uint8_t command_byte;
    switch (command_mode)
    {
    case PACE_COMMAND_LK_ENABLE:
        command_byte = 0x88U;
        break;
    case PACE_COMMAND_LK_DISABLE:
        command_byte = 0x80U;
        break;
    case PACE_COMMAND_LK_STOP:
        command_byte = 0x81U;
        break;
    default:
        return false;
    }
    if (!pace_lk_prepare(canonical_index, command_mode, encoded))
    {
        return false;
    }
    encoded->data[0] = command_byte;
    return true;
}

bool pace_lk_decode_feedback(const uint8_t data[PACE_CAN_CLASSIC_MAX_DATA],
                             uint8_t dlc,
                             pace_lk_feedback_t *feedback)
{
    if ((data == 0) || (feedback == 0) || (dlc != 8U))
    {
        return false;
    }
    feedback->status = data[0];
    feedback->temperature_c = (int8_t)data[1];
    feedback->iq_raw = (int16_t)(((uint16_t)data[3] << 8) | data[2]);
    feedback->speed_raw = (int16_t)(((uint16_t)data[5] << 8) | data[4]);
    feedback->encoder_raw = (uint16_t)(((uint16_t)data[7] << 8) | data[6]);
    feedback->turn_count = 0;
    feedback->position_rad = ((float)feedback->encoder_raw * (2.0f * PACE_PI_F) / 65535.0f) - PACE_PI_F;
    feedback->velocity_rad_s = (float)feedback->speed_raw * PACE_LK_DEG_TO_RAD;
    feedback->current_a = (float)feedback->iq_raw * PACE_LK_CURRENT_A_PER_COMMAND;
    feedback->torque_nm = feedback->current_a * PACE_LK_TORQUE_CONSTANT_NM_PER_A;
    return true;
}
