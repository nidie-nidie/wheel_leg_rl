#include "dm_motor.h"

#include <math.h>
#include <string.h>

#include "pace_motor_manifest.h"

static float pace_clampf(float value, float minimum, float maximum, bool *saturated)
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

static uint16_t pace_dm_float_to_uint(float value,
                                      float minimum,
                                      float maximum,
                                      uint8_t bits,
                                      bool *saturated)
{
    const uint32_t maximum_code = (1UL << bits) - 1UL;
    const float clamped = pace_clampf(value, minimum, maximum, saturated);
    const float normalized = (clamped - minimum) / (maximum - minimum);
    return (uint16_t)(normalized * (float)maximum_code);
}

float pace_dm_uint_to_float(uint16_t raw, float minimum, float maximum, uint8_t bits)
{
    const uint32_t maximum_code = (1UL << bits) - 1UL;
    return ((float)raw * (maximum - minimum) / (float)maximum_code) + minimum;
}

bool pace_dm_pack_mit(uint8_t canonical_index,
                      float position_rad,
                      float velocity_rad_s,
                      float kp,
                      float kd,
                      float torque_nm,
                      pace_encoded_command_t *encoded)
{
    const pace_motor_manifest_entry_t *entry;
    bool saturated = false;

    if ((encoded == 0) || !isfinite(position_rad) || !isfinite(velocity_rad_s) ||
        !isfinite(kp) || !isfinite(kd) || !isfinite(torque_nm))
    {
        return false;
    }
    entry = pace_motor_manifest_get(canonical_index);
    if ((entry == 0) || (entry->motor_type != PACE_MOTOR_TYPE_DM8009))
    {
        return false;
    }

    memset(encoded, 0, sizeof(*encoded));
    encoded->canonical_index = canonical_index;
    encoded->motor_type = PACE_MOTOR_TYPE_DM8009;
    encoded->command_mode = PACE_COMMAND_DM_MIT;
    encoded->can_id = entry->command_can_id;
    encoded->dlc = 8U;
    encoded->raw.dm.q = pace_dm_float_to_uint(position_rad, PACE_DM_POSITION_MIN_RAD, PACE_DM_POSITION_MAX_RAD, 16U, &saturated);
    encoded->raw.dm.dq = pace_dm_float_to_uint(velocity_rad_s, PACE_DM_VELOCITY_MIN_RAD_S, PACE_DM_VELOCITY_MAX_RAD_S, 12U, &saturated);
    encoded->raw.dm.kp = pace_dm_float_to_uint(kp, PACE_DM_KP_MIN, PACE_DM_KP_MAX, 12U, &saturated);
    encoded->raw.dm.kd = pace_dm_float_to_uint(kd, PACE_DM_KD_MIN, PACE_DM_KD_MAX, 12U, &saturated);
    encoded->raw.dm.tau = pace_dm_float_to_uint(torque_nm, PACE_DM_TORQUE_MIN_NM, PACE_DM_TORQUE_MAX_NM, 12U, &saturated);

    encoded->data[0] = (uint8_t)(encoded->raw.dm.q >> 8);
    encoded->data[1] = (uint8_t)encoded->raw.dm.q;
    encoded->data[2] = (uint8_t)(encoded->raw.dm.dq >> 4);
    encoded->data[3] = (uint8_t)(((encoded->raw.dm.dq & 0x0FU) << 4) | (encoded->raw.dm.kp >> 8));
    encoded->data[4] = (uint8_t)encoded->raw.dm.kp;
    encoded->data[5] = (uint8_t)(encoded->raw.dm.kd >> 4);
    encoded->data[6] = (uint8_t)(((encoded->raw.dm.kd & 0x0FU) << 4) | (encoded->raw.dm.tau >> 8));
    encoded->data[7] = (uint8_t)encoded->raw.dm.tau;
    encoded->saturated = saturated;
    return true;
}

bool pace_dm_pack_special(uint8_t canonical_index,
                          pace_command_mode_t command_mode,
                          pace_encoded_command_t *encoded)
{
    const pace_motor_manifest_entry_t *entry;
    uint8_t command_byte;

    if (encoded == 0)
    {
        return false;
    }
    entry = pace_motor_manifest_get(canonical_index);
    if ((entry == 0) || (entry->motor_type != PACE_MOTOR_TYPE_DM8009))
    {
        return false;
    }
    switch (command_mode)
    {
    case PACE_COMMAND_DM_ENABLE:
        command_byte = 0xFCU;
        break;
    case PACE_COMMAND_DM_DISABLE:
        command_byte = 0xFDU;
        break;
    case PACE_COMMAND_DM_SAVE_ZERO:
        command_byte = 0xFEU;
        break;
    default:
        return false;
    }

    memset(encoded, 0, sizeof(*encoded));
    encoded->canonical_index = canonical_index;
    encoded->motor_type = PACE_MOTOR_TYPE_DM8009;
    encoded->command_mode = command_mode;
    encoded->can_id = entry->command_can_id;
    encoded->dlc = 8U;
    memset(encoded->data, 0xFF, sizeof(encoded->data));
    encoded->data[7] = command_byte;
    return true;
}

bool pace_dm_decode_feedback(const uint8_t data[PACE_CAN_CLASSIC_MAX_DATA],
                             uint8_t dlc,
                             pace_dm_feedback_t *feedback)
{
    if ((data == 0) || (feedback == 0) || (dlc != 8U))
    {
        return false;
    }
    feedback->state = data[0] >> 4;
    feedback->q_raw = (uint16_t)(((uint16_t)data[1] << 8) | data[2]);
    feedback->dq_raw = (uint16_t)(((uint16_t)data[3] << 4) | ((uint16_t)data[4] >> 4));
    feedback->tau_raw = (uint16_t)((((uint16_t)data[4] & 0x0FU) << 8) | data[5]);
    feedback->position_rad = pace_dm_uint_to_float(feedback->q_raw, PACE_DM_POSITION_MIN_RAD, PACE_DM_POSITION_MAX_RAD, 16U);
    feedback->velocity_rad_s = pace_dm_uint_to_float(feedback->dq_raw, PACE_DM_VELOCITY_MIN_RAD_S, PACE_DM_VELOCITY_MAX_RAD_S, 12U);
    feedback->torque_nm = pace_dm_uint_to_float(feedback->tau_raw, PACE_DM_TORQUE_MIN_NM, PACE_DM_TORQUE_MAX_NM, 12U);
    feedback->temperature_mos_c = data[6];
    feedback->temperature_rotor_c = data[7];
    return true;
}
