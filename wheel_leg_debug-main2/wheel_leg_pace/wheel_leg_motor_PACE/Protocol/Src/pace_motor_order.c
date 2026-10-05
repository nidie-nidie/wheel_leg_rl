#include "pace_motor_manifest.h"

#define PACE_MANIFEST_ENTRY(index_, name_, type_, slot_, device_id_, command_id_, feedback_id_, mujoco_, isaac_, pos_sign_, vel_sign_, torque_sign_, zero_, ratio_, family_, verified_) \
    { (uint8_t)(index_), (name_), (type_), (uint8_t)(slot_), (uint16_t)(device_id_), (uint16_t)(command_id_), (uint16_t)(feedback_id_), (mujoco_), (isaac_),                 \
      (int8_t)(pos_sign_), (int8_t)(vel_sign_), (int8_t)(torque_sign_), (zero_), (ratio_), (family_), (verified_) },

const pace_motor_manifest_entry_t pace_motor_manifest[PACE_MOTOR_COUNT] = {
    PACE_MOTOR_MANIFEST_ROWS(PACE_MANIFEST_ENTRY)};

#undef PACE_MANIFEST_ENTRY

const pace_motor_manifest_entry_t *pace_motor_manifest_get(uint8_t canonical_index)
{
    if (canonical_index >= PACE_MOTOR_COUNT)
    {
        return 0;
    }
    return &pace_motor_manifest[canonical_index];
}

uint8_t pace_motor_index_from_can_id(uint16_t can_id, pace_motor_type_t motor_type)
{
    uint8_t index;
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        const pace_motor_manifest_entry_t *entry = &pace_motor_manifest[index];
        if ((entry->motor_type == motor_type) &&
            ((entry->command_can_id == can_id) || (entry->feedback_can_id == can_id)))
        {
            return index;
        }
    }
    return PACE_MOTOR_INDEX_INVALID;
}

uint8_t pace_motor_index_from_feedback_id(uint16_t can_id)
{
    uint8_t index;
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        if (pace_motor_manifest[index].feedback_can_id == can_id)
        {
            return index;
        }
    }
    return PACE_MOTOR_INDEX_INVALID;
}
