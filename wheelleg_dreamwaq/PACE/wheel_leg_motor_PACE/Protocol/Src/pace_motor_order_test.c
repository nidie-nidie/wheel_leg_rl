#include "pace_motor_manifest.h"

int pace_motor_order_self_test(void)
{
    uint8_t i;
    uint8_t j;
    for (i = 0U; i < PACE_MOTOR_COUNT; ++i)
    {
        const pace_motor_manifest_entry_t *entry = pace_motor_manifest_get(i);
        if ((entry == 0) || (entry->canonical_index != i) ||
            (pace_motor_index_from_can_id(entry->feedback_can_id, entry->motor_type) != i))
        {
            return 0;
        }
        for (j = (uint8_t)(i + 1U); j < PACE_MOTOR_COUNT; ++j)
        {
            if ((entry->device_id == pace_motor_manifest[j].device_id) ||
                (entry->feedback_can_id == pace_motor_manifest[j].feedback_can_id))
            {
                return 0;
            }
        }
    }
    return (pace_motor_manifest[PACE_MOTOR_R_REAR].hardware_slot == 2U) &&
           (pace_motor_manifest[PACE_MOTOR_R_FRONT].hardware_slot == 3U);
}
