#ifndef PACE_MOTOR_MANIFEST_H
#define PACE_MOTOR_MANIFEST_H

#include <stdbool.h>
#include <stdint.h>

#include "pace_motor_order.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PACE_MOTOR_MANIFEST_HASH 0x4D4F5431UL

typedef struct
{
    uint8_t canonical_index;
    const char *motor_name;
    pace_motor_type_t motor_type;
    uint8_t hardware_slot;
    uint16_t device_id;
    uint16_t command_can_id;
    uint16_t feedback_can_id;
    const char *mujoco_joint;
    const char *isaac_joint;
    int8_t position_sign;
    int8_t velocity_sign;
    int8_t torque_sign;
    float position_zero_rad;
    float gear_ratio;
    pace_actuator_family_t actuator_family;
    bool calibration_verified;
} pace_motor_manifest_entry_t;

/*
 * Firmware source of truth for order and IDs. Signs and zero offsets remain
 * unverified until the first unloaded hardware commissioning check.
 */
#define PACE_MOTOR_MANIFEST_ROWS(X) \
    X(PACE_MOTOR_L_FRONT, "L_front", PACE_MOTOR_TYPE_DM8009, 0U, 1U, 0x001U, 0x011U, "jIJ", "jIJ", 1, 1, 1, 0.0f, 1.0f, PACE_ACTUATOR_LEG_DM, false) \
    X(PACE_MOTOR_L_REAR, "L_rear", PACE_MOTOR_TYPE_DM8009, 1U, 2U, 0x002U, 0x012U, "jIO", "jIO", 1, 1, 1, 0.0f, 1.0f, PACE_ACTUATOR_LEG_DM, false) \
    X(PACE_MOTOR_R_REAR, "R_rear", PACE_MOTOR_TYPE_DM8009, 2U, 3U, 0x003U, 0x013U, "jAG", "jAG", 1, 1, 1, 0.0f, 1.0f, PACE_ACTUATOR_LEG_DM, false) \
    X(PACE_MOTOR_R_FRONT, "R_front", PACE_MOTOR_TYPE_DM8009, 3U, 6U, 0x006U, 0x016U, "jAB", "jAB", 1, 1, 1, 0.0f, 1.0f, PACE_ACTUATOR_LEG_DM, false) \
    X(PACE_MOTOR_L_WHEEL, "L_wheel", PACE_MOTOR_TYPE_LK9025, 0U, 4U, 0x144U, 0x144U, "jwheel_left", "jwheel_left", 1, 1, 1, 0.0f, 1.0f, PACE_ACTUATOR_WHEEL_LK, false) \
    X(PACE_MOTOR_R_WHEEL, "R_wheel", PACE_MOTOR_TYPE_LK9025, 1U, 5U, 0x145U, 0x145U, "jwheel_right", "jwheel_right", 1, 1, 1, 0.0f, 1.0f, PACE_ACTUATOR_WHEEL_LK, false)

extern const pace_motor_manifest_entry_t pace_motor_manifest[PACE_MOTOR_COUNT];

const pace_motor_manifest_entry_t *pace_motor_manifest_get(uint8_t canonical_index);
uint8_t pace_motor_index_from_can_id(uint16_t can_id, pace_motor_type_t motor_type);
uint8_t pace_motor_index_from_feedback_id(uint16_t can_id);

#ifdef __cplusplus
}
#endif

#endif
