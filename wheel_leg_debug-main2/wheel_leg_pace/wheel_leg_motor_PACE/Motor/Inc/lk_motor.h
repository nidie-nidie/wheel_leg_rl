#ifndef PACE_LK_MOTOR_H
#define PACE_LK_MOTOR_H

#include <stdbool.h>
#include <stdint.h>

#include "pace_motor_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PACE_LK_TORQUE_CONSTANT_NM_PER_A (0.32f)
#define PACE_LK_CURRENT_COMMAND_PER_A (124.1212121212121f)
#define PACE_LK_CURRENT_A_PER_COMMAND (0.008056640625f)
#define PACE_LK_TORQUE_MIN_NM (-2.41f)
#define PACE_LK_TORQUE_MAX_NM (2.41f)
#define PACE_LK_RAD_TO_DEG (57.2957795131f)
#define PACE_LK_DEG_TO_RAD (0.0174532925f)

bool pace_lk_pack_torque(uint8_t canonical_index,
                         float torque_nm,
                         pace_encoded_command_t *encoded);
bool pace_lk_pack_velocity(uint8_t canonical_index,
                           float velocity_rad_s,
                           pace_encoded_command_t *encoded);
bool pace_lk_pack_special(uint8_t canonical_index,
                          pace_command_mode_t command_mode,
                          pace_encoded_command_t *encoded);
bool pace_lk_decode_feedback(const uint8_t data[PACE_CAN_CLASSIC_MAX_DATA],
                             uint8_t dlc,
                             pace_lk_feedback_t *feedback);

#ifdef __cplusplus
}
#endif

#endif
