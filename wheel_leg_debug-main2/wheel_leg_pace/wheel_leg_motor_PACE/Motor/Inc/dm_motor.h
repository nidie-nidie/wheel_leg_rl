#ifndef PACE_DM_MOTOR_H
#define PACE_DM_MOTOR_H

#include <stdbool.h>
#include <stdint.h>

#include "pace_motor_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PACE_DM_POSITION_MIN_RAD (-12.5f)
#define PACE_DM_POSITION_MAX_RAD (12.5f)
#define PACE_DM_VELOCITY_MIN_RAD_S (-45.0f)
#define PACE_DM_VELOCITY_MAX_RAD_S (45.0f)
#define PACE_DM_TORQUE_MIN_NM (-54.0f)
#define PACE_DM_TORQUE_MAX_NM (54.0f)
#define PACE_DM_KP_MIN (0.0f)
#define PACE_DM_KP_MAX (500.0f)
#define PACE_DM_KD_MIN (0.0f)
#define PACE_DM_KD_MAX (5.0f)

bool pace_dm_pack_mit(uint8_t canonical_index,
                      float position_rad,
                      float velocity_rad_s,
                      float kp,
                      float kd,
                      float torque_nm,
                      pace_encoded_command_t *encoded);
bool pace_dm_pack_special(uint8_t canonical_index,
                          pace_command_mode_t command_mode,
                          pace_encoded_command_t *encoded);
bool pace_dm_decode_feedback(const uint8_t data[PACE_CAN_CLASSIC_MAX_DATA],
                             uint8_t dlc,
                             pace_dm_feedback_t *feedback);
float pace_dm_uint_to_float(uint16_t raw, float minimum, float maximum, uint8_t bits);

#ifdef __cplusplus
}
#endif

#endif
