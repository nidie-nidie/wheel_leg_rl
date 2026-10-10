#ifndef PACE_MOTOR_TYPES_H
#define PACE_MOTOR_TYPES_H

#include <stdbool.h>
#include <stdint.h>

#include "pace_motor_order.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PACE_CAN_CLASSIC_MAX_DATA 8U

typedef enum
{
    PACE_COMMAND_NONE = 0,
    PACE_COMMAND_DM_MIT,
    PACE_COMMAND_DM_ENABLE,
    PACE_COMMAND_DM_DISABLE,
    PACE_COMMAND_DM_SAVE_ZERO,
    PACE_COMMAND_LK_TORQUE,
    PACE_COMMAND_LK_VELOCITY,
    PACE_COMMAND_LK_ENABLE,
    PACE_COMMAND_LK_DISABLE,
    PACE_COMMAND_LK_STOP
} pace_command_mode_t;

typedef struct
{
    uint16_t q;
    uint16_t dq;
    uint16_t kp;
    uint16_t kd;
    uint16_t tau;
} pace_dm_command_raw_t;

typedef struct
{
    int32_t primary;
} pace_lk_command_raw_t;

typedef union
{
    pace_dm_command_raw_t dm;
    pace_lk_command_raw_t lk;
} pace_command_raw_t;

typedef struct
{
    uint8_t canonical_index;
    pace_motor_type_t motor_type;
    pace_command_mode_t command_mode;
    uint16_t can_id;
    uint8_t dlc;
    uint8_t data[PACE_CAN_CLASSIC_MAX_DATA];
    pace_command_raw_t raw;
    bool saturated;
} pace_encoded_command_t;

typedef struct
{
    uint8_t state;
    uint16_t q_raw;
    uint16_t dq_raw;
    uint16_t tau_raw;
    float position_rad;
    float velocity_rad_s;
    float torque_nm;
    uint8_t temperature_mos_c;
    uint8_t temperature_rotor_c;
} pace_dm_feedback_t;

typedef struct
{
    uint8_t status;
    int8_t temperature_c;
    int16_t iq_raw;
    int16_t speed_raw;
    uint16_t encoder_raw;
    int32_t turn_count;
    float position_rad;
    float velocity_rad_s;
    float current_a;
    float torque_nm;
} pace_lk_feedback_t;

typedef union
{
    pace_dm_feedback_t dm;
    pace_lk_feedback_t lk;
} pace_feedback_t;

typedef struct
{
    pace_motor_type_t motor_type;
    bool online;
    bool rx_valid;
    bool tx_valid;
    uint32_t last_rx_timestamp_us;
    uint32_t last_tx_timestamp_us;
    uint32_t rx_age_us;
    uint32_t tx_age_us;
    uint32_t rx_count;
    uint32_t tx_count;
    pace_feedback_t feedback;
    pace_encoded_command_t last_confirmed_command;
} pace_motor_state_t;

#ifdef __cplusplus
}
#endif

#endif
