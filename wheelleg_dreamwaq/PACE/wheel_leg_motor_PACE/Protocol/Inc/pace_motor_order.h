#ifndef PACE_MOTOR_ORDER_H
#define PACE_MOTOR_ORDER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PACE_MOTOR_ORDER_VERSION 1U
#define PACE_MOTOR_COUNT 6U
#define PACE_DM_MOTOR_COUNT 4U
#define PACE_LK_MOTOR_COUNT 2U
#define PACE_MOTOR_INDEX_INVALID 0xFFU

typedef enum
{
    PACE_MOTOR_L_FRONT = 0,
    PACE_MOTOR_L_REAR = 1,
    PACE_MOTOR_R_REAR = 2,
    PACE_MOTOR_R_FRONT = 3,
    PACE_MOTOR_L_WHEEL = 4,
    PACE_MOTOR_R_WHEEL = 5
} pace_motor_index_t;

typedef enum
{
    PACE_MOTOR_TYPE_DM8009 = 0,
    PACE_MOTOR_TYPE_LK9025 = 1
} pace_motor_type_t;

typedef enum
{
    PACE_ACTUATOR_LEG_DM = 0,
    PACE_ACTUATOR_WHEEL_LK = 1
} pace_actuator_family_t;

typedef char pace_motor_count_must_be_six[(PACE_MOTOR_COUNT == 6U) ? 1 : -1];
typedef char pace_right_rear_must_precede_front[(PACE_MOTOR_R_REAR < PACE_MOTOR_R_FRONT) ? 1 : -1];

#ifdef __cplusplus
}
#endif

#endif
