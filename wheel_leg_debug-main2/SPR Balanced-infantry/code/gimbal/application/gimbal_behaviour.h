#ifndef GIMBAL_BEHAVIOUR_H
#define GIMBAL_BEHAVIOUR_H

#include "struct_typedef.h"
#include "gimbal_task.h"

#define RC_LEFT_Y 3
#define RC_LEFT_X 2
#define RC_RIGHT_Y 1
#define RC_RIGHT_X 0

#define RC_DEADBAND 10

#define rc_deadband_limit(input, output, dealine)        \
    {                                                    \
        if ((input) > (dealine) || (input) < -(dealine)) \
        {                                                \
            (output) = (input);                          \
        }                                                \
        else                                             \
        {                                                \
            (output) = 0;                                \
        }                                                \
    }

extern void gimbal_behaviour_mode_set(gimbal_move_t *gimbal_mode_set);
extern void gimbal_behaviour_control_set(float *add_yaw, float *add_pitch, gimbal_move_t *gimbal_control_set);

#endif
