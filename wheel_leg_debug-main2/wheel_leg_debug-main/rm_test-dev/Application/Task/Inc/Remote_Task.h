#ifndef __REMOTE_TASK_H
#define __REMOTE_TASK_H

#include "main.h"
#include "Chassis_Task.h"
#include "Remote_Control.h"
#include "INS_Task.h"

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

#define CHASSIS_X_CHANNEL DT7_CH_RV          // 前后的遥控器通道号码
#define CHASSIS_Y_CHANNEL DT7_CH_RV          // 左右的遥控器通道号码
#define CHASSIS_WZ_CHANNEL DT7_CH_LH         // 旋转的遥控器通道号码
#define CHASSIS_ANGLE_CHANNEL DT7_CH_ROLLER  // 腿摆角的遥控器通道号码
#define CHASSIS_LENGTH_CHANNEL DT7_CH_ROLLER // 腿长的遥控器通道号码
#define CHASSIS_ROLL_CHANNEL DT7_CH_ROLLER   // ROLL角的遥控器通道号码
#define CHASSIS_RC_DEADLINE 20               // 摇杆死区

void Remote_task(void);

#endif
