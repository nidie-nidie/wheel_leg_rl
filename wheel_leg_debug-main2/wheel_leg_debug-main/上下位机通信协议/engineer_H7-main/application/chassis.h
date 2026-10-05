#ifndef CHASSIS_H
#define CHASSIS_H
#include "kf_imu.h"
#include "bsp_dwt.h"
#include "M3508motors.h"
#include "pid.h"

#include "robot_def.h"
#include "RC_protocol.h"

#include "fdcan.h"
#include "bsp_fdcan.h"

extern M3508motor_t motor_lf,motor_lb,motor_rf,motor_rb;


void CHASSIS_INIT(void);

void CHASSIS_TASK(void);

#endif // CHASSIS_H
