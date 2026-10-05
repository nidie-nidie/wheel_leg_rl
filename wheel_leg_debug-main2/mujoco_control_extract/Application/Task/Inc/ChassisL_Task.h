#ifndef __CHASSISL_TASK_H
#define __CHASSISL_TASK_H

#include "main.h"
#include "Motor.h"
#include "Chassis_Task.h"

extern void ChassisL_task(void);
extern void ChassisL_init(void);
extern void ChassisL_feedback_update(void);
extern void ChassisL_control_loop(void);

#endif
