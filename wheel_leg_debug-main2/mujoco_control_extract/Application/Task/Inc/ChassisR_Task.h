#ifndef __CHASSISR_TASK_H
#define __CHASSISR_TASK_H

#include "main.h"
#include "Motor.h"
#include "Chassis_Task.h"

extern void ChassisR_task(void);
extern void ChassisR_init(void);
extern void Pensation_init(void);
extern void ChassisR_feedback_update(void);
extern void ChassisR_control_loop(void);

#endif
