#ifndef SUCKER_H
#define SUCKER_H
#include "robot_def.h"

#include "usart.h"
#include "RC_protocol.h"
#include "pid.h"
#include "kf_imu.h"
#include "usart.h"
#include "tim.h"
#include "robot_cmd.h"
void SUCKER_INIT(void);
	
void SUCKER_TASK(void);


#endif // CMD_H