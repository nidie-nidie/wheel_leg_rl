#ifndef __OBSERVE_TASK_H
#define __OBSERVE_TASK_H

#include "stdint.h"
#include "INS_Task.h"
#include "Chassis_Task.h"
#include "main.h"

extern void Observe_task(void);
extern void xvEstimateKF_Init(KalmanFilter_t *EstimateKF);
extern void xvEstimateKF_Update(KalmanFilter_t *EstimateKF, float acc, float vel);
extern float RAMP_float(float final, float now, float ramp);

#endif
