#ifndef __OBSERVE_TASK_H
#define __OBSERVE_TASK_H

#include "stdint.h"
#include "INS_Task.h"
#include "Chassis_Task.h"
#include "main.h"

typedef struct
{
	float wr_radps;
	float wl_radps;
	float vrb_mps;
	float vlb_mps;
	float aver_v_mps;
	float accel_measure_mps2;
	float filtered_v_mps;
	float filtered_a_mps2;
	float kalman_k[4];
	uint32_t update_tick_ms;
} Observe_Diagnostic_t;

extern volatile Observe_Diagnostic_t observe_diag;

extern void Observe_task(void);
extern void xvEstimateKF_Init(KalmanFilter_t *EstimateKF);
extern void xvEstimateKF_Update(KalmanFilter_t *EstimateKF, float acc, float vel);
extern float RAMP_float(float final, float now, float ramp);

#endif
