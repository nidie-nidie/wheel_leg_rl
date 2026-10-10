#ifndef WHEEL_LEG_MOTOR_PACE_TIM_H
#define WHEEL_LEG_MOTOR_PACE_TIM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

extern TIM_HandleTypeDef htim5;
extern TIM_HandleTypeDef htim6;

void MX_TIM5_Init(void);
void MX_TIM6_Init(void);

#ifdef __cplusplus
}
#endif

#endif
