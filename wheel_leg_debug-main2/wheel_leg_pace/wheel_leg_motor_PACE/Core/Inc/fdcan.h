#ifndef WHEEL_LEG_MOTOR_PACE_FDCAN_H
#define WHEEL_LEG_MOTOR_PACE_FDCAN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

extern FDCAN_HandleTypeDef hfdcan3;
void MX_FDCAN3_Init(void);

#ifdef __cplusplus
}
#endif

#endif
