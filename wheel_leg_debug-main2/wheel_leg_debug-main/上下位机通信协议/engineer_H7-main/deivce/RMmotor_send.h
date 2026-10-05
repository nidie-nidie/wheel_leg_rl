#ifndef RMMOTOR_SEND_H
#define RMMOTOR_SEND_H

#include "bsp_fdcan.h"

void RMmotor_send_4(unsigned short id, FDCAN_HandleTypeDef *hcan, short cmd1, short cmd2, short cmd3, short cmd4);

#endif // RMMOTOR_SEND_H
