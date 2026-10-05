#ifndef WHEEL_LEG_MOTOR_PACE_USART_H
#define WHEEL_LEG_MOTOR_PACE_USART_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

extern UART_HandleTypeDef huart7;
extern DMA_HandleTypeDef hdma_uart7_rx;
extern DMA_HandleTypeDef hdma_uart7_tx;

void MX_UART7_Init(void);

#ifdef __cplusplus
}
#endif

#endif
