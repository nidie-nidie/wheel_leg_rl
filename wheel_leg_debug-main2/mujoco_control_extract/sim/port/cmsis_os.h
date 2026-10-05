#ifndef SIM_PORT_CMSIS_OS_H
#define SIM_PORT_CMSIS_OS_H

#include <stdint.h>
#include <stdlib.h>

typedef uint32_t TickType_t;

void osDelay(uint32_t ms);
TickType_t osKernelSysTick(void);

#define pvPortMalloc malloc

#endif

