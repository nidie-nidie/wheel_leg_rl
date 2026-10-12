#ifndef PACE_HOST_CMSIS_OS_H
#define PACE_HOST_CMSIS_OS_H
#include <stdint.h>
typedef int32_t osStatus;
#define osOK 0
osStatus osDelay(uint32_t millisec);
#endif
