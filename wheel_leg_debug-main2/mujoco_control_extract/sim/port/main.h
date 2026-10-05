#ifndef SIM_PORT_MAIN_H
#define SIM_PORT_MAIN_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef float fp32;

#ifndef PI
#define PI 3.14159265358979323846f
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

#ifndef M_PI_6
#define M_PI_6 0.52359877559829887308
#endif

#ifndef M_2PI
#define M_2PI 6.28318530717958647692f
#endif

#ifndef GRAVITY
#define GRAVITY 9.81f
#endif

#ifndef __PACKED_STRUCT
#define __PACKED_STRUCT struct __attribute__((packed))
#endif

uint32_t HAL_GetTick(void);

#endif

