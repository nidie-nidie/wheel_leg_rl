#ifndef PACE_APP_H
#define PACE_APP_H

#include <stdbool.h>

#include "pace_experiment.h"

#ifdef __cplusplus
extern "C" {
#endif

bool pace_app_init(void);
void pace_app_capture_task(void const *argument);
void pace_app_transport_task(void const *argument);
pace_experiment_state_t pace_app_state(void);

#ifdef __cplusplus
}
#endif

#endif
