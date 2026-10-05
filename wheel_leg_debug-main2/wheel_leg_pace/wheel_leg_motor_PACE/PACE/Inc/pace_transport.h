#ifndef PACE_TRANSPORT_H
#define PACE_TRANSPORT_H

#include <stdbool.h>
#include <stdint.h>

#include "pace_capture.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    bool inflight;
    bool failed;
    uint32_t frames_sent;
    uint32_t start_failures;
    uint32_t dma_failures;
} pace_transport_stats_t;

void pace_transport_init(pace_capture_t *capture);
void pace_transport_kick(void);
const pace_transport_stats_t *pace_transport_stats(void);

#ifdef __cplusplus
}
#endif

#endif
