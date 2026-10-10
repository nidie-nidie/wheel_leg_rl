#ifndef PACE_TIME_H
#define PACE_TIME_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*pace_time_tick_callback_t)(uint32_t now_us);

bool pace_time_init(void);
uint32_t pace_time_now_us(void);
uint32_t pace_time_expand_fdcan_timestamp(uint16_t raw_timestamp, uint32_t observed_now_us);
void pace_time_set_tick_callback(pace_time_tick_callback_t callback);
uint32_t pace_time_tick_count(void);

#ifdef __cplusplus
}
#endif

#endif
