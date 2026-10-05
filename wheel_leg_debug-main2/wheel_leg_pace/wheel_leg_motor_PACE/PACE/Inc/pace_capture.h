#ifndef PACE_CAPTURE_H
#define PACE_CAPTURE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PACE_CAPTURE_SLOT_SIZE 128U
#define PACE_CAPTURE_SLOT_COUNT 128U

typedef struct
{
    uint16_t length;
    uint8_t data[PACE_CAPTURE_SLOT_SIZE];
} pace_capture_slot_t;

typedef struct
{
    pace_capture_slot_t slots[PACE_CAPTURE_SLOT_COUNT];
    volatile uint16_t head;
    volatile uint16_t tail;
    volatile bool overflow;
    uint32_t pushed_frames;
    uint32_t popped_frames;
    uint32_t rejected_frames;
    uint16_t high_water;
} pace_capture_t;

void pace_capture_init(pace_capture_t *capture);
bool pace_capture_push(pace_capture_t *capture, const uint8_t *frame, uint16_t length);
bool pace_capture_peek(const pace_capture_t *capture, const uint8_t **frame, uint16_t *length);
bool pace_capture_pop(pace_capture_t *capture);
uint16_t pace_capture_depth(const pace_capture_t *capture);
bool pace_capture_overflowed(const pace_capture_t *capture);

#ifdef __cplusplus
}
#endif

#endif
