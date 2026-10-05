#include "pace_capture.h"

#include <string.h>

static uint16_t pace_capture_next(uint16_t index)
{
    index++;
    return (index >= PACE_CAPTURE_SLOT_COUNT) ? 0U : index;
}

void pace_capture_init(pace_capture_t *capture)
{
    if (capture != 0)
    {
        memset(capture, 0, sizeof(*capture));
    }
}

uint16_t pace_capture_depth(const pace_capture_t *capture)
{
    uint16_t head;
    uint16_t tail;
    if (capture == 0)
    {
        return 0U;
    }
    head = capture->head;
    tail = capture->tail;
    return (head >= tail) ? (uint16_t)(head - tail) :
                            (uint16_t)(PACE_CAPTURE_SLOT_COUNT - tail + head);
}

bool pace_capture_push(pace_capture_t *capture, const uint8_t *frame, uint16_t length)
{
    uint16_t next;
    uint16_t depth;
    if ((capture == 0) || (frame == 0) || (length == 0U) ||
        (length > PACE_CAPTURE_SLOT_SIZE))
    {
        return false;
    }
    next = pace_capture_next(capture->head);
    if (next == capture->tail)
    {
        capture->overflow = true;
        capture->rejected_frames++;
        return false;
    }
    capture->slots[capture->head].length = length;
    memcpy(capture->slots[capture->head].data, frame, length);
    capture->head = next;
    capture->pushed_frames++;
    depth = pace_capture_depth(capture);
    if (depth > capture->high_water)
    {
        capture->high_water = depth;
    }
    return true;
}

bool pace_capture_peek(const pace_capture_t *capture, const uint8_t **frame, uint16_t *length)
{
    if ((capture == 0) || (frame == 0) || (length == 0) ||
        (capture->tail == capture->head))
    {
        return false;
    }
    *frame = capture->slots[capture->tail].data;
    *length = capture->slots[capture->tail].length;
    return true;
}

bool pace_capture_pop(pace_capture_t *capture)
{
    if ((capture == 0) || (capture->tail == capture->head))
    {
        return false;
    }
    capture->tail = pace_capture_next(capture->tail);
    capture->popped_frames++;
    return true;
}

bool pace_capture_overflowed(const pace_capture_t *capture)
{
    return (capture != 0) && capture->overflow;
}
