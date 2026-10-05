#include "pace_transport.h"

#include <string.h>

#include "pace_uart_port.h"

static pace_capture_t *pace_transport_capture;
static pace_transport_stats_t pace_transport_state;

static void pace_transport_start_next(void)
{
    const uint8_t *frame;
    uint16_t length;
    if ((pace_transport_capture == 0) || pace_transport_state.inflight ||
        !pace_capture_peek(pace_transport_capture, &frame, &length))
    {
        return;
    }
    if (!pace_uart_port_send_dma(frame, length))
    {
        pace_transport_state.start_failures++;
        return;
    }
    pace_transport_state.inflight = true;
}

static void pace_transport_tx_complete(bool success)
{
    if (!pace_transport_state.inflight)
    {
        return;
    }
    if (!success)
    {
        pace_transport_state.failed = true;
        pace_transport_state.dma_failures++;
        pace_transport_state.inflight = false;
        return;
    }
    (void)pace_capture_pop(pace_transport_capture);
    pace_transport_state.frames_sent++;
    pace_transport_state.inflight = false;
    pace_transport_start_next();
}

void pace_transport_init(pace_capture_t *capture)
{
    pace_transport_capture = capture;
    memset(&pace_transport_state, 0, sizeof(pace_transport_state));
    pace_uart_port_set_tx_callback(pace_transport_tx_complete);
}

void pace_transport_kick(void)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    pace_transport_start_next();
    if (primask == 0U)
    {
        __enable_irq();
    }
}

const pace_transport_stats_t *pace_transport_stats(void)
{
    return &pace_transport_state;
}
