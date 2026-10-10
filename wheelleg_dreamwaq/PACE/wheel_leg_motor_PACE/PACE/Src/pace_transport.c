#include "pace_transport.h"

#include <string.h>

#include "pace_uart_port.h"

#define PACE_UART_TX_TIMEOUT_MS 20U
#define PACE_UART_TX_MAX_CONSECUTIVE_FAILURES 3U

static pace_capture_t *pace_transport_capture;
static pace_transport_stats_t pace_transport_state;
static uint8_t pace_transport_consecutive_failures;

static void pace_transport_start_next(void)
{
    const uint8_t *frame;
    uint16_t length;
    if ((pace_transport_capture == 0) || pace_transport_state.inflight ||
        pace_transport_state.failed ||
        !pace_capture_peek(pace_transport_capture, &frame, &length))
    {
        return;
    }
    pace_transport_state.inflight = true;
    if (!pace_uart_port_send_blocking(frame, length, PACE_UART_TX_TIMEOUT_MS))
    {
        pace_transport_state.inflight = false;
        pace_transport_state.start_failures++;
        pace_transport_state.retries++;
        pace_transport_consecutive_failures++;
        if (pace_transport_consecutive_failures >=
            PACE_UART_TX_MAX_CONSECUTIVE_FAILURES)
        {
            pace_transport_state.failed = true;
        }
        return;
    }
    (void)pace_capture_pop(pace_transport_capture);
    pace_transport_state.frames_sent++;
    pace_transport_consecutive_failures = 0U;
    pace_transport_state.inflight = false;
}

void pace_transport_init(pace_capture_t *capture)
{
    pace_transport_capture = capture;
    memset(&pace_transport_state, 0, sizeof(pace_transport_state));
    pace_transport_consecutive_failures = 0U;
    pace_uart_port_set_tx_callback(0);
}

void pace_transport_kick(void)
{
    /* Only pace_app_transport_task calls this function. Keeping UART ownership
     * in one task avoids queue/DMA state transitions from two tasks plus ISR. */
    pace_transport_start_next();
}

const pace_transport_stats_t *pace_transport_stats(void)
{
    return &pace_transport_state;
}
