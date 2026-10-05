#ifndef PACE_UART_PORT_H
#define PACE_UART_PORT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*pace_uart_rx_callback_t)(const uint8_t *data, uint16_t length);
typedef void (*pace_uart_tx_callback_t)(bool success);

typedef struct
{
    volatile bool tx_busy;
    volatile bool tx_error;
    uint32_t tx_complete_count;
    uint32_t tx_error_count;
    uint32_t rx_event_count;
    uint32_t rx_error_count;
} pace_uart_port_stats_t;

void pace_uart_port_init(void);
void pace_uart_port_set_tx_callback(pace_uart_tx_callback_t callback);
bool pace_uart_port_start_rx(uint8_t *buffer, uint16_t capacity,
                             pace_uart_rx_callback_t callback);
bool pace_uart_port_send_dma(const uint8_t *data, uint16_t length);
bool pace_uart_port_is_busy(void);
const pace_uart_port_stats_t *pace_uart_port_stats(void);

#ifdef __cplusplus
}
#endif

#endif
