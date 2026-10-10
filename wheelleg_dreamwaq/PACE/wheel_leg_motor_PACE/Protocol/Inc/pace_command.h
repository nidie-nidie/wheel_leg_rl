#ifndef PACE_COMMAND_H
#define PACE_COMMAND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PACE_HOST_COMMAND_MAGIC_0 0x50U
#define PACE_HOST_COMMAND_MAGIC_1 0x43U
#define PACE_HOST_COMMAND_VERSION 1U
#define PACE_HOST_COMMAND_MIN_SIZE 14U
#define PACE_HOST_COMMAND_MAX_SIZE 128U

typedef enum
{
    PACE_HOST_CONFIGURE = 0x01,
    PACE_HOST_START = 0x02,
    PACE_HOST_STOP = 0x03,
    PACE_HOST_STATUS = 0x04,
    PACE_HOST_ABORT = 0x05
} pace_host_command_id_t;

typedef struct
{
    bool sequence_initialized;
    uint32_t last_sequence;
} pace_command_parser_t;

typedef struct
{
    pace_host_command_id_t command_id;
    uint32_t sequence;
    const uint8_t *payload;
    uint16_t payload_length;
} pace_host_command_t;

void pace_command_parser_init(pace_command_parser_t *parser);
bool pace_command_decode(pace_command_parser_t *parser,
                         const uint8_t *frame,
                         size_t available,
                         pace_host_command_t *command);

#ifdef __cplusplus
}
#endif

#endif
