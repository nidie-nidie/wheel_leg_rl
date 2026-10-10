#include "pace_command.h"

#include "pace_crc.h"

static uint16_t pace_command_get_u16(const uint8_t *input, size_t offset)
{
    return (uint16_t)((uint16_t)input[offset] | ((uint16_t)input[offset + 1U] << 8));
}

static uint32_t pace_command_get_u32(const uint8_t *input, size_t offset)
{
    return (uint32_t)input[offset] |
           ((uint32_t)input[offset + 1U] << 8) |
           ((uint32_t)input[offset + 2U] << 16) |
           ((uint32_t)input[offset + 3U] << 24);
}

static bool pace_command_id_is_valid(uint8_t command_id)
{
    return (command_id >= (uint8_t)PACE_HOST_CONFIGURE) &&
           (command_id <= (uint8_t)PACE_HOST_ABORT);
}

void pace_command_parser_init(pace_command_parser_t *parser)
{
    if (parser != 0)
    {
        parser->sequence_initialized = false;
        parser->last_sequence = 0U;
    }
}

bool pace_command_decode(pace_command_parser_t *parser,
                         const uint8_t *frame,
                         size_t available,
                         pace_host_command_t *command)
{
    uint16_t frame_length;
    uint32_t sequence;
    uint32_t expected_crc;
    uint32_t actual_crc;

    if ((parser == 0) || (frame == 0) || (command == 0) ||
        (available < PACE_HOST_COMMAND_MIN_SIZE) ||
        (frame[0] != PACE_HOST_COMMAND_MAGIC_0) ||
        (frame[1] != PACE_HOST_COMMAND_MAGIC_1) ||
        (frame[2] != PACE_HOST_COMMAND_VERSION) ||
        !pace_command_id_is_valid(frame[3]))
    {
        return false;
    }

    frame_length = pace_command_get_u16(frame, 4U);
    if ((frame_length < PACE_HOST_COMMAND_MIN_SIZE) ||
        (frame_length > PACE_HOST_COMMAND_MAX_SIZE) ||
        (available < frame_length))
    {
        return false;
    }
    expected_crc = pace_command_get_u32(frame, (size_t)frame_length - 4U);
    actual_crc = pace_crc32_iso_hdlc(frame, (size_t)frame_length - 4U);
    if (expected_crc != actual_crc)
    {
        return false;
    }

    sequence = pace_command_get_u32(frame, 6U);
    if (parser->sequence_initialized && (sequence != (parser->last_sequence + 1U)))
    {
        return false;
    }
    parser->sequence_initialized = true;
    parser->last_sequence = sequence;
    command->command_id = (pace_host_command_id_t)frame[3];
    command->sequence = sequence;
    command->payload = &frame[10];
    command->payload_length = (uint16_t)(frame_length - PACE_HOST_COMMAND_MIN_SIZE);
    return true;
}
