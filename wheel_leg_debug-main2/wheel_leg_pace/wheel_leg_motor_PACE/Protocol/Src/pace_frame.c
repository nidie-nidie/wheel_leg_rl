#include "pace_frame.h"

#include <string.h>

#include "pace_crc.h"

typedef char pace_sample_wire_size_check[(26U + (4U * 16U) + (2U * 16U) + 4U == PACE_SAMPLE_FRAME_SIZE) ? 1 : -1];
typedef char pace_status_wire_size_check[(60U + 4U == PACE_STATUS_BASE_FRAME_SIZE) ? 1 : -1];

static void pace_put_u16(uint8_t *output, size_t offset, uint16_t value)
{
    output[offset] = (uint8_t)(value & 0xFFU);
    output[offset + 1U] = (uint8_t)(value >> 8);
}

static void pace_put_i16(uint8_t *output, size_t offset, int16_t value)
{
    pace_put_u16(output, offset, (uint16_t)value);
}

static void pace_put_u32(uint8_t *output, size_t offset, uint32_t value)
{
    output[offset] = (uint8_t)(value & 0xFFU);
    output[offset + 1U] = (uint8_t)((value >> 8) & 0xFFU);
    output[offset + 2U] = (uint8_t)((value >> 16) & 0xFFU);
    output[offset + 3U] = (uint8_t)((value >> 24) & 0xFFU);
}

static uint16_t pace_get_u16(const uint8_t *input, size_t offset)
{
    return (uint16_t)((uint16_t)input[offset] | ((uint16_t)input[offset + 1U] << 8));
}

static uint32_t pace_get_u32(const uint8_t *input, size_t offset)
{
    return (uint32_t)input[offset] |
           ((uint32_t)input[offset + 1U] << 8) |
           ((uint32_t)input[offset + 2U] << 16) |
           ((uint32_t)input[offset + 3U] << 24);
}

static void pace_put_float(uint8_t *output, size_t offset, float value)
{
    uint32_t raw;
    memcpy(&raw, &value, sizeof(raw));
    pace_put_u32(output, offset, raw);
}

static bool pace_frame_begin(uint8_t *output,
                             size_t capacity,
                             pace_frame_type_t frame_type,
                             uint16_t frame_length,
                             uint32_t sequence)
{
    if ((output == 0) || (capacity < frame_length))
    {
        return false;
    }
    memset(output, 0, frame_length);
    output[0] = PACE_FRAME_MAGIC_0;
    output[1] = PACE_FRAME_MAGIC_1;
    output[2] = PACE_PROTOCOL_VERSION;
    output[3] = (uint8_t)frame_type;
    pace_put_u16(output, 4U, frame_length);
    pace_put_u32(output, 6U, sequence);
    return true;
}

static size_t pace_frame_finish(uint8_t *output, uint16_t frame_length)
{
    const uint32_t crc = pace_crc32_iso_hdlc(output, (size_t)frame_length - 4U);
    pace_put_u32(output, (size_t)frame_length - 4U, crc);
    return frame_length;
}

size_t pace_frame_encode_session_header(uint8_t *output,
                                        size_t capacity,
                                        uint32_t sequence,
                                        const pace_session_header_data_t *data)
{
    if ((data == 0) || !pace_frame_begin(output, capacity, PACE_FRAME_SESSION_HEADER,
                                         PACE_SESSION_HEADER_FRAME_SIZE, sequence))
    {
        return 0U;
    }
    output[10] = (uint8_t)data->session_type;
    output[11] = data->motor_order_version;
    output[12] = data->wire_endianness;
    pace_put_u32(output, 14U, data->session_id);
    pace_put_u16(output, 18U, data->sample_rate_hz);
    pace_put_u16(output, 20U, data->status_rate_hz);
    pace_put_u32(output, 22U, data->start_timestamp_us);
    pace_put_u32(output, 26U, data->experiment_config_hash);
    pace_put_u16(output, 30U, data->sample_frame_size_bytes);
    pace_put_u32(output, 34U, data->uart_baud);
    pace_put_u32(output, 38U, data->can_nominal_bitrate);
    pace_put_u32(output, 42U, data->scale_manifest_hash);
    pace_put_u32(output, 46U, data->firmware_build_id);
    pace_put_u32(output, 50U, data->robot_variant);
    return pace_frame_finish(output, PACE_SESSION_HEADER_FRAME_SIZE);
}

size_t pace_frame_encode_sample(uint8_t *output,
                                size_t capacity,
                                uint32_t sequence,
                                const pace_sample_frame_data_t *data)
{
    uint8_t index;
    if ((data == 0) || !pace_frame_begin(output, capacity, PACE_FRAME_SAMPLE,
                                         PACE_SAMPLE_FRAME_SIZE, sequence))
    {
        return 0U;
    }
    pace_put_u32(output, 10U, data->sample_time_us);
    output[14] = data->stage_id;
    output[15] = data->config_seq;
    output[16] = data->active_mask;
    output[17] = data->online_mask;
    output[18] = data->tx_valid_mask;
    output[19] = data->rx_valid_mask;
    output[20] = data->saturation_mask;
    output[21] = 0U;
    pace_put_u16(output, 22U, data->safety_flags);
    pace_put_u16(output, 24U, data->can_error_flags);

    for (index = 0U; index < PACE_DM_MOTOR_COUNT; ++index)
    {
        const size_t offset = 26U + ((size_t)index * 16U);
        const pace_sample_dm_block_t *block = &data->dm[index];
        pace_put_u16(output, offset, block->cmd_q_raw);
        pace_put_u16(output, offset + 2U, (uint16_t)(block->cmd_dq_raw & 0x0FFFU));
        pace_put_u16(output, offset + 4U, (uint16_t)(block->cmd_tau_raw & 0x0FFFU));
        pace_put_u16(output, offset + 6U, block->fb_q_raw);
        pace_put_u16(output, offset + 8U, (uint16_t)(block->fb_dq_raw & 0x0FFFU));
        pace_put_u16(output, offset + 10U, (uint16_t)(block->fb_tau_raw & 0x0FFFU));
        pace_put_u16(output, offset + 12U, block->tx_age_us);
        pace_put_u16(output, offset + 14U, block->rx_age_us);
    }
    for (index = 0U; index < PACE_LK_MOTOR_COUNT; ++index)
    {
        const size_t offset = 90U + ((size_t)index * 16U);
        const pace_sample_lk_block_t *block = &data->lk[index];
        pace_put_u32(output, offset, (uint32_t)block->cmd_primary_raw);
        pace_put_u16(output, offset + 4U, block->fb_encoder_raw);
        pace_put_i16(output, offset + 6U, block->fb_turn_count);
        pace_put_i16(output, offset + 8U, block->fb_speed_raw);
        pace_put_i16(output, offset + 10U, block->fb_iq_raw);
        pace_put_u16(output, offset + 12U, block->tx_age_us);
        pace_put_u16(output, offset + 14U, block->rx_age_us);
    }
    return pace_frame_finish(output, PACE_SAMPLE_FRAME_SIZE);
}

size_t pace_frame_encode_stage_config(uint8_t *output,
                                      size_t capacity,
                                      uint32_t sequence,
                                      const pace_stage_config_frame_data_t *data)
{
    uint8_t index;
    if ((data == 0) || !pace_frame_begin(output, capacity, PACE_FRAME_STAGE_CONFIG,
                                         PACE_STAGE_CONFIG_FRAME_SIZE, sequence))
    {
        return 0U;
    }
    output[10] = data->config_seq;
    output[11] = data->stage_id;
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        output[12U + index] = data->command_mode[index];
        pace_put_u16(output, 18U + ((size_t)index * 2U), data->command_rate_hz[index]);
    }
    for (index = 0U; index < PACE_DM_MOTOR_COUNT; ++index)
    {
        pace_put_u16(output, 30U + ((size_t)index * 2U), data->dm_kp_raw[index]);
        pace_put_u16(output, 38U + ((size_t)index * 2U), data->dm_kd_raw[index]);
    }
    output[46] = data->excitation_type;
    pace_put_float(output, 48U, data->amplitude);
    pace_put_float(output, 52U, data->frequency_start_hz);
    pace_put_float(output, 56U, data->frequency_end_hz);
    pace_put_u32(output, 60U, data->duration_ms);
    pace_put_u16(output, 64U, data->safety_limit_set_id);
    pace_put_u32(output, 68U, data->experiment_config_hash);
    return pace_frame_finish(output, PACE_STAGE_CONFIG_FRAME_SIZE);
}

size_t pace_frame_encode_status(uint8_t *output,
                                size_t capacity,
                                uint32_t sequence,
                                const pace_status_frame_data_t *data)
{
    uint8_t index;
    uint16_t frame_length;
    if ((data == 0) || (data->extension_length > PACE_STATUS_MAX_EXTENSION_SIZE) ||
        ((data->extension_length > 0U) && (data->extension_tlv == 0)))
    {
        return 0U;
    }
    frame_length = (uint16_t)(PACE_STATUS_BASE_FRAME_SIZE + data->extension_length);
    if (!pace_frame_begin(output, capacity, PACE_FRAME_STATUS, frame_length, sequence))
    {
        return 0U;
    }
    pace_put_u32(output, 10U, data->status_time_us);
    memcpy(&output[14], data->motor_state_nibbles, 3U);
    for (index = 0U; index < PACE_DM_MOTOR_COUNT; ++index)
    {
        output[17U + index] = (uint8_t)data->dm_temp_mos[index];
        output[21U + index] = (uint8_t)data->dm_temp_rotor[index];
    }
    for (index = 0U; index < PACE_LK_MOTOR_COUNT; ++index)
    {
        output[25U + index] = (uint8_t)data->lk_temp[index];
    }
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        pace_put_u16(output, 27U + ((size_t)index * 2U), data->tx_count_delta[index]);
        pace_put_u16(output, 39U + ((size_t)index * 2U), data->rx_count_delta[index]);
    }
    pace_put_u16(output, 51U, data->enqueue_fail_delta);
    pace_put_u16(output, 53U, data->tx_event_lost_delta);
    pace_put_u16(output, 55U, data->can_error_delta);
    output[57] = data->tx_fifo_high_water;
    output[58] = data->can_state;
    output[59] = data->extension_length;
    if (data->extension_length > 0U)
    {
        memcpy(&output[60], data->extension_tlv, data->extension_length);
    }
    return pace_frame_finish(output, frame_length);
}

size_t pace_frame_encode_event(uint8_t *output,
                               size_t capacity,
                               uint32_t sequence,
                               const pace_event_frame_data_t *data)
{
    if ((data == 0) || !pace_frame_begin(output, capacity, PACE_FRAME_EVENT,
                                         PACE_EVENT_FRAME_SIZE, sequence))
    {
        return 0U;
    }
    pace_put_u32(output, 10U, data->event_time_us);
    pace_put_u16(output, 14U, data->event_code);
    output[16] = data->severity;
    output[17] = data->motor_index;
    pace_put_u32(output, 18U, data->argument0);
    pace_put_u32(output, 22U, data->argument1);
    return pace_frame_finish(output, PACE_EVENT_FRAME_SIZE);
}

size_t pace_frame_encode_footer(uint8_t *output,
                                size_t capacity,
                                uint32_t sequence,
                                const pace_footer_frame_data_t *data)
{
    if ((data == 0) || !pace_frame_begin(output, capacity, PACE_FRAME_FOOTER,
                                         PACE_FOOTER_FRAME_SIZE, sequence))
    {
        return 0U;
    }
    pace_put_u32(output, 10U, data->stop_time_us);
    pace_put_u32(output, 14U, data->total_frames);
    pace_put_u32(output, 18U, data->dropped_frames);
    output[22] = data->overflow ? 1U : 0U;
    output[23] = data->statistics_complete ? 1U : 0U;
    pace_put_u16(output, 24U, data->stop_reason);
    pace_put_u32(output, 26U, data->experiment_config_hash);
    return pace_frame_finish(output, PACE_FOOTER_FRAME_SIZE);
}

bool pace_frame_validate(const uint8_t *frame,
                         size_t available,
                         pace_frame_type_t *frame_type,
                         uint32_t *sequence,
                         uint16_t *frame_length)
{
    uint16_t length;
    uint32_t expected_crc;
    uint32_t actual_crc;
    uint8_t type;

    if ((frame == 0) || (available < 14U) ||
        (frame[0] != PACE_FRAME_MAGIC_0) || (frame[1] != PACE_FRAME_MAGIC_1) ||
        (frame[2] != PACE_PROTOCOL_VERSION))
    {
        return false;
    }
    length = pace_get_u16(frame, 4U);
    type = frame[3];
    if ((length < 14U) || (available < length))
    {
        return false;
    }
    switch ((pace_frame_type_t)type)
    {
    case PACE_FRAME_SESSION_HEADER:
        if (length != PACE_SESSION_HEADER_FRAME_SIZE)
        {
            return false;
        }
        break;
    case PACE_FRAME_SAMPLE:
        if (length != PACE_SAMPLE_FRAME_SIZE)
        {
            return false;
        }
        break;
    case PACE_FRAME_STAGE_CONFIG:
        if (length != PACE_STAGE_CONFIG_FRAME_SIZE)
        {
            return false;
        }
        break;
    case PACE_FRAME_STATUS:
        if ((length < PACE_STATUS_BASE_FRAME_SIZE) || (length > PACE_STATUS_MAX_FRAME_SIZE) ||
            ((uint16_t)frame[59] + PACE_STATUS_BASE_FRAME_SIZE != length))
        {
            return false;
        }
        break;
    case PACE_FRAME_EVENT:
        if (length != PACE_EVENT_FRAME_SIZE)
        {
            return false;
        }
        break;
    case PACE_FRAME_FOOTER:
        if (length != PACE_FOOTER_FRAME_SIZE)
        {
            return false;
        }
        break;
    default:
        return false;
    }

    expected_crc = pace_get_u32(frame, (size_t)length - 4U);
    actual_crc = pace_crc32_iso_hdlc(frame, (size_t)length - 4U);
    if (expected_crc != actual_crc)
    {
        return false;
    }
    if (frame_type != 0)
    {
        *frame_type = (pace_frame_type_t)type;
    }
    if (sequence != 0)
    {
        *sequence = pace_get_u32(frame, 6U);
    }
    if (frame_length != 0)
    {
        *frame_length = length;
    }
    return true;
}
