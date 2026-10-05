#ifndef PACE_FRAME_H
#define PACE_FRAME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pace_motor_order.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PACE_PROTOCOL_VERSION 1U
#define PACE_FRAME_MAGIC_0 0x50U
#define PACE_FRAME_MAGIC_1 0x41U
#define PACE_COMMON_HEADER_SIZE 10U
#define PACE_SAMPLE_FRAME_SIZE 126U
#define PACE_STATUS_BASE_FRAME_SIZE 64U
#define PACE_STATUS_MAX_FRAME_SIZE 128U
#define PACE_STATUS_MAX_EXTENSION_SIZE 64U
#define PACE_SESSION_HEADER_FRAME_SIZE 58U
#define PACE_STAGE_CONFIG_FRAME_SIZE 76U
#define PACE_EVENT_FRAME_SIZE 30U
#define PACE_FOOTER_FRAME_SIZE 34U

typedef enum
{
    PACE_FRAME_SESSION_HEADER = 0x01,
    PACE_FRAME_SAMPLE = 0x02,
    PACE_FRAME_STAGE_CONFIG = 0x03,
    PACE_FRAME_STATUS = 0x04,
    PACE_FRAME_EVENT = 0x05,
    PACE_FRAME_FOOTER = 0x06
} pace_frame_type_t;

typedef enum
{
    PACE_SESSION_COMMISSIONING = 1,
    PACE_SESSION_FINAL_IDENTIFICATION = 2
} pace_session_type_t;

typedef struct
{
    uint16_t cmd_q_raw;
    uint16_t cmd_dq_raw;
    uint16_t cmd_tau_raw;
    uint16_t fb_q_raw;
    uint16_t fb_dq_raw;
    uint16_t fb_tau_raw;
    uint16_t tx_age_us;
    uint16_t rx_age_us;
} pace_sample_dm_block_t;

typedef struct
{
    int32_t cmd_primary_raw;
    uint16_t fb_encoder_raw;
    int16_t fb_turn_count;
    int16_t fb_speed_raw;
    int16_t fb_iq_raw;
    uint16_t tx_age_us;
    uint16_t rx_age_us;
} pace_sample_lk_block_t;

typedef struct
{
    uint32_t sample_time_us;
    uint8_t stage_id;
    uint8_t config_seq;
    uint8_t active_mask;
    uint8_t online_mask;
    uint8_t tx_valid_mask;
    uint8_t rx_valid_mask;
    uint8_t saturation_mask;
    uint16_t safety_flags;
    uint16_t can_error_flags;
    pace_sample_dm_block_t dm[PACE_DM_MOTOR_COUNT];
    pace_sample_lk_block_t lk[PACE_LK_MOTOR_COUNT];
} pace_sample_frame_data_t;

typedef struct
{
    uint32_t status_time_us;
    uint8_t motor_state_nibbles[3];
    int8_t dm_temp_mos[PACE_DM_MOTOR_COUNT];
    int8_t dm_temp_rotor[PACE_DM_MOTOR_COUNT];
    int8_t lk_temp[PACE_LK_MOTOR_COUNT];
    uint16_t tx_count_delta[PACE_MOTOR_COUNT];
    uint16_t rx_count_delta[PACE_MOTOR_COUNT];
    uint16_t enqueue_fail_delta;
    uint16_t tx_event_lost_delta;
    uint16_t can_error_delta;
    uint8_t tx_fifo_high_water;
    uint8_t can_state;
    const uint8_t *extension_tlv;
    uint8_t extension_length;
} pace_status_frame_data_t;

typedef struct
{
    pace_session_type_t session_type;
    uint8_t motor_order_version;
    uint8_t wire_endianness;
    uint32_t session_id;
    uint16_t sample_rate_hz;
    uint16_t status_rate_hz;
    uint32_t start_timestamp_us;
    uint32_t experiment_config_hash;
    uint16_t sample_frame_size_bytes;
    uint32_t uart_baud;
    uint32_t can_nominal_bitrate;
    uint32_t scale_manifest_hash;
    uint32_t firmware_build_id;
    uint32_t robot_variant;
} pace_session_header_data_t;

typedef struct
{
    uint8_t config_seq;
    uint8_t stage_id;
    uint8_t command_mode[PACE_MOTOR_COUNT];
    uint16_t command_rate_hz[PACE_MOTOR_COUNT];
    uint16_t dm_kp_raw[PACE_DM_MOTOR_COUNT];
    uint16_t dm_kd_raw[PACE_DM_MOTOR_COUNT];
    uint8_t excitation_type;
    float amplitude;
    float frequency_start_hz;
    float frequency_end_hz;
    uint32_t duration_ms;
    uint16_t safety_limit_set_id;
    uint32_t experiment_config_hash;
} pace_stage_config_frame_data_t;

typedef struct
{
    uint32_t event_time_us;
    uint16_t event_code;
    uint8_t severity;
    uint8_t motor_index;
    uint32_t argument0;
    uint32_t argument1;
} pace_event_frame_data_t;

typedef struct
{
    uint32_t stop_time_us;
    uint32_t total_frames;
    uint32_t dropped_frames;
    bool overflow;
    bool statistics_complete;
    uint16_t stop_reason;
    uint32_t experiment_config_hash;
} pace_footer_frame_data_t;

size_t pace_frame_encode_session_header(uint8_t *output, size_t capacity, uint32_t sequence,
                                        const pace_session_header_data_t *data);
size_t pace_frame_encode_sample(uint8_t *output, size_t capacity, uint32_t sequence,
                                const pace_sample_frame_data_t *data);
size_t pace_frame_encode_stage_config(uint8_t *output, size_t capacity, uint32_t sequence,
                                      const pace_stage_config_frame_data_t *data);
size_t pace_frame_encode_status(uint8_t *output, size_t capacity, uint32_t sequence,
                                const pace_status_frame_data_t *data);
size_t pace_frame_encode_event(uint8_t *output, size_t capacity, uint32_t sequence,
                               const pace_event_frame_data_t *data);
size_t pace_frame_encode_footer(uint8_t *output, size_t capacity, uint32_t sequence,
                                const pace_footer_frame_data_t *data);
bool pace_frame_validate(const uint8_t *frame, size_t available, pace_frame_type_t *frame_type,
                         uint32_t *sequence, uint16_t *frame_length);

#ifdef __cplusplus
}
#endif

#endif
