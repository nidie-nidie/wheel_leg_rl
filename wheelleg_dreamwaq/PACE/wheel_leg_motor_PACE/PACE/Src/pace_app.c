#include "pace_app.h"

#include <string.h>

#include "cmsis_os.h"
#include "dm_motor.h"
#include "lk_motor.h"
#include "main.h"
#include "pace_can_port.h"
#include "pace_capture.h"
#include "pace_command.h"
#include "pace_experiment_config.h"
#include "pace_frame.h"
#include "pace_motor_manifest.h"
#include "pace_motor_registry.h"
#include "pace_safety.h"
#include "pace_time.h"
#include "pace_transport.h"
#include "pace_uart_port.h"

#define PACE_UART_RX_DMA_SIZE 256U
#define PACE_HOST_RX_RING_SIZE 512U
#define PACE_TICK_RING_SIZE 16U
#define PACE_ALL_MOTORS_MASK 0x3FU
#define PACE_ARM_POLL_DIVIDER (PACE_CONTROL_RATE_HZ / 100U)
#define PACE_SAMPLE_DIVIDER (PACE_CONTROL_RATE_HZ / PACE_SAMPLE_RATE_HZ)
#define PACE_STATUS_DIVIDER (PACE_SAMPLE_RATE_HZ / PACE_STATUS_RATE_HZ)

#if ((PACE_CONTROL_RATE_HZ % PACE_SAMPLE_RATE_HZ) != 0U)
#error "PACE sample rate must divide the control rate"
#endif

#if ((PACE_SAMPLE_RATE_HZ % PACE_STATUS_RATE_HZ) != 0U)
#error "PACE status rate must divide the sample rate"
#endif

typedef enum
{
    PACE_EVENT_STAGE_CHANGED = 1,
    PACE_EVENT_SESSION_COMPLETE = 2,
    PACE_EVENT_SAFETY_ABORT = 3,
    PACE_EVENT_CAPTURE_OVERFLOW = 4,
    PACE_EVENT_UART_FAILURE = 5,
    PACE_EVENT_CAN_SEND_FAILURE = 6,
    PACE_EVENT_ARM_TIMEOUT = 7,
    PACE_EVENT_COUNTER_SATURATION = 8,
    PACE_EVENT_HOST_ABORT = 9,
    PACE_EVENT_SCHEDULER_OVERRUN = 10
} pace_app_event_code_t;

typedef enum
{
    PACE_STOP_COMPLETE = 0,
    PACE_STOP_HOST_REQUEST = 1,
    PACE_STOP_HOST_ABORT = 2,
    PACE_STOP_SAFETY = 3,
    PACE_STOP_CAPTURE_OVERFLOW = 4,
    PACE_STOP_UART_FAILURE = 5,
    PACE_STOP_CAN_FAILURE = 6,
    PACE_STOP_ARM_TIMEOUT = 7,
    PACE_STOP_SCHEDULER_OVERRUN = 8
} pace_app_stop_reason_t;

static pace_motor_registry_t pace_registry;
static pace_capture_t pace_capture;
static pace_experiment_t pace_experiment;
static pace_experiment_config_t pace_default_config;
static pace_safety_limits_t pace_safety_limits;
static pace_command_parser_t pace_command_parser;

static uint8_t pace_uart_rx_dma[PACE_UART_RX_DMA_SIZE];
static uint8_t pace_host_rx_ring[PACE_HOST_RX_RING_SIZE];
static volatile uint16_t pace_host_rx_head;
static volatile uint16_t pace_host_rx_tail;
static volatile bool pace_host_rx_overflow;
static uint8_t pace_host_command_buffer[PACE_HOST_COMMAND_MAX_SIZE];
static uint16_t pace_host_command_length;

static uint32_t pace_tick_ring[PACE_TICK_RING_SIZE];
static volatile uint8_t pace_tick_head;
static volatile uint8_t pace_tick_tail;
static volatile bool pace_tick_overflow;

static uint32_t pace_stream_sequence;
static uint32_t pace_sample_count;
static uint32_t pace_running_tick_count;
static uint32_t pace_session_start_us;
static uint32_t pace_arm_start_us;
static uint32_t pace_arm_tick_count;
static uint8_t pace_arm_enable_count[PACE_MOTOR_COUNT];
static uint8_t pace_arm_enable_pending_mask;
static uint8_t pace_safe_command_pending_mask;
static uint16_t pace_current_safety_flags;
static pace_app_stop_reason_t pace_normal_stop_reason;
static bool pace_session_header_emitted;
static bool pace_session_statistics_complete;

static uint32_t pace_previous_tx_count[PACE_MOTOR_COUNT];
static uint32_t pace_previous_rx_count[PACE_MOTOR_COUNT];
static uint32_t pace_previous_enqueue_fail;
static uint32_t pace_previous_tx_event_lost;
static uint32_t pace_previous_can_error;
static uint32_t pace_session_queue_full_base;
static uint32_t pace_session_hal_error_base;
static uint32_t pace_session_tx_event_lost_base;
static uint32_t pace_session_unmatched_event_base;

static bool pace_terminal_event_pending;
static pace_event_frame_data_t pace_terminal_event;
static bool pace_footer_pending;
static pace_app_stop_reason_t pace_footer_reason;
static uint32_t pace_footer_time_us;

static uint16_t pace_ring_next_u16(uint16_t value, uint16_t size)
{
    value++;
    return (value >= size) ? 0U : value;
}

static uint8_t pace_ring_next_u8(uint8_t value, uint8_t size)
{
    value++;
    return (value >= size) ? 0U : value;
}

static uint16_t pace_age_u16(bool valid, uint32_t age_us)
{
    if (!valid)
    {
        return 0U;
    }
    return (age_us > 65535U) ? 65535U : (uint16_t)age_us;
}

static uint32_t pace_app_relative_time(uint32_t now_us)
{
    return pace_session_header_emitted ? (now_us - pace_session_start_us) : 0U;
}

static void pace_app_uart_rx(const uint8_t *data, uint16_t length)
{
    uint16_t index;
    for (index = 0U; index < length; ++index)
    {
        const uint16_t next = pace_ring_next_u16(pace_host_rx_head, PACE_HOST_RX_RING_SIZE);
        if (next == pace_host_rx_tail)
        {
            pace_host_rx_overflow = true;
            return;
        }
        pace_host_rx_ring[pace_host_rx_head] = data[index];
        pace_host_rx_head = next;
    }
}

static void pace_app_tick_isr(uint32_t now_us)
{
    const uint8_t next = pace_ring_next_u8(pace_tick_head, PACE_TICK_RING_SIZE);
    if (next == pace_tick_tail)
    {
        pace_tick_overflow = true;
        return;
    }
    pace_tick_ring[pace_tick_head] = now_us;
    pace_tick_head = next;
}

static bool pace_app_pop_tick(uint32_t *now_us)
{
    if ((now_us == 0) || (pace_tick_tail == pace_tick_head))
    {
        return false;
    }
    *now_us = pace_tick_ring[pace_tick_tail];
    pace_tick_tail = pace_ring_next_u8(pace_tick_tail, PACE_TICK_RING_SIZE);
    return true;
}

static bool pace_app_push_frame(const uint8_t *frame, uint16_t length)
{
    if (!pace_capture_push(&pace_capture, frame, length))
    {
        return false;
    }
    pace_stream_sequence++;
    return true;
}

static bool pace_app_emit_event(uint32_t now_us,
                                uint16_t event_code,
                                uint8_t severity,
                                uint8_t motor_index,
                                uint32_t argument0,
                                uint32_t argument1)
{
    uint8_t frame[PACE_EVENT_FRAME_SIZE];
    pace_event_frame_data_t data;
    size_t length;

    if (!pace_session_header_emitted)
    {
        return true;
    }
    memset(&data, 0, sizeof(data));
    data.event_time_us = pace_app_relative_time(now_us);
    data.event_code = event_code;
    data.severity = severity;
    data.motor_index = motor_index;
    data.argument0 = argument0;
    data.argument1 = argument1;
    length = pace_frame_encode_event(frame, sizeof(frame), pace_stream_sequence, &data);
    return (length > 0U) && pace_app_push_frame(frame, (uint16_t)length);
}

static void pace_app_schedule_terminal(uint32_t now_us,
                                       pace_app_event_code_t event_code,
                                       pace_app_stop_reason_t stop_reason,
                                       uint32_t argument0,
                                       uint32_t argument1)
{
    if (!pace_session_header_emitted || pace_footer_pending)
    {
        return;
    }
    memset(&pace_terminal_event, 0, sizeof(pace_terminal_event));
    pace_terminal_event.event_time_us = pace_app_relative_time(now_us);
    pace_terminal_event.event_code = (uint16_t)event_code;
    pace_terminal_event.severity = (stop_reason == PACE_STOP_COMPLETE ||
                                    stop_reason == PACE_STOP_HOST_REQUEST) ? 0U : 2U;
    pace_terminal_event.motor_index = PACE_MOTOR_INDEX_INVALID;
    pace_terminal_event.argument0 = argument0;
    pace_terminal_event.argument1 = argument1;
    pace_terminal_event_pending = true;
    pace_footer_pending = true;
    pace_footer_reason = stop_reason;
    pace_footer_time_us = pace_app_relative_time(now_us);
}

static void pace_app_flush_terminal_frames(void)
{
    uint8_t frame[PACE_CAPTURE_SLOT_SIZE];
    size_t length;

    if (!pace_session_header_emitted || pace_transport_stats()->failed)
    {
        return;
    }
    if (pace_capture_depth(&pace_capture) >= (PACE_CAPTURE_SLOT_COUNT - 1U))
    {
        return;
    }
    if (pace_terminal_event_pending)
    {
        length = pace_frame_encode_event(frame, sizeof(frame), pace_stream_sequence,
                                         &pace_terminal_event);
        if ((length == 0U) || !pace_app_push_frame(frame, (uint16_t)length))
        {
            return;
        }
        pace_terminal_event_pending = false;
    }
    if (pace_footer_pending)
    {
        pace_footer_frame_data_t footer;
        memset(&footer, 0, sizeof(footer));
        footer.stop_time_us = pace_footer_time_us;
        footer.total_frames = pace_stream_sequence + 1U;
        footer.dropped_frames = pace_capture.rejected_frames;
        footer.overflow = pace_capture_overflowed(&pace_capture);
        footer.statistics_complete = pace_session_statistics_complete && !footer.overflow;
        footer.stop_reason = (uint16_t)pace_footer_reason;
        footer.experiment_config_hash = pace_experiment.config.experiment_config_hash;
        length = pace_frame_encode_footer(frame, sizeof(frame), pace_stream_sequence, &footer);
        if ((length == 0U) || !pace_app_push_frame(frame, (uint16_t)length))
        {
            return;
        }
        pace_footer_pending = false;
        pace_session_header_emitted = false;
    }
}

static void pace_app_enter_fault(uint32_t now_us,
                                 pace_app_event_code_t event_code,
                                 pace_app_stop_reason_t stop_reason,
                                 uint32_t argument0,
                                 uint32_t argument1)
{
    if (pace_experiment.state == PACE_STATE_FAULT)
    {
        return;
    }
    pace_experiment_abort(&pace_experiment);
    pace_session_statistics_complete = false;
    pace_safe_command_pending_mask = PACE_ALL_MOTORS_MASK;
    pace_app_schedule_terminal(now_us, event_code, stop_reason, argument0, argument1);
}

static bool pace_app_send_encoded(const pace_encoded_command_t *encoded)
{
    return pace_can_port_send(encoded) == PACE_CAN_SEND_OK;
}

static void pace_app_try_safe_commands(void)
{
    uint8_t index;
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        pace_encoded_command_t encoded;
        bool packed;
        if ((pace_safe_command_pending_mask & (uint8_t)(1U << index)) == 0U)
        {
            continue;
        }
        if (index < PACE_DM_MOTOR_COUNT)
        {
            packed = pace_dm_pack_special(index, PACE_COMMAND_DM_DISABLE, &encoded);
        }
        else
        {
            packed = pace_lk_pack_special(index, PACE_COMMAND_LK_STOP, &encoded);
        }
        if (packed && pace_app_send_encoded(&encoded))
        {
            pace_safe_command_pending_mask &= (uint8_t)~(1U << index);
        }
    }
}

static void pace_app_snapshot_registry(uint32_t now_us,
                                       pace_motor_registry_t *snapshot)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    pace_motor_registry_refresh_ages(&pace_registry, now_us,
                                     pace_safety_limits.feedback_timeout_us);
    *snapshot = pace_registry;
    if (primask == 0U)
    {
        __enable_irq();
    }
}

static bool pace_app_all_feedback_valid(const pace_motor_registry_t *registry)
{
    uint8_t index;
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        if (!registry->motors[index].rx_valid)
        {
            return false;
        }
    }
    return true;
}

static void pace_app_reset_session_counters(void)
{
    pace_can_port_stats_t stats;
    uint32_t primask;
    uint8_t index;

    primask = __get_PRIMASK();
    __disable_irq();
    stats = *pace_can_port_stats();
    pace_host_rx_overflow = false;
    pace_tick_overflow = false;
    pace_tick_tail = pace_tick_head;
    if (primask == 0U)
    {
        __enable_irq();
    }
    pace_stream_sequence = 0U;
    pace_sample_count = 0U;
    pace_running_tick_count = 0U;
    pace_session_start_us = 0U;
    pace_arm_tick_count = 0U;
    pace_current_safety_flags = 0U;
    pace_session_header_emitted = false;
    pace_session_statistics_complete = true;
    pace_terminal_event_pending = false;
    pace_footer_pending = false;
    pace_capture_init(&pace_capture);
    pace_transport_init(&pace_capture);
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        pace_previous_tx_count[index] = stats.confirmed_count[index];
        pace_previous_rx_count[index] = stats.rx_count[index];
    }
    pace_previous_enqueue_fail = 0U;
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        pace_previous_enqueue_fail += stats.enqueue_fail_count[index];
    }
    pace_previous_tx_event_lost = stats.tx_event_lost_count;
    pace_previous_can_error = stats.can_error_count + stats.hal_error_count +
                              stats.unmatched_tx_event_count;
    pace_session_queue_full_base = stats.queue_full_count;
    pace_session_hal_error_base = stats.hal_error_count;
    pace_session_tx_event_lost_base = stats.tx_event_lost_count;
    pace_session_unmatched_event_base = stats.unmatched_tx_event_count;
}

static void pace_app_invalidate_motor_history(void)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    pace_motor_registry_init(&pace_registry);
    if (primask == 0U)
    {
        __enable_irq();
    }
}

static bool pace_app_emit_session_header(uint32_t now_us)
{
    uint8_t frame[PACE_SESSION_HEADER_FRAME_SIZE];
    pace_session_header_data_t header;
    size_t length;

    memset(&header, 0, sizeof(header));
    header.session_type = pace_experiment.config.session_type;
    header.motor_order_version = PACE_MOTOR_ORDER_VERSION;
    header.wire_endianness = 1U;
    header.session_id = now_us ^ PACE_FIRMWARE_BUILD_ID;
    header.sample_rate_hz = PACE_SAMPLE_RATE_HZ;
    header.status_rate_hz = PACE_STATUS_RATE_HZ;
    header.start_timestamp_us = now_us;
    header.experiment_config_hash = pace_experiment.config.experiment_config_hash;
    header.sample_frame_size_bytes = PACE_SAMPLE_FRAME_SIZE;
    header.uart_baud = PACE_UART_BAUD;
    header.can_nominal_bitrate = PACE_CAN_NOMINAL_BITRATE;
    header.scale_manifest_hash = PACE_MOTOR_MANIFEST_HASH;
    header.firmware_build_id = PACE_FIRMWARE_BUILD_ID;
    header.robot_variant = PACE_ROBOT_VARIANT_ID;
    length = pace_frame_encode_session_header(frame, sizeof(frame), pace_stream_sequence,
                                              &header);
    if ((length == 0U) || !pace_app_push_frame(frame, (uint16_t)length))
    {
        return false;
    }
    pace_session_start_us = now_us;
    pace_session_header_emitted = true;
    return true;
}

static bool pace_app_arm_configured_session(uint32_t now_us)
{
    if (!pace_experiment_arm(&pace_experiment))
    {
        return false;
    }
    pace_app_reset_session_counters();
    pace_app_invalidate_motor_history();
    pace_arm_start_us = now_us;
    memset(pace_arm_enable_count, 0, sizeof(pace_arm_enable_count));
    pace_arm_enable_pending_mask = PACE_ALL_MOTORS_MASK;
    pace_safe_command_pending_mask = 0U;
    pace_normal_stop_reason = PACE_STOP_COMPLETE;
    return true;
}

/* Match the reference ChassisL/ChassisR initialization sequence.  Those two
 * RTOS tasks run in parallel: every millisecond the left task sends one frame
 * and the right task sends one frame.  Each motor receives ten enable frames
 * before that side advances to the next motor. */
static bool pace_app_send_enable_pair(uint8_t left_index, uint8_t right_index)
{
    pace_encoded_command_t left;
    pace_encoded_command_t right;
    bool left_packed;
    bool right_packed;

    if (left_index < PACE_DM_MOTOR_COUNT)
    {
        left_packed = pace_dm_pack_special(left_index,
                                           PACE_COMMAND_DM_ENABLE, &left);
        right_packed = pace_dm_pack_special(right_index,
                                            PACE_COMMAND_DM_ENABLE, &right);
    }
    else
    {
        left_packed = pace_lk_pack_special(left_index,
                                           PACE_COMMAND_LK_ENABLE, &left);
        right_packed = pace_lk_pack_special(right_index,
                                            PACE_COMMAND_LK_ENABLE, &right);
    }
    return left_packed && right_packed &&
           pace_app_send_encoded(&left) && pace_app_send_encoded(&right);
}

static bool pace_app_run_reference_enable_sequence(void)
{
    uint8_t repeat;

    for (repeat = 0U; repeat < PACE_MOTOR_ENABLE_REPEAT_COUNT; ++repeat)
    {
        if (!pace_app_send_enable_pair(PACE_MOTOR_L_FRONT,
                                       PACE_MOTOR_R_REAR))
        {
            return false;
        }
        osDelay(PACE_MOTOR_ENABLE_INTERVAL_MS);
    }
    for (repeat = 0U; repeat < PACE_MOTOR_ENABLE_REPEAT_COUNT; ++repeat)
    {
        if (!pace_app_send_enable_pair(PACE_MOTOR_L_REAR,
                                       PACE_MOTOR_R_FRONT))
        {
            return false;
        }
        osDelay(PACE_MOTOR_ENABLE_INTERVAL_MS);
    }
    for (repeat = 0U; repeat < PACE_MOTOR_ENABLE_REPEAT_COUNT; ++repeat)
    {
        if (!pace_app_send_enable_pair(PACE_MOTOR_L_WHEEL,
                                       PACE_MOTOR_R_WHEEL))
        {
            return false;
        }
        osDelay(PACE_MOTOR_ENABLE_INTERVAL_MS);
    }
    return true;
}

#if PACE_ONE_WAY_AUTOSTART
static bool pace_app_run_reference_autostart(void)
{
    osDelay(PACE_AUTOSTART_DELAY_MS);
    pace_experiment_default_config(&pace_default_config,
                                   PACE_SESSION_COMMISSIONING);
    if (!pace_experiment_configure(&pace_experiment, &pace_default_config) ||
        !pace_app_run_reference_enable_sequence())
    {
        return false;
    }

    /* TIM6 has been running during the reference startup delay. Discard its
     * pre-session ticks so capture begins from a clean control boundary. */
    pace_tick_tail = pace_tick_head;
    pace_tick_overflow = false;
    if (!pace_app_arm_configured_session(pace_time_now_us()))
    {
        return false;
    }
    /* The reference sequence above already delivered all enable frames. */
    pace_arm_enable_pending_mask = 0U;
    return pace_app_emit_session_header(pace_time_now_us());
}
#endif

static bool pace_app_emit_stage_config(uint32_t now_us)
{
    uint8_t frame[PACE_STAGE_CONFIG_FRAME_SIZE];
    pace_stage_config_frame_data_t data;
    size_t length;
    if (!pace_experiment_fill_stage_frame(&pace_experiment, &data))
    {
        return false;
    }
    length = pace_frame_encode_stage_config(frame, sizeof(frame), pace_stream_sequence, &data);
    if ((length == 0U) || !pace_app_push_frame(frame, (uint16_t)length))
    {
        return false;
    }
    return pace_app_emit_event(now_us, PACE_EVENT_STAGE_CHANGED, 0U,
                               PACE_MOTOR_INDEX_INVALID, data.stage_id, data.config_seq);
}

static uint16_t pace_app_can_error_flags(const pace_can_port_stats_t *stats)
{
    uint16_t flags = stats->can_state;
    if (stats->tx_event_lost_count != pace_session_tx_event_lost_base)
    {
        flags |= (uint16_t)(1U << 8);
    }
    if (stats->unmatched_tx_event_count != pace_session_unmatched_event_base)
    {
        flags |= (uint16_t)(1U << 9);
    }
    if ((stats->queue_full_count != pace_session_queue_full_base) ||
        (stats->hal_error_count != pace_session_hal_error_base))
    {
        flags |= (uint16_t)(1U << 10);
    }
    return flags;
}

static bool pace_app_emit_sample(uint32_t now_us,
                                 const pace_experiment_output_t *output,
                                 const pace_motor_registry_t *registry,
                                 const pace_can_port_stats_t *can_stats)
{
    uint8_t frame[PACE_SAMPLE_FRAME_SIZE];
    pace_sample_frame_data_t data;
    uint8_t index;
    size_t length;

    memset(&data, 0, sizeof(data));
    data.sample_time_us = pace_app_relative_time(now_us);
    data.stage_id = (uint8_t)output->stage_id;
    data.config_seq = output->config_seq;
    data.active_mask = output->active_mask;
    data.safety_flags = pace_current_safety_flags;
    data.can_error_flags = pace_app_can_error_flags(can_stats);

    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        const pace_motor_state_t *state = &registry->motors[index];
        const uint8_t bit = (uint8_t)(1U << index);
        const bool command_mode_matches = state->tx_valid &&
            (state->last_confirmed_command.command_mode == output->command_mode[index]);
        if (state->online)
        {
            data.online_mask |= bit;
        }
        if (command_mode_matches)
        {
            data.tx_valid_mask |= bit;
            if (state->last_confirmed_command.saturated)
            {
                data.saturation_mask |= bit;
            }
        }
        if (state->rx_valid)
        {
            data.rx_valid_mask |= bit;
        }
        if (index < PACE_DM_MOTOR_COUNT)
        {
            pace_sample_dm_block_t *block = &data.dm[index];
            if (command_mode_matches)
            {
                block->cmd_q_raw = state->last_confirmed_command.raw.dm.q;
                block->cmd_dq_raw = state->last_confirmed_command.raw.dm.dq;
                block->cmd_tau_raw = state->last_confirmed_command.raw.dm.tau;
            }
            if (state->rx_valid)
            {
                block->fb_q_raw = state->feedback.dm.q_raw;
                block->fb_dq_raw = state->feedback.dm.dq_raw;
                block->fb_tau_raw = state->feedback.dm.tau_raw;
            }
            block->tx_age_us = pace_age_u16(command_mode_matches, state->tx_age_us);
            block->rx_age_us = pace_age_u16(state->rx_valid, state->rx_age_us);
        }
        else
        {
            const uint8_t lk_index = (uint8_t)(index - PACE_DM_MOTOR_COUNT);
            pace_sample_lk_block_t *block = &data.lk[lk_index];
            if (command_mode_matches)
            {
                block->cmd_primary_raw = state->last_confirmed_command.raw.lk.primary;
            }
            if (state->rx_valid)
            {
                block->fb_encoder_raw = state->feedback.lk.encoder_raw;
                block->fb_turn_count = (int16_t)(uint16_t)state->feedback.lk.turn_count;
                block->fb_speed_raw = state->feedback.lk.speed_raw;
                block->fb_iq_raw = state->feedback.lk.iq_raw;
            }
            block->tx_age_us = pace_age_u16(command_mode_matches, state->tx_age_us);
            block->rx_age_us = pace_age_u16(state->rx_valid, state->rx_age_us);
        }
    }
    length = pace_frame_encode_sample(frame, sizeof(frame), pace_stream_sequence, &data);
    return (length > 0U) && pace_app_push_frame(frame, (uint16_t)length);
}

static uint16_t pace_app_delta_u16(uint32_t current,
                                   uint32_t *previous,
                                   bool *saturated)
{
    const uint32_t delta = current - *previous;
    *previous = current;
    if (delta > 65535U)
    {
        *saturated = true;
        return 65535U;
    }
    return (uint16_t)delta;
}

static bool pace_app_emit_status(uint32_t now_us,
                                 const pace_motor_registry_t *registry,
                                 const pace_can_port_stats_t *stats)
{
    uint8_t frame[PACE_STATUS_MAX_FRAME_SIZE];
    pace_status_frame_data_t data;
    uint32_t enqueue_fail = 0U;
    uint32_t can_error;
    bool saturated = false;
    uint8_t index;
    size_t length;

    memset(&data, 0, sizeof(data));
    data.status_time_us = pace_app_relative_time(now_us);
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        const pace_motor_state_t *state = &registry->motors[index];
        const uint8_t state_code = (index < PACE_DM_MOTOR_COUNT) ?
            (uint8_t)(state->feedback.dm.state & 0x0FU) :
            (uint8_t)(state->feedback.lk.status & 0x0FU);
        data.motor_state_nibbles[index / 2U] |=
            (uint8_t)(state_code << ((index % 2U) * 4U));
        data.tx_count_delta[index] = pace_app_delta_u16(stats->confirmed_count[index],
                                                        &pace_previous_tx_count[index],
                                                        &saturated);
        data.rx_count_delta[index] = pace_app_delta_u16(stats->rx_count[index],
                                                        &pace_previous_rx_count[index],
                                                        &saturated);
        enqueue_fail += stats->enqueue_fail_count[index];
        if (index < PACE_DM_MOTOR_COUNT)
        {
            data.dm_temp_mos[index] = (int8_t)state->feedback.dm.temperature_mos_c;
            data.dm_temp_rotor[index] = (int8_t)state->feedback.dm.temperature_rotor_c;
        }
        else
        {
            data.lk_temp[index - PACE_DM_MOTOR_COUNT] = state->feedback.lk.temperature_c;
        }
    }
    can_error = stats->can_error_count + stats->hal_error_count +
                stats->unmatched_tx_event_count;
    data.enqueue_fail_delta = pace_app_delta_u16(enqueue_fail,
                                                 &pace_previous_enqueue_fail,
                                                 &saturated);
    data.tx_event_lost_delta = pace_app_delta_u16(stats->tx_event_lost_count,
                                                  &pace_previous_tx_event_lost,
                                                  &saturated);
    data.can_error_delta = pace_app_delta_u16(can_error,
                                              &pace_previous_can_error,
                                              &saturated);
    data.tx_fifo_high_water = stats->tx_fifo_high_water;
    data.can_state = stats->can_state;
    if (saturated)
    {
        pace_session_statistics_complete = false;
        if (!pace_app_emit_event(now_us, PACE_EVENT_COUNTER_SATURATION, 1U,
                                 PACE_MOTOR_INDEX_INVALID, enqueue_fail, can_error))
        {
            return false;
        }
    }
    length = pace_frame_encode_status(frame, sizeof(frame), pace_stream_sequence, &data);
    return (length > 0U) && pace_app_push_frame(frame, (uint16_t)length);
}

static bool pace_app_send_experiment_commands(const pace_experiment_output_t *output)
{
    uint8_t index;
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        pace_encoded_command_t encoded;
        bool packed = false;
        if ((output->send_mask & (uint8_t)(1U << index)) == 0U)
        {
            continue;
        }
        if (index < PACE_DM_MOTOR_COUNT)
        {
            packed = pace_dm_pack_mit(index,
                                      output->dm_position_rad[index],
                                      output->dm_velocity_rad_s[index],
                                      output->dm_kp[index],
                                      output->dm_kd[index],
                                      output->dm_torque_nm[index],
                                      &encoded);
        }
        else if (output->command_mode[index] == PACE_COMMAND_LK_TORQUE)
        {
            packed = pace_lk_pack_torque(index,
                                         output->lk_primary[index - PACE_DM_MOTOR_COUNT],
                                         &encoded);
        }
        else if (output->command_mode[index] == PACE_COMMAND_LK_VELOCITY)
        {
            packed = pace_lk_pack_velocity(index,
                                           output->lk_primary[index - PACE_DM_MOTOR_COUNT],
                                           &encoded);
        }
        if (!packed || encoded.saturated || !pace_app_send_encoded(&encoded))
        {
            return false;
        }
    }
    return true;
}

static bool pace_app_send_arm_probe(void)
{
    uint8_t index;
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        pace_encoded_command_t encoded;
        bool packed;
        if (index < PACE_DM_MOTOR_COUNT)
        {
            packed = pace_dm_pack_mit(index, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                      &encoded);
        }
        else
        {
            packed = pace_lk_pack_torque(index, 0.0f, &encoded);
        }
        if (!packed || !pace_app_send_encoded(&encoded))
        {
            return false;
        }
    }
    return true;
}

static void pace_app_try_arm_enable(void)
{
    uint8_t index;
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        pace_encoded_command_t encoded;
        bool packed;
        if ((pace_arm_enable_pending_mask & (uint8_t)(1U << index)) == 0U)
        {
            continue;
        }
        if (index < PACE_DM_MOTOR_COUNT)
        {
            packed = pace_dm_pack_special(index, PACE_COMMAND_DM_ENABLE, &encoded);
        }
        else
        {
            packed = pace_lk_pack_special(index, PACE_COMMAND_LK_ENABLE, &encoded);
        }
        if (packed && pace_app_send_encoded(&encoded))
        {
            pace_arm_enable_count[index]++;
            if (pace_arm_enable_count[index] >= PACE_MOTOR_ENABLE_REPEAT_COUNT)
            {
                pace_arm_enable_pending_mask &= (uint8_t)~(1U << index);
            }
        }
    }
}

static void pace_app_handle_armed_tick(uint32_t now_us,
                                       const pace_motor_registry_t *registry)
{
    pace_arm_tick_count++;
    pace_app_try_arm_enable();
    if ((pace_arm_enable_pending_mask == 0U) &&
        ((pace_arm_tick_count % PACE_ARM_POLL_DIVIDER) == 0U) &&
        !pace_app_send_arm_probe())
    {
        if ((now_us - pace_arm_start_us) >= (PACE_ARM_TIMEOUT_MS * 1000U))
        {
            pace_app_enter_fault(now_us, PACE_EVENT_ARM_TIMEOUT,
                                 PACE_STOP_ARM_TIMEOUT, 0U, 0U);
        }
        return;
    }
    if (pace_app_all_feedback_valid(registry))
    {
        if (!pace_experiment_begin(&pace_experiment, registry, now_us))
        {
            pace_app_enter_fault(now_us, PACE_EVENT_CAPTURE_OVERFLOW,
                                 PACE_STOP_CAPTURE_OVERFLOW, 0U, 0U);
        }
        return;
    }
    if ((now_us - pace_arm_start_us) >= (PACE_ARM_TIMEOUT_MS * 1000U))
    {
        pace_app_enter_fault(now_us, PACE_EVENT_ARM_TIMEOUT,
                             PACE_STOP_ARM_TIMEOUT, 0U, 0U);
    }
}

static void pace_app_finish_session(uint32_t now_us)
{
    pace_safe_command_pending_mask = PACE_ALL_MOTORS_MASK;
    pace_app_schedule_terminal(now_us, PACE_EVENT_SESSION_COMPLETE,
                               pace_normal_stop_reason, pace_sample_count, 0U);
}

static void pace_app_handle_running_tick(uint32_t now_us,
                                         const pace_motor_registry_t *registry,
                                         const pace_can_port_stats_t *can_stats)
{
    pace_experiment_output_t output;
    bool emit_sample;
    const bool can_fault = ((can_stats->can_state &
                            (PACE_CAN_STATE_PASSIVE | PACE_CAN_STATE_BUS_OFF)) != 0U) ||
                           (can_stats->tx_event_lost_count !=
                            pace_session_tx_event_lost_base) ||
                           (can_stats->unmatched_tx_event_count !=
                            pace_session_unmatched_event_base);

    pace_current_safety_flags = pace_safety_check(registry, &pace_safety_limits,
                                                  now_us, pace_session_start_us,
                                                  can_fault,
                                                  pace_capture_overflowed(&pace_capture));
    if (pace_current_safety_flags != PACE_SAFETY_NONE)
    {
        pace_app_enter_fault(now_us, PACE_EVENT_SAFETY_ABORT, PACE_STOP_SAFETY,
                             pace_current_safety_flags,
                             pace_app_can_error_flags(can_stats));
        return;
    }
    if (!pace_experiment_step(&pace_experiment, now_us, &output))
    {
        pace_app_enter_fault(now_us, PACE_EVENT_SCHEDULER_OVERRUN,
                             PACE_STOP_SCHEDULER_OVERRUN, 0U, 0U);
        return;
    }
    if (output.session_complete)
    {
        pace_app_finish_session(now_us);
        return;
    }
    if (output.stage_changed && !pace_app_emit_stage_config(now_us))
    {
        pace_app_enter_fault(now_us, PACE_EVENT_CAPTURE_OVERFLOW,
                             PACE_STOP_CAPTURE_OVERFLOW, 0U, 0U);
        return;
    }
    if (!pace_app_send_experiment_commands(&output))
    {
        pace_app_enter_fault(now_us, PACE_EVENT_CAN_SEND_FAILURE,
                             PACE_STOP_CAN_FAILURE, output.stage_id,
                             output.send_mask);
        return;
    }
    /* CAN commands run on every 500 Hz control tick. Keep one continuous
     * decimation phase for the whole session so stage boundaries cannot
     * create a short 2 ms sample interval. */
    emit_sample = ((pace_running_tick_count % PACE_SAMPLE_DIVIDER) == 0U);
    pace_running_tick_count++;
    if (!emit_sample)
    {
        return;
    }
    if (!pace_app_emit_sample(now_us, &output, registry, can_stats))
    {
        pace_app_enter_fault(now_us, PACE_EVENT_CAPTURE_OVERFLOW,
                             PACE_STOP_CAPTURE_OVERFLOW, pace_capture_depth(&pace_capture),
                             pace_capture.rejected_frames);
        return;
    }
    pace_sample_count++;
    if ((pace_sample_count % PACE_STATUS_DIVIDER) == 0U)
    {
        if (!pace_app_emit_status(now_us, registry, can_stats))
        {
            pace_app_enter_fault(now_us, PACE_EVENT_CAPTURE_OVERFLOW,
                                 PACE_STOP_CAPTURE_OVERFLOW,
                                 pace_capture_depth(&pace_capture),
                                 pace_capture.rejected_frames);
        }
    }
}

static void pace_app_handle_tick(uint32_t now_us)
{
    pace_motor_registry_t registry_snapshot;
    pace_can_port_stats_t can_stats_snapshot;
    uint32_t primask;

    pace_app_snapshot_registry(now_us, &registry_snapshot);
    if ((pace_sample_count % PACE_STATUS_DIVIDER) == 0U)
    {
        pace_can_port_refresh_protocol_state();
    }
    primask = __get_PRIMASK();
    __disable_irq();
    can_stats_snapshot = *pace_can_port_stats();
    if (primask == 0U)
    {
        __enable_irq();
    }

    if (pace_experiment.state == PACE_STATE_ARMED)
    {
        pace_app_handle_armed_tick(now_us, &registry_snapshot);
    }
    else if ((pace_experiment.state == PACE_STATE_RUNNING) ||
             (pace_experiment.state == PACE_STATE_STOPPING))
    {
        pace_app_handle_running_tick(now_us, &registry_snapshot, &can_stats_snapshot);
    }
}

static void pace_app_remove_command_bytes(uint16_t count)
{
    if (count >= pace_host_command_length)
    {
        pace_host_command_length = 0U;
        return;
    }
    memmove(pace_host_command_buffer, &pace_host_command_buffer[count],
            pace_host_command_length - count);
    pace_host_command_length = (uint16_t)(pace_host_command_length - count);
}

static bool pace_app_ready_for_new_session(void)
{
    return (pace_capture_depth(&pace_capture) == 0U) &&
           !pace_transport_stats()->inflight &&
           !pace_footer_pending &&
           (pace_safe_command_pending_mask == 0U) &&
           (pace_can_port_pending_count() == 0U);
}

static bool pace_app_final_contract_is_ready(void)
{
    return (PACE_FINAL_CONTRACT_READY == 1U) &&
           (PACE_FINAL_CONTRACT_VERSION > 0U) &&
           (PACE_FINAL_CONTRACT_HASH != 0UL) &&
           (PACE_FINAL_ACTION_RATE_HZ > 0U) &&
           (PACE_FINAL_COMMAND_HOLD_METHOD > 0U) &&
           (PACE_FINAL_DM_POSITION_AMPLITUDE_RAD > 0.0f) &&
           (PACE_FINAL_DM_BANDWIDTH_HZ > 0.0f) &&
           (PACE_FINAL_LK_COMMAND_MODE > 0U) &&
           (PACE_FINAL_SAFETY_LIMIT_SET_ID > 0U);
}

static void pace_app_handle_host_command(const pace_host_command_t *command)
{
    uint32_t now_us = pace_time_now_us();
    if (command == 0)
    {
        return;
    }
    switch (command->command_id)
    {
    case PACE_HOST_CONFIGURE:
        if ((command->payload_length <= 1U) && pace_app_ready_for_new_session())
        {
            pace_session_type_t session_type = PACE_SESSION_COMMISSIONING;
            if (command->payload_length == 1U)
            {
                session_type = (pace_session_type_t)command->payload[0];
            }
            if (session_type == PACE_SESSION_COMMISSIONING)
            {
                pace_experiment_default_config(&pace_default_config, session_type);
                (void)pace_experiment_configure(&pace_experiment, &pace_default_config);
            }
            else if ((session_type == PACE_SESSION_FINAL_IDENTIFICATION) &&
                     pace_app_final_contract_is_ready())
            {
                pace_experiment_default_config(&pace_default_config, session_type);
                pace_default_config.experiment_config_hash = PACE_FINAL_CONTRACT_HASH;
                pace_default_config.safety_limit_set_id =
                    PACE_FINAL_SAFETY_LIMIT_SET_ID;
                (void)pace_experiment_configure(&pace_experiment,
                                                &pace_default_config);
            }
        }
        break;
    case PACE_HOST_START:
        if ((command->payload_length == 0U) && pace_app_ready_for_new_session())
        {
            if (pace_app_arm_configured_session(now_us) &&
                !pace_app_emit_session_header(now_us))
            {
                pace_app_enter_fault(now_us, PACE_EVENT_CAPTURE_OVERFLOW,
                                     PACE_STOP_CAPTURE_OVERFLOW, 0U, 0U);
            }
        }
        break;
    case PACE_HOST_STOP:
        if (pace_experiment.state == PACE_STATE_RUNNING)
        {
            if (pace_experiment_request_stop(&pace_experiment, now_us))
            {
                pace_normal_stop_reason = PACE_STOP_HOST_REQUEST;
            }
        }
        else if (pace_experiment.state == PACE_STATE_ARMED)
        {
            pace_safe_command_pending_mask = PACE_ALL_MOTORS_MASK;
            pace_normal_stop_reason = PACE_STOP_HOST_REQUEST;
            pace_app_schedule_terminal(now_us, PACE_EVENT_SESSION_COMPLETE,
                                       PACE_STOP_HOST_REQUEST, 0U, 0U);
            pace_experiment_reset(&pace_experiment);
        }
        break;
    case PACE_HOST_STATUS:
        if (pace_session_header_emitted)
        {
            pace_motor_registry_t registry_snapshot;
            pace_can_port_stats_t can_stats_snapshot;
            uint32_t primask;
            pace_app_snapshot_registry(now_us, &registry_snapshot);
            primask = __get_PRIMASK();
            __disable_irq();
            can_stats_snapshot = *pace_can_port_stats();
            if (primask == 0U)
            {
                __enable_irq();
            }
            if (!pace_app_emit_status(now_us, &registry_snapshot, &can_stats_snapshot))
            {
                pace_app_enter_fault(now_us, PACE_EVENT_CAPTURE_OVERFLOW,
                                     PACE_STOP_CAPTURE_OVERFLOW, 0U, 0U);
            }
        }
        break;
    case PACE_HOST_ABORT:
        pace_app_enter_fault(now_us, PACE_EVENT_HOST_ABORT,
                             PACE_STOP_HOST_ABORT, 0U, 0U);
        break;
    default:
        break;
    }
}

static void pace_app_parse_host_stream(void)
{
    while (pace_host_rx_tail != pace_host_rx_head)
    {
        const uint8_t byte = pace_host_rx_ring[pace_host_rx_tail];
        pace_host_rx_tail = pace_ring_next_u16(pace_host_rx_tail,
                                               PACE_HOST_RX_RING_SIZE);
        if (pace_host_command_length >= PACE_HOST_COMMAND_MAX_SIZE)
        {
            pace_app_remove_command_bytes(1U);
        }
        pace_host_command_buffer[pace_host_command_length++] = byte;

        for (;;)
        {
            uint16_t frame_length;
            pace_host_command_t command;
            if (pace_host_command_length < 2U)
            {
                break;
            }
            if ((pace_host_command_buffer[0] != PACE_HOST_COMMAND_MAGIC_0) ||
                (pace_host_command_buffer[1] != PACE_HOST_COMMAND_MAGIC_1))
            {
                pace_app_remove_command_bytes(1U);
                continue;
            }
            if (pace_host_command_length < 6U)
            {
                break;
            }
            frame_length = (uint16_t)pace_host_command_buffer[4] |
                           ((uint16_t)pace_host_command_buffer[5] << 8);
            if ((frame_length < PACE_HOST_COMMAND_MIN_SIZE) ||
                (frame_length > PACE_HOST_COMMAND_MAX_SIZE))
            {
                pace_app_remove_command_bytes(1U);
                continue;
            }
            if (pace_host_command_length < frame_length)
            {
                break;
            }
            if (pace_command_decode(&pace_command_parser, pace_host_command_buffer,
                                    frame_length, &command))
            {
                pace_app_handle_host_command(&command);
                pace_app_remove_command_bytes(frame_length);
            }
            else
            {
                pace_app_remove_command_bytes(1U);
            }
        }
    }
}

bool pace_app_init(void)
{
    memset(pace_host_rx_ring, 0, sizeof(pace_host_rx_ring));
    memset(pace_tick_ring, 0, sizeof(pace_tick_ring));
    pace_host_rx_head = 0U;
    pace_host_rx_tail = 0U;
    pace_host_rx_overflow = false;
    pace_host_command_length = 0U;
    pace_tick_head = 0U;
    pace_tick_tail = 0U;
    pace_tick_overflow = false;
    pace_safe_command_pending_mask = 0U;
    pace_motor_registry_init(&pace_registry);
    pace_capture_init(&pace_capture);
    pace_experiment_init(&pace_experiment);
    pace_safety_default_limits(&pace_safety_limits);
    pace_command_parser_init(&pace_command_parser);
    pace_uart_port_init();
    pace_transport_init(&pace_capture);
    pace_can_port_init(&pace_registry);
    pace_time_set_tick_callback(pace_app_tick_isr);

#if PACE_ONE_WAY_AUTOSTART
    if (!pace_time_init() ||
        !pace_can_port_start())
#else
    if (!pace_uart_port_start_rx(pace_uart_rx_dma, sizeof(pace_uart_rx_dma),
                                 pace_app_uart_rx) ||
        !pace_time_init() ||
        !pace_can_port_start())
#endif
    {
        return false;
    }
    return true;
}

void pace_app_capture_task(void const *argument)
{
    (void)argument;
#if PACE_ONE_WAY_AUTOSTART
    if (!pace_app_run_reference_autostart())
    {
        pace_app_enter_fault(pace_time_now_us(), PACE_EVENT_CAN_SEND_FAILURE,
                             PACE_STOP_CAN_FAILURE, 0U, 0U);
    }
#endif
    for (;;)
    {
        uint32_t now_us;
#if !PACE_ONE_WAY_AUTOSTART
        pace_app_parse_host_stream();
        if (pace_host_rx_overflow &&
            ((pace_experiment.state == PACE_STATE_ARMED) ||
             (pace_experiment.state == PACE_STATE_RUNNING) ||
             (pace_experiment.state == PACE_STATE_STOPPING)))
        {
            pace_app_enter_fault(pace_time_now_us(), PACE_EVENT_UART_FAILURE,
                                 PACE_STOP_UART_FAILURE, 0U, 0U);
        }
#endif
        if (pace_tick_overflow &&
            ((pace_experiment.state == PACE_STATE_ARMED) ||
             (pace_experiment.state == PACE_STATE_RUNNING) ||
             (pace_experiment.state == PACE_STATE_STOPPING)))
        {
            pace_app_enter_fault(pace_time_now_us(), PACE_EVENT_SCHEDULER_OVERRUN,
                                 PACE_STOP_SCHEDULER_OVERRUN, 0U, 0U);
        }
        while (pace_app_pop_tick(&now_us))
        {
            pace_app_handle_tick(now_us);
        }
        if (pace_transport_stats()->failed)
        {
            pace_app_enter_fault(pace_time_now_us(), PACE_EVENT_UART_FAILURE,
                                 PACE_STOP_UART_FAILURE, 0U, 0U);
        }
        pace_app_try_safe_commands();
        pace_app_flush_terminal_frames();
        osDelay(1U);
    }
}

void pace_app_transport_task(void const *argument)
{
    (void)argument;
    for (;;)
    {
        pace_transport_kick();
        osDelay(1U);
    }
}

pace_experiment_state_t pace_app_state(void)
{
    return pace_experiment.state;
}
