/* Tests execute production app/Board/encoder code; only HAL/RTOS are faked. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "host_hal.h"
#include "dm_motor.h"
#include "lk_motor.h"
#include "pace_app.h"
#include "pace_can_port.h"
#include "pace_capture.h"
#include "pace_command.h"
#include "pace_crc.h"
#include "pace_experiment_config.h"
#include "pace_frame.h"
#include "pace_motor_manifest.h"
#include "pace_motor_registry.h"
#include "pace_safety.h"
#include "pace_time.h"
#include "pace_transport.h"
#include "pace_uart_port.h"

#define CHECK(test) do { if (!(test)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #test); exit(1); } } while (0)
#ifdef PACE_HAS_CONTRACT_TEST
extern void pace_test_experiment_contract(void);
#endif

typedef struct
{
    uint32_t frames, headers, stages, samples, statuses, footers, events;
    uint32_t recipe_hash, dropped_frames, footer_total_frames;
    uint32_t last_sample_time_us, last_footer_sequence;
    uint32_t stage_mask;
    uint16_t stop_reason;
    uint8_t statistics_complete, overflow;
    uint32_t event_mask;
} stream_summary_t;

static uint32_t host_command_sequence;
static uint16_t get16(const uint8_t *bytes, size_t offset)
{ return (uint16_t)((uint16_t)bytes[offset] | (uint16_t)((uint16_t)bytes[offset + 1U] << 8)); }
static uint32_t get32(const uint8_t *bytes, size_t offset)
{ return (uint32_t)bytes[offset] | ((uint32_t)bytes[offset + 1U] << 8) | ((uint32_t)bytes[offset + 2U] << 16) | ((uint32_t)bytes[offset + 3U] << 24); }
static void put32(uint8_t *bytes, size_t offset, uint32_t value)
{ bytes[offset] = (uint8_t)value; bytes[offset + 1U] = (uint8_t)(value >> 8); bytes[offset + 2U] = (uint8_t)(value >> 16); bytes[offset + 3U] = (uint8_t)(value >> 24); }

static stream_summary_t inspect_stream(void)
{
    stream_summary_t summary = {0};
    size_t offset = 0U;
    const uint8_t *bytes = host_uart_bytes();
    const size_t size = host_uart_size();
    bool have_sample = false;
    uint8_t current_stage_seq = 0U;
    while (offset < size)
    {
        pace_frame_type_t type;
        uint16_t length;
        uint32_t sequence;
        const uint8_t *frame = bytes + offset;
        CHECK(pace_frame_validate(frame, size - offset, &type, &sequence, &length));
        CHECK(frame[2] == PACE_PROTOCOL_VERSION);
        CHECK(sequence == summary.frames);
        CHECK(summary.footers == 0U); /* Terminal frame is always last. */
        if (summary.frames == 0U) CHECK(type == PACE_FRAME_SESSION_HEADER);
        summary.frames++;
        switch (type)
        {
        case PACE_FRAME_SESSION_HEADER:
            summary.headers++;
            CHECK(summary.headers == 1U);
            CHECK(frame[13] == PACE_TIMESTAMP_TIM5_ISR_SERVICE);
            CHECK(get16(frame, 18U) == PACE_SAMPLE_RATE_HZ);
            CHECK(get16(frame, 32U) == PACE_CONTROL_RATE_HZ);
            summary.recipe_hash = get32(frame, 26U);
            CHECK(summary.recipe_hash != 0U);
            break;
        case PACE_FRAME_STAGE_CONFIG:
            CHECK(length == PACE_STAGE_CONFIG_FRAME_SIZE);
            current_stage_seq = frame[10];
            CHECK(current_stage_seq >= 1U && current_stage_seq <= PACE_EXPERIMENT_MAX_STAGES);
            CHECK((summary.stage_mask & (1U << (current_stage_seq - 1U))) == 0U);
            summary.stage_mask |= (1U << (current_stage_seq - 1U));
            CHECK(get32(frame, 68U) == summary.recipe_hash);
            CHECK(get16(frame, 74U) == ((frame[72] == PACE_ROLE_STOP) ? PACE_EXPERIMENT_STAGE_COUNT : current_stage_seq));
            CHECK(get16(frame, 76U) == PACE_ACTION_UPDATE_RATE_HZ);
            CHECK(frame[78] == PACE_HOLD_ZERO_ORDER);
            CHECK(frame[79] == PACE_CENTER_INITIAL_FEEDBACK);
            summary.stages++;
            break;
        case PACE_FRAME_SAMPLE:
        {
            const uint32_t sample_time = get32(frame, 10U);
            CHECK(frame[15] == current_stage_seq);
            if (have_sample) CHECK(sample_time - summary.last_sample_time_us == 4000U);
            have_sample = true;
            summary.last_sample_time_us = sample_time;
            summary.samples++;
            break;
        }
        case PACE_FRAME_STATUS: summary.statuses++; break;
        case PACE_FRAME_EVENT:
            summary.events++;
            if (get16(frame, 14U) < 32U) summary.event_mask |= (1U << get16(frame, 14U));
            break;
        case PACE_FRAME_FOOTER:
            summary.footers++;
            summary.last_footer_sequence = sequence;
            summary.footer_total_frames = get32(frame, 14U);
            summary.dropped_frames = get32(frame, 18U);
            summary.overflow = frame[22];
            summary.statistics_complete = frame[23];
            summary.stop_reason = get16(frame, 24U);
            CHECK(get32(frame, 26U) == summary.recipe_hash);
            CHECK(summary.footer_total_frames == summary.frames);
            break;
        default: CHECK(false); break;
        }
        offset += length;
    }
    CHECK(offset == size);
    return summary;
}

static void drain_transport(void)
{
    uint32_t previous, attempt;
    for (attempt = 0U; attempt < PACE_CAPTURE_SLOT_COUNT + 4U; ++attempt)
    {
        previous = pace_transport_stats()->frames_sent;
        pace_app_transport_poll();
        if (previous == pace_transport_stats()->frames_sent) break;
    }
}

static void send_host(pace_host_command_id_t command, int session_type, bool fragmented)
{
    uint8_t frame[15] = {0};
    const uint16_t length = (session_type >= 0) ? 15U : 14U;
    frame[0] = PACE_HOST_COMMAND_MAGIC_0;
    frame[1] = PACE_HOST_COMMAND_MAGIC_1;
    frame[2] = PACE_HOST_COMMAND_VERSION;
    frame[3] = (uint8_t)command;
    frame[4] = (uint8_t)length;
    put32(frame, 6U, host_command_sequence++);
    if (session_type >= 0) frame[10] = (uint8_t)session_type;
    put32(frame, length - 4U, pace_crc32_iso_hdlc(frame, length - 4U));
    if (fragmented)
    {
        host_uart_receive(frame, 3U);
        pace_app_poll();
        host_uart_receive(frame + 3U, length - 3U);
    }
    else host_uart_receive(frame, length);
    pace_app_poll();
}

static void initialize_app(void)
{
    host_reset();
    host_command_sequence = 0U;
    CHECK(pace_app_init());
    CHECK(pace_app_state() == PACE_STATE_SAFE_IDLE);
    CHECK(host_can_queued() == 0U);
    CHECK(host_uart_size() == 0U);
    CHECK((host_can_notifications() & FDCAN_IT_RX_FIFO0_MESSAGE_LOST) != 0U);
}

static void regular_step(bool transport)
{
    host_set_time(host_now_us() + 1000U);
    host_can_confirm_all(); /* Deliberately later than HAL enqueue. */
    if ((host_now_us() % 2000U) == 0U)
    {
        host_can_feedback_all();
        host_tick();
    }
    pace_app_poll();
    if (transport) pace_app_transport_poll();
}

static void start_running(void)
{
    uint32_t attempts;
    initialize_app();
#if !PACE_ONE_WAY_AUTOSTART
    send_host(PACE_HOST_CONFIGURE, PACE_SESSION_COMMISSIONING, true);
    CHECK(pace_app_state() == PACE_STATE_CONFIGURED);
    send_host(PACE_HOST_START, -1, true);
    CHECK(pace_app_state() == PACE_STATE_ARMED);
#endif
    for (attempts = 0U; attempts < PACE_AUTOSTART_DELAY_MS + PACE_ARM_TIMEOUT_MS; ++attempts)
    {
        regular_step(true);
        CHECK(pace_app_state() != PACE_STATE_FAULT);
        if (pace_app_state() == PACE_STATE_RUNNING)
        {
            regular_step(true); regular_step(true); drain_transport();
            return;
        }
    }
    CHECK(false);
}

static void finish_fault(uint16_t reason)
{
    uint32_t count;
    stream_summary_t summary;
    for (count = 0U; count < 500U; ++count) regular_step(true);
    drain_transport();
    summary = inspect_stream();
    CHECK(summary.footers == 1U);
    CHECK(summary.stop_reason == reason);
    CHECK(summary.statistics_complete == 0U);
    CHECK(pace_app_state() == PACE_STATE_FAULT);
}

static void test_can_confirmation_and_time(void)
{
    pace_motor_registry_t registry;
    pace_encoded_command_t command;
    host_reset();
    pace_motor_registry_init(&registry);
    pace_can_port_init(&registry);
    CHECK(pace_time_init());
    CHECK(pace_can_port_start());
    host_set_fdcan_epoch_lag(100U);
    host_set_time(65530U);
    CHECK(pace_dm_pack_mit(0U, 0.1f, 0.0f, 20.0f, 0.6f, 0.0f, &command));
    CHECK(pace_can_port_send(&command) == PACE_CAN_SEND_OK);
    CHECK(!registry.motors[0].tx_valid && registry.motors[0].tx_count == 0U);
    CHECK(pace_can_port_pending_count() == 1U);
    host_set_time(65680U);
    CHECK(host_can_confirm_next());
    CHECK(registry.motors[0].last_tx_timestamp_us == 65680U);
    CHECK(pace_can_port_stats()->last_enqueue_to_tx_us == 150U);
    CHECK(registry.motors[0].tx_count == 1U);
    CHECK(pace_can_port_pending_count() == 0U);
    host_can_motor_feedback(0U, 32767U);
    CHECK(registry.motors[0].last_rx_timestamp_us == 65680U);
    host_set_time(UINT32_MAX - 100U);
    CHECK(pace_can_port_send(&command) == PACE_CAN_SEND_OK);
    host_set_time(100U);
    CHECK(host_can_confirm_next());
    CHECK(pace_can_port_stats()->last_enqueue_to_tx_us == 201U);
    CHECK(registry.motors[0].last_tx_timestamp_us == 100U);
    puts("PASS CAN asynchronous confirmation and TIM5 service time across wraps");
}

static void test_can_validation(void)
{
    pace_motor_registry_t registry;
    FDCAN_RxHeaderTypeDef header = {0};
    uint8_t data[64] = {0xA2U, 25U, 0U, 0U, 0U, 0U, 1U, 2U};
    host_reset();
    pace_motor_registry_init(&registry); pace_can_port_init(&registry);
    header.Identifier = pace_motor_manifest[4].feedback_can_id;
    header.DataLength = FDCAN_DLC_BYTES_8;
    host_set_time(1000U);
    host_can_feedback(&header, data);
    CHECK(registry.motors[4].rx_count == 1U);
    header.DataLength = 7U; host_can_feedback(&header, data);
    header.DataLength = FDCAN_DLC_BYTES_8; header.RxFrameType = FDCAN_REMOTE_FRAME;
    host_can_feedback(&header, data);
    header.RxFrameType = FDCAN_DATA_FRAME; header.IdType = FDCAN_EXTENDED_ID;
    host_can_feedback(&header, data);
    header.IdType = FDCAN_STANDARD_ID; header.FDFormat = FDCAN_FD_CAN;
    host_can_feedback(&header, data);
    /* Real HAL copies all 64 bytes before Board validates classic/DLC8. */
    header.DataLength = FDCAN_DLC_BYTES_64; memset(data + 8U, 0xA5, 56U);
    host_can_feedback(&header, data);
    header.DataLength = FDCAN_DLC_BYTES_8;
    header.FDFormat = FDCAN_CLASSIC_CAN; data[0] = 0x9AU;
    host_can_feedback(&header, data);
    header.Identifier = 0x7FEU; data[0] = 0xA1U;
    host_can_feedback(&header, data);
    CHECK(registry.motors[4].rx_count == 1U);
    CHECK(pace_can_port_stats()->rx_invalid_count == 7U);
    header.Identifier = pace_motor_manifest[4].feedback_can_id;
    host_can_feedback(&header, data);
    CHECK(registry.motors[4].rx_count == 2U);
    host_can_rx_lost();
    CHECK(pace_can_port_stats()->rx_fifo_lost_count == 1U);
    CHECK(pace_can_port_stats()->rx_fifo_full_count == 1U);
    puts("PASS CAN short/64-byte-FD/non-data/non-classic/unknown/opcode rejection and RX loss counters");
}

static void test_registry_age(void)
{
    pace_motor_registry_t registry;
    pace_safety_limits_t limits;
    pace_motor_state_t *state;
    uint8_t index;
    pace_motor_registry_init(&registry);
    pace_safety_default_limits(&limits);
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        registry.motors[index].rx_valid = registry.motors[index].tx_valid = true;
        registry.motors[index].last_rx_timestamp_us = registry.motors[index].last_tx_timestamp_us = 100000U;
    }
    state = &registry.motors[0];
    state->last_rx_timestamp_us = state->last_tx_timestamp_us = 100300U;
    pace_motor_registry_refresh_ages(&registry, 100000U, 30000U);
    CHECK(state->rx_age_us == 0U && state->tx_age_us == 0U && !state->online);
    CHECK(!state->tx_valid);
    CHECK(pace_safety_check(&registry, &limits, 100000U, 0U, false, false) == PACE_SAFETY_FEEDBACK_TIMEOUT);
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        /* A new valid confirmation restores history; ordinary wrap is valid. */
        registry.motors[index].tx_valid = true;
        registry.motors[index].last_rx_timestamp_us = registry.motors[index].last_tx_timestamp_us = UINT32_MAX - 99U;
    }
    pace_motor_registry_refresh_ages(&registry, 100U, 30000U);
    CHECK(state->rx_age_us == 200U && state->tx_age_us == 200U && state->online && state->tx_valid);
    CHECK(pace_safety_check(&registry, &limits, 100U, 0U, false, false) == PACE_SAFETY_NONE);
    pace_motor_registry_refresh_ages(&registry, 30100U, 30000U);
    CHECK(!state->online && state->rx_age_us == 30200U);
    puts("PASS future RX faults/future TX invalidates and ordinary uint32 wrap stays valid");
}

static void test_capture_transport(void)
{
    pace_capture_t capture;
    uint8_t frame[PACE_EVENT_FRAME_SIZE];
    pace_event_frame_data_t event = {0};
    uint16_t count, length;
    const uint8_t *peeked;
    host_reset(); pace_uart_port_init(); pace_capture_init(&capture);
    for (count = 0U; count < PACE_CAPTURE_SLOT_COUNT - 1U; ++count)
    {
        CHECK(pace_frame_encode_event(frame, sizeof(frame), count, &event) == sizeof(frame));
        CHECK(pace_capture_push(&capture, frame, sizeof(frame)));
    }
    CHECK(pace_capture_depth(&capture) == PACE_CAPTURE_SLOT_COUNT - 1U);
    CHECK(!pace_capture_push(&capture, frame, sizeof(frame)));
    CHECK(capture.rejected_frames == 1U && capture.high_water == PACE_CAPTURE_SLOT_COUNT - 1U);
    for (count = 0U; count < PACE_CAPTURE_SLOT_COUNT - 1U; ++count)
    {
        CHECK(pace_capture_peek(&capture, &peeked, &length));
        CHECK(length == sizeof(frame) && get32(peeked, 6U) == count);
        CHECK(pace_capture_pop(&capture));
    }
    CHECK(pace_capture_depth(&capture) == 0U);
    pace_capture_init(&capture);
    CHECK(pace_capture_push(&capture, frame, sizeof(frame)));
    pace_transport_init(&capture);
    host_uart_fail_next(2U);
    pace_transport_kick(); pace_transport_kick();
    CHECK(pace_capture_depth(&capture) == 1U && host_uart_size() == 0U);
    CHECK(pace_transport_stats()->retries == 2U && !pace_transport_stats()->failed);
    pace_transport_kick();
    CHECK(pace_capture_depth(&capture) == 0U && host_uart_size() == sizeof(frame));
    CHECK(pace_transport_stats()->frames_sent == 1U);
    CHECK(pace_capture_push(&capture, frame, sizeof(frame)));
    host_uart_fail_next(3U);
    pace_transport_kick(); pace_transport_kick(); pace_transport_kick();
    CHECK(pace_transport_stats()->failed && pace_capture_depth(&capture) == 1U);
    pace_transport_kick();
    CHECK(pace_capture_depth(&capture) == 1U);
    puts("PASS capture FIFO/full/wrap and transport retry ownership");
}

static void test_enable_sequence(void)
{
    static const uint8_t expected_left[3] = {0U, 1U, 4U};
    static const uint8_t expected_right[3] = {2U, 3U, 5U};
    host_can_message_t message;
    uint32_t index;
    uint32_t start;
    initialize_app();
#if PACE_ONE_WAY_AUTOSTART
    for (index = 0U; index < PACE_AUTOSTART_DELAY_MS - 1U; ++index) regular_step(true);
    CHECK(host_can_queued() == 0U && host_uart_size() == 0U);
    start = PACE_AUTOSTART_DELAY_MS * 1000U;
#else
    send_host(PACE_HOST_CONFIGURE, PACE_SESSION_COMMISSIONING, false);
    send_host(PACE_HOST_START, -1, false);
    start = host_now_us(); /* START poll itself queued the first pair. */
#endif
    for (index = 0U; index < 3U * PACE_MOTOR_ENABLE_REPEAT_COUNT; ++index)
    {
        const uint8_t pair = (uint8_t)(index / PACE_MOTOR_ENABLE_REPEAT_COUNT);
        host_set_time(start + index * 1000U);
        pace_app_poll();
        CHECK(host_can_queued() == 2U);
        CHECK(host_can_peek(&message));
        CHECK(message.header.Identifier == pace_motor_manifest[expected_left[pair]].command_can_id);
        CHECK(message.enqueue_time_us == start + index * 1000U);
        CHECK(message.data[0] == ((pair < 2U) ? 0xFFU : 0x88U));
        CHECK(host_can_confirm_next());
        CHECK(host_can_peek(&message));
        CHECK(message.header.Identifier == pace_motor_manifest[expected_right[pair]].command_can_id);
        CHECK(host_can_confirm_next());
    }
    CHECK(pace_can_port_pending_count() == 0U);
    puts("PASS nonblocking autostart/paired enable order and 1 ms repetition");
}

static stream_summary_t run_complete(void)
{
    pace_experiment_config_t config;
    uint32_t max_ms = 5000U, stage, count;
    stream_summary_t summary;
    start_running();
    pace_experiment_default_config(&config, PACE_SESSION_COMMISSIONING);
    for (stage = 0U; stage < config.stage_count; ++stage) max_ms += config.stages[stage].duration_ms;
    for (count = 0U; count < max_ms; ++count)
    {
        regular_step(true);
        CHECK(pace_app_state() != PACE_STATE_FAULT);
        if (pace_app_state() == PACE_STATE_COMPLETE) break;
    }
    CHECK(count < max_ms);
    for (count = 0U; count < 200U; ++count) regular_step(true);
    drain_transport();
    summary = inspect_stream();
    CHECK(summary.headers == 1U && summary.footers == 1U);
    CHECK(summary.stages == 26U && summary.stage_mask == ((1U << 26U) - 1U));
    CHECK(summary.samples == (max_ms - 5000U) / 4U && summary.statuses == summary.samples / 25U);
    CHECK(summary.stop_reason == 0U && summary.statistics_complete == 1U);
    CHECK(summary.overflow == 0U && summary.dropped_frames == 0U);
    CHECK(summary.recipe_hash == config.experiment_config_hash);
    CHECK(pace_can_port_pending_count() == 0U && host_can_queued() == 0U);
    printf("PASS production 26-stage complete: %u samples, %u frames, %zu bytes, recipe %08X\n",
           summary.samples, summary.frames, host_uart_size(), summary.recipe_hash);
    return summary;
}

static void test_fault_paths(void)
{
    stream_summary_t summary;
    uint32_t count;
    start_running();
    host_can_confirm_all();
    host_set_time(host_now_us() + 2000U); host_can_feedback_all(); host_tick();
    host_set_time(host_now_us() + 2000U); host_can_feedback_all(); host_tick();
    pace_app_poll();
    CHECK(pace_app_state() == PACE_STATE_FAULT);
    CHECK(host_can_queued() == 6U); /* Safe commands only; no stale control burst. */
    finish_fault(8U);
    summary = inspect_stream(); CHECK((summary.event_mask & (1U << 10U)) != 0U);

    start_running(); host_can_confirm_all(); host_can_rx_lost();
    regular_step(true); regular_step(true);
    CHECK(pace_app_state() == PACE_STATE_FAULT);
    finish_fault(3U);
    summary = inspect_stream(); CHECK((summary.event_mask & (1U << 3U)) != 0U);

    start_running(); host_can_set_bus_off(true);
    regular_step(true); regular_step(true); finish_fault(3U);

    start_running();
    for (count = 0U; count < 700U && pace_app_state() != PACE_STATE_FAULT; ++count) regular_step(false);
    CHECK(pace_app_state() == PACE_STATE_FAULT);
    CHECK(count < 700U);
    for (count = 0U; count < 20U; ++count) regular_step(true); drain_transport();
    summary = inspect_stream();
    CHECK(summary.footers == 1U && summary.stop_reason == 4U);
    CHECK(summary.overflow == 1U && summary.dropped_frames > 0U && summary.statistics_complete == 0U);

    start_running(); host_can_confirm_all();
    host_uart_fail_next(3U);
    /* Ask the real sample path to populate the real transport queue. */
    regular_step(false); regular_step(false); regular_step(false); regular_step(false);
    pace_app_transport_poll(); pace_app_transport_poll(); pace_app_transport_poll(); pace_app_poll();
    CHECK(pace_transport_stats()->failed && pace_app_state() == PACE_STATE_FAULT);
    summary = inspect_stream(); CHECK(summary.footers == 0U);
    host_can_confirm_all(); pace_app_poll(); drain_transport();
    summary = inspect_stream(); CHECK(summary.footers == 0U);
    puts("PASS app scheduler/RX loss/bus-off/queue overflow/UART failure terminal behavior");
}

static void test_safe_confirmation_order(void)
{
    stream_summary_t summary;
    uint32_t index;
    start_running();
    CHECK(host_can_queued() == 6U); /* An ordinary command cycle is still in flight. */
    host_can_rx_lost();
    host_set_time(host_now_us() + 2000U); host_can_feedback_all(); host_tick(); pace_app_poll();
    CHECK(pace_app_state() == PACE_STATE_FAULT && host_can_queued() == 8U);
    /* Confirm only the six old MIT/torque commands, not the new safe frames. */
    for (index = 0U; index < 6U; ++index) CHECK(host_can_confirm_next());
    pace_app_poll(); drain_transport();
    CHECK(host_can_queued() == 6U);
    summary = inspect_stream(); CHECK(summary.footers == 0U);
    host_can_confirm_all(); pace_app_poll(); drain_transport();
    summary = inspect_stream();
    CHECK(summary.footers == 1U && summary.stop_reason == 3U);
    CHECK(pace_can_port_pending_count() == 0U);
    puts("PASS ordinary TX confirmation cannot release safe-command inflight mask");
}

static void test_snapshot_current_time(void)
{
    start_running(); host_can_confirm_all();
    host_set_time(host_now_us() + 1000U); host_tick(); /* Queued historical tick. */
    host_set_time(host_now_us() + 1000U); host_can_feedback_all();
    pace_app_poll(); /* Snapshot clock is newer than feedback, not historical tick. */
    CHECK(pace_app_state() == PACE_STATE_RUNNING);
    CHECK(host_can_queued() <= 6U);
    puts("PASS historical tick plus newer ISR feedback uses current snapshot clock");
}

static void test_feedback_timeout(void)
{
    uint32_t index;
    start_running();
    for (index = 0U; index < 40U; ++index)
    {
        host_set_time(host_now_us() + 1000U);
        host_can_confirm_all();
        if ((host_now_us() % 2000U) == 0U) host_tick();
        pace_app_poll(); pace_app_transport_poll();
        if (pace_app_state() == PACE_STATE_FAULT) break;
    }
    CHECK(index < 40U);
    finish_fault(3U);
    puts("PASS stale feedback produces safe fault and incomplete footer");
}

static void test_future_snapshot_validity(void)
{
    uint32_t now;
    uint8_t index;
    pace_encoded_command_t command;
    size_t offset;
    const uint8_t *last_sample = 0;
    uint32_t sample_count = 0U;
    start_running();
    host_can_confirm_all();
    now = host_now_us();
    host_set_time(now + 4000U);
    host_can_feedback_all(); /* Deliberately future relative to the snapshot clock. */
    host_set_time(now + 2000U);
    host_tick(); pace_app_poll();
    CHECK(pace_app_state() == PACE_STATE_FAULT);
    finish_fault(3U);

    start_running();
    host_can_confirm_all();
    regular_step(true); regular_step(true); /* Next running tick emits no sample. */
    now = host_now_us();
    host_set_time(now + 6000U);
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        const bool packed = (index < PACE_DM_MOTOR_COUNT) ?
            pace_dm_pack_mit(index, 0.0f, 0.0f, 20.0f, 0.6f, 0.0f, &command) :
            pace_lk_pack_torque(index, 0.0f, &command);
        CHECK(packed && pace_can_port_send(&command) == PACE_CAN_SEND_OK);
    }
    host_can_confirm_all(); /* Real Board records future service timestamps. */
    host_set_time(now + 2000U);
    host_can_feedback_all(); host_tick(); pace_app_poll(); drain_transport();
    CHECK(pace_app_state() == PACE_STATE_RUNNING);
    for (offset = 0U; offset < host_uart_size(); )
    {
        uint16_t length;
        pace_frame_type_t type;
        CHECK(pace_frame_validate(host_uart_bytes() + offset, host_uart_size() - offset,
                                   &type, 0, &length));
        if (type == PACE_FRAME_SAMPLE)
        {
            last_sample = host_uart_bytes() + offset;
            sample_count++;
        }
        offset += length;
    }
    CHECK(sample_count == 2U && last_sample != 0);
    CHECK(last_sample[18] == 0U); /* Future TX confirmations are not fresh commands. */
    CHECK(last_sample[17] == 0x3FU && last_sample[19] == 0x3FU);
    CHECK(get16(last_sample, 22U) == 0U);
    puts("PASS real app future RX safety fault and future TX Sample validity mask");
}

static void test_stop_confirmation_timeout(void)
{
    stream_summary_t summary;
    uint32_t start;
    start_running(); host_can_confirm_all();
#if !PACE_ONE_WAY_AUTOSTART
    send_host(PACE_HOST_ABORT, -1, true);
#else
    host_can_rx_lost(); regular_step(true); regular_step(true);
#endif
    CHECK(pace_app_state() == PACE_STATE_FAULT);
    CHECK(host_can_queued() == 6U);
    drain_transport();
    summary = inspect_stream(); CHECK(summary.footers == 0U);
    start = host_now_us();
    host_set_time(start + 1999000U); pace_app_poll(); drain_transport();
    summary = inspect_stream(); CHECK(summary.footers == 0U);
    host_set_time(start + 2001000U); pace_app_poll(); drain_transport();
    summary = inspect_stream();
    CHECK(summary.footers == 1U && summary.stop_reason == 9U && !summary.statistics_complete);
    CHECK((summary.event_mask & (1U << 11U)) != 0U);
    puts("PASS unconfirmed safe command blocks footer until explicit 2 s failure");
}

#if !PACE_ONE_WAY_AUTOSTART
static void test_host_control(void)
{
    stream_summary_t summary;
    uint32_t count, starts;
    initialize_app();
    send_host(PACE_HOST_CONFIGURE, PACE_SESSION_FINAL_IDENTIFICATION, true);
    CHECK(pace_app_state() == PACE_STATE_SAFE_IDLE && host_can_queued() == 0U);
    send_host(PACE_HOST_CONFIGURE, PACE_SESSION_COMMISSIONING, true);
    CHECK(pace_app_state() == PACE_STATE_CONFIGURED);
    send_host(PACE_HOST_CONFIGURE, PACE_SESSION_FINAL_IDENTIFICATION, true);
    CHECK(pace_app_state() == PACE_STATE_SAFE_IDLE);
    send_host(PACE_HOST_START, -1, true);
    CHECK(pace_app_state() == PACE_STATE_SAFE_IDLE && host_can_queued() == 0U);
    send_host(PACE_HOST_CONFIGURE, PACE_SESSION_COMMISSIONING, true);
    CHECK(pace_app_state() == PACE_STATE_CONFIGURED);
    send_host(PACE_HOST_START, -1, true);
    starts = host_uart_rx_starts();
    host_uart_receive_error();
    CHECK(host_uart_rx_starts() == starts); /* HAL error ISR does not restart DMA. */
    pace_app_poll();
    CHECK(host_uart_rx_starts() == starts + 1U);
    CHECK(pace_uart_port_stats()->rx_restart_count == 1U);
    for (count = 0U; count < 200U && pace_app_state() != PACE_STATE_RUNNING; ++count) regular_step(true);
    CHECK(pace_app_state() == PACE_STATE_RUNNING);
    send_host(PACE_HOST_STOP, -1, true);
    CHECK(pace_app_state() == PACE_STATE_STOPPING);
    for (count = 0U; count < PACE_STAGE_STOP_DURATION_MS + 100U; ++count)
    {
        regular_step(true);
        if (pace_app_state() == PACE_STATE_COMPLETE) break;
    }
    CHECK(pace_app_state() == PACE_STATE_COMPLETE);
    drain_transport();
    summary = inspect_stream();
    CHECK(summary.footers == 1U && summary.stop_reason == 1U && summary.statistics_complete == 1U);
    CHECK(pace_can_port_pending_count() == 0U);
    /* Same app object can configure and run another session after footer drained. */
    host_uart_clear();
    send_host(PACE_HOST_CONFIGURE, PACE_SESSION_COMMISSIONING, false);
    CHECK(pace_app_state() == PACE_STATE_CONFIGURED);
    send_host(PACE_HOST_START, -1, false);
    CHECK(pace_app_state() == PACE_STATE_ARMED);
    send_host(PACE_HOST_STOP, -1, false);
    CHECK(host_can_queued() >= 6U);
    drain_transport(); summary = inspect_stream(); CHECK(summary.footers == 0U);
    host_can_confirm_all(); pace_app_poll(); drain_transport();
    summary = inspect_stream(); CHECK(summary.footers == 1U && summary.stop_reason == 1U);
    puts("PASS host fragmented commands/final lock/RX task recovery/STOP/restart");
}
#endif

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "--fixture") == 0)
    {
        (void)run_complete();
        CHECK(host_uart_write_file(argv[2]));
        printf("Fixture: %s (%zu bytes; synthetic steady feedback, production C wire)\n", argv[2], host_uart_size());
        return 0;
    }
    CHECK(argc == 1);
#ifdef PACE_HAS_CONTRACT_TEST
    pace_test_experiment_contract();
#endif
    test_can_confirmation_and_time();
    test_can_validation();
    test_registry_age();
    test_capture_transport();
    test_enable_sequence();
    (void)run_complete();
    test_fault_paths();
    test_safe_confirmation_order();
    test_snapshot_current_time();
    test_feedback_timeout();
    test_future_snapshot_validity();
    test_stop_confirmation_timeout();
#if !PACE_ONE_WAY_AUTOSTART
    test_host_control();
#endif
    initialize_app();
    CHECK(pace_can_port_stats()->queued_count[0] == 0U && !pace_transport_stats()->failed);
    printf("PASS all native checks (%s)\n", PACE_ONE_WAY_AUTOSTART ? "passive" : "active");
    return 0;
}
