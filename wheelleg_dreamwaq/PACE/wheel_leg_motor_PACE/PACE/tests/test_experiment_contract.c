/* Run against the production experiment/excitation/frame implementation. */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pace_crc.h"
#include "pace_excitation.h"
#include "pace_experiment.h"
#include "pace_experiment_config.h"

static void contract_feedback(pace_motor_registry_t *registry, uint32_t now_us)
{
    uint8_t motor;
    memset(registry, 0, sizeof(*registry));
    for (motor = 0U; motor < PACE_MOTOR_COUNT; ++motor)
    {
        registry->motors[motor].rx_valid = true;
        registry->motors[motor].online = true;
        registry->motors[motor].last_rx_timestamp_us = now_us;
        if (motor < PACE_DM_MOTOR_COUNT)
        {
            registry->motors[motor].feedback.dm.position_rad = (float)motor * 0.1f;
        }
    }
}

static void contract_begin(pace_experiment_t *experiment, pace_motor_registry_t *registry,
                           const pace_experiment_config_t *config, uint32_t now_us)
{
    pace_experiment_init(experiment);
    assert(pace_experiment_configure(experiment, config));
    assert(pace_experiment_arm(experiment));
    contract_feedback(registry, now_us);
    assert(pace_experiment_begin(experiment, registry, now_us));
}

static void contract_crc(uint8_t *frame, size_t length)
{
    uint32_t crc = pace_crc32_iso_hdlc(frame, length - 4U);
    frame[length - 4U] = (uint8_t)crc;
    frame[length - 3U] = (uint8_t)(crc >> 8);
    frame[length - 2U] = (uint8_t)(crc >> 16);
    frame[length - 1U] = (uint8_t)(crc >> 24);
}

void pace_test_experiment_contract(void)
{
    pace_experiment_config_t config;
    pace_experiment_config_t changed;
    pace_experiment_t experiment;
    pace_motor_registry_t registry;
    pace_experiment_output_t output;
    pace_stage_config_frame_data_t stage_frame;
    pace_session_header_data_t header;
    uint8_t wire[PACE_STAGE_CONFIG_FRAME_SIZE];
    uint8_t legacy[PACE_LEGACY_STAGE_CONFIG_FRAME_SIZE];
    uint32_t start = 100000U;
    uint32_t now;
    uint32_t sequence;
    uint32_t recipe_hash;
    uint16_t length;
    uint8_t stage;
    uint8_t motor;
    uint8_t fit_leg = 0U;
    uint8_t valid_leg = 0U;
    uint8_t fit_torque = 0U;
    uint8_t valid_torque = 0U;
    uint8_t fit_velocity = 0U;
    uint8_t valid_velocity = 0U;
    float held[PACE_DM_MOTOR_COUNT];

    pace_experiment_default_config(&config, PACE_SESSION_COMMISSIONING);
    assert(config.stage_count == 26U);
    assert(pace_experiment_config_is_valid(&config));
    recipe_hash = config.experiment_config_hash;
    assert(recipe_hash != 0U);
    for (stage = 0U; stage < config.stage_count; ++stage)
    {
        const pace_stage_definition_t *definition = &config.stages[stage];
        assert(definition->segment_id == (uint16_t)(stage + 1U));
        assert(definition->action_update_rate_hz == 50U);
        assert(definition->command_hold_method == PACE_HOLD_ZERO_ORDER);
        assert(pace_experiment_stage_bus_load_percent(definition) <= 70.0f);
        if (definition->stage_id == PACE_STAGE_DM_FIT) ++fit_leg;
        if (definition->stage_id == PACE_STAGE_DM_VALIDATION) ++valid_leg;
        if ((definition->stage_id == PACE_STAGE_LK_TORQUE) && (definition->role == PACE_ROLE_FIT)) ++fit_torque;
        if ((definition->stage_id == PACE_STAGE_LK_TORQUE) && (definition->role == PACE_ROLE_VALIDATION)) ++valid_torque;
        if ((definition->stage_id == PACE_STAGE_LK_VELOCITY) && (definition->role == PACE_ROLE_FIT)) ++fit_velocity;
        if ((definition->stage_id == PACE_STAGE_LK_VELOCITY) && (definition->role == PACE_ROLE_VALIDATION)) ++valid_velocity;
    }
    assert(fit_leg == 8U && valid_leg == 4U);
    assert(fit_torque == 4U && valid_torque == 2U);
    assert(fit_velocity == 4U && valid_velocity == 2U);
    assert(config.stages[1].group_mask == 0x03U);
    assert(config.stages[3].group_mask == 0x0CU);
    assert(config.stages[5].excitation_multiplier[2] == -1.0f);
    assert(config.stages[7].excitation_multiplier[2] == 1.0f);

    changed = config;
    changed.stages[1].phase_rad = 0.125f;
    assert(pace_experiment_config_hash(&changed) != recipe_hash);
    assert(!pace_experiment_config_is_valid(&changed));
    changed.experiment_config_hash = pace_experiment_config_hash(&changed);
    assert(pace_experiment_config_is_valid(&changed));
    changed.stages[1].amplitude = 0.20f;
    changed.experiment_config_hash = pace_experiment_config_hash(&changed);
    assert(!pace_experiment_config_is_valid(&changed));
    changed = config;
    changed.stages[1].duration_ms = 0xFFFFFFFFUL;
    changed.experiment_config_hash = pace_experiment_config_hash(&changed);
    assert(!pace_experiment_config_is_valid(&changed));
    changed = config;
    changed.stages[1].role = PACE_ROLE_VALIDATION;
    changed.experiment_config_hash = pace_experiment_config_hash(&changed);
    assert(!pace_experiment_config_is_valid(&changed));
    changed = config;
    changed.stages[1].command_rate_hz[0] = 0U;
    changed.experiment_config_hash = pace_experiment_config_hash(&changed);
    assert(!pace_experiment_config_is_valid(&changed));
    pace_experiment_default_config(&changed, PACE_SESSION_FINAL_IDENTIFICATION);
    assert(changed.stage_count == 0U);
    assert(!pace_experiment_final_config(&changed));

    contract_begin(&experiment, &registry, &config, start);
    assert(pace_experiment_step(&experiment, start, &output));
    now = start + config.stages[0].duration_ms * 1000U;
    assert(pace_experiment_step(&experiment, now, &output));
    assert(output.stage_id == PACE_STAGE_DM_FIT && output.active_mask == 0x03U);
    assert(output.stage_changed && output.target_updated && output.send_mask == 0x3FU);
    memcpy(held, output.dm_position_rad, sizeof(held));
    sequence = output.target_update_sequence;
    for (stage = 1U; stage < 10U; ++stage)
    {
        assert(pace_experiment_step(&experiment, now + (uint32_t)stage * 2000U, &output));
        assert(!output.target_updated && output.target_update_sequence == sequence);
        assert((output.send_mask & 0x03U) == 0x03U);
        for (motor = 0U; motor < PACE_DM_MOTOR_COUNT; ++motor)
        {
            assert(output.dm_position_rad[motor] == held[motor]);
        }
    }
    assert(pace_experiment_step(&experiment, now + 20000U, &output));
    assert(output.target_updated && output.target_update_sequence == sequence + 1U);
    assert(output.dm_position_rad[0] != held[0]);
    assert(pace_experiment_step(&experiment, now + 20000U, &output));
    assert(output.send_mask == 0U && !output.target_updated);
    assert(pace_experiment_fill_stage_frame(&experiment, &stage_frame));
    assert(stage_frame.experiment_config_hash == recipe_hash);
    assert(stage_frame.dm_center_rad[3] == 0.3f);
    assert(pace_frame_encode_stage_config(wire, sizeof(wire), 17U, &stage_frame) == 128U);
    assert(wire[2] == 2U && wire[72] == PACE_ROLE_FIT && wire[73] == 3U);
    assert(wire[74] == 2U && wire[76] == 50U && wire[78] == PACE_HOLD_ZERO_ORDER);
    assert(pace_frame_validate(wire, sizeof(wire), 0, &sequence, &length));
    assert(length == 128U && sequence == 17U);
    assert(pace_frame_encode_stage_config(wire, 127U, 17U, &stage_frame) == 0U);

    memset(&header, 0, sizeof(header));
    header.timestamp_timebase = PACE_TIMESTAMP_TIM5_ISR_SERVICE;
    header.control_rate_hz = 500U;
    assert(pace_frame_encode_session_header(wire, sizeof(wire), 1U, &header) == 58U);
    assert(wire[13] == 1U && wire[32] == 0xF4U && wire[33] == 1U);
    memset(legacy, 0, sizeof(legacy));
    legacy[0] = PACE_FRAME_MAGIC_0;
    legacy[1] = PACE_FRAME_MAGIC_1;
    legacy[2] = 1U;
    legacy[3] = PACE_FRAME_STAGE_CONFIG;
    legacy[4] = 76U;
    contract_crc(legacy, sizeof(legacy));
    assert(pace_frame_validate(legacy, sizeof(legacy), 0, 0, 0));
    legacy[2] = 2U;
    contract_crc(legacy, sizeof(legacy));
    assert(!pace_frame_validate(legacy, sizeof(legacy), 0, 0, 0));

    assert(pace_experiment_request_stop(&experiment, now + 20300U));
    assert(!pace_experiment_step(&experiment, now + 20000U, &output));
    assert(experiment.state == PACE_STATE_STOPPING);
    assert(pace_experiment_step(&experiment, now + 20300U + 999999U, &output));
    assert(!output.session_complete);
    assert(pace_experiment_step(&experiment, now + 20300U + 1000000U, &output));
    assert(output.session_complete);

    start = 0xFFFFC000UL;
    contract_begin(&experiment, &registry, &config, start);
    now = start;
    for (stage = 0U; stage < config.stage_count; ++stage)
    {
        assert(pace_experiment_step(&experiment, now, &output));
        assert(output.stage_changed);
        assert(pace_experiment_fill_stage_frame(&experiment, &stage_frame));
        assert(stage_frame.segment_id == (uint16_t)(stage + 1U));
        now += config.stages[stage].duration_ms * 1000U;
    }
    assert(pace_experiment_step(&experiment, now, &output));
    assert(output.session_complete && experiment.state == PACE_STATE_COMPLETE);

    pace_experiment_init(&experiment);
    assert(pace_experiment_configure(&experiment, &config));
    assert(pace_experiment_arm(&experiment));
    contract_feedback(&registry, 100000U);
    registry.motors[0].last_rx_timestamp_us = 100300U;
    assert(!pace_experiment_begin(&experiment, &registry, 100000U));
    registry.motors[0].last_rx_timestamp_us = 100000U - 30001U;
    assert(!pace_experiment_begin(&experiment, &registry, 100000U));
    registry.motors[0].last_rx_timestamp_us = 100000U;
    registry.motors[0].feedback.dm.position_rad = 12.49f;
    assert(!pace_experiment_begin(&experiment, &registry, 100000U));

    for (stage = 0U; stage < 4U; ++stage)
    {
        unsigned sample;
        for (sample = 0U; sample <= 6000U; ++sample)
        {
            const float value = pace_excitation_validation((float)sample * 0.001f,
                6.0f, 0.12f, 0.35f, 6.0f, 0.47f + (float)stage * 0.23f);
            assert(fabsf(value) <= 0.120001f);
        }
    }
    printf("Recipe hash: %08X, config bytes: %u, experiment bytes: %u\n",
        (unsigned)recipe_hash, (unsigned)sizeof(config), (unsigned)sizeof(experiment));
    printf("PASS experiment contract: 26 segments, content hash, 50 Hz hold, wrap/STOP, V1/V2 wire\n");
}

#ifdef PACE_EXPERIMENT_TEST_STANDALONE
int main(void)
{
    pace_test_experiment_contract();
    return 0;
}
#endif
