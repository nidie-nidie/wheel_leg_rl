#include "pace_experiment.h"

#include <string.h>

#include "dm_motor.h"
#include "pace_excitation.h"
#include "pace_experiment_config.h"

#define PACE_CAN_BITS_PER_COMMAND_FEEDBACK_PAIR 260U

static const float pace_dm_excitation_sign[PACE_DM_MOTOR_COUNT] = {
    1.0f, -1.0f, -1.0f, 1.0f};
static const float pace_lk_excitation_sign[PACE_LK_MOTOR_COUNT] = {
    1.0f, -1.0f};

static void pace_stage_set_modes(pace_stage_definition_t *stage,
                                 pace_command_mode_t lk_mode)
{
    uint8_t index;
    for (index = 0U; index < PACE_DM_MOTOR_COUNT; ++index)
    {
        stage->command_mode[index] = (uint8_t)PACE_COMMAND_DM_MIT;
    }
    for (index = 0U; index < PACE_LK_MOTOR_COUNT; ++index)
    {
        stage->command_mode[PACE_DM_MOTOR_COUNT + index] = (uint8_t)lk_mode;
    }
}

static void pace_stage_set_rates(pace_stage_definition_t *stage,
                                 uint16_t dm_rate_hz,
                                 uint16_t lk_rate_hz)
{
    uint8_t index;
    for (index = 0U; index < PACE_DM_MOTOR_COUNT; ++index)
    {
        stage->command_rate_hz[index] = dm_rate_hz;
    }
    for (index = 0U; index < PACE_LK_MOTOR_COUNT; ++index)
    {
        stage->command_rate_hz[PACE_DM_MOTOR_COUNT + index] = lk_rate_hz;
    }
}

void pace_experiment_default_config(pace_experiment_config_t *config,
                                    pace_session_type_t session_type)
{
    pace_stage_definition_t *stage;
    if (config == 0)
    {
        return;
    }
    memset(config, 0, sizeof(*config));
    config->session_type = session_type;
    config->experiment_config_hash = PACE_DEFAULT_CONFIG_HASH;
    config->safety_limit_set_id = PACE_DEFAULT_SAFETY_LIMIT_SET_ID;
    config->dm_kp = PACE_DM_BASELINE_KP;
    config->dm_kd = PACE_DM_BASELINE_KD;

    stage = &config->stages[0];
    stage->stage_id = PACE_STAGE_STATIC;
    stage->duration_ms = PACE_STAGE_STATIC_DURATION_MS;
    pace_stage_set_modes(stage, PACE_COMMAND_LK_TORQUE);
    pace_stage_set_rates(stage, 100U, 100U);

    stage = &config->stages[1];
    stage->stage_id = PACE_STAGE_DM_FIT;
    stage->duration_ms = PACE_STAGE_DM_FIT_DURATION_MS;
    stage->excitation_type = PACE_EXCITATION_CHIRP;
    stage->amplitude = PACE_DM_FIT_AMPLITUDE_RAD;
    stage->frequency_start_hz = PACE_DM_FIT_FREQUENCY_START_HZ;
    stage->frequency_end_hz = PACE_DM_FIT_FREQUENCY_END_HZ;
    pace_stage_set_modes(stage, PACE_COMMAND_LK_TORQUE);
    pace_stage_set_rates(stage, 500U, 100U);

    stage = &config->stages[2];
    stage->stage_id = PACE_STAGE_DM_VALIDATION;
    stage->duration_ms = PACE_STAGE_DM_VALIDATION_DURATION_MS;
    stage->excitation_type = PACE_EXCITATION_VALIDATION;
    stage->amplitude = PACE_DM_VALIDATION_AMPLITUDE_RAD;
    stage->frequency_start_hz = PACE_DM_VALIDATION_FREQUENCY_START_HZ;
    stage->frequency_end_hz = PACE_DM_VALIDATION_FREQUENCY_END_HZ;
    stage->phase_rad = 0.47f;
    pace_stage_set_modes(stage, PACE_COMMAND_LK_TORQUE);
    pace_stage_set_rates(stage, 500U, 100U);

    stage = &config->stages[3];
    stage->stage_id = PACE_STAGE_LK_TORQUE;
    stage->duration_ms = PACE_STAGE_LK_TORQUE_DURATION_MS;
    stage->excitation_type = PACE_EXCITATION_CHIRP;
    stage->amplitude = PACE_LK_TORQUE_AMPLITUDE_NM;
    stage->frequency_start_hz = PACE_LK_TORQUE_FREQUENCY_START_HZ;
    stage->frequency_end_hz = PACE_LK_TORQUE_FREQUENCY_END_HZ;
    pace_stage_set_modes(stage, PACE_COMMAND_LK_TORQUE);
    pace_stage_set_rates(stage, 100U, 500U);

    stage = &config->stages[4];
    stage->stage_id = PACE_STAGE_LK_VELOCITY;
    stage->duration_ms = PACE_STAGE_LK_VELOCITY_DURATION_MS;
    stage->excitation_type = PACE_EXCITATION_CHIRP;
    stage->amplitude = PACE_LK_VELOCITY_AMPLITUDE_RAD_S;
    stage->frequency_start_hz = PACE_LK_VELOCITY_FREQUENCY_START_HZ;
    stage->frequency_end_hz = PACE_LK_VELOCITY_FREQUENCY_END_HZ;
    pace_stage_set_modes(stage, PACE_COMMAND_LK_VELOCITY);
    pace_stage_set_rates(stage, 100U, 500U);

    stage = &config->stages[5];
    stage->stage_id = PACE_STAGE_STOP;
    stage->duration_ms = PACE_STAGE_STOP_DURATION_MS;
    pace_stage_set_modes(stage, PACE_COMMAND_LK_TORQUE);
    pace_stage_set_rates(stage, 100U, 100U);
}

void pace_experiment_init(pace_experiment_t *experiment)
{
    if (experiment == 0)
    {
        return;
    }
    memset(experiment, 0, sizeof(*experiment));
    experiment->state = PACE_STATE_SAFE_IDLE;
}

float pace_experiment_stage_bus_load_percent(const pace_stage_definition_t *stage)
{
    uint32_t commands_per_second = 0U;
    uint8_t index;
    if (stage == 0)
    {
        return 100.0f;
    }
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        commands_per_second += stage->command_rate_hz[index];
    }
    return ((float)(commands_per_second * PACE_CAN_BITS_PER_COMMAND_FEEDBACK_PAIR) /
            (float)PACE_CAN_NOMINAL_BITRATE) * 100.0f;
}

bool pace_experiment_config_is_valid(const pace_experiment_config_t *config)
{
    static const pace_stage_id_t expected_stage_ids[PACE_EXPERIMENT_STAGE_COUNT] = {
        PACE_STAGE_STATIC,
        PACE_STAGE_DM_FIT,
        PACE_STAGE_DM_VALIDATION,
        PACE_STAGE_LK_TORQUE,
        PACE_STAGE_LK_VELOCITY,
        PACE_STAGE_STOP};
    uint8_t stage_index;
    uint8_t motor_index;
    if ((config == 0) ||
        ((config->session_type != PACE_SESSION_COMMISSIONING) &&
         (config->session_type != PACE_SESSION_FINAL_IDENTIFICATION)) ||
        (config->dm_kp != PACE_DM_BASELINE_KP) ||
        (config->dm_kd != PACE_DM_BASELINE_KD))
    {
        return false;
    }
    for (stage_index = 0U; stage_index < PACE_EXPERIMENT_STAGE_COUNT; ++stage_index)
    {
        const pace_stage_definition_t *stage = &config->stages[stage_index];
        if ((stage->stage_id != expected_stage_ids[stage_index]) ||
            (stage->duration_ms == 0U) ||
            (pace_experiment_stage_bus_load_percent(stage) > 70.0f))
        {
            return false;
        }
        for (motor_index = 0U; motor_index < PACE_MOTOR_COUNT; ++motor_index)
        {
            const uint16_t rate = stage->command_rate_hz[motor_index];
            if ((rate == 0U) || (rate > PACE_CONTROL_RATE_HZ) ||
                ((PACE_CONTROL_RATE_HZ % rate) != 0U))
            {
                return false;
            }
            if ((motor_index < PACE_DM_MOTOR_COUNT) &&
                (stage->command_mode[motor_index] != (uint8_t)PACE_COMMAND_DM_MIT))
            {
                return false;
            }
            if ((motor_index >= PACE_DM_MOTOR_COUNT) &&
                (stage->command_mode[motor_index] != (uint8_t)PACE_COMMAND_LK_TORQUE) &&
                (stage->command_mode[motor_index] != (uint8_t)PACE_COMMAND_LK_VELOCITY))
            {
                return false;
            }
        }
    }
    return true;
}

bool pace_experiment_configure(pace_experiment_t *experiment,
                               const pace_experiment_config_t *config)
{
    if ((experiment == 0) || !pace_experiment_config_is_valid(config) ||
        ((experiment->state != PACE_STATE_SAFE_IDLE) &&
         (experiment->state != PACE_STATE_CONFIGURED) &&
         (experiment->state != PACE_STATE_COMPLETE)))
    {
        return false;
    }
    experiment->config = *config;
    experiment->state = PACE_STATE_CONFIGURED;
    return true;
}

bool pace_experiment_arm(pace_experiment_t *experiment)
{
    if ((experiment == 0) || (experiment->state != PACE_STATE_CONFIGURED))
    {
        return false;
    }
    experiment->state = PACE_STATE_ARMED;
    return true;
}

bool pace_experiment_begin(pace_experiment_t *experiment,
                           const pace_motor_registry_t *registry,
                           uint32_t now_us)
{
    uint8_t index;
    if ((experiment == 0) || (registry == 0) ||
        (experiment->state != PACE_STATE_ARMED))
    {
        return false;
    }
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        if (!registry->motors[index].rx_valid)
        {
            return false;
        }
    }
    for (index = 0U; index < PACE_DM_MOTOR_COUNT; ++index)
    {
        experiment->dm_center_rad[index] = registry->motors[index].feedback.dm.position_rad;
    }
    experiment->stage_index = 0U;
    experiment->config_seq = 1U;
    experiment->session_start_us = now_us;
    experiment->stage_start_us = now_us;
    experiment->tick_in_stage = 0U;
    experiment->stage_announcement_pending = true;
    experiment->state = PACE_STATE_RUNNING;
    return true;
}

static uint8_t pace_experiment_active_mask(pace_stage_id_t stage_id)
{
    if ((stage_id == PACE_STAGE_DM_FIT) || (stage_id == PACE_STAGE_DM_VALIDATION))
    {
        return 0x0FU;
    }
    if ((stage_id == PACE_STAGE_LK_TORQUE) || (stage_id == PACE_STAGE_LK_VELOCITY))
    {
        return 0x30U;
    }
    return 0U;
}

static void pace_experiment_fill_commands(const pace_experiment_t *experiment,
                                          const pace_stage_definition_t *stage,
                                          uint32_t now_us,
                                          pace_experiment_output_t *output)
{
    const float time_s = (float)(now_us - experiment->stage_start_us) * 0.000001f;
    const float duration_s = (float)stage->duration_ms * 0.001f;
    float excitation = 0.0f;
    uint8_t index;

    if (stage->excitation_type == PACE_EXCITATION_CHIRP)
    {
        excitation = pace_excitation_chirp(time_s, duration_s, stage->amplitude,
                                           stage->frequency_start_hz,
                                           stage->frequency_end_hz,
                                           stage->phase_rad);
    }
    else if (stage->excitation_type == PACE_EXCITATION_VALIDATION)
    {
        excitation = pace_excitation_validation(time_s, duration_s, stage->amplitude,
                                                stage->frequency_start_hz,
                                                stage->frequency_end_hz,
                                                stage->phase_rad);
    }

    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        const uint16_t rate = stage->command_rate_hz[index];
        const uint32_t period_ticks = PACE_CONTROL_RATE_HZ / rate;
        output->command_mode[index] = (pace_command_mode_t)stage->command_mode[index];
        if ((experiment->tick_in_stage % period_ticks) == 0U)
        {
            output->send_mask |= (uint8_t)(1U << index);
        }
    }
    for (index = 0U; index < PACE_DM_MOTOR_COUNT; ++index)
    {
        float offset = 0.0f;
        if ((stage->stage_id == PACE_STAGE_DM_FIT) ||
            (stage->stage_id == PACE_STAGE_DM_VALIDATION))
        {
            offset = pace_dm_excitation_sign[index] * excitation;
        }
        output->dm_position_rad[index] = experiment->dm_center_rad[index] + offset;
        output->dm_velocity_rad_s[index] = 0.0f;
        output->dm_kp[index] = experiment->config.dm_kp;
        output->dm_kd[index] = experiment->config.dm_kd;
        output->dm_torque_nm[index] = 0.0f;
    }
    for (index = 0U; index < PACE_LK_MOTOR_COUNT; ++index)
    {
        if ((stage->stage_id == PACE_STAGE_LK_TORQUE) ||
            (stage->stage_id == PACE_STAGE_LK_VELOCITY))
        {
            output->lk_primary[index] = pace_lk_excitation_sign[index] * excitation;
        }
        else
        {
            output->lk_primary[index] = 0.0f;
        }
    }
}

bool pace_experiment_step(pace_experiment_t *experiment,
                          uint32_t now_us,
                          pace_experiment_output_t *output)
{
    const pace_stage_definition_t *stage;
    if ((experiment == 0) || (output == 0) ||
        ((experiment->state != PACE_STATE_RUNNING) &&
         (experiment->state != PACE_STATE_STOPPING)))
    {
        return false;
    }
    memset(output, 0, sizeof(*output));
    stage = &experiment->config.stages[experiment->stage_index];
    while ((now_us - experiment->stage_start_us) >= (stage->duration_ms * 1000U))
    {
        if (experiment->stage_index >= (PACE_EXPERIMENT_STAGE_COUNT - 1U))
        {
            experiment->state = PACE_STATE_COMPLETE;
            output->session_complete = true;
            return true;
        }
        experiment->stage_start_us += stage->duration_ms * 1000U;
        experiment->stage_index++;
        experiment->config_seq++;
        experiment->tick_in_stage = 0U;
        experiment->stage_announcement_pending = true;
        stage = &experiment->config.stages[experiment->stage_index];
        if (stage->stage_id == PACE_STAGE_STOP)
        {
            experiment->state = PACE_STATE_STOPPING;
        }
    }

    output->stage_changed = experiment->stage_announcement_pending;
    experiment->stage_announcement_pending = false;
    output->stage_id = stage->stage_id;
    output->config_seq = experiment->config_seq;
    output->active_mask = pace_experiment_active_mask(stage->stage_id);
    pace_experiment_fill_commands(experiment, stage, now_us, output);
    experiment->tick_in_stage++;
    return true;
}

bool pace_experiment_request_stop(pace_experiment_t *experiment, uint32_t now_us)
{
    if ((experiment == 0) || (experiment->state != PACE_STATE_RUNNING))
    {
        return false;
    }
    experiment->stage_index = PACE_EXPERIMENT_STAGE_COUNT - 1U;
    experiment->stage_start_us = now_us;
    experiment->tick_in_stage = 0U;
    experiment->config_seq++;
    experiment->stage_announcement_pending = true;
    experiment->state = PACE_STATE_STOPPING;
    return true;
}

void pace_experiment_abort(pace_experiment_t *experiment)
{
    if (experiment != 0)
    {
        experiment->state = PACE_STATE_FAULT;
    }
}

void pace_experiment_reset(pace_experiment_t *experiment)
{
    if (experiment != 0)
    {
        experiment->state = PACE_STATE_SAFE_IDLE;
        experiment->stage_index = 0U;
        experiment->config_seq = 0U;
        experiment->tick_in_stage = 0U;
        experiment->stage_announcement_pending = false;
    }
}

bool pace_experiment_fill_stage_frame(const pace_experiment_t *experiment,
                                      pace_stage_config_frame_data_t *frame_data)
{
    pace_encoded_command_t encoded;
    const pace_stage_definition_t *stage;
    uint8_t index;
    if ((experiment == 0) || (frame_data == 0) ||
        (experiment->stage_index >= PACE_EXPERIMENT_STAGE_COUNT))
    {
        return false;
    }
    stage = &experiment->config.stages[experiment->stage_index];
    memset(frame_data, 0, sizeof(*frame_data));
    frame_data->config_seq = experiment->config_seq;
    frame_data->stage_id = (uint8_t)stage->stage_id;
    memcpy(frame_data->command_mode, stage->command_mode, sizeof(frame_data->command_mode));
    memcpy(frame_data->command_rate_hz, stage->command_rate_hz, sizeof(frame_data->command_rate_hz));
    if (!pace_dm_pack_mit(PACE_MOTOR_L_FRONT, 0.0f, 0.0f,
                          experiment->config.dm_kp, experiment->config.dm_kd,
                          0.0f, &encoded))
    {
        return false;
    }
    for (index = 0U; index < PACE_DM_MOTOR_COUNT; ++index)
    {
        frame_data->dm_kp_raw[index] = encoded.raw.dm.kp;
        frame_data->dm_kd_raw[index] = encoded.raw.dm.kd;
    }
    frame_data->excitation_type = (uint8_t)stage->excitation_type;
    frame_data->amplitude = stage->amplitude;
    frame_data->frequency_start_hz = stage->frequency_start_hz;
    frame_data->frequency_end_hz = stage->frequency_end_hz;
    frame_data->duration_ms = stage->duration_ms;
    frame_data->safety_limit_set_id = experiment->config.safety_limit_set_id;
    frame_data->experiment_config_hash = experiment->config.experiment_config_hash;
    return true;
}
