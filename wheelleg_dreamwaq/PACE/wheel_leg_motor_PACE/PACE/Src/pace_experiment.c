#include "pace_experiment.h"

#include <math.h>
#include <string.h>

#include "dm_motor.h"
#include "pace_excitation.h"
#include "pace_experiment_config.h"
#include "pace_motor_manifest.h"

#define PACE_CAN_BITS_PER_COMMAND_FEEDBACK_PAIR 260U
#define PACE_HASH_RECIPE_VERSION 2U
#define PACE_TIME_HALF_RANGE_US 0x80000000UL

/* Excitation recipes are separate from the unchanged hardware axis signs.
 * Order here is jIJ, jIO, jAG, jAB, left wheel, right wheel. */
static const float pace_leg_groups[4][PACE_MOTOR_COUNT] = {
    {1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, -1.0f, 1.0f, 0.0f, 0.0f},
    {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f},
    {1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 0.0f}};

static pace_stage_definition_t *pace_recipe_append(
    pace_experiment_config_t *config, pace_stage_id_t id,
    pace_experiment_role_t role, uint32_t duration_ms,
    pace_command_mode_t wheel_mode, float amplitude,
    float frequency_start_hz, float frequency_end_hz, float phase_rad,
    const float *multipliers)
{
    pace_stage_definition_t *stage;
    uint8_t motor;
    if (config->stage_count >= PACE_EXPERIMENT_MAX_STAGES)
    {
        return 0;
    }
    stage = &config->stages[config->stage_count++];
    stage->stage_id = id;
    stage->segment_id = config->stage_count;
    stage->role = role;
    stage->duration_ms = duration_ms;
    stage->amplitude = amplitude;
    stage->frequency_start_hz = frequency_start_hz;
    stage->frequency_end_hz = frequency_end_hz;
    stage->phase_rad = phase_rad;
    stage->excitation_type = (role == PACE_ROLE_VALIDATION) ?
        PACE_EXCITATION_VALIDATION : ((amplitude > 0.0f) ?
        PACE_EXCITATION_CHIRP : PACE_EXCITATION_NONE);
    stage->action_update_rate_hz = PACE_ACTION_UPDATE_RATE_HZ;
    stage->command_hold_method = PACE_HOLD_ZERO_ORDER;
    stage->center_source = PACE_CENTER_INITIAL_FEEDBACK;
    for (motor = 0U; motor < PACE_MOTOR_COUNT; ++motor)
    {
        stage->command_mode[motor] = (uint8_t)((motor < PACE_DM_MOTOR_COUNT) ?
            PACE_COMMAND_DM_MIT : wheel_mode);
        stage->excitation_multiplier[motor] = (multipliers != 0) ? multipliers[motor] : 0.0f;
        if (stage->excitation_multiplier[motor] != 0.0f)
        {
            stage->group_mask |= (uint8_t)(1U << motor);
        }
        stage->command_rate_hz[motor] = (stage->excitation_multiplier[motor] != 0.0f) ? 500U : 100U;
    }
    return stage;
}

static void pace_recipe_build(pace_experiment_config_t *config,
                              float leg_amplitude, float leg_bandwidth,
                              float torque_amplitude, float velocity_amplitude,
                              float wheel_bandwidth)
{
    uint8_t group;
    uint8_t level;
    uint8_t mode;
    uint8_t wheel;
    float multipliers[PACE_MOTOR_COUNT];
    pace_recipe_append(config, PACE_STAGE_STATIC, PACE_ROLE_STATIC,
        PACE_STAGE_STATIC_DURATION_MS, PACE_COMMAND_LK_TORQUE, 0.0f, 0.0f, 0.0f, 0.0f, 0);
    for (group = 0U; group < 4U; ++group)
    {
        for (level = 0U; level < 2U; ++level)
        {
            pace_recipe_append(config, PACE_STAGE_DM_FIT, PACE_ROLE_FIT,
                PACE_STAGE_DM_FIT_DURATION_MS, PACE_COMMAND_LK_TORQUE,
                leg_amplitude * ((level == 0U) ? 0.5f : 1.0f),
                PACE_DM_FIT_FREQUENCY_START_HZ, leg_bandwidth,
                0.0f, pace_leg_groups[group]);
        }
    }
    for (group = 0U; group < 4U; ++group)
    {
        pace_recipe_append(config, PACE_STAGE_DM_VALIDATION, PACE_ROLE_VALIDATION,
            PACE_STAGE_DM_VALIDATION_DURATION_MS, PACE_COMMAND_LK_TORQUE,
            leg_amplitude * 0.8f, PACE_DM_VALIDATION_FREQUENCY_START_HZ,
            leg_bandwidth * 1.2f, 0.47f + (float)group * 0.23f, pace_leg_groups[group]);
    }
    for (mode = 0U; mode < 2U; ++mode)
    {
        const pace_stage_id_t id = (mode == 0U) ? PACE_STAGE_LK_TORQUE : PACE_STAGE_LK_VELOCITY;
        const pace_command_mode_t command = (mode == 0U) ? PACE_COMMAND_LK_TORQUE : PACE_COMMAND_LK_VELOCITY;
        const float amplitude = (mode == 0U) ? torque_amplitude : velocity_amplitude;
        const float start_hz = (mode == 0U) ? PACE_LK_TORQUE_FREQUENCY_START_HZ : PACE_LK_VELOCITY_FREQUENCY_START_HZ;
        const float end_hz = (wheel_bandwidth > 0.0f) ? wheel_bandwidth :
            ((mode == 0U) ? PACE_LK_TORQUE_FREQUENCY_END_HZ : PACE_LK_VELOCITY_FREQUENCY_END_HZ);
        const uint32_t fit_ms = (mode == 0U) ? PACE_STAGE_LK_TORQUE_DURATION_MS : PACE_STAGE_LK_VELOCITY_DURATION_MS;
        for (wheel = 0U; wheel < PACE_LK_MOTOR_COUNT; ++wheel)
        {
            memset(multipliers, 0, sizeof(multipliers));
            multipliers[PACE_DM_MOTOR_COUNT + wheel] = (wheel == 0U) ? 1.0f : -1.0f;
            for (level = 0U; level < 2U; ++level)
            {
                pace_recipe_append(config, id, PACE_ROLE_FIT, fit_ms, command,
                    amplitude * ((level == 0U) ? 0.5f : 1.0f), start_hz, end_hz, 0.0f, multipliers);
            }
        }
        for (wheel = 0U; wheel < PACE_LK_MOTOR_COUNT; ++wheel)
        {
            memset(multipliers, 0, sizeof(multipliers));
            multipliers[PACE_DM_MOTOR_COUNT + wheel] = (wheel == 0U) ? 1.0f : -1.0f;
            pace_recipe_append(config, id, PACE_ROLE_VALIDATION,
                PACE_STAGE_DM_VALIDATION_DURATION_MS, command, amplitude * 0.75f,
                start_hz * 1.5f, end_hz * 1.1f, 0.71f + (float)wheel * 0.31f, multipliers);
        }
    }
    pace_recipe_append(config, PACE_STAGE_STOP, PACE_ROLE_STOP,
        PACE_STAGE_STOP_DURATION_MS, PACE_COMMAND_LK_TORQUE, 0.0f, 0.0f, 0.0f, 0.0f, 0);
}

void pace_experiment_default_config(pace_experiment_config_t *config,
                                    pace_session_type_t session_type)
{
    if (config == 0)
    {
        return;
    }
    memset(config, 0, sizeof(*config));
    /* A caller cannot obtain a final recipe by relabeling commissioning. */
    if (session_type != PACE_SESSION_COMMISSIONING)
    {
        return;
    }
    config->session_type = PACE_SESSION_COMMISSIONING;
    config->safety_limit_set_id = PACE_DEFAULT_SAFETY_LIMIT_SET_ID;
    config->dm_kp = PACE_DM_BASELINE_KP;
    config->dm_kd = PACE_DM_BASELINE_KD;
    pace_recipe_build(config, PACE_DM_FIT_AMPLITUDE_RAD, PACE_DM_FIT_FREQUENCY_END_HZ,
        PACE_LK_TORQUE_AMPLITUDE_NM, PACE_LK_VELOCITY_AMPLITUDE_RAD_S, 0.0f);
    config->experiment_config_hash = pace_experiment_config_hash(config);
}

bool pace_experiment_final_config(pace_experiment_config_t *config)
{
    static const float centers[PACE_DM_MOTOR_COUNT] = {
        PACE_FINAL_DM_L_FRONT_NOMINAL_RAD, PACE_FINAL_DM_L_REAR_NOMINAL_RAD,
        PACE_FINAL_DM_R_REAR_NOMINAL_RAD, PACE_FINAL_DM_R_FRONT_NOMINAL_RAD};
    uint8_t stage;
    if (config == 0)
    {
        return false;
    }
    memset(config, 0, sizeof(*config));
    if ((PACE_FINAL_CONTRACT_READY != 1U) || (PACE_FINAL_ZERO_VERIFIED != 1U) ||
        (PACE_FINAL_CONTRACT_VERSION == 0U) || (PACE_FINAL_CONTRACT_HASH == 0U) ||
        (PACE_FINAL_ACTION_RATE_HZ == 0U) ||
        (PACE_FINAL_COMMAND_HOLD_METHOD != (uint32_t)PACE_HOLD_ZERO_ORDER) ||
        (PACE_FINAL_DM_TORQUE_FEEDFORWARD_POLICY != 0U) ||
        (PACE_FINAL_LK_COMMAND_MODE != (uint32_t)PACE_COMMAND_LK_VELOCITY) ||
        (PACE_FINAL_SAFETY_LIMIT_SET_ID != PACE_DEFAULT_SAFETY_LIMIT_SET_ID) ||
        !(PACE_FINAL_DM_KP > 0.0f) || !(PACE_FINAL_DM_KD > 0.0f) ||
        !(PACE_FINAL_DM_POSITION_AMPLITUDE_RAD > 0.0f) ||
        !(PACE_FINAL_DM_BANDWIDTH_HZ > 0.0f) ||
        !(PACE_FINAL_LK_TORQUE_AMPLITUDE_NM > 0.0f) ||
        !(PACE_FINAL_LK_VELOCITY_AMPLITUDE_RAD_S > 0.0f) ||
        !(PACE_FINAL_LK_BANDWIDTH_HZ > 0.0f))
    {
        return false;
    }
    config->session_type = PACE_SESSION_FINAL_IDENTIFICATION;
    config->contract_version = PACE_FINAL_CONTRACT_VERSION;
    config->safety_limit_set_id = PACE_FINAL_SAFETY_LIMIT_SET_ID;
    config->dm_kp = PACE_FINAL_DM_KP;
    config->dm_kd = PACE_FINAL_DM_KD;
    pace_recipe_build(config, PACE_FINAL_DM_POSITION_AMPLITUDE_RAD, PACE_FINAL_DM_BANDWIDTH_HZ,
        PACE_FINAL_LK_TORQUE_AMPLITUDE_NM, PACE_FINAL_LK_VELOCITY_AMPLITUDE_RAD_S,
        PACE_FINAL_LK_BANDWIDTH_HZ);
    for (stage = 0U; stage < config->stage_count; ++stage)
    {
        config->stages[stage].action_update_rate_hz = PACE_FINAL_ACTION_RATE_HZ;
        config->stages[stage].command_hold_method = (pace_command_hold_method_t)PACE_FINAL_COMMAND_HOLD_METHOD;
        config->stages[stage].center_source = PACE_CENTER_VERIFIED_CONTRACT;
        memcpy(config->stages[stage].dm_center_rad, centers, sizeof(centers));
    }
    config->experiment_config_hash = pace_experiment_config_hash(config);
    if ((config->experiment_config_hash != PACE_FINAL_CONTRACT_HASH) ||
        !pace_experiment_config_is_valid(config))
    {
        memset(config, 0, sizeof(*config));
        return false;
    }
    return true;
}

static bool pace_stage_frame_from_definition(const pace_experiment_config_t *config,
                                             const pace_stage_definition_t *stage,
                                             uint8_t config_seq,
                                             const float *centers,
                                             pace_stage_config_frame_data_t *data)
{
    pace_encoded_command_t encoded;
    uint8_t motor;
    memset(data, 0, sizeof(*data));
    if (!pace_dm_pack_mit(PACE_MOTOR_L_FRONT, 0.0f, 0.0f,
        config->dm_kp, config->dm_kd, 0.0f, &encoded) || encoded.saturated)
    {
        return false;
    }
    data->config_seq = config_seq;
    data->stage_id = (uint8_t)stage->stage_id;
    memcpy(data->command_mode, stage->command_mode, sizeof(data->command_mode));
    memcpy(data->command_rate_hz, stage->command_rate_hz, sizeof(data->command_rate_hz));
    for (motor = 0U; motor < PACE_DM_MOTOR_COUNT; ++motor)
    {
        data->dm_kp_raw[motor] = encoded.raw.dm.kp;
        data->dm_kd_raw[motor] = encoded.raw.dm.kd;
    }
    data->excitation_type = (uint8_t)stage->excitation_type;
    data->amplitude = stage->amplitude;
    data->frequency_start_hz = stage->frequency_start_hz;
    data->frequency_end_hz = stage->frequency_end_hz;
    data->duration_ms = stage->duration_ms;
    data->safety_limit_set_id = config->safety_limit_set_id;
    data->experiment_config_hash = config->experiment_config_hash;
    data->role = (uint8_t)stage->role;
    data->group_mask = stage->group_mask;
    data->segment_id = stage->segment_id;
    data->action_update_rate_hz = stage->action_update_rate_hz;
    data->command_hold_method = (uint8_t)stage->command_hold_method;
    data->center_source = (uint8_t)stage->center_source;
    data->phase_rad = stage->phase_rad;
    memcpy(data->dm_center_rad, centers, sizeof(data->dm_center_rad));
    memcpy(data->excitation_multiplier, stage->excitation_multiplier, sizeof(data->excitation_multiplier));
    return true;
}

static uint32_t pace_hash_bytes(uint32_t crc, const uint8_t *data, size_t length)
{
    size_t offset;
    uint8_t bit;
    for (offset = 0U; offset < length; ++offset)
    {
        crc ^= data[offset];
        for (bit = 0U; bit < 8U; ++bit)
        {
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320UL : 0U);
        }
    }
    return crc;
}

static uint32_t pace_hash_u32(uint32_t crc, uint32_t value)
{
    const uint8_t bytes[4] = {(uint8_t)value, (uint8_t)(value >> 8),
        (uint8_t)(value >> 16), (uint8_t)(value >> 24)};
    return pace_hash_bytes(crc, bytes, sizeof(bytes));
}

uint32_t pace_experiment_config_hash(const pace_experiment_config_t *config)
{
    uint32_t crc = 0xFFFFFFFFUL;
    uint32_t raw;
    uint8_t stage;
    uint8_t wire[PACE_STAGE_CONFIG_FRAME_SIZE];
    pace_stage_config_frame_data_t data;
    if ((config == 0) || (config->stage_count == 0U) ||
        (config->stage_count > PACE_EXPERIMENT_MAX_STAGES))
    {
        return 0U;
    }
    crc = pace_hash_u32(crc, PACE_HASH_RECIPE_VERSION);
    crc = pace_hash_u32(crc, (uint32_t)config->session_type);
    crc = pace_hash_u32(crc, config->contract_version);
    crc = pace_hash_u32(crc, config->stage_count);
    crc = pace_hash_u32(crc, PACE_CONTROL_RATE_HZ);
    crc = pace_hash_u32(crc, PACE_SAMPLE_RATE_HZ);
    crc = pace_hash_u32(crc, PACE_STATUS_RATE_HZ);
    crc = pace_hash_u32(crc, PACE_CAN_NOMINAL_BITRATE);
    crc = pace_hash_u32(crc, PACE_UART_BAUD);
    crc = pace_hash_u32(crc, PACE_MOTOR_MANIFEST_HASH);
    /* Bind the actual axis/ID/zero contents as well as their declared tag. */
    for (stage = 0U; stage < PACE_MOTOR_COUNT; ++stage)
    {
        const pace_motor_manifest_entry_t *entry = &pace_motor_manifest[stage];
        crc = pace_hash_u32(crc, entry->canonical_index);
        crc = pace_hash_u32(crc, (uint32_t)entry->motor_type);
        crc = pace_hash_u32(crc, entry->hardware_slot);
        crc = pace_hash_u32(crc, entry->device_id);
        crc = pace_hash_u32(crc, entry->command_can_id);
        crc = pace_hash_u32(crc, entry->feedback_can_id);
        crc = pace_hash_u32(crc, (uint32_t)(int32_t)entry->position_sign);
        crc = pace_hash_u32(crc, (uint32_t)(int32_t)entry->velocity_sign);
        crc = pace_hash_u32(crc, (uint32_t)(int32_t)entry->torque_sign);
        memcpy(&raw, &entry->position_zero_rad, sizeof(raw));
        crc = pace_hash_u32(crc, raw);
        memcpy(&raw, &entry->gear_ratio, sizeof(raw));
        crc = pace_hash_u32(crc, raw);
        crc = pace_hash_u32(crc, (uint32_t)entry->actuator_family);
        crc = pace_hash_u32(crc, entry->calibration_verified ? 1U : 0U);
        crc = pace_hash_u32(crc, (uint32_t)strlen(entry->mujoco_joint));
        crc = pace_hash_bytes(crc, (const uint8_t *)entry->mujoco_joint, strlen(entry->mujoco_joint));
        crc = pace_hash_u32(crc, (uint32_t)strlen(entry->isaac_joint));
        crc = pace_hash_bytes(crc, (const uint8_t *)entry->isaac_joint, strlen(entry->isaac_joint));
    }
    memcpy(&raw, &config->dm_kp, sizeof(raw));
    crc = pace_hash_u32(crc, raw);
    memcpy(&raw, &config->dm_kd, sizeof(raw));
    crc = pace_hash_u32(crc, raw);
    crc = pace_hash_u32(crc, config->safety_limit_set_id);
    for (stage = 0U; stage < config->stage_count; ++stage)
    {
        if (!pace_stage_frame_from_definition(config, &config->stages[stage],
            (uint8_t)(stage + 1U), config->stages[stage].dm_center_rad, &data))
        {
            return 0U;
        }
        data.experiment_config_hash = 0U;
        if (pace_frame_encode_stage_config(wire, sizeof(wire), 0U, &data) != sizeof(wire))
        {
            return 0U;
        }
        /* Only recipe payload: no common header, CRC, or self-referential hash.
         * Configured centers are zeros for INITIAL_FEEDBACK; runtime centers
         * belong to each recorded stage and the immutable raw content. */
        crc = pace_hash_bytes(crc, &wire[10], 114U);
    }
    return crc ^ 0xFFFFFFFFUL;
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
    uint8_t stage_index;
    uint8_t motor;
    uint64_t total_us = 0U;
    if ((config == 0) || (config->stage_count < 2U) ||
        (config->stage_count > PACE_EXPERIMENT_MAX_STAGES) ||
        ((config->session_type != PACE_SESSION_COMMISSIONING) &&
         (config->session_type != PACE_SESSION_FINAL_IDENTIFICATION)) ||
        !isfinite(config->dm_kp) || !isfinite(config->dm_kd) ||
        !(config->dm_kp > 0.0f) || !(config->dm_kd > 0.0f) ||
        (config->safety_limit_set_id == 0U) ||
        (config->stages[0].stage_id != PACE_STAGE_STATIC) ||
        (config->stages[config->stage_count - 1U].stage_id != PACE_STAGE_STOP))
    {
        return false;
    }
    if ((config->session_type == PACE_SESSION_COMMISSIONING) &&
        ((config->dm_kp != PACE_DM_BASELINE_KP) || (config->dm_kd != PACE_DM_BASELINE_KD)))
    {
        return false;
    }
    if ((config->session_type == PACE_SESSION_FINAL_IDENTIFICATION) &&
        ((PACE_FINAL_CONTRACT_READY != 1U) || (PACE_FINAL_ZERO_VERIFIED != 1U) ||
         (config->contract_version != PACE_FINAL_CONTRACT_VERSION) ||
         (config->safety_limit_set_id != PACE_DEFAULT_SAFETY_LIMIT_SET_ID) ||
         (config->experiment_config_hash != PACE_FINAL_CONTRACT_HASH)))
    {
        return false;
    }
    for (stage_index = 0U; stage_index < config->stage_count; ++stage_index)
    {
        const pace_stage_definition_t *stage = &config->stages[stage_index];
        uint8_t actual_mask = 0U;
        if ((stage->stage_id < PACE_STAGE_STATIC) || (stage->stage_id > PACE_STAGE_STOP) ||
            (stage->segment_id != (uint16_t)(stage_index + 1U)) ||
            (stage->duration_ms == 0U) ||
            (stage->duration_ms >= (PACE_TIME_HALF_RANGE_US / 1000U)) ||
            (pace_experiment_stage_bus_load_percent(stage) > 70.0f) ||
            (stage->role > PACE_ROLE_STOP) ||
            !isfinite(stage->amplitude) || (stage->amplitude < 0.0f) ||
            !isfinite(stage->frequency_start_hz) || !isfinite(stage->frequency_end_hz) ||
            !isfinite(stage->phase_rad) ||
            (stage->frequency_start_hz < 0.0f) || (stage->frequency_end_hz < 0.0f) ||
            (stage->action_update_rate_hz == 0U) ||
            (stage->action_update_rate_hz > PACE_CONTROL_RATE_HZ) ||
            ((PACE_CONTROL_RATE_HZ % stage->action_update_rate_hz) != 0U) ||
            (stage->command_hold_method != PACE_HOLD_ZERO_ORDER) ||
            ((stage->center_source != PACE_CENTER_INITIAL_FEEDBACK) &&
             (stage->center_source != PACE_CENTER_VERIFIED_CONTRACT)) ||
            (stage->excitation_type > PACE_EXCITATION_VALIDATION) ||
            (((stage->role == PACE_ROLE_FIT) || (stage->role == PACE_ROLE_VALIDATION)) &&
             ((stage->amplitude <= 0.0f) || (stage->frequency_start_hz <= 0.0f) ||
              (stage->frequency_end_hz <= 0.0f) || (stage->excitation_type == PACE_EXCITATION_NONE))))
        {
            return false;
        }
        total_us += (uint64_t)stage->duration_ms * 1000U;
        for (motor = 0U; motor < PACE_MOTOR_COUNT; ++motor)
        {
            const uint16_t rate = stage->command_rate_hz[motor];
            const float multiplier = stage->excitation_multiplier[motor];
            if ((rate == 0U) || (rate > PACE_CONTROL_RATE_HZ) ||
                ((PACE_CONTROL_RATE_HZ % rate) != 0U) ||
                !isfinite(multiplier) || (fabsf(multiplier) > 1.0f))
            {
                return false;
            }
            if (multiplier != 0.0f)
            {
                actual_mask |= (uint8_t)(1U << motor);
            }
            if (motor < PACE_DM_MOTOR_COUNT)
            {
                if ((stage->command_mode[motor] != (uint8_t)PACE_COMMAND_DM_MIT) ||
                    !isfinite(stage->dm_center_rad[motor]) ||
                    (fabsf(stage->dm_center_rad[motor]) + stage->amplitude * fabsf(multiplier) > 12.5f))
                {
                    return false;
                }
            }
            else if (((stage->command_mode[motor] != (uint8_t)PACE_COMMAND_LK_TORQUE) &&
                      (stage->command_mode[motor] != (uint8_t)PACE_COMMAND_LK_VELOCITY)) ||
                     ((stage->stage_id == PACE_STAGE_LK_TORQUE) &&
                      (stage->command_mode[motor] != (uint8_t)PACE_COMMAND_LK_TORQUE)) ||
                     ((stage->stage_id == PACE_STAGE_LK_VELOCITY) &&
                      (stage->command_mode[motor] != (uint8_t)PACE_COMMAND_LK_VELOCITY)))
            {
                return false;
            }
        }
        if (((stage->stage_id == PACE_STAGE_STATIC) && (stage->role != PACE_ROLE_STATIC)) ||
            ((stage->stage_id == PACE_STAGE_STOP) && (stage->role != PACE_ROLE_STOP)) ||
            ((stage->stage_id == PACE_STAGE_DM_FIT) && (stage->role != PACE_ROLE_FIT)) ||
            ((stage->stage_id == PACE_STAGE_DM_VALIDATION) && (stage->role != PACE_ROLE_VALIDATION)) ||
            (((stage->stage_id == PACE_STAGE_LK_TORQUE) || (stage->stage_id == PACE_STAGE_LK_VELOCITY)) &&
             (stage->role != PACE_ROLE_FIT) && (stage->role != PACE_ROLE_VALIDATION) &&
             (stage->role != PACE_ROLE_DIAGNOSTIC)) ||
            ((config->session_type == PACE_SESSION_COMMISSIONING) &&
             (stage->center_source != PACE_CENTER_INITIAL_FEEDBACK)) ||
            ((config->session_type == PACE_SESSION_FINAL_IDENTIFICATION) &&
             (stage->center_source != PACE_CENTER_VERIFIED_CONTRACT)))
        {
            return false;
        }
        if ((config->session_type == PACE_SESSION_COMMISSIONING) &&
            ((((stage->stage_id == PACE_STAGE_DM_FIT) || (stage->stage_id == PACE_STAGE_DM_VALIDATION)) &&
              (stage->amplitude > PACE_DM_FIT_AMPLITUDE_RAD)) ||
             ((stage->stage_id == PACE_STAGE_LK_TORQUE) && (stage->amplitude > PACE_LK_TORQUE_AMPLITUDE_NM)) ||
             ((stage->stage_id == PACE_STAGE_LK_VELOCITY) && (stage->amplitude > PACE_LK_VELOCITY_AMPLITUDE_RAD_S))))
        {
            return false;
        }
        for (motor = 0U; motor < PACE_DM_MOTOR_COUNT; ++motor)
        {
            if ((stage->center_source == PACE_CENTER_INITIAL_FEEDBACK) &&
                (stage->dm_center_rad[motor] != 0.0f))
            {
                return false;
            }
        }
        if ((actual_mask != stage->group_mask) ||
            (((stage->stage_id == PACE_STAGE_STATIC) || (stage->stage_id == PACE_STAGE_STOP)) &&
             ((actual_mask != 0U) || (stage->amplitude != 0.0f) || (stage->excitation_type != PACE_EXCITATION_NONE))) ||
            (((stage->stage_id == PACE_STAGE_DM_FIT) || (stage->stage_id == PACE_STAGE_DM_VALIDATION)) &&
             ((actual_mask == 0U) || ((actual_mask & 0xF0U) != 0U))) ||
            (((stage->stage_id == PACE_STAGE_LK_TORQUE) || (stage->stage_id == PACE_STAGE_LK_VELOCITY)) &&
             ((actual_mask == 0U) || ((actual_mask & 0xCFU) != 0U))))
        {
            return false;
        }
    }
    return (total_us < PACE_TIME_HALF_RANGE_US) &&
        (config->experiment_config_hash != 0U) &&
        (config->experiment_config_hash == pace_experiment_config_hash(config));
}

void pace_experiment_init(pace_experiment_t *experiment)
{
    if (experiment != 0)
    {
        memset(experiment, 0, sizeof(*experiment));
        experiment->state = PACE_STATE_SAFE_IDLE;
    }
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

static void pace_experiment_reset_stage_clock(pace_experiment_t *experiment)
{
    experiment->tick_in_stage = 0U;
    experiment->command_cycle_valid_mask = 0U;
    experiment->target_cycle_valid = false;
    experiment->stage_announcement_pending = true;
}

bool pace_experiment_begin(pace_experiment_t *experiment,
                           const pace_motor_registry_t *registry,
                           uint32_t now_us)
{
    uint8_t motor;
    uint8_t stage;
    if ((experiment == 0) || (registry == 0) || (experiment->state != PACE_STATE_ARMED))
    {
        return false;
    }
    for (motor = 0U; motor < PACE_MOTOR_COUNT; ++motor)
    {
        const pace_motor_state_t *state = &registry->motors[motor];
        const uint32_t age = now_us - state->last_rx_timestamp_us;
        if (!state->rx_valid || !state->online ||
            (age >= PACE_TIME_HALF_RANGE_US) || (age > PACE_INITIAL_FEEDBACK_MAX_AGE_US))
        {
            return false;
        }
        if (motor < PACE_DM_MOTOR_COUNT)
        {
            const float center = state->feedback.dm.position_rad;
            if (!isfinite(center) || (fabsf(center) > 12.5f))
            {
                return false;
            }
            for (stage = 0U; stage < experiment->config.stage_count; ++stage)
            {
                const pace_stage_definition_t *definition = &experiment->config.stages[stage];
                if ((definition->center_source == PACE_CENTER_INITIAL_FEEDBACK) &&
                    (fabsf(center) + definition->amplitude * fabsf(definition->excitation_multiplier[motor]) > 12.5f))
                {
                    return false;
                }
            }
            experiment->dm_center_rad[motor] = center;
        }
    }
    experiment->stage_index = 0U;
    experiment->config_seq = 1U;
    experiment->session_start_us = now_us;
    experiment->stage_start_us = now_us;
    experiment->target_update_sequence = 0U;
    pace_experiment_reset_stage_clock(experiment);
    experiment->state = PACE_STATE_RUNNING;
    return true;
}

static void pace_experiment_fill_commands(pace_experiment_t *experiment,
                                          const pace_stage_definition_t *stage,
                                          uint32_t now_us,
                                          pace_experiment_output_t *output)
{
    const uint32_t elapsed_us = now_us - experiment->stage_start_us;
    const uint32_t update_us = 1000000U / stage->action_update_rate_hz;
    const uint32_t target_cycle = elapsed_us / update_us;
    const uint32_t held_elapsed_us = target_cycle * update_us;
    const float time_s = (float)held_elapsed_us * 0.000001f;
    const float duration_s = (float)stage->duration_ms * 0.001f;
    float excitation = 0.0f;
    uint8_t index;
    output->target_updated = !experiment->target_cycle_valid ||
        (experiment->last_target_cycle != target_cycle);
    if (output->target_updated)
    {
        experiment->target_update_sequence++;
        experiment->last_target_cycle = target_cycle;
        experiment->target_cycle_valid = true;
    }
    output->target_update_sequence = experiment->target_update_sequence;
    /* This is the recipe update grid, not a measured motor execution time. */
    output->target_update_time_us = experiment->stage_start_us + held_elapsed_us;
    if (stage->excitation_type == PACE_EXCITATION_CHIRP)
    {
        excitation = pace_excitation_chirp(time_s, duration_s, stage->amplitude,
            stage->frequency_start_hz, stage->frequency_end_hz, stage->phase_rad);
    }
    else if (stage->excitation_type == PACE_EXCITATION_VALIDATION)
    {
        excitation = pace_excitation_validation(time_s, duration_s, stage->amplitude,
            stage->frequency_start_hz, stage->frequency_end_hz, stage->phase_rad);
    }
    for (index = 0U; index < PACE_MOTOR_COUNT; ++index)
    {
        const uint32_t command_cycle = elapsed_us / (1000000U / stage->command_rate_hz[index]);
        const uint8_t bit = (uint8_t)(1U << index);
        output->command_mode[index] = (pace_command_mode_t)stage->command_mode[index];
        if (((experiment->command_cycle_valid_mask & bit) == 0U) ||
            (experiment->last_command_cycle[index] != command_cycle))
        {
            output->send_mask |= bit;
            experiment->last_command_cycle[index] = command_cycle;
            experiment->command_cycle_valid_mask |= bit;
        }
    }
    for (index = 0U; index < PACE_DM_MOTOR_COUNT; ++index)
    {
        const float center = (stage->center_source == PACE_CENTER_INITIAL_FEEDBACK) ?
            experiment->dm_center_rad[index] : stage->dm_center_rad[index];
        output->dm_position_rad[index] = center + stage->excitation_multiplier[index] * excitation;
        output->dm_velocity_rad_s[index] = 0.0f;
        output->dm_kp[index] = experiment->config.dm_kp;
        output->dm_kd[index] = experiment->config.dm_kd;
        output->dm_torque_nm[index] = 0.0f;
    }
    for (index = 0U; index < PACE_LK_MOTOR_COUNT; ++index)
    {
        output->lk_primary[index] = stage->excitation_multiplier[PACE_DM_MOTOR_COUNT + index] * excitation;
    }
}

bool pace_experiment_step(pace_experiment_t *experiment,
                          uint32_t now_us,
                          pace_experiment_output_t *output)
{
    const pace_stage_definition_t *stage;
    if ((experiment == 0) || (output == 0) ||
        ((experiment->state != PACE_STATE_RUNNING) && (experiment->state != PACE_STATE_STOPPING)))
    {
        return false;
    }
    memset(output, 0, sizeof(*output));
    /* Valid sessions last less than half the uint32 time range. A historical
     * queued tick preceding a host STOP must never skip the stop duration. */
    if ((now_us - experiment->stage_start_us) >= PACE_TIME_HALF_RANGE_US)
    {
        return false;
    }
    stage = &experiment->config.stages[experiment->stage_index];
    while ((now_us - experiment->stage_start_us) >= (stage->duration_ms * 1000U))
    {
        if (experiment->stage_index >= (experiment->config.stage_count - 1U))
        {
            experiment->state = PACE_STATE_COMPLETE;
            output->session_complete = true;
            return true;
        }
        experiment->stage_start_us += stage->duration_ms * 1000U;
        experiment->stage_index++;
        experiment->config_seq++;
        pace_experiment_reset_stage_clock(experiment);
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
    output->active_mask = stage->group_mask;
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
    experiment->stage_index = experiment->config.stage_count - 1U;
    experiment->stage_start_us = now_us;
    experiment->config_seq++;
    pace_experiment_reset_stage_clock(experiment);
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
        experiment->target_update_sequence = 0U;
        pace_experiment_reset_stage_clock(experiment);
        experiment->stage_announcement_pending = false;
    }
}

bool pace_experiment_fill_stage_frame(const pace_experiment_t *experiment,
                                      pace_stage_config_frame_data_t *frame_data)
{
    const pace_stage_definition_t *stage;
    const float *centers;
    if ((experiment == 0) || (frame_data == 0) ||
        (experiment->stage_index >= experiment->config.stage_count))
    {
        return false;
    }
    stage = &experiment->config.stages[experiment->stage_index];
    centers = (stage->center_source == PACE_CENTER_INITIAL_FEEDBACK) ?
        experiment->dm_center_rad : stage->dm_center_rad;
    return pace_stage_frame_from_definition(&experiment->config, stage,
        experiment->config_seq, centers, frame_data);
}
