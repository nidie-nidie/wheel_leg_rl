#ifndef PACE_EXPERIMENT_H
#define PACE_EXPERIMENT_H

#include <stdbool.h>
#include <stdint.h>

#include "pace_frame.h"
#include "pace_motor_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PACE_EXPERIMENT_STAGE_COUNT 6U

typedef enum
{
    PACE_STATE_BOOT = 0,
    PACE_STATE_SAFE_IDLE,
    PACE_STATE_CONFIGURED,
    PACE_STATE_ARMED,
    PACE_STATE_RUNNING,
    PACE_STATE_STOPPING,
    PACE_STATE_COMPLETE,
    PACE_STATE_FAULT
} pace_experiment_state_t;

typedef enum
{
    PACE_STAGE_STATIC = 1,
    PACE_STAGE_DM_FIT = 2,
    PACE_STAGE_DM_VALIDATION = 3,
    PACE_STAGE_LK_TORQUE = 4,
    PACE_STAGE_LK_VELOCITY = 5,
    PACE_STAGE_STOP = 6
} pace_stage_id_t;

typedef enum
{
    PACE_EXCITATION_NONE = 0,
    PACE_EXCITATION_CHIRP = 1,
    PACE_EXCITATION_VALIDATION = 2
} pace_excitation_type_t;

typedef struct
{
    pace_stage_id_t stage_id;
    uint32_t duration_ms;
    uint8_t command_mode[PACE_MOTOR_COUNT];
    uint16_t command_rate_hz[PACE_MOTOR_COUNT];
    pace_excitation_type_t excitation_type;
    float amplitude;
    float frequency_start_hz;
    float frequency_end_hz;
    float phase_rad;
} pace_stage_definition_t;

typedef struct
{
    pace_session_type_t session_type;
    uint32_t experiment_config_hash;
    uint16_t safety_limit_set_id;
    float dm_kp;
    float dm_kd;
    pace_stage_definition_t stages[PACE_EXPERIMENT_STAGE_COUNT];
} pace_experiment_config_t;

typedef struct
{
    bool stage_changed;
    bool session_complete;
    pace_stage_id_t stage_id;
    uint8_t config_seq;
    uint8_t active_mask;
    uint8_t send_mask;
    pace_command_mode_t command_mode[PACE_MOTOR_COUNT];
    float dm_position_rad[PACE_DM_MOTOR_COUNT];
    float dm_velocity_rad_s[PACE_DM_MOTOR_COUNT];
    float dm_kp[PACE_DM_MOTOR_COUNT];
    float dm_kd[PACE_DM_MOTOR_COUNT];
    float dm_torque_nm[PACE_DM_MOTOR_COUNT];
    float lk_primary[PACE_LK_MOTOR_COUNT];
} pace_experiment_output_t;

typedef struct
{
    pace_experiment_state_t state;
    pace_experiment_config_t config;
    uint8_t stage_index;
    uint8_t config_seq;
    uint32_t session_start_us;
    uint32_t stage_start_us;
    uint32_t tick_in_stage;
    float dm_center_rad[PACE_DM_MOTOR_COUNT];
    bool stage_announcement_pending;
} pace_experiment_t;

void pace_experiment_default_config(pace_experiment_config_t *config,
                                    pace_session_type_t session_type);
void pace_experiment_init(pace_experiment_t *experiment);
bool pace_experiment_configure(pace_experiment_t *experiment,
                               const pace_experiment_config_t *config);
bool pace_experiment_arm(pace_experiment_t *experiment);
bool pace_experiment_begin(pace_experiment_t *experiment,
                           const pace_motor_registry_t *registry,
                           uint32_t now_us);
bool pace_experiment_step(pace_experiment_t *experiment,
                          uint32_t now_us,
                          pace_experiment_output_t *output);
bool pace_experiment_request_stop(pace_experiment_t *experiment, uint32_t now_us);
void pace_experiment_abort(pace_experiment_t *experiment);
void pace_experiment_reset(pace_experiment_t *experiment);
float pace_experiment_stage_bus_load_percent(const pace_stage_definition_t *stage);
bool pace_experiment_config_is_valid(const pace_experiment_config_t *config);
bool pace_experiment_fill_stage_frame(const pace_experiment_t *experiment,
                                      pace_stage_config_frame_data_t *frame_data);

#ifdef __cplusplus
}
#endif

#endif
