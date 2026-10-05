#ifndef STEER_STATE_MACHINE_H
#define STEER_STATE_MACHINE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    STEER_PHASE_IDLE = 0,
    STEER_PHASE_PREPARE,
    STEER_PHASE_ACTIVE,
    STEER_PHASE_BRAKE,
    STEER_PHASE_RECOVER,
    STEER_PHASE_ABORT,
} SteerPhase;

typedef enum
{
    STEER_DRIVE_STAND = 0,
    STEER_DRIVE_FORWARD,
    STEER_DRIVE_JUMP,
} SteerDriveMode;

typedef struct
{
    float turn_leg_length;
    float leg_rate;
    float yaw_rate_max;
    float yaw_accel_limit;
    float active_time_limit;
    float active_planar_error_limit;
    float input_deadband;
    float prepare_l0_tolerance;
    float brake_gyro_tolerance;
    float brake_linear_velocity_tolerance;
    float brake_hold_time;
    float recover_linear_velocity_tolerance;
    float recover_l0_tolerance;
    float abort_roll_limit;
    float abort_pitch_limit;
} SteerStateMachineConfig;

typedef struct
{
    float dt;
    float input;
    float current_yaw;
    float gyro_z;
    float left_l0;
    float right_l0;
    float body_x;
    float body_y;
    float body_v_x;
    float body_v_y;
    float base_leg_set;
    float base_yaw_hold;
    float roll;
    float pitch;
    SteerDriveMode drive_mode;
    int chassis_safe;
    int start_enabled;
    int jump_active;
} SteerObservation;

typedef struct
{
    SteerPhase phase;
    int active;
    int yaw_lock;
    int force_stand;
    int planar_lock;
    float leg_set;
    float yaw_hold;
    float yaw_rate_ref;
    float target_x;
    float target_y;
} SteerCommand;

typedef struct
{
    SteerStateMachineConfig config;
    SteerPhase phase;
    float leg_set_cmd;
    float yaw_hold_cmd;
    float yaw_rate_cmd;
    float recover_leg_set;
    float anchor_x;
    float anchor_y;
    float brake_stable_time;
    float phase_time;
    int cycle_limited;
    int session_active;
    int initialized;
} SteerStateMachine;

void SteerStateMachine_Init(SteerStateMachine *machine,
                            const SteerStateMachineConfig *config);
void SteerStateMachine_SetConfig(SteerStateMachine *machine,
                                 const SteerStateMachineConfig *config);
void SteerStateMachine_Step(SteerStateMachine *machine,
                            const SteerObservation *observation,
                            SteerCommand *command);
const char *SteerStateMachine_PhaseName(SteerPhase phase);

#ifdef __cplusplus
}
#endif

#endif
