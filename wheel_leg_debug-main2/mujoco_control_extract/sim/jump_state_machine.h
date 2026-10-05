#ifndef JUMP_STATE_MACHINE_H
#define JUMP_STATE_MACHINE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    JUMP_PHASE_STAND = 0,
    JUMP_PHASE_COMPRESS = 1,
    JUMP_PHASE_THRUST = 2,
    JUMP_PHASE_TUCK = 3,
    JUMP_PHASE_PRELAND = 4,
    JUMP_PHASE_BUFFER = 5,
    JUMP_PHASE_RECOVER = 6,
} JumpPhase;

typedef struct
{
    int compression_enabled;
    float initial_leg_length;
    float min_leg_length;
    float max_leg_length;

    float compress_target;
    float compress_rate;
    float compress_support_scale;
    float compress_tolerance;
    float compress_hold_time;
    float compress_timeout;

    float extend_l0;
    float extend_end_margin;
    float extend_rate;
    float thrust_min_time;
    float thrust_timeout;
    int extend_hold_steps;

    float tuck_min_time;
    float tuck_timeout;

    float preland_clearance;
    float preland_l0;
    float preland_rate;
    float preland_pid_scale;
    float preland_timeout;

    float touchdown_force;
    float touchdown_hold;

    float buffer_l0;
    float buffer_rate;
    float buffer_support_scale;
    float buffer_pid_scale;
    float buffer_hold_time;
    float buffer_timeout;

    float recover_rate;
    float recover_time;
    float recover_stable_time;
    float recover_timeout;
    float recover_l0_tolerance;
    float recover_roll_limit;
    float recover_pitch_limit;
    float recover_pitch_rate;
    float recover_speed_limit;
    float wheel_recover_blend_time;

    float request_roll_limit;
    float request_pitch_limit;
    float request_speed_limit;
    float request_z_speed_limit;

    float landing_roll_f0_kp;
    float landing_roll_f0_kd;
    float landing_contact_f0_kp;
    float landing_balance_f0_limit;

    float landing_roll_l0_kp;
    float landing_roll_l0_kd;
    float landing_balance_l0_limit;

    float landing_clearance_l0_kp;
    float landing_clearance_l0_rate;
    float landing_clearance_l0_limit;

    float lqr_tp_weight;
    float split_tp_weight;
    float pitch_tp_weight;
    float leg_swing_tp_weight;
} JumpStateMachineConfig;

typedef struct
{
    float left_l0;
    float right_l0;
    float roll_error;
    float roll_rate;
    float pitch_error;
    float pitch_rate;
    float body_forward_v;
    float body_lateral_v;
    float body_z_vel;
    float wheel_clearance[2];
    float wheel_contact_normal[2];
    int airborne;
} JumpObservation;

typedef struct
{
    JumpPhase phase;
    int active;
    int just_finished;
    int airborne_pose_enabled;

    float compress_l0_set;
    float extend_l0_set;
    float landing_l0_set;
    float landing_support_scale;
    float landing_pid_scale;
    float landing_balance_f0;
    float landing_balance_l0;

    float lqr_tp_weight;
    float split_tp_weight;
    float pitch_tp_weight;
    float leg_swing_tp_weight;
    float wheel_balance_blend;
} JumpCommand;

typedef struct
{
    JumpStateMachineConfig config;
    JumpCommand command;

    int has_been_airborne;
    int extend_ready_steps;
    float compress_elapsed;
    float compress_hold_elapsed;
    float phase_elapsed;
    float landing_elapsed;
    float touch_elapsed;
    float recover_elapsed;
    float recover_stable_elapsed;
    float previous_min_clearance;
    int clearance_initialized;
} JumpStateMachine;

void JumpStateMachine_Init(JumpStateMachine *machine,
                           const JumpStateMachineConfig *config);
void JumpStateMachine_SetConfig(JumpStateMachine *machine,
                                const JumpStateMachineConfig *config);
int JumpStateMachine_Request(JumpStateMachine *machine,
                             const JumpObservation *observation);
void JumpStateMachine_Step(JumpStateMachine *machine,
                           const JumpObservation *observation,
                           float dt);
const JumpCommand *JumpStateMachine_GetCommand(const JumpStateMachine *machine);

#ifdef __cplusplus
}
#endif

#endif
