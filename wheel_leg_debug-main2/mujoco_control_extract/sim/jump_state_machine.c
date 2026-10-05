#include "jump_state_machine.h"

#include <math.h>
#include <string.h>

static float jump_clamp(float value, float min_value, float max_value)
{
    if (value < min_value)
    {
        return min_value;
    }
    if (value > max_value)
    {
        return max_value;
    }
    return value;
}

static float jump_move_toward(float value, float target, float rate, float dt)
{
    const float step = fabsf(rate) * fmaxf(dt, 0.0f);
    return value + jump_clamp(target - value, -step, step);
}

static float jump_min_clearance(const JumpObservation *observation)
{
    return fminf(observation->wheel_clearance[0],
                 observation->wheel_clearance[1]);
}

static int jump_first_touch(const JumpStateMachine *machine,
                            const JumpObservation *observation)
{
    const int force_contact =
        machine->has_been_airborne &&
        (observation->wheel_contact_normal[0] >= machine->config.touchdown_force ||
         observation->wheel_contact_normal[1] >= machine->config.touchdown_force);
    const int geometric_contact =
        machine->has_been_airborne &&
        !observation->airborne &&
        jump_min_clearance(observation) <= 0.003f;
    return force_contact || geometric_contact;
}

static int jump_is_grounded(const JumpStateMachine *machine,
                            const JumpObservation *observation)
{
    const int force_grounded =
        observation->wheel_contact_normal[0] >= machine->config.touchdown_force &&
        observation->wheel_contact_normal[1] >= machine->config.touchdown_force;
    const int geometry_grounded =
        !observation->airborne &&
        observation->wheel_clearance[0] <= 0.003f &&
        observation->wheel_clearance[1] <= 0.003f;
    return force_grounded || geometry_grounded;
}

static int jump_stable_touch(const JumpStateMachine *machine,
                             const JumpObservation *observation)
{
    return observation->wheel_contact_normal[0] >= machine->config.touchdown_force &&
           observation->wheel_contact_normal[1] >= machine->config.touchdown_force;
}

static void jump_set_phase(JumpStateMachine *machine, JumpPhase phase)
{
    if (machine->command.phase != phase)
    {
        machine->phase_elapsed = 0.0f;
    }
    machine->command.phase = phase;
    machine->command.airborne_pose_enabled = phase == JUMP_PHASE_TUCK;

    machine->command.lqr_tp_weight =
        phase == JUMP_PHASE_STAND
            ? 1.0f
            : ((phase == JUMP_PHASE_COMPRESS ||
                phase == JUMP_PHASE_RECOVER)
                   ? machine->config.lqr_tp_weight
                   : 0.0f);
    machine->command.split_tp_weight =
        phase == JUMP_PHASE_STAND
            ? 1.0f
            : machine->config.split_tp_weight;
    machine->command.pitch_tp_weight =
        phase >= JUMP_PHASE_COMPRESS && phase <= JUMP_PHASE_RECOVER
            ? machine->config.pitch_tp_weight
            : 0.0f;
    machine->command.leg_swing_tp_weight =
        phase == JUMP_PHASE_THRUST
            ? machine->config.leg_swing_tp_weight
            : 0.0f;
    if (phase == JUMP_PHASE_STAND)
    {
        machine->command.wheel_balance_blend = 1.0f;
    }
    else if (phase != JUMP_PHASE_RECOVER)
    {
        machine->command.wheel_balance_blend = 0.0f;
    }
}

static void jump_reset_outputs(JumpStateMachine *machine)
{
    machine->command.compress_l0_set = machine->config.initial_leg_length;
    machine->command.extend_l0_set = machine->config.initial_leg_length;
    machine->command.landing_l0_set = machine->config.initial_leg_length;
    machine->command.landing_support_scale = 1.0f;
    machine->command.landing_pid_scale = 1.0f;
    machine->command.landing_balance_f0 = 0.0f;
    machine->command.landing_balance_l0 = 0.0f;
    machine->command.wheel_balance_blend = 1.0f;
}

static void jump_finish(JumpStateMachine *machine)
{
    machine->command.active = 0;
    machine->command.just_finished = 1;
    machine->has_been_airborne = 0;
    machine->extend_ready_steps = 0;
    machine->landing_elapsed = 0.0f;
    machine->touch_elapsed = 0.0f;
    machine->recover_elapsed = 0.0f;
    machine->recover_stable_elapsed = 0.0f;
    machine->clearance_initialized = 0;
    jump_reset_outputs(machine);
    jump_set_phase(machine, JUMP_PHASE_STAND);
}

static void jump_update_landing_balance(JumpStateMachine *machine,
                                        const JumpObservation *observation,
                                        float dt)
{
    const JumpPhase phase = machine->command.phase;
    if (phase < JUMP_PHASE_PRELAND || phase > JUMP_PHASE_RECOVER)
    {
        machine->command.landing_balance_f0 = 0.0f;
        machine->command.landing_balance_l0 = 0.0f;
        return;
    }

    const float roll_term =
        machine->config.landing_roll_f0_kp * observation->roll_error +
        machine->config.landing_roll_f0_kd * observation->roll_rate;
    const float contact_term =
        machine->config.landing_contact_f0_kp *
        (observation->wheel_contact_normal[0] -
         observation->wheel_contact_normal[1]);
    machine->command.landing_balance_f0 =
        jump_clamp(roll_term + contact_term,
                   -machine->config.landing_balance_f0_limit,
                   machine->config.landing_balance_f0_limit);

    float balance_target;
    if (phase == JUMP_PHASE_PRELAND)
    {
        const float clearance_error =
            observation->wheel_clearance[0] -
            observation->wheel_clearance[1];
        balance_target =
            jump_clamp(machine->config.landing_clearance_l0_kp *
                           clearance_error,
                       -machine->config.landing_clearance_l0_limit,
                       machine->config.landing_clearance_l0_limit);
    }
    else
    {
        balance_target =
            jump_clamp(-machine->config.landing_roll_l0_kp *
                           observation->roll_error -
                           machine->config.landing_roll_l0_kd *
                               observation->roll_rate,
                       -machine->config.landing_balance_l0_limit,
                       machine->config.landing_balance_l0_limit);
    }

    machine->command.landing_balance_l0 =
        jump_move_toward(machine->command.landing_balance_l0,
                         balance_target,
                         machine->config.landing_clearance_l0_rate,
                         dt);
}

void JumpStateMachine_Init(JumpStateMachine *machine,
                           const JumpStateMachineConfig *config)
{
    if (machine == 0 || config == 0)
    {
        return;
    }

    memset(machine, 0, sizeof(*machine));
    machine->config = *config;
    jump_reset_outputs(machine);
    jump_set_phase(machine, JUMP_PHASE_STAND);
}

void JumpStateMachine_SetConfig(JumpStateMachine *machine,
                                const JumpStateMachineConfig *config)
{
    if (machine == 0 || config == 0)
    {
        return;
    }
    machine->config = *config;
}

int JumpStateMachine_Request(JumpStateMachine *machine,
                             const JumpObservation *observation)
{
    if (machine == 0 || observation == 0 || machine->command.active)
    {
        return 0;
    }
    if (fabsf(observation->roll_error) > machine->config.request_roll_limit ||
        fabsf(observation->pitch_error) > machine->config.request_pitch_limit ||
        fabsf(observation->body_forward_v) > machine->config.request_speed_limit ||
        fabsf(observation->body_lateral_v) > machine->config.request_speed_limit ||
        fabsf(observation->body_z_vel) > machine->config.request_z_speed_limit ||
        !jump_is_grounded(machine, observation))
    {
        return 0;
    }

    machine->command.active = 1;
    machine->command.just_finished = 0;
    machine->has_been_airborne = 0;
    machine->extend_ready_steps = 0;
    machine->compress_elapsed = 0.0f;
    machine->compress_hold_elapsed = 0.0f;
    machine->landing_elapsed = 0.0f;
    machine->touch_elapsed = 0.0f;
    machine->recover_elapsed = 0.0f;
    machine->recover_stable_elapsed = 0.0f;
    machine->command.compress_l0_set =
        fmaxf(machine->config.compress_target,
              fmaxf(observation->left_l0, observation->right_l0));
    machine->command.landing_l0_set = machine->config.initial_leg_length;
    machine->command.extend_l0_set =
        fmaxf(observation->left_l0, observation->right_l0);
    machine->command.landing_support_scale = 1.0f;
    machine->command.landing_pid_scale = 1.0f;
    machine->command.landing_balance_f0 = 0.0f;
    machine->command.landing_balance_l0 = 0.0f;
    machine->command.wheel_balance_blend = 0.0f;
    machine->previous_min_clearance = jump_min_clearance(observation);
    machine->clearance_initialized = 1;
    jump_set_phase(machine,
                   machine->config.compression_enabled
                       ? JUMP_PHASE_COMPRESS
                       : JUMP_PHASE_THRUST);
    return 1;
}

void JumpStateMachine_Step(JumpStateMachine *machine,
                           const JumpObservation *observation,
                           float dt)
{
    if (machine == 0 || observation == 0)
    {
        return;
    }

    machine->command.just_finished = 0;
    if (!machine->command.active)
    {
        jump_set_phase(machine, JUMP_PHASE_STAND);
        return;
    }

    dt = fmaxf(dt, 0.0f);
    machine->phase_elapsed += dt;
    const float min_clearance = jump_min_clearance(observation);
    float clearance_rate = 0.0f;
    if (machine->clearance_initialized && dt > 0.0f)
    {
        clearance_rate =
            (min_clearance - machine->previous_min_clearance) / dt;
    }
    machine->previous_min_clearance = min_clearance;
    machine->clearance_initialized = 1;
    if (observation->airborne)
    {
        machine->has_been_airborne = 1;
    }

    switch (machine->command.phase)
    {
    case JUMP_PHASE_COMPRESS:
        machine->compress_elapsed += dt;
        machine->command.compress_l0_set =
            jump_move_toward(machine->command.compress_l0_set,
                             machine->config.compress_target,
                             machine->config.compress_rate,
                             dt);

        if (machine->command.compress_l0_set <=
                machine->config.compress_target + 1.0e-4f &&
            observation->left_l0 <=
                machine->config.compress_target +
                    machine->config.compress_tolerance &&
            observation->right_l0 <=
                machine->config.compress_target +
                    machine->config.compress_tolerance)
        {
            machine->compress_hold_elapsed += dt;
        }
        else
        {
            machine->compress_hold_elapsed = 0.0f;
        }

        if (machine->compress_hold_elapsed >=
            machine->config.compress_hold_time)
        {
            machine->extend_ready_steps = 0;
            machine->command.extend_l0_set =
                fmaxf(observation->left_l0, observation->right_l0);
            jump_set_phase(machine, JUMP_PHASE_THRUST);
        }
        else if (machine->compress_elapsed >=
                 machine->config.compress_timeout)
        {
            jump_finish(machine);
        }
        break;

    case JUMP_PHASE_THRUST:
    {
        machine->command.extend_l0_set =
            jump_move_toward(machine->command.extend_l0_set,
                             machine->config.extend_l0,
                             machine->config.extend_rate,
                             dt);

        const float extend_ready_l0 =
            machine->config.extend_l0 - machine->config.extend_end_margin;
        const int legs_extended =
            observation->left_l0 >= extend_ready_l0 &&
            observation->right_l0 >= extend_ready_l0;
        if (observation->airborne &&
            legs_extended &&
            machine->phase_elapsed >= machine->config.thrust_min_time)
        {
            ++machine->extend_ready_steps;
        }
        else
        {
            machine->extend_ready_steps = 0;
        }

        if (machine->extend_ready_steps >=
            machine->config.extend_hold_steps)
        {
            jump_set_phase(machine, JUMP_PHASE_TUCK);
        }
        else if (machine->phase_elapsed >= machine->config.thrust_timeout)
        {
            jump_finish(machine);
        }
        break;
    }

    case JUMP_PHASE_TUCK:
        machine->landing_elapsed = 0.0f;
        machine->touch_elapsed = 0.0f;
        machine->recover_elapsed = 0.0f;

        if (machine->phase_elapsed >= machine->config.tuck_min_time &&
            jump_first_touch(machine, observation))
        {
            machine->command.landing_l0_set =
                jump_clamp(fmaxf(observation->left_l0,
                                 observation->right_l0),
                           machine->config.buffer_l0,
                           machine->config.preland_l0);
            machine->command.landing_support_scale =
                machine->config.buffer_support_scale;
            machine->command.landing_pid_scale =
                machine->config.buffer_pid_scale;
            jump_set_phase(machine, JUMP_PHASE_BUFFER);
        }
        else if (machine->phase_elapsed >= machine->config.tuck_min_time &&
                 machine->has_been_airborne &&
                 min_clearance < machine->config.preland_clearance &&
                 (observation->body_z_vel < -0.05f ||
                  clearance_rate < -0.05f))
        {
            machine->command.landing_l0_set =
                jump_clamp(fmaxf(observation->left_l0,
                                 observation->right_l0),
                           machine->config.initial_leg_length,
                           machine->config.preland_l0);
            machine->command.landing_support_scale = 0.0f;
            machine->command.landing_pid_scale =
                machine->config.preland_pid_scale;
            jump_set_phase(machine, JUMP_PHASE_PRELAND);
        }
        else if (machine->phase_elapsed >= machine->config.tuck_timeout)
        {
            machine->command.landing_l0_set =
                jump_clamp(fmaxf(observation->left_l0,
                                 observation->right_l0),
                           machine->config.initial_leg_length,
                           machine->config.preland_l0);
            machine->command.landing_support_scale = 0.0f;
            machine->command.landing_pid_scale =
                machine->config.preland_pid_scale;
            jump_set_phase(machine, JUMP_PHASE_PRELAND);
        }
        break;

    case JUMP_PHASE_PRELAND:
        machine->landing_elapsed += dt;
        machine->command.landing_support_scale = 0.0f;
        machine->command.landing_pid_scale =
            machine->config.preland_pid_scale;
        machine->command.landing_l0_set =
            jump_move_toward(machine->command.landing_l0_set,
                             machine->config.preland_l0,
                             machine->config.preland_rate,
                             dt);

        if (jump_first_touch(machine, observation))
        {
            machine->landing_elapsed = 0.0f;
            machine->touch_elapsed = 0.0f;
            machine->command.landing_support_scale =
                machine->config.buffer_support_scale;
            machine->command.landing_pid_scale =
                machine->config.buffer_pid_scale;
            jump_set_phase(machine, JUMP_PHASE_BUFFER);
        }
        else if (machine->landing_elapsed >=
                 machine->config.preland_timeout)
        {
            machine->landing_elapsed = 0.0f;
            machine->touch_elapsed = 0.0f;
            machine->command.landing_support_scale =
                machine->config.buffer_support_scale;
            machine->command.landing_pid_scale =
                machine->config.buffer_pid_scale;
            jump_set_phase(machine, JUMP_PHASE_BUFFER);
        }
        break;

    case JUMP_PHASE_BUFFER:
        machine->landing_elapsed += dt;
        machine->command.landing_l0_set =
            jump_move_toward(machine->command.landing_l0_set,
                             machine->config.buffer_l0,
                             machine->config.buffer_rate,
                             dt);
        machine->command.landing_support_scale =
            machine->config.buffer_support_scale;
        machine->command.landing_pid_scale =
            machine->config.buffer_pid_scale;

        if (jump_stable_touch(machine, observation))
        {
            machine->touch_elapsed += dt;
        }
        else
        {
            machine->touch_elapsed = 0.0f;
        }

        if (machine->touch_elapsed >= machine->config.touchdown_hold &&
            machine->landing_elapsed >= machine->config.buffer_hold_time)
        {
            machine->landing_elapsed = 0.0f;
            machine->recover_elapsed = 0.0f;
            machine->command.landing_support_scale = 1.0f;
            machine->command.landing_pid_scale = 0.65f;
            jump_set_phase(machine, JUMP_PHASE_RECOVER);
        }
        else if (machine->landing_elapsed >= machine->config.buffer_timeout &&
                 !observation->airborne &&
                 jump_min_clearance(observation) <= 0.003f)
        {
            machine->landing_elapsed = 0.0f;
            machine->recover_elapsed = 0.0f;
            machine->recover_stable_elapsed = 0.0f;
            machine->command.landing_support_scale = 1.0f;
            machine->command.landing_pid_scale = 0.65f;
            jump_set_phase(machine, JUMP_PHASE_RECOVER);
        }
        break;

    case JUMP_PHASE_RECOVER:
        machine->recover_elapsed += dt;
        machine->command.landing_l0_set =
            jump_move_toward(machine->command.landing_l0_set,
                             machine->config.initial_leg_length,
                             machine->config.recover_rate,
                             dt);
        machine->command.landing_support_scale = 1.0f;
        machine->command.landing_pid_scale = 0.65f;
        if (machine->config.wheel_recover_blend_time > 0.0f)
        {
            machine->command.wheel_balance_blend =
                jump_clamp(machine->recover_elapsed /
                               machine->config.wheel_recover_blend_time,
                           0.0f,
                           1.0f);
        }
        else
        {
            machine->command.wheel_balance_blend = 1.0f;
        }

        const int actual_leg_ready =
            fabsf(observation->left_l0 - machine->config.initial_leg_length) <
                machine->config.recover_l0_tolerance &&
            fabsf(observation->right_l0 - machine->config.initial_leg_length) <
                machine->config.recover_l0_tolerance;
        const int attitude_ready =
            fabsf(observation->roll_error) < machine->config.recover_roll_limit &&
            fabsf(observation->pitch_error) < machine->config.recover_pitch_limit &&
            fabsf(observation->pitch_rate) < machine->config.recover_pitch_rate;
        const int motion_ready =
            fabsf(observation->body_forward_v) <
                machine->config.recover_speed_limit &&
            fabsf(observation->body_lateral_v) <
                machine->config.recover_speed_limit;
        const int stable_recovery =
            jump_stable_touch(machine, observation) &&
            attitude_ready &&
            motion_ready &&
            (actual_leg_ready ||
             machine->recover_elapsed >= machine->config.recover_timeout);

        if (stable_recovery)
        {
            machine->recover_stable_elapsed += dt;
        }
        else
        {
            machine->recover_stable_elapsed = 0.0f;
        }

        if (machine->recover_elapsed >= machine->config.recover_time &&
            machine->recover_stable_elapsed >=
                machine->config.recover_stable_time)
        {
            jump_finish(machine);
        }
        break;

    case JUMP_PHASE_STAND:
    default:
        break;
    }

    jump_update_landing_balance(machine, observation, dt);
}

const JumpCommand *JumpStateMachine_GetCommand(const JumpStateMachine *machine)
{
    return machine != 0 ? &machine->command : 0;
}
