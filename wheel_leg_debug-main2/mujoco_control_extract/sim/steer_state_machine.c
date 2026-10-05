#include "steer_state_machine.h"

#include <math.h>
#include <string.h>

static float steer_clamp(float value, float min_value, float max_value)
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

static float steer_abs(float value)
{
    return value < 0.0f ? -value : value;
}

static float steer_move_toward(float value, float target, float rate, float dt)
{
    const float max_step = fabsf(rate) * fmaxf(dt, 0.0f);
    if (max_step <= 0.0f)
    {
        return target;
    }
    return value + steer_clamp(target - value, -max_step, max_step);
}

static int steer_l0_near_target(const SteerStateMachine *machine,
                                const SteerObservation *observation,
                                float target,
                                float tolerance)
{
    const float command_error = steer_abs(machine->leg_set_cmd - target);
    const float left_error = steer_abs(observation->left_l0 - target);
    const float right_error = steer_abs(observation->right_l0 - target);
    return command_error <= tolerance &&
           left_error <= tolerance &&
           right_error <= tolerance;
}

static float steer_planar_speed(const SteerObservation *observation)
{
    return sqrtf(observation->body_v_x * observation->body_v_x +
                 observation->body_v_y * observation->body_v_y);
}

static float steer_planar_error(const SteerStateMachine *machine,
                                const SteerObservation *observation)
{
    const float error_x = observation->body_x - machine->anchor_x;
    const float error_y = observation->body_y - machine->anchor_y;
    return sqrtf(error_x * error_x + error_y * error_y);
}

static float steer_planar_yaw_scale(const SteerStateMachine *machine,
                                    const SteerObservation *observation)
{
    const float limit = machine->config.active_planar_error_limit;
    if (limit <= 0.0f)
    {
        return 1.0f;
    }

    const float error = steer_planar_error(machine, observation);
    const float slowdown_start = 0.5f * limit;
    const float emergency_limit = 1.5f * limit;
    if (error <= slowdown_start)
    {
        return 1.0f;
    }
    if (error >= emergency_limit)
    {
        return 0.0f;
    }

    return (emergency_limit - error) /
           (emergency_limit - slowdown_start);
}

static void steer_set_phase(SteerStateMachine *machine, SteerPhase phase)
{
    if (machine->phase != phase)
    {
        machine->phase = phase;
        machine->phase_time = 0.0f;
        if (phase != STEER_PHASE_BRAKE)
        {
            machine->brake_stable_time = 0.0f;
        }
    }
}

static int steer_request_active(const SteerStateMachine *machine, float input)
{
    return steer_abs(input) > machine->config.input_deadband;
}

static int steer_abort_requested(const SteerStateMachine *machine,
                                 const SteerObservation *observation)
{
    if (!observation->start_enabled ||
        !observation->chassis_safe ||
        observation->jump_active ||
        observation->drive_mode == STEER_DRIVE_JUMP)
    {
        return 1;
    }

    if (steer_abs(observation->roll) > machine->config.abort_roll_limit ||
        steer_abs(observation->pitch) > machine->config.abort_pitch_limit)
    {
        return 1;
    }

    return 0;
}

static void steer_fill_command(const SteerStateMachine *machine,
                               const SteerObservation *observation,
                               SteerCommand *command)
{
    memset(command, 0, sizeof(*command));
    command->phase = machine->phase;
    command->leg_set = observation->base_leg_set;
    command->yaw_hold = observation->base_yaw_hold;
    command->yaw_lock = 1;
    command->target_x = observation->body_x;
    command->target_y = observation->body_y;

    if (machine->phase != STEER_PHASE_IDLE &&
        machine->phase != STEER_PHASE_ABORT)
    {
        command->active = 1;
        command->planar_lock = 1;
        command->leg_set = machine->leg_set_cmd;
        command->yaw_hold = machine->yaw_hold_cmd;
        command->yaw_rate_ref = machine->yaw_rate_cmd;
        command->target_x = machine->anchor_x;
        command->target_y = machine->anchor_y;
        command->force_stand = 1;
    }
}

void SteerStateMachine_Init(SteerStateMachine *machine,
                            const SteerStateMachineConfig *config)
{
    if (machine == 0)
    {
        return;
    }

    memset(machine, 0, sizeof(*machine));
    SteerStateMachine_SetConfig(machine, config);
    machine->phase = STEER_PHASE_IDLE;
}

void SteerStateMachine_SetConfig(SteerStateMachine *machine,
                                 const SteerStateMachineConfig *config)
{
    if (machine == 0 || config == 0)
    {
        return;
    }

    machine->config = *config;
    machine->config.turn_leg_length =
        fmaxf(0.0f, machine->config.turn_leg_length);
    machine->config.leg_rate = fabsf(machine->config.leg_rate);
    machine->config.yaw_rate_max = fabsf(machine->config.yaw_rate_max);
    machine->config.yaw_accel_limit =
        fabsf(machine->config.yaw_accel_limit);
    machine->config.active_time_limit =
        fmaxf(0.0f, machine->config.active_time_limit);
    machine->config.active_planar_error_limit =
        fmaxf(0.0f, machine->config.active_planar_error_limit);
    machine->config.input_deadband =
        fmaxf(0.0f, machine->config.input_deadband);
    machine->config.prepare_l0_tolerance =
        fmaxf(0.0f, machine->config.prepare_l0_tolerance);
    machine->config.brake_gyro_tolerance =
        fmaxf(0.0f, machine->config.brake_gyro_tolerance);
    machine->config.brake_linear_velocity_tolerance =
        fmaxf(0.0f, machine->config.brake_linear_velocity_tolerance);
    machine->config.brake_hold_time =
        fmaxf(0.0f, machine->config.brake_hold_time);
    machine->config.recover_linear_velocity_tolerance =
        fmaxf(0.0f, machine->config.recover_linear_velocity_tolerance);
    machine->config.recover_l0_tolerance =
        fmaxf(0.0f, machine->config.recover_l0_tolerance);
    machine->config.abort_roll_limit =
        fmaxf(0.0f, machine->config.abort_roll_limit);
    machine->config.abort_pitch_limit =
        fmaxf(0.0f, machine->config.abort_pitch_limit);
    machine->initialized = 1;
}

void SteerStateMachine_Step(SteerStateMachine *machine,
                            const SteerObservation *observation,
                            SteerCommand *command)
{
    if (machine == 0 || observation == 0 || command == 0)
    {
        return;
    }

    if (!machine->initialized)
    {
        SteerStateMachineConfig config = {
            .turn_leg_length = 0.20f,
            .leg_rate = 0.30f,
            .yaw_rate_max = 0.50f,
            .yaw_accel_limit = 1.00f,
            .active_time_limit = 0.0f,
            .active_planar_error_limit = 0.60f,
            .input_deadband = 0.05f,
            .prepare_l0_tolerance = 0.012f,
            .brake_gyro_tolerance = 0.08f,
            .brake_linear_velocity_tolerance = 0.04f,
            .brake_hold_time = 0.12f,
            .recover_linear_velocity_tolerance = 0.05f,
            .recover_l0_tolerance = 0.002f,
            .abort_roll_limit = 0.30f,
            .abort_pitch_limit = 0.35f,
        };
        SteerStateMachine_Init(machine, &config);
    }

    const float dt = fmaxf(observation->dt, 0.0f);
    const float input = steer_clamp(observation->input, -1.0f, 1.0f);
    const int requested = steer_request_active(machine, input);
    const int abort = steer_abort_requested(machine, observation);
    const SteerPhase previous_phase = machine->phase;
    machine->phase_time += dt;

    if (abort)
    {
        if (previous_phase == STEER_PHASE_IDLE ||
            (previous_phase == STEER_PHASE_ABORT &&
             machine->recover_leg_set <= 0.0f))
        {
            machine->leg_set_cmd = observation->base_leg_set;
            machine->recover_leg_set = observation->base_leg_set;
            machine->anchor_x = observation->body_x;
            machine->anchor_y = observation->body_y;
        }
        machine->yaw_hold_cmd = observation->current_yaw;
        machine->yaw_rate_cmd = 0.0f;
        steer_set_phase(machine, STEER_PHASE_ABORT);
    }

    switch (machine->phase)
    {
    case STEER_PHASE_IDLE:
        machine->leg_set_cmd = observation->base_leg_set;
        machine->recover_leg_set = observation->base_leg_set;
        machine->yaw_hold_cmd = observation->base_yaw_hold;
        machine->yaw_rate_cmd = 0.0f;
        if (!requested)
        {
            machine->cycle_limited = 0;
            machine->session_active = 0;
            machine->anchor_x = observation->body_x;
            machine->anchor_y = observation->body_y;
        }
        if (!abort && requested && observation->drive_mode == STEER_DRIVE_STAND)
        {
            machine->recover_leg_set = observation->base_leg_set;
            machine->leg_set_cmd = observation->base_leg_set;
            if (!machine->session_active)
            {
                machine->anchor_x = observation->body_x;
                machine->anchor_y = observation->body_y;
                machine->session_active = 1;
            }
            machine->yaw_hold_cmd = observation->current_yaw;
            machine->cycle_limited = 0;
            steer_set_phase(machine, STEER_PHASE_PREPARE);
        }
        break;

    case STEER_PHASE_PREPARE:
        machine->yaw_rate_cmd =
            steer_move_toward(machine->yaw_rate_cmd,
                              0.0f,
                              machine->config.yaw_accel_limit,
                              dt);
        if (!requested)
        {
            steer_set_phase(machine, STEER_PHASE_RECOVER);
            break;
        }
        machine->yaw_hold_cmd = observation->current_yaw;
        machine->leg_set_cmd =
            steer_move_toward(machine->leg_set_cmd,
                              machine->config.turn_leg_length,
                              machine->config.leg_rate,
                              dt);
        if (steer_l0_near_target(machine,
                                 observation,
                                 machine->config.turn_leg_length,
                                 machine->config.prepare_l0_tolerance))
        {
            steer_set_phase(machine, STEER_PHASE_ACTIVE);
        }
        break;

    case STEER_PHASE_ACTIVE:
        if (!requested)
        {
            steer_set_phase(machine, STEER_PHASE_BRAKE);
            break;
        }
        machine->leg_set_cmd =
            steer_move_toward(machine->leg_set_cmd,
                              machine->config.turn_leg_length,
                              machine->config.leg_rate,
                              dt);
        machine->yaw_rate_cmd =
            steer_move_toward(
                machine->yaw_rate_cmd,
                input * machine->config.yaw_rate_max *
                    steer_planar_yaw_scale(machine, observation),
                machine->config.yaw_accel_limit,
                dt);
        machine->yaw_hold_cmd += machine->yaw_rate_cmd * dt;
        if ((machine->config.active_time_limit > 0.0f &&
             machine->phase_time >= machine->config.active_time_limit) ||
            (machine->config.active_planar_error_limit > 0.0f &&
             steer_planar_error(machine, observation) >=
                 1.5f * machine->config.active_planar_error_limit))
        {
            machine->cycle_limited = 1;
            steer_set_phase(machine, STEER_PHASE_BRAKE);
        }
        break;

    case STEER_PHASE_BRAKE:
        if (requested && !machine->cycle_limited)
        {
            steer_set_phase(machine, STEER_PHASE_ACTIVE);
            break;
        }
        machine->leg_set_cmd =
            steer_move_toward(machine->leg_set_cmd,
                              machine->config.turn_leg_length,
                              machine->config.leg_rate,
                              dt);
        machine->yaw_rate_cmd =
            steer_move_toward(machine->yaw_rate_cmd,
                              0.0f,
                              machine->config.yaw_accel_limit,
                              dt);
        machine->yaw_hold_cmd += machine->yaw_rate_cmd * dt;
        if (steer_abs(machine->yaw_rate_cmd) <= 0.01f &&
            steer_abs(observation->gyro_z) <=
                machine->config.brake_gyro_tolerance &&
            steer_planar_speed(observation) <=
                machine->config.brake_linear_velocity_tolerance)
        {
            machine->brake_stable_time += dt;
        }
        else
        {
            machine->brake_stable_time = 0.0f;
        }
        if (machine->brake_stable_time >=
            machine->config.brake_hold_time)
        {
            steer_set_phase(machine, STEER_PHASE_RECOVER);
        }
        break;

    case STEER_PHASE_RECOVER:
        if (requested && !machine->cycle_limited &&
            observation->drive_mode == STEER_DRIVE_STAND)
        {
            steer_set_phase(machine, STEER_PHASE_PREPARE);
            break;
        }
        machine->leg_set_cmd =
            steer_move_toward(machine->leg_set_cmd,
                              machine->recover_leg_set,
                              machine->config.leg_rate,
                              dt);
        machine->yaw_rate_cmd =
            steer_move_toward(machine->yaw_rate_cmd,
                              0.0f,
                              machine->config.yaw_accel_limit,
                              dt);
        if (steer_abs(machine->leg_set_cmd - machine->recover_leg_set) <=
                machine->config.recover_l0_tolerance &&
            steer_planar_speed(observation) <=
                machine->config.recover_linear_velocity_tolerance)
        {
            machine->leg_set_cmd = machine->recover_leg_set;
            if (!requested &&
                steer_abs(observation->base_leg_set - machine->recover_leg_set) <=
                    1.0e-4f)
            {
                machine->cycle_limited = 0;
                steer_set_phase(machine, STEER_PHASE_IDLE);
            }
        }
        break;

    case STEER_PHASE_ABORT:
        if (!abort)
        {
            if (requested && observation->drive_mode == STEER_DRIVE_STAND)
            {
                machine->yaw_hold_cmd = observation->current_yaw;
                machine->cycle_limited = 0;
                steer_set_phase(machine, STEER_PHASE_PREPARE);
            }
            else if (steer_abs(machine->leg_set_cmd - machine->recover_leg_set) >
                     machine->config.recover_l0_tolerance)
            {
                steer_set_phase(machine, STEER_PHASE_RECOVER);
            }
            else
            {
                if (!requested)
                {
                    machine->session_active = 0;
                }
                steer_set_phase(machine, STEER_PHASE_IDLE);
            }
        }
        break;
    }

    steer_fill_command(machine, observation, command);
}

const char *SteerStateMachine_PhaseName(SteerPhase phase)
{
    switch (phase)
    {
    case STEER_PHASE_IDLE:
        return "idle";
    case STEER_PHASE_PREPARE:
        return "prepare";
    case STEER_PHASE_ACTIVE:
        return "active";
    case STEER_PHASE_BRAKE:
        return "brake";
    case STEER_PHASE_RECOVER:
        return "recover";
    case STEER_PHASE_ABORT:
        return "abort";
    default:
        return "unknown";
    }
}
