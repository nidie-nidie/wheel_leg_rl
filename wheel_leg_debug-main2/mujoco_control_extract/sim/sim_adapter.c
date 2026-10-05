#include "sim_adapter.h"

#include <math.h>
#include <string.h>

#include "ChassisL_Task.h"
#include "ChassisR_Task.h"
#include "Chassis_Task.h"
#include "INS_Task.h"
#include "Motor.h"
#include "jump_state_machine.h"
#include "robot_param.h"

extern INS_t INS;

static float sim_stand_l0_pitch = M_PI_2;
static int sim_safe_mit_enabled = 1;
static int sim_drive_forward = 0;
static float sim_position_hold_blend = 1.0f;
static int sim_yaw_lock = 1;
static int sim_airborne_pose_valid = 0;
static float sim_body_z_vel = 0.0f;
static float sim_body_forward_v = 0.0f;
static float sim_body_lateral_v = 0.0f;
static JumpStateMachine sim_jump_machine;
static JumpObservation sim_jump_observation;
static JumpStateMachineConfig sim_jump_config = {
    .compression_enabled = 1,
    .initial_leg_length = INIT_LEG_LENGTH,
    .min_leg_length = MIN_LEG_LENGTH,
    .max_leg_length = MAX_LEG_LENGTH,
    .compress_target = MUJOCO_JUMP_COMPRESS_TARGET,
    .compress_rate = MUJOCO_JUMP_COMPRESS_RATE,
    .compress_support_scale = MUJOCO_JUMP_COMPRESS_SUPPORT_SCALE,
    .compress_tolerance = MUJOCO_JUMP_COMPRESS_TOLERANCE,
    .compress_hold_time = MUJOCO_JUMP_COMPRESS_HOLD_TIME,
    .compress_timeout = MUJOCO_JUMP_COMPRESS_TIMEOUT,
    .extend_l0 = MUJOCO_JUMP_EXTEND_L0,
    .extend_end_margin = MUJOCO_JUMP_EXTEND_END_MARGIN,
    .extend_rate = MUJOCO_JUMP_EXTEND_RATE,
    .thrust_min_time = MUJOCO_JUMP_THRUST_MIN_TIME,
    .thrust_timeout = MUJOCO_JUMP_THRUST_TIMEOUT,
    .extend_hold_steps = 2,
    .tuck_min_time = MUJOCO_JUMP_TUCK_MIN_TIME,
    .tuck_timeout = MUJOCO_JUMP_TUCK_TIMEOUT,
    .preland_clearance = MUJOCO_JUMP_PRELAND_CLEARANCE,
    .preland_l0 = MUJOCO_JUMP_PRELAND_L0,
    .preland_rate = MUJOCO_JUMP_PRELAND_RATE,
    .preland_pid_scale = MUJOCO_JUMP_PRELAND_PID_SCALE,
    .preland_timeout = MUJOCO_JUMP_PRELAND_TIMEOUT,
    .touchdown_force = MUJOCO_JUMP_TOUCHDOWN_FORCE,
    .touchdown_hold = MUJOCO_JUMP_TOUCHDOWN_HOLD,
    .buffer_l0 = MUJOCO_JUMP_BUFFER_L0,
    .buffer_rate = MUJOCO_JUMP_BUFFER_RATE,
    .buffer_support_scale = MUJOCO_JUMP_BUFFER_SUPPORT_SCALE,
    .buffer_pid_scale = MUJOCO_JUMP_BUFFER_PID_SCALE,
    .buffer_hold_time = MUJOCO_JUMP_BUFFER_HOLD_TIME,
    .buffer_timeout = MUJOCO_JUMP_BUFFER_TIMEOUT,
    .recover_rate = MUJOCO_JUMP_RECOVER_RATE,
    .recover_time = MUJOCO_JUMP_RECOVER_TIME,
    .recover_stable_time = MUJOCO_JUMP_RECOVER_STABLE_TIME,
    .recover_timeout = MUJOCO_JUMP_RECOVER_TIMEOUT,
    .recover_l0_tolerance = MUJOCO_JUMP_RECOVER_L0_TOLERANCE,
    .recover_roll_limit = MUJOCO_JUMP_RECOVER_ROLL_LIMIT,
    .recover_pitch_limit = MUJOCO_JUMP_RECOVER_PITCH_LIMIT,
    .recover_pitch_rate = MUJOCO_JUMP_RECOVER_PITCH_RATE,
    .recover_speed_limit = MUJOCO_JUMP_RECOVER_SPEED_LIMIT,
    .wheel_recover_blend_time = MUJOCO_JUMP_WHEEL_RECOVER_BLEND_TIME,
    .request_roll_limit = MUJOCO_JUMP_REQUEST_ROLL_LIMIT,
    .request_pitch_limit = MUJOCO_JUMP_REQUEST_PITCH_LIMIT,
    .request_speed_limit = MUJOCO_JUMP_REQUEST_SPEED_LIMIT,
    .request_z_speed_limit = MUJOCO_JUMP_REQUEST_Z_SPEED_LIMIT,
    .landing_roll_f0_kp = MUJOCO_JUMP_LANDING_ROLL_F0_KP,
    .landing_roll_f0_kd = MUJOCO_JUMP_LANDING_ROLL_F0_KD,
    .landing_contact_f0_kp = MUJOCO_JUMP_LANDING_CONTACT_F0_KP,
    .landing_balance_f0_limit = MUJOCO_JUMP_LANDING_BALANCE_F0_LIMIT,
    .landing_roll_l0_kp = MUJOCO_JUMP_LANDING_ROLL_L0_KP,
    .landing_roll_l0_kd = MUJOCO_JUMP_LANDING_ROLL_L0_KD,
    .landing_balance_l0_limit = MUJOCO_JUMP_LANDING_BALANCE_L0_LIMIT,
    .landing_clearance_l0_kp = MUJOCO_JUMP_LANDING_CLEARANCE_L0_KP,
    .landing_clearance_l0_rate = MUJOCO_JUMP_LANDING_CLEARANCE_L0_RATE,
    .landing_clearance_l0_limit = MUJOCO_JUMP_LANDING_CLEARANCE_L0_LIMIT,
    .lqr_tp_weight = MUJOCO_JUMP_LQR_TP_WEIGHT,
    .split_tp_weight = MUJOCO_JUMP_SPLIT_TP_WEIGHT,
    .pitch_tp_weight = MUJOCO_JUMP_PITCH_TP_WEIGHT,
    .leg_swing_tp_weight = MUJOCO_JUMP_LEG_SWING_TP_WEIGHT,
};
static float sim_airborne_joint_target[4];
static float sim_airborne_pose_kp = MUJOCO_JUMP_TUCK_KP;
static float sim_airborne_pose_kd = MUJOCO_JUMP_TUCK_KD;
static float sim_airborne_pose_torque_limit = MUJOCO_JUMP_TUCK_TORQUE_LIMIT;

/*
 * Legacy MuJoCo tuning values are retained by the adapter API only.
 * The real controller does not consume these variables.
 */
float mujoco_jump_thrust_ff = MUJOCO_JUMP_THRUST_FF;
static float mujoco_jump_pitch_target = MUJOCO_JUMP_PITCH_TARGET;
static float mujoco_jump_pitch_tp_kp = MUJOCO_JUMP_PITCH_TP_KP;
static float mujoco_jump_pitch_tp_kd = MUJOCO_JUMP_PITCH_TP_KD;
static float mujoco_jump_pitch_tp_limit = MUJOCO_JUMP_PITCH_TP_LIMIT;
float mujoco_jump_compress_l0_set = INIT_LEG_LENGTH;
static float mujoco_jump_compress_target = MUJOCO_JUMP_COMPRESS_TARGET;
static float mujoco_jump_compress_rate = MUJOCO_JUMP_COMPRESS_RATE;
float mujoco_jump_compress_support_scale = MUJOCO_JUMP_COMPRESS_SUPPORT_SCALE;
static float mujoco_jump_compress_tolerance = MUJOCO_JUMP_COMPRESS_TOLERANCE;
static float mujoco_jump_compress_hold_time = MUJOCO_JUMP_COMPRESS_HOLD_TIME;
static float mujoco_jump_compress_timeout = MUJOCO_JUMP_COMPRESS_TIMEOUT;
static float mujoco_jump_leg_swing_offset = MUJOCO_JUMP_LEG_SWING_OFFSET;
static float mujoco_jump_leg_swing_kp = MUJOCO_JUMP_LEG_SWING_KP;
static float mujoco_jump_leg_swing_kd = MUJOCO_JUMP_LEG_SWING_KD;
static float mujoco_jump_leg_swing_limit = MUJOCO_JUMP_LEG_SWING_LIMIT;
static float mujoco_jump_extend_l0 = MUJOCO_JUMP_EXTEND_L0;
float mujoco_jump_extend_l0_set = INIT_LEG_LENGTH;
static float mujoco_jump_extend_end_margin = MUJOCO_JUMP_EXTEND_END_MARGIN;
float mujoco_jump_landing_l0_set = INIT_LEG_LENGTH;
float mujoco_jump_landing_support_scale = 1.0f;
float mujoco_jump_landing_pid_scale = 1.0f;
float mujoco_jump_landing_balance_f0 = 0.0f;
float mujoco_jump_landing_balance_l0 = 0.0f;
static float mujoco_jump_lqr_tp_weight = 1.0f;
static float mujoco_jump_split_tp_weight = 1.0f;
static float mujoco_jump_pitch_tp_weight = 0.0f;
static float mujoco_jump_leg_swing_tp_weight = 0.0f;

static float sim_mit_torque(float position_set, float velocity_set, float kp, float kd, float torque_ff,
                            const DM_Motor_Info_Typedef *motor)
{
    return kp * (position_set - motor->Data.Position) +
           kd * (velocity_set - motor->Data.Velocity) +
           torque_ff;
}

static float sim_clamp_float(float value, float min_value, float max_value)
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

static float sim_airborne_pose_torque(int joint_index)
{
    const float torque = sim_mit_torque(sim_airborne_joint_target[joint_index],
                                        0.0f,
                                        sim_airborne_pose_kp,
                                        sim_airborne_pose_kd,
                                        0.0f,
                                        chassis_move.joint_motor[joint_index]);
    return sim_clamp_float(torque,
                           -sim_airborne_pose_torque_limit,
                           sim_airborne_pose_torque_limit);
}

static void sim_sync_jump_command(void)
{
    static int last_phase = JUMP_PHASE_STAND;
    const JumpCommand *command = JumpStateMachine_GetCommand(&sim_jump_machine);
    if (command == 0)
    {
        return;
    }

    if ((int)command->phase != last_phase)
    {
        jump_time_l = 0;
        jump_time_r = 0;
        PID_Calc_Clear(&legl_pid);
        PID_Calc_Clear(&legr_pid);
        last_phase = (int)command->phase;
    }

    chassis_move.jump_flag = (uint8_t)command->phase;
    chassis_move.jump_flag2 = (uint8_t)command->phase;
    mujoco_jump_compress_l0_set = command->compress_l0_set;
    mujoco_jump_extend_l0_set = command->extend_l0_set;
    mujoco_jump_landing_l0_set = command->landing_l0_set;
    mujoco_jump_landing_support_scale = command->landing_support_scale;
    mujoco_jump_landing_pid_scale = command->landing_pid_scale;
    mujoco_jump_landing_balance_f0 = command->landing_balance_f0;
    mujoco_jump_landing_balance_l0 = command->landing_balance_l0;
    mujoco_jump_lqr_tp_weight = command->lqr_tp_weight;
    mujoco_jump_split_tp_weight = command->split_tp_weight;
    mujoco_jump_pitch_tp_weight = command->pitch_tp_weight;
    mujoco_jump_leg_swing_tp_weight = command->leg_swing_tp_weight;

    if (command->just_finished)
    {
        chassis_move.leg_set = INIT_LEG_LENGTH;
        chassis_move.last_leg_set = INIT_LEG_LENGTH;
    }
}

static void sim_update_jump_body_observation(void)
{
    sim_jump_observation.left_l0 = left.L0;
    sim_jump_observation.right_l0 = right.L0;
    sim_jump_observation.roll_error = INS.Roll - chassis_move.roll_set;
    sim_jump_observation.roll_rate = INS.Gyro[0];
    sim_jump_observation.pitch_error = INS.Pitch - MUJOCO_WHEEL_BALANCE_PITCH_TARGET;
    sim_jump_observation.pitch_rate = INS.Gyro[1];
    sim_jump_observation.body_forward_v = sim_body_forward_v;
    sim_jump_observation.body_lateral_v = sim_body_lateral_v;
    sim_jump_observation.body_z_vel = sim_body_z_vel;
}

static void sim_apply_native_wheel_balance(SimControllerOutput *output)
{
    const float pitch_term = MUJOCO_WHEEL_BALANCE_PITCH_KP * (INS.Pitch - MUJOCO_WHEEL_BALANCE_PITCH_TARGET) +
                             MUJOCO_WHEEL_BALANCE_PITCH_KD * INS.Gyro[1];
    const float pos_term = sim_drive_forward
                               ? 0.0f
                               : MUJOCO_WHEEL_BALANCE_POS_KP * sim_position_hold_blend *
                                     (chassis_move.x_filter - chassis_move.x_set);
    const float vel_term = MUJOCO_WHEEL_BALANCE_VEL_KD * (chassis_move.v_filter - chassis_move.v_set);
    const float drive_term = MUJOCO_WHEEL_BALANCE_DRIVE_KFF * chassis_move.v_set;
    const float yaw_term = sim_yaw_lock
                               ? MUJOCO_WHEEL_BALANCE_YAW_KP * (INS.Yaw - chassis_move.turn_set) +
                                     MUJOCO_WHEEL_BALANCE_YAW_KD * INS.Gyro[2]
                               : 0.0f;
    const float common = sim_clamp_float(pitch_term + pos_term + vel_term + drive_term,
                                         -MUJOCO_WHEEL_BALANCE_LIMIT,
                                         MUJOCO_WHEEL_BALANCE_LIMIT);

    output->wheel_torque[0] = common - yaw_term;
    output->wheel_torque[1] = common + yaw_term;
}

static void sim_apply_stand_joint_targets(void)
{
    float phi1_phi4_l[2];
    float phi1_phi4_r[2];

    CalcPhi1AndPhi4(sim_stand_l0_pitch, chassis_move.leg_set, phi1_phi4_l);
    CalcPhi1AndPhi4(sim_stand_l0_pitch, chassis_move.leg_set, phi1_phi4_r);

    if (isnan(phi1_phi4_l[0]) || isnan(phi1_phi4_l[1]) ||
        isnan(phi1_phi4_r[0]) || isnan(phi1_phi4_r[1]))
    {
        return;
    }

    left.position_set[0] = -theta_transform(phi1_phi4_l[1], -J0_ANGLE_OFFSET, J0_DIRECTION, 1);
    left.position_set[1] = -theta_transform(phi1_phi4_l[0], -J1_ANGLE_OFFSET, J1_DIRECTION, 1);
    right.position_set[0] = -theta_transform(phi1_phi4_r[0], -J2_ANGLE_OFFSET, J2_DIRECTION, 1);
    right.position_set[1] = -theta_transform(phi1_phi4_r[1], -J3_ANGLE_OFFSET, J3_DIRECTION, 1);
}

void SimController_SetStandL0Pitch(float l0_pitch)
{
    sim_stand_l0_pitch = l0_pitch;
}

void SimController_SetSafeMitEnabled(int enabled)
{
    sim_safe_mit_enabled = enabled != 0;
}

void SimController_Init(void)
{
    memset(&INS, 0, sizeof(INS));
    memset(&sim_jump_observation, 0, sizeof(sim_jump_observation));

    sim_body_z_vel = 0.0f;
    INS.ins_flag = 1;
    chassis_move.start_flag = 1;
    chassis_move.mode = CHASSIS_STAND_UP;
    chassis_move.leg_set = INIT_LEG_LENGTH;
    chassis_move.last_leg_set = INIT_LEG_LENGTH;
    mujoco_jump_landing_l0_set = INIT_LEG_LENGTH;
    mujoco_jump_landing_support_scale = 1.0f;
    mujoco_jump_landing_pid_scale = 1.0f;
    mujoco_jump_landing_balance_f0 = 0.0f;
    mujoco_jump_landing_balance_l0 = 0.0f;
    chassis_move.roll_set = INIT_ROLL;
    chassis_move.turn_set = 0.0f;
    chassis_move.x_set = 0.0f;
    chassis_move.v_set = 0.0f;

    ChassisL_init();
    ChassisR_init();
    Pensation_init();
    JumpStateMachine_Init(&sim_jump_machine, &sim_jump_config);
    sim_sync_jump_command();
    ConsoleStandUp();
    sim_apply_stand_joint_targets();
}

void SimController_SetState(const SimControllerState *state)
{
    if (state == 0)
    {
        return;
    }

    for (int i = 0; i < 4; ++i)
    {
        DM_8009_Motor[i].Data.Position = state->joint_pos[i];
        DM_8009_Motor[i].Data.Velocity = state->joint_vel[i];
        DM_8009_Motor[i].Data.Torque = state->joint_torque_fdb[i];
    }

    for (int i = 0; i < 2; ++i)
    {
        LK_9025_Motor[i].Data.Velocity = state->wheel_vel[i];
    }

    INS.Roll = state->roll;
    INS.Pitch = state->pitch;
    INS.Yaw = state->yaw;
    INS.YawTotalAngle = state->yaw;

    for (int i = 0; i < 3; ++i)
    {
        INS.Gyro[i] = state->gyro[i];
        INS.MotionAccel_b[i] = state->accel_body[i];
        INS.MotionAccel_n[i] = state->accel_world[i];
    }

    chassis_move.x_filter = state->body_x;
    chassis_move.v_filter = state->body_v;
    sim_body_z_vel = state->body_z_vel;
    sim_body_forward_v = state->body_v;
    sim_body_lateral_v = state->body_v_y;
}

void SimController_SetCommand(float v_set, float x_set, float leg_set, float roll_set, float yaw_set)
{
    chassis_move.v_set = v_set;
    chassis_move.x_set = x_set;
    chassis_move.leg_set = leg_set;
    chassis_move.roll_set = roll_set;
    chassis_move.turn_set = yaw_set;
}

void SimController_SetDriveContext(int drive_forward, float position_hold_blend, int yaw_lock)
{
    sim_drive_forward = drive_forward != 0;
    sim_position_hold_blend = sim_clamp_float(position_hold_blend, 0.0f, 1.0f);
    sim_yaw_lock = yaw_lock != 0;
}

void SimController_SetJumpThrust(float thrust_ff)
{
    mujoco_jump_thrust_ff = thrust_ff;
}

void SimController_SetJumpPitchTp(float target, float kp, float kd, float limit)
{
    mujoco_jump_pitch_target = target;
    mujoco_jump_pitch_tp_kp = kp;
    mujoco_jump_pitch_tp_kd = kd;
    mujoco_jump_pitch_tp_limit = limit;
}

void SimController_SetJumpCompression(int enabled,
                                      float target,
                                      float rate,
                                      float support_scale,
                                      float tolerance,
                                      float hold_time,
                                      float timeout)
{
    sim_jump_config.compression_enabled = enabled != 0;
    mujoco_jump_compress_target = sim_clamp_float(target,
                                                   MIN_LEG_LENGTH,
                                                   INIT_LEG_LENGTH);
    mujoco_jump_compress_rate = fabsf(rate);
    mujoco_jump_compress_support_scale = sim_clamp_float(support_scale, 0.0f, 1.0f);
    mujoco_jump_compress_tolerance = fabsf(tolerance);
    mujoco_jump_compress_hold_time = fmaxf(0.0f, hold_time);
    mujoco_jump_compress_timeout = fmaxf(mujoco_jump_compress_hold_time, timeout);
    sim_jump_config.compress_target = mujoco_jump_compress_target;
    sim_jump_config.compress_rate = mujoco_jump_compress_rate;
    sim_jump_config.compress_support_scale = mujoco_jump_compress_support_scale;
    sim_jump_config.compress_tolerance = mujoco_jump_compress_tolerance;
    sim_jump_config.compress_hold_time = mujoco_jump_compress_hold_time;
    sim_jump_config.compress_timeout = mujoco_jump_compress_timeout;
    JumpStateMachine_SetConfig(&sim_jump_machine, &sim_jump_config);
}

void SimController_SetJumpLegSwing(float offset, float kp, float kd, float limit)
{
    mujoco_jump_leg_swing_offset = offset;
    mujoco_jump_leg_swing_kp = kp;
    mujoco_jump_leg_swing_kd = kd;
    mujoco_jump_leg_swing_limit = fabsf(limit);
}

void SimController_SetJumpTpWeights(float lqr_weight,
                                    float split_weight,
                                    float pitch_weight,
                                    float leg_swing_weight)
{
    sim_jump_config.lqr_tp_weight = lqr_weight;
    sim_jump_config.split_tp_weight = split_weight;
    sim_jump_config.pitch_tp_weight = pitch_weight;
    sim_jump_config.leg_swing_tp_weight = leg_swing_weight;
    JumpStateMachine_SetConfig(&sim_jump_machine, &sim_jump_config);
}

void SimController_SetJumpExtend(float extend_l0,
                                 float end_margin,
                                 float extend_rate)
{
    mujoco_jump_extend_l0 = sim_clamp_float(extend_l0,
                                            MIN_LEG_LENGTH,
                                            MAX_LEG_LENGTH);
    mujoco_jump_extend_end_margin = fmaxf(0.0f, end_margin);
    sim_jump_config.extend_l0 = mujoco_jump_extend_l0;
    sim_jump_config.extend_end_margin =
        fminf(mujoco_jump_extend_end_margin,
              fmaxf(0.0f, mujoco_jump_extend_l0 - MIN_LEG_LENGTH));
    sim_jump_config.extend_rate = fmaxf(0.0f, extend_rate);
    JumpStateMachine_SetConfig(&sim_jump_machine, &sim_jump_config);
}

void SimController_SetJumpLandingLegLengths(float preland_l0, float buffer_l0)
{
    sim_jump_config.preland_l0 = sim_clamp_float(preland_l0,
                                                 INIT_LEG_LENGTH,
                                                 MAX_LEG_LENGTH);
    sim_jump_config.buffer_l0 = sim_clamp_float(buffer_l0,
                                                MIN_LEG_LENGTH,
                                                sim_jump_config.preland_l0);
    JumpStateMachine_SetConfig(&sim_jump_machine, &sim_jump_config);
}

void SimController_SetJumpPrelandClearance(float clearance)
{
    sim_jump_config.preland_clearance = fmaxf(0.0f, clearance);
    JumpStateMachine_SetConfig(&sim_jump_machine, &sim_jump_config);
}

void SimController_SetJumpPrelandPidScale(float pid_scale)
{
    sim_jump_config.preland_pid_scale = fmaxf(0.0f, pid_scale);
    JumpStateMachine_SetConfig(&sim_jump_machine, &sim_jump_config);
}

void SimController_SetJumpLandingDynamics(float preland_rate,
                                          float buffer_rate,
                                          float buffer_support_scale,
                                          float buffer_pid_scale)
{
    sim_jump_config.preland_rate = fmaxf(0.0f, preland_rate);
    sim_jump_config.buffer_rate = fmaxf(0.0f, buffer_rate);
    sim_jump_config.buffer_support_scale = fmaxf(0.0f, buffer_support_scale);
    sim_jump_config.buffer_pid_scale = fmaxf(0.0f, buffer_pid_scale);
    JumpStateMachine_SetConfig(&sim_jump_machine, &sim_jump_config);
}

void SimController_SetJumpLandingBalance(float roll_kp,
                                         float roll_kd,
                                         float contact_kp,
                                         float limit)
{
    sim_jump_config.landing_roll_f0_kp = fmaxf(0.0f, roll_kp);
    sim_jump_config.landing_roll_f0_kd = fmaxf(0.0f, roll_kd);
    sim_jump_config.landing_contact_f0_kp = fmaxf(0.0f, contact_kp);
    sim_jump_config.landing_balance_f0_limit = fmaxf(0.0f, limit);
    JumpStateMachine_SetConfig(&sim_jump_machine, &sim_jump_config);
}

void SimController_SetJumpLandingL0Balance(float roll_kp,
                                           float roll_kd,
                                           float limit)
{
    sim_jump_config.landing_roll_l0_kp = fmaxf(0.0f, roll_kp);
    sim_jump_config.landing_roll_l0_kd = fmaxf(0.0f, roll_kd);
    sim_jump_config.landing_balance_l0_limit = fmaxf(0.0f, limit);
    JumpStateMachine_SetConfig(&sim_jump_machine, &sim_jump_config);
}

void SimController_SetJumpLandingClearanceBalance(float kp,
                                                  float rate,
                                                  float limit)
{
    sim_jump_config.landing_clearance_l0_kp = fmaxf(0.0f, kp);
    sim_jump_config.landing_clearance_l0_rate = fmaxf(0.0f, rate);
    sim_jump_config.landing_clearance_l0_limit = fmaxf(0.0f, limit);
    JumpStateMachine_SetConfig(&sim_jump_machine, &sim_jump_config);
}

void SimController_SetJumpWheelRecoverBlendTime(float blend_time)
{
    sim_jump_config.wheel_recover_blend_time = fmaxf(0.0f, blend_time);
    JumpStateMachine_SetConfig(&sim_jump_machine, &sim_jump_config);
}

void SimController_SetAirbornePoseTarget(const float joint_pos[4])
{
    if (joint_pos == 0)
    {
        sim_airborne_pose_valid = 0;
        return;
    }

    memcpy(sim_airborne_joint_target, joint_pos, sizeof(sim_airborne_joint_target));
    sim_airborne_pose_valid = 1;
}

void SimController_SetAirbornePoseGains(float kp, float kd, float torque_limit)
{
    sim_airborne_pose_kp = kp;
    sim_airborne_pose_kd = kd;
    sim_airborne_pose_torque_limit = fabsf(torque_limit);
}

void SimController_SetAirborne(int airborne)
{
    sim_jump_observation.airborne = airborne != 0;
}

void SimController_SetFlightObservation(int airborne,
                                        const float wheel_clearance_m[2],
                                        const float wheel_contact_normal_n[2])
{
    SimController_SetAirborne(airborne);
    if (wheel_clearance_m != 0)
    {
        sim_jump_observation.wheel_clearance[0] = wheel_clearance_m[0];
        sim_jump_observation.wheel_clearance[1] = wheel_clearance_m[1];
    }
    if (wheel_contact_normal_n != 0)
    {
        sim_jump_observation.wheel_contact_normal[0] = wheel_contact_normal_n[0];
        sim_jump_observation.wheel_contact_normal[1] = wheel_contact_normal_n[1];
    }
}

void SimController_SetMode(int mode)
{
    chassis_move.mode = (ChassisMode_e)mode;
}

int SimController_RequestJump(void)
{
    sim_update_jump_body_observation();
    JumpObservation request_observation = sim_jump_observation;
    const int clearly_airborne =
        request_observation.airborne &&
        request_observation.wheel_clearance[0] > 0.03f &&
        request_observation.wheel_clearance[1] > 0.03f;
    if (clearly_airborne)
    {
        return 0;
    }

    request_observation.body_forward_v = 0.0f;
    request_observation.body_lateral_v = 0.0f;
    request_observation.body_z_vel = 0.0f;
    request_observation.airborne = 0;
    request_observation.wheel_clearance[0] = 0.0f;
    request_observation.wheel_clearance[1] = 0.0f;
    request_observation.wheel_contact_normal[0] =
        fmaxf(request_observation.wheel_contact_normal[0],
              sim_jump_config.touchdown_force);
    request_observation.wheel_contact_normal[1] =
        fmaxf(request_observation.wheel_contact_normal[1],
              sim_jump_config.touchdown_force);

    if (!JumpStateMachine_Request(&sim_jump_machine, &request_observation))
    {
        return 0;
    }
    jump_time_l = 0;
    jump_time_r = 0;
    sim_sync_jump_command();
    return 1;
}

int SimController_IsJumping(void)
{
    return chassis_move.jump_flag != 0 || chassis_move.jump_flag2 != 0;
}

float SimController_GetJumpWheelBalanceBlend(void)
{
    const JumpCommand *command = JumpStateMachine_GetCommand(&sim_jump_machine);
    return command != 0 ? command->wheel_balance_blend : 1.0f;
}

void SimController_Step(float dt)
{
    ChassisL_feedback_update();
    ChassisR_feedback_update();

    ChassisR_control_loop();
    ChassisL_control_loop();

    ChassisConsole();
    sim_apply_stand_joint_targets();

    sim_update_jump_body_observation();
    JumpStateMachine_Step(&sim_jump_machine, &sim_jump_observation, dt);
    sim_sync_jump_command();
}

void SimController_GetOutput(SimControllerOutput *output)
{
    if (output == 0)
    {
        return;
    }

    memset(output, 0, sizeof(*output));

    if (chassis_move.start_flag != 1)
    {
        return;
    }

    switch (chassis_move.mode)
    {
    case CHASSIS_STAND_UP:
        output->joint_torque[0] = sim_mit_torque(left.position_set[0], 0.0f, DEBUG_POS_KP, DEBUG_POS_KD, 0.0f,
                                                 chassis_move.joint_motor[0]);
        output->joint_torque[1] = sim_mit_torque(left.position_set[1], 0.0f, DEBUG_POS_KP, DEBUG_POS_KD, 0.0f,
                                                 chassis_move.joint_motor[1]);
        output->joint_torque[2] = sim_mit_torque(right.position_set[0], 0.0f, DEBUG_POS_KP, DEBUG_POS_KD, 0.0f,
                                                 chassis_move.joint_motor[2]);
        output->joint_torque[3] = sim_mit_torque(right.position_set[1], 0.0f, DEBUG_POS_KP, DEBUG_POS_KD, 0.0f,
                                                 chassis_move.joint_motor[3]);
        output->wheel_torque[0] = left.wheel_T;
        output->wheel_torque[1] = right.wheel_T;
        break;

    case CHASSIS_CALIBRATE:
        output->joint_torque[0] = sim_mit_torque(0.0f, left.velocity_set[0], 0.0f, CALIBRATE_VEL_KD, 0.0f,
                                                 chassis_move.joint_motor[0]);
        output->joint_torque[1] = sim_mit_torque(0.0f, left.velocity_set[1], 0.0f, CALIBRATE_VEL_KD, 0.0f,
                                                 chassis_move.joint_motor[1]);
        output->joint_torque[2] = sim_mit_torque(0.0f, right.velocity_set[0], 0.0f, CALIBRATE_VEL_KD, 0.0f,
                                                 chassis_move.joint_motor[2]);
        output->joint_torque[3] = sim_mit_torque(0.0f, right.velocity_set[1], 0.0f, CALIBRATE_VEL_KD, 0.0f,
                                                 chassis_move.joint_motor[3]);
        break;

    case CHASSIS_SAFE:
    {
        // Mirror the real control task: MIT zero pose/velocity targets with
        // torque feedforward from the VMC output.
        const float safe_kp = sim_safe_mit_enabled ? NORMAL_POS_KP : 0.0f;
        const float safe_kd = sim_safe_mit_enabled ? NORMAL_POS_KD : 0.0f;
        output->joint_torque[0] = sim_mit_torque(0.0f, 0.0f, safe_kp, safe_kd, -left.torque_set[1],
                                                 chassis_move.joint_motor[0]);
        output->joint_torque[1] = sim_mit_torque(0.0f, 0.0f, safe_kp, safe_kd, -left.torque_set[0],
                                                 chassis_move.joint_motor[1]);
        output->joint_torque[2] = sim_mit_torque(0.0f, 0.0f, safe_kp, safe_kd, -right.torque_set[0],
                                                 chassis_move.joint_motor[2]);
        output->joint_torque[3] = sim_mit_torque(0.0f, 0.0f, safe_kp, safe_kd, -right.torque_set[1],
                                                 chassis_move.joint_motor[3]);
        output->wheel_torque[0] = left.wheel_T;
        output->wheel_torque[1] = right.wheel_T;
        break;
    }

    case CHASSIS_OFF:
    default:
        output->joint_torque[0] = sim_mit_torque(0.0f, 0.0f, 0.0f, ZERO_FORCE_VEL_KD, 0.0f,
                                                 chassis_move.joint_motor[0]);
        output->joint_torque[1] = sim_mit_torque(0.0f, 0.0f, 0.0f, ZERO_FORCE_VEL_KD, 0.0f,
                                                 chassis_move.joint_motor[1]);
        output->joint_torque[2] = sim_mit_torque(0.0f, 0.0f, 0.0f, ZERO_FORCE_VEL_KD, 0.0f,
                                                 chassis_move.joint_motor[2]);
        output->joint_torque[3] = sim_mit_torque(0.0f, 0.0f, 0.0f, ZERO_FORCE_VEL_KD, 0.0f,
                                                 chassis_move.joint_motor[3]);
        break;
    }

    if (chassis_move.mode == CHASSIS_CALIBRATE)
    {
        output->wheel_torque[0] = left.wheel_T;
        output->wheel_torque[1] = right.wheel_T;
    }
}

void SimController_GetControlBreakdown(SimControllerControlBreakdown *breakdown)
{
    if (breakdown == 0)
    {
        return;
    }

    memset(breakdown, 0, sizeof(*breakdown));
    const vmc_leg_t *legs[2] = {&left, &right};
    for (int leg = 0; leg < 2; ++leg)
    {
        breakdown->l0[leg] = legs[leg]->L0;
        breakdown->l0_set[leg] = chassis_move.leg_set;
        breakdown->d_l0[leg] = legs[leg]->d_L0;
        breakdown->phi0[leg] = legs[leg]->phi0;
        breakdown->d_phi0[leg] = legs[leg]->d_phi0;
        breakdown->theta[leg] = legs[leg]->theta;
        breakdown->d_theta[leg] = legs[leg]->d_theta;
        breakdown->f0_total[leg] = legs[leg]->F0;
        const float cos_theta = cosf(legs[leg]->theta);
        if (fabsf(cos_theta) > 1.0e-4f)
        {
            breakdown->f0_gravity[leg] = BODY_GRAVITY / cos_theta;
            breakdown->f0_leg_pid[leg] =
                legs[leg]->F0 - breakdown->f0_gravity[leg];
        }
        breakdown->tp_total[leg] = legs[leg]->Tp;
    }

    breakdown->joint_torque_f0[0] = -left.j21 * left.F0;
    breakdown->joint_torque_f0[1] = -left.j11 * left.F0;
    breakdown->joint_torque_f0[2] = -right.j11 * right.F0;
    breakdown->joint_torque_f0[3] = -right.j21 * right.F0;
    breakdown->joint_torque_tp[0] = -left.j22 * left.Tp;
    breakdown->joint_torque_tp[1] = -left.j12 * left.Tp;
    breakdown->joint_torque_tp[2] = -right.j12 * right.Tp;
    breakdown->joint_torque_tp[3] = -right.j22 * right.Tp;
    for (int joint = 0; joint < 4; ++joint)
    {
        breakdown->joint_torque_vmc[joint] =
            breakdown->joint_torque_f0[joint] +
            breakdown->joint_torque_tp[joint];
    }
    if (chassis_move.mode == CHASSIS_SAFE && sim_safe_mit_enabled)
    {
        for (int joint = 0; joint < 4; ++joint)
        {
            breakdown->joint_torque_mit[joint] =
                sim_mit_torque(0.0f,
                               0.0f,
                               NORMAL_POS_KP,
                               NORMAL_POS_KD,
                               0.0f,
                               chassis_move.joint_motor[joint]);
        }
    }
    breakdown->tp_weight[0] = 1.0f;
    breakdown->tp_weight[1] = 1.0f;
}
