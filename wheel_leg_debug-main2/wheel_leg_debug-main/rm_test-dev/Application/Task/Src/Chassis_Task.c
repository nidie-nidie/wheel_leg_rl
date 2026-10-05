#include "Remote_Task.h"
#include "PS2_Task.h"
#include "Chassis_Task.h"

chassis_t chassis_move = {
    .mode = CHASSIS_OFF,
    .error_code = 0,
};
vmc_leg_t left;
vmc_leg_t right;

uint8_t left_flag;
uint8_t right_flag;

uint32_t jump_time_r;
uint32_t jump_time_l;

uint32_t CHASS_TIME = 1;
bool chass_is_calibrated = false;

Calibrate_s CALIBRATE = {
    .velocity = {0.0f, 0.0f, 0.0f, 0.0f},
    .stop_time = {0, 0, 0, 0},
    .reached = {false, false, false, false},
    .left_reached = false,
    .right_reached = false,
    .calibrated = false,
};

LegHardstopRefTest_s leg_hardstop_ref_test;

static const int8_t leg_hardstop_ref_direction[4] = {1, -1, -1, 1};

PID_Info_TypeDef stand_up_pid;
PID_Info_TypeDef legl_pid;
PID_Info_TypeDef legr_pid;

PID_Info_TypeDef roll_pid;
PID_Info_TypeDef tp_pid;
PID_Info_TypeDef turn_pid;

float safe_debug_pitch_target;
float safe_debug_pitch_err;
float safe_debug_pitch_tp;
float safe_debug_left_theta_tp;
float safe_debug_right_theta_tp;
float safe_debug_left_tp_cmd;
float safe_debug_right_tp_cmd;
float safe_debug_phi0_bias;
float safe_debug_left_phi0_cmd;
float safe_debug_right_phi0_cmd;
float safe_debug_wheel_pitch_assist;
float safe_debug_phi0_trim;
float safe_debug_phi0_trim_gate;
float safe_debug_phi0_trim_enable;
float safe_debug_left_theta_eq = -SAFE_DEBUG_LEFT_PHI0_BIAS_SIGN * SAFE_PHI0_EQ_TRIM;
float safe_debug_right_theta_eq = -SAFE_DEBUG_RIGHT_PHI0_BIAS_SIGN * SAFE_PHI0_EQ_TRIM;
float safe_debug_left_wheel_lqr_body;
float safe_debug_right_wheel_lqr_body;
float safe_debug_common_wheel_lqr_body;
float safe_debug_pitch_phi0_trim;
float safe_debug_leg_set_ramp;

volatile uint8_t joint_map_test_side;
volatile float joint_map_test_tp_cmd;

static uint8_t safe_debug_leg_set_ramp_active;
static uint32_t safe_debug_leg_set_ramp_last_tick;

uint32_t CHASS_FSM_TIME = 3; // 3ms的底盘控制周期，对齐底盘控制频率

void mySaturate(float *in, float min, float max)
{
    if (*in < min)
    {
        *in = min;
    }
    else if (*in > max)
    {
        *in = max;
    }
}

void ChassisApplySafePitchTp(vmc_leg_t *leg, int8_t side_sign)
{
#if (SAFE_USE_EXPLICIT_PITCH_TP)
    float pitch_tp;
    float theta_tp;
    float tp_cmd;

    safe_debug_pitch_target = SAFE_PITCH_TARGET;
    safe_debug_pitch_err = SAFE_PITCH_TARGET - INS.Pitch;

    pitch_tp = SAFE_PITCH_TP_KP * safe_debug_pitch_err - SAFE_PITCH_TP_KD * INS.Gyro[1];
    mySaturate(&pitch_tp, -SAFE_PITCH_TP_LIMIT, SAFE_PITCH_TP_LIMIT);
    safe_debug_pitch_tp = pitch_tp;

    theta_tp = SAFE_THETA_TP_KP * (0.0f - leg->theta) + SAFE_THETA_TP_KD * (0.0f - leg->d_theta);
    mySaturate(&theta_tp, -SAFE_THETA_TP_LIMIT, SAFE_THETA_TP_LIMIT);

    tp_cmd = theta_tp + ((float)side_sign) * SAFE_PITCH_TP_SIGN * pitch_tp + chassis_move.leg_tp;
    tp_cmd *= SAFE_DEBUG_TP_CMD_SCALE;
    leg->Tp = tp_cmd;

    if (leg == &left)
    {
        safe_debug_left_theta_tp = theta_tp;
        safe_debug_left_tp_cmd = tp_cmd;
    }
    else
    {
        safe_debug_right_theta_tp = theta_tp;
        safe_debug_right_tp_cmd = tp_cmd;
    }
#else
    (void)leg;
    (void)side_sign;
#endif
}

float ChassisCalcSafeWheelTorque(float wheel_velocity, float wheel_direction)
{
    float wheel_t = -SAFE_WHEEL_DAMPING_KD * wheel_velocity;
    float pitch_assist = 0.0f;

#if (SAFE_DEBUG_WHEEL_PITCH_ASSIST_ENABLE)
    pitch_assist = SAFE_DEBUG_WHEEL_PITCH_SIGN *
                   (SAFE_DEBUG_WHEEL_PITCH_KP * (SAFE_PITCH_TARGET - INS.Pitch) -
                    SAFE_DEBUG_WHEEL_PITCH_KD * INS.Gyro[1]);
    mySaturate(&pitch_assist,
               -SAFE_DEBUG_WHEEL_PITCH_LIMIT,
               SAFE_DEBUG_WHEEL_PITCH_LIMIT);
#endif

    safe_debug_wheel_pitch_assist = pitch_assist;
    wheel_t += wheel_direction * pitch_assist;
    mySaturate(&wheel_t, -SAFE_WHEEL_T_LIMIT, SAFE_WHEEL_T_LIMIT);
    return wheel_t;
}

static bool ChassisSafePhi0TrimGate(void)
{
    float torque_limit = MAX_JOINT_TORQUE - SAFE_PHI0_TRIM_TORQUE_MARGIN;
    float wheel_limit = SAFE_WHEEL_T_LIMIT - SAFE_PHI0_TRIM_WHEEL_T_MARGIN;

    if (torque_limit < 0.0f)
    {
        torque_limit = 0.0f;
    }
    if (wheel_limit < 0.0f)
    {
        wheel_limit = 0.0f;
    }

    if (chassis_move.mode != CHASSIS_SAFE || chassis_move.start_flag == 0U)
    {
        return false;
    }
    if (chassis_move.recover_flag != 0U || chassis_move.jump_flag != 0U || chassis_move.jump_flag2 != 0U)
    {
        return false;
    }
    if (fabs(chassis_move.v_filter) > SAFE_PHI0_TRIM_MAX_ABS_V)
    {
        return false;
    }
    if (fabs(INS.Gyro[1]) > SAFE_PHI0_TRIM_MAX_ABS_PITCH_RATE)
    {
        return false;
    }
    if (left.L0 < SAFE_PHI0_TRIM_MIN_L0 || left.L0 > SAFE_PHI0_TRIM_MAX_L0 ||
        right.L0 < SAFE_PHI0_TRIM_MIN_L0 || right.L0 > SAFE_PHI0_TRIM_MAX_L0)
    {
        return false;
    }
    if (fabs(left.wheel_T) > wheel_limit || fabs(right.wheel_T) > wheel_limit)
    {
        return false;
    }
    if (fabs(left.torque_set[0]) > torque_limit || fabs(left.torque_set[1]) > torque_limit ||
        fabs(right.torque_set[0]) > torque_limit || fabs(right.torque_set[1]) > torque_limit)
    {
        return false;
    }

    return true;
}

static void ChassisUpdateSafeLegSetRamp(void)
{
#if (SAFE_DEBUG_LEG_SET_RAMP_ENABLE)
    uint32_t now = HAL_GetTick();
    float target_leg_set = INIT_LEG_LENGTH;

    if (chassis_move.mode != CHASSIS_SAFE || chassis_move.start_flag == 0U)
    {
        safe_debug_leg_set_ramp_active = 0U;
        safe_debug_leg_set_ramp = chassis_move.leg_set;
        safe_debug_leg_set_ramp_last_tick = now;
        return;
    }

    if (safe_debug_leg_set_ramp_active == 0U)
    {
        float measured_leg_set = 0.5f * (left.L0 + right.L0);

        if (isnan(measured_leg_set) ||
            measured_leg_set < MIN_LEG_LENGTH ||
            measured_leg_set > MAX_LEG_LENGTH)
        {
            measured_leg_set = chassis_move.leg_set;
        }

        safe_debug_leg_set_ramp = measured_leg_set;
        mySaturate(&safe_debug_leg_set_ramp, MIN_LEG_LENGTH, MAX_LEG_LENGTH);
        safe_debug_leg_set_ramp_last_tick = now;
        safe_debug_leg_set_ramp_active = 1U;
    }
    else
    {
        float dt = ((float)(now - safe_debug_leg_set_ramp_last_tick)) / 1000.0f;
        float max_step;
        float delta;

        if (dt > 0.05f)
        {
            dt = 0.05f;
        }

        max_step = SAFE_DEBUG_LEG_SET_RAMP_RATE * dt;
        delta = target_leg_set - safe_debug_leg_set_ramp;
        mySaturate(&delta, -max_step, max_step);
        safe_debug_leg_set_ramp += delta;
        mySaturate(&safe_debug_leg_set_ramp, MIN_LEG_LENGTH, MAX_LEG_LENGTH);
        safe_debug_leg_set_ramp_last_tick = now;
    }

    chassis_move.leg_set = safe_debug_leg_set_ramp;
    chassis_move.last_leg_set = chassis_move.leg_set;
#else
    safe_debug_leg_set_ramp = chassis_move.leg_set;
#endif
}

void UpdateCalibrateStatus(void)
{
    if ((chassis_move.mode == CHASSIS_CALIBRATE) &&
        fabs(chassis_move.joint_motor[0]->Data.Position) < ZERO_POS_THRESHOLD &&
        fabs(chassis_move.joint_motor[1]->Data.Position) < ZERO_POS_THRESHOLD &&
        fabs(chassis_move.joint_motor[2]->Data.Position) < ZERO_POS_THRESHOLD &&
        fabs(chassis_move.joint_motor[3]->Data.Position) < ZERO_POS_THRESHOLD)
    {
        CALIBRATE.calibrated = true;
    }

    // 校准模式的相关反馈数据
    uint32_t now = HAL_GetTick();
    if (chassis_move.mode == CHASSIS_CALIBRATE)
    {
        for (uint8_t i = 0; i < 4; i++)
        {
            CALIBRATE.velocity[i] = chassis_move.joint_motor[i]->Data.Velocity;
            if (CALIBRATE.velocity[i] > CALIBRATE_STOP_VELOCITY)
            { // 速度大于阈值时重置计时
                CALIBRATE.reached[i] = false;
                CALIBRATE.stop_time[i] = now;
            }
            else
            {
                if (now - CALIBRATE.stop_time[i] > CALIBRATE_STOP_TIME)
                {
                    CALIBRATE.reached[i] = true;
                }
            }
        }
    }
}

/**
 * @brief          异常处理
 * @param[in]      none
 * @retval         none
 */
static void LegHardstopRefTest_StopAll(void)
{
    for (uint8_t i = 0U; i < 4U; i++)
    {
        leg_hardstop_ref_test.velocity_cmd[i] = 0.0f;
        leg_hardstop_ref_test.torque_cmd[i] = 0.0f;
    }
    leg_hardstop_ref_test.active = 0U;
}

static void LegHardstopRefTest_CaptureSide(uint8_t side)
{
    if (side == 0U)
    {
        leg_hardstop_ref_test.snapshot_pitch[0] = INS.Pitch;
        leg_hardstop_ref_test.snapshot_phi0[0] = left.phi0;
        leg_hardstop_ref_test.snapshot_theta[0] = left.theta;
        leg_hardstop_ref_test.snapshot_L0[0] = left.L0;
        leg_hardstop_ref_test.snapshot_joint_pos[0] = chassis_move.joint_motor[0]->Data.Position;
        leg_hardstop_ref_test.snapshot_joint_pos[1] = chassis_move.joint_motor[1]->Data.Position;
        leg_hardstop_ref_test.snapshot_valid_mask |= 0x01U;
    }
    else
    {
        leg_hardstop_ref_test.snapshot_pitch[1] = INS.Pitch;
        leg_hardstop_ref_test.snapshot_phi0[1] = right.phi0;
        leg_hardstop_ref_test.snapshot_theta[1] = right.theta;
        leg_hardstop_ref_test.snapshot_L0[1] = right.L0;
        leg_hardstop_ref_test.snapshot_joint_pos[2] = chassis_move.joint_motor[2]->Data.Position;
        leg_hardstop_ref_test.snapshot_joint_pos[3] = chassis_move.joint_motor[3]->Data.Position;
        leg_hardstop_ref_test.snapshot_valid_mask |= 0x02U;
    }
}

void LegHardstopRefTest_Update(void)
{
#if (LEG_HARDSTOP_REF_TEST_ENABLE)
    static uint8_t last_start_flag = 0U;
    uint32_t now = HAL_GetTick();

    if (chassis_move.start_flag == 0U)
    {
        LegHardstopRefTest_StopAll();
        last_start_flag = 0U;
        return;
    }

    if (last_start_flag == 0U)
    {
        leg_hardstop_ref_test.active = 1U;
        leg_hardstop_ref_test.done = 0U;
        leg_hardstop_ref_test.fault = 0U;
        leg_hardstop_ref_test.reached_mask = 0U;
        leg_hardstop_ref_test.snapshot_valid_mask = 0U;
        leg_hardstop_ref_test.start_tick = now;
        for (uint8_t i = 0U; i < 4U; i++)
        {
            leg_hardstop_ref_test.stop_candidate_tick[i] = now;
            leg_hardstop_ref_test.velocity_cmd[i] = 0.0f;
            leg_hardstop_ref_test.torque_cmd[i] =
                (float)leg_hardstop_ref_direction[i] * LEG_HARDSTOP_REF_TEST_TORQUE;
            leg_hardstop_ref_test.start_joint_pos[i] =
                (chassis_move.joint_motor[i] != 0) ?
                chassis_move.joint_motor[i]->Data.Position : 0.0f;
            leg_hardstop_ref_test.snapshot_joint_pos[i] = 0.0f;
        }
        for (uint8_t side = 0U; side < 2U; side++)
        {
            leg_hardstop_ref_test.snapshot_pitch[side] = 0.0f;
            leg_hardstop_ref_test.snapshot_phi0[side] = 0.0f;
            leg_hardstop_ref_test.snapshot_theta[side] = 0.0f;
            leg_hardstop_ref_test.snapshot_L0[side] = 0.0f;
        }
    }
    last_start_flag = 1U;

    if (leg_hardstop_ref_test.active == 0U)
    {
        return;
    }

    if ((now - leg_hardstop_ref_test.start_tick) > LEG_HARDSTOP_REF_TEST_TIMEOUT_MS)
    {
        leg_hardstop_ref_test.fault = 1U;
        LegHardstopRefTest_StopAll();
        chassis_move.start_flag = 0U;
        chassis_move.mode = CHASSIS_OFF;
        return;
    }

    for (uint8_t i = 0U; i < 4U; i++)
    {
        DM_Motor_Info_Typedef *motor = chassis_move.joint_motor[i];
        uint8_t bit = (uint8_t)(1U << i);

        if ((motor == 0) || ((now - motor->last_fdb_time) > LEG_HARDSTOP_REF_TEST_MAX_FDB_AGE_MS))
        {
            leg_hardstop_ref_test.fault = 2U;
            LegHardstopRefTest_StopAll();
            chassis_move.start_flag = 0U;
            chassis_move.mode = CHASSIS_OFF;
            return;
        }
        if (fabsf(motor->Data.Torque) > LEG_HARDSTOP_REF_TEST_MAX_TORQUE)
        {
            leg_hardstop_ref_test.fault = 3U;
            LegHardstopRefTest_StopAll();
            chassis_move.start_flag = 0U;
            chassis_move.mode = CHASSIS_OFF;
            return;
        }

        if ((leg_hardstop_ref_test.reached_mask & bit) != 0U)
        {
            leg_hardstop_ref_test.velocity_cmd[i] = 0.0f;
            leg_hardstop_ref_test.torque_cmd[i] =
                (float)leg_hardstop_ref_direction[i] * LEG_HARDSTOP_REF_TEST_HOLD_TORQUE;
            continue;
        }

        leg_hardstop_ref_test.velocity_cmd[i] = 0.0f;
        leg_hardstop_ref_test.torque_cmd[i] =
            (float)leg_hardstop_ref_direction[i] * LEG_HARDSTOP_REF_TEST_TORQUE;

        if ((now - leg_hardstop_ref_test.start_tick) < LEG_HARDSTOP_REF_TEST_START_GRACE_MS)
        {
            leg_hardstop_ref_test.stop_candidate_tick[i] = now;
        }
        else if ((fabsf(motor->Data.Position - leg_hardstop_ref_test.start_joint_pos[i]) >=
                  LEG_HARDSTOP_REF_TEST_MIN_TRAVEL) &&
                 (fabsf(motor->Data.Velocity) < LEG_HARDSTOP_REF_TEST_STOP_VELOCITY) &&
                 (fabsf(motor->Data.Torque) >= LEG_HARDSTOP_REF_TEST_MIN_STOP_TORQUE))
        {
            if ((now - leg_hardstop_ref_test.stop_candidate_tick[i]) >= LEG_HARDSTOP_REF_TEST_STOP_TIME_MS)
            {
                leg_hardstop_ref_test.reached_mask |= bit;
            }
        }
        else
        {
            leg_hardstop_ref_test.stop_candidate_tick[i] = now;
        }
    }

    if (((leg_hardstop_ref_test.reached_mask & 0x03U) == 0x03U) &&
        ((leg_hardstop_ref_test.snapshot_valid_mask & 0x01U) == 0U))
    {
        LegHardstopRefTest_CaptureSide(0U);
        leg_hardstop_ref_test.velocity_cmd[0] = 0.0f;
        leg_hardstop_ref_test.velocity_cmd[1] = 0.0f;
        leg_hardstop_ref_test.torque_cmd[0] = 0.0f;
        leg_hardstop_ref_test.torque_cmd[1] = 0.0f;
    }
    if (((leg_hardstop_ref_test.reached_mask & 0x0CU) == 0x0CU) &&
        ((leg_hardstop_ref_test.snapshot_valid_mask & 0x02U) == 0U))
    {
        LegHardstopRefTest_CaptureSide(1U);
        leg_hardstop_ref_test.velocity_cmd[2] = 0.0f;
        leg_hardstop_ref_test.velocity_cmd[3] = 0.0f;
        leg_hardstop_ref_test.torque_cmd[2] = 0.0f;
        leg_hardstop_ref_test.torque_cmd[3] = 0.0f;
    }

    if (leg_hardstop_ref_test.snapshot_valid_mask == 0x03U)
    {
        leg_hardstop_ref_test.done = 1U;
        LegHardstopRefTest_StopAll();
        chassis_move.start_flag = 0U;
        chassis_move.mode = CHASSIS_OFF;
    }
#else
    LegHardstopRefTest_StopAll();
#endif
}

void ChassisHandleException(void)
{
    if ((ENABLE_ALARM_RC_OFFLINE && GetRcOffline() || (ENABLE_ALARM_PS2_OFFLINE && ps2_lost)))
    {
        chassis_move.error_code |= DBUS_ERROR_OFFSET;
    }
    else
    {
        chassis_move.error_code &= ~DBUS_ERROR_OFFSET;
    }

    for (uint8_t i = 0; i < 4; i++)
    {
        if (fabs(chassis_move.joint_motor[i]->Data.Torque) > MAX_TORQUE_PROTECT)
        {
            chassis_move.error_code |= JOINT_ERROR_OFFSET;
            break;
        }
    }

    if ((chassis_move.mode == CHASSIS_OFF || chassis_move.mode == CHASSIS_SAFE) &&
        fabs(stand_up_pid.Param.LimitOutput) != 0.0f)
    {
        PID_Calc_Clear(&stand_up_pid);
    }
}

/**
 * @brief          设置模式
 * @param[in]      none
 * @retval         none
 */
void ChassisSetMode(void)
{
    if (chassis_move.error_code & DBUS_ERROR_OFFSET)
    { // 遥控器出错时的状态处理
        chassis_move.mode = CHASSIS_SAFE;
        ChassisUpdateSafeLegSetRamp();
        return;
    }

    if (chassis_move.error_code & JOINT_ERROR_OFFSET)
    { // 关节电机出错时的状态处理
        chassis_move.mode = CHASSIS_SAFE;
        ChassisUpdateSafeLegSetRamp();
        return;
    }

    if (chassis_move.mode == CHASSIS_CALIBRATE)
    { // 校准完成后才退出校准
        if (CALIBRATE.left_reached && CALIBRATE.right_reached && CALIBRATE.calibrated)
        {
            chassis_move.mode = CHASSIS_SAFE;
        }else{
            return; // 校准未完成前不允许切出校准模式
        }
    }

    if (CALIBRATE.toggle)
    { // 切入底盘校准
        CALIBRATE.toggle = false;
        chassis_move.mode = CHASSIS_CALIBRATE;
        CALIBRATE.calibrated = false;

        uint32_t now = HAL_GetTick();
        for (uint8_t i = 0; i < 4; i++)
        {
            CALIBRATE.reached[i] = false;
            CALIBRATE.stop_time[i] = now;
        }

        return;
    }

    if (chassis_move.recover_flag == 1) // 倒地自恢复状态
    {
        chassis_move.mode = CHASSIS_OFF_HOOK;
    }

    ChassisUpdateSafeLegSetRamp();
}

static void UpdateLegPositionSet(void)
{
    float phi1_phi4_l[2], phi1_phi4_r[2];
    float phi0_bias = 0.0f;
    float left_phi0_cmd = M_PI_2;
    float right_phi0_cmd = M_PI_2;

    safe_debug_pitch_target = SAFE_PITCH_TARGET;
    safe_debug_pitch_err = SAFE_PITCH_TARGET - INS.Pitch;

    if (chassis_move.mode == CHASSIS_SAFE)
    {
#if (SAFE_STATIC_PHI0_TRIM_ENABLE)
        bool trim_gate = ChassisSafePhi0TrimGate();

        safe_debug_phi0_trim_enable = 1.0f;
        safe_debug_phi0_trim_gate = trim_gate ? 1.0f : 0.0f;

        if (chassis_move.start_flag == 0U)
        {
            safe_debug_phi0_trim = 0.0f;
        }
        else if (trim_gate)
        {
            safe_debug_phi0_trim += SAFE_PHI0_TRIM_SIGN *
                                    SAFE_PHI0_TRIM_KI *
                                    safe_debug_pitch_err *
                                    ((float)CHASS_FSM_TIME / 1000.0f);
            mySaturate(&safe_debug_phi0_trim, -SAFE_PHI0_TRIM_LIMIT, SAFE_PHI0_TRIM_LIMIT);
        }

#if (SAFE_STATIC_PITCH_PHI0_TRIM_ENABLE)
        if (chassis_move.start_flag == 0U)
        {
            safe_debug_pitch_phi0_trim = 0.0f;
        }
        else
        {
            float pitch_phi0_trim_target =
                SAFE_DEBUG_PITCH_TO_PHI0_SIGN *
                SAFE_STATIC_PITCH_PHI0_TRIM_KP *
                safe_debug_pitch_err;
            float pitch_phi0_trim_delta;
            float pitch_phi0_trim_step =
                SAFE_STATIC_PITCH_PHI0_TRIM_RATE_LIMIT *
                ((float)CHASS_FSM_TIME / 1000.0f);

            mySaturate(&pitch_phi0_trim_target,
                       -SAFE_STATIC_PITCH_PHI0_TRIM_LIMIT,
                       SAFE_STATIC_PITCH_PHI0_TRIM_LIMIT);

            pitch_phi0_trim_delta = pitch_phi0_trim_target - safe_debug_pitch_phi0_trim;
            mySaturate(&pitch_phi0_trim_delta,
                       -pitch_phi0_trim_step,
                       pitch_phi0_trim_step);
            safe_debug_pitch_phi0_trim += pitch_phi0_trim_delta;
        }
#else
        safe_debug_pitch_phi0_trim = 0.0f;
#endif

        phi0_bias = SAFE_PHI0_EQ_TRIM + safe_debug_phi0_trim + safe_debug_pitch_phi0_trim;
#if (FULL_LQR_THETA_ZERO_OFFSET_TEST_ENABLE)
        // Treat the measured non-zero symmetric theta as a coordinate zero
        // offset. Keep MIT position IK and the LQR theta error centered on the
        // same raw pose during this experiment.
        phi0_bias = FULL_LQR_THETA_ZERO_OFFSET_TEST_TRIM;
#endif
        mySaturate(&phi0_bias, -SAFE_PHI0_TRIM_LIMIT, SAFE_PHI0_TRIM_LIMIT);

        left_phi0_cmd = M_PI_2 + SAFE_DEBUG_LEFT_PHI0_BIAS_SIGN * phi0_bias;
        right_phi0_cmd = M_PI_2 + SAFE_DEBUG_RIGHT_PHI0_BIAS_SIGN * phi0_bias;
#else
        safe_debug_phi0_trim_enable = 0.0f;
        safe_debug_phi0_trim_gate = 0.0f;
        safe_debug_phi0_trim = 0.0f;
        safe_debug_pitch_phi0_trim = 0.0f;

        phi0_bias = SAFE_DEBUG_PITCH_TO_PHI0_SIGN *
                    SAFE_DEBUG_PITCH_TO_PHI0_GAIN *
                    (SAFE_PITCH_TARGET - INS.Pitch);
        mySaturate(&phi0_bias, -SAFE_DEBUG_PITCH_TO_PHI0_LIMIT, SAFE_DEBUG_PITCH_TO_PHI0_LIMIT);

        left_phi0_cmd = M_PI_2 + SAFE_DEBUG_LEFT_PHI0_BIAS_SIGN * phi0_bias;
        right_phi0_cmd = M_PI_2 + SAFE_DEBUG_RIGHT_PHI0_BIAS_SIGN * phi0_bias;
#endif
    }
    else
    {
        safe_debug_phi0_trim_enable = 0.0f;
        safe_debug_phi0_trim_gate = 0.0f;
        safe_debug_phi0_trim = 0.0f;
        safe_debug_pitch_phi0_trim = 0.0f;
    }
    safe_debug_phi0_bias = phi0_bias;
    safe_debug_left_phi0_cmd = left_phi0_cmd;
    safe_debug_right_phi0_cmd = right_phi0_cmd;
    safe_debug_left_theta_eq = M_PI_2 - left_phi0_cmd;
    safe_debug_right_theta_eq = M_PI_2 - right_phi0_cmd;

    CalcPhi1AndPhi4(left_phi0_cmd, chassis_move.leg_set, phi1_phi4_l);
    CalcPhi1AndPhi4(right_phi0_cmd, chassis_move.leg_set, phi1_phi4_r);

    if (!(isnan(phi1_phi4_l[0]) || isnan(phi1_phi4_l[1]) || isnan(phi1_phi4_r[0]) ||
          isnan(phi1_phi4_r[1])))
    {
        left.position_set[0] =
            -theta_transform(phi1_phi4_l[1], -J0_ANGLE_OFFSET, J0_DIRECTION, 1);
        left.position_set[1] =
            -theta_transform(phi1_phi4_l[0], -J1_ANGLE_OFFSET, J1_DIRECTION, 1);
        right.position_set[0] =
            -theta_transform(phi1_phi4_r[0], -J2_ANGLE_OFFSET, J2_DIRECTION, 1);
        right.position_set[1] =
            -theta_transform(phi1_phi4_r[1], -J3_ANGLE_OFFSET, J3_DIRECTION, 1);
    }
}

/**
 * @brief          计算控制量
 * @param[in]      none
 * @retval         none
 */
void ChassisConsole(void)
{
    switch (chassis_move.mode)
    {
    case CHASSIS_CALIBRATE:
    {
        ConsoleCalibrate();
    }
    break;
    case CHASSIS_OFF_HOOK:
    {
        // ConsoleOffHook();
    }
    break;
    case CHASSIS_STAND_UP:
    {
        ConsoleStandUp();
    }
    break;
    case CHASSIS_SAFE:
    {
        ChassisUpdateSafeLegSetRamp();
        UpdateLegPositionSet();
    }
    break;
    case CHASSIS_OFF:
    default:
    {
        // ConsoleZeroForce();
    }
    }
}

/**
 * @brief  Console function for chassis calibration mode, 计算校准时的电机控制量
 */
void ConsoleCalibrate(void)
{
    left.velocity_set[1] = -CALIBRATE_VELOCITY;
    left.velocity_set[0] = CALIBRATE_VELOCITY;
    right.velocity_set[1] = CALIBRATE_VELOCITY;
    right.velocity_set[0] = -CALIBRATE_VELOCITY;

    left.wheel_T = 0;
    right.wheel_T = 0;
}

/**
 * @brief  Console function for chassis stand-up mode, 计算初始站起时的电机控制量
 */
void ConsoleStandUp(void)
{
    // ===腿部位置控制===
    UpdateLegPositionSet();

    left.wheel_T = 0;
    right.wheel_T = 0;

    // right.position_set[0] = -0.4223f; // 站立时的固定角度，暂时先写死
    // right.position_set[1] = 0.4223f;  // 站立时的固定角度，暂时先写死

    // 检测设定角度是否超过电机角度限制
    // left.position_set[0] =
    //     fp32_constrain(left.position_set[0], MIN_J0_ANGLE, MAX_J0_ANGLE);
    // left.position_set[1] =
    //     fp32_constrain(left.position_set[1], MIN_J1_ANGLE, MAX_J1_ANGLE);
    // right.position_set[0] =
    //     fp32_constrain(right.position_set[0], MIN_J2_ANGLE, MAX_J2_ANGLE);
    // right.position_set[1] =
    //     fp32_constrain(right.position_set[1], MIN_J3_ANGLE, MAX_J3_ANGLE);

    // ===驱动轮pid控制===
    float feedforward = -220;
    PID_Calculate(&stand_up_pid, 0, INS.Pitch); // 以IMU Pitch角为反馈，0为目标，计算站起pid控制量
    left.wheel_T = (feedforward + stand_up_pid.Output) * W0_DIRECTION;
    right.wheel_T = (feedforward + stand_up_pid.Output) * W1_DIRECTION;
}
