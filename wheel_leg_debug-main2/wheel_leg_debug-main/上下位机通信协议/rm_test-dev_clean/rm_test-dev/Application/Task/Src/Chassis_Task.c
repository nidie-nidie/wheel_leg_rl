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

PID_Info_TypeDef stand_up_pid;
PID_Info_TypeDef legl_pid;
PID_Info_TypeDef legr_pid;

PID_Info_TypeDef roll_pid;
PID_Info_TypeDef tp_pid;
PID_Info_TypeDef turn_pid;

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
        return;
    }

    if (chassis_move.error_code & JOINT_ERROR_OFFSET)
    { // 关节电机出错时的状态处理
        chassis_move.mode = CHASSIS_SAFE;
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
    case CHASSIS_OFF:
    case CHASSIS_SAFE:
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
    float phi1_phi4_l[2], phi1_phi4_r[2];
    CalcPhi1AndPhi4(M_PI_2, chassis_move.leg_set, phi1_phi4_l);
    CalcPhi1AndPhi4(M_PI_2, chassis_move.leg_set, phi1_phi4_r);

    // 当解算出的角度正常时，设置目标角度
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