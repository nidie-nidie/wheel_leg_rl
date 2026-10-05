#include "ChassisR_Task.h"

void ChassisR_task(void)
{
    while (INS.ins_flag == 0)
    { // 等待加速度收敛
        osDelay(1);
    }

    ChassisR_init();  // 初始化右边两个关节电机和右边轮毂电机的id和控制模式、初始化腿部
    Pensation_init(); // 补偿pid初始化

    while (1)
    {
        // 处理异常
        ChassisHandleException();
        // 设置底盘模式
        ChassisSetMode();
        // 底盘状态量
        UpdateCalibrateStatus();

        ChassisR_feedback_update(); // 更新数据

        ChassisR_control_loop(); // 控制计算

        // 计算控制量
        ChassisConsole();

#if (LEG_HARDSTOP_REF_TEST_ENABLE)
        LegHardstopRefTest_Update();
#endif

        if (chassis_move.start_flag == 1)
        {
            switch (chassis_move.mode)
            {
            case CHASSIS_STAND_UP:
            {
                // DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[2], 0, 0, NORMAL_POS_KP, NORMAL_POS_KD, 0);
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[2], right.position_set[0], 0, DEBUG_POS_KP, DEBUG_POS_KD, 0);
                osDelay(CHASS_TIME);
                // DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[3], 0, 0, NORMAL_POS_KP, NORMAL_POS_KD, 0);
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[3], right.position_set[1], 0, DEBUG_POS_KP, DEBUG_POS_KD, 0);
                osDelay(CHASS_TIME);
                LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[1], 0, right.wheel_T);
                osDelay(CHASS_TIME);
            }
            break;
            case CHASSIS_CALIBRATE:
            {
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[2], 0, right.velocity_set[0], 0, CALIBRATE_VEL_KD, 0);
                osDelay(CHASS_TIME);
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[3], 0, right.velocity_set[1], 0, CALIBRATE_VEL_KD, 0);
                osDelay(CHASS_TIME);
                LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[1], 0, right.wheel_T);
                osDelay(CHASS_TIME);

                if (!LEG_HARDSTOP_REF_TEST_ENABLE &&
                    CALIBRATE.reached[2] && CALIBRATE.reached[3] && !CALIBRATE.right_reached)
                {
                    CALIBRATE.right_reached = true; // 只校准一次，避免误触
                    DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[2], DM_Motor_Save_Zero_Position);
                    DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[2], DM_Motor_Enable);
                    osDelay(CHASS_TIME);
                    DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[3], DM_Motor_Save_Zero_Position);
                    DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[3], DM_Motor_Enable);
                    osDelay(CHASS_TIME * 2);
                }
            }
            break;
            case CHASSIS_SAFE: // 正常车子状态，执行控制量
            {
#if (LEG_HARDSTOP_REF_TEST_ENABLE)
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[2], 0,
                                       leg_hardstop_ref_test.velocity_cmd[2], 0,
                                       LEG_HARDSTOP_REF_TEST_VEL_KD,
                                       leg_hardstop_ref_test.torque_cmd[2]);
                osDelay(CHASS_TIME);
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[3], 0,
                                       leg_hardstop_ref_test.velocity_cmd[3], 0,
                                       LEG_HARDSTOP_REF_TEST_VEL_KD,
                                       leg_hardstop_ref_test.torque_cmd[3]);
#elif (JOINT_MAP_TEST_ENABLE)
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[2], 0, 0,
                                       JOINT_MAP_TEST_MIT_KP, JOINT_MAP_TEST_MIT_KD, -right.torque_set[0]);
                osDelay(CHASS_TIME);
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[3], 0, 0,
                                       JOINT_MAP_TEST_MIT_KP, JOINT_MAP_TEST_MIT_KD, -right.torque_set[1]);
#elif (CHASSIS_FULL_LQR_ENABLE)
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[2], right.position_set[0], 0,
                                       SAFE_DEBUG_MIT_POS_KP, SAFE_DEBUG_MIT_POS_KD, -right.torque_set[0]);
                osDelay(CHASS_TIME);
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[3], right.position_set[1], 0,
                                       SAFE_DEBUG_MIT_POS_KP, SAFE_DEBUG_MIT_POS_KD, -right.torque_set[1]);
#else
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[2], right.position_set[0], 0, SAFE_DEBUG_MIT_POS_KP, SAFE_DEBUG_MIT_POS_KD, -right.torque_set[0]);
                osDelay(CHASS_TIME);
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[3], right.position_set[1], 0, SAFE_DEBUG_MIT_POS_KP, SAFE_DEBUG_MIT_POS_KD, -right.torque_set[1]);
#endif
                osDelay(CHASS_TIME);
#if (LEG_HARDSTOP_REF_TEST_ENABLE)
                LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[1], 0, 0.0f);
#else
                LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[1], 0, right.wheel_T);
#endif
                osDelay(CHASS_TIME);
            }
            break;
            default:
            {
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[2], 0, 0, 0, ZERO_FORCE_VEL_KD, 0);
                osDelay(CHASS_TIME);
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[3], 0, 0, 0, ZERO_FORCE_VEL_KD, 0);
                osDelay(CHASS_TIME);
                LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[1], 0, 0);
                // LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[1], 0, 1);
                osDelay(CHASS_TIME);
            }
            }
        }
        else
        {
            // 失能状态，电机停止
            DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[2], 0, 0, 0, 0, 0);
            osDelay(CHASS_TIME);
            DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[3], 0, 0, 0, 0, 0);
            osDelay(CHASS_TIME);
            LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[1], 0, 0);
            osDelay(CHASS_TIME);
        }
    }
}

void ChassisR_init(void)
{
    chassis_move.joint_motor[2] = &DM_8009_Motor[2];
    chassis_move.joint_motor[3] = &DM_8009_Motor[3];

    chassis_move.wheel_motor[1] = &LK_9025_Motor[1];

    // pid init
    float Stand_Up_PID_Param[PID_PARAMETER_NUM] = {KP_CHASSIS_STAND_UP, KI_CHASSIS_STAND_UP, KD_CHASSIS_STAND_UP, 0, 0, MAX_IOUT_CHASSIS_STAND_UP, MAX_OUT_CHASSIS_STAND_UP};
    PID_Init(&stand_up_pid, PID_POSITION, Stand_Up_PID_Param);

    // pid init
    float LegR_PID_Param[PID_PARAMETER_NUM] = {KP_CHASSIS_LEG_LENGTH_LENGTH, KI_CHASSIS_LEG_LENGTH_LENGTH, KD_CHASSIS_LEG_LENGTH_LENGTH, ALPHA_LEG_LENGTH_LENGTH, 0, MAX_IOUT_CHASSIS_LEG_LENGTH_LENGTH, MAX_OUT_CHASSIS_LEG_LENGTH_LENGTH};
    PID_Init(&legr_pid, PID_POSITION, LegR_PID_Param);

    VMC_init(&right); // 给杆长赋值

    for (int j = 0; j < 10; j++)
    {
        DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[2], DM_Motor_Enable);
        osDelay(1);
    }
    for (int j = 0; j < 10; j++)
    {
        DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[3], DM_Motor_Enable);
        osDelay(1);
    }
    for (int j = 0; j < 10; j++)
    {
        LK_Motor_Command(&FDCAN3_TxFrame, chassis_move.wheel_motor[1], LK_Motor_Enable); // 右边轮毂电机
        osDelay(1);
    }

    // // 保存零点位置
    // DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[2], DM_Motor_Save_Zero_Position);
    // osDelay(10);
    // DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[3], DM_Motor_Save_Zero_Position);
    // osDelay(10);
}

void Pensation_init(void)
{ // 补偿pid初始化：横滚角补偿、防劈叉补偿、偏航角补偿
    float Roll_PID_Param[PID_PARAMETER_NUM] = {KP_CHASSIS_ROLL_ANGLE, KI_CHASSIS_ROLL_ANGLE, KD_CHASSIS_ROLL_ANGLE, 0, 0, MAX_IOUT_CHASSIS_ROLL_ANGLE, MAX_OUT_CHASSIS_ROLL_ANGLE};
    PID_Init(&roll_pid, PID_POSITION, Roll_PID_Param);
    float Tp_PID_Param[PID_PARAMETER_NUM] = {KP_CHASSIS_TP, KI_CHASSIS_TP, KD_CHASSIS_TP, 0, 0, MAX_IOUT_CHASSIS_TP, MAX_OUT_CHASSIS_TP};
    PID_Init(&tp_pid, PID_POSITION, Tp_PID_Param);
    float Turn_PID_Param[PID_PARAMETER_NUM] = {KP_CHASSIS_TURN, KI_CHASSIS_TURN, KD_CHASSIS_TURN, 0, 0, MAX_IOUT_CHASSIS_TURN, MAX_OUT_CHASSIS_TURN};
    PID_Init(&turn_pid, PID_POSITION, Turn_PID_Param);
}

void ChassisR_feedback_update(void)
{
    right.phi1 = theta_transform(-chassis_move.joint_motor[2]->Data.Position, J2_ANGLE_OFFSET, J2_DIRECTION, 1);
    right.phi4 = theta_transform(-chassis_move.joint_motor[3]->Data.Position, J3_ANGLE_OFFSET, J3_DIRECTION, 1);

    chassis_move.myPithR = INS.Pitch;
    chassis_move.myPithGyroR = INS.Gyro[1];

    // 计算倒地自起的腿长补偿，防止劈叉
    chassis_move.total_yaw = INS.YawTotalAngle;
    chassis_move.roll = INS.Roll;
    chassis_move.theta_err = 0.0f - (right.theta + left.theta);

    if (INS.Pitch < M_PI_6 && INS.Pitch > -M_PI_6)
    { // 根据pitch角度判断倒地自起是否完成
        chassis_move.recover_flag = 0;
    }
}

float LQR_K_R[12];
float x_r[6];    // x0是theta，x1是theta_dot，x2是x，x3是x_dot，x4是phi，x5是phi_dot
float T_Tp_r[2]; // T是轮子力矩，Tp是腿部力矩
void ChassisR_control_loop(void)
{
    float theta_target = 0.0f;

    VMC_calc_1(&right, chassis_move.myPithR, chassis_move.myPithGyroR, ((float)CHASS_FSM_TIME) / 1000.0f); // 计算theta和d_theta给lqr用，同时也计算右腿长L0,该任务控制周期是3*0.001秒

    LQR_K_calc(right.L0, LQR_K_R); // 根据当前腿长计算lqr控制器的增益

    // 补偿pid计算
    // chassis->turn_T=PID_Calc(&Turn_Pid, chassis->total_yaw, chassis->turn_set); //yaw轴pid计算
    chassis_move.turn_T = turn_pid.Param.KP * (chassis_move.turn_set - chassis_move.total_yaw) - turn_pid.Param.KD * INS.Gyro[2]; // 这样计算更稳一点

    chassis_move.roll_f0 = roll_pid.Param.KP * (chassis_move.roll_set - chassis_move.roll) - roll_pid.Param.KD * INS.Gyro[0]; // roll轴pid计算，目标roll角是roll_set，当前roll角是INS.Roll，反馈用roll角速度
    mySaturate(&chassis_move.roll_f0, -roll_pid.Param.LimitOutput, roll_pid.Param.LimitOutput);

    chassis_move.leg_tp = PID_Calculate(&tp_pid, 0.0f, chassis_move.theta_err); // 防劈叉pid计算

    if (chassis_move.mode == CHASSIS_SAFE &&
        (!CHASSIS_FULL_LQR_ENABLE || FULL_LQR_THETA_ZERO_OFFSET_TEST_ENABLE))
    {
#if (SAFE_STATIC_PHI0_TRIM_ENABLE)
        theta_target = safe_debug_right_theta_eq;
#endif
    }

    x_r[0] = X0_OFFSET + (right.theta - theta_target);                 // SAFE debug: theta error around phi0_eq
    x_r[1] = X1_OFFSET + (right.d_theta - 0.0f);                       // theta_dot误差，目标theta_dot是0
    x_r[2] = X2_OFFSET + (chassis_move.x_filter - chassis_move.x_set); // x误差，目标x是滤波后的x_set
    x_r[3] = X3_OFFSET + (chassis_move.v_filter - chassis_move.v_set); // x_dot误差，目标x_dot是滤波后的v_set
    x_r[4] = X4_OFFSET + (chassis_move.myPithR - (0.0f));              // phi误差，目标phi是0
    x_r[5] = X5_OFFSET + (chassis_move.myPithGyroR - 0.0f);            // phi_dot误差，目标phi_dot是0

    CalcLQR(LQR_K_R, x_r, T_Tp_r); // 计算lqr控制量，得到的控制量是单位为牛顿的轮子的力矩和腿部力

    safe_debug_right_wheel_lqr_body = (float)W1_DIRECTION * T_Tp_r[0];
    safe_debug_common_wheel_lqr_body =
        0.5f * (safe_debug_left_wheel_lqr_body + safe_debug_right_wheel_lqr_body);

    right.wheel_T = T_Tp_r[0] - chassis_move.turn_T; // 轮毂电机输出力矩，减去yaw轴补偿
    right.Tp = T_Tp_r[1] + chassis_move.leg_tp;      // 右边髋关节输出力矩 + 防劈叉补偿
    if (chassis_move.mode == CHASSIS_SAFE && !CHASSIS_FULL_LQR_ENABLE)
    {
#if (SAFE_STATIC_PHI0_TRIM_ENABLE)
        right.Tp = SAFE_DEBUG_LQR_TP_SIGN * SAFE_STATIC_LQR_TP_SCALE * T_Tp_r[1] + chassis_move.leg_tp;
#else
        right.Tp = SAFE_DEBUG_LQR_TP_SIGN * SAFE_LQR_TP_SCALE * T_Tp_r[1] + chassis_move.leg_tp;
#endif
        safe_debug_right_tp_cmd = right.Tp;
    }

    if (chassis_move.jump_flag == 1 || chassis_move.jump_flag == 2 || chassis_move.jump_flag == 3)
    {
        if (chassis_move.jump_flag == 1)
        {                                                                                                            // 压缩阶段
            right.F0 = BODY_GRAVITY / arm_cos_f32(right.theta) + PID_Calculate(&legr_pid, MIN_LEG_LENGTH, right.L0); // 前馈+pd

            if (right.L0 < MIN_LEG_LENGTH + 0.02f)
            {
                jump_time_r++;
            }
            if (jump_time_r >= 10 && jump_time_l >= 10)
            {
                jump_time_r = 0;
                jump_time_l = 0;
                chassis_move.jump_flag = 2; // 压缩完毕进入上升加速阶段
                chassis_move.jump_flag2 = 2;
            }
        }
        else if (chassis_move.jump_flag == 2)
        {                                                                                                            // 上升加速阶段
            right.F0 = BODY_GRAVITY / arm_cos_f32(right.theta) + PID_Calculate(&legr_pid, MAX_LEG_LENGTH, right.L0); // 前馈+pd

            if (right.L0 > MAX_LEG_LENGTH - 0.03f)
            {
                jump_time_r++;
            }
            if (jump_time_r >= 2 && jump_time_l >= 2)
            {
                jump_time_r = 0;
                jump_time_l = 0;
                chassis_move.jump_flag = 3; // 上升完毕进入缩腿阶段
                chassis_move.jump_flag2 = 3;
            }
        }
        else if (chassis_move.jump_flag == 3)
        {                                                                   // 缩腿阶段
            right.F0 = PID_Calculate(&legr_pid, INIT_LEG_LENGTH, right.L0); // pd
            // chassis_move.theta_set = 0.0f;

            chassis_move.x_filter = 0.0f;
            chassis_move.x_set = chassis_move.x_filter;
            if (right.L0 < INIT_LEG_LENGTH + 0.05f)
            {
                jump_time_r++;
            }
            if (jump_time_r >= 3 && jump_time_l >= 3)
            {
                jump_time_r = 0;
                jump_time_l = 0;
                chassis_move.leg_set = INIT_LEG_LENGTH;
                chassis_move.last_leg_set = INIT_LEG_LENGTH;
                chassis_move.jump_flag = 0; // 缩腿完毕
                chassis_move.jump_flag2 = 0;
            }
        }
    }
    else
    {
        float gravity_ff_scale = (chassis_move.mode == CHASSIS_SAFE) ? SAFE_LEG_GRAVITY_FF_SCALE : 1.0f;
        right.F0 = gravity_ff_scale * BODY_GRAVITY / arm_cos_f32(right.theta) + PID_Calculate(&legr_pid, chassis_move.leg_set, right.L0); // 前馈+pd
    }

    right_flag = ground_detection(&right); // 右腿离地检测

    if (chassis_move.recover_flag == 0)
    { // 倒地自起不需要检测是否离地
        if ((right_flag == 1 && left_flag == 1 && right.leg_flag == 0 && chassis_move.jump_flag != 1 && chassis_move.jump_flag2 != 1 && chassis_move.jump_flag2 != 2 && chassis_move.jump_flag != 2) || chassis_move.jump_flag == 3)
        { // 当两腿同时离地并且遥控器没有在控制腿的伸缩时，才认为离地
          // 排除跳跃的压缩阶段和跳跃的缩腿阶段
            right.wheel_T = 0.0f;
            right.Tp = LQR_K_R[6] * (right.theta - 0.0f) + LQR_K_R[7] * (right.d_theta - 0.0f);

            chassis_move.x_filter = 0.0f;
            chassis_move.x_set = chassis_move.x_filter;

            right.Tp = right.Tp + chassis_move.leg_tp;
        }
        else
        {                       // 没有离地
            right.leg_flag = 0; // 置为0

            if (chassis_move.jump_flag == 0)
            { // 不跳跃的时候需要roll轴补偿

                right.F0 = right.F0 - chassis_move.roll_f0; // roll轴补偿取反然后加上去
            }
        }
    }
    else if (chassis_move.recover_flag == 1)
    {
        right.Tp = 0.0f;
        right.F0 = 0.0f;
    }

    if (!CHASSIS_FULL_LQR_ENABLE && chassis_move.mode == CHASSIS_SAFE && chassis_move.recover_flag == 0 &&
        chassis_move.jump_flag == 0 && chassis_move.jump_flag2 == 0 &&
        !(left_flag == 1 && right_flag == 1))
    {
        ChassisApplySafePitchTp(&right, 1);
    }

#if (JOINT_MAP_TEST_ENABLE)
    // Override every normal controller output immediately before the VMC
    // mapping. The unselected leg receives zero force/torque commands.
    right.F0 = 0.0f;
    right.Tp = 0.0f;
    right.wheel_T = 0.0f;
    if (chassis_move.start_flag == 1U && chassis_move.mode == CHASSIS_SAFE &&
        chassis_move.error_code == 0U && joint_map_test_side == 2U)
    {
        right.Tp = joint_map_test_tp_cmd;
    }
#endif

    VMC_calc_2(&right); // 计算期望的关节输出力矩

#if (JOINT_MAP_TEST_ENABLE)
    mySaturate(&right.torque_set[0], -JOINT_MAP_TEST_MOTOR_TORQUE_LIMIT, JOINT_MAP_TEST_MOTOR_TORQUE_LIMIT);
    mySaturate(&right.torque_set[1], -JOINT_MAP_TEST_MOTOR_TORQUE_LIMIT, JOINT_MAP_TEST_MOTOR_TORQUE_LIMIT);
#endif

    if (chassis_move.mode == CHASSIS_SAFE && !CHASSIS_FULL_LQR_ENABLE)
    {
#if (SAFE_DEBUG_COMMON_WHEEL_TEST_ENABLE)
        right.wheel_T = (float)W1_DIRECTION * SAFE_DEBUG_COMMON_WHEEL_TEST_T;
        safe_debug_common_wheel_lqr_body = SAFE_DEBUG_COMMON_WHEEL_TEST_T;
#else
        // SAFE debug: low-scale LQR wheel torque plus damping.
        right.wheel_T = 0.0f;
        if (chassis_move.wheel_motor[1] != 0)
        {
            right.wheel_T = ChassisCalcSafeWheelTorque(chassis_move.wheel_motor[1]->Data.Velocity,
                                                       (float)W1_DIRECTION);
        }
#if (SAFE_STATIC_PHI0_TRIM_ENABLE)
#if (SAFE_STATIC_COMMON_WHEEL_ENABLE)
        right.wheel_T += (float)W1_DIRECTION *
                         SAFE_STATIC_LQR_WHEEL_SCALE *
                         safe_debug_common_wheel_lqr_body;
#else
        right.wheel_T += SAFE_STATIC_LQR_WHEEL_SCALE * (T_Tp_r[0] - chassis_move.turn_T);
#endif
#else
        right.wheel_T += SAFE_LQR_WHEEL_SCALE * (T_Tp_r[0] - chassis_move.turn_T);
#endif
#endif
    }

    // Keep the SAFE wheel-torque software limit in both staged-debug and
    // complete-LQR paths. The motor driver has a wider final hardware limit.
    if (chassis_move.mode == CHASSIS_SAFE)
    {
        mySaturate(&right.wheel_T, -SAFE_WHEEL_T_LIMIT, SAFE_WHEEL_T_LIMIT);
    }

    if (chassis_move.jump_flag == 1 || chassis_move.jump_flag == 2 || chassis_move.jump_flag == 3)
    { // 跳跃时允许使用更高的关节期望扭矩
        mySaturate(&right.torque_set[0], MIN_JOINT_TORQUE_JUMP, MAX_JOINT_TORQUE_JUMP);
        mySaturate(&right.torque_set[1], MIN_JOINT_TORQUE_JUMP, MAX_JOINT_TORQUE_JUMP);
    }
    else
    { // 普通状态限制关节期望扭矩
        mySaturate(&right.torque_set[0], MIN_JOINT_TORQUE, MAX_JOINT_TORQUE);
        mySaturate(&right.torque_set[1], MIN_JOINT_TORQUE, MAX_JOINT_TORQUE);
    }
}
