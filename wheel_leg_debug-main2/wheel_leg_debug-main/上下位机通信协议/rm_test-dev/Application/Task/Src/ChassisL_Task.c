#include "ChassisL_Task.h"

void ChassisL_task(void)
{
    while (INS.ins_flag == 0)
    { // 等待加速度收敛
        osDelay(1);
    }
    ChassisL_init(); // 初始化左边两个关节电机和左边轮毂电机的id和控制模式、初始化腿部

    while (1)
    {
        ChassisL_feedback_update(); // 更新数据

        ChassisL_control_loop(); // 控制计算

        if (chassis_move.start_flag == 1)
        {
            switch (chassis_move.mode)
            {
            case CHASSIS_STAND_UP:
            {
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[0], left.position_set[0], 0, DEBUG_POS_KP, DEBUG_POS_KD, 0);
                osDelay(CHASS_TIME);
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[1], left.position_set[1], 0, DEBUG_POS_KP, DEBUG_POS_KD, 0);
                osDelay(CHASS_TIME);
                LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[0], 0, left.wheel_T);
                osDelay(CHASS_TIME);
            }
            break;
            case CHASSIS_CALIBRATE:
            {
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[0], 0, left.velocity_set[0], 0, CALIBRATE_VEL_KD, 0);
                osDelay(CHASS_TIME);
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[1], 0, left.velocity_set[1], 0, CALIBRATE_VEL_KD, 0);
                osDelay(CHASS_TIME);
                LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[0], 0, left.wheel_T);
                osDelay(CHASS_TIME);

                if (CALIBRATE.reached[0] && CALIBRATE.reached[1] && !CALIBRATE.left_reached)
                {
                    CALIBRATE.left_reached = true; // 只校准一次，避免误触

                    DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[0], DM_Motor_Save_Zero_Position);
                    DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[0], DM_Motor_Enable);
                    osDelay(CHASS_TIME);
                    DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[1], DM_Motor_Save_Zero_Position);
                    DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[1], DM_Motor_Enable);
                    osDelay(CHASS_TIME * 2);
                }
            }
            break;
            case CHASSIS_SAFE: // 正常车子状态，执行控制量
            {
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[0], 0, 0, NORMAL_POS_KP, NORMAL_POS_KD, -left.torque_set[1]);
                osDelay(CHASS_TIME);
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[1], 0, 0, NORMAL_POS_KP, NORMAL_POS_KD, -left.torque_set[0]);
                osDelay(CHASS_TIME);
                LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[0], 0, left.wheel_T);
                osDelay(CHASS_TIME);
            }
            break;
            default:
            {
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[0], 0, 0, 0, ZERO_FORCE_VEL_KD, 0);
                osDelay(CHASS_TIME);
                DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[1], 0, 0, 0, ZERO_FORCE_VEL_KD, 0);
                osDelay(CHASS_TIME);
                LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[0], 0, 1);
                // LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[0], 0, 0);
                osDelay(CHASS_TIME);
            }
            }
        }
        else
        {
            // 失能状态，电机停止
            DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[0], 0, 0, 0, 0, 0);
            osDelay(CHASS_TIME);
            DM_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.joint_motor[1], 0, 0, 0, 0, 0);
            osDelay(CHASS_TIME);
            LK_Motor_CAN_TxMessage(&FDCAN3_TxFrame, chassis_move.wheel_motor[0], 0, 0);
            osDelay(CHASS_TIME);
        }
    }
}

void ChassisL_init(void)
{
    chassis_move.joint_motor[0] = &DM_8009_Motor[0];
    chassis_move.joint_motor[1] = &DM_8009_Motor[1];

    chassis_move.wheel_motor[0] = &LK_9025_Motor[0];

    // pid init
    float LegL_PID_Param[PID_PARAMETER_NUM] = {KP_CHASSIS_LEG_LENGTH_LENGTH, KI_CHASSIS_LEG_LENGTH_LENGTH, KD_CHASSIS_LEG_LENGTH_LENGTH, ALPHA_LEG_LENGTH_LENGTH, 0, MAX_IOUT_CHASSIS_LEG_LENGTH_LENGTH, MAX_OUT_CHASSIS_LEG_LENGTH_LENGTH};
    PID_Init(&legl_pid, PID_POSITION, LegL_PID_Param);

    VMC_init(&left); // 给杆长赋值

    for (int j = 0; j < 10; j++)
    {
        DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[0], DM_Motor_Enable);
        osDelay(1);
    }
    for (int j = 0; j < 10; j++)
    {
        DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[1], DM_Motor_Enable);
        osDelay(1);
    }
    for (int j = 0; j < 10; j++)
    {
        LK_Motor_Command(&FDCAN3_TxFrame, chassis_move.wheel_motor[0], LK_Motor_Enable); // 左边轮毂电机
        osDelay(1);
    }

    // // 保存零点位置
    // DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[0], DM_Motor_Save_Zero_Position);
    // osDelay(10);
    // DM_Motor_Command(&FDCAN3_TxFrame, chassis_move.joint_motor[1], DM_Motor_Save_Zero_Position);
    // osDelay(10);
}

void ChassisL_feedback_update(void)
{
    left.phi4 = theta_transform(-chassis_move.joint_motor[0]->Data.Position, J0_ANGLE_OFFSET, J0_DIRECTION, 1);
    left.phi1 = theta_transform(-chassis_move.joint_motor[1]->Data.Position, J1_ANGLE_OFFSET, J1_DIRECTION, 1);

    chassis_move.myPithL = 0.0f - INS.Pitch;
    chassis_move.myPithGyroL = 0.0f - INS.Gyro[1];
}

float LQR_K_L[12];
float x_l[6];    // x0是theta，x1是theta_dot，x2是x，x3是x_dot，x4是phi，x5是phi_dot
float T_Tp_l[2]; // T是轮子力矩，Tp是腿部力矩
void ChassisL_control_loop(void)
{
    VMC_calc_1(&left, chassis_move.myPithL, chassis_move.myPithGyroL, ((float)CHASS_FSM_TIME) / 1000.0f); // 计算theta和d_theta给lqr用，同时也计算左腿长L0,该任务控制周期是3*0.001秒

    LQR_K_calc(left.L0, LQR_K_L); // 根据当前腿长计算lqr控制器的增益

    x_l[0] = X0_OFFSET + (left.theta - 0.0f);                           // theta误差，目标theta是0
    x_l[1] = X1_OFFSET + (left.d_theta - 0.0f);                         // theta_dot误差，目标theta_dot是0
    x_l[2] = -X2_OFFSET + (chassis_move.x_set - chassis_move.x_filter); // x误差，目标x是滤波后的x_set
    x_l[3] = -X3_OFFSET + (chassis_move.v_set - chassis_move.v_filter); // x_dot误差，目标x_dot是滤波后的v_set
    x_l[4] = -X4_OFFSET + (chassis_move.myPithL - (0.0f));              // phi误差，目标phi是0
    x_l[5] = X5_OFFSET + (chassis_move.myPithGyroL - 0.0f);             // phi_dot误差，目标phi_dot是0

    CalcLQR(LQR_K_L, x_l, T_Tp_l); // 计算lqr控制量，得到的控制量是单位为牛顿的轮子的力矩和腿部力

    left.wheel_T = T_Tp_l[0] - chassis_move.turn_T; // 轮毂电机输出力矩，减去yaw轴补偿
    left.Tp = T_Tp_l[1] + chassis_move.leg_tp;      // 左边髋关节输出力矩 + 防劈叉补偿

    if (chassis_move.jump_flag2 == 1 || chassis_move.jump_flag2 == 2 || chassis_move.jump_flag2 == 3)
    {
        if (chassis_move.jump_flag2 == 1)
        {                                                                                                         // 压缩阶段
            left.F0 = BODY_GRAVITY / arm_cos_f32(left.theta) + PID_Calculate(&legl_pid, MIN_LEG_LENGTH, left.L0); // 前馈+pd

            if (left.L0 < MIN_LEG_LENGTH + 0.02f)
            {
                jump_time_l++;
            }
            if (jump_time_l >= 10 && jump_time_r >= 10)
            {
                jump_time_l = 0;
                jump_time_r = 0;
                chassis_move.jump_flag2 = 2;
                chassis_move.jump_flag = 2; // 压缩完毕进入上升加速阶段
            }
        }
        else if (chassis_move.jump_flag2 == 2)
        {                                                                                                         // 上升加速阶段
            left.F0 = BODY_GRAVITY / arm_cos_f32(left.theta) + PID_Calculate(&legl_pid, MAX_LEG_LENGTH, left.L0); // 前馈+pd

            if (left.L0 > MAX_LEG_LENGTH - 0.03f)
            {
                jump_time_l++;
            }
            if (jump_time_l >= 2 && jump_time_r >= 2)
            {
                jump_time_l = 0;
                jump_time_r = 0;
                chassis_move.jump_flag2 = 3;
                chassis_move.jump_flag = 3; // 上升完毕进入缩腿阶段
            }
        }
        else if (chassis_move.jump_flag2 == 3)
        {                                                                 // 缩腿阶段
            left.F0 = PID_Calculate(&legl_pid, INIT_LEG_LENGTH, left.L0); // pd
            // chassis_move.theta_set = 0.0f;

            chassis_move.x_filter = 0.0f;
            chassis_move.x_set = chassis_move.x_filter;
            if (left.L0 < INIT_LEG_LENGTH + 0.05f)
            {
                jump_time_l++;
            }
            if (jump_time_l >= 3 && jump_time_r >= 3)
            {
                jump_time_l = 0;
                jump_time_r = 0;
                chassis_move.leg_set = INIT_LEG_LENGTH;
                chassis_move.last_leg_set = INIT_LEG_LENGTH;
                chassis_move.jump_flag2 = 0;
                chassis_move.jump_flag = 0;
            }
        }
    }
    else
    {
        left.F0 = BODY_GRAVITY / arm_cos_f32(left.theta) + PID_Calculate(&legl_pid, chassis_move.leg_set, left.L0); // 前馈+pd
    }

    left_flag = ground_detection(&left); // 左腿离地检测

    if (chassis_move.recover_flag == 0)
    { // 倒地自起不需要检测是否离地
        if ((right_flag == 1 && left_flag == 1 && left.leg_flag == 0 && chassis_move.jump_flag2 != 1 && chassis_move.jump_flag != 1 && chassis_move.jump_flag2 != 2 && chassis_move.jump_flag != 2) || chassis_move.jump_flag2 == 3)
        { // 当两腿同时离地并且遥控器没有在控制腿的伸缩时，才认为离地
          // 排除跳跃的压缩阶段和跳跃的缩腿阶段
            left.wheel_T = 0.0f;
            left.Tp = LQR_K_L[6] * (left.theta - 0.0f) + LQR_K_L[7] * (left.d_theta - 0.0f);

            chassis_move.x_filter = 0.0f;
            chassis_move.x_set = chassis_move.x_filter;

            left.Tp = left.Tp + chassis_move.leg_tp;
        }
        else
        {                      // 没有离地
            left.leg_flag = 0; // 置为0

            if (chassis_move.jump_flag2 == 0)
            { // 不跳跃的时候需要roll轴补偿

                left.F0 = left.F0 - chassis_move.roll_f0; // roll轴补偿取反然后加上去
            }
        }
    }
    else if (chassis_move.recover_flag == 1)
    {
        left.Tp = 0.0f;
        left.F0 = 0.0f;
    }

    VMC_calc_2(&left); // 计算期望的关节输出力矩

    // 这里不对输出的力矩进行限制，直接让电机输出
    // if (chassis_move.jump_flag2 == 1 || chassis_move.jump_flag2 == 2 || chassis_move.jump_flag2 == 3)
    // { // 跳跃的时候需要更大扭矩
    //     mySaturate(&left.torque_set[0], MIN_JOINT_TORQUE_JUMP, MAX_JOINT_TORQUE_JUMP);
    //     mySaturate(&left.torque_set[1], MIN_JOINT_TORQUE_JUMP, MAX_JOINT_TORQUE_JUMP);
    // }
    // else
    // { // 不跳跃的时候最大为额定扭矩
    //     mySaturate(&left.torque_set[0], MIN_JOINT_TORQUE, MAX_JOINT_TORQUE);
    //     mySaturate(&left.torque_set[1], MIN_JOINT_TORQUE, MAX_JOINT_TORQUE);
    // }
}