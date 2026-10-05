#include "Remote_Task.h"
#include "Detect_Task.h"
#include "cmsis_os.h"
#include "math.h"

#include "robot_param.h"

uint32_t REMOTE_TIME = 10; // rc手柄任务周期是10ms

void Remote_task(void)
{
	while (INS.ins_flag == 0)
	{ // 等待惯导(INS)初始化收敛完成
		osDelay(1);
	}

	// pid init
	float Stand_Up_PID_Param[PID_PARAMETER_NUM] = {KP_CHASSIS_STAND_UP, KI_CHASSIS_STAND_UP, KD_CHASSIS_STAND_UP, 0, 0, MAX_IOUT_CHASSIS_STAND_UP, MAX_OUT_CHASSIS_STAND_UP};
	PID_Init(&stand_up_pid, PID_POSITION, Stand_Up_PID_Param);

	float dt = (float)(REMOTE_TIME) / 1000.0f;
	static float last_jump_vrb = 0;
	static uint32_t vbat_low_count = 0;

	int16_t rc_x = 0, rc_wz = 0;
	int16_t rc_length = 0, rc_angle = 0;
	int16_t rc_roll = 0;

	while (1)
	{
		// 处理异常
		ChassisHandleException();
		// 设置底盘模式
		ChassisSetMode();
		// 底盘状态量
		UpdateCalibrateStatus();

		// 遥控器输入处理
		if (chassis_move.start_flag == 1 && (chassis_move.recover_flag == 1 ||
											 chassis_move.mode == CHASSIS_CALIBRATE ||
											 chassis_move.mode == CHASSIS_STAND_UP))
		{
			chassis_move.v_set = 0.0f; // 速度清零
		}
		else
		{
			if (remote_ctrl.rc.s[DT7_SW_LEFT] == DT7_SW_DOWN)
			{
				// Power Off (遥控器左SW拨杆打下，关闭底盘)
				chassis_move.start_flag = 0;
				chassis_move.recover_flag = 0;

				vbat_low_count = 0;
			}
			else if (remote_ctrl.rc.s[DT7_SW_LEFT] == DT7_SW_UP || remote_ctrl.rc.s[DT7_SW_LEFT] == DT7_SW_MID)
			{
				if (vbus_low_warning && (chassis_move.start_flag == 1))
					vbat_low_count++;

				// 低电量保护逻辑，低电量持续一段时间后自动断电保护
				if (vbat_low_count > VBAT_LOW_WARNING_THRESHOLD)
				{
					chassis_move.start_flag = 0;
					chassis_move.recover_flag = 0;
				}
				else
				{
					// Power On (遥控器SWD拨杆打上，启动底盘)
					chassis_move.start_flag = 1;
				}
			}

			if (ENABLE_CHASSIS_CALIBRATE && remote_ctrl.rc.s[DT7_SW_LEFT] == DT7_SW_UP && remote_ctrl.rc.s[DT7_SW_RIGHT] == DT7_SW_UP)
			{
				CALIBRATE.toggle = true;
			}

			if (remote_ctrl.rc.s[DT7_SW_LEFT] == DT7_SW_UP && remote_ctrl.rc.s[DT7_SW_RIGHT] == DT7_SW_DOWN)
			{
				chassis_move.mode = CHASSIS_STAND_UP;
			}

			// 倒地检测与自恢复逻辑 (当俯仰角过大判定为倒地时)
			if (chassis_move.recover_flag == 0 && ((chassis_move.myPithR < ((-3.1415926f) / 4.0f) && chassis_move.myPithR > ((-3.1415926f) / 2.0f)) || (chassis_move.myPithR > (3.1415926f / 4.0f) && chassis_move.myPithR < (3.1415926f / 2.0f))))
			{
				chassis_move.recover_flag = 1; // 需要恢复倒地
				chassis_move.leg_set = 0.08f;  // 恢复原始腿长
			}

			if (chassis_move.start_flag == 1)
			{
				rc_deadband_limit(remote_ctrl.rc.ch[CHASSIS_X_CHANNEL], rc_x, CHASSIS_RC_DEADLINE);
				// 6s 电池供电时的速度映射 (动力更强，倍率更高)
				chassis_move.v_set = ((float)remote_ctrl.rc.ch[CHASSIS_X_CHANNEL]) * (-0.00097f * 1.3f); // 摇杆向前推为负
				chassis_move.x_set = rc_x * DT7_RC_TO_ONE * MAX_SPEED_VECTOR_VX;
				// chassis_move.x_set = chassis_move.x_set + chassis_move.v_set * dt;

				// 左摇杆水平方向控制偏航角 (Yaw转向)
				chassis_move.turn_set += ((float)remote_ctrl.rc.ch[DT7_CH_LH]) * (-0.0000625f);

				// SWA 拨杆控制横滚角 (Roll姿态)
				if (remote_ctrl.rc.ch[DT7_CH_ROLLER] == 0)
				{
					chassis_move.roll_set = -0.03f;
				}
				else
				{
					chassis_move.roll_set += ((float)remote_ctrl.rc.ch[DT7_CH_RH]) * (0.000016f);
				}
				mySaturate(&chassis_move.roll_set, -0.40f, 0.40f);

				// SWC 拨杆结合左摇杆垂直方向控制腿长
				if (remote_ctrl.rc.ch[DT7_CH_ROLLER] == 0)
				{
					chassis_move.leg_set = 0.08f;
				}
				else
				{
					chassis_move.leg_set = (remote_ctrl.rc.ch[DT7_CH_LV] + 1024) * (0.00007f) + 0.072f;
				}

				mySaturate(&chassis_move.leg_set, 0.072f, 0.21f);

				if (fabsf(chassis_move.last_leg_set - chassis_move.leg_set) > 0.0001f)
				{						// 遥控器设置腿长在变化
					right.leg_flag = 1; // 为1标志位遥控器正在控制腿长，此时不进行离地检测，因为在腿长瞬间变短时离地检测会误判为离地
					left.leg_flag = 1;
				}
				chassis_move.last_leg_set = chassis_move.leg_set;

				// // SWB 拨杆结合拨轮控制跳跃逻辑
				// if (remote_ctrl.toggle.swb == 0)
				// {
				// 	// chassis_move.jump_flag=0;
				// 	// chassis_move.jump_flag2=0;
				// }
				// else
				// {
				// 	// 触发跳跃条件：SWC未归零，拨轮拨动产生上升沿，且当前腿长允许跳跃
				// 	if (remote_ctrl.toggle.swc != 0 && remote_ctrl.var.b < 500 && last_jump_vrb >= 500 && chassis_move.leg_set <= 0.16f)
				// 	{
				// 		if (chassis_move.vbus_mode == 2)
				// 		{
				// 			// 6s 电源模式下才允许跳跃（保证动力足够）
				// 			chassis_move.jump_flag = 1;
				// 			chassis_move.jump_flag2 = 1;
				// 		}
				// 	}
				// }

				// last_jump_vrb = remote_ctrl.var.b;
			}

			else
			{
				chassis_move.v_set = 0.0f; // 速度清零
				// chassis_move.x_set = chassis_move.x_filter;		// 保持位置
				chassis_move.turn_set = chassis_move.total_yaw; // 保持偏航
				chassis_move.leg_set = 0.08f;					// 原始腿长
				chassis_move.roll_set = -0.03f;
			}
		}
		// else
		// {
		// 	// 遥控器离线 (失控保护)
		// 	chassis_move.start_flag = 0;
		// 	chassis_move.recover_flag = 0;
		// 	vbat_low_count = 0;

		// 	chassis_move.v_set = 0.0f; // 速度清零
		// 	// chassis_move.x_set = chassis_move.x_filter;		// 保持位置
		// 	chassis_move.turn_set = chassis_move.total_yaw; // 保持偏航
		// 	chassis_move.leg_set = 0.08f;					// 原始腿长
		// 	chassis_move.roll_set = -0.03f;
		// }

		// 计算控制量
		ChassisConsole();

		osDelay(REMOTE_TIME);
	}
}