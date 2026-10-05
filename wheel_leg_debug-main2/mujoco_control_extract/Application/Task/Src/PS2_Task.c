/**
  *********************************************************************
  * @file      ps2_task.c/h
  * @brief     本任务是读取并处理ps2手柄接收器的数据，
  *            将遥控器数据转换为底盘运动速度、偏航角转角、目标腿长等
  * @note
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */

#include "PS2_Task.h"
#include "Detect_Task.h"
#include "User_Lib.h"
#include "cmsis_os.h"

ps2data_t ps2data;

uint16_t Handkey;														  // 按键值读取临时存储区
uint8_t Comd[2] = {0x01, 0x42};											  // 开始命令。请求数据
uint8_t Data[9] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}; // 数据存储数组
uint16_t MASK[] = {
	PSB_SELECT,
	PSB_L3,
	PSB_R3,
	PSB_START,
	PSB_PAD_UP,
	PSB_PAD_RIGHT,
	PSB_PAD_DOWN,
	PSB_PAD_LEFT,
	PSB_L2,
	PSB_R2,
	PSB_L1,
	PSB_R1,
	PSB_GREEN,
	PSB_RED,
	PSB_BLUE,
	PSB_PINK,
}; // 按键值屏蔽掩码

bool ps2_lost = false;		// ps2手柄离线标志位，初始为离线
// bool ps2_lost = true;		// ps2手柄离线标志位，初始为离线
uint32_t last_operate_time; // 上次操作时间
uint16_t ps2_mode;

uint32_t vbat_low_count = 0;

uint32_t PS2_TIME = 10; // ps2手柄任务周期是10ms
void pstwo_task(void)
{
	while (INS.ins_flag == 0)
	{ // 等待惯导(INS)初始化收敛完成
		osDelay(1);
	}

	PS2_SetInit(); // 初始化ps2手柄通信接口

	float dt = (float)(PS2_TIME) / 1000.0f;

	while (1)
	{
		if (Data[1] != PS2_MODE_ANALOG)
		{
			// ps2_lost = true;
			PS2_SetInit();
		}

		ps2data.last_lx = ps2data.lx;
		ps2data.last_ly = ps2data.ly;
		ps2data.last_rx = ps2data.rx;
		ps2data.last_ry = ps2data.ry;

		PS2_data_read(&ps2data);					   // 读数据
		PS2_data_process(&ps2data, &chassis_move, dt); // 处理数据，下发底盘控制命令

		osDelay(PS2_TIME);
	}
}

uint32_t GetPs2IdleTime(void) { return HAL_GetTick() - last_operate_time; }

// 向手柄发送命令
void PS2_Cmd(uint8_t CMD)
{
	volatile uint16_t ref = 0x01;
	Data[1] = 0;
	for (ref = 0x01; ref < 0x0100; ref <<= 1)
	{
		if (ref & CMD)
		{
			HAL_GPIO_WritePin(GPIOA, GPIO_PIN_2, GPIO_PIN_SET); // 发送一位高电平 DO_H;
		}
		else
			HAL_GPIO_WritePin(GPIOA, GPIO_PIN_2, GPIO_PIN_RESET); // DO_L

		HAL_GPIO_WritePin(GPIOE, GPIO_PIN_13, GPIO_PIN_SET); // 时钟拉高
		DWT_Delay(0.000005f);
		HAL_GPIO_WritePin(GPIOE, GPIO_PIN_13, GPIO_PIN_RESET);
		DWT_Delay(0.000005f);
		HAL_GPIO_WritePin(GPIOE, GPIO_PIN_13, GPIO_PIN_SET);
		if (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0))
			Data[1] = ref | Data[1];
	}
	DWT_Delay(0.000016f);
}

/**************************************************************************
Function: Read the control of the ps2 handle
Input   : none
Output  : none
功能说明：读取PS2手柄的控制量
内部调用：无
返回  值：无
**************************************************************************/
uint8_t reve_flag = 0;
void PS2_data_read(ps2data_t *data)
{
	// 获取按键值
	data->key = PS2_DataKey(); // 获取按键值

	// 获取左摇杆X轴方向模拟量
	data->lx = PS2_AnologData(PSS_LX);

	// 获取左摇杆Y轴方向模拟量
	data->ly = PS2_AnologData(PSS_LY);

	// 获取右摇杆X轴方向模拟量
	data->rx = PS2_AnologData(PSS_RX);

	// 获取右摇杆Y轴方向模拟量
	data->ry = PS2_AnologData(PSS_RY);

	if ((data->ry <= 255 && data->ry > 192) || (data->ry < 64 && data->ry >= 0))
	{
		data->rx = 127;
	}
	if ((data->rx <= 255 && data->rx > 192) || (data->rx < 64 && data->rx >= 0))
	{
		data->ry = 128;
	}
}

void PS2_data_process(ps2data_t *data, chassis_t *chassis, float dt)
{
	// if (Data[1] == PS2_MODE_ANALOG)
	// {
	// 	ps2_lost = false;
	// }
	// else
	// {
	// 	chassis->start_flag = 0; // 手柄处于非正常状态，强制底盘失能
	// }
	if (data->last_key != PSB_START && data->key == PSB_START && chassis->start_flag == 0) // 上电
	{
		// 手柄上的Start按键被按下
		chassis->start_flag = 1;
		chassis->mode = CHASSIS_SAFE; // 先进入安全状态，等待后续条件满足才切换到其他状态
	}
	else if ((data->last_key != PSB_START && data->key == PSB_START && chassis->start_flag == 1) || (vbat_low_count > VBAT_LOW_WARNING_THRESHOLD)) // 下电或者低电量保护逻辑，低电量持续一段时间后自动断电保护
	{
		// 手柄上的Start按键被按下
		chassis->start_flag = 0;
		chassis->recover_flag = 0;

		vbat_low_count = 0;
	}

	if (vbus_low_warning && (chassis_move.start_flag == 1))
		vbat_low_count++;

	if (chassis->recover_flag == 0 && ((chassis->myPithR < ((-M_PI_4)) && chassis->myPithR > ((-M_PI_2)) || (chassis->myPithR > (M_PI_4) && chassis->myPithR < (M_PI_2)))))
	{
		chassis->recover_flag = 1;			// 需要恢复倒地
		chassis->leg_set = INIT_LEG_LENGTH; // 恢复原始腿长
	}

	if (data->last_key != PSB_SELECT && data->key == PSB_SELECT && chassis->prejump_flag == 0 && chassis->start_flag == 1)
	{
		// 手柄上的select按键按一下置1
		chassis->prejump_flag = 1; // 预跳跃标志位置1
	}
	else if (data->last_key != PSB_SELECT && data->key == PSB_SELECT && chassis->prejump_flag == 1)
	{
		// 手柄上的select按键再按一下清0
		chassis->prejump_flag = 0;
	}

	if (data->last_key != PSB_PAD_UP && data->key == PSB_PAD_UP && chassis->prejump_flag == 1 && chassis->jump_flag == 0 && chassis->jump_flag2 == 0)
	{
		// 手柄上的左侧的十字键的上按键按下，执行跳跃
		// 只有当预跳跃标志为1且落地缓冲完成后才能开始跳跃
		chassis->jump_flag = 1;
		chassis->jump_flag2 = 1;
	}

	// 判断数据变化
	if (ps2data.key != ps2data.last_key || ps2data.lx != ps2data.last_lx || ps2data.ly != ps2data.last_ly || ps2data.rx != ps2data.last_rx || ps2data.ry != ps2data.last_ry)
	{
		last_operate_time = HAL_GetTick(); // 更新上次操作时间
	}

	data->last_key = data->key;

	if (chassis->start_flag == 1)									  // 使能
	{																  // 启动
		chassis->target_v = ((float)(data->ry - 128)) * (-0.008f);	  // 往前大于0
		slope_following(&chassis->target_v, &chassis->v_set, 0.005f); //	坡度跟随

		chassis->x_set = chassis->x_set + chassis->v_set * dt;

		chassis->turn_set = chassis->turn_set + (data->rx - 127) * (-0.0005f); // 向右打为正

		chassis->roll_target = chassis->roll_set + ((float)(data->lx - 127)) * (-0.00007f);
		slope_following(&chassis->roll_target, &chassis->roll_set, 0.0075f);

		mySaturate(&chassis->roll_set, MIN_ROLL, MAX_ROLL);

		chassis->leg_set = chassis->leg_set;
		// chassis->leg_set = chassis->leg_set + ((float)(data->ly - 128)) * (-0.000016f);
		mySaturate(&chassis->leg_set, MIN_LEG_LENGTH, MAX_LEG_LENGTH);

		// if (fabsf(chassis->last_leg_set - chassis->leg_set) > 0.0001f)
		// {						// 遥控器设置腿长在变化
		// 	right.leg_flag = 1; // 为1标志位遥控器正在控制腿长，此时不进行离地检测，因为在腿长瞬间变短时离地检测会误判为离地
		// 	left.leg_flag = 1;
		// }
		chassis->last_leg_set = chassis->leg_set;

		if (data->key == PSB_L2)
		{
			chassis->roll_set = INIT_ROLL;
		}
		// else if (ENABLE_CHASSIS_CALIBRATE && data->key == PSB_R2)
		else if ((ENABLE_CHASSIS_CALIBRATE && data->key == PSB_R2) && !chass_is_calibrated)
		{
			chass_is_calibrated = true; // 只校准一次，避免误触
			CALIBRATE.toggle = true;
		}
		else if (data->key == PSB_L1)
		{
			chassis->mode = CHASSIS_STAND_UP;
			chassis->leg_set = INIT_LEG_LENGTH;
		}
	}
	else if (chassis->start_flag == 0)
	{											// 失能
		chassis->mode = CHASSIS_OFF;					// 进入底盘关闭模式

		chassis->v_set = 0.0f;					// 速度清零
		chassis->x_set = chassis->x_filter;		// 保持位置
		chassis->turn_set = chassis->total_yaw; // 保持偏航
		chassis->leg_set = INIT_LEG_LENGTH;		// 原始腿长
		chassis->roll_set = INIT_ROLL;
	}
}

// 判断是否为红灯模式,0x41=模拟绿灯，0x73=模拟红灯
// 返回值：0为红灯模式
//		  其它为绿灯模式
uint8_t PS2_RedLight(void)
{
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_RESET);
	PS2_Cmd(Comd[0]); // 开始命令
	PS2_Cmd(Comd[1]); // 请求数据
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_SET);
	if (Data[1] == 0X73)
		return 0;
	else
		return 1;
}
// 读取手柄数据
void PS2_ReadData(void)
{
	volatile uint8_t byte = 0;
	volatile uint16_t ref = 0x01;

	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_RESET); // CS_L
	PS2_Cmd(Comd[0]);									  // 开始命令
	PS2_Cmd(Comd[1]);									  // 请求数据
	for (byte = 2; byte < 9; byte++)					  // 开始接收数据
	{
		for (ref = 0x01; ref < 0x100; ref <<= 1)
		{
			HAL_GPIO_WritePin(GPIOE, GPIO_PIN_13, GPIO_PIN_SET); // CLK_H
			DWT_Delay(0.000005f);
			HAL_GPIO_WritePin(GPIOE, GPIO_PIN_13, GPIO_PIN_RESET); // CLK_L
			DWT_Delay(0.000005f);
			HAL_GPIO_WritePin(GPIOE, GPIO_PIN_13, GPIO_PIN_SET); // CLK_H
			if (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0))			 // DI
				Data[byte] = ref | Data[byte];
		}
		DWT_Delay(0.000016f);
	}
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_SET); // CS_H

	// 更新模式
	switch (Data[1])
	{
	case PS2_MODE_DIGITAL:
	case PS2_MODE_ANALOG:
	case PS2_MODE_VIBRATION:
	case PS2_MODE_CONFIG:
		ps2_mode = Data[1];
		break;
	default:
		ps2_mode = PS2_MODE_ERROR;
	}
}

// 将读取到的PS2数据进行处理,只提取按键值
// 只有当对应按键按下时对应位为0，未按下为1
uint8_t PS2_DataKey()
{
	uint8_t index;

	PS2_ClearData();
	PS2_ReadData();

	Handkey = (Data[4] << 8) | Data[3]; // 提取16个按键  按下为0，未按下为1
	for (index = 0; index < 16; index++)
	{
		if ((Handkey & (1 << (MASK[index] - 1))) == 0)
			return index + 1;
	}
	return 0; // 没有任何按键按下
}

// 得到一个摇杆的模拟量	 范围0~256
uint8_t PS2_AnologData(uint8_t button)
{
	return Data[button];
}

// 清除数据缓冲区
void PS2_ClearData()
{
	uint8_t a;
	for (a = 0; a < 9; a++)
		Data[a] = 0x00;
}
/******************************************************
Function:    void PS2_Vibration(u8 motor1, u8 motor2)
Description: 手柄震动函数
Calls:		 void PS2_Cmd(u8 CMD);
Input: motor1:右侧小震动电机 0x00关，其它开
	   motor2:左侧大震动电机 0x40~0xFF 电机电压值越大 震动越强
******************************************************/
void PS2_Vibration(uint8_t motor1, uint8_t motor2)
{
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_RESET); // CS_L
	DWT_Delay(0.000016f);
	PS2_Cmd(0x01); // 开始命令
	PS2_Cmd(0x42); // 请求数据
	PS2_Cmd(0X00);
	PS2_Cmd(motor1);
	PS2_Cmd(motor2);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_SET); // CS_H
	DWT_Delay(0.000016f);
}

// short poll
void PS2_ShortPoll(void)
{
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_RESET); // CS_L
	DWT_Delay(0.000016f);
	PS2_Cmd(0x01);
	PS2_Cmd(0x42);
	PS2_Cmd(0X00);
	PS2_Cmd(0x00);
	PS2_Cmd(0x00);
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_SET); // CS_H
	DWT_Delay(0.000016f);
}

// 进入配置
void PS2_EnterConfing(void)
{
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_RESET); // CS_L
	DWT_Delay(0.000016f);
	PS2_Cmd(0x01);
	PS2_Cmd(0x43);
	PS2_Cmd(0X00);
	PS2_Cmd(0x01);
	PS2_Cmd(0x00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_SET); // CS_H
	DWT_Delay(0.000016f);
}
// 模拟模式设置
void PS2_TurnOnAnalogMode(void)
{
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_RESET); // CS_L
	PS2_Cmd(0x01);
	PS2_Cmd(0x44);
	PS2_Cmd(0X00);
	PS2_Cmd(0x01); // analog=0x01;digital=0x00  软件设置发送模式
	PS2_Cmd(0x03); // 0x03软件设置，并且不能通过手柄上的MODE键更改模式
				   // 0xEE软件设置，并且可以通过手柄上的MODE键更改模式
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_SET); // CS_H
	DWT_Delay(0.000016f);
}
// 配置震动
void PS2_VibrationMode(void)
{
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_RESET); // CS_L
	DWT_Delay(0.000016f);
	PS2_Cmd(0x01);
	PS2_Cmd(0x4D);
	PS2_Cmd(0X00);
	PS2_Cmd(0x00);
	PS2_Cmd(0X01);
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_SET); // CS_H
	DWT_Delay(0.000016f);
}
// 完成配置退出
void PS2_ExitConfing(void)
{
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_RESET); // CS_L
	DWT_Delay(0.000016f);
	PS2_Cmd(0x01);
	PS2_Cmd(0x43);
	PS2_Cmd(0X00);
	PS2_Cmd(0x00);
	PS2_Cmd(0x5A);
	PS2_Cmd(0x5A);
	PS2_Cmd(0x5A);
	PS2_Cmd(0x5A);
	PS2_Cmd(0x5A);
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_9, GPIO_PIN_SET); // CS_H
	DWT_Delay(0.000016f);
}

// 手柄配置初始化
void PS2_SetInit(void)
{
	PS2_ShortPoll();
	PS2_ShortPoll();
	PS2_ShortPoll();
	PS2_EnterConfing();		// 进入配置模式
	PS2_TurnOnAnalogMode(); // 开启红灯、模拟模式，并选择是否保存
	// PS2_VibrationMode();	// 开启震动模式
	PS2_ExitConfing(); // 完成配置并退出
}