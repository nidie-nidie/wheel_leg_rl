/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : Motor.c
 * @brief          : Motor interfaces functions
 * @author         : GrassFan Wang
 * @date           : 2025/01/22
 * @version        : v1.0
 ******************************************************************************
 * @attention      : None
 ******************************************************************************
 */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "Motor.h"

/**
 * @brief The structure that contains the Information of chassis motor.Use DJI M3508 motor.
 */
LK_Motor_Info_Typedef LK_9025_Motor[2] = {

	[0] = {
		.Control_Mode = LK_TORQUE,
		.FDCANFrame = {
			.TxIdentifier = 0x04,
			.RxIdentifier = 0x144, // 0x140 + can_id
		},
		.last_fdb_time = 0,
		.offline = true,
	},
	[1] = {
		.Control_Mode = LK_TORQUE,
		.FDCANFrame = {
			.TxIdentifier = 0x05,
			.RxIdentifier = 0x145,
		},
		.last_fdb_time = 0,
		.offline = true,
	},
};
//------------------------------------------------------------------------------

/**
 * @brief The structure that contains the Information of joint motor.Use DM 8009p motor.
 */
DM_Motor_Info_Typedef DM_8009_Motor[4] = {

	[0] = {
		.Control_Mode = DM_MIT,
		.Param_Range = {
			.P_MAX = DM_P_MAX,
			.P_MIN = DM_P_MIN,
			.V_MAX = DM_V_MAX,
			.V_MIN = DM_V_MIN,
			.T_MAX = DM_T_MAX,
			.T_MIN = DM_T_MIN,
		},
		.FDCANFrame = {
			.TxIdentifier = 0x01, // can id
			.RxIdentifier = 0x11, // master id
		},
		.last_fdb_time = 0,
		.offline = true,
	},

	[1] = {
		.Control_Mode = DM_MIT,
		.Param_Range = {
			.P_MAX = DM_P_MAX,
			.P_MIN = DM_P_MIN,
			.V_MAX = DM_V_MAX,
			.V_MIN = DM_V_MIN,
			.T_MAX = DM_T_MAX,
			.T_MIN = DM_T_MIN,
		},
		.FDCANFrame = {
			.TxIdentifier = 0x02,
			.RxIdentifier = 0x12,
		},
		.last_fdb_time = 0,
		.offline = true,
	},

	[2] = {
		.Control_Mode = DM_MIT,
		.Param_Range = {
			.P_MAX = DM_P_MAX,
			.P_MIN = DM_P_MIN,
			.V_MAX = DM_V_MAX,
			.V_MIN = DM_V_MIN,
			.T_MAX = DM_T_MAX,
			.T_MIN = DM_T_MIN,
		},
		.FDCANFrame = {
			.TxIdentifier = 0x03,
			.RxIdentifier = 0x13,
		},
		.last_fdb_time = 0,
		.offline = true,
	},

	[3] = {
		.Control_Mode = DM_MIT,
		.Param_Range = {
			.P_MAX = DM_P_MAX,
			.P_MIN = DM_P_MIN,
			.V_MAX = DM_V_MAX,
			.V_MIN = DM_V_MIN,
			.T_MAX = DM_T_MAX,
			.T_MIN = DM_T_MIN,
		},
		.FDCANFrame = {
			.TxIdentifier = 0x06,
			.RxIdentifier = 0x16,
		},
		.last_fdb_time = 0,
		.offline = true,
	},

};
//------------------------------------------------------------------------------

/**
 * @brief  将编码器值转换为角度(累计，最大到float极值)
 */
static float DJI_Motor_Encoder_To_Anglesum(DJI_Motor_Data_Typedef *, float, uint16_t);
/**
 * @brief  将编码器值转换为角度(范围 -180~180度)
 */
static float DJI_Motor_Encoder_To_Angle(DJI_Motor_Data_Typedef *, float, uint16_t);

float F_Loop_Constrain(float Input, float Min_Value, float Max_Value);

static float uint_to_float(int X_int, float X_min, float X_max, int Bits);

static int float_to_uint(float x, float x_min, float x_max, int bits);

//------------------------------------------------------------------------------

/**
 * @brief  Update the DJI motor Information
 * @param  Identifier  pointer to the specifies the standard identifier.
 * @param  Rx_Buf  pointer to the can receive data
 * @param  DJI_Motor pointer to a DJI_Motor_Info_t structure
 *         that contains the information of DJI motor
 * @retval None
 */
void DJI_Motor_Info_Update(uint32_t *Identifier, uint8_t *Rx_Buf, DJI_Motor_Info_Typedef *DJI_Motor)
{
	/* check the Identifier */
	if (*Identifier != DJI_Motor->FDCANFrame.RxIdentifier)
		return;

	/* transforms the  general motor data */
	DJI_Motor->Data.Temperature = Rx_Buf[6];
	DJI_Motor->Data.Encoder = ((int16_t)Rx_Buf[0] << 8 | (int16_t)Rx_Buf[1]);
	DJI_Motor->Data.Velocity = ((int16_t)Rx_Buf[2] << 8 | (int16_t)Rx_Buf[3]);
	DJI_Motor->Data.Current = ((int16_t)Rx_Buf[4] << 8 | (int16_t)Rx_Buf[5]);

	/* transform the Encoder to angle */
	switch (DJI_Motor->Type)
	{
	case DJI_GM6020:

		DJI_Motor->Data.Angle = DJI_Motor_Encoder_To_Angle(&DJI_Motor->Data, 1.f, 8192); // 6020电机减速比为1:1，拥有绝对位置
		break;

	case DJI_M3508:
		DJI_Motor->Data.Angle = DJI_Motor_Encoder_To_Angle(&DJI_Motor->Data, 3591.f / 187.f, 8192);
		break;

	case DJI_M2006:
		DJI_Motor->Data.Angle = DJI_Motor_Encoder_To_Angle(&DJI_Motor->Data, 36.f, 8192);
		break;

	default:
		break;
	}
}
//------------------------------------------------------------------------------

/**
 * @brief  float loop constrain
 * @param  Input    the specified variables
 * @param  minValue minimum number of the specified variables
 * @param  maxValue maximum number of the specified variables
 * @retval variables
 */
float F_Loop_Constrain(float Input, float Min_Value, float Max_Value)
{
	if (Max_Value < Min_Value)
	{
		return Input;
	}

	float len = Max_Value - Min_Value;

	if (Input > Max_Value)
	{
		do
		{
			Input -= len;
		} while (Input > Max_Value);
	}
	else if (Input < Min_Value)
	{
		do
		{
			Input += len;
		} while (Input < Min_Value);
	}
	return Input;
}
//------------------------------------------------------------------------------

/**
 * @brief  transform the Encoder(0-8192) to anglesum(3.4E38)
 * @param  *Info        pointer to a Motor_Data_Typedef structure that
 *					             contains the infomation for the specified motor
 * @param  torque_ratio the specified motor torque ratio
 * @param  MAXEncoder   the specified motor max Encoder number
 * @retval anglesum
 */
static float DJI_Motor_Encoder_To_Anglesum(DJI_Motor_Data_Typedef *Data, float Torque_Ratio, uint16_t MAXEncoder)
{
	float res1 = 0, res2 = 0;

	if (Data == NULL)
		return 0;

	/* Judge the motor Initlized */
	if (Data->Initlized != true)
	{
		/* update the last Encoder */
		Data->Last_Encoder = Data->Encoder;

		/* reset the angle */
		Data->Angle = 0;

		/* Set the init flag */
		Data->Initlized = true;
	}

	/* get the possiable min Encoder err */
	if (Data->Encoder < Data->Last_Encoder)
	{
		res1 = Data->Encoder - Data->Last_Encoder + MAXEncoder;
	}
	else if (Data->Encoder > Data->Last_Encoder)
	{
		res1 = Data->Encoder - Data->Last_Encoder - MAXEncoder;
	}
	res2 = Data->Encoder - Data->Last_Encoder;

	/* update the last Encoder */
	Data->Last_Encoder = Data->Encoder;

	/* transforms the Encoder data to tolangle */
	if (fabsf(res1) > fabsf(res2))
	{
		Data->Angle += (float)res2 / (MAXEncoder * Torque_Ratio) * 360.f;
	}
	else
	{
		Data->Angle += (float)res1 / (MAXEncoder * Torque_Ratio) * 360.f;
	}

	return Data->Angle;
}
//------------------------------------------------------------------------------

/**
 * @brief  transform the Encoder(0-8192) to angle(-180-180)
 * @param  *Data        pointer to a Motor_Data_Typedef structure that
 *					             contains the Data for the specified motor
 * @param  torque_ratio the specified motor torque ratio
 * @param  MAXEncoder   the specified motor max Encoder number
 * @retval angle
 */
float DJI_Motor_Encoder_To_Angle(DJI_Motor_Data_Typedef *Data, float torque_ratio, uint16_t MAXEncoder)
{
	float Encoder_Err = 0.f;

	/* check the motor init */
	if (Data->Initlized != true)
	{
		/* update the last Encoder */
		Data->Last_Encoder = Data->Encoder;

		/* reset the angle */
		Data->Angle = Data->Encoder / (MAXEncoder * torque_ratio) * 360.f;

		/* config the init flag */
		Data->Initlized = true;
	}

	Encoder_Err = Data->Encoder - Data->Last_Encoder;

	/* 0 -> MAXEncoder */
	if (Encoder_Err > MAXEncoder * 0.5f)
	{
		Data->Angle += (float)(Encoder_Err - MAXEncoder) / (MAXEncoder * torque_ratio) * 360.f;
	}
	/* MAXEncoder-> 0 */
	else if (Encoder_Err < -MAXEncoder * 0.5f)
	{
		Data->Angle += (float)(Encoder_Err + MAXEncoder) / (MAXEncoder * torque_ratio) * 360.f;
	}
	else
	{
		Data->Angle += (float)(Encoder_Err) / (MAXEncoder * torque_ratio) * 360.f;
	}

	/* update the last Encoder */
	Data->Last_Encoder = Data->Encoder;

	/* loop constrain */
	Data->Angle = F_Loop_Constrain(Data->Angle, -180.f, 180.f);

	return Data->Angle;
}

/**
 * ...
 * @param  *FDCAN_TxFrame: pointer to the FDCAN_TxFrame_TypeDef.
 * @param  *DM_Motor: pointer to the DM_Motor
 * @param  CMD: Transmit Command  (DM_Motor_CMD_Type_e)
 * @retval None
 */
void DM_Motor_Command(FDCAN_TxFrame_TypeDef *FDCAN_TxFrame, DM_Motor_Info_Typedef *DM_Motor, uint8_t CMD)
{

	FDCAN_TxFrame->Header.Identifier = DM_Motor->FDCANFrame.TxIdentifier + DM_MODE_MIT;

	FDCAN_TxFrame->Data[0] = 0xFF;
	FDCAN_TxFrame->Data[1] = 0xFF;
	FDCAN_TxFrame->Data[2] = 0xFF;
	FDCAN_TxFrame->Data[3] = 0xFF;
	FDCAN_TxFrame->Data[4] = 0xFF;
	FDCAN_TxFrame->Data[5] = 0xFF;
	FDCAN_TxFrame->Data[6] = 0xFF;

	switch (CMD)
	{

	case DM_Motor_Enable:
		FDCAN_TxFrame->Data[7] = 0xFC;
		break;

	case DM_Motor_Disable:
		FDCAN_TxFrame->Data[7] = 0xFD;
		break;

	case DM_Motor_Save_Zero_Position:
		FDCAN_TxFrame->Data[7] = 0xFE;
		break;

	default:
		break;
	}

	USER_FDCAN_AddMessageToTxFifoQ(FDCAN_TxFrame);
}

/**
 * @brief  CAN Transmit DM motor Information
 * @param  *FDCAN_TxFrame  pointer to the FDCAN_TxFrame_TypeDef.
 * @param  *DM_Motor  pointer to the DM_Motor
 * @param  Postion Velocity KP KD Torgue: Target
 * @retval None
 */
void DM_Motor_CAN_TxMessage(FDCAN_TxFrame_TypeDef *FDCAN_TxFrame, DM_Motor_Info_Typedef *DM_Motor, float Postion, float Velocity, float KP, float KD, float Torque)
{

	if (DM_Motor->Control_Mode == DM_MIT)
	{

		uint16_t Postion_Tmp, Velocity_Tmp, Torque_Tmp, KP_Tmp, KD_Tmp;

		Postion_Tmp = float_to_uint(Postion, -DM_Motor->Param_Range.P_MAX, DM_Motor->Param_Range.P_MAX, 16);
		Velocity_Tmp = float_to_uint(Velocity, -DM_Motor->Param_Range.V_MAX, DM_Motor->Param_Range.V_MAX, 12);
		Torque_Tmp = float_to_uint(Torque, -DM_Motor->Param_Range.T_MAX, DM_Motor->Param_Range.T_MAX, 12);
		KP_Tmp = float_to_uint(KP, DM_KP_MIN, DM_KP_MAX, 12);
		KD_Tmp = float_to_uint(KD, DM_KD_MIN, DM_KD_MAX, 12);

		FDCAN_TxFrame->Header.Identifier = DM_Motor->FDCANFrame.TxIdentifier + DM_MODE_MIT;
		// FDCAN_TxFrame->Header.DataLength = 8; // DLC = 8

		FDCAN_TxFrame->Data[0] = (uint8_t)(Postion_Tmp >> 8);
		FDCAN_TxFrame->Data[1] = (uint8_t)(Postion_Tmp);
		FDCAN_TxFrame->Data[2] = (uint8_t)(Velocity_Tmp >> 4);
		FDCAN_TxFrame->Data[3] = (uint8_t)((Velocity_Tmp & 0x0F) << 4) | (uint8_t)(KP_Tmp >> 8);
		FDCAN_TxFrame->Data[4] = (uint8_t)(KP_Tmp);
		FDCAN_TxFrame->Data[5] = (uint8_t)(KD_Tmp >> 4);
		FDCAN_TxFrame->Data[6] = (uint8_t)((KD_Tmp & 0x0F) << 4) | (uint8_t)(Torque_Tmp >> 8);
		FDCAN_TxFrame->Data[7] = (uint8_t)(Torque_Tmp);
	}
	else if (DM_Motor->Control_Mode == DM_POSITION_VELOCITY)
	{

		uint8_t *Postion_Tmp, *Velocity_Tmp;

		Postion_Tmp = (uint8_t *)&Postion;
		Velocity_Tmp = (uint8_t *)&Velocity;

		FDCAN_TxFrame->Header.Identifier = DM_Motor->FDCANFrame.TxIdentifier + DM_MODE_POS;
		// FDCAN_TxFrame->Header.DataLength = 8; // DLC = 8

		FDCAN_TxFrame->Data[0] = *(Postion_Tmp);
		FDCAN_TxFrame->Data[1] = *(Postion_Tmp + 1);
		FDCAN_TxFrame->Data[2] = *(Postion_Tmp + 2);
		FDCAN_TxFrame->Data[3] = *(Postion_Tmp + 3);
		FDCAN_TxFrame->Data[4] = *(Velocity_Tmp);
		FDCAN_TxFrame->Data[5] = *(Velocity_Tmp + 1);
		FDCAN_TxFrame->Data[6] = *(Velocity_Tmp + 2);
		FDCAN_TxFrame->Data[7] = *(Velocity_Tmp + 3);
	}
	else if (DM_Motor->Control_Mode == DM_VELOCITY)
	{

		uint8_t *Velocity_Tmp;
		Velocity_Tmp = (uint8_t *)&Velocity;

		FDCAN_TxFrame->Header.Identifier = DM_Motor->FDCANFrame.TxIdentifier + DM_MODE_SPEED;
		// FDCAN_TxFrame->Header.DataLength = 4; // DLC = 4

		FDCAN_TxFrame->Data[0] = *(Velocity_Tmp);
		FDCAN_TxFrame->Data[1] = *(Velocity_Tmp + 1);
		FDCAN_TxFrame->Data[2] = *(Velocity_Tmp + 2);
		FDCAN_TxFrame->Data[3] = *(Velocity_Tmp + 3);
		FDCAN_TxFrame->Data[4] = 0;
		FDCAN_TxFrame->Data[5] = 0;
		FDCAN_TxFrame->Data[6] = 0;
		FDCAN_TxFrame->Data[7] = 0;
	}

	USER_FDCAN_AddMessageToTxFifoQ(FDCAN_TxFrame);
}
//------------------------------------------------------------------------------

/**
 * @brief  Update the DM_Motor Information
 * @param  Identifier:  pointer to the specifies the standard identifier.
 * @param  Rx_Buf:  pointer to the can receive data
 * @param  DM_Motor: pointer to a DM_Motor_Info_Typedef structure that contains the information of DM_Motor
 * @retval None
 */
void DM_Motor_Info_Update(uint32_t *Identifier, uint8_t *Rx_Buf, DM_Motor_Info_Typedef *DM_Motor)
{

	if (*Identifier != DM_Motor->FDCANFrame.RxIdentifier)
		return;

	DM_Motor->Data.State = Rx_Buf[0] >> 4;
	DM_Motor->Data.P_int = ((uint16_t)(Rx_Buf[1]) << 8) | ((uint16_t)(Rx_Buf[2]));
	DM_Motor->Data.V_int = ((uint16_t)(Rx_Buf[3]) << 4) | ((uint16_t)(Rx_Buf[4]) >> 4);
	DM_Motor->Data.T_int = ((uint16_t)(Rx_Buf[4] & 0xF) << 8) | ((uint16_t)(Rx_Buf[5]));
	DM_Motor->Data.Torque = uint_to_float(DM_Motor->Data.T_int, -DM_Motor->Param_Range.T_MAX, DM_Motor->Param_Range.T_MAX, 12);
	DM_Motor->Data.Position = uint_to_float(DM_Motor->Data.P_int, -DM_Motor->Param_Range.P_MAX, DM_Motor->Param_Range.P_MAX, 16);
	DM_Motor->Data.Velocity = uint_to_float(DM_Motor->Data.V_int, -DM_Motor->Param_Range.V_MAX, DM_Motor->Param_Range.V_MAX, 12);

	DM_Motor->Data.Temperature_MOS = (float)(Rx_Buf[6]);
	DM_Motor->Data.Temperature_Rotor = (float)(Rx_Buf[7]);

	// check the motor stable running
	uint32_t now = HAL_GetTick();
	if (now - DM_Motor->last_fdb_time > MOTOR_STABLE_RUNNING_TIME)
	{
		DM_Motor->offline = true;
	}
	else
	{
		DM_Motor->offline = false;
	}
	DM_Motor->last_fdb_time = now;
}
//------------------------------------------------------------------------------

/**
 * @brief  Send command to LK Motor
 * @param  *FDCAN_TxFrame: pointer to the FDCAN_TxFrame_TypeDef.
 * @param  *LK_Motor: pointer to the LK_Motor_Info_Typedef structure
 * @param  CMD: Transmit Command  (LK_Motor_CMD_Type_e)
 * @retval None
 */
void LK_Motor_Command(FDCAN_TxFrame_TypeDef *FDCAN_TxFrame, LK_Motor_Info_Typedef *LK_Motor, uint8_t CMD)
{

	FDCAN_TxFrame->Header.Identifier = LK_Motor->FDCANFrame.TxIdentifier + LK_STDID_OFFESET;

	FDCAN_TxFrame->Data[1] = 0x00;
	FDCAN_TxFrame->Data[2] = 0x00;
	FDCAN_TxFrame->Data[3] = 0x00;
	FDCAN_TxFrame->Data[4] = 0x00;
	FDCAN_TxFrame->Data[5] = 0x00;
	FDCAN_TxFrame->Data[6] = 0x00;
	FDCAN_TxFrame->Data[7] = 0x00;

	switch (CMD)
	{

	case LK_Motor_Enable:
		FDCAN_TxFrame->Data[0] = 0x88;
		break;

	case LK_Motor_Disable:
		FDCAN_TxFrame->Data[0] = 0x80;
		break;

	case LK_Motor_Stop:
		FDCAN_TxFrame->Data[0] = 0x81;
		break;

	default:
		break;
	}

	USER_FDCAN_AddMessageToTxFifoQ(FDCAN_TxFrame);
}
//------------------------------------------------------------------------------

/**
 * @brief  Update the LK_Motor Information
 * @param  Identifier:  pointer to the specifies the standard identifier.
 * @param  Rx_Buf:  pointer to the can receive data
 * @param  LK_Motor: pointer to a LK_Motor_Info_Typedef structure that contains the information of LK_Motor
 * @retval None
 */
void LK_Motor_Info_Update(uint32_t *Identifier, uint8_t *Rx_Buf, LK_Motor_Info_Typedef *LK_Motor)
{

	if (*Identifier != LK_Motor->FDCANFrame.RxIdentifier)
		return;

	LK_Motor->Data.temprature = Rx_Buf[1];
	LK_Motor->Data.iq = ((uint16_t)(Rx_Buf[3]) << 8) | ((uint16_t)(Rx_Buf[2]));
	LK_Motor->Data.speed = ((uint16_t)(Rx_Buf[5]) << 4) | (uint16_t)(Rx_Buf[4]);
	LK_Motor->Data.encoder = ((uint16_t)(Rx_Buf[7] & 0xF) << 8) | (uint16_t)(Rx_Buf[6]);

	LK_Motor->Data.Current = LK_Motor->Data.iq * LK_MF_CONTROL_TO_CURRENT;
	LK_Motor->Data.Position = uint_to_float(LK_Motor->Data.encoder, -M_PI, M_PI, 16);
	LK_Motor->Data.Velocity = LK_Motor->Data.speed * LK_DEGREE_TO_RAD;
	LK_Motor->Data.Temperature = (float)LK_Motor->Data.temprature;

	// check the motor stable running
	uint32_t now = HAL_GetTick();
	if (now - LK_Motor->last_fdb_time > MOTOR_STABLE_RUNNING_TIME)
	{
		LK_Motor->offline = true;
	}
	else
	{
		LK_Motor->offline = false;
	}
	LK_Motor->last_fdb_time = now;
}
//------------------------------------------------------------------------------

/**
 * @brief  CAN Transmit LK motor Information
 * @param  *FDCAN_TxFrame  pointer to the FDCAN_TxFrame_TypeDef.
 * @param  *LK_Motor  pointer to the LK_Motor
 * @param  Postion Velocity KP KD Torgue: Target
 * @retval None
 */
void LK_Motor_CAN_TxMessage(FDCAN_TxFrame_TypeDef *FDCAN_TxFrame, LK_Motor_Info_Typedef *LK_Motor, float Velocity, float Torque)
{

	if (LK_Motor->Control_Mode == LK_TORQUE)
	{
		int16_t iqControl = (int16_t)(fp32_constrain(Torque, LK_MIN_MF_TORQUE, LK_MAX_MF_TORQUE) / LK_TORQUE_COEFFICIENT *
									  LK_CURRENT_TO_MF_CONTROL);
		FDCAN_TxFrame->Header.Identifier = LK_Motor->FDCANFrame.TxIdentifier + LK_STDID_OFFESET;
		// FDCAN_TxFrame->Header.DataLength = 8; // DLC = 8
		FDCAN_TxFrame->Data[0] = 0xA1;
		FDCAN_TxFrame->Data[1] = 0x00;
		FDCAN_TxFrame->Data[2] = 0x00;
		FDCAN_TxFrame->Data[3] = 0x00;
		FDCAN_TxFrame->Data[4] = *(uint8_t *)(&iqControl);
		FDCAN_TxFrame->Data[5] = *((uint8_t *)(&iqControl) + 1);
		FDCAN_TxFrame->Data[6] = 0x00;
		FDCAN_TxFrame->Data[7] = 0x00;
	}
	else if (LK_Motor->Control_Mode == LK_VELOCITY)
	{
		int32_t speedControl = (int32_t)(Velocity * LK_RAD_TO_DEGREE * 100);
		FDCAN_TxFrame->Header.Identifier = LK_Motor->FDCANFrame.TxIdentifier + LK_STDID_OFFESET;
		// FDCAN_TxFrame->Header.DataLength = 8; // DLC = 8
		FDCAN_TxFrame->Data[0] = 0xA2;
		FDCAN_TxFrame->Data[1] = 0x00;
		FDCAN_TxFrame->Data[2] = 0x00;
		FDCAN_TxFrame->Data[3] = 0x00;
		FDCAN_TxFrame->Data[4] = *(uint8_t *)(&speedControl);
		FDCAN_TxFrame->Data[5] = *((uint8_t *)(&speedControl) + 1);
		FDCAN_TxFrame->Data[6] = *((uint8_t *)(&speedControl) + 2);
		FDCAN_TxFrame->Data[7] = *((uint8_t *)(&speedControl) + 3);
	}

	USER_FDCAN_AddMessageToTxFifoQ(FDCAN_TxFrame);
}
//------------------------------------------------------------------------------

/*	TOOLS	*/

/**
************************************************************************
* @brief:       uint_to_float: 将无符号整数映射转换为浮点数
* @param[in]:   X_int:     待转换的无符号整型值
* @param[in]:   X_min:     浮点目标范围最小值
* @param[in]:   X_max:     浮点目标范围最大值
* @param[in]:   Bits:      输入无符号整数的位宽
* @retval:      映射后的浮点数
* @details:     将 [0, 2^Bits-1] 区间内的整数线性映射到 [X_min, X_max] 区间
************************************************************************
**/
static float uint_to_float(int X_int, float X_min, float X_max, int Bits)
{

	float span = X_max - X_min;
	float offset = X_min;
	return ((float)X_int) * span / ((float)((1 << Bits) - 1)) + offset;
}

/**
************************************************************************
* @brief:       float_to_uint: 将浮点数映射转换为无符号整数
* @param[in]:   x:         待转换的浮点数
* @param[in]:   x_min:     浮点输入范围最小值
* @param[in]:   x_max:     浮点输入范围最大值
* @param[in]:   bits:      目标无符号整数位宽
* @retval:      映射后的无符号整型数值（以 int 返回）
* @details:     将 [x_min, x_max] 区间内的浮点数线性映射到 [0, 2^bits-1] 区间
************************************************************************
**/
static int float_to_uint(float x, float x_min, float x_max, int bits)
{

	float span = x_max - x_min;
	float offset = x_min;
	return (int)((x - offset) * ((float)((1 << bits) - 1)) / span);
}

// 限幅函数
fp32 fp32_constrain(fp32 Value, fp32 minValue, fp32 maxValue)
{
	if (Value < minValue)
		return minValue;
	else if (Value > maxValue)
		return maxValue;
	else
		return Value;
}

/**
 * @brief          扫描所有电机，检测是否有离线电机
 * @return         true: 有离线电机 false: 全部在线
 */
bool ScanOfflineMotor(void)
{
	for (uint32_t i = 0; i < 2; i++)
	{
		if (LK_9025_Motor[i].offline)
		{
			return true;
		}
	}
	for (uint32_t i = 0; i < 4; i++)
	{
		if (DM_8009_Motor[i].offline)
		{
			return true;
		}
	}
	return false;
}