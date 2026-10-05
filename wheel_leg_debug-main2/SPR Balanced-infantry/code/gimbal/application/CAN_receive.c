#include "CAN_receive.h"
#include "cmsis_os.h"
#include "detect_task.h"
#include "shoot_task.h"
#include "main.h"
#include "referee.h"
#include "gimbal_task.h"
#include "config.h"
#include <string.h>

extern CAN_HandleTypeDef hcan1;
extern CAN_HandleTypeDef hcan2;
extern gimbal_move_t gimbal_move;

static CAN_TxHeaderTypeDef yaw_tx_message;
static uint8_t yaw_can_send_data[8];
static CAN_TxHeaderTypeDef pitch_tx_message;
static uint8_t pitch_can_send_data[8];
static CAN_TxHeaderTypeDef shoot_tx_message;
static uint8_t shoot_can_send_data[8];
static CAN_TxHeaderTypeDef fricion_tx_message;
static uint8_t fricion_can_send_data[8];
static CAN_TxHeaderTypeDef chassis_tx_message;
static uint8_t chassis_can_send_data[8];
static CAN_TxHeaderTypeDef chassis_tx_superpower;
static uint8_t chassis_can_send_superpower[8];

MIT_Motor_t motor_gimbal[2];
motor_measure_t motor_shoot[3];
SuperCap_t SuperCap;

//数据限幅
static float Data_Clipping(float Data,float Data_Min,float Data_Max)
{
	if(Data>Data_Max)return Data_Max;
	else if(Data<Data_Min)return Data_Min;
	return Data;
}

//根据DM通信协议，将整型uint转换成浮点数float
static float DM_uint_to_float(int x,float x_min,float x_max,int bits)
{
	float span=x_max-x_min;
	float offset=x_min;
	return (float)(((float)x)*span/((float)(1<<bits)-1)+offset);
}

//根据DM通信协议，将浮点数float转换成整型uint
uint16_t DM_float_to_uint(float x,float x_min,float x_max,int bits)
{
	float span=x_max-x_min;
	float offset=x_min;
	return (uint16_t)((x-offset)/span*((float)((1<<bits)-1)));
}

void Calc_2006_Angle(motor_measure_t *motor)
{

	if (motor->ecd - motor->last_ecd > 4095.5)
	{
		motor->round--;
	}
	else if (motor->ecd - motor->last_ecd < -4095.5)
	{
		motor->round++;
	}	

	motor->angle = (float)(motor->round * 8191 + motor->ecd)/8191*3.1415926f*2/360*1;
}

void Calc_3508_Angle(motor_measure_t *motor)
{

	if (motor->ecd - motor->last_ecd > 4095.5)
	{
		motor->round--;
	}
	else if (motor->ecd - motor->last_ecd < -4095.5)
	{
		motor->round++;
	}	

	motor->angle = (float)(motor->round * 8191 + motor->ecd)/8191*3.1415926f*2/3591*187;
}

void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
  CAN_RxHeaderTypeDef rx_header;

  uint8_t rx_data[8];

  HAL_CAN_GetRxMessage(hcan, CAN_RX_FIFO0, &rx_header, rx_data);

  if (hcan == &hcan1)
  {
    switch (rx_header.StdId)
    {
				case 0x05:
				{
					uint16_t Position_int=(int16_t)((uint16_t)rx_data[1]<<8|rx_data[2]);
					uint16_t Speed_int=(int16_t)((uint16_t)rx_data[3]<<4|rx_data[4]>>4);
					uint16_t Torque_int=(int16_t)((uint16_t)(rx_data[4] & 0x0F)<<8|rx_data[5]);
					
					motor_gimbal[0].Status=(rx_data[0]>>4) & 0x0F;
					motor_gimbal[0].Position=DM_uint_to_float(Position_int,DM_4310_PMIN,DM_4310_PMAX,16);
					motor_gimbal[0].Speed=DM_uint_to_float(Speed_int,DM_4310_VMIN,DM_4310_VMAX,12);
					motor_gimbal[0].Torque=DM_uint_to_float(Torque_int,DM_4310_TMIN,DM_4310_TMAX,12);
					detect_hook(YAW_GIMBAL_MOTOR_TOE);
					break;
				}
				case 0x207:
				{
					motor_shoot[0].last_ecd=motor_shoot[0].ecd;
					motor_shoot[0].temperate=rx_data[6];
					motor_shoot[0].ecd=(rx_data[0]<<8|rx_data[1]);
					motor_shoot[0].speed_rpm=(rx_data[2]<<8|rx_data[3]);
					motor_shoot[0].given_current=(rx_data[4]<<8|rx_data[5]);
					
					motor_shoot[0].Torque=(float)(motor_shoot[0].given_current)/10000*10*0.18f;
					motor_shoot[0].speed_rads=motor_shoot[0].speed_rpm*3.14159/30/36*1/5*2;
					Calc_2006_Angle(&motor_shoot[0]);
					motor_shoot[0].last_angle=motor_shoot[0].angle;
					detect_hook(TRIGGER_SHOOT_MOTOR_TOE);
					break;
				}
				case 0x051: //超级电容控制板消息
				{
					SuperCap.errorCode=rx_data[0];
						uint8_t power_bytes[4] = {rx_data[1], rx_data[2], rx_data[3], rx_data[4]};
						memcpy(&SuperCap.chassisPower, power_bytes, 4);//需要调用<string.h>
					SuperCap.capEnergy=rx_data[7];
					detect_hook(SUPERPOWER);
					break;
				}
				case 0x031: //来自底盘的消息
				{
				gimbal_move.TwoBoardControlGimbal.LeftPhi0=(((int8_t)rx_data[0])*0.1f);
				gimbal_move.TwoBoardControlGimbal.LeftLength=(((int8_t)rx_data[1])*0.01f);
				gimbal_move.TwoBoardControlGimbal.RightPhi0=(((int8_t)rx_data[2])*0.1f);
				gimbal_move.TwoBoardControlGimbal.RightLength=(((int8_t)rx_data[3])*0.01f);
				gimbal_move.TwoBoardControlGimbal.ChassisState=rx_data[4];
					detect_hook(CHASSISTOGIMBAL);
					break;
				}
			default:
			{
				break;
			}
    }
  }
  else if (hcan == &hcan2)
  {
    switch (rx_header.StdId)
    {
				case 0x06:
				{
					uint16_t Position_int=(int16_t)((uint16_t)rx_data[1]<<8|rx_data[2]);
					uint16_t Speed_int=(int16_t)((uint16_t)rx_data[3]<<4|rx_data[4]>>4);
					uint16_t Torque_int=(int16_t)((uint16_t)(rx_data[4] & 0x0F)<<8|rx_data[5]);
					
					motor_gimbal[1].Status=(rx_data[0]>>4) & 0x0F;
					motor_gimbal[1].Position=DM_uint_to_float(Position_int,DM_4310_PMIN,DM_4310_PMAX,16);
					motor_gimbal[1].Speed=DM_uint_to_float(Speed_int,DM_4310_VMIN,DM_4310_VMAX,12);
					motor_gimbal[1].Torque=DM_uint_to_float(Torque_int,DM_4310_TMIN,DM_4310_TMAX,12);
					detect_hook(PITCH_GIMBAL_MOTOR_TOE);
					break;
				}
				case 0x205:
				{
					motor_shoot[1].last_ecd=motor_shoot[0].ecd;
					motor_shoot[1].temperate=rx_data[6];
					motor_shoot[1].ecd=(rx_data[0]<<8|rx_data[1]);
					motor_shoot[1].speed_rpm=(rx_data[2]<<8|rx_data[3]);
					motor_shoot[1].given_current=(rx_data[4]<<8|rx_data[5]);
					
					motor_shoot[1].Torque=(float)(motor_shoot[1].given_current)/16384*20*0.3f;
					motor_shoot[1].speed_rads=motor_shoot[1].speed_rpm*3.14159/30;
					Calc_3508_Angle(&motor_shoot[1]);
					motor_shoot[1].last_angle=motor_shoot[1].angle;
					detect_hook(LEFT_SHOOT_MOTOR_TOE);
					break;
				}
				case 0x206:
				{
					motor_shoot[2].last_ecd=motor_shoot[0].ecd;
					motor_shoot[2].temperate=rx_data[6];
					motor_shoot[2].ecd=(rx_data[0]<<8|rx_data[1]);
					motor_shoot[2].speed_rpm=(rx_data[2]<<8|rx_data[3]);
					motor_shoot[2].given_current=(rx_data[4]<<8|rx_data[5]);
					
					motor_shoot[2].Torque=(float)(motor_shoot[2].given_current)/16384*20*0.3f;
					motor_shoot[2].speed_rads=motor_shoot[2].speed_rpm*3.14159/30;
					Calc_3508_Angle(&motor_shoot[2]);
					motor_shoot[2].last_angle=motor_shoot[2].angle;
					detect_hook(RIGHT_SHOOT_MOTOR_TOE);
					break;
				}
				
			default:
			{
				break;
			}
    }
  }
}

extern void CAN_cmd_SuperPower(void)
{
	uint16_t power = gimbal_move.TwoBoardControlGimbal.chassis_power_limit;
	uint16_t energy = gimbal_move.TwoBoardControlGimbal.buffer_energy_chassis;
	
  uint32_t send_mail_box;
  chassis_tx_superpower.StdId = 0x061;
  chassis_tx_superpower.IDE = CAN_ID_STD;
  chassis_tx_superpower.RTR = CAN_RTR_DATA;
  chassis_tx_superpower.DLC = 0x08;	
	// 小端序：低地址放低字节
	chassis_can_send_superpower[0] = 0x01;   //希望 enableDCDC=1，其它位为0
	chassis_can_send_superpower[1] = power & 0xFF;        // 功率低8位
	chassis_can_send_superpower[2] = (power >> 8) & 0xFF; // 功率高8位
	chassis_can_send_superpower[3] = energy & 0xFF;       // 能量低8位
	chassis_can_send_superpower[4] = (energy >> 8) & 0xFF;// 能量高8位
  chassis_can_send_superpower[5] = 0;
  chassis_can_send_superpower[6] = 0;
  chassis_can_send_superpower[7] = 0;
  HAL_CAN_AddTxMessage(&hcan1, &chassis_tx_superpower, chassis_can_send_superpower, &send_mail_box);
}

extern void CAN_cmd_chassis(void)
{
	int32_t yaw=(int32_t)(gimbal_move.TwoBoardControlGimbal.relative_angle_yaw*1000000);
	
  uint32_t send_mail_box;
  chassis_tx_message.StdId = 0x081;
  chassis_tx_message.IDE = CAN_ID_STD;
  chassis_tx_message.RTR = CAN_RTR_DATA;
  chassis_tx_message.DLC = 0x08;
  chassis_can_send_data[0] = (int8_t)(gimbal_move.TwoBoardControlGimbal.vx_set*10);
  chassis_can_send_data[1] = (int8_t)(gimbal_move.TwoBoardControlGimbal.vy_set*10);
  chassis_can_send_data[2] = (int8_t)(gimbal_move.TwoBoardControlGimbal.roll_set*100);
  chassis_can_send_data[3] = (int8_t)gimbal_move.TwoBoardControlGimbal.legL_mode;
  chassis_can_send_data[4] = gimbal_move.TwoBoardControlGimbal.mode;
  chassis_can_send_data[5] = ((yaw) >> 24)&0xff;
  chassis_can_send_data[6] = ((yaw) >> 16)&0xff;
  chassis_can_send_data[7] = ((yaw) >>  8)&0xff;
  HAL_CAN_AddTxMessage(&hcan1, &chassis_tx_message, chassis_can_send_data, &send_mail_box);
}

extern void CAN_cmd_chassis2(void)
{
  uint32_t send_mail_box;
  chassis_tx_message.StdId = 0x082;
  chassis_tx_message.IDE = CAN_ID_STD;
  chassis_tx_message.RTR = CAN_RTR_DATA;
  chassis_tx_message.DLC = 0x08;
  chassis_can_send_data[0] = gimbal_move.TwoBoardControlGimbal.chassis_power_limit<<8;
  chassis_can_send_data[1] = gimbal_move.TwoBoardControlGimbal.chassis_power_limit;
  chassis_can_send_data[2] = gimbal_move.TwoBoardControlGimbal.buffer_energy_chassis<<8;
  chassis_can_send_data[3] = gimbal_move.TwoBoardControlGimbal.buffer_energy_chassis;
  chassis_can_send_data[4] = (int8_t)(gimbal_move.TwoBoardControlGimbal.LeftLengthSet*100);
  chassis_can_send_data[5] = (int8_t)(gimbal_move.TwoBoardControlGimbal.LeftPhi0Set*10);
  chassis_can_send_data[6] = (int8_t)(gimbal_move.TwoBoardControlGimbal.RightLengthSet*100);
  chassis_can_send_data[7] = (int8_t)(gimbal_move.TwoBoardControlGimbal.RightPhi0Set*10);
  HAL_CAN_AddTxMessage(&hcan1, &chassis_tx_message, chassis_can_send_data, &send_mail_box);
}

extern void CAN_cmd_yaw_init(void)
{
  uint32_t send_mail_box;
  yaw_tx_message.StdId = 0x05;
  yaw_tx_message.IDE = CAN_ID_STD;
  yaw_tx_message.RTR = CAN_RTR_DATA;
  yaw_tx_message.DLC = 0x08;
  yaw_can_send_data[0] = 0xFF;
  yaw_can_send_data[1] = 0xFF;
  yaw_can_send_data[2] = 0xFF;
  yaw_can_send_data[3] = 0xFF;
  yaw_can_send_data[4] = 0xFF;
  yaw_can_send_data[5] = 0xFF;
  yaw_can_send_data[6] = 0xFF;
  yaw_can_send_data[7] = 0xFC;
  HAL_CAN_AddTxMessage(&hcan1, &yaw_tx_message, yaw_can_send_data, &send_mail_box);
}

extern void CAN_cmd_yaw_clean(void)
{
  uint32_t send_mail_box;
  yaw_tx_message.StdId = 0x05;
  yaw_tx_message.IDE = CAN_ID_STD;
  yaw_tx_message.RTR = CAN_RTR_DATA;
  yaw_tx_message.DLC = 0x08;
  yaw_can_send_data[0] = 0xFF;
  yaw_can_send_data[1] = 0xFF;
  yaw_can_send_data[2] = 0xFF;
  yaw_can_send_data[3] = 0xFF;
  yaw_can_send_data[4] = 0xFF;
  yaw_can_send_data[5] = 0xFF;
  yaw_can_send_data[6] = 0xFF;
  yaw_can_send_data[7] = 0xFB;
  HAL_CAN_AddTxMessage(&hcan1, &yaw_tx_message, yaw_can_send_data, &send_mail_box);
}

extern void CAN_cmd_gimbal_yaw(float Tor)
{
  Tor=Data_Clipping(Tor,DM_4310_TMIN,DM_4310_TMAX);
  uint16_t Torque=DM_float_to_uint(Tor,DM_4310_TMIN,DM_4310_TMAX,12);
	
  uint32_t send_mail_box;
  yaw_tx_message.StdId = 0x05;
  yaw_tx_message.IDE = CAN_ID_STD;
  yaw_tx_message.RTR = CAN_RTR_DATA;
  yaw_tx_message.DLC = 0x08;
  yaw_can_send_data[0] = 0;
  yaw_can_send_data[1] = 0;
  yaw_can_send_data[2] = 0;
  yaw_can_send_data[3] = 0;
  yaw_can_send_data[4] = 0;
  yaw_can_send_data[5] = 0;
  yaw_can_send_data[6] = Torque>>8;
  yaw_can_send_data[7] = Torque;
  HAL_CAN_AddTxMessage(&hcan1, &yaw_tx_message, yaw_can_send_data, &send_mail_box);
}

extern void CAN_cmd_pitch_init(void)
{
  uint32_t send_mail_box;
  pitch_tx_message.StdId = 0x06;
  pitch_tx_message.IDE = CAN_ID_STD;
  pitch_tx_message.RTR = CAN_RTR_DATA;
  pitch_tx_message.DLC = 0x08;
  pitch_can_send_data[0] = 0xFF;
  pitch_can_send_data[1] = 0xFF;
  pitch_can_send_data[2] = 0xFF;
  pitch_can_send_data[3] = 0xFF;
  pitch_can_send_data[4] = 0xFF;
  pitch_can_send_data[5] = 0xFF;
  pitch_can_send_data[6] = 0xFF;
  pitch_can_send_data[7] = 0xFC;
  HAL_CAN_AddTxMessage(&hcan2, &pitch_tx_message, pitch_can_send_data, &send_mail_box);
}

extern void CAN_cmd_pitch_clean(void)
{
  uint32_t send_mail_box;
  pitch_tx_message.StdId = 0x06;
  pitch_tx_message.IDE = CAN_ID_STD;
  pitch_tx_message.RTR = CAN_RTR_DATA;
  pitch_tx_message.DLC = 0x08;
  pitch_can_send_data[0] = 0xFF;
  pitch_can_send_data[1] = 0xFF;
  pitch_can_send_data[2] = 0xFF;
  pitch_can_send_data[3] = 0xFF;
  pitch_can_send_data[4] = 0xFF;
  pitch_can_send_data[5] = 0xFF;
  pitch_can_send_data[6] = 0xFF;
  pitch_can_send_data[7] = 0xFB;
  HAL_CAN_AddTxMessage(&hcan2, &pitch_tx_message, pitch_can_send_data, &send_mail_box);
}

extern void CAN_cmd_gimbal_pitch(float Tor)
{
  Tor=Data_Clipping(Tor,DM_4310_TMIN,DM_4310_TMAX);
  uint16_t Torque=DM_float_to_uint(Tor,DM_4310_TMIN,DM_4310_TMAX,12);
	
  uint32_t send_mail_box;
  pitch_tx_message.StdId = 0x06;
  pitch_tx_message.IDE = CAN_ID_STD;
  pitch_tx_message.RTR = CAN_RTR_DATA;
  pitch_tx_message.DLC = 0x08;
  pitch_can_send_data[0] = 0;
  pitch_can_send_data[1] = 0;
  pitch_can_send_data[2] = 0;
  pitch_can_send_data[3] = 0;
  pitch_can_send_data[4] = 0;
  pitch_can_send_data[5] = 0;
  pitch_can_send_data[6] = Torque>>8;
  pitch_can_send_data[7] = Torque;
  HAL_CAN_AddTxMessage(&hcan2, &pitch_tx_message, pitch_can_send_data, &send_mail_box);
}

extern void CAN_cmd_shoot(int16_t shoot)
{
  uint32_t send_mail_box;
  shoot_tx_message.StdId = 0x1FF;
  shoot_tx_message.IDE = CAN_ID_STD;
  shoot_tx_message.RTR = CAN_RTR_DATA;
  shoot_tx_message.DLC = 0x08;
  shoot_can_send_data[0] = 0;
  shoot_can_send_data[1] = 0;
  shoot_can_send_data[2] = 0;
  shoot_can_send_data[3] = 0;
  shoot_can_send_data[4] = (shoot >> 8);
  shoot_can_send_data[5] = shoot;
  shoot_can_send_data[6] = 0;
  shoot_can_send_data[7] = 0;
  HAL_CAN_AddTxMessage(&hcan1, &shoot_tx_message, shoot_can_send_data, &send_mail_box);
}

extern void CAN_cmd_fricion(int16_t left_friction, int16_t rigit_friction)
{
  uint32_t send_mail_box;
  fricion_tx_message.StdId = 0x1FF;
  fricion_tx_message.IDE = CAN_ID_STD;
  fricion_tx_message.RTR = CAN_RTR_DATA;
  fricion_tx_message.DLC = 0x08;
  fricion_can_send_data[0] = (left_friction >> 8);
  fricion_can_send_data[1] = left_friction;
  fricion_can_send_data[2] = (rigit_friction >> 8);
  fricion_can_send_data[3] = rigit_friction;
  fricion_can_send_data[4] = 0;
  fricion_can_send_data[5] = 0;
  fricion_can_send_data[6] = 0;
  fricion_can_send_data[7] = 0;
  HAL_CAN_AddTxMessage(&hcan2, &fricion_tx_message, fricion_can_send_data, &send_mail_box);
}

const motor_measure_t *trigger_measure_point(void)
{
	return &motor_shoot[0];
}
const motor_measure_t *left_motor_measure_point(void)
{
	return &motor_shoot[1];
}
const motor_measure_t *right_motor_measure_point(void)
{
	return &motor_shoot[2];
}
const MIT_Motor_t *yaw_motor_measure_point(void)
{
	return &motor_gimbal[0];
}
const MIT_Motor_t *pitch_motor_measure_point(void)
{
	return &motor_gimbal[1];
}
