#include "gimbal_behaviour.h"
#include "gimbal_task.h"
#include "arm_math.h"
#include "detect_task.h"
#include "user_lib.h"
#include "USART_receive.h"
#include "config.h"
#include "filter.h"
#include "referee.h"
#include "config.h"

static void gimbal_behavour_set(gimbal_move_t *gimbal_mode_set);
static void gimbal_zero_force_control(float *yaw, float *pitch, gimbal_move_t *gimbal_control_set);
static void gimbal_absolute_angle_control(float *yaw, float *pitch, gimbal_move_t *gimbal_control_set);
static void gimbal_auto_angle_control(float *yaw, float *pitch, gimbal_move_t *gimbal_control_set);
static void gimbal_return(float *yaw, float *pitch, gimbal_move_t *gimbal_control_set);

extern int Z_flag;
uint8_t ZeroToFollowFlag=0;
int A_flag = 0, A_flag_last = 0;
int D_flag = 0, D_flag_last = 0;

extern void gimbal_behaviour_mode_set(gimbal_move_t *gimbal_mode_set)
{
	if (gimbal_mode_set == NULL)
	{
		return;
	}
	//云台模式设置
	gimbal_behavour_set(gimbal_mode_set);
}

static void gimbal_behavour_set(gimbal_move_t *gimbal_mode_set)
{
	if (gimbal_mode_set == NULL)
	{
		return;
	}
	
	//通过遥控器设置云台模式，进而控制云台
	if (switch_is_up(rc_ctrl.rc.s[1])||PT_switch_is_up(remote_data.mode_sw)) //上
	{
//			__disable_irq();          // 关闭所有可屏蔽中断，如果进入到这里，给下板发送的消息也会暂停，进而底盘和头无力
//			HAL_NVIC_SystemReset();   // 执行系统复位	
#if Gimbal_Auto_Debug		
		gimbal_mode_set->gimbal_behaviour=GIMBAL_AUTO;
#else
		gimbal_mode_set->gimbal_behaviour=GIMBAL_OPERATION;
#endif
	}
	else if (switch_is_mid(rc_ctrl.rc.s[1])||PT_switch_is_mid(remote_data.mode_sw)) //中
	{
			gimbal_mode_set->gimbal_behaviour=GIMBAL_ZERO_FORCE;
			
			//头无力自救，自救后手动切换
			if(remote_data.fn_2==1||ZeroToFollowFlag||gimbal_mode_set->TwoBoardControlGimbal.ChassisState==2)//处于正常站立状态
			{
			ZeroToFollowFlag=1;
			gimbal_mode_set->gimbal_behaviour=GIMBAL_ABSOLUTE_ANGLE;
			}
	}
	else if (switch_is_down(rc_ctrl.rc.s[1])||PT_switch_is_down(remote_data.mode_sw)) //下
	{
		ZeroToFollowFlag=0;
		gimbal_mode_set->gimbal_behaviour=GIMBAL_ZERO_FORCE;
	}
	
	//遥控器掉线保护
	if (toe_is_error(DBUS_TOE))
	{
		gimbal_mode_set->gimbal_behaviour=GIMBAL_ZERO_FORCE;
	}
	
	gimbal_mode_set->last_gimbal_behaviour=gimbal_mode_set->gimbal_behaviour;
}

extern void gimbal_behaviour_control_set(float *add_yaw, float *add_pitch, gimbal_move_t *gimbal_control_set)
{
	if (add_yaw == NULL || add_pitch == NULL || gimbal_control_set == NULL)
	{
		return;
	}
	//通过不同的模式，选择不同的控制量
	switch(gimbal_control_set->gimbal_behaviour)
	{
		case GIMBAL_ZERO_FORCE:
		{
			gimbal_zero_force_control(add_yaw,add_pitch,gimbal_control_set);
			break;
		}
		case GIMBAL_ABSOLUTE_ANGLE:
		{
			gimbal_absolute_angle_control(add_yaw,add_pitch,gimbal_control_set);
			break;
		}
		case GIMBAL_AUTO:
		{
			gimbal_auto_angle_control(add_yaw,add_pitch,gimbal_control_set);
			break;
		}
		case GIMBAL_RETURN:
		{
			gimbal_return(add_yaw,add_pitch,gimbal_control_set);
			break;
		}
		case GIMBAL_OPERATION:
		{
			gimbal_absolute_angle_control(add_yaw,add_pitch,gimbal_control_set);
			if(Z_flag)
			{
			gimbal_zero_force_control(add_yaw,add_pitch,gimbal_control_set);
			}
			if(remote_data.mouse_right==1)
			{
			gimbal_auto_angle_control(add_yaw,add_pitch,gimbal_control_set);
			}
			
			break;
		}
 default:
 {
	 break;
 }
	}
}

static void gimbal_return(float *yaw, float *pitch, gimbal_move_t *gimbal_control_set)
{
	if (yaw == NULL || pitch == NULL || gimbal_control_set == NULL)
	{
		return;
	}
		*yaw = 0.0f;
		*pitch = 0.0f;
}

static void gimbal_zero_force_control(float *yaw, float *pitch, gimbal_move_t *gimbal_control_set)
{
	if (yaw == NULL || pitch == NULL || gimbal_control_set == NULL)
	{
		return;
	}
	
		*yaw = 0.0f;
		*pitch = 0.0f;
}

static void gimbal_absolute_angle_control(float *yaw, float *pitch, gimbal_move_t *gimbal_control_set)
{
	if (yaw == NULL || pitch == NULL || gimbal_control_set == NULL)
	{
		return;
	}
	
	static int16_t yaw_channel = 0, pitch_channel = 0;
	//遥控器输入死区，因为遥控器存在差异，摇杆在中间，其值不一定为零
	
#if PT_link_en
	rc_deadband_limit(remote_data.ch_3, yaw_channel, RC_DEADBAND);
	rc_deadband_limit(remote_data.ch_2, pitch_channel, RC_DEADBAND);
#else
	rc_deadband_limit(rc_ctrl.rc.ch[RC_RIGHT_X], yaw_channel, RC_DEADBAND);
	rc_deadband_limit(rc_ctrl.rc.ch[RC_RIGHT_Y], pitch_channel, RC_DEADBAND);
#endif
	//对遥控器通道值乘以一个系数，使控制时符合手感,键鼠同理
	
	if(gimbal_control_set->gimbal_behaviour==GIMBAL_OPERATION)
	{
	*yaw = remote_data.mouse_x * 0.00004f;
	*pitch = remote_data.mouse_y * 0.0001f;
	}
	else
	{
	*yaw = yaw_channel * 0.0000055f     ;//+  remote_data.mouse_x * 0.00004f;
	*pitch = pitch_channel * 0.0000053f ;//+  remote_data.mouse_y * 0.0001f;
	}
/*************************键鼠AD一键转头*******************************************************/	
	
	if(!(gimbal_control_set->TwoBoardControlGimbal.mode==8))
	{
		//用上flag表示长按也只触发一次，防止反复触发
		A_flag_last=A_flag;
		A_flag=(remote_data.key & KEY_PRESSED_OFFSET_A)?1:0;
		if(A_flag-A_flag_last==1&&!(remote_data.key&KEY_PRESSED_OFFSET_SHIFT))
		{
		*yaw=-0;
		}
		
		D_flag_last=D_flag;
		D_flag=(remote_data.key & KEY_PRESSED_OFFSET_D)?1:0;
		if(D_flag-D_flag_last==1&&!(remote_data.key&KEY_PRESSED_OFFSET_SHIFT))
		{
		*yaw=0;
		}
	}
}

static void gimbal_auto_angle_control(float *yaw, float *pitch, gimbal_move_t *gimbal_control_set)
{
	if (yaw == NULL || pitch == NULL || gimbal_control_set == NULL)
	{
		return;
	}
		//add值为0，实际是set=target，不需要add
	static int16_t yaw_channel = 0, pitch_channel = 0;
	//遥控器输入死区，因为遥控器存在差异，摇杆在中间，其值不一定为零
#if PT_link_en
	rc_deadband_limit(remote_data.ch_3, yaw_channel, RC_DEADBAND);
	rc_deadband_limit(remote_data.ch_2, pitch_channel, RC_DEADBAND);
#else
	rc_deadband_limit(rc_ctrl.rc.ch[RC_RIGHT_X], yaw_channel, RC_DEADBAND);
	rc_deadband_limit(rc_ctrl.rc.ch[RC_RIGHT_Y], pitch_channel, RC_DEADBAND);
#endif
	//对遥控器通道值乘以一个系数，使控制时符合手感
	*yaw = yaw_channel * 0.0000055f;
	*pitch = pitch_channel * 0.0000053f;
}
