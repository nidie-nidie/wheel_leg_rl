#include "gimbal_task.h"
#include "gimbal_behaviour.h"
#include "cmsis_os.h"
#include "main.h"
#include "INS_task.h"
#include "arm_math.h"
#include "detect_task.h"
#include "config.h"
#include "user_lib.h"
#include "USART_receive.h"
#include "referee.h"
#include "custom_ui_draw.h"

extern tank_ui_t tank;

gimbal_move_t gimbal_move;
static uint8_t send_time=0;
static uint8_t SendChassisSwitch=0;
static void gimbal_data_send(void);
static void gimbal_init(void);
static void gimbal_feedback_update(void);
static void gimbal_set_mode(void);
static void gimbal_absolute_angle_limit(gimbal_control_t *gimbal_motor, float add);
static void gimbal_set_control(void);
static void gimbal_motor_absolute_angle_control(gimbal_control_t *gimbal_motor);
static void gimbal_motor_auto_control(gimbal_control_t *gimbal_motor);
static void gimbal_control_loop(void);
static void gimbal_to_chassis(void);
static void gimbal_motor_relative_angle_control(gimbal_control_t *gimbal_motor);
static void gimbal_motor_wake(void);
static void updateUI(void);
static uint8_t SendFrequencyFlag=0;
uint8_t relative_angle_yaw_flag=0;
uint8_t GimbalOffsetAngleFlag=0;

int V_flag = 0, V_flag_last = 0;
int Z_flag = 0, Z_flag_last = 0,Z_flag_time=0;
int shift_flag = 0, shift_flag_last = 0,shift_MODE=0;
int R_flag = 0, R_flag_last = 0;
int B_flag = 0, B_flag_last = 0;
int C_flag = 0, C_flag_last = 0;
int F_flag = 0, F_flag_last = 0;
float MaxVxSet=0;
float KeyVx=0,KeyVy=0,KeyRoll=0;

void gimbal_task(void const *pvParameters)
{
	//推迟开始任务，防止初始化时电机等其他部件还没有正常运行
  vTaskDelay(201);
  //云台初始化
  gimbal_init();
			//判断电机是否都上线和使能
			while(toe_is_error(YAW_GIMBAL_MOTOR_TOE)||toe_is_error(PITCH_GIMBAL_MOTOR_TOE)
						||gimbal_move.gimbal_yaw_motor.gimbal_motor.gimbal_motor_measure->Status!=1
						||gimbal_move.gimbal_pitch_motor.gimbal_motor.gimbal_motor_measure->Status!=1)
			{
				gimbal_motor_wake();
				gimbal_feedback_update();
				gimbal_data_send();
				vTaskDelay(1);
			}
  while (1)
  {
		//设置云台控制模式
		gimbal_set_mode();
		//云台数据反馈
		gimbal_feedback_update();
		//设置云台控制量
		gimbal_set_control();
		//计算云台控制量
		gimbal_control_loop();
		//给自瞄发送数据
		gimbal_data_send();
		//给底盘发送消息控制整车
		gimbal_to_chassis();
		//UI数据更新
		updateUI();

		//控制电机
		SendFrequencyFlag++;
		if(SendFrequencyFlag>=2)//为了保证CAN线负载率，这里手动给电机降频
		{
			SendFrequencyFlag=0;
					if (toe_is_error(DBUS_TOE)||toe_is_error(YAW_GIMBAL_MOTOR_TOE)||toe_is_error(PITCH_GIMBAL_MOTOR_TOE))
					{
						if(gimbal_move.gimbal_yaw_motor.gimbal_motor.gimbal_motor_measure->Status!=1||gimbal_move.gimbal_pitch_motor.gimbal_motor.gimbal_motor_measure->Status!=1)
						{
							gimbal_motor_wake();
						}
						else
						{
							CAN_cmd_gimbal_pitch(0);
							CAN_cmd_gimbal_yaw(0);
						}
					}
					else
					{
						if(gimbal_move.gimbal_yaw_motor.gimbal_motor.gimbal_motor_measure->Status!=1||gimbal_move.gimbal_pitch_motor.gimbal_motor.gimbal_motor_measure->Status!=1)
						{
							gimbal_motor_wake();
						}
						else
						{
							CAN_cmd_gimbal_pitch(-gimbal_move.gimbal_pitch_motor.gimbal_motor.set_Tp);
							CAN_cmd_gimbal_yaw(-gimbal_move.gimbal_yaw_motor.gimbal_motor.set_Tp);
						}
					}
		}
			
    vTaskDelay(1);
  }
}


static void gimbal_init(void)
{
	//通过指针的方式获取电机数据和陀螺仪，使task和application解耦
	gimbal_move.gimbal_yaw_motor.gimbal_motor.gimbal_motor_measure=yaw_motor_measure_point();
	gimbal_move.gimbal_pitch_motor.gimbal_motor.gimbal_motor_measure=pitch_motor_measure_point();
	gimbal_move.ins_data.INS_Angle=get_INS_angle_point();
	gimbal_move.ins_data.INS_Gyro=get_gyro_data_point();
	gimbal_move.ins_data.INS_Accel=get_accel_data_point();
	//absolute_angle用控制精度高噪声小，relative_angle做限位更有效
	//初始化云台中值和限位
	gimbal_move.gimbal_yaw_motor.offset_angle=-0.02f;
	gimbal_move.gimbal_yaw_motor.min_relative_angle=-3.14f*2;//不做限位则给到大于3.14，刚好3.14可能会刚好卡着不能无限转
	gimbal_move.gimbal_yaw_motor.max_relative_angle=3.14f*2;
	gimbal_move.gimbal_pitch_motor.offset_angle=2.678f;
	gimbal_move.gimbal_pitch_motor.min_relative_angle=-0.3f;
	gimbal_move.gimbal_pitch_motor.max_relative_angle=0.4f;
	//初始化云台电机PID，包括两个电机，两种要求的双环PID，所以2*2*2+2，但是力大撞飞，没有给自瞄单独调PID
	gimbal_PID_init(&gimbal_move.gimbal_yaw_motor.gimbal_motor_absolute_angle_pid,20,0,0,0,0,0.001,10,0);//20
	gimbal_PID_init(&gimbal_move.gimbal_yaw_motor.gimbal_motor_auto_angle_pid,0,0,0,0,0,0.001,0,0);
	PID_init(&gimbal_move.gimbal_yaw_motor.gimbal_motor_absolute_speed_pid,1,0,0,0,0.001,5,0);
	PID_init(&gimbal_move.gimbal_yaw_motor.gimbal_motor_auto_speed_pid,0,0,0,0,0.001,0,0);
	gimbal_PID_init(&gimbal_move.gimbal_pitch_motor.gimbal_motor_absolute_angle_pid,10,20,0.8,0,0.1,0.001,5,2);//事实上不加可能更好I
	gimbal_PID_init(&gimbal_move.gimbal_pitch_motor.gimbal_motor_auto_angle_pid,0,0,0,0,0,0.001,0,0);
	PID_init(&gimbal_move.gimbal_pitch_motor.gimbal_motor_absolute_speed_pid,1,0,0,0,0.001,5,0);
	PID_init(&gimbal_move.gimbal_pitch_motor.gimbal_motor_auto_speed_pid,0,0,0,0,0.001,0,0);
	//yaw的相对角PID，用于底盘自救规则，但是现在是头无力自救，所以没用到
	gimbal_PID_init(&gimbal_move.gimbal_yaw_motor.gimbal_motor_relative_angle_pid,10,0,0,0,0,0.001,4,0);
	PID_init(&gimbal_move.gimbal_yaw_motor.gimbal_motor_relative_speed_pid,1,5,0,3,0.001,5,5);
	//数据更新，用于下面的初始化启动状态
	gimbal_feedback_update();
	//初始化启动状态
	gimbal_move.gimbal_yaw_motor.absolute_angle_set=gimbal_move.gimbal_yaw_motor.absolute_angle;
	gimbal_move.gimbal_yaw_motor.absolute_speed_set=gimbal_move.gimbal_yaw_motor.absolute_speed;
	gimbal_move.gimbal_pitch_motor.absolute_angle_set=gimbal_move.gimbal_pitch_motor.absolute_angle;
	gimbal_move.gimbal_pitch_motor.absolute_speed_set=gimbal_move.gimbal_pitch_motor.absolute_speed;
	//初始化云台状态
	gimbal_move.gimbal_behaviour=GIMBAL_ZERO_FORCE;
	gimbal_move.last_gimbal_behaviour=GIMBAL_ZERO_FORCE;
}

static void gimbal_feedback_update(void)
{
	//INS_data数据更新，没有用extern的写法，浪费一丁丁性能使task间解耦，使代码更安全,最好的办法其实是用freertos的东西如队列
	//写法问题，对于本代码，调云台时哪个不对，在这边手动改正负号，方向是前方为正x，左方为正y，上方为正z，右手法则，大拇指指向xyz，四指方向是数值减小的方向
	gimbal_move.ins_data.yaw=-gimbal_move.ins_data.INS_Angle[0];
	gimbal_move.ins_data.pitch=gimbal_move.ins_data.INS_Angle[2];
	gimbal_move.ins_data.roll=-gimbal_move.ins_data.INS_Angle[1];
	gimbal_move.ins_data.wx=-gimbal_move.ins_data.INS_Gyro[1];
	gimbal_move.ins_data.wy=gimbal_move.ins_data.INS_Gyro[0];
	gimbal_move.ins_data.wz=-gimbal_move.ins_data.INS_Gyro[2];
	gimbal_move.ins_data.ax=gimbal_move.ins_data.INS_Accel[0];
	gimbal_move.ins_data.ay=gimbal_move.ins_data.INS_Accel[1];
	gimbal_move.ins_data.az=gimbal_move.ins_data.INS_Accel[2];

	//是新机器人时，修改正负号使absolute_angle和relative_angle及speed方向一直，有助于之后的控制
	/*我的定义是：前方为正x，左方为正y，上方为正z，右手法则，大拇指指向xyz，四指方向是数值减小的方向*/
	gimbal_move.gimbal_pitch_motor.relative_angle=-rad_format(gimbal_move.gimbal_pitch_motor.gimbal_motor.gimbal_motor_measure->Position-gimbal_move.gimbal_pitch_motor.offset_angle);
	gimbal_move.gimbal_pitch_motor.relative_speed=-gimbal_move.gimbal_pitch_motor.gimbal_motor.gimbal_motor_measure->Speed;
	gimbal_move.gimbal_pitch_motor.absolute_angle=gimbal_move.ins_data.pitch;
	gimbal_move.gimbal_pitch_motor.absolute_speed=gimbal_move.ins_data.wy;
		//当pitch俯仰时，对偏航（Yaw）轴角速度测量有影响，通过解算消除pitch的影响
		//真实Yaw角速度 = (机身在倾斜状态下的角速度) 投影到 (水平Yaw轴上)，可以从极端情况，即pitch为0和90时看代码
	gimbal_move.gimbal_yaw_motor.relative_angle=-rad_format(gimbal_move.gimbal_yaw_motor.gimbal_motor.gimbal_motor_measure->Position-gimbal_move.gimbal_yaw_motor.offset_angle);
	gimbal_move.gimbal_yaw_motor.relative_speed=-gimbal_move.gimbal_yaw_motor.gimbal_motor.gimbal_motor_measure->Speed;
	gimbal_move.gimbal_yaw_motor.absolute_angle=gimbal_move.ins_data.yaw;
	gimbal_move.gimbal_yaw_motor.absolute_speed=arm_cos_f32(gimbal_move.gimbal_pitch_motor.relative_angle)*gimbal_move.ins_data.wz+arm_sin_f32(gimbal_move.gimbal_pitch_motor.relative_angle)*gimbal_move.ins_data.wx;
	
}

static void gimbal_set_mode(void)
{
	gimbal_behaviour_mode_set(&gimbal_move);
}

static void gimbal_set_control(void)
{
	float add_yaw_angle = 0.0f;
	float add_pitch_angle = 0.0f;
	gimbal_behaviour_control_set(&add_yaw_angle,&add_pitch_angle,&gimbal_move);
			auto_shoot.NUC_GG_Detect++;
			if(auto_shoot.NUC_GG_Detect>=600)
			{
				auto_shoot.NUC_GG_Detect=600;
			}
	//自瞄控制是上位机直接发目标角度，而遥控控制一般是增量角度
	if(gimbal_move.gimbal_behaviour==GIMBAL_AUTO)
	{
		//这里对于某些限位要求大的云台，可以做个差继续算最大添加角，这里暂时不多做修改
		//自瞄时被控制，不需要时保持不动
		if(auto_shoot.mode!=0&&(auto_shoot.NUC_GG_Detect)<=550)
		{
		gimbal_move.gimbal_yaw_motor.absolute_angle_set=auto_shoot.yaw;
		gimbal_move.gimbal_pitch_motor.absolute_angle_set=auto_shoot.pitch;
		}
		else
		{
		gimbal_move.gimbal_yaw_motor.absolute_angle_set=gimbal_move.gimbal_yaw_motor.absolute_angle+add_yaw_angle;
		gimbal_move.gimbal_pitch_motor.absolute_angle_set=gimbal_move.gimbal_pitch_motor.absolute_angle+add_pitch_angle;
		}
	}
	else if(gimbal_move.gimbal_behaviour==GIMBAL_RETURN)
	{
		gimbal_move.gimbal_yaw_motor.absolute_angle_set=gimbal_move.gimbal_yaw_motor.absolute_angle;
		gimbal_move.gimbal_pitch_motor.absolute_angle_set=add_pitch_angle;
	}
	else
	{
		if(remote_data.mouse_right==1)
		{
			if(auto_shoot.mode!=0&&(auto_shoot.NUC_GG_Detect)<=550)
			{
			gimbal_move.gimbal_yaw_motor.absolute_angle_set=auto_shoot.yaw;
			gimbal_move.gimbal_pitch_motor.absolute_angle_set=auto_shoot.pitch;
			}
			else
			{
			gimbal_move.gimbal_yaw_motor.absolute_angle_set=gimbal_move.gimbal_yaw_motor.absolute_angle+add_yaw_angle;
			gimbal_move.gimbal_pitch_motor.absolute_angle_set=gimbal_move.gimbal_pitch_motor.absolute_angle+add_pitch_angle;
			}
		}
	//在设置控制量之前就进行电子限位，即设置不到限位外
	//对于有滑环的云台，yaw一般不做限位
	//普通限位可以直接是set>xx,set=xx,但是陀螺仪角度会漂，所以要结合电机角
	//即，我们set是一个增量（error），将error放到电机角里，然后set>xx,set=xx，error=set-get，得出最大可增量
	gimbal_absolute_angle_limit(&gimbal_move.gimbal_pitch_motor,add_pitch_angle);
	gimbal_absolute_angle_limit(&gimbal_move.gimbal_yaw_motor,add_yaw_angle);
	}
	
	//无力后将set与ref一致，防止下次启动直接乱动
	if(gimbal_move.gimbal_behaviour==GIMBAL_ZERO_FORCE||gimbal_move.gimbal_behaviour==GIMBAL_RETURN)
	{
		gimbal_move.gimbal_yaw_motor.absolute_angle_set=gimbal_move.gimbal_yaw_motor.absolute_angle;
		gimbal_move.gimbal_pitch_motor.absolute_angle_set=gimbal_move.gimbal_pitch_motor.absolute_angle;
	}
}


static void gimbal_absolute_angle_limit(gimbal_control_t *gimbal_motor, float add)
{
  static float bias_angle;
  static float angle_set;
  if (gimbal_motor == NULL)
  {
    return;
  }
  //当前控制误差角度
  bias_angle = rad_format(gimbal_motor->absolute_angle_set - gimbal_motor->absolute_angle);
  //云台相对角度+ 误差角度 + 新增角度 如果大于最大机械角度
	if(gimbal_motor->relative_angle+bias_angle+add>gimbal_motor->max_relative_angle)
	{
    //如果是往最大机械角度控制方向
    if (add > 0.0f)
    {
      //计算出一个最大的添加角度，
      add = gimbal_motor->max_relative_angle - gimbal_motor->relative_angle - bias_angle;
    }
	}
	else if(gimbal_motor->relative_angle+bias_angle+add<gimbal_motor->min_relative_angle)
	{
    if (add < 0.0f)
    {
      add = gimbal_motor->min_relative_angle - gimbal_motor->relative_angle - bias_angle;
    }
	}
	
	angle_set=gimbal_motor->absolute_angle_set;
	gimbal_motor->absolute_angle_set=rad_format(angle_set + add);
}

static void gimbal_control_loop(void)
{
	//通过不同的模式选择不同的控制方式，就算是PID，也因自瞄和操作手要求不一样，参数也不一样,但是4310力大砖飞，自瞄和操作手都说好
	if(gimbal_move.gimbal_behaviour==GIMBAL_ABSOLUTE_ANGLE)
	{
		gimbal_motor_absolute_angle_control(&gimbal_move.gimbal_pitch_motor);
		gimbal_motor_absolute_angle_control(&gimbal_move.gimbal_yaw_motor);
	}
	else if(gimbal_move.gimbal_behaviour==GIMBAL_AUTO)
	{
		gimbal_motor_auto_control(&gimbal_move.gimbal_pitch_motor);
		gimbal_motor_auto_control(&gimbal_move.gimbal_yaw_motor);
	}
	else if(gimbal_move.gimbal_behaviour==GIMBAL_RETURN)//自救头摆正
	{
		gimbal_motor_absolute_angle_control(&gimbal_move.gimbal_pitch_motor);//pitch不动
		gimbal_move.gimbal_pitch_motor.gimbal_motor.set_Tp=0;
		gimbal_motor_relative_angle_control(&gimbal_move.gimbal_yaw_motor);
	}
	else if(gimbal_move.gimbal_behaviour==GIMBAL_OPERATION)
	{
			if(remote_data.mouse_right==1)
			{
		gimbal_motor_auto_control(&gimbal_move.gimbal_pitch_motor);
		gimbal_motor_auto_control(&gimbal_move.gimbal_yaw_motor);
			}
			else
			{
		gimbal_motor_absolute_angle_control(&gimbal_move.gimbal_pitch_motor);
		gimbal_motor_absolute_angle_control(&gimbal_move.gimbal_yaw_motor);
			}
	}
	else
	{
		gimbal_motor_absolute_angle_control(&gimbal_move.gimbal_pitch_motor);
		gimbal_motor_absolute_angle_control(&gimbal_move.gimbal_yaw_motor);
		
		gimbal_move.gimbal_pitch_motor.gimbal_motor.set_Tp=0;
		gimbal_move.gimbal_yaw_motor.gimbal_motor.set_Tp=0;
	}
}
static void gimbal_motor_absolute_angle_control(gimbal_control_t *gimbal_motor)
{
	if (gimbal_motor == NULL)
	{
		return;
	}
	 //角度环，速度环串级pid
	gimbal_motor->absolute_speed_set=gimbal_PID_calc(&gimbal_motor->gimbal_motor_absolute_angle_pid,gimbal_motor->absolute_angle,gimbal_motor->absolute_angle_set);
	gimbal_motor->gimbal_motor.set_Tp=PID_calc(&gimbal_motor->gimbal_motor_absolute_speed_pid,gimbal_motor->absolute_speed,gimbal_motor->absolute_speed_set);
}

static void gimbal_motor_relative_angle_control(gimbal_control_t *gimbal_motor)
{
	if (gimbal_motor == NULL)
	{
		return;
	}
	
	if((gimbal_motor->relative_angle)<1.57f&&(gimbal_motor->relative_angle)>-1.57f)
	{
		gimbal_motor->relative_angle_set=0;
	}
	else
	{
		gimbal_motor->relative_angle_set=3.14;
	}
	
	gimbal_motor->relative_speed_set=gimbal_PID_calc(&gimbal_motor->gimbal_motor_relative_angle_pid,gimbal_motor->relative_angle,gimbal_motor->relative_angle_set);
	gimbal_motor->gimbal_motor.set_Tp=PID_calc(&gimbal_motor->gimbal_motor_relative_speed_pid,gimbal_motor->relative_speed,gimbal_motor->relative_speed_set);
}

static void gimbal_motor_auto_control(gimbal_control_t *gimbal_motor)
{
	if (gimbal_motor == NULL)
	{
		return;
	}
	 //角度环，速度环串级pid
	gimbal_motor->absolute_speed_set=gimbal_PID_calc(&gimbal_motor->gimbal_motor_absolute_angle_pid,gimbal_motor->absolute_angle,gimbal_motor->absolute_angle_set);
	gimbal_motor->gimbal_motor.set_Tp=PID_calc(&gimbal_motor->gimbal_motor_absolute_speed_pid,gimbal_motor->absolute_speed,gimbal_motor->absolute_speed_set);
}

static void gimbal_to_chassis(void)
{
	float x_set,y_set;
	int16_t vx_channel,vy_channel,roll_channel,L_channel;
		
#if PT_link_en
	rc_deadband_limit(remote_data.ch_0, vy_channel, RC_DEADBAND);
	rc_deadband_limit(remote_data.ch_1, vx_channel, RC_DEADBAND);
	rc_deadband_limit(remote_data.ch_0, roll_channel, 400);
	rc_deadband_limit(remote_data.wheel, L_channel, RC_DEADBAND);//给channel死区限制，有些遥控器的channel默认值可能不为0
#else
	rc_deadband_limit(rc_ctrl.rc.ch[RC_LEFT_X], vy_channel, RC_DEADBAND);
	rc_deadband_limit(rc_ctrl.rc.ch[RC_LEFT_Y], vx_channel, RC_DEADBAND);
	rc_deadband_limit(rc_ctrl.rc.ch[RC_LEFT_X], roll_channel, RC_DEADBAND);
	rc_deadband_limit(rc_ctrl.rc.ch[4], L_channel, RC_DEADBAND);//给channel死区限制，有些遥控器的channel默认值可能不为0
#endif
	
	//小陀螺时左右速度设置
	y_set=limit_symmetric((float)vy_channel/660.0f*5.0f*0.5f,2.5);
	gimbal_move.TwoBoardControlGimbal.vy_set=y_set;
	//前进速度设置
	x_set=limit_symmetric((float)vx_channel/660.0f*5.0f*0.5f,2.5);
	gimbal_move.TwoBoardControlGimbal.vx_set=x_set;
	//Roll轴补偿
	gimbal_move.TwoBoardControlGimbal.roll_set=(float)roll_channel/660*0.3f;
	//腿长设置
	if(L_channel>400)
	{
	gimbal_move.TwoBoardControlGimbal.legL_mode=2;
	}
	else if (L_channel<-400)
	{
	gimbal_move.TwoBoardControlGimbal.legL_mode=3;
	}
	else
	{
	gimbal_move.TwoBoardControlGimbal.legL_mode=1;
	}

	//yaw相关数据
	gimbal_move.TwoBoardControlGimbal.relative_angle_yaw=gimbal_move.gimbal_yaw_motor.relative_angle;
	//底盘模式
#if PT_link_en
	gimbal_move.TwoBoardControlGimbal.mode=remote_data.mode_sw;//0这里挡位不一样，换遥控器时注意一下
	if(remote_data.pause==1&&remote_data.mode_sw==1)
	{
	gimbal_move.TwoBoardControlGimbal.mode=8;//多个模式给小陀螺
	}
#else
	gimbal_move.TwoBoardControlGimbal.mode=2;//rc_ctrl.rc.s[1];//2
#endif
	
	gimbal_move.TwoBoardControlGimbal.chassis_power_limit=robot_state.chassis_power_limit;
	gimbal_move.TwoBoardControlGimbal.buffer_energy_chassis=power_heat_data_t.chassis_power_buffer;
	
	if(remote_data.mouse_middle==1)
	{
	MaxVxSet=2.5f;
	}
	else
	{
	MaxVxSet=1.8f;
	}
	
	/************************键鼠控制*****************************/
	if(gimbal_move.gimbal_behaviour==GIMBAL_OPERATION)
	{
			//默认在操作手模式下，底盘起立的
			gimbal_move.TwoBoardControlGimbal.mode=1;
			//ws前后控制
			KeyVx=limit_symmetric(KeyVx+(remote_data.key&KEY_PRESSED_OFFSET_W)*0.015f-(remote_data.key&KEY_PRESSED_OFFSET_S)*0.015f,MaxVxSet);
			if(!(remote_data.key&KEY_PRESSED_OFFSET_W)&&!((remote_data.key&KEY_PRESSED_OFFSET_S)))
			{
			KeyVx=0;
			}
			x_set=KeyVx;
			gimbal_move.TwoBoardControlGimbal.vx_set=x_set;

			//qe侧身
			if((remote_data.key & KEY_PRESSED_OFFSET_Q)?1:0)
			{
			gimbal_move.TwoBoardControlGimbal.roll_set=-0.3f;
			}
			else if((remote_data.key & KEY_PRESSED_OFFSET_E)?1:0)
			{
			gimbal_move.TwoBoardControlGimbal.roll_set=0.3f;
			}
			else
			{
			gimbal_move.TwoBoardControlGimbal.roll_set=0.0f;
			}
			//shift底盘小陀螺
			shift_flag=(remote_data.key & KEY_PRESSED_OFFSET_SHIFT)?1:0;
//			if(shift_flag==1&&shift_flag_last==0)
//			{
//				shift_MODE=!shift_MODE;
//			}
			if(shift_flag==1)
			{
				gimbal_move.TwoBoardControlGimbal.mode=8;
			}
				shift_flag_last=shift_flag;
			//ad小陀螺左右控制
			if(gimbal_move.TwoBoardControlGimbal.mode==8)
			{
				KeyVy=limit_symmetric(KeyVy-(remote_data.key&KEY_PRESSED_OFFSET_A)*0.004f+(remote_data.key&KEY_PRESSED_OFFSET_D)*0.004f,MaxVxSet);
				if(!(remote_data.key&KEY_PRESSED_OFFSET_A)&&!((remote_data.key&KEY_PRESSED_OFFSET_D)))
				{
				KeyVy=0;
				}
				y_set=KeyVy;
				gimbal_move.TwoBoardControlGimbal.vy_set=y_set;
			}
			//ctrl和F伸腿收腿
			gimbal_move.TwoBoardControlGimbal.legL_mode=1;
			if(remote_data.key & KEY_PRESSED_OFFSET_CTRL)
			{
				gimbal_move.TwoBoardControlGimbal.legL_mode=2;
			}
			if(remote_data.key & KEY_PRESSED_OFFSET_F)
			{
				gimbal_move.TwoBoardControlGimbal.legL_mode=3;
			}
				
			//预留的R键跳跃	
			R_flag_last=R_flag;
			R_flag=(remote_data.key & KEY_PRESSED_OFFSET_R)?1:0;
			if(R_flag==1)
			{
			gimbal_move.TwoBoardControlGimbal.mode=9;
			}
			
			//C键蹭台阶键
			C_flag_last=C_flag;
			C_flag=(remote_data.key & KEY_PRESSED_OFFSET_C)?1:0;
			if(C_flag==1)
			{
			gimbal_move.TwoBoardControlGimbal.mode=7;
			gimbal_move.TwoBoardControlGimbal.legL_mode=3;
			}
			
			//z键自启
			Z_flag_last=Z_flag;
			Z_flag=(remote_data.key & KEY_PRESSED_OFFSET_Z)?1:0;
			if(Z_flag==1)
			{
				Z_flag_time++;
				if(Z_flag_time<10)
				{
				gimbal_move.TwoBoardControlGimbal.mode=0;
				}
				else
				{
				gimbal_move.TwoBoardControlGimbal.mode=1;
				}
				//自救时头无力
				gimbal_move.gimbal_pitch_motor.gimbal_motor.set_Tp=0;
				gimbal_move.gimbal_yaw_motor.gimbal_motor.set_Tp=0;
			}
			else
			{
			Z_flag_time=0;
			}

			//V键一键转头
			V_flag_last=V_flag;
			V_flag=(remote_data.key & KEY_PRESSED_OFFSET_V)?1:0;
			if(V_flag-V_flag_last==1)
			{
				GimbalOffsetAngleFlag=!GimbalOffsetAngleFlag;
			}
			
			if(!GimbalOffsetAngleFlag)
			{
				gimbal_move.TwoBoardControlGimbal.relative_angle_yaw=gimbal_move.gimbal_yaw_motor.relative_angle;
			}
			else
			{
				gimbal_move.TwoBoardControlGimbal.relative_angle_yaw=rad_format(gimbal_move.gimbal_yaw_motor.relative_angle+3.14f);
				gimbal_move.TwoBoardControlGimbal.vx_set=-gimbal_move.TwoBoardControlGimbal.vx_set;
				gimbal_move.TwoBoardControlGimbal.roll_set=-gimbal_move.TwoBoardControlGimbal.roll_set;
						if(gimbal_move.TwoBoardControlGimbal.last_mode==8)
						{
						gimbal_move.TwoBoardControlGimbal.vx_set=-gimbal_move.TwoBoardControlGimbal.vx_set;
						gimbal_move.TwoBoardControlGimbal.vy_set=-gimbal_move.TwoBoardControlGimbal.vy_set;
						}
			}
			
			if(Z_flag==1&&remote_data.key & KEY_PRESSED_OFFSET_CTRL)	
			{
			gimbal_move.TwoBoardControlGimbal.mode=6;
				//F键控变腿
				F_flag_last=F_flag;
				F_flag=(remote_data.key & KEY_PRESSED_OFFSET_F)?1:0;
				if(F_flag==1&&F_flag_last==0)
				{
					gimbal_move.TwoBoardControlGimbal.LeftLengthSet=gimbal_move.TwoBoardControlGimbal.LeftLengthSet+0.1f;
					if(gimbal_move.TwoBoardControlGimbal.LeftLengthSet>0.41f)
					{
					gimbal_move.TwoBoardControlGimbal.LeftLengthSet=0.1f;
					}
					gimbal_move.TwoBoardControlGimbal.RightLengthSet=gimbal_move.TwoBoardControlGimbal.LeftLengthSet;
				}
				if(remote_data.mouse_left==1)
				{
				gimbal_move.TwoBoardControlGimbal.LeftPhi0Set=gimbal_move.TwoBoardControlGimbal.LeftPhi0+0.2f;
				gimbal_move.TwoBoardControlGimbal.RightPhi0Set=gimbal_move.TwoBoardControlGimbal.RightPhi0+0.2f;
				}
				if(remote_data.mouse_right==1)
				{
				gimbal_move.TwoBoardControlGimbal.LeftPhi0Set=gimbal_move.TwoBoardControlGimbal.LeftPhi0-0.2f;
				gimbal_move.TwoBoardControlGimbal.RightPhi0Set=gimbal_move.TwoBoardControlGimbal.RightPhi0-0.2f;
				}
			}
			
				if(remote_data.mouse_middle==1)
				{
				gimbal_move.TwoBoardControlGimbal.mode=5;
				}
			
			//B键实现软重启，有需要可以之后先发消息再重启，使得底盘也重启
			B_flag_last=B_flag;
			B_flag=(remote_data.key & KEY_PRESSED_OFFSET_B)?1:0;
			if(B_flag-B_flag_last==1)
			{
					__disable_irq();          // 关闭所有可屏蔽中断
					HAL_NVIC_SystemReset();   // 执行系统复位
			}
		}

#if Gimbal_Auto_Debug		
		gimbal_move.TwoBoardControlGimbal.mode=0;//一键关底盘
#else
		//gimbal_move.TwoBoardControlGimbal.mode=0;//一键关底盘
#endif
	if(!toe_is_error(DBUS_TOE))//遥控器掉线时不发消息，用于底盘监测遥控器是否在线
	{
	//发送数据
		if(SendChassisSwitch==0)
		{
			SendChassisSwitch=1;
			CAN_cmd_chassis();
		}
		else
		{
			SendChassisSwitch=0;
			CAN_cmd_chassis2();
		}
	}
}
	

static void gimbal_data_send(void)
{
  send_time++;
  //5*1ms,每5ms给自瞄发送一次信息
  if (send_time % 5==0)
  {
    user_data_pack_handle();
  }
	if(send_time==20)
	{
    send_time = 0;
		CAN_cmd_SuperPower();
	}
}

static void gimbal_motor_wake(void)
{
	if(gimbal_move.gimbal_yaw_motor.gimbal_motor.gimbal_motor_measure->Status==0x0D)
	{
	CAN_cmd_yaw_clean();
	}
	else if(gimbal_move.gimbal_yaw_motor.gimbal_motor.gimbal_motor_measure->Status==0x00)
	{
	CAN_cmd_yaw_init();
	}
	else
	{
	CAN_cmd_gimbal_yaw(0);
	}
		
	if(gimbal_move.gimbal_pitch_motor.gimbal_motor.gimbal_motor_measure->Status==0x0D)
	{
	CAN_cmd_pitch_clean();
	}
	else if(gimbal_move.gimbal_pitch_motor.gimbal_motor.gimbal_motor_measure->Status==0x00)
	{
	CAN_cmd_pitch_init();
	}
	else
	{
	CAN_cmd_gimbal_pitch(0);
	}
}

static void updateUI(void)
{
		//简笔腿UI的动态更新
		gimbal_move.dynamicUI.LegAngleLeft=acos((gimbal_move.TwoBoardControlGimbal.LeftLength*gimbal_move.TwoBoardControlGimbal.LeftLength+0.21*0.21-0.25*0.25)/2/0.21/gimbal_move.TwoBoardControlGimbal.LeftLength);
		gimbal_move.dynamicUI.LeftX1=(int32_t)(150*arm_cos_f32(PI-gimbal_move.TwoBoardControlGimbal.LeftPhi0-gimbal_move.dynamicUI.LegAngleLeft));
		gimbal_move.dynamicUI.LeftY1=-(int32_t)(150*arm_sin_f32(PI-gimbal_move.TwoBoardControlGimbal.LeftPhi0-gimbal_move.dynamicUI.LegAngleLeft));
		gimbal_move.dynamicUI.LeftX2=-(int32_t)(gimbal_move.TwoBoardControlGimbal.LeftLength*600*arm_cos_f32(gimbal_move.TwoBoardControlGimbal.LeftPhi0));
		gimbal_move.dynamicUI.LeftY2=-(int32_t)(gimbal_move.TwoBoardControlGimbal.LeftLength*600*arm_sin_f32(gimbal_move.TwoBoardControlGimbal.LeftPhi0));
		gimbal_move.dynamicUI.LegAngleRight=acos((gimbal_move.TwoBoardControlGimbal.RightLength*gimbal_move.TwoBoardControlGimbal.RightLength+0.21*0.21-0.25*0.25)/2/0.21/gimbal_move.TwoBoardControlGimbal.RightLength);
		gimbal_move.dynamicUI.RightX1=(int32_t)(150*arm_cos_f32(PI-gimbal_move.TwoBoardControlGimbal.RightPhi0-gimbal_move.dynamicUI.LegAngleRight));
		gimbal_move.dynamicUI.RightY1=-(int32_t)(150*arm_sin_f32(PI-gimbal_move.TwoBoardControlGimbal.RightPhi0-gimbal_move.dynamicUI.LegAngleRight));
		gimbal_move.dynamicUI.RightX2=-(int32_t)(gimbal_move.TwoBoardControlGimbal.RightLength*600*arm_cos_f32(gimbal_move.TwoBoardControlGimbal.RightPhi0));
		gimbal_move.dynamicUI.RightY2=-(int32_t)(gimbal_move.TwoBoardControlGimbal.RightLength*600*arm_sin_f32(gimbal_move.TwoBoardControlGimbal.RightPhi0));
    //ui界面
    cal_draw_gimbal_relative_angle_tangle(&tank);
}
