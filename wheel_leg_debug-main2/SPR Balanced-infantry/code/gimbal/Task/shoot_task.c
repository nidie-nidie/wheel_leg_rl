#include "shoot_task.h"
#include "main.h"
#include "cmsis_os.h"
#include "arm_math.h"
#include "CAN_receive.h"
#include "USART_receive.h"
#include "detect_task.h"
#include "pid.h"
#include "referee.h"
#include "config.h"
#include "user_lib.h"
#include "referee.h"

extern gimbal_move_t gimbal_move;
shoot_move_t shoot_control;
static void shoot_feedback_update(void);
static void shoot_init(void);
static void shoot_set_mode(void);
static void shoot_set_control(void);
static void trigger_motor_turn_back(void);
static void shoot_control_loop(void);
//static void shoot_single_detect(void);
static void shoot_heat_detect(void);

uint16_t MouseLeftTime=0;

int X_flag = 0, X_flag_last = 0;

float speed_cut=0;
uint16_t Fn2Time=0;

void fire_task(void const *pvParameters)
{
  //空闲一段时间
  vTaskDelay(300);
	//发射机构初始化
	shoot_init();
  	//判断发射机构相关的电机是否都在线
		while (toe_is_error(LEFT_SHOOT_MOTOR_TOE)||toe_is_error(RIGHT_SHOOT_MOTOR_TOE)||toe_is_error(TRIGGER_SHOOT_MOTOR_TOE) )
		{
			vTaskDelay(1);
		}
  while (1)
  {
		//发射模式选择
		shoot_set_mode();
		//发射机构数据更新
		shoot_feedback_update();
		//射击控制量设置
		shoot_set_control();
		//热量检测
		shoot_heat_detect();
		//射击控制量计算
		shoot_control_loop();
		
    if(toe_is_error(LEFT_SHOOT_MOTOR_TOE)||toe_is_error(RIGHT_SHOOT_MOTOR_TOE)||toe_is_error(TRIGGER_SHOOT_MOTOR_TOE) )
		{
			CAN_cmd_shoot(0);
			CAN_cmd_fricion(0,0);
		}
		else
		{
//			CAN_cmd_shoot(0);
//			CAN_cmd_fricion(0,0);
			CAN_cmd_shoot(shoot_control.trigger_motor.shoot_motor.set_current);
			CAN_cmd_fricion(shoot_control.left_fricion_motor.shoot_motor.set_current, shoot_control.right_fricion_motor.shoot_motor.set_current);
		}
		
    vTaskDelay(1);
  }
}

static void shoot_init(void)
{
	//模式默认停下
	shoot_control.shoot_mode=SHOOT_STOP;
	//电机指针获取电机数据
	shoot_control.left_fricion_motor.shoot_motor.shoot_motor_measure=left_motor_measure_point();
	shoot_control.right_fricion_motor.shoot_motor.shoot_motor_measure=right_motor_measure_point();
	shoot_control.trigger_motor.shoot_motor.shoot_motor_measure=trigger_measure_point();
	//初始化PID
	PID_init(&shoot_control.left_fricion_motor.shoot_motor_absolute_speed_pid,30,0,0,0,0.001,4000,0);//滑环线5A持续电流，所以10A/16384*5/2=4096
	PID_init(&shoot_control.right_fricion_motor.shoot_motor_absolute_speed_pid,30,0,0,0,0.001,4000,0);
	PID_init(&shoot_control.trigger_motor.shoot_motor_absolute_speed_pid,1200,0,0,0,0.001,10000,0);//注意2006的MAX是10000
	//数据更新
	shoot_feedback_update();
	//堵转时间初始化
	shoot_control.block_time=0;
	//单发时间初始化
	shoot_control.single_fire_time=0;
	//热量检测初始化
	shoot_control.heat_detect_fire=0;
	shoot_control.heat_timeout=0;
	//自瞄需要的发弹个数
	shoot_control.count=0;
	
	shoot_control.BulletSpeedFlag=0;
}

static void shoot_feedback_update(void)
{
	shoot_control.left_fricion_motor.speed=shoot_control.left_fricion_motor.shoot_motor.shoot_motor_measure->speed_rads;
	shoot_control.right_fricion_motor.speed=shoot_control.right_fricion_motor.shoot_motor.shoot_motor_measure->speed_rads;
	shoot_control.trigger_motor.speed=shoot_control.trigger_motor.shoot_motor.shoot_motor_measure->speed_rads;
	//拨盘角度控制实现单发，但是不准，后面要变为判断慢速度环+判断是否发射的逻辑
	shoot_control.trigger_motor.angle=shoot_control.trigger_motor.shoot_motor.shoot_motor_measure->angle;
	
	shoot_control.bullet_speed=shoot_data_t.bullet_speed;
	//超射速自动降低射速,但是这里只降一次，如果可以，可以多级降速
	if(shoot_control.bullet_speed>25.0f&&shoot_control.BulletSpeedFlag==0)
	{
	shoot_control.BulletSpeedFlag=1;
	speed_cut=speed_cut-50;
	}
}

static void shoot_set_mode(void)
{
	shoot_control.last_shoot_mode=shoot_control.shoot_mode;
			
	//自锁，按一下Fn1就是选择开还是关摩擦轮
	if(remote_data.fn_1==1&&shoot_control.LastFn1==0)
	{
		shoot_control.FrictionSwitch=!shoot_control.FrictionSwitch;
	}
		if(shoot_control.FrictionSwitch==0)
		{
			shoot_control.shoot_mode = SHOOT_STOP;
		}
		else
		{
			shoot_control.shoot_mode = SHOOT_READY;
		}
		
	if(remote_data.trigger==1)
	{
#if Gimbal_Auto_Debug
		shoot_control.shoot_mode = SHOOT_AUTO;
#else
		shoot_control.shoot_mode = SHOOT_BULLET;
#endif
	}
	
	/******************键鼠控制摩擦轮部分*****************/
	
	if(gimbal_move.gimbal_behaviour==GIMBAL_OPERATION)
	{

			//X键开关摩擦轮
			X_flag_last=X_flag;
			X_flag=(remote_data.key & KEY_PRESSED_OFFSET_X)?1:0;
			if(X_flag-X_flag_last==1 && !((remote_data.key & KEY_PRESSED_OFFSET_CTRL)?1:0))
			{
				shoot_control.FrictionSwitch=!shoot_control.FrictionSwitch;
			}
			
			//左键连发
			if(remote_data.mouse_left == 1&&!((remote_data.key & KEY_PRESSED_OFFSET_CTRL)?1:0)&&!((remote_data.key & KEY_PRESSED_OFFSET_Z)?1:0))
			{
				shoot_control.shoot_mode=SHOOT_BULLET;
			}
			
			//右键加左键自瞄
			if(remote_data.mouse_right == 1 && remote_data.mouse_left == 1 && !((remote_data.key & KEY_PRESSED_OFFSET_CTRL)?1:0)&&!((remote_data.key & KEY_PRESSED_OFFSET_Z)?1:0))
			{
			shoot_control.shoot_mode=SHOOT_AUTO;
				if(game_state.game_progress!=4)
				{
					shoot_control.shoot_mode=SHOOT_READY;
				}
			}
	}
	
	/*************************DT7遥控器控制部分，换图传链路就可以不管*************************/
	if (switch_is_up(rc_ctrl.rc.s[RIGHT_SWITCH]))
	{
			shoot_control.shoot_mode=SHOOT_BULLET;
	}
	else if(switch_is_mid(rc_ctrl.rc.s[RIGHT_SWITCH]))
	{
		shoot_control.shoot_mode = SHOOT_READY;
	}
	else if(switch_is_down(rc_ctrl.rc.s[RIGHT_SWITCH]))
	{
		shoot_control.shoot_mode = SHOOT_STOP;
	}
	/*************************因为用的数据不一样，不影响，只是没删除*************************/
	
	shoot_control.LastFn1=remote_data.fn_1;
}
static void shoot_set_control(void)
{
	switch (shoot_control.shoot_mode)
	{
	case SHOOT_STOP:
		{
			//设置速度为零，可以再加个模式是直接零电流
			shoot_control.left_fricion_motor.speed_set=0;
			shoot_control.right_fricion_motor.speed_set=0;
			shoot_control.trigger_motor.speed_set=0;
			//可以再加一个pid是maxout设置，实现一个加速电机限幅，但是我暂时给一个固定maxout了
			break;
		}
	case SHOOT_READY:
		{
			//给摩擦轮设置一个速度，注意，我代码里摩擦轮速度反馈单位是rad/s
			shoot_control.left_fricion_motor.speed_set=LEFT_FRICTION_SPEED+speed_cut;
			shoot_control.right_fricion_motor.speed_set=RIGHT_FRICTION_SPEED-speed_cut;
			shoot_control.trigger_motor.speed_set=0;
			
			if(remote_data.fn_2==1||(((remote_data.key & KEY_PRESSED_OFFSET_CTRL)?1:0) && ((remote_data.key & KEY_PRESSED_OFFSET_X)?1:0)))
			{
			shoot_control.trigger_motor.speed_set=-TRIGGER_SPEED/4;
			}
			break;
		}
	case SHOOT_BULLET:
		{
			shoot_control.left_fricion_motor.speed_set=LEFT_FRICTION_SPEED+speed_cut;
			shoot_control.right_fricion_motor.speed_set=RIGHT_FRICTION_SPEED-speed_cut;
			shoot_control.trigger_motor.speed_set=TRIGGER_SPEED;//2*PI/9*10=7，根据设计，一圈9个弹丸，10Hz发射频率就是设置7rad/s
			
			if(remote_data.fn_2==1||(((remote_data.key & KEY_PRESSED_OFFSET_CTRL)?1:0) && ((remote_data.key & KEY_PRESSED_OFFSET_X)?1:0)))
			{
			shoot_control.trigger_motor.speed_set=-TRIGGER_SPEED/4;
			}
			//卡弹检测
			trigger_motor_turn_back();
			break;
		}
	case SHOOT_SINGLE_BULLET:
		{
			//单发检测
//			shoot_single_detect();
			
			shoot_control.left_fricion_motor.speed_set=LEFT_FRICTION_SPEED+speed_cut;
			shoot_control.right_fricion_motor.speed_set=RIGHT_FRICTION_SPEED-speed_cut;
			shoot_control.trigger_motor.speed_set=TRIGGER_SPEED;//旋转速度转太快怕来不及控制

			if(auto_shoot.mode==2)
			{
				shoot_control.trigger_motor.angle_set = shoot_control.trigger_motor.angle + 0.2f;
				shoot_control.count++;
			}
			
			if (shoot_control.trigger_motor.angle_set - shoot_control.trigger_motor.angle > 0)
			{
			shoot_control.trigger_motor.speed_set=TRIGGER_SPEED;
			}
			else
			{
			shoot_control.trigger_motor.speed_set=0;
			}
			
			//卡弹检测
			trigger_motor_turn_back();
				
				//如果单发成功就不再旋转
				if(shoot_control.single_fire)
				{
					shoot_control.trigger_motor.speed_set=0;
				}
			break;
		}
	case SHOOT_AUTO:
		{
			
			//通过控制拨盘旋转实现发射，修改angle_set得一次几发
			shoot_control.left_fricion_motor.speed_set=LEFT_FRICTION_SPEED+speed_cut;
			shoot_control.right_fricion_motor.speed_set=RIGHT_FRICTION_SPEED-speed_cut;

			if(auto_shoot.mode==2)
			{
				shoot_control.trigger_motor.angle_set = shoot_control.trigger_motor.angle + 0.2f;
				shoot_control.count++;
			}
			
			if (shoot_control.trigger_motor.angle_set - shoot_control.trigger_motor.angle > 0)
			{
			shoot_control.trigger_motor.speed_set=TRIGGER_SPEED;
			}
			else
			{
			shoot_control.trigger_motor.speed_set=0;
			}
			
			
//			//单发检测
//			shoot_single_detect();
//			
//			//需要发送而单发成功了则重置单发
//			if(auto_shoot.mode==2&&shoot_control.single_fire==1)
//			{
//			shoot_control.single_fire=0;
//			}
//			
//			//需要发射或单发未完成则继续转
//			if(auto_shoot.mode==2||shoot_control.single_fire==0)//需要发射时会是2
//			{
//				shoot_control.trigger_motor.speed_set=TRIGGER_SPEED*3;//感觉不怎么射可以考虑将拨盘速度变快					
//			}
//			else
//			{
//				shoot_control.trigger_motor.speed_set=0;
//			}
//			shoot_control.left_fricion_motor.speed_set=LEFT_FRICTION_SPEED+speed_cut;
//			shoot_control.right_fricion_motor.speed_set=RIGHT_FRICTION_SPEED-speed_cut;


			//卡弹检测
			if(shoot_control.trigger_motor.speed_set==TRIGGER_SPEED)
			{
			trigger_motor_turn_back();
			}
			break;
		}
 default:
 {
	 break;
 }
	}
}

static void trigger_motor_turn_back(void)
{
	//机械后面要求不需要反转了
	
//	//当拨盘转不起来的时候开始计时（速度明显小于当前set值，单边保证反转不会不小心清零）
//	if(shoot_control.trigger_motor.speed>BLOCK_TRIGGER_SPEED)
//	{
//		shoot_control.block_time++;
//	}
//	else
//	{
//		shoot_control.block_time=0;
//	}
//	
//	//刚开始计时，也许是误判或弹丸挤压导致的小卡，所以继续转
//	if(shoot_control.block_time<BLOCK_TIME)
//	{
//    shoot_control.trigger_motor.speed_set = TRIGGER_SPEED;
//	}
//	//计时一段时间，也许是进入异物卡住弹道，将拨盘反转退出
//	else if(shoot_control.block_time>BLOCK_TIME&&shoot_control.block_time<REVERSE_TIME)
//	{
//		shoot_control.trigger_motor.speed_set = -TRIGGER_SPEED/3;
//	}
////	//计时反转一段时间，重新变回正常状态（也许应该再加个判断，循环多次自动关掉电机防止电机烧坏）
//	else if (shoot_control.block_time>REVERSE_TIME)
//	{
//		shoot_control.block_time=0;
//	}
}

static void shoot_control_loop(void)
{
	//左摩擦轮
	PID_calc(&shoot_control.left_fricion_motor.shoot_motor_absolute_speed_pid,shoot_control.left_fricion_motor.speed,shoot_control.left_fricion_motor.speed_set);
	shoot_control.left_fricion_motor.shoot_motor.set_current=(int16_t)shoot_control.left_fricion_motor.shoot_motor_absolute_speed_pid.out;
	//右摩擦轮
	PID_calc(&shoot_control.right_fricion_motor.shoot_motor_absolute_speed_pid,shoot_control.right_fricion_motor.speed,shoot_control.right_fricion_motor.speed_set);
	shoot_control.right_fricion_motor.shoot_motor.set_current=(int16_t)shoot_control.right_fricion_motor.shoot_motor_absolute_speed_pid.out;
	//拨盘电机
	PID_calc(&shoot_control.trigger_motor.shoot_motor_absolute_speed_pid,shoot_control.trigger_motor.speed,shoot_control.trigger_motor.speed_set);
	shoot_control.trigger_motor.shoot_motor.set_current=(int16_t)shoot_control.trigger_motor.shoot_motor_absolute_speed_pid.out;
}

//大FU没有研发出来，单发没必要了，单发监测参考中科大电控教程
//static void shoot_single_detect(void)
//{
//	//摩擦轮碰到弹丸时，为了保持速度，扭矩会增大
//	if(shoot_control.left_fricion_motor.shoot_motor.shoot_motor_measure->given_current>1000)
//	{
//		shoot_control.single_fire_time++;
//	}
//	//防止误判，扭矩增大一段时间后就确认单发了
//	if(shoot_control.single_fire_time>=7)
//	{
//		shoot_control.single_fire_time=0;
//		shoot_control.single_fire=1;
//	}
//}

float raw_torque=0;
float energy_smooth=0;
uint8_t DetectState=0;
#define ENERGY_TAU 5            // 低通时间常数

uint16_t DeltaHeat=0;
static void shoot_heat_detect(void)
{
	//自己热量监测做不好，参试引入方差、机器学习什么的，但是还是不太行
//	raw_torque=(shoot_control.left_fricion_motor.shoot_motor.shoot_motor_measure->Torque-shoot_control.right_fricion_motor.shoot_motor.shoot_motor_measure->Torque)/2;
//	float sq = raw_torque * raw_torque;        // 偏差平方
//	energy_smooth=energy_smooth + (sq - energy_smooth) / ENERGY_TAU;
//	
//	if(energy_smooth>0.07f&&DetectState==0)
//	{
//	DetectState=1;
//	}
//	else if(DetectState==1&&energy_smooth<0.03f)
//	{
//	DetectState=0;
//	shoot_control.heat_detect_fire=shoot_control.heat_detect_fire+10;
//	}
//	//热量冷却
//	shoot_control.heat_timeout++;
//	if(shoot_control.heat_timeout>=100)//1000
//	{
//		shoot_control.heat_timeout=0;
//		shoot_control.heat_detect_fire=shoot_control.heat_detect_fire-robot_state.shooter_17mm_cooling_rate*0.1;
//		if(shoot_control.heat_detect_fire<0.0f)
//		{
//			shoot_control.heat_detect_fire=0;
//		}
//	}
//	//热量融合
//	shoot_control.current_heat=shoot_control.heat_detect_fire;
//	//即将超热量则不射
//	if(shoot_control.current_heat+15>=robot_state.shooter_17mm_cooling_limit||power_heat_data_t.shooter_heat0+15>=robot_state.shooter_17mm_cooling_limit)
//	{
//		shoot_control.trigger_motor.speed_set=0;
//	}
//参考的是同济的热量前馈，但是感觉效果也一般，还是有概率超热量，还是中科大的热量融合可能比较好，但是自己在连发下一直有问题ORZ
			DeltaHeat=(robot_state.shooter_17mm_cooling_limit-power_heat_data_t.shooter_heat0);
			if(DeltaHeat>0&&DeltaHeat<=40)
			{
				shoot_control.trigger_motor.speed_set=shoot_control.trigger_motor.speed_set/2;
			}
			else if(DeltaHeat>40&&DeltaHeat<=130)
			{
				shoot_control.trigger_motor.speed_set=shoot_control.trigger_motor.speed_set;
			}
			else if(DeltaHeat>130)
			{
				shoot_control.trigger_motor.speed_set=shoot_control.trigger_motor.speed_set*1.5f;
			}
			//即将超热量则不射
			if(power_heat_data_t.shooter_heat0+30>=robot_state.shooter_17mm_cooling_limit)
			{
					shoot_control.trigger_motor.speed_set=0;
			}

}
