////哨兵比赛时的行为状态机，明年大概率用不到，只是暂时保留

///**
// * @file radar_task.c
// * @author Lkhannn
// * @brief navigation
// * @version 1.0
// * @date 2024-04-24
// *
// * @copyright Copyright (c) 2024 SPR
// *
// */
//#include "radar_task.h"
//#include "gimbal_task.h"
//#include "gimbal_behaviour.h"
//#include "cmsis_os.h"
//#include "main.h"
//#include "INS_task.h"
//#include "arm_math.h"
//#include "detect_task.h"
//#include "user_lib.h"
//#include "USART_receive.h"
//#include "config.h"
//#include "referee.h"
//#include "fsm.h"
//#include "CAN_receive.h"
//#include "shoot_task.h"

//#define pointA Ring_Shaped_Elevated_Ground
//#define attack_mode 1//1-outpost 2-robot
//FSM location_fsm;

//int current_event = 0;
//int des = 0;

//extern int target_lost_time;

//radar_control_t radar_control;
//int test_flag = 0;

//inline float f32_abs(float a)
//{
//	if(a >= 0) return a;
//	else return -a;
//}

//enum destination{
//  Born_Place = 1,
//  Restoration_Zone,
//  Road_Zone,
//  Outpost_Zone,
//  Ring_Shaped_Elevated_Ground,
//	Trapezoid_shaped_Elevated_Ground,
//};

//enum location_event{
//	start = 1,
//  arrive_road,
//  outpost_HP_low,
//  outpost_HP_high,
//  arrive_born,
//  arrive_ring,
//	outpost_time_dead,
//  under_attack,
//  low_HP,
//  no_attack,
//  arrive_re,
//  re_finish,
//	out_patrol,
//	face_centre_road,
//	face_centre_born,
//	radar_dead,
//	try_again,
//	all_over,
//};

//enum location_state
//{
//	OFF = 1,
//  NAV_ROAD,
//  WAIT_ROAD,
//  NAV_BORN,
//  NAV_RING,
//  SPIN_WAIT,
//  WAIT_RING,
//  CRAZY_SPIN,
//  NAV_RE,
//  WAIT_HP,
//	BACK_CENTRE,
//	WAIT_BORN,
//};


//static void decision_data_update(void);
//static void HP_deduction_check(uint16_t current_HP);
//static void sentry_patrol_check(uint32_t patrol_status);
//static void des_change_check(void);

//void off(void);
//void nav_road(void);	
//void wait_road(void);
//void nav_born(void);
//void nav_ring(void);
//void wait_ring(void);
//void spin_wait(void);
//void crazy_spin(void);
//void nav_re(void);
//void wait_hp(void);
//void back_centre(void);
//void wait_born(void);

//FsmTable location_table[] =
//{
//		{OFF, start, NAV_ROAD, nav_road},
//		{NAV_ROAD,radar_dead,NAV_BORN,nav_born},
//    {NAV_ROAD,arrive_road,WAIT_ROAD,wait_road},
//		{WAIT_ROAD,outpost_HP_low,NAV_BORN,nav_born},
//    {NAV_BORN,arrive_born,WAIT_BORN,wait_born},
//		{WAIT_BORN,try_again,NAV_ROAD,nav_road},
//		{WAIT_BORN,all_over,SPIN_WAIT,spin_wait},
//		
//};



//void radar_task(void const *pvParameters)
//{
//  vTaskDelay(RADAR_TASK_INIT_TIME);

//  FSM_Regist(&location_fsm, location_table);
//	location_fsm.curState = OFF;
//	location_fsm.size = sizeof(location_table) / sizeof(FsmTable);
//	location_fsm.transfer_flag = 0;
//	current_event = start;
//	
//	decision_data_update();
//  
//  while (1)
//  {
//    if(test_flag == 1)//(switch_is_up(rc_ctrl.rc.s[GIMBAL_MODE_CHANNEL])) //left_up
//    {
//      FSM_EventHandle(&location_fsm, current_event);
//    }
//    else
//    {
//      location_fsm.curState = OFF;
//      current_event = start;
//			radar_control.spin_flag = 0;
//			radar_control.crazy_spin_flag = 0;
//			des = 0;
//    }
//		
//		decision_data_update();
//	
//		//send referee data to radar
//		//CAN_cmd_navigation(des);

//    vTaskDelay(RADAR_CONTROL_TIME);
//  }
//}

//static void decision_data_update(void)
//{
//	radar_control.my_outpost = Get_Outpost_HP(Get_Team());
//	radar_control.enemy_outpost = Get_Outpost_HP(!(Get_Team()));
//	radar_control.vision_lost_time = target_lost_time;
//	//radar_control.game_remain_time = get_game_stage_remain_time();
//	radar_control.my_HP = get_robot_HP();
//	radar_control.patrol_status = get_sentry_patrol_zone_status();
//	radar_control.bullet_allowance = get_bullet_allowance();
//	HP_deduction_check(radar_control.my_HP);
//	sentry_patrol_check(radar_control.patrol_status);
//}

//static void HP_deduction_check(uint16_t current_HP)
//{
//  static int hurt_time = 0;
//	if(current_HP < radar_control.previous_HP)
//  {
//    radar_control.HP_deduction_flag = 1;
//    hurt_time = 0;
//  }
//  else
//  {
//    hurt_time++;
//    if(hurt_time > 3000)
//    {
//      radar_control.HP_deduction_flag = 0;
//    }
//  }
//	radar_control.previous_HP = current_HP;
//}

//static void sentry_patrol_check(uint32_t patrol_status)
//{
//	//static int update_flag = 0;
//	if(patrol_status == 1)
//	{
//		radar_control.leave_time = radar_control.game_remain_time;
//	}
//}


//static void des_change_check(void)
//{
//	static int ongoing_duration = 0; 
//	static int last_time = 0; 
//	
//	if(auto_chassis.ongoing_flag == 1) {
//		if(last_time != 0)
//		{
//			ongoing_duration += (last_time - radar_control.game_remain_time);
//		}

//		last_time = radar_control.game_remain_time;

//		if(ongoing_duration >= 30) 
//		{
//			ongoing_duration = 0;
//			radar_control.change_des_flag = 1; 
//			return;
//		}
//	}
//	else
//	{
//		ongoing_duration = 0;
//		last_time = 0;
//	}
//	
//	radar_control.change_des_flag = 2; 
//	return;
//}


//void off(void)
//{
//  des = 0;
//  return;
//}


//void nav_road(void)
//{
//  des = Road_Zone;

//	radar_control.region_search_flag = 1;
//	if(radar_control.bullet_allowance > 300)
//	{
//		radar_control.shoot_outpost_flag = 1;
//	}
//	else
//	{
//		radar_control.shoot_outpost_flag = 0;
//	}

//	des_change_check();
//	
//	if(auto_chassis.ongoing_flag == 2 && auto_chassis.destination == Road_Zone)
//	{
//		current_event = arrive_road;
//		location_fsm.transfer_flag = 1;
//		return;
//	}
//	
//	if(radar_control.change_des_flag == 1)
//	{
//		current_event = radar_dead;
//		radar_control.shoot_outpost_flag = 0;
//		radar_control.region_search_flag = 0;
//		location_fsm.transfer_flag = 1;
//		radar_control.change_des_flag = 2; 

//		return;
//		
//	}
//	
//  return;
//}


//void wait_road(void)
//{
//	des = Road_Zone;
//	
//	radar_control.region_search_flag = 1;
//	if(radar_control.bullet_allowance > 300)
//	{
//		radar_control.shoot_outpost_flag = 1;
//	}
//	else
//	{
//		radar_control.shoot_outpost_flag = 0;
//	}

//	if(radar_control.my_outpost <= 300 || radar_control.game_remain_time < 240)
//  {
//		current_event = outpost_HP_low;
//		location_fsm.transfer_flag = 1;
//		radar_control.shoot_outpost_flag = 0;
//		radar_control.region_search_flag = 0;
//		return;
//	}
//	if(radar_control.enemy_outpost == 0)
//	{
//		if(radar_control.my_outpost < 600)
//    {
//      current_event = outpost_HP_low;
//			radar_control.shoot_outpost_flag = 0;
//			radar_control.region_search_flag = 0;
//			location_fsm.transfer_flag = 1;
//    }
//		
//	}
//  return;
//}

//void nav_ring(void)
//{
//  des = Ring_Shaped_Elevated_Ground;

//	if(auto_chassis.ongoing_flag == 2 && auto_chassis.destination == Ring_Shaped_Elevated_Ground)
//	{
//		current_event = arrive_ring;
//		location_fsm.transfer_flag = 1;
//		return;
//	}

//	if(radar_control.my_outpost < 200 || radar_control.game_remain_time < 240)
//	{
//		if(auto_chassis.ramp_flag == 2)
//		{
//			current_event = outpost_time_dead;
//			location_fsm.transfer_flag = 1;
//			return;
//		}
//	}
//		
//  return;

//}

//void wait_ring(void)
//{
//  des = Ring_Shaped_Elevated_Ground;
//	
//	if(radar_control.my_outpost < 200 || radar_control.game_remain_time < 240)
//	{
//		if(auto_chassis.ramp_flag == 2)
//		{
//			current_event = outpost_time_dead;
//			location_fsm.transfer_flag = 1;
//			return;
//		}
//	}
//}


//void nav_born(void)
//{
//  des = Born_Place;
//  
//  if(auto_chassis.ongoing_flag == 2 && auto_chassis.destination == Born_Place)
//	{
//		current_event = arrive_born;
//		location_fsm.transfer_flag = 1;
//		return;
//	}
//	
//  return;
//}
//		
//		
//void back_centre(void)
//{
//	static int back_flag = 0;
//	radar_control.back_centre_flag = 1;
//	if(f32_abs(gimbal_control.gimbal_yaw_motor.absolute_angle)<0.2f)
//	{
//		back_flag++;
//		if(back_flag == 1000)
//		{
//			back_flag = 0;
//			radar_control.back_centre_flag = 0;
//			if(des == Road_Zone)
//			{
//				current_event = face_centre_road;
//			}
//			else
//			{
//				current_event = face_centre_born;
//      }
//			location_fsm.transfer_flag = 1;
//		}
//	}
//	
//	return;
//}


//void wait_born(void)
//{
//	if(radar_control.my_outpost > 600 && radar_control.game_remain_time > 330)
//	{
//		current_event = try_again;
//	}
//	else
//	{
//		current_event = all_over;
//	}
//	location_fsm.transfer_flag = 1;
//}

//		
//void spin_wait(void)
//{
//  radar_control.spin_flag = 1;
//}







//void crazy_spin()
//{
//  radar_control.crazy_spin_flag = 1;
//	
//	if(radar_control.my_HP < 180)
//  {
//		radar_control.crazy_spin_flag = 0;
//    current_event = low_HP;
//    location_fsm.transfer_flag = 1;
//    return;
//  }
//	if(radar_control.leave_time - radar_control.game_remain_time>25)
//	{
//		radar_control.crazy_spin_flag = 0;
//    current_event = out_patrol;
//    location_fsm.transfer_flag = 1;
//		return;
//	}
//  if(radar_control.HP_deduction_flag == 0)
//  {
//		radar_control.crazy_spin_flag = 0;
//		radar_control.spin_flag = 1;
//    current_event = no_attack;
//    location_fsm.transfer_flag = 1;
//    return;
//  }

//}
//		

//void nav_re(void)
//{
//  des = Restoration_Zone;
//  
//  if(auto_chassis.ongoing_flag == 2 && auto_chassis.destination == Restoration_Zone)
//	{
//		current_event = arrive_re;
//		location_fsm.transfer_flag = 1;
//		return;
//	}
//	if(radar_control.leave_time - radar_control.game_remain_time>25)
//	{
//		current_event = out_patrol;
//		location_fsm.transfer_flag = 1;
//		return;
//	}
//	
//  return;

//}


//void wait_hp(void)
//{
//  if(radar_control.my_HP==400)
//  {
//    current_event = re_finish;
//    location_fsm.transfer_flag = 1;
//    return;
//  }
//  if(radar_control.leave_time - radar_control.game_remain_time>25)
//  {
//    current_event = re_finish;
//    location_fsm.transfer_flag = 1;
//    return;
//  }
//}


