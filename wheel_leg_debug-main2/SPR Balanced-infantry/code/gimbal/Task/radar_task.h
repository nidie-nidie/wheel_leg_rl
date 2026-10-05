#ifndef __radar_TASK_H__
#define __radar_TASK_H__



#define RADAR_TASK_INIT_TIME 201
#define RADAR_CONTROL_TIME 2

#include "user_lib.h"

extern void radar_task(void const *pvParameters);
extern int spin_flag;
extern int crazy_spin_flag;

typedef struct
{
    int spin_flag;
		int crazy_spin_flag;
		int HP_deduction_flag;
		int vision_lost_time;
		uint16_t leave_time;
		uint16_t enemy_outpost;
		uint16_t my_outpost;
		uint16_t game_remain_time;
		uint16_t my_HP;
		uint16_t previous_HP;
		uint32_t patrol_status;
		uint16_t bullet_allowance;
		int shoot_outpost_flag;
		int region_search_flag;
		int back_centre_flag;
		int change_des_flag;

} radar_control_t;

extern radar_control_t radar_control;

#endif
