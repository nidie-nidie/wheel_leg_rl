#include "UI_task.h"

ROBOT_STATE_e ROBOT_STATE;//机器人状态

char str1[]="hello";

uint8_t temp_id=2;

uint8_t is_inited;

void UI_INIT(void)
{
	//搜索ID
	//清空UI
	
	//初次绘制
	if(dianguan_cmd.robot_status.robot_id==RED_2)
	{
		temp_id=RED_2;
	}	
		if(dianguan_cmd.robot_status.robot_id==BLUE_2)
	{
		temp_id=BLUE_2;
	}	
	
}
	
	
void UI_TASK(void)
{
//	if(temp_id==0)
//	{
//		UI_INIT();
//	}
//	else
//	{
	 rf_ui_write_string(&dianguan_cmd, "sd1", str1,  strlen(str1),  15,  5,  250,  250,  1,  temp_id);
	 rf_ui_string_upgrade(&dianguan_cmd);
//	}
	
}


void UI_UPLOAD(void)//上传
{
	if(ROBOT_STATE==NORMAL){
	strcpy(ui_g_Ungroup_ROBOT_STATE->string, "NORMAL");}
	if(ROBOT_STATE==CUSTOM){
	strcpy(ui_g_Ungroup_ROBOT_STATE->string, "CUSTOM");}
	
//	if(ARM_CMD_data.sucker_mode==OFF){
//	strcpy(ui_g_Ungroup_SUCKER_STATE->string, "OFF");}
//		if(ARM_CMD_data.sucker_mode==ON){
//	strcpy(ui_g_Ungroup_SUCKER_STATE->string, "ON ");}

}
