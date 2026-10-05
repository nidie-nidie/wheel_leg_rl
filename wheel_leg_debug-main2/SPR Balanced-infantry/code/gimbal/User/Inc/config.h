/**
 * @file config.h
 * @Function 用于面对部分特殊需求时进行一键切换
 */

#ifndef __CONFIG_H
#define __CONFIG_H

//线程开关
#define GIMBAL_TASK			//云台线程
#define REFEREE_TASK		//裁判系统收发线程
#define SHOOT_TASK			//发射线程
#define DETECT_TASK			//错误检测线程
#define INS_TASK				//姿态传感器线程

#define PT_link_en 1 		//1启用图传链路控制
#define Gimbal_Auto_Debug 0 //1时一键修改模式进行视觉调试模式

#endif

