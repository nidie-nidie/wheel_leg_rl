#ifndef _MAHONY_FILTER_H
#define _MAHONY_FILTER_H

#include <math.h>
#include <stdlib.h>
#include "stm32h7xx.h"
#include "arm_math.h"

/*************************************
创建时间：2023年09月02日
功能简述：实现 Mahony 姿态解算算法的模块化封装
***************************************/

#define DEG2RAD 0.0174533f // 角度转弧度系数 (π/180)
#define RAD2DEG 57.295671f // 弧度转角度系数 (180/π)

// 定义三轴数据结构体
typedef struct Axis3f_t
{
	float x;
	float y;
	float z;
} Axis3f;

// 定义 MAHONY_FILTER_t 结构体，用于封装 Mahony 滤波器的数据和函数
struct MAHONY_FILTER_t
{
	// 滤波参数
	float Kp, Ki;	  // 比例和积分系数（用于控制加速度计修正陀螺仪的速度）
	float dt;		  // 采样时间间隔（解算周期，单位：秒）
	Axis3f gyro, acc; // 陀螺仪和加速度计的输入数据

	// 状态变量
	float exInt, eyInt, ezInt; // 误差积分累计值（用于消除陀螺仪零偏）
	float q0, q1, q2, q3;	   // 四元数（表示当前姿态）
	float rMat[3][3];		   // 旋转矩阵（方向余弦矩阵 DCM）

	// 输出结果
	float pitch, roll, yaw; // 姿态角：俯仰角(Pitch)、横滚角(Roll)、偏航角(Yaw)

	// 函数指针（实现类似面向对象的封装）
	void (*mahony_init)(struct MAHONY_FILTER_t *mahony_filter, float Kp, float Ki, float dt);
	void (*mahony_input)(struct MAHONY_FILTER_t *mahony_filter, Axis3f gyro, Axis3f acc);
	void (*mahony_update)(struct MAHONY_FILTER_t *mahony_filter);
	void (*mahony_output)(struct MAHONY_FILTER_t *mahony_filter);
	void (*RotationMatrix_update)(struct MAHONY_FILTER_t *mahony_filter);
};

// 外部函数声明
void mahony_init(struct MAHONY_FILTER_t *mahony_filter, float Kp, float Ki, float dt); // 初始化滤波器参数
void mahony_input(struct MAHONY_FILTER_t *mahony_filter, Axis3f gyro, Axis3f acc);	   // 输入传感器数据
void mahony_update(struct MAHONY_FILTER_t *mahony_filter);							   // 执行一次姿态解算更新
void mahony_output(struct MAHONY_FILTER_t *mahony_filter);							   // 计算并输出欧拉角
void RotationMatrix_update(struct MAHONY_FILTER_t *mahony_filter);					   // 更新旋转矩阵

#endif
