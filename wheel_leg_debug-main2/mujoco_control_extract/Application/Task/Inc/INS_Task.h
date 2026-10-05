/**
 ******************************************************************************
 * @file    ins_task.h
 * @author  Wang Hongxi
 * @version V2.0.0
 * @date    2022/2/23
 * @brief
 ******************************************************************************
 * @attention
 *
 ******************************************************************************
 **/

//  IMU 惯性导航任务
// 主要逻辑：读取 BMI088 传感器数据 -> Mahony 滤波解算四元数 -> 剔除重力加速度 -> 提取纯运动加速度 -> 加速度死区处理 -> 积分求速度和位移 -> 处理偏航角(Yaw)过零点连续化。

#ifndef INS_TASK_H
#define INS_TASK_H

/* Includes ------------------------------------------------------------------*/
#include "stdint.h"
#include "cmsis_os.h"

#include "BMI088driver.h"
#include "QuaternionEKF.h"
#include "PID.h"
#include "Config.h"
#include "Mahony_Filter.h"
#include "QuaternionEKF.h"

#include "tim.h"
#include "bsp_pwm.h"
#include "bsp_dwt.h"

#define AXIS_X 0
#define AXIS_Y 1
#define AXIS_Z 2

/* Exported types ------------------------------------------------------------*/
/**
 * @brief typedef structure that contains the information for the INS.
 */
typedef struct
{
	float q[4]; // 四元数解算值

	float Gyro[3];			// 陀螺仪角速度 (rad/s)
	float Accel[3];			// 加速度计数据
	float MotionAccel_b[3]; // 机体系下的运动加速度 (Body Frame)
	float MotionAccel_n[3]; // 导航系下的运动加速度 (Navigation/Earth Frame)

	float AccelLPF; // 加速度低通滤波系数

	// 机体坐标系轴在导航坐标系下的向量表示
	float xn[3];
	float yn[3];
	float zn[3];

	float atanxz;
	float atanyz;

	// 姿态角 (欧拉角)
	float Roll;
	float Pitch;
	float Yaw;
	float YawTotalAngle; // 累计的连续偏航角总角度
	float YawAngleLast;	 // 上一次的偏航角数值 (用于判断过零点)
	float YawRoundCount; // 偏航角旋转的圈数

	float v_n; // 导航系下的水平运动速度积分
	float x_n; // 导航系下的水平运动位移积分

	uint8_t ins_flag; // 惯导系统初始化/运行状态标志位
} INS_t;

/**
 * @brief IMU传感器的安装误差校准参数, demo中可为空
 *
 */
typedef struct
{
	uint8_t flag;

	float scale[3]; // 比例缩放参数

	float Yaw;	 // 安装偏航误差补偿
	float Pitch; // 安装俯仰误差补偿
	float Roll;	 // 安装横滚误差补偿
} IMU_Param_t;

/* Externs---------------------------------------------------------*/
extern void INS_task(void);
extern INS_t INS;

// 坐标系转换：机体系(Body Frame) 转 世界/导航系(Earth Frame)
void BodyFrameToEarthFrame(const float *vecBF, float *vecEF, float *q);
// 坐标系转换：世界/导航系(Earth Frame) 转 机体系(Body Frame)
void EarthFrameToBodyFrame(const float *vecEF, float *vecBF, float *q);

#endif // INS_TASK_H
