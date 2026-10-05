#ifndef __VMC_CALC_H
#define __VMC_CALC_H

#include "main.h"
#include "INS_Task.h"
#include "robot_param.h"

typedef struct
{
	/*左右两腿的公共参数，固定不变*/
	float l5; // AE长度 //单位为m
	float l1; // 单位为m
	float l2; // 单位为m
	float l3; // 单位为m
	float l4; // 单位为m

	float XB, YB; // B点的坐标
	float XD, YD; // D点的坐标

	float XC, YC;	// C点的直角坐标
	float L0, phi0; // C点的极坐标
	float alpha;
	float d_alpha;

	float lBD; // BD两点的距离

	float d_phi0;	 // 现在C点角度phi0的变换率
	float last_phi0; // 上一次C点角度，用于计算角度phi0的变换率d_phi0

	float A0, B0, C0; // 中间变量
	float phi2, phi3;
	float phi1, phi4;

	/* MuJoCo XML 主动关节坐标：前支路 q_front，后支路 q_rear。 */
	float q_front, q_rear;
	/* 推荐模型的轮轴坐标，s=+base Y，z=+base Z。 */
	float wheel_s, wheel_z;
	/* J_H = d(L0, phi0) / d(q_front, q_rear)，按行存储。 */
	float jh_l0_qfront, jh_l0_qrear;
	float jh_phi0_qfront, jh_phi0_qrear;
	/* 原始 MuJoCo 主动关节力矩：[front, rear]。 */
	float raw_joint_torque[2];
	uint8_t kinematics_valid;

	float j11, j12, j21, j22; // 旧 phi1/phi4 接口及遥测兼容系数

	float torque_set[2];   // 0=phi1/后支路，1=phi4/前支路（旧发送接口）
	float velocity_set[2]; // 0是前腿，1是后腿的关节电机速度控制量
	float position_set[2]; // 0是前腿，1是后腿的关节电机位置控制量
	float wheel_T;		   // 轮毂电机的扭矩控制量

	float F0;
	float Tp;
	float F02;

	float theta;
	float d_theta; // theta的一阶导数
	float last_d_theta;
	float dd_theta; // theta的二阶导数

	float d_L0;	 // L0的一阶导数
	float dd_L0; // L0的二阶导数
	float last_L0;
	float last_d_L0;

	float FN; // 支持力

	uint8_t first_flag;
	uint8_t leg_flag; // 腿长完成标志

	float aver[4]; // 支持力均值滤波数组
	float aver_fn; // 支持力均值滤波结果
} vmc_leg_t;

extern void VMC_init(vmc_leg_t *vmc); // 给杆长赋值

extern void VMC_calc_1(vmc_leg_t *vmc, float Pitch, float PithGyro, float dt); // 计算theta和d_theta给lqr用，同时也计算腿长L0
extern void VMC_calc_2(vmc_leg_t *vmc);										   // 计算期望的关节输出力矩

extern uint8_t ground_detection(vmc_leg_t *vmc); // 腿离地检测

extern void CalcPhi1AndPhi4(float phi0, float l0, float phi1_phi4[2]);

extern void LQR_K_calc(float len, float k[12]);
extern void CalcLQR(float k[12], float x[6], float T_Tp[2]);

#endif
