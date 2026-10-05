#include "VMC_Calc.h"

// 建模见 https://zhuanlan.zhihu.com/p/563048952 1.2.2
void VMC_init(vmc_leg_t *vmc) // 给杆长赋值
{
	vmc->l5 = LEG_L5; // AE长度 //单位为m
	vmc->l1 = LEG_L1; // 单位为m
	vmc->l2 = LEG_L2; // 单位为m
	vmc->l3 = LEG_L3; // 单位为m
	vmc->l4 = LEG_L4; // 单位为m

	vmc->aver[0] = 0.0f;
	vmc->aver[1] = 0.0f;
	vmc->aver[2] = 0.0f;
	vmc->aver[3] = 0.0f;
	vmc->aver_fn = 0.0f;
}

void VMC_calc_1(vmc_leg_t *vmc, float Pitch, float PithGyro, float dt) // 计算theta和d_theta给lqr用，同时也计算腿长L0
{
	// L0和phi0的计算见 https://zhuanlan.zhihu.com/p/563048952 1.2.2与 https://zhuanlan.zhihu.com/p/613007726 公式(1.4)
	vmc->YD = vmc->l4 * arm_sin_f32(vmc->phi4);			  // D的y坐标
	vmc->YB = vmc->l1 * arm_sin_f32(vmc->phi1);			  // B的y坐标
	vmc->XD = vmc->l5 + vmc->l4 * arm_cos_f32(vmc->phi4); // D的x坐标
	vmc->XB = vmc->l1 * arm_cos_f32(vmc->phi1);			  // B的x坐标

	vmc->lBD = sqrt((vmc->XD - vmc->XB) * (vmc->XD - vmc->XB) + (vmc->YD - vmc->YB) * (vmc->YD - vmc->YB));

	vmc->A0 = 2 * vmc->l2 * (vmc->XD - vmc->XB);
	vmc->B0 = 2 * vmc->l2 * (vmc->YD - vmc->YB);
	vmc->C0 = vmc->l2 * vmc->l2 + vmc->lBD * vmc->lBD - vmc->l3 * vmc->l3;
	vmc->phi2 = 2 * atan2f((vmc->B0 + sqrt(vmc->A0 * vmc->A0 + vmc->B0 * vmc->B0 - vmc->C0 * vmc->C0)), vmc->A0 + vmc->C0);
	vmc->phi3 = atan2f(vmc->YB - vmc->YD + vmc->l2 * arm_sin_f32(vmc->phi2), vmc->XB - vmc->XD + vmc->l2 * arm_cos_f32(vmc->phi2));
	// C点直角坐标
	vmc->XC = vmc->l1 * arm_cos_f32(vmc->phi1) + vmc->l2 * arm_cos_f32(vmc->phi2);
	vmc->YC = vmc->l1 * arm_sin_f32(vmc->phi1) + vmc->l2 * arm_sin_f32(vmc->phi2);
	// C点极坐标
	vmc->L0 = sqrt((vmc->XC - vmc->l5 / 2.0f) * (vmc->XC - vmc->l5 / 2.0f) + vmc->YC * vmc->YC);

	vmc->phi0 = atan2f(vmc->YC, (vmc->XC - vmc->l5 / 2.0f)); // phi0用于计算lqr需要的theta
	vmc->alpha = M_PI_2 - vmc->phi0;

	if (vmc->first_flag == 0)
	{
		vmc->last_phi0 = vmc->phi0;
		vmc->first_flag = 1;
	}
	vmc->d_phi0 = (vmc->phi0 - vmc->last_phi0) / dt; // 计算phi0变化率，d_phi0用于计算lqr需要的d_theta
	vmc->d_alpha = 0.0f - vmc->d_phi0;

	vmc->theta = M_PI_2 - Pitch - vmc->phi0;  // 得到状态变量1
	vmc->d_theta = (-PithGyro - vmc->d_phi0); // 得到状态变量2

	vmc->last_phi0 = vmc->phi0;

	vmc->d_L0 = (vmc->L0 - vmc->last_L0) / dt;		// 腿长L0的一阶导数
	vmc->dd_L0 = (vmc->d_L0 - vmc->last_d_L0) / dt; // 腿长L0的二阶导数

	vmc->last_d_L0 = vmc->d_L0;
	vmc->last_L0 = vmc->L0;

	vmc->dd_theta = (vmc->d_theta - vmc->last_d_theta) / dt;
	vmc->last_d_theta = vmc->d_theta;
}

void VMC_calc_2(vmc_leg_t *vmc) // 计算期望的关节输出力矩
{
	// 计算雅可比矩阵系数见 https://zhuanlan.zhihu.com/p/613007726 公式(2.13)下面的最终结果
	vmc->j11 = (vmc->l1 * arm_sin_f32(vmc->phi0 - vmc->phi3) * arm_sin_f32(vmc->phi1 - vmc->phi2)) / arm_sin_f32(vmc->phi3 - vmc->phi2);
	vmc->j12 = (vmc->l1 * arm_cos_f32(vmc->phi0 - vmc->phi3) * arm_sin_f32(vmc->phi1 - vmc->phi2)) / (vmc->L0 * arm_sin_f32(vmc->phi3 - vmc->phi2));
	vmc->j21 = (vmc->l4 * arm_sin_f32(vmc->phi0 - vmc->phi2) * arm_sin_f32(vmc->phi3 - vmc->phi4)) / arm_sin_f32(vmc->phi3 - vmc->phi2);
	vmc->j22 = (vmc->l4 * arm_cos_f32(vmc->phi0 - vmc->phi2) * arm_sin_f32(vmc->phi3 - vmc->phi4)) / (vmc->L0 * arm_sin_f32(vmc->phi3 - vmc->phi2));

	vmc->torque_set[0] = vmc->j11 * vmc->F0 + vmc->j12 * vmc->Tp; // 得到Front电机的输出轴期望力矩，F0为五连杆机构末端沿腿的推力
	vmc->torque_set[1] = vmc->j21 * vmc->F0 + vmc->j22 * vmc->Tp; // 得到Back电机的输出轴期望力矩，Tp为沿中心轴的力矩
}

uint8_t ground_detection(vmc_leg_t *vmc)
{

	vmc->FN = vmc->F0 * arm_cos_f32(vmc->theta) + vmc->Tp * arm_sin_f32(vmc->theta) / vmc->L0 + 0.6f * (INS.MotionAccel_n[2] - vmc->dd_L0 * arm_cos_f32(vmc->theta) + 2.0f * vmc->d_L0 * vmc->d_theta * arm_sin_f32(vmc->theta) + vmc->L0 * vmc->dd_theta * arm_sin_f32(vmc->theta) + vmc->L0 * vmc->d_theta * vmc->d_theta * arm_cos_f32(vmc->theta));

	vmc->aver[0] = vmc->aver[1];
	vmc->aver[1] = vmc->aver[2];
	vmc->aver[2] = vmc->aver[3];
	vmc->aver[3] = vmc->FN;

	vmc->aver_fn = 0.25f * vmc->aver[0] + 0.25f * vmc->aver[1] + 0.25f * vmc->aver[2] + 0.25f * vmc->aver[3]; // 对支持力进行均值滤波

	if (vmc->aver_fn < TAKE_OFF_FN_THRESHOLD)
	{ // 离地了

		return 1;
	}
	else
	{
		return 0;
	}
}

/**
 * @brief 通过L0和Phi0的值计算关节phi1和phi4
 * @param[in]  phi0
 * @param[in]  l0
 * @param[out] phi1_phi4 phi1和phi4
 * @note 用于位置控制时求逆解
 */
void CalcPhi1AndPhi4(float phi0, float l0, float phi1_phi4[2])
{
	// float L5_2_pow;
	// float Lca2, Lce2;
	// float cos_phi11, cos_phi12, cos_phi41, cos_phi42;
	// float phi11, phi12, phi41, phi42;
	// float phi1, phi4;

	// L5_2_pow = (LEG_L5 / 2) * (LEG_L5 / 2); //(LEG_L5 / 2)^2
	// Lca2 = l0 * l0 + L5_2_pow + l0 * LEG_L5 * cos(phi0);
	// Lce2 = l0 * l0 + L5_2_pow - l0 * LEG_L5 * cos(phi0);

	// cos_phi11 = (L5_2_pow + Lca2 - l0 * l0) / (LEG_L5 * sqrt(Lca2));
	// cos_phi12 = (LEG_L1 * LEG_L1 + Lca2 - LEG_L2 * LEG_L2) / (2 * LEG_L1 * sqrt(Lca2));
	// cos_phi41 = (L5_2_pow + Lce2 - l0 * l0) / (LEG_L5 * sqrt(Lce2));
	// cos_phi42 = (LEG_L4 * LEG_L4 + Lce2 - LEG_L3 * LEG_L3) / (2 * LEG_L5 * sqrt(Lce2));

	// phi11 = acos(cos_phi11);
	// phi12 = acos(cos_phi12);

	// phi41 = acos(cos_phi41);
	// phi42 = acos(cos_phi42);

	// phi1 = phi11 + phi12;
	// phi4 = M_PI - (phi41 + phi42);

	// phi1_phi4[0] = phi1;
	// phi1_phi4[1] = phi4;

	// 下述是LEG_L5为0时的简化版本
	float cos_beta1, cos_beta2;
	float beta1, beta2;

	cos_beta1 = (LEG_L1 * LEG_L1 + l0 * l0 - LEG_L2 * LEG_L2) / (2 * LEG_L1 * l0);
	cos_beta2 = (LEG_L4 * LEG_L4 + l0 * l0 - LEG_L3 * LEG_L3) / (2 * LEG_L4 * l0);
	beta1 = acos(cos_beta1);
	beta2 = acos(cos_beta2);
	phi1_phi4[0] = phi0 + beta1; // phi1
	phi1_phi4[1] = phi0 - beta2; // phi4
}

// 三次多项式拟合系数
float Poly_Coefficient[12][4] = {
	{-157.0203f, 179.8485f, -85.4172f, 0.0537f},
	{-1.0899f, 3.0072f, -6.6735f, 0.1905f},
	{-27.7037f, 27.7682f, -9.7108f, -0.2497f},
	{-32.3706f, 32.7508f, -12.0579f, -0.3533f},
	{-38.5250f, 54.5544f, -30.4327f, 8.2300f},
	{-5.2760f, 8.9537f, -5.7839f, 1.8671f},
	{152.7053f, -130.2649f, 28.7526f, 5.6772f},
	{16.2673f, -16.3037f, 5.2455f, 0.1397f},
	{-21.5602f, 27.7447f, -13.9335f, 3.1997f},
	{-24.4104f, 31.6150f, -16.0527f, 3.7713f},
	{224.1583f, -227.6099f, 81.5551f, 0.1116f},
	{50.1510f, -51.5745f, 18.8992f, -0.3485f},
};

/**
 * @brief 获取K矩阵
 * @param[in]  len 腿长
 * @param[in] k K矩阵
 */
// https://zhuanlan.zhihu.com/p/563048952 1.2.1
void LQR_K_calc(float len, float k[12])
{
	float t1 = len;
	float t2 = len * len;
	float t3 = len * len * len;
	for (int i = 0; i < 12; i++)
	{
		k[i] = Poly_Coefficient[i][0] * t3 + Poly_Coefficient[i][1] * t2 + Poly_Coefficient[i][2] * t1 + Poly_Coefficient[i][3];
	}
}

void CalcLQR(float k[12], float x[6], float T_Tp[2])
{
	T_Tp[0] = k[0] * x[0] + k[1] * x[1] + k[2] * x[2] + k[3] * x[3] + k[4] * x[4] +
			  k[5] * x[5];
	T_Tp[1] = k[6] * x[0] + k[7] * x[1] + k[8] * x[2] + k[9] * x[3] + k[10] * x[4] +
			  k[11] * x[5];
}