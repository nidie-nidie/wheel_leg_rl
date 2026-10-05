#include "VMC_Calc.h"

#if MUJOCO_VMC_KINEMATICS_MODE == MUJOCO_VMC_MODEL_CAD_ONE_CLOSURE_ID
/*
 * HTML 4.3/5.5 推荐的 CAD 一次闭链模型。
 * 坐标原点为本侧共同髋轴 I，s=+base Y，z=+base Z；输入角直接使用
 * MuJoCo XML 的原始主动关节 q_front/q_rear。I-L-P 共线，P-W 与 L-M
 * 平行，因此只需求一次 J/L 两圆交点 M：
 *
 *   P = k_a L
 *   W = P + k_b (M - L)
 *   y = [L0, phi0]^T,  phi0 = atan2(-Wz, Ws)
 *
 * 这些量来自 wheel_leg_urdf4_self_mesh_all.xml 的名义参考几何；微小的
 * XML 导出不共线残差按 CAD 约束忽略。
 */
static uint8_t VMC_calc_cad_one_closure(vmc_leg_t *vmc)
{
    const float v_ij_s = 0.0967494f;
    const float v_ij_z = 0.00674587f;
    const float v_il_s = -0.0963553f;
    const float v_il_z = 0.0109404f;
    const float radius_jm = 0.115000002623f;
    const float radius_lm = 0.115000045454f;
    const float scale_a = 2.216813311434f; /* |IP| / |IL| */
    const float scale_b = 2.243476979061f; /* |PW| / |LM| */
    const float eps = 1.0e-7f;

    const float cf = cosf(vmc->q_front);
    const float sf = sinf(vmc->q_front);
    const float cr = cosf(vmc->q_rear);
    const float sr = sinf(vmc->q_rear);

    const float js = cf * v_ij_s - sf * v_ij_z;
    const float jz = sf * v_ij_s + cf * v_ij_z;
    const float ls = cr * v_il_s - sr * v_il_z;
    const float lz = sr * v_il_s + cr * v_il_z;

    const float center_s = ls - js;
    const float center_z = lz - jz;
    const float rho_sq = center_s * center_s + center_z * center_z;
    if (!(rho_sq > eps * eps))
    {
        return 0;
    }

    const float rho = sqrtf(rho_sq);
    if (rho > radius_jm + radius_lm + eps ||
        rho < fabsf(radius_jm - radius_lm) - eps)
    {
        return 0;
    }

    const float e_s = center_s / rho;
    const float e_z = center_z / rho;
    const float a = (radius_jm * radius_jm - radius_lm * radius_lm + rho_sq) /
                    (2.0f * rho);
    float h_sq = radius_jm * radius_jm - a * a;
    if (h_sq < -eps)
    {
        return 0;
    }
    if (h_sq < 0.0f)
    {
        h_sq = 0.0f;
    }

    /* 固定物理装配支路 sigma_M=+1，e_perp=[-e_z,+e_s]^T。 */
    const float h = sqrtf(h_sq);
    const float ms = js + a * e_s - h * e_z;
    const float mz = jz + a * e_z + h * e_s;

    const float ws = scale_b * ms + (scale_a - scale_b) * ls;
    const float wz = scale_b * mz + (scale_a - scale_b) * lz;
    const float l0_sq = ws * ws + wz * wz;
    if (!(l0_sq > eps * eps))
    {
        return 0;
    }
    const float l0 = sqrtf(l0_sq);

    /*
     * 隐式解析求导：M 同时满足 |M-J|=|JM|、|M-L|=|LM|。
     * D_J、D_L 的两列依次对应 q_front、q_rear。
     */
    const float dj_s[2] = {-jz, 0.0f};
    const float dj_z[2] = {js, 0.0f};
    const float dl_s[2] = {0.0f, -lz};
    const float dl_z[2] = {0.0f, ls};
    const float u_s = ms - js;
    const float u_z = mz - jz;
    const float v_s = ms - ls;
    const float v_z = mz - lz;
    const float closure_det = u_s * v_z - u_z * v_s;
    if (!(fabsf(closure_det) > eps))
    {
        return 0;
    }

    float dw_s[2];
    float dw_z[2];
    for (uint8_t column = 0; column < 2; ++column)
    {
        const float b_j = u_s * dj_s[column] + u_z * dj_z[column];
        const float b_l = v_s * dl_s[column] + v_z * dl_z[column];
        const float dm_s = (v_z * b_j - u_z * b_l) / closure_det;
        const float dm_z = (-v_s * b_j + u_s * b_l) / closure_det;

        dw_s[column] = scale_b * dm_s + (scale_a - scale_b) * dl_s[column];
        dw_z[column] = scale_b * dm_z + (scale_a - scale_b) * dl_z[column];
    }

    const float jh_l0_qfront = (ws * dw_s[0] + wz * dw_z[0]) / l0;
    const float jh_l0_qrear = (ws * dw_s[1] + wz * dw_z[1]) / l0;
    const float jh_phi0_qfront = (wz * dw_s[0] - ws * dw_z[0]) / l0_sq;
    const float jh_phi0_qrear = (wz * dw_s[1] - ws * dw_z[1]) / l0_sq;
    const float phi0 = atan2f(-wz, ws);

    if (!isfinite(l0) || !isfinite(phi0) ||
        !isfinite(jh_l0_qfront) || !isfinite(jh_l0_qrear) ||
        !isfinite(jh_phi0_qfront) || !isfinite(jh_phi0_qrear))
    {
        return 0;
    }

    vmc->wheel_s = ws;
    vmc->wheel_z = wz;
    vmc->XC = ws;
    vmc->YC = -wz;
    vmc->L0 = l0;
    vmc->phi0 = phi0;
    vmc->alpha = M_PI_2 - phi0;
    vmc->jh_l0_qfront = jh_l0_qfront;
    vmc->jh_l0_qrear = jh_l0_qrear;
    vmc->jh_phi0_qfront = jh_phi0_qfront;
    vmc->jh_phi0_qrear = jh_phi0_qrear;
    return 1;
}

static void VMC_clear_cad_kinematics_outputs(vmc_leg_t *vmc)
{
	vmc->jh_l0_qfront = 0.0f;
	vmc->jh_l0_qrear = 0.0f;
	vmc->jh_phi0_qfront = 0.0f;
	vmc->jh_phi0_qrear = 0.0f;
	vmc->raw_joint_torque[0] = 0.0f;
	vmc->raw_joint_torque[1] = 0.0f;
	vmc->j11 = 0.0f;
	vmc->j12 = 0.0f;
	vmc->j21 = 0.0f;
	vmc->j22 = 0.0f;
	vmc->torque_set[0] = 0.0f;
	vmc->torque_set[1] = 0.0f;
}
#endif

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
	vmc->kinematics_valid = 0;
	vmc->raw_joint_torque[0] = 0.0f;
	vmc->raw_joint_torque[1] = 0.0f;
}

void VMC_calc_1(vmc_leg_t *vmc, float Pitch, float PithGyro, float dt) // 计算theta和d_theta给lqr用，同时也计算腿长L0
{
#if MUJOCO_VMC_KINEMATICS_MODE == MUJOCO_VMC_MODEL_CAD_ONE_CLOSURE_ID
	const uint8_t was_valid = vmc->kinematics_valid;
	vmc->kinematics_valid = VMC_calc_cad_one_closure(vmc);
	if (!vmc->kinematics_valid)
	{
		VMC_clear_cad_kinematics_outputs(vmc);
		return;
	}
	const uint8_t reset_derivative_history = (vmc->first_flag == 0U || was_valid == 0U);
#else
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
	vmc->kinematics_valid = 1;
#endif

#if MUJOCO_VMC_KINEMATICS_MODE == MUJOCO_VMC_MODEL_CAD_ONE_CLOSURE_ID
	if (reset_derivative_history)
	{
		vmc->last_phi0 = vmc->phi0;
		vmc->last_L0 = vmc->L0;
		vmc->last_d_L0 = 0.0f;
		vmc->d_phi0 = 0.0f;
		vmc->d_L0 = 0.0f;
		vmc->dd_L0 = 0.0f;
		vmc->first_flag = 1;
	}
	else
	{
		const float delta_phi0 = atan2f(sinf(vmc->phi0 - vmc->last_phi0),
									 cosf(vmc->phi0 - vmc->last_phi0));
		vmc->d_phi0 = delta_phi0 / dt;
		vmc->d_L0 = (vmc->L0 - vmc->last_L0) / dt;
		vmc->dd_L0 = (vmc->d_L0 - vmc->last_d_L0) / dt;
	}
#else
	if (vmc->first_flag == 0)
	{
		vmc->last_phi0 = vmc->phi0;
		vmc->first_flag = 1;
	}

	vmc->d_phi0 = (vmc->phi0 - vmc->last_phi0) / dt; // 计算phi0变化率，d_phi0用于计算lqr需要的d_theta
#endif
	vmc->d_alpha = 0.0f - vmc->d_phi0;

	vmc->theta = M_PI_2 - Pitch - vmc->phi0;  // 得到状态变量1
	vmc->d_theta = (-PithGyro - vmc->d_phi0); // 得到状态变量2

	vmc->last_phi0 = vmc->phi0;

#if MUJOCO_VMC_KINEMATICS_MODE != MUJOCO_VMC_MODEL_CAD_ONE_CLOSURE_ID
	vmc->d_L0 = (vmc->L0 - vmc->last_L0) / dt;		// 腿长L0的一阶导数
	vmc->dd_L0 = (vmc->d_L0 - vmc->last_d_L0) / dt; // 腿长L0的二阶导数
#endif

	vmc->last_d_L0 = vmc->d_L0;
	vmc->last_L0 = vmc->L0;

#if MUJOCO_VMC_KINEMATICS_MODE == MUJOCO_VMC_MODEL_CAD_ONE_CLOSURE_ID
	if (reset_derivative_history)
	{
		vmc->last_d_theta = vmc->d_theta;
	}
#endif
	vmc->dd_theta = (vmc->d_theta - vmc->last_d_theta) / dt;
	vmc->last_d_theta = vmc->d_theta;
}

void VMC_calc_2(vmc_leg_t *vmc) // 计算期望的关节输出力矩
{
#if MUJOCO_VMC_KINEMATICS_MODE == MUJOCO_VMC_MODEL_CAD_ONE_CLOSURE_ID
	if (!vmc->kinematics_valid)
	{
		VMC_clear_cad_kinematics_outputs(vmc);
		return;
	}

	/* Tp 在此定义为与 phi0 共轭的虚拟腿力矩。 */
	vmc->raw_joint_torque[0] = vmc->jh_l0_qfront * vmc->F0 +
								   vmc->jh_phi0_qfront * vmc->Tp;
	vmc->raw_joint_torque[1] = vmc->jh_l0_qrear * vmc->F0 +
								   vmc->jh_phi0_qrear * vmc->Tp;

	/*
	 * 旧发送接口随后执行 [-phi4, -phi1]。这里显式构造兼容量，保证
	 * 最终 MuJoCo actuator 直接收到 [tau_qfront, tau_qrear]，不重复翻转。
	 */
	vmc->j11 = -vmc->jh_l0_qrear;
	vmc->j12 = -vmc->jh_phi0_qrear;
	vmc->j21 = -vmc->jh_l0_qfront;
	vmc->j22 = -vmc->jh_phi0_qfront;
	vmc->torque_set[0] = -vmc->raw_joint_torque[1];
	vmc->torque_set[1] = -vmc->raw_joint_torque[0];
#else
	// 计算雅可比矩阵系数见 https://zhuanlan.zhihu.com/p/613007726 公式(2.13)下面的最终结果
	vmc->j11 = (vmc->l1 * arm_sin_f32(vmc->phi0 - vmc->phi3) * arm_sin_f32(vmc->phi1 - vmc->phi2)) / arm_sin_f32(vmc->phi3 - vmc->phi2);
	vmc->j12 = (vmc->l1 * arm_cos_f32(vmc->phi0 - vmc->phi3) * arm_sin_f32(vmc->phi1 - vmc->phi2)) / (vmc->L0 * arm_sin_f32(vmc->phi3 - vmc->phi2));
	vmc->j21 = (vmc->l4 * arm_sin_f32(vmc->phi0 - vmc->phi2) * arm_sin_f32(vmc->phi3 - vmc->phi4)) / arm_sin_f32(vmc->phi3 - vmc->phi2);
	vmc->j22 = (vmc->l4 * arm_cos_f32(vmc->phi0 - vmc->phi2) * arm_sin_f32(vmc->phi3 - vmc->phi4)) / (vmc->L0 * arm_sin_f32(vmc->phi3 - vmc->phi2));

	vmc->torque_set[0] = vmc->j11 * vmc->F0 + vmc->j12 * vmc->Tp; // phi1分支/J1(J2)的期望力矩
	vmc->torque_set[1] = vmc->j21 * vmc->F0 + vmc->j22 * vmc->Tp; // phi4分支/J0(J3)的期望力矩
	vmc->raw_joint_torque[0] = -vmc->torque_set[1];
	vmc->raw_joint_torque[1] = -vmc->torque_set[0];
#endif
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
	{-182.3050f, 196.6642f, -89.5626f, 0.2684f},
	{-0.5441f, 2.8161f, -6.8860f, 0.1985f},
	{-33.5189f, 31.2920f, -10.3542f, -0.2141f},
	{-39.4331f, 37.1658f, -12.9313f, -0.3137f},
	{-39.1161f, 55.0711f, -30.5840f, 8.2785f},
	{-4.5533f, 8.5200f, -5.7114f, 1.8779f},
	{197.4371f, -157.9305f, 34.0006f, 5.4496f},
	{20.1883f, -18.9250f, 5.8190f, 0.1169f},
	{-23.3938f, 28.9784f, -14.1897f, 3.2199f},
	{-26.4950f, 33.1094f, -16.4262f, 3.8212f},
	{269.8723f, -255.4974f, 86.7485f, -0.2000f},
	{60.1130f, -57.7089f, 20.0702f, -0.4254f},
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
