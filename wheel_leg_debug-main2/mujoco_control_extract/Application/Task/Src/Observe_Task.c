/**
  *********************************************************************
  * @file      observe_task.c/h
  * @brief     观测器任务：使用卡尔曼滤波器对底盘运动速度进行估计，并积分求位移
  * @note
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */

#include "Observe_Task.h"
#include "Kalman_Filter.h"
#include "cmsis_os.h"

KalmanFilter_t vaEstimateKF; // 速度和加速度的卡尔曼滤波器结构体 (v: velocity, a: acceleration)

float vaEstimateKF_F[4] = {1.0f, 0.003f,
						   0.0f, 1.0f}; // 状态转移矩阵 F，采样时间(dt)设置为0.003s (3ms)

float vaEstimateKF_P[4] = {1.0f, 0.0f,
						   0.0f, 1.0f}; // 估计误差协方差矩阵 P 的初始值

float vaEstimateKF_Q[4] = {1.0f, 0.0f,
						   0.0f, 1.0f}; // 过程噪声协方差矩阵 Q 的初始值

float vaEstimateKF_R[4] = {200.0f, 0.0f,
						   0.0f, 200.0f}; // 测量噪声协方差矩阵 R 的初始值

float vaEstimateKF_K[4]; // 卡尔曼增益 K

const float vaEstimateKF_H[4] = {1.0f, 0.0f,
								 0.0f, 1.0f}; // 观测矩阵 H (单位矩阵，因为直接观测速度和加速度)

float vel_acc[2];
uint32_t OBSERVE_TIME = 3; // 观测任务运行周期 3ms, 与底盘控制任务的运行周期相同，保证每次控制更新时都能得到最新的速度估计值

void Observe_task(void)
{
	while (INS.ins_flag == 0)
	{ // 等待惯导(INS)初始化收敛完成
		osDelay(1);
	}
	static float wr, wl = 0.0f;
	static float vrb, vlb = 0.0f;
	static float aver_v = 0.0f;

	xvEstimateKF_Init(&vaEstimateKF);

	while (1)
	{
		// 参考https://zhuanlan.zhihu.com/p/689921165
		wr = -chassis_move.wheel_motor[0]->Data.Velocity - INS.Gyro[1] + right.d_alpha;										   // 右轮相对地面的转动角速度（由电机速度、机体俯仰角速度、腿部关节角速度合成，这里规定顺时针为正）
		vrb = wr * WHEEL_RADIUS + right.L0 * right.d_theta * arm_cos_f32(right.theta) + right.d_L0 * arm_sin_f32(right.theta); // 运动学正解：计算右侧机体在 b系(机体坐标系) 下的线速度

		wl = -chassis_move.wheel_motor[1]->Data.Velocity + INS.Gyro[1] + left.d_alpha;									  // 左轮相对地面的转动角速度（由电机速度、机体俯仰角速度、腿部关节角速度合成，这里规定顺时针为正）
		vlb = wl * WHEEL_RADIUS + left.L0 * left.d_theta * arm_cos_f32(left.theta) + left.d_L0 * arm_sin_f32(left.theta); // 运动学正解：计算左侧机体在 b系(机体坐标系) 下的线速度

		aver_v = (vrb - vlb) / 2.0f; // 取左右两侧速度的平均值，得到底盘中心的前进速度
		xvEstimateKF_Update(&vaEstimateKF, -INS.MotionAccel_b[0], aver_v);

		// 原地旋转的时候由于左右轮速度反向，算出来的平均速度 v_filter 和位移 x_filter 应当都近似为0
		chassis_move.v_filter = vel_acc[0];																		 // 得到卡尔曼滤波后的速度															 // 获取经过卡尔曼滤波后的底盘最优速度估计值
		chassis_move.x_filter = chassis_move.x_filter + chassis_move.v_filter * ((float)OBSERVE_TIME / 1000.0f); // 对速度进行积分，得到底盘位移

		// 如果想直接用轮子速度，不做融合的话可以这样
		//	chassis_move.v_filter=(chassis_move.wheel_motor[0]->Data.Velocity-chassis_move.wheel_motor[1]->Data.Velocity)*(-0.0603f)/2.0f;//0.0603是轮子半径，电机反馈的是角速度，乘半径后得到线速度，数学模型中定义的是轮子顺时针为正，所以要乘个负号
		//	chassis_move.x_filter=chassis_move.x_filter+chassis_move.v_filter*((float)OBSERVE_TIME/1000.0f);

		osDelay(OBSERVE_TIME);
	}
}

void xvEstimateKF_Init(KalmanFilter_t *EstimateKF)
{
	Kalman_Filter_Init(EstimateKF, 2, 0, 2); // 状态量为2维(速度,加速度)，没有控制量(0维)，测量/观测量为2维(编码器算出的速度, IMU测出的加速度)

	memcpy(EstimateKF->F_data, vaEstimateKF_F, sizeof(vaEstimateKF_F));
	memcpy(EstimateKF->P_data, vaEstimateKF_P, sizeof(vaEstimateKF_P));
	memcpy(EstimateKF->Q_data, vaEstimateKF_Q, sizeof(vaEstimateKF_Q));
	memcpy(EstimateKF->R_data, vaEstimateKF_R, sizeof(vaEstimateKF_R));
	memcpy(EstimateKF->H_data, vaEstimateKF_H, sizeof(vaEstimateKF_H));
}

void xvEstimateKF_Update(KalmanFilter_t *EstimateKF, float acc, float vel)
{
	memcpy(EstimateKF->Q_data, vaEstimateKF_Q, sizeof(vaEstimateKF_Q));
	memcpy(EstimateKF->R_data, vaEstimateKF_R, sizeof(vaEstimateKF_R));

	// 更新卡尔曼滤波器的测量值向量
	EstimateKF->MeasuredVector[0] = vel; // 观测到的速度 (来自轮腿运动学正解)
	EstimateKF->MeasuredVector[1] = acc; // 观测到的加速度 (来自IMU)

	// 调用卡尔曼滤波器更新函数进行状态估计
	Kalman_Filter_Update(EstimateKF);

	// 提取滤波后的最优估计值
	for (uint8_t i = 0; i < 2; i++)
	{
		vel_acc[i] = EstimateKF->FilteredValue[i];
	}
}

float RAMP_float(float final, float now, float ramp)
{
	float buffer = 0;

	buffer = final - now;

	if (buffer > 0)
	{
		if (buffer > ramp)
		{
			now += ramp;
		}
		else
		{
			now += buffer;
		}
	}
	else
	{
		if (buffer < -ramp)
		{
			now += -ramp;
		}
		else
		{
			now += buffer;
		}
	}

	return now;
}