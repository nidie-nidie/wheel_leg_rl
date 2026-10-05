/**
 * @Author: Li Zhenxing
 * @Date: 2024/11/27 17:18:37
 * @LastEditors: Li Zhenxing
 * @LastEditTime: 2024/11/27 17:18:37
 * Description: 卡尔曼滤波,6轴imu数据融合,计算数据类型为double(最好在有双精度浮点数加速计算平台上运行)
 * Copyright: Copyright (©) 2024 Li Zhenxing. All rights reserved.
 *
 */
// ekf融合imu数据
//规定欧拉角范围(-pi,pi]
#include "tim.h"
#include "gpio.h"
#include "bsp_dwt.h"
#include "kf_imu.h"
#include "BMI088driver.h"
#include "math.h"
#include "string.h"
#include "kalman_filter.h"

#define gravity 9.78984
#define imu_PI 3.1415926
#define BAIS_stable 0.4

static unsigned char init_allow = 0;
static unsigned char allow_exti_flag = 0;
static unsigned int dwt_count = 0; //单位秒s
IMU_DATA imu_data;
//算法相关数据
static double dt = 0;
static float wz_bais = 0.00028;
static float wy_bais = 0.002 - 0.0059;
static float wx_bais = 0.002 - 0.0031;
static double wz = 0.0;
static double wy = 0.0;
static double wx = 0.0;

static double euler_vector[2 + 3] = {3, 1, 0, 0, 0};
static double last_euler_vector[2 + 3] = {3, 1, 0, 0, 0};
static double mea_euler_vector[2 + 3] = {3, 1, 0, 0, 0};
static double minus_euler_vector[2 + 3] = {3, 1, 0, 0, 0};
static double P[2 + 9] = {3, 3,
                          0, 0, 0,
                          0, 0, 0,
                          0, 0, 0};
static double P_minus[2 + 9] = {3, 3,
                                0, 0, 0,
                                0, 0, 0,
                                0, 0, 0};
static double last_P[2 + 9] = {3, 3,
                               0, 0, 0,
                               0, 0, 0,
                               0, 0, 0};
static double Q[2 + 9] = {3, 3,
                          1, 0, 0,
                          0, 1, 0,
                          0, 0, 1};
static double R[2 + 9] = {3, 3,
                          1000, 0, 0,
                          0, 1000, 0,
                          0, 0, 1000};
static double H[2 + 9] = {3, 3,
                          1, 0, 0,
                          0, 1, 0,
                          0, 0, 1};
static double K[2 + 9] = {3, 3,
                          0, 0, 0,
                          0, 0, 0,
                          0, 0, 0};
static double accel[3] = {0};
static double last_accel[3] = {0};
static double last_yaw = 0;
double a_lpfk = 20; //加速度计低通滤波系数
static void imu_process(void);
static void accel2euler(double a[3], double euler[3]);
static void get_x_minus(double last_euler_vector[], double dt, double wz, double wy, double wx, double x_minus[]);

void kf_imu_init()
{
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_4);
    while (BMI088_init())
    {
        ;
    }
    allow_exti_flag = 1;
    while (1)
    {
        if (init_allow == 1)
            break;
        DWT_Delay(0.001);
    }
    dt = DWT_GetDeltaT(&dwt_count);
    imu_process();
    double euler[3] = {0};
    accel[0] = (double)imu_data.accel[0];
    accel[1] = (double)imu_data.accel[1];
    accel[2] = (double)imu_data.accel[2];
    accel2euler(accel, euler);
    //初始化加速度计估计初始pitch和roll
    last_euler_vector[2] = euler[0];
    last_euler_vector[3] = euler[1];
    last_euler_vector[4] = euler[2];
    last_accel[0] = accel[0];
    last_accel[1] = accel[1];
    last_accel[2] = accel[2];
}

void kf_imu_upgrade(void)
{
    imu_process();
    dt = DWT_GetDeltaT(&dwt_count);
    imu_data.gyro[0] = imu_data.gyro[0] - wx_bais;
    imu_data.gyro[1] = imu_data.gyro[1] - wy_bais;
    imu_data.gyro[2] = imu_data.gyro[2] - wz_bais;
    wx = (double)imu_data.gyro[0];
    wy = (double)imu_data.gyro[1];
    wz = (double)imu_data.gyro[2];
    double euler[3] = {0};
    accel[0] = (double)imu_data.accel[0];
    accel[1] = (double)imu_data.accel[1];
    accel[2] = (double)imu_data.accel[2];
    accel[0] = last_accel[0] * (1 - a_lpfk * dt) + accel[0] * a_lpfk * dt;
    accel[1] = last_accel[1] * (1 - a_lpfk * dt) + accel[1] * a_lpfk * dt;
    accel[2] = last_accel[2] * (1 - a_lpfk * dt) + accel[2] * a_lpfk * dt;
    imu_data.accelLPF[0] = accel[0];
    imu_data.accelLPF[1] = accel[1];
    imu_data.accelLPF[2] = accel[2];
    accel2euler(accel, euler);
    mea_euler_vector[2] = euler[0];
    mea_euler_vector[3] = euler[1];
    mea_euler_vector[4] = euler[2];
    last_accel[0] = accel[0];
    last_accel[1] = accel[1];
    last_accel[2] = accel[2];
    get_x_minus(last_euler_vector, dt, wz, wy, wx, minus_euler_vector);
    kf_get_P_minus(NULL, last_P, Q, P_minus);
    if (fabs(sqrt(wx * wx + wy * wy)) > BAIS_stable || fabs(sqrt(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]) - gravity) > BAIS_stable)
    {
        // imu运动，加速度计不修正
        euler_vector[2] = minus_euler_vector[2];
        euler_vector[3] = minus_euler_vector[3];
        euler_vector[4] = minus_euler_vector[4];
        kf_copy(last_euler_vector, last_P, euler_vector, P_minus);
    }
    else
    {
        kf_get_K(P_minus, H, R, K);
        kf_get_x_hat(minus_euler_vector, K, mea_euler_vector, H, euler_vector);
        euler_vector[2] = minus_euler_vector[2]; //偏航角不修正
        kf_get_P(K, H, P_minus, P);
        kf_copy(last_euler_vector, last_P, euler_vector, P);
    }
    imu_data.euler[0] = euler_vector[2 + 0] * 57.295779513;
    imu_data.euler[1] = euler_vector[2 + 1] * 57.295779513;
    imu_data.euler[2] = euler_vector[2 + 2] * 57.295779513;
    imu_data.euler_rad[0] = euler_vector[2 + 0];
    imu_data.euler_rad[1] = euler_vector[2 + 1];
    imu_data.euler_rad[2] = euler_vector[2 + 2];
    if (imu_data.euler_rad[0] - last_yaw < -3)
    {
        imu_data.yaw_ring_num++;
    }
    else if (imu_data.euler_rad[0] - last_yaw > 3)
    {
        imu_data.yaw_ring_num--;
    }
    imu_data.yawTotal = imu_data.euler_rad[0] + imu_data.yaw_ring_num * 2 * 3.1415926;
    last_yaw = imu_data.euler_rad[0];
}

//加速度计估计pitch和roll，欧拉角变换顺序z-y-x,加速度a1-3 ax ay az
static void accel2euler(double a[3], double euler[3])
{
    double r = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    euler[1] = asin(-a[0] / r);
    euler[2] = atan2(a[1], a[2]);
}
static void get_x_minus(double last_euler_vector[], double dt, double wz, double wy, double wx, double x_minus[])
{
    double temp_euler[3] = {0};
    double temp_q[4] = {0};
    double temp_q_v[2 + 4] = {4, 1, 0, 0, 0, 0};
    temp_euler[0] = last_euler_vector[0 + 2];
    temp_euler[1] = last_euler_vector[1 + 2];
    temp_euler[2] = last_euler_vector[2 + 2];
    euler2quaternion(temp_euler, temp_q);
    temp_q_v[2] = temp_q[0];
    temp_q_v[3] = temp_q[1];
    temp_q_v[4] = temp_q[2];
    temp_q_v[5] = temp_q[3];
    double state_transmit[2 + 4 * 4] = {4, 4,
                                        1, -0.5 * wx * dt, -0.5 * wy * dt, -0.5 * wz * dt,
                                        0.5 * wx * dt, 1, 0.5 * wz * dt, -0.5 * wy * dt,
                                        0.5 * wy * dt, -0.5 * wz * dt, 1, 0.5 * wx * dt,
                                        0.5 * wz * dt, 0.5 * wy * dt, -0.5 * wx * dt, 1};
    user_mat_multiply(state_transmit, temp_q_v, temp_q_v);
    //四元数迭代后做归一化处理
    double q_norm = sqrt(temp_q_v[0 + 2] * temp_q_v[0 + 2] + temp_q_v[1 + 2] * temp_q_v[1 + 2] + temp_q_v[2 + 2] * temp_q_v[2 + 2] + temp_q_v[3 + 2] * temp_q_v[3 + 2]);
    temp_q[0] = temp_q_v[2] / q_norm;
    temp_q[1] = temp_q_v[3] / q_norm;
    temp_q[2] = temp_q_v[4] / q_norm;
    temp_q[3] = temp_q_v[5] / q_norm;
    quaternion2euler(temp_q, temp_euler);
    x_minus[2] = temp_euler[0];
    x_minus[3] = temp_euler[1];
    x_minus[4] = temp_euler[2];
}
//欧拉角变换顺序为z-y-x
void quaternion2euler(double q[4], double euler[3])
{
    /*
    Yaw = atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 2.0 * (q[0] * q[0] + q[1] * q[1]) - 1.0) * 57.295779513;
    Pitch = atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), 2.0 * (q[0] * q[0] + q[3] * q[3]) - 1.0) * 57.295779513;
    Roll = asin(-2.0 * (q[1] * q[3] - q[0] * q[2])) * 57.295779513;
    */
    euler[0] = atan2(2 * (q[0] * q[3] + q[1] * q[2]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]);
    if ((2 * (q[0] * q[2] - q[3] * q[1]) > 1) || (2 * (q[0] * q[2] - q[3] * q[1]) < -1))
    {
        return;
    }
    euler[1] = asin(2 * (q[0] * q[2] - q[3] * q[1]));
    euler[2] = atan2(2 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]);
}
//欧拉角转四元数 欧拉角变换顺序为z-y-x
void euler2quaternion(double euler[3], double q[4])
{
    double c1 = cos(euler[0] / 2);
    double s1 = sin(euler[0] / 2);
    double c2 = cos(euler[1] / 2);
    double s2 = sin(euler[1] / 2);
    double c3 = cos(euler[2] / 2);
    double s3 = sin(euler[2] / 2);
    q[0] = c1 * c2 * c3 + s1 * s2 * s3;
    q[1] = c1 * c2 * s3 - s1 * s2 * c3;
    q[2] = c1 * s2 * c3 + s1 * c2 * s3;
    q[3] = s1 * c2 * c3 - c1 * s2 * s3;
}

#define DES_TEMP 40.0f
#define KP 300.f
#define KI 0.3f
#define KD 10.f
#define MAX_OUT 9990
#define Err_I_MAX (MAX_OUT / 2 / KI)
float out = 0;
float err = 0;
float err_l = 0;
float err_ll = 0;
float err_i = 0;
uint8_t forceStop = 0;
void imu_process(void)
{
    BMI088_read(imu_data.gyro, imu_data.accel, &(imu_data.temperature));
    err_ll = err_l;
    err_l = err;
    err = DES_TEMP - imu_data.temperature;
    err_i += err;
    //积分限幅
    if (err_i > Err_I_MAX)
    {
        err_i = Err_I_MAX;
    }
    if (err_i < -Err_I_MAX)
    {
        err_i = -Err_I_MAX;
    }
    out = KP * err + KI * err_i + KD * (err - err_l);
    if (out > MAX_OUT)
        out = MAX_OUT;
    if (out < 0)
        out = 0.f;

    if (forceStop == 1)
    {
        out = 0.0f;
    }
    htim3.Instance->CCR4 = (uint16_t)out;
}
//角速度计中断 ！一定要等rtos初始化之后才允许进入中断！
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if (allow_exti_flag == 1)
    {
        if (GPIO_Pin == ACC_INT_Pin)
        {
            init_allow = 1;
        }
        else if (GPIO_Pin == GYRO_INT_Pin)
        {
        }
    }
}
