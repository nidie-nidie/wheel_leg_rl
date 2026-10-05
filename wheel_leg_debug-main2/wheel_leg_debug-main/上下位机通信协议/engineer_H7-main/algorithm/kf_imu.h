#ifndef KF_IMU_H
#define KF_IMU_H

typedef struct
{
    float accel[3];
    float gyro[3];//第三个是偏航角速度
    float temperature;
    double euler[3];
    double euler_rad[3]; // yaw-pitch-roll
    double accelLPF[3];  //低通滤波后的加速度计数据
    double yawTotal;     //从启动开始计算yaw轴总角度
    int yaw_ring_num;    //从启动开始计算总圈数
} IMU_DATA;

void kf_imu_init(void);
void kf_imu_upgrade(void);
void euler2quaternion(double euler[3], double q[4]);
void quaternion2euler(double q[4], double euler[3]);

#endif // KF_IMU_H
