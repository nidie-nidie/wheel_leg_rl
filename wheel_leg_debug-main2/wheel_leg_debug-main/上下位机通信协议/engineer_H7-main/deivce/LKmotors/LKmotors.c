#include "LKmotors.h"
#include "math.h"
#include "string.h"
#define DEVICE_STD_ID (0x140)
/*将电机从开启状态（上电后默认状态）切换到关闭状态，清除电机转动圈数及之前接收的控
制指令，LED 由常亮转为慢闪。此时电机仍然可以回复控制命令，但不会执行动作。*/
void turn_off_LKmotor(unsigned short id, FDCAN_HandleTypeDef *hcan)
{
    unsigned char data[8] = {0};
    data[0] = 0x80;
    fdcanx_send_data(hcan, DEVICE_STD_ID + id, data, 8);
}
/*将电机从关闭状态切换到开启状态，LED 由慢闪转为常亮。此时再发送控制指令即可控制电机动作。*/
void turn_on_LKmotor(unsigned short id, FDCAN_HandleTypeDef *hcan)
{
    unsigned char data[8] = {0};
    data[0] = 0x88;
    fdcanx_send_data(hcan, DEVICE_STD_ID + id, data, 8);
}
/*停止电机，但不清除电机运行状态。再次发送控制指令即可控制电机动作。*/
void stop_LKmotor(unsigned short id, FDCAN_HandleTypeDef *hcan)
{
    unsigned char data[8] = {0};
    data[0] = 0x81;
    fdcanx_send_data(hcan, DEVICE_STD_ID + id, data, 8);
}

void MF9025v2_setTor(unsigned short id, FDCAN_HandleTypeDef *hcan, double torque)
{
    // T=Ct*I MF9025v2转矩常数 0.32 N*m/A
    double current = torque / 0.32;
    double currentRange = 16.5;
    double proportion = current / currentRange;
    short iqControl = 2048 * proportion;
    unsigned char data[8] = {0};
    data[0] = 0xA1;
    data[4] = *(uint8_t *)(&iqControl);
    data[5] = *((uint8_t *)(&iqControl) + 1);
    fdcanx_send_data(hcan, DEVICE_STD_ID + id, data, 8);
}
// pos_fbk 0~2pi
void MF9025v2_fbkdata(MF9025v2_t *MF9025_motor, unsigned char data[])
{
    MF9025_motor->para.temperature = (char)data[1]; // get temperature
    // MF9025_motor->para.tor_fbk = (double)((data[2]) | data[3] << 8) * (16.5 / 2048.0) * 0.32;            //电流
    // MF9025_motor->para.vel_fbk = ((double)((data[4]) | data[5] << 8)) / 57.295779513;                    //速度 rad/s
    // MF9025_motor->para.pos_fbk = (double)((uint16_t)((data[7] << 8) + data[6])) * 3.1415926 * 2 / 65535; //位置 rad
    MF9025_motor->para.tor_fbk = (double)((int16_t)((data[2]) | data[3] << 8)) * (16.5 / 2048.0) * 0.32; //电流
    MF9025_motor->para.vel_fbk = (double)((int16_t)((data[4]) | data[5] << 8)) / 57.295779513;           //速度 rad/s
    MF9025_motor->para.pos_fbk = (double)((uint16_t)((data[7] << 8) + data[6])) * 3.1415926 * 2 / 65535; //位置 rad
    if (MF9025_motor->is_first == 0)
    {
        MF9025_motor->init_angle = MF9025_motor->para.pos_fbk;
        MF9025_motor->last_pos = MF9025_motor->para.pos_fbk;
        MF9025_motor->total_angle = 0;
        MF9025_motor->ring_num = 0;
        MF9025_motor->is_first = 1;
    }
    if (MF9025_motor->para.pos_fbk - MF9025_motor->last_pos < -3.1415926 * 1.2)
        MF9025_motor->ring_num += 1;
    if (MF9025_motor->para.pos_fbk - MF9025_motor->last_pos > 3.1415926 * 1.2)
        MF9025_motor->ring_num -= 1;
    MF9025_motor->total_angle = MF9025_motor->ring_num * 3.1415926 * 2 + MF9025_motor->para.pos_fbk - MF9025_motor->init_angle;
    MF9025_motor->last_pos = MF9025_motor->para.pos_fbk;
}

// id 1~32
// torque 减速后输出扭矩
void MG4010_setTor(unsigned short id, FDCAN_HandleTypeDef *hcan, double torque)
{
    // T=Ct*I MG4010E-i10v3转矩常数 0.07 N*m/A  减速比   10:1
    double current = torque / 0.07;
    double currentRange = 33;
    double proportion = current / currentRange;
    short iqControl = 2048 * proportion;
    unsigned char data[8] = {0};
    data[0] = 0xA1;
    data[4] = *(uint8_t *)(&iqControl);
    data[5] = *((uint8_t *)(&iqControl) + 1);
    fdcanx_send_data(hcan, DEVICE_STD_ID + id, data, 8);
}
// pos_fbk -pi~pi
void MG4010_fbkdata(MG4010_t *MG4010_motor, unsigned char data[])
{
    MG4010_motor->para.temperature = (char)data[1];                                                       // get temperature
    MG4010_motor->para.tor_fbk = (double)((int16_t)((data[2]) | data[3] << 8)) * (33 / 2048.0) * (0.07);  //电流
    MG4010_motor->para.vel_fbk = (double)((int16_t)((data[4]) | data[5] << 8)) / 57.295779513 / 10;       //速度 rad/s 不知道为什么需要多除以一个10才对
    double temp_pos = (double)((uint16_t)((data[7] << 8) + data[6])) * 3.1415926 * 2 / 65535 - 3.1415926; //位置 rad
    MG4010_motor->para.pos_fbk = temp_pos - MG4010_motor->zero_pos;
    if (MG4010_motor->para.pos_fbk > 3.1415926)
        MG4010_motor->para.pos_fbk -= 2 * 3.1415926;
    if (MG4010_motor->para.pos_fbk < -3.1415926)
        MG4010_motor->para.pos_fbk += 2 * 3.1415926;
    //通过最大加速度滤掉速度反馈数据的毛刺（高频噪声）
    double max_accel = 37;
    if (MG4010_motor->para.vel_fbk - MG4010_motor->last_para.vel_fbk > max_accel * 0.001) // 0.001默认数据回传值的时间间隔
        MG4010_motor->para.vel_fbk = MG4010_motor->last_para.vel_fbk + max_accel * 0.001;
    else if (MG4010_motor->para.vel_fbk - MG4010_motor->last_para.vel_fbk < -max_accel * 0.001) // 0.001默认数据回传值的时间间隔
        MG4010_motor->para.vel_fbk = MG4010_motor->last_para.vel_fbk - max_accel * 0.001;
    //滤掉角度反馈值的高频噪声
    double max_vel = 34;                                                                // 320rpm
    if (MG4010_motor->para.pos_fbk - MG4010_motor->last_para.pos_fbk > max_vel * 0.001) // 0.001默认数据回传值的时间间隔
        MG4010_motor->para.pos_fbk = MG4010_motor->last_para.pos_fbk + max_vel * 0.001;
    else if (MG4010_motor->para.pos_fbk - MG4010_motor->last_para.pos_fbk < -max_vel * 0.001) // 0.001默认数据回传值的时间间隔
        MG4010_motor->para.pos_fbk = MG4010_motor->last_para.pos_fbk - max_vel * 0.001;
    memcpy((void *)&(MG4010_motor->last_para), (const void *)&(MG4010_motor->para), sizeof(MG4010_motor->para));
    if (MG4010_motor->not_first == 0) //非非第一次，即第一次受到数据
    {
        //对于GM6020来说手动设置初始角度用于标定零点
        MG4010_motor->last_para.pos_fbk = MG4010_motor->para.pos_fbk;
        MG4010_motor->total_angle = MG4010_motor->para.pos_fbk;
        MG4010_motor->ring_num = 0;
        MG4010_motor->not_first = 1;
    }
    if (MG4010_motor->para.pos_fbk - MG4010_motor->last_para.pos_fbk < -3.1415926 * 1.2)
        MG4010_motor->ring_num += 1;
    if (MG4010_motor->para.pos_fbk - MG4010_motor->last_para.pos_fbk > 3.1415926 * 1.2)
        MG4010_motor->ring_num -= 1;
    MG4010_motor->total_angle = MG4010_motor->ring_num * 3.1415926 * 2 + MG4010_motor->para.pos_fbk;
    MG4010_motor->last_para.pos_fbk = MG4010_motor->para.pos_fbk;
}
// id 1~32
void MG4010_init(MG4010_t *MG4010_motor, char id, double zero_pos)
{
    memset((void *)MG4010_motor, 0, sizeof(MG4010_t));
    MG4010_motor->id = id;
    MG4010_motor->gear_ratio = 10;
    MG4010_motor->zero_pos = zero_pos;
}
