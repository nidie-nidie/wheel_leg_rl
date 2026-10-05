#include "M3508motors.h"
#include "string.h"
//定义3508电机转子的转矩常数 (0.3*187/3591) N*m/A 原装减速箱减速比为3591/187
const double M3508_ROTOR_TORQUE_CONSTANT = 0.3 * 187 / 3591;

void M3508_fbkdata(M3508motor_t *m3508_motor, unsigned char data[])//看来已经转化过了
{
    m3508_motor->para.pos_fbk = ((((short)data[0]) << 8) + data[1]) * (360.0 / 8191) * (3.1415926 / 180) - 3.1415926;                //->degree->rad
    m3508_motor->para.vel_fbk = (double)((short)((((short)data[2]) << 8) + data[3])) * (2 * 3.1415926) / 60;                         // rpm->rad/s
    m3508_motor->para.tor_fbk = (double)((short)((((short)data[4]) << 8) + data[5])) * (20.0 / 16384) * M3508_ROTOR_TORQUE_CONSTANT; //->A->N*m
    m3508_motor->para.temperature = data[6];
    if (m3508_motor->not_first == 0) //非非第一次，即第一次受到数据
    {
        m3508_motor->init_angle = m3508_motor->para.pos_fbk;
        m3508_motor->last_pos = m3508_motor->para.pos_fbk;
        m3508_motor->total_angle = 0;
        m3508_motor->ring_num = 0;
        m3508_motor->not_first = 1;
    }
    if (m3508_motor->para.pos_fbk - m3508_motor->last_pos < -3.1415926 * 1.2)
        m3508_motor->ring_num += 1;
    if (m3508_motor->para.pos_fbk - m3508_motor->last_pos > 3.1415926 * 1.2)
        m3508_motor->ring_num -= 1;
    m3508_motor->total_angle = m3508_motor->ring_num * 3.1415926 * 2 + m3508_motor->para.pos_fbk - m3508_motor->init_angle;
    m3508_motor->last_pos = m3508_motor->para.pos_fbk;
}

void M3508_torSet2cmd(M3508motor_t *m3508_motor, double torque)
{
    m3508_motor->cmd.tor_set = torque;
    double current_cmd = torque / M3508_ROTOR_TORQUE_CONSTANT;
    if (current_cmd > 20)
        current_cmd = 20;
    if (current_cmd < -20)
        current_cmd = -20;
    int temp_cmd = (current_cmd / 20) * 16384;
    if (temp_cmd > 16384)
        temp_cmd = 16384;
    if (temp_cmd < -16384)
        temp_cmd = -16384;
    m3508_motor->cmd.cmdSignal = temp_cmd;
}

//这里的id就是电调灯闪次数，结构体里的id是电机接收报文的id
void M3508_init(M3508motor_t *m3508_motor, int id, double gear_ratio)
{
    memset((void *)m3508_motor, 0, sizeof(M3508motor_t));
    m3508_motor->id = id + 0x200;
    m3508_motor->gear_ratio = gear_ratio;
    m3508_motor->power_limit = 140;
}




