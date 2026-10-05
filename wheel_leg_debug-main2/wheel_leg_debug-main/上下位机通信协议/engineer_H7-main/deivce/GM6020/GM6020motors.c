#include "GM6020motors.h"
#include "string.h"
//转矩常数 741 mN·m/A
const double GM6020_ROTOR_TORQUE_CONSTANT = 0.741;

void GM6020_fbkdata(GM6020motor_t *GM6020_motor, unsigned char data[])
{
    double temp_pos = ((((short)data[0]) << 8) + data[1]) * (360.0 / 8191) * (3.1415926 / 180) - 3.1415926; //->degree->rad
    GM6020_motor->para.pos_fbk = temp_pos - GM6020_motor->zero_pos;
    if (GM6020_motor->para.pos_fbk > 3.1415926)
        GM6020_motor->para.pos_fbk -= 2 * 3.1415926;
    if (GM6020_motor->para.pos_fbk < -3.1415926)
        GM6020_motor->para.pos_fbk += 2 * 3.1415926;
    GM6020_motor->para.vel_fbk = (double)((short)((((short)data[2]) << 8) + data[3])) * (2 * 3.1415926) / 60;                          // rpm->rad/s
    GM6020_motor->para.tor_fbk = (double)((short)((((short)data[4]) << 8) + data[5])) * (20.0 / 16384) * GM6020_ROTOR_TORQUE_CONSTANT; //->A->N*m
    GM6020_motor->para.temperature = data[6];
    if (GM6020_motor->not_first == 0) //非非第一次，即第一次受到数据
    {
        //对于GM6020来说手动设置初始角度用于标定零点
        GM6020_motor->last_pos = GM6020_motor->para.pos_fbk;
        GM6020_motor->total_angle = GM6020_motor->para.pos_fbk;
        GM6020_motor->ring_num = 0;
        GM6020_motor->not_first = 1;
    }
    if (GM6020_motor->para.pos_fbk - GM6020_motor->last_pos < -3.1415926 * 1.2)
        GM6020_motor->ring_num += 1;
    if (GM6020_motor->para.pos_fbk - GM6020_motor->last_pos > 3.1415926 * 1.2)
        GM6020_motor->ring_num -= 1;
    GM6020_motor->total_angle = GM6020_motor->ring_num * 3.1415926 * 2 + GM6020_motor->para.pos_fbk;
    GM6020_motor->last_pos = GM6020_motor->para.pos_fbk;
}
// zero_pos 用户希望的电机0位置对应的电机返回的角度
// id 灯闪次数即id
void GM6020_init(GM6020motor_t *GM6020_motor, int id, double zero_pos)
{
    memset((void *)GM6020_motor, 0, sizeof(GM6020motor_t));
    GM6020_motor->id = 0x204 + id;
    GM6020_motor->zero_pos = zero_pos;
}
