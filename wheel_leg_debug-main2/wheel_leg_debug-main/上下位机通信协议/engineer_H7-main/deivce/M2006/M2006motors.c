#include "M2006motors.h"
#include "string.h"
//定义2006电机转子的转矩常数 (0.18*1/36) N*m/A
const double M2006_ROTOR_TORQUE_CONSTANT = 0.18 * 1 / 36;

	void M2006_fbkdata(M2006motor_t *M2006_motor, unsigned char data[])
{
    M2006_motor->para.pos_fbk = ((((short)data[0]) << 8) + data[1]) * (360.0 / 8191) * (3.1415926 / 180) - 3.1415926;                //->degree->rad
    M2006_motor->para.vel_fbk = (double)((short)((((short)data[2]) << 8) + data[3])) * (2 * 3.1415926) / 60;                         // rpm->rad/s
    M2006_motor->para.tor_fbk = (double)((short)((((short)data[4]) << 8) + data[5])) * (20.0 / 16384) * M2006_ROTOR_TORQUE_CONSTANT; //->A->N*m

    if (M2006_motor->not_first == 0) //非非第一次，即第一次受到数据
    {
        M2006_motor->init_angle = M2006_motor->para.pos_fbk;
        M2006_motor->last_pos = M2006_motor->para.pos_fbk;
        M2006_motor->total_angle = 0;
        M2006_motor->ring_num = 0;
        M2006_motor->not_first = 1;
    }
    if (M2006_motor->para.pos_fbk - M2006_motor->last_pos < -3.1415926 * 1.2)
        M2006_motor->ring_num += 1;
    if (M2006_motor->para.pos_fbk - M2006_motor->last_pos > 3.1415926 * 1.2)
        M2006_motor->ring_num -= 1;
    M2006_motor->total_angle = M2006_motor->ring_num * 3.1415926 * 2 + M2006_motor->para.pos_fbk - M2006_motor->init_angle;
    M2006_motor->last_pos = M2006_motor->para.pos_fbk;
}
// torque 转子扭矩
void M2006_torSet2cmd(M2006motor_t *M2006_motor, double torque)
{
    M2006_motor->cmd.tor_set = torque;
    double current_cmd = torque / M2006_ROTOR_TORQUE_CONSTANT;
    M2006_motor->cmd.cmdSignal = (current_cmd / 20) * 16384;
}
// id 灯闪次数
void M2006_init(M2006motor_t *M2006_motor, int id, double gear_ratio)
{
    memset((void *)M2006_motor, 0, sizeof(M2006motor_t));
    M2006_motor->id = id + 0x200;
    M2006_motor->gear_ratio = gear_ratio;
}
