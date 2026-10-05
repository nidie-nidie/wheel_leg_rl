#ifndef M2006MOTORS_H
#define M2006MOTORS_H

typedef struct
{
    unsigned short id; //这里是接收报文id
    double gear_ratio; //速比，减速比为大于1的值，加速比为小于1的值
    struct
    {
        double tor_set;  //扭矩设置（转子扭矩）
        short cmdSignal; //实际发送报文的数据
    } cmd;
    struct
    {
        // M2006电机无温度反馈
        //  unsigned char temperature; // degree
        double tor_fbk; // N*m
        double vel_fbk; // rad/s
        double pos_fbk; // rad -pi~pi
    } para;             //反馈报文的数据
    double last_pos;
    int ring_num;
    double total_angle; //启动时为0,记录总转角
    char not_first;     // 0为第一次收到数据
    double init_angle;
} M2006motor_t;

void M2006_fbkdata(M2006motor_t *m2006_motor, unsigned char data[]);
void M2006_torSet2cmd(M2006motor_t *m2006_motor, double torque);
void M2006_init(M2006motor_t *M2006_motor, int id, double gear_ratio);

#endif // M2006MOTORS_H
