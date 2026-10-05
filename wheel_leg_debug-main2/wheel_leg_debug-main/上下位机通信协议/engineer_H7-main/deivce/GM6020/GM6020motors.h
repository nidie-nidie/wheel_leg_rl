#ifndef GM6020MOTORS_H
#define GM6020MOTORS_H

typedef struct
{
    unsigned short id; //这里是接收报文id
    struct
    {
        // 6020电机用电压控制，切换到电流控制比较麻烦，而且控制报文id会变
        short cmdSignal; //实际发送报文的数据
    } cmd;
    struct
    {
        unsigned char temperature; // degree
        double tor_fbk;            // N*m
        double vel_fbk;            // rad/s
        double pos_fbk;            // rad -pi~pi
    } para;                        //反馈报文的数据
    double last_pos;
    int ring_num;
    double total_angle; //与其他减速电机不同,启动时不为0,记录总转角
    char not_first;     // 0为第一次收到数据
    double zero_pos;    //用户设定的电机0角度值
} GM6020motor_t;

void GM6020_fbkdata(GM6020motor_t *GM6020_motor, unsigned char data[]);
void GM6020_init(GM6020motor_t *GM6020_motor, int id, double zero_pos);

#endif // GM6020MOTORS_H
