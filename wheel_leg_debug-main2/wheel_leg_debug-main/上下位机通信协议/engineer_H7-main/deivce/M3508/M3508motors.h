#ifndef M3508MOTORS_H
#define M3508MOTORS_H
/*
RoboMaster M3508电机
根据负载特性曲线(使用c620电调做负载闭环)估计得到
转矩常数 (0.3*187/3591) N*m/A
额定转速约50*(3591/187)rad/s
额定转矩约为3/(3591/187) N*m
估计电机输出功率为140w
*/
typedef struct
{
    unsigned short id; //这里是接收报文id
    double gear_ratio; //速比，减速比为大于1的值，加速比为小于1的值
    double power_limit;
    struct
    {
        double tor_set;  //扭矩设置
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
    double total_angle; //启动时为0,记录总转角
    char not_first;     // 0为第一次收到数据
    double init_angle;
    int start_flag;
} M3508motor_t;

void M3508_fbkdata(M3508motor_t *m3508_motor, unsigned char data[]);
void M3508_torSet2cmd(M3508motor_t *m3508_motor, double torque);
void M3508_init(M3508motor_t *m3508_motor, int id, double gear_ratio);

#endif // M3508MOTORS_H
