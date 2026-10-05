#ifndef LK_MOTORS_H
#define LK_MOTORS_H
#include "bsp_fdcan.h"
/*
转矩闭环控制命令（该命令仅在 MF、MH、MG 电机上实现）
主机发送该命令以控制电机的转矩电流输出，控制值 iqControl 为 int16_t 类型，数值范围-2048~
2048，对应 MF 电机实际转矩电流范围-16.5A~16.5A，对应 MG 电机实际转矩电流范围-33A~33A，
母线电流和电机的实际扭矩因不同电机而异
*/

/*
LK MF9025v2
电势常数 0.05 V/rpm
转矩常数 0.32 N*m/A
线电阻   0.5 Ohm
*/
/*
LK MG4010E-i10v3
电势常数 (1/108.3) V/rpm
转矩常数 0.07 N*m/A
减速比   10:1
*/
typedef struct
{
    char id;
    struct
    {
        double tor_set;
    } cmd;
    struct
    {
        unsigned char temperature;
        double tor_fbk;
        double vel_fbk;
        double pos_fbk;
    } para;
    double last_pos;
    int ring_num;
    double total_angle; //启动时为0,记录总转角
    char is_first;      // 0为第一次
    double init_angle;
} MF9025v2_t;

typedef struct
{
    char id;           // ID 1~32
    double gear_ratio; //速比，减速比为大于1的值，加速比为小于1的值
    struct
    {
        double tor_set;
    } cmd;
    struct
    {
        unsigned char temperature;
        double tor_fbk;
        double vel_fbk;
        double pos_fbk;
    } para;
    struct
    {
        unsigned char temperature;
        double tor_fbk;
        double vel_fbk;
        double pos_fbk;
    } last_para;
    int ring_num;
    double total_angle; //与其他减速电机不同,启动时不为0,记录总转角
    char not_first;     // 0为第一次收到数据
    double zero_pos;    //用户设定的电机0角度值
} MG4010_t;

void turn_off_LKmotor(unsigned short id, FDCAN_HandleTypeDef *hcan);
void turn_on_LKmotor(unsigned short id, FDCAN_HandleTypeDef *hcan);
void stop_LKmotor(unsigned short id, FDCAN_HandleTypeDef *hcan);
void MF9025v2_setTor(unsigned short id, FDCAN_HandleTypeDef *hcan, double torque);
void MF9025v2_fbkdata(MF9025v2_t *MF9025_motor, unsigned char data[]);
void MG4010_init(MG4010_t *MG4010_motor, char id, double zero_pos);
void MG4010_setTor(unsigned short id, FDCAN_HandleTypeDef *hcan, double torque);
void MG4010_fbkdata(MG4010_t *MG4010_motor, unsigned char data[]);

#endif // LK_MOTORS_H
