#ifndef SCARA_KINEMATICS_H
#define SCARA_KINEMATICS_H

#include "main.h"
#include <math.h>
//#include "VideoTransmitter.h"
#define ARMLENGHT1 220 // unit mm 单位 毫米
#define ARMLENGHT2 220
#define ARMLENGHT3 210
// #define MAIN_ARM_ZERO_POINT -0.96
// #define MID_ARM_ZERO_POINT  0.43
// #define YAW_ZERO_POINT      0.03
// #define PITCH_ZERO_POINT    0.66
#define sqrt2 1.414213f

#pragma pack(1)
typedef struct
{
    float init_angle1;
    float init_angle2;
    float init_yaw;
    float init_pitch;
    float init_roll;
    float init_Z;

} initial_state;

typedef struct
{
    float delta_x;
    float delta_y;
    float delta_z;
    float delta_yaw;
    float delta_pitch;
    float delta_roll;
} slightly_controll_data;
#pragma pack()

void scara_inverse_kinematics(float x, float y, float L1, float L2, uint8_t handcoor, float angles[2]);
double polyval_calc(double p[], double x, uint8_t terms);
void check_boundary_scara(float x, float y, float res_xy[2]);
void scara_forward_kinematics(float angle1, float angle2, float L1, float L2, float res_xy[2]);
void GC_get_target_angles_slightly(slightly_controll_data data_pack, float result[6]);
void StateInit(float angle1, float angle2, float angle3, float angle4, float z, float roll_angle);
void RecordMode(uint8_t HeadByte);
void GetCurrentState(float angle1, float angle2, float angle3, float angle4, float z, float roll_angle);
void PushToCube(float result[6], float length);
void check_boundary_scara_lefthand(float x, float y, float res_xy[2]);

//enum PID_MODE
//{
//    PID_POSITION = 0,
//    PID_DELTA = 10
//};

//typedef struct
//{
//    uint8_t mode;
//    float Kp;
//    float Ki;
//    float Kd;

//    float max_out;  
//    float max_iout; 

//    float set;
//    float fdb;

//    float out;
//    float Pout;
//    float Iout;
//    float Dout;
//    float Dbuf[3];  //used in Delta mode
//    float error[3]; 

//} pid_type_def;
///**
//  * @brief          pid struct data init
//  * @param[out]     pid: PID struct data pointer
//  * @param[in]      mode: PID_POSITION: normal pid
//  *                       PID_DELTA: delta pid
//  * @param          Kp
//  * @param          Ki
//  * @param          Kd
//  * @param[in]      max_out: pid max out
//  * @param[in]      max_iout: pid max iout
//  * @retval         none
//  */
//extern void PID_init(pid_type_def *pid, uint8_t mode, float Kp,float Ki,float Kd, float max_out, float max_iout);

///**
//  * @brief          pid calculate 
//  * @param[out]     pid: PID struct data point
//  * @param[in]      ref: feedback data 
//  * @param[in]      set: set point
//  * @retval         pid out
//  */
//extern float PID_calc(pid_type_def *pid, float ref, float set);

///**
//  * @brief          pid out clear
//  * @param[out]     pid: PID struct data point
//  * @retval         none
//  */

#endif
