#ifndef N100IMU_H
#define N100IMU_H

#include "stdint.h"
#include "CRC8_CRC16.h"
#include "string.h"
#include "bsp_dwt.h"

typedef struct
{
    uint8_t data_buf[100];
    struct
    {
        float no_data_time;
        float pack_loss_rate;
        uint32_t count;             //用于记时
        uint8_t rx_pack_state[100]; //记录当前及100个数据包中的接收状态，0什么都不表示，1表示接收正常，2表示丢包
    } tsm;                          // tansportation state monitor
    __packed struct
    {
        uint8_t SOF;
        uint8_t data_type;
        uint8_t data_length;
        uint8_t seq;
        uint8_t crc8_check_sum;
    } frame_header;
    struct
    {
        float gyroscope_x;          // unit: rad/s
        float gyroscope_y;          // unit: rad/s
        float gyroscope_z;          // unit: rad/s
        float accelerometer_x;      // m/s^2
        float accelerometer_y;      // m/s^2
        float accelerometer_z;      // m/s^2
        float magnetometer_x;       // mG
        float magnetometer_y;       // mG
        float magnetometer_z;       // mG
        float imu_temperature;      // C
        float Pressure;             // Pa
        float pressure_temperature; // C
        uint32_t Timestamp;         // us
    } msg_imu;
    struct
    {
        float RollSpeed;    // unit: rad/s
        float PitchSpeed;   // unit: rad/s
        float HeadingSpeed; // unit: rad/s
        float Roll;         // unit: rad
        float Pitch;        // unit: rad
        float Heading;      // unit: rad
        float Qw;           // w          //Quaternion
        float Qx;           // x
        float Qy;           // y
        float Qz;           // z
        uint32_t Timestamp; // unit: us
    } msg_AHRS;
} N100imu_t;

void n100imu_fbkdata(N100imu_t *n100);
void n100imu_call(N100imu_t *n100, uint8_t data[]);

#endif // N100IMU_H
