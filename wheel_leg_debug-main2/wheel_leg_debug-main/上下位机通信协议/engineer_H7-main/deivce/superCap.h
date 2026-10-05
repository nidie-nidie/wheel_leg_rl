#ifndef SUPERCAP_H
#define SUPERCAP_H

#include "stdint.h"

typedef struct
{
    struct
    {
        uint8_t cap_cmd_buf[8];
        uint16_t chassis_power_buf;   //缓冲能量
        uint16_t chassis_power_limit; //底盘功率限制
        uint8_t mode;                 // 0x02关掉电容（底盘直连）0x03接入电容
    } cap_cmd;
    struct
    {
        float cap_vol;       //电容电压
        float chassis_power; //底盘功率
        uint8_t mode;        //状态码
    } cap_data;
} superCap_t;

void cap_cmd_buf_upgrade(superCap_t *cap);
void cap_fbkdata(superCap_t *cap, uint8_t rx_data[8]);

#endif // SUPERCAP_H
