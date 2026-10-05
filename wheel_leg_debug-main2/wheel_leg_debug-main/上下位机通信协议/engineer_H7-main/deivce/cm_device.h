#ifndef CM_DEVICE_H
#define CM_DEVICE_H

#include "bsp_dwt.h"

#ifndef MAX_DEVICE_NUM
#define MAX_DEVICE_NUM (100)
#endif

typedef enum
{
    DEVICE_OK = 1,
    DEVICE_NOT_CONNECTED = 2
} DEVICE_STATE;

typedef struct
{
    uint32_t device_count; //用于dwt计算时间
    double no_data_time;
    double time_out;
    uint16_t device_index; //设备状态列表的系数1~MAX_DEVICE_NUM
    DEVICE_STATE state;
} CM_t; // connection monitor

extern DEVICE_STATE device_state_list[MAX_DEVICE_NUM];

extern CM_t video_cm;//图传离线
extern CM_t rc_cm;//遥控离线

void device_init(CM_t *cm, double time_out, uint16_t device_index);
void device_fbk(CM_t *cm);
void device_refresh(CM_t *cm);

#endif
