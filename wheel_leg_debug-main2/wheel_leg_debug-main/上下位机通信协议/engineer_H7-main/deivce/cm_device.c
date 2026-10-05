#include "cm_device.h"

CM_t video_cm;//图传离线
CM_t rc_cm;//遥控离线

DEVICE_STATE device_state_list[MAX_DEVICE_NUM];

void device_init(CM_t *cm, double time_out, uint16_t device_index)
{
    cm->time_out = time_out;
    if (device_index > MAX_DEVICE_NUM)
        device_index = MAX_DEVICE_NUM;
    if (device_index < 1)
        device_index = 1;
    cm->device_index = device_index;
    cm->state = DEVICE_OK;
}
void device_fbk(CM_t *cm)
{
    //更新计时点
    cm->no_data_time = 0;
    DWT_GetDeltaT(&(cm->device_count));
}//在回调中调用


void device_refresh(CM_t *cm)
{
    float dt = DWT_GetDeltaT(&(cm->device_count));
    cm->no_data_time += dt;
    //这个结构体里面的time_cnt加加
    if (cm->no_data_time > cm->time_out)
    {
        cm->state = DEVICE_NOT_CONNECTED;
        if (cm->device_index - 1 < 0)
            return;
        device_state_list[cm->device_index - 1] = DEVICE_NOT_CONNECTED;
    }
    else
    {
        cm->state = DEVICE_OK;
        if (cm->device_index - 1 < 0)
            return;
        device_state_list[cm->device_index - 1] = DEVICE_OK;
    }
}
