#include "superCap.h"

void cap_cmd_buf_upgrade(superCap_t *cap)
{
    uint16_t temp_cpb = 0;
    uint16_t temp_cpl = 0;
    temp_cpb = cap->cap_cmd.chassis_power_buf;
    temp_cpl = cap->cap_cmd.chassis_power_limit;
    cap->cap_cmd.cap_cmd_buf[0] = (temp_cpb >> 8);
    cap->cap_cmd.cap_cmd_buf[1] = temp_cpb;
    cap->cap_cmd.cap_cmd_buf[2] = (temp_cpl >> 8);
    cap->cap_cmd.cap_cmd_buf[3] = temp_cpl;
    cap->cap_cmd.cap_cmd_buf[4] = cap->cap_cmd.mode;
}
void cap_fbkdata(superCap_t *cap, uint8_t rx_data[8])
{
    int16_t temp_vol = 0;
    int16_t temp_power = 0;
    temp_vol = (((int16_t)rx_data[0]) << 8) + rx_data[1];
    temp_power = (((int16_t)rx_data[2]) << 8) + rx_data[3];
    cap->cap_data.cap_vol = ((float)temp_vol) / 1000.0f;
    cap->cap_data.chassis_power = ((float)temp_power) / 1000.0f;
    cap->cap_data.mode = rx_data[4];
}
