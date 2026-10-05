#include "RMmotor_send.h"

unsigned char RMmotors_tx_buf[8] = {0};

void RMmotor_send_4(unsigned short id, FDCAN_HandleTypeDef *hcan, short cmd1, short cmd2, short cmd3, short cmd4)
{
    RMmotors_tx_buf[0] = (cmd1 >> 8);
    RMmotors_tx_buf[1] = cmd1;
    RMmotors_tx_buf[2] = (cmd2 >> 8);
    RMmotors_tx_buf[3] = cmd2;
    RMmotors_tx_buf[4] = (cmd3 >> 8);
    RMmotors_tx_buf[5] = cmd3;
    RMmotors_tx_buf[6] = (cmd4 >> 8);
    RMmotors_tx_buf[7] = cmd4;
    fdcanx_send_data(hcan, id, RMmotors_tx_buf, 8);
}
