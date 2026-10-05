#include "serialplot.h"
#include "string.h"
#include "usbd_cdc_if.h"

static unsigned char serialplot_tx_buf[200] = {0xab};
// data 数据 num 位置编号 第一个为1 第二个为2
void serialplot_add(double data, int num)
{
    memcpy((void *)&(serialplot_tx_buf[1 + 8 * (num - 1)]), &data, 8);
}
// num 数据个数
void serialplot_send(int num)
{
    CDC_Transmit_HS(serialplot_tx_buf, 1 + 8 * num);
}
