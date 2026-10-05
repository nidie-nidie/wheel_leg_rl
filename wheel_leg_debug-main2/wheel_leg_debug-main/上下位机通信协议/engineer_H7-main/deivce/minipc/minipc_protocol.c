#include "minipc_protocol.h"

minipc_t minipc;//小电脑结构体

uint8_t size;

uint8_t buff_test[50];
//在接收数据的回调中使用，接收数据到缓冲区，并清空no_data_time
void minipc_rec(minipc_t *pc, uint8_t pc_data_buf[])
{
    //移位
    for (int i = 0; i < 99; i++)
        pc->rx_pack_state[99 - i] = pc->rx_pack_state[98 - i];
    // crc校验
	  size=sizeof(pc->minipc2mcu);
	  memcpy(buff_test, (const void *)pc_data_buf,sizeof(pc->minipc2mcu));
//    if (Verify_CRC16_Check_Sum(pc_data_buf, sizeof(pc->minipc2mcu)))
//    {
//        pc->rx_pack_state[0] = 1;
        memcpy((void *)&(pc->minipc2mcu), (const void *)pc_data_buf, sizeof(pc->minipc2mcu));
//    }
//    else
//    pc->rx_pack_state[0] = 2;
//    pc->no_data_time = 0;
//    DWT_GetDeltaT(&(pc->minipc_count)); //更新记时点
}


//放在循环中，更新no_data_time和pack_loss_rate，并将mcu数据附上crc校验，更新发送缓冲区
void minipc_upgrade(minipc_t *pc)
{
    float dt = DWT_GetDeltaT(&(pc->minipc_count));
    pc->no_data_time += dt;//典型
	
    int pack_ok = 0;
    int pack_loss = 0;
    for (int i = 0; i < 100; i++)
    {
        if (pc->rx_pack_state[i] == 1)
            pack_ok++;
        else if (pc->rx_pack_state[i] == 2)
            pack_loss++;
        else
            break;
    }
    if ((pack_loss + pack_ok) > 0)
        pc->pack_loss_rate = ((float)pack_loss) / ((float)(pack_loss + pack_ok));
		
    pc->mcu2minipc.header = 0x5a;//先设置帧头
    Append_CRC16_Check_Sum((uint8_t *)&(pc->mcu2minipc), sizeof(pc->mcu2minipc));//添加
    memcpy((void *)(pc->mcu2minipc_buf), (const void *)&(pc->mcu2minipc), sizeof(pc->mcu2minipc));//将帧头等发送的结构体 添加到缓冲区
}

