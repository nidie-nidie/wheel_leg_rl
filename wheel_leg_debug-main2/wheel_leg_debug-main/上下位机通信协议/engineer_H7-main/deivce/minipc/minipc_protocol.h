#ifndef MINIPC_PROTOCOL_H
#define MINIPC_PROTOCOL_H

#include "stdint.h"
#include "string.h"
#include "CRC8_CRC16.h"
#include "bsp_dwt.h"

typedef enum __attribute__((packed))
{
	
	FOR_BACK=0,
	LEFT_RIGHT,
	
}chassis_direct_e;

typedef enum __attribute__((packed))
{
	ANGLE=0,
	SPEED,
	KEEP,
}Ctrl_mode_e;//增量式还是绝对式

typedef struct
{
    uint8_t mcu2minipc_buf[100];
    float no_data_time;
    float pack_loss_rate;
    __packed struct
    {
        uint8_t header; // = 0x5A;
        float max_realangle;  // r:
        float min_realangle;   // rad
			  float finesse_realangle;   // rad
        float pitch_realangle; //
        float roll_realangle;
        float z_realheight;
        uint16_t checksum; // = 0; // c!
    } mcu2minipc;
    __packed struct
    {
        uint8_t header; // = 0xA5;
        float max_angle_ctrl;
        float min_angle_ctrl;
        float finesse_angle_ctrl;
        float pitch_angle_ctrl;
			  Ctrl_mode_e roll_mode;//0为绝对 1为增量 2为保持
			  float roll_angle_ctrl;
			  Ctrl_mode_e z_mode;//0为绝对 1为增量
			  float z_ctrl;
			  

			  float y_erro;//y轴差距
			  float x_erro;//x轴差距
			  
			
        uint16_t checksum; // = 0;
    } minipc2mcu;
    uint32_t minipc_count;      //用于记时
    uint8_t rx_pack_state[100]; //记录当前及100个数据包中的接收状态，0什么都不表示，1表示接收正常，2表示丢包
} minipc_t;


extern minipc_t minipc;//小电脑结构体

void minipc_rec(minipc_t *pc, uint8_t pc_data_buf[]);
void minipc_upgrade(minipc_t *pc);

#endif // MINIPC_PROTOCOL_H
