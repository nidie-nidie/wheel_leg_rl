#include "custom_ui_draw.h"
#include "bsp_usart.h"
#include "gimbal_task.h"
#include "arm_math.h"
#include "CAN_receive.h"
#include "shoot_task.h"
#include "referee.h"

//外部引用数据用于画UI
extern shoot_move_t shoot_control;
extern SuperCap_t SuperCap;
extern gimbal_move_t gimbal_move;
extern float LegAngleLeft,LegLLeft;
extern auto_shoot_t auto_shoot;

uint16_t send_id;
uint16_t receive_id;
tank_ui_t tank;

/***********************************打包发送***********************************/
extern DMA_HandleTypeDef hdma_usart6_tx;
extern UART_HandleTypeDef huart6;
uint8_t seq = 0;
//长度定义
#define MAX_SIZE 128
#define frameheader_len 5
#define cmd_len 2
#define crc_len 2
//发送函数
void referee_data_pack_handle(uint8_t sof, uint16_t cmd_id, uint8_t *p_data, uint16_t len)
{
    uint8_t tx_buff[MAX_SIZE];
    uint16_t frame_length = frameheader_len + cmd_len + crc_len + len;

    memset(tx_buff, 0, MAX_SIZE); // 存储数据的数组清零

    /*****帧头打包*****/
    tx_buff[0] = sof;                                  // 数据帧起始字节
    memcpy(&tx_buff[1], (uint8_t *)&len, sizeof(len)); // 数据帧中data的长度
    tx_buff[3] = seq;                                  // 包序号
    append_CRC8_check_sum(tx_buff, frameheader_len);   // 帧头校验CRC8

    /*****命令码打包*****/
    memcpy(&tx_buff[frameheader_len], (uint8_t *)&cmd_id, cmd_len);

    /*****数据打包*****/
    memcpy(&tx_buff[frameheader_len + cmd_len], p_data, len);
    append_CRC16_check_sum(tx_buff, frame_length); // 一帧数据校验CRC16
    if (seq == 0xff)
        seq = 0;
    else
        seq++;
		
    //HAL_UART_Transmit(&huart6,tx_buff,frame_length,13);
    usart6_tx_dma_enable(tx_buff, frame_length);
}

void draw_get_robot_id(void)//通过裁判系统ID获得当前机器人画UI需要的send_id和receive_id
{
    switch (robot_state.robot_id)
    {
    case RED_HERO:
        send_id = 1;
        receive_id = 0x0101;
        break;
    case RED_ENGINEER:
        send_id = 2;
        receive_id = 0x0102;
        break;
    case RED_STANDARD_1:
        send_id = 3;
        receive_id = 0x0103;
        break;
    case RED_STANDARD_2:
        send_id = 4;
        receive_id = 0x0104;
        break;
    case RED_STANDARD_3:
        send_id = 5;
        receive_id = 0x0105;
        break;
    case RED_AERIAL:
        send_id = 6;
        receive_id = 0x0106;
        break;
    case BLUE_HERO:
        send_id = 101;
        receive_id = 0x0165;
        break;
    case BLUE_ENGINEER:
        send_id = 102;
        receive_id = 0x0166;
        break;
    case BLUE_STANDARD_1:
        send_id = 103;
        receive_id = 0x0167;
        break;
    case BLUE_STANDARD_2:
        send_id = 104;
        receive_id = 0x0168;
        break;
    case BLUE_STANDARD_3:
        send_id = 105;
        receive_id = 0x0169;
        break;
    case BLUE_AERIAL:
        send_id = 106;
        receive_id = 0x016A;
        break;
    default:
        send_id = 0;
        receive_id = 0;
        break;
    }
}

void draw_init_all(uint16_t sender_id, uint16_t receiver_id)
{	
	clear_all_layer(sender_id, receiver_id);//清屏
  osDelay(100);
	draw_shoot_mode(sender_id, receiver_id, 1);//初始化左上角的发射模式
	osDelay(100);
	draw_capacity_bar(sender_id, receiver_id, 1);//初始化超级电容容量进度条
	osDelay(100);
	draw_capacity_bar_deadline(sender_id, receiver_id, 1);//初始化进度条的警告线
	osDelay(100);
	Leg_Simple_drawing(sender_id, receiver_id, 1);//腿当前状态的简笔画
	osDelay(100);
	draw_aim(sender_id, receiver_id, 1);//发射的框框，辅助发射和判断自瞄与手瞄
	osDelay(100);
	draw_chassis_mode(sender_id, receiver_id, 1);//初始化左上角的底盘模式
	osDelay(100);
	draw_shoot_circle(sender_id, receiver_id, 1);//dwc想要的弹口发射框
	osDelay(100);
	
	draw_gimbal_relative_angle_tangle(sender_id, receiver_id, 1);//画云台和显示yaw轴相对角度
	osDelay(100);
	draw_single_tangle_test(sender_id, receiver_id, 1);
	osDelay(100);

}

void clear_all_layer(uint16_t sender_id, uint16_t receiver_id)
{

    ext_client_custom_graphic_delete_t init_struct;

    init_struct.header_data_t.data_cmd_id = 0x0100;//选手端删除图传
    init_struct.header_data_t.receiver_ID = receiver_id;
    init_struct.header_data_t.sender_ID = sender_id;

    init_struct.operate_tpye = 2;//删除所有图层
    init_struct.layer = 0;//这个应该就无所谓了
		
		//0xA5即帧头，0x0301表示想要画UI
    referee_data_pack_handle(0xA5, 0x0301, (uint8_t *)&init_struct, sizeof(init_struct));
}

void draw_shoot_mode(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type)
{
    ext_client_custom_character_t draw_init;
    draw_init.header_data_t.data_cmd_id = 0x0110;//选手端绘字符
    draw_init.header_data_t.sender_ID = sender_id;
    draw_init.header_data_t.receiver_ID = receiver_id;
	
    draw_init.grapic_data_struct.graphic_name[0] = 0;
    draw_init.grapic_data_struct.graphic_name[1] = 0;
    draw_init.grapic_data_struct.graphic_name[2] = 1;
    draw_init.grapic_data_struct.operate_tpye = op_type;//图形操作，1：增加，2：修改
    draw_init.grapic_data_struct.graphic_tpye = 7;//字符
    draw_init.grapic_data_struct.layer = 1;//图层数（1）

    draw_init.grapic_data_struct.start_angle = 40; //字符大小16

    draw_init.grapic_data_struct.width = 4;//线宽2
    draw_init.grapic_data_struct.start_x = 60;//1300，820
    draw_init.grapic_data_struct.start_y = 820;
    draw_init.grapic_data_struct.end_x = 0;//0
    draw_init.grapic_data_struct.end_y = 0;
    draw_init.grapic_data_struct.radius = 0;
	
    memset(draw_init.data, 0, sizeof(draw_init.data));//清除数据，防止上次数据的干扰
    draw_init.grapic_data_struct.color = 2;//绿色
		
		switch(shoot_control.shoot_mode)
		{
			case SHOOT_STOP:
        draw_init.grapic_data_struct.end_angle = 10; 
        draw_init.data[0] = 'S';
        draw_init.data[1] = 'h';
        draw_init.data[2] = 'o';
        draw_init.data[3] = 'o';
        draw_init.data[4] = 't';
        draw_init.data[5] = '_';
        draw_init.data[6] = 'S';
        draw_init.data[7] = 't';
        draw_init.data[8] = 'o';
        draw_init.data[9] = 'p';
				break;
			case SHOOT_READY:
        draw_init.grapic_data_struct.end_angle = 11; 
        draw_init.data[0] = 'S';
        draw_init.data[1] = 'h';
        draw_init.data[2] = 'o';
        draw_init.data[3] = 'o';
        draw_init.data[4] = 't';
        draw_init.data[5] = '_';
        draw_init.data[6] = 'R';
        draw_init.data[7] = 'e';
        draw_init.data[8] = 'a';
        draw_init.data[9] = 'd';
        draw_init.data[10] = 'y';
				break;
			case SHOOT_AUTO:
        draw_init.grapic_data_struct.end_angle = 10; 
        draw_init.data[0] = 'S';
        draw_init.data[1] = 'h';
        draw_init.data[2] = 'o';
        draw_init.data[3] = 'o';
        draw_init.data[4] = 't';
        draw_init.data[5] = '_';
        draw_init.data[6] = 'A';
        draw_init.data[7] = 'u';
        draw_init.data[8] = 't';
        draw_init.data[9] = 'o';
				break;
			default:
        draw_init.grapic_data_struct.end_angle = 12; 
        draw_init.data[0] = 'S';
        draw_init.data[1] = 'h';
        draw_init.data[2] = 'o';
        draw_init.data[3] = 'o';
        draw_init.data[4] = 't';
        draw_init.data[5] = '_';
        draw_init.data[6] = 'B';
        draw_init.data[7] = 'u';
        draw_init.data[8] = 'l';
        draw_init.data[9] = 'l';
        draw_init.data[10] = 'e';
        draw_init.data[11] = 't';
				break;
		}
    referee_data_pack_handle(0xA5, 0x0301, (uint8_t *)&draw_init, sizeof(draw_init));
}

void draw_capacity_bar(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type)
{
    ext_client_custom_graphic_double_t draw_init;
    draw_init.header_data_t.data_cmd_id = 0x0102;//绘制2个图形
    draw_init.header_data_t.sender_ID = sender_id;
    draw_init.header_data_t.receiver_ID = receiver_id;

    draw_init.grapic_data_struct[0].graphic_name[0] = 0;
    draw_init.grapic_data_struct[0].graphic_name[1] = 0;
    draw_init.grapic_data_struct[0].graphic_name[2] = 2;

    draw_init.grapic_data_struct[0].graphic_tpye = 1;
    draw_init.grapic_data_struct[0].operate_tpye = op_type;
    draw_init.grapic_data_struct[0].layer = 8;
    draw_init.grapic_data_struct[0].color = 4;// 紫红
    draw_init.grapic_data_struct[0].width = 3;
    draw_init.grapic_data_struct[0].start_x = 960-128;
    draw_init.grapic_data_struct[0].start_y = 100;
    draw_init.grapic_data_struct[0].end_x = 960+128;
    draw_init.grapic_data_struct[0].end_y = 100+30;

    draw_init.grapic_data_struct[1].graphic_name[0] = 0;
    draw_init.grapic_data_struct[1].graphic_name[1] = 0;
    draw_init.grapic_data_struct[1].graphic_name[2] = 3;
    draw_init.grapic_data_struct[1].graphic_tpye = 0;
    draw_init.grapic_data_struct[1].operate_tpye = op_type;
    draw_init.grapic_data_struct[1].layer = 9;
    draw_init.grapic_data_struct[1].color = 2;//绿色
    draw_init.grapic_data_struct[1].width = 28;
    draw_init.grapic_data_struct[1].start_x = 960-128 + 2;
    draw_init.grapic_data_struct[1].start_y = 100 + 15;
    draw_init.grapic_data_struct[1].end_x = 960+128 - (256-SuperCap.capEnergy);
    draw_init.grapic_data_struct[1].end_y = 100 + 15;

    referee_data_pack_handle(0xA5, 0x0301, (uint8_t *)&draw_init, sizeof(draw_init));
}

void draw_capacity_bar_deadline(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type)
{
    ext_client_custom_graphic_single_t draw_test;

    draw_test.header_data_t.data_cmd_id = 0x0101;
    draw_test.header_data_t.sender_ID = sender_id;
    draw_test.header_data_t.receiver_ID = receiver_id;

    draw_test.grapic_data_struct.graphic_name[0] = 0;
    draw_test.grapic_data_struct.graphic_name[1] = 0;
    draw_test.grapic_data_struct.graphic_name[2] = 4;

    draw_test.grapic_data_struct.graphic_tpye = 0;
    draw_test.grapic_data_struct.operate_tpye = op_type;
    draw_test.grapic_data_struct.layer = 9;
    draw_test.grapic_data_struct.color = 4;
    draw_test.grapic_data_struct.width = 2;
    draw_test.grapic_data_struct.start_x = 960-128 + 2 + 100;
    draw_test.grapic_data_struct.start_y = 100 + 15 + 45;
    draw_test.grapic_data_struct.end_x = 960-128 + 2 + 100;
    draw_test.grapic_data_struct.end_y = 100 + 15 - 45;

    referee_data_pack_handle(0xA5, 0x0301, (uint8_t *)&draw_test, sizeof(draw_test));
}

void Leg_Simple_drawing(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type)
{
    ext_client_custom_graphic_five_t draw_init;
    draw_init.header_data_t.data_cmd_id = 0x0103;
    draw_init.header_data_t.sender_ID = sender_id;
    draw_init.header_data_t.receiver_ID = receiver_id;
	
		//中间身体
    draw_init.grapic_data_struct[0].graphic_name[0] = 0;
    draw_init.grapic_data_struct[0].graphic_name[1] = 0;
    draw_init.grapic_data_struct[0].graphic_name[2] = 5;
    draw_init.grapic_data_struct[0].operate_tpye = op_type;
    draw_init.grapic_data_struct[0].graphic_tpye = 0;//直线
    draw_init.grapic_data_struct[0].layer = 9;//图层数
    draw_init.grapic_data_struct[0].color = 3;//橙色
    draw_init.grapic_data_struct[0].width = 3;//线宽
    draw_init.grapic_data_struct[0].start_x = 1600;
    draw_init.grapic_data_struct[0].start_y = 800;
    draw_init.grapic_data_struct[0].end_x = 1600+200;
    draw_init.grapic_data_struct[0].end_y = 800;
    draw_init.grapic_data_struct[0].radius = 0;

    draw_init.grapic_data_struct[1].graphic_name[0] = 0;
    draw_init.grapic_data_struct[1].graphic_name[1] = 0;
    draw_init.grapic_data_struct[1].graphic_name[2] = 6;
    draw_init.grapic_data_struct[1].operate_tpye = op_type;
    draw_init.grapic_data_struct[1].graphic_tpye = 0;//直线
    draw_init.grapic_data_struct[1].layer = 9;//图层数
    draw_init.grapic_data_struct[1].color = 4;//紫红色
    draw_init.grapic_data_struct[1].width = 3;//线宽
    draw_init.grapic_data_struct[1].start_x = 1600+100;
    draw_init.grapic_data_struct[1].start_y = 800;
    draw_init.grapic_data_struct[1].end_x = 1600+100+gimbal_move.dynamicUI.LeftX1;
    draw_init.grapic_data_struct[1].end_y = 800+gimbal_move.dynamicUI.LeftY1;
    draw_init.grapic_data_struct[1].radius = 0;
		
		//左小腿
    draw_init.grapic_data_struct[2].graphic_name[0] = 0;
    draw_init.grapic_data_struct[2].graphic_name[1] = 0;
    draw_init.grapic_data_struct[2].graphic_name[2] = 7;
    draw_init.grapic_data_struct[2].operate_tpye = op_type;
    draw_init.grapic_data_struct[2].graphic_tpye = 0;//直线
    draw_init.grapic_data_struct[2].layer = 9;//图层数
    draw_init.grapic_data_struct[2].color = 4;//紫红色
    draw_init.grapic_data_struct[2].width = 3;//线宽
    draw_init.grapic_data_struct[2].start_x = 1600+100+gimbal_move.dynamicUI.LeftX1;
    draw_init.grapic_data_struct[2].start_y = 800+gimbal_move.dynamicUI.LeftY1;
    draw_init.grapic_data_struct[2].end_x = 1600+100+gimbal_move.dynamicUI.LeftX2;
    draw_init.grapic_data_struct[2].end_y = 800+gimbal_move.dynamicUI.LeftY2;
    draw_init.grapic_data_struct[2].radius = 0;

    draw_init.grapic_data_struct[3].graphic_name[0] = 0;
    draw_init.grapic_data_struct[3].graphic_name[1] = 0;
    draw_init.grapic_data_struct[3].graphic_name[2] = 8;
    draw_init.grapic_data_struct[3].operate_tpye = op_type;
    draw_init.grapic_data_struct[3].graphic_tpye = 0;//直线
    draw_init.grapic_data_struct[3].layer = 9;//图层数
    draw_init.grapic_data_struct[3].color = 6;//青色
    draw_init.grapic_data_struct[3].width = 3;//线宽
    draw_init.grapic_data_struct[3].start_x = 1600+100;
    draw_init.grapic_data_struct[3].start_y = 800;
    draw_init.grapic_data_struct[3].end_x = 1600+100+gimbal_move.dynamicUI.RightX1;
    draw_init.grapic_data_struct[3].end_y = 800+gimbal_move.dynamicUI.RightY1;
    draw_init.grapic_data_struct[3].radius = 0;
		
		//右小腿
    draw_init.grapic_data_struct[4].graphic_name[0] = 0;
    draw_init.grapic_data_struct[4].graphic_name[1] = 0;
    draw_init.grapic_data_struct[4].graphic_name[2] = 9;
    draw_init.grapic_data_struct[4].operate_tpye = op_type;
    draw_init.grapic_data_struct[4].graphic_tpye = 0;//直线
    draw_init.grapic_data_struct[4].layer = 9;//图层数
    draw_init.grapic_data_struct[4].color = 6;//青色
    draw_init.grapic_data_struct[4].width = 3;//线宽
    draw_init.grapic_data_struct[4].start_x = 1600+100+gimbal_move.dynamicUI.RightX1;
    draw_init.grapic_data_struct[4].start_y = 800+gimbal_move.dynamicUI.RightY1;
    draw_init.grapic_data_struct[4].end_x = 1600+100+gimbal_move.dynamicUI.RightX2;
    draw_init.grapic_data_struct[4].end_y = 800+gimbal_move.dynamicUI.RightY2;
    draw_init.grapic_data_struct[4].radius = 0;
	
    referee_data_pack_handle(0xA5, 0x0301, (uint8_t *)&draw_init, sizeof(draw_init));
}

void draw_aim(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type)
{
    ext_client_custom_graphic_single_t  draw_init;
    draw_init.header_data_t.data_cmd_id = 0x0101;//绘制1个
    draw_init.header_data_t.sender_ID = sender_id;
    draw_init.header_data_t.receiver_ID = receiver_id;

    draw_init.grapic_data_struct.graphic_name[0] = 0;
    draw_init.grapic_data_struct.graphic_name[1] = 0;
    draw_init.grapic_data_struct.graphic_name[2] = 10;

    draw_init.grapic_data_struct.graphic_tpye = 1;
    draw_init.grapic_data_struct.operate_tpye = op_type;
    draw_init.grapic_data_struct.layer = 9;
    draw_init.grapic_data_struct.width = 3;
    draw_init.grapic_data_struct.start_x = 730;
    draw_init.grapic_data_struct.start_y = 700;
    draw_init.grapic_data_struct.end_x = 1170;
    draw_init.grapic_data_struct.end_y = 400;
	
		if(auto_shoot.mode==1)
		{
	    draw_init.grapic_data_struct.color = 4;// 紫红
		}
		else
		{
	    draw_init.grapic_data_struct.color= 2;// 绿色

		}
    referee_data_pack_handle(0xA5, 0x0301, (uint8_t *)&draw_init, sizeof(draw_init));
}

void draw_chassis_mode(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type)
{
    ext_client_custom_character_t draw_init;
    draw_init.header_data_t.data_cmd_id = 0x0110;//选手端绘字符
    draw_init.header_data_t.sender_ID = sender_id;
    draw_init.header_data_t.receiver_ID = receiver_id;
	
    draw_init.grapic_data_struct.graphic_name[0] = 0;
    draw_init.grapic_data_struct.graphic_name[1] = 0;
    draw_init.grapic_data_struct.graphic_name[2] = 11;
    draw_init.grapic_data_struct.operate_tpye = op_type;//图形操作，1：增加，2：修改
    draw_init.grapic_data_struct.graphic_tpye = 7;//字符
    draw_init.grapic_data_struct.layer = 2;//图层数（1）

    draw_init.grapic_data_struct.start_angle = 40; //字符大小16

    draw_init.grapic_data_struct.width = 4;//线宽2
    draw_init.grapic_data_struct.start_x = 60;//1300，820
    draw_init.grapic_data_struct.start_y = 720;
    draw_init.grapic_data_struct.end_x = 0;//0
    draw_init.grapic_data_struct.end_y = 0;
    draw_init.grapic_data_struct.radius = 0;
	
    memset(draw_init.data, 0, sizeof(draw_init.data));//清除数据，防止上次数据的干扰
    draw_init.grapic_data_struct.color = 2;//绿色
		
		switch(gimbal_move.TwoBoardControlGimbal.mode)
		{
			case 0:
        draw_init.grapic_data_struct.end_angle = 12; 
        draw_init.data[0] = 'C';
        draw_init.data[1] = 'h';
        draw_init.data[2] = 'a';
        draw_init.data[3] = 's';
        draw_init.data[4] = 's';
        draw_init.data[5] = 'i';
        draw_init.data[6] = 'S';
        draw_init.data[7] = '_';
        draw_init.data[8] = 'Z';
        draw_init.data[9] = 'e';
        draw_init.data[10] = 'r';
        draw_init.data[11] = 'o';
				break;
			case 8:
        draw_init.grapic_data_struct.end_angle = 11; 
        draw_init.data[0] = 'C';
        draw_init.data[1] = 'h';
        draw_init.data[2] = 'a';
        draw_init.data[3] = 's';
        draw_init.data[4] = 's';
        draw_init.data[5] = 'i';
        draw_init.data[6] = 'S';
        draw_init.data[7] = '_';
        draw_init.data[8] = 'T';
        draw_init.data[9] = 'o';
        draw_init.data[10] = 'p';
				break;
			default:
        draw_init.grapic_data_struct.end_angle = 14; 
        draw_init.data[0] = 'C';
        draw_init.data[1] = 'h';
        draw_init.data[2] = 'a';
        draw_init.data[3] = 's';
        draw_init.data[4] = 's';
        draw_init.data[5] = 'i';
        draw_init.data[6] = 'S';
        draw_init.data[7] = '_';
        draw_init.data[8] = 'F';
        draw_init.data[9] = 'o';
        draw_init.data[10] = 'l';
        draw_init.data[11] = 'l';
        draw_init.data[12] = 'o';
        draw_init.data[13] = 'w';
				break;
		}
    referee_data_pack_handle(0xA5, 0x0301, (uint8_t *)&draw_init, sizeof(draw_init));
}


void draw_shoot_circle(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type)
{
    ext_client_custom_graphic_double_t draw_init;
    draw_init.header_data_t.data_cmd_id = 0x0101;//绘制1个
    draw_init.header_data_t.sender_ID = sender_id;
    draw_init.header_data_t.receiver_ID = receiver_id;

    draw_init.grapic_data_struct[0].graphic_name[0] = 0;
    draw_init.grapic_data_struct[0].graphic_name[1] = 0;
    draw_init.grapic_data_struct[0].graphic_name[2] = 12;

    draw_init.grapic_data_struct[0].graphic_tpye = 2;
    draw_init.grapic_data_struct[0].operate_tpye = op_type;
    draw_init.grapic_data_struct[0].layer = 9;
    draw_init.grapic_data_struct[0].width = 3;
    draw_init.grapic_data_struct[0].start_x = 960;
    draw_init.grapic_data_struct[0].start_y = 400;
    draw_init.grapic_data_struct[0].radius = 50;

	  draw_init.grapic_data_struct[0].color = 0;//己方颜色
    referee_data_pack_handle(0xA5, 0x0301, (uint8_t *)&draw_init, sizeof(draw_init));
}

void draw_gimbal_relative_angle_tangle(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type)
{

    ext_client_custom_graphic_double_t draw_init;

    draw_init.header_data_t.data_cmd_id = 0x0102;
    draw_init.header_data_t.sender_ID = sender_id;
    draw_init.header_data_t.receiver_ID = receiver_id;


    draw_init.grapic_data_struct[0].graphic_name[0] = 0;
    draw_init.grapic_data_struct[0].graphic_name[1] = 0;
    draw_init.grapic_data_struct[0].graphic_name[2] = 13;

    draw_init.grapic_data_struct[0].graphic_tpye = 2;//正圆
    draw_init.grapic_data_struct[0].operate_tpye = op_type;			//图形操作，1：增加
    draw_init.grapic_data_struct[0].layer = 9;                      //图层数（3）
    draw_init.grapic_data_struct[0].color = 6;                      //青色
    draw_init.grapic_data_struct[0].width = 6;                      //线宽（6）
    draw_init.grapic_data_struct[0].start_x = 1600;//1600
    draw_init.grapic_data_struct[0].start_y = 500;//500
    draw_init.grapic_data_struct[0].radius = 50;                     

    draw_init.grapic_data_struct[1].graphic_name[0] = 0;
    draw_init.grapic_data_struct[1].graphic_name[1] = 0;
    draw_init.grapic_data_struct[1].graphic_name[2] = 14;
    draw_init.grapic_data_struct[1].graphic_tpye = 0;
    draw_init.grapic_data_struct[1].operate_tpye = op_type;
    draw_init.grapic_data_struct[1].layer = 7;
    draw_init.grapic_data_struct[1].color = 3;
    draw_init.grapic_data_struct[1].width = 4;
    draw_init.grapic_data_struct[1].start_x = 1600;
    draw_init.grapic_data_struct[1].start_y = 500;
    draw_init.grapic_data_struct[1].end_x = 1600 + tank.gx;
    draw_init.grapic_data_struct[1].end_y = 500 + tank.gy;

    referee_data_pack_handle(0xA5, 0x0301, (uint8_t *)&draw_init, sizeof(draw_init));
}

void cal_draw_gimbal_relative_angle_tangle(tank_ui_t *tank)
{
    tank->r = 50 * 2.2;
    tank->angle = (float)(gimbal_move.gimbal_yaw_motor.relative_angle-1.57f+3.14f);
    tank->gx = (cosf(tank->angle) * (float)tank->r);
    tank->gy = (sinf(tank->angle) * (float)tank->r);
}

void draw_single_tangle_test(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type)
{
    ext_client_custom_graphic_single_t draw_test;

    draw_test.header_data_t.data_cmd_id = 0x0101;
    draw_test.header_data_t.sender_ID = sender_id;
    draw_test.header_data_t.receiver_ID = receiver_id;

    draw_test.grapic_data_struct.graphic_name[0] = 0;
    draw_test.grapic_data_struct.graphic_name[1] = 0;
    draw_test.grapic_data_struct.graphic_name[2] = 15;

    draw_test.grapic_data_struct.graphic_tpye = 1;
    draw_test.grapic_data_struct.operate_tpye = op_type;
    draw_test.grapic_data_struct.layer = 7;
    draw_test.grapic_data_struct.color = 2;
    draw_test.grapic_data_struct.width = 4;
    draw_test.grapic_data_struct.start_x = 1600;
    draw_test.grapic_data_struct.start_y = 500;
    draw_test.grapic_data_struct.end_x = 1600 + 0;
    draw_test.grapic_data_struct.end_y = 500 + 100;

    referee_data_pack_handle(0xA5, 0x0301, (uint8_t *)&draw_test, sizeof(draw_test));
}
