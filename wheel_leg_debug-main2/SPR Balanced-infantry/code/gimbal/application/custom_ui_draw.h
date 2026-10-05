#ifndef CUSTOM_UI_DRAW_H
#define CUSTOM_UI_DRAW_H

#include "main.h"
#include "CRC8_CRC16.h"
#include "protocol.h"
#include "string.h"
#include "cmsis_os.h"

typedef __packed struct
{
  uint16_t data_cmd_id;//内容id(删除、绘制x个图形，绘制字符)
  uint16_t sender_ID;//发送者id
  uint16_t receiver_ID;//接收者id
} ext_student_interactive_header_data_t;

typedef __packed struct
{
  ext_student_interactive_header_data_t header_data_t;//基础的帧头数据
  uint8_t operate_tpye;//操作：0 空操作 ,1 删除图层 ，2 删除所有
  uint8_t layer;			 //图层 0~9
} ext_client_custom_graphic_delete_t;

typedef __packed struct
{
  uint8_t graphic_name[3];
  uint32_t operate_tpye : 3;//图形操作
  uint32_t graphic_tpye : 3;//图形类别
  uint32_t layer : 4;				//图层数
  uint32_t color : 4;				//颜色
  uint32_t start_angle : 9;	//起始角度
  uint32_t end_angle : 9;		//终止角度
  uint32_t width : 10;			//线宽
  uint32_t start_x : 11;		
  uint32_t start_y : 11;		
  uint32_t radius : 10;//字体大小或半径
  uint32_t end_x : 11;		
  uint32_t end_y : 11;		
} graphic_data_struct_t;//图形数据结构体

//绘制字符
typedef __packed struct
{
  ext_student_interactive_header_data_t header_data_t;
  graphic_data_struct_t grapic_data_struct;
  uint8_t data[30];
} ext_client_custom_character_t;
//绘制1个图形
typedef __packed struct
{
  ext_student_interactive_header_data_t header_data_t;
  graphic_data_struct_t grapic_data_struct;
} ext_client_custom_graphic_single_t;
//绘制2个图形
typedef __packed struct
{
  ext_student_interactive_header_data_t header_data_t;
  graphic_data_struct_t grapic_data_struct[2];
 } ext_client_custom_graphic_double_t;
//绘制5个图形
typedef __packed struct
{
  ext_student_interactive_header_data_t header_data_t;
  graphic_data_struct_t grapic_data_struct[5];
} ext_client_custom_graphic_five_t;

typedef struct
{
  int16_t gx;
  int16_t gy;
  float angle;
  float r;
} tank_ui_t;

extern void draw_get_robot_id(void);
extern void draw_init_all(uint16_t sender_id, uint16_t receiver_id);
extern void clear_all_layer(uint16_t sender_id, uint16_t receiver_id);
extern void draw_shoot_mode(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type);
extern void Leg_Simple_drawing(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type);
extern void draw_capacity_bar(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type);
extern void draw_capacity_bar_deadline(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type);
extern void draw_aim(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type);
extern void draw_chassis_mode(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type);
extern void draw_shoot_circle(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type);
extern void draw_gimbal_relative_angle_tangle(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type);
extern void cal_draw_gimbal_relative_angle_tangle(tank_ui_t *tank);
extern void draw_single_tangle_test(uint16_t sender_id, uint16_t receiver_id, uint16_t op_type);
extern uint16_t send_id;
extern uint16_t receive_id;
#endif
