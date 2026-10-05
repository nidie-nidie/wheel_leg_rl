#ifndef USART_RECEIVE_H
#define USART_RECEIVE_H

#include "struct_typedef.h"

#define USART1_RX_BUF_NUM 56u
#define USER_FRAME_LENGTH 28u

extern void user_usart_init(void);
extern void user_data_pack_handle(void);

__packed typedef struct
{
//	uint8_t header;
//	uint8_t enemy_color;
//	uint8_t aaa;
//	uint8_t bbb;
//	float pitch;
//	float yaw;
//	uint8_t ccc;
//	uint8_t ddd;
//	uint8_t eee;
//	uint8_t end;
//	
  uint8_t head[2];// = {'S', 'P'};
  uint8_t mode;  // 0: 空闲, 1: 自瞄, 2: 小符, 3: 大符
  float q[4];    // wxyz顺序
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;
  uint16_t bullet_count;  // 子弹累计发送次数
  uint8_t tail;// = 0xef;  // 帧尾校验
	
} user_send_data_t;

__packed typedef struct
{
//  uint8_t header;
//	uint8_t mode;
//	float yaw;
//	float pitch;
//	uint8_t aaa;
//	uint8_t bbb;
//	uint8_t ccc;
//	uint8_t ddd;
//	uint8_t eee;
//	uint8_t ender;
//	
	
  uint8_t head[2];// = {'S', 'P'};
  uint8_t mode;  // 0: 不控制, 1: 控制云台但不开火，2: 控制云台且开火
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
  uint8_t tail;// = 0xef;  // 帧尾校验
	
		uint16_t NUC_GG_Detect;
} auto_shoot_t;

extern auto_shoot_t auto_shoot;

#endif
