#ifndef RC_PROTOCAL_H
#define RC_PROTOCAL_H
#include "struct_typedef.h"
#include "main.h"
#include "referee.h"
#include "robot_def.h"
extern referee_t video_cmd;

#define REMOTE_BUFF_SIZE 18

typedef __packed struct
{
    __packed struct
    {
        int16_t ch[5];
        char s[2];
			  int16_t dial;
    } rc;
    __packed struct
    {
        int16_t x;
        int16_t y;
        int16_t z;
        uint8_t press_l;
        uint8_t press_r;
    } mouse;
    __packed struct
    {
        uint16_t v;
    } key;

} RC_ctrl_t;

void Get_RC_ctrl_t(volatile uint8_t *rxBuf);

extern RC_ctrl_t rc_ctrl;
extern uint16_t rc_w_flag;
extern uint16_t rc_s_flag;
extern uint16_t rc_a_flag;
extern uint16_t rc_d_flag;
extern uint16_t rc_q_flag;
extern uint16_t rc_e_flag;
extern uint16_t rc_shift_flag;
extern uint16_t rc_ctrl_flag;
extern uint8_t  rc_press_left;
extern uint8_t  rc_press_right;
extern uint16_t rc_r_flag;
extern uint16_t rc_f_flag;
extern uint16_t rc_g_flag;
extern uint16_t rc_z_flag;
extern uint16_t rc_x_flag;
extern uint16_t rc_c_flag;
extern uint16_t rc_v_flag;
extern uint16_t rc_b_flag;



extern uint16_t w_flag;
extern uint16_t s_flag;
extern uint16_t a_flag;
extern uint16_t d_flag;
extern uint16_t q_flag;
extern uint16_t e_flag;
extern uint16_t shift_flag;
extern uint16_t ctrl_flag;

extern uint16_t r_flag;
extern uint16_t f_flag;
extern uint16_t g_flag;
extern uint16_t z_flag;
extern uint16_t x_flag;
extern uint16_t c_flag;
extern uint16_t v_flag;
extern uint16_t b_flag;
extern uint8_t press_left;
extern uint8_t press_right;


// 全局变量定义
extern KeyComboCounter_t rc_v_counter;
extern KeyComboCounter_t rc_b_counter;
extern KeyComboCounter_t rc_g_counter;
extern KeyComboCounter_t rc_x_counter;
extern KeyComboCounter_t rc_z_counter;
extern KeyComboCounter_t rc_c_counter;
extern KeyComboCounter_t rc_r_counter;
extern KeyComboCounter_t rc_f_counter;

#endif
