#include "RC_protocol.h"

uint16_t w_flag;
uint16_t s_flag;
uint16_t a_flag;
uint16_t d_flag;
uint16_t q_flag;
uint16_t e_flag;
uint16_t shift_flag;
uint16_t ctrl_flag;

uint16_t r_flag;
uint16_t f_flag;
uint16_t g_flag;
uint16_t z_flag;
uint16_t x_flag;
uint16_t c_flag;
uint16_t v_flag;
uint16_t b_flag;

uint8_t press_left;
uint8_t press_right;




uint16_t rc_w_flag;
uint16_t rc_s_flag;
uint16_t rc_a_flag;
uint16_t rc_d_flag;
uint16_t rc_q_flag;
uint16_t rc_e_flag;
uint16_t rc_shift_flag;
uint16_t rc_ctrl_flag;

uint16_t rc_r_flag;
uint16_t rc_f_flag;
uint16_t rc_g_flag;
uint16_t rc_z_flag;
uint16_t rc_x_flag;
uint16_t rc_c_flag;
uint16_t rc_v_flag;
uint16_t rc_b_flag;

RC_ctrl_t rc_ctrl;

uint8_t rc_press_left;
uint8_t rc_press_right;

// 自己定义键盘
KeyCode_BUFF_t key_code_buff;

// 全局变量定义
KeyComboCounter_t rc_v_counter = {0,0,0, key_v, key_shift, key_ctrl,0,0,0}; // 单独v键
KeyComboCounter_t rc_b_counter = {0,0,0, key_b, key_shift, key_ctrl,0,0,0}; // 单独b键
KeyComboCounter_t rc_g_counter = {0,0,0, key_g, key_shift, key_ctrl,0,0,0}; // 单独g键
KeyComboCounter_t rc_x_counter = {0,0,0, key_x, key_shift, key_ctrl,0,0,0}; // 单独x键
KeyComboCounter_t rc_z_counter = {0,0,0, key_z, key_shift, key_ctrl,0,0,0}; // 单独z键
KeyComboCounter_t rc_c_counter = {0,0,0, key_c, key_shift, key_ctrl,0,0,0}; // 单独c键
KeyComboCounter_t rc_r_counter = {0,0,0, key_r, key_shift, key_ctrl,0,0,0}; // 单独r键
KeyComboCounter_t rc_f_counter = {0,0,0, key_f, key_shift, key_ctrl,0,0,0}; // 单独f键

uint8_t is_key_pressed(uint16_t now, uint16_t last, KeyBit_t bit) {
    return ((now >> bit) & 0x01) && !((last >> bit) & 0x01);
}

static uint8_t is_key_held(uint16_t now, KeyBit_t bit) {
    return (now >> bit) & 0x01;
}

static void update_combo_counter(KeyCode_BUFF_t *status, KeyComboCounter_t *counter) {
    if (
        is_key_pressed(status->key_code, status->key_code_last, counter->primary) &&
        !is_key_held(status->key_code, counter->shift_key) &&
        !is_key_held(status->key_code, counter->ctrl_key)
    ) {
        counter->single_press_count++;
        counter->single_press_flag = 1;
    } else {
        counter->single_press_flag = 0;
    }

    if (
        is_key_pressed(status->key_code, status->key_code_last, counter->primary) &&
        is_key_held(status->key_code, counter->shift_key) &&
        !is_key_held(status->key_code, counter->ctrl_key)
    ) {
        counter->shift_press_count++;
        counter->shift_press_flag = 1;
    } else {
        counter->shift_press_flag = 0;
    }

    if (
        is_key_pressed(status->key_code, status->key_code_last, counter->primary) &&
        !is_key_held(status->key_code, counter->shift_key) &&
        is_key_held(status->key_code, counter->ctrl_key)
    ) {
        counter->ctrl_press_count++;
        counter->ctrl_press_flag = 1;
    } else {
        counter->ctrl_press_flag = 0;
    }
}

#define RC_CH_VALUE_OFFSET ((uint16_t)1024)
void Get_RC_ctrl_t(volatile uint8_t *rxBuf)
{
    rc_ctrl.rc.ch[0] = (rxBuf[0] | (rxBuf[1] << 8)) & 0x07ff;
    rc_ctrl.rc.ch[1] = (((rxBuf[1] >> 3) & 0xff) | (rxBuf[2] << 5)) & 0x07ff;
    rc_ctrl.rc.ch[2] = (((rxBuf[2] >> 6) & 0xff) | (rxBuf[3] << 2) |
                        (rxBuf[4] << 10)) & 0x07ff;
    rc_ctrl.rc.ch[3] = (((rxBuf[4] >> 1) & 0xff) | (rxBuf[5] << 7)) & 0x07ff;
    rc_ctrl.rc.s[0] = ((rxBuf[5] >> 4) & 0x0003);
    rc_ctrl.rc.s[1] = ((rxBuf[5] >> 4) & 0x000C) >> 2;
    rc_ctrl.mouse.x = rxBuf[6] | (rxBuf[7] << 8);
    rc_ctrl.mouse.y = rxBuf[8] | (rxBuf[9] << 8);
    rc_ctrl.mouse.z = rxBuf[10] | (rxBuf[11] << 8);
    rc_ctrl.mouse.press_l = rxBuf[12];
    rc_ctrl.mouse.press_r = rxBuf[13];
    rc_ctrl.key.v = rxBuf[14] | (rxBuf[15] << 8);
    rc_ctrl.rc.ch[4] = rxBuf[16] | (rxBuf[17] << 8);
    rc_ctrl.rc.dial = ((rxBuf[16] | (rxBuf[17] << 8)) & 0x07FF) - RC_CH_VALUE_OFFSET;

    rc_ctrl.rc.ch[0] -= RC_CH_VALUE_OFFSET;
    rc_ctrl.rc.ch[1] -= RC_CH_VALUE_OFFSET;
    rc_ctrl.rc.ch[2] -= RC_CH_VALUE_OFFSET;
    rc_ctrl.rc.ch[3] -= RC_CH_VALUE_OFFSET;
    rc_ctrl.rc.ch[4] -= RC_CH_VALUE_OFFSET;

    rc_w_flag = (rxBuf[14] & 0x01);
    rc_s_flag = (rxBuf[14] & 0x02);
    rc_a_flag = (rxBuf[14] & 0x04);
    rc_d_flag = (rxBuf[14] & 0x08);
    rc_q_flag = (rxBuf[14] & 0x40);
    rc_e_flag = (rxBuf[14] & 0x80);
    rc_shift_flag = (rxBuf[14] & 0x10);
    rc_ctrl_flag = (rxBuf[14] & 0x20);

    rc_press_left = rc_ctrl.mouse.press_l;
    rc_press_right = rc_ctrl.mouse.press_r;

    rc_r_flag = rc_ctrl.key.v & (0x00 | 0x01 << 8);
    rc_f_flag = rc_ctrl.key.v & (0x00 | 0x02 << 8);
    rc_g_flag = rc_ctrl.key.v & (0x00 | 0x04 << 8);
    rc_z_flag = rc_ctrl.key.v & (0x00 | 0x08 << 8);
    rc_x_flag = rc_ctrl.key.v & (0x00 | 0x10 << 8);
    rc_c_flag = rc_ctrl.key.v & (0x00 | 0x20 << 8);
    rc_v_flag = rc_ctrl.key.v & (0x00 | 0x40 << 8);
    rc_b_flag = rc_ctrl.key.v & (0x00 | 0x80 << 8);

    if (rc_w_flag) rc_w_flag = 1;
    if (rc_s_flag) rc_s_flag = 1;
    if (rc_a_flag) rc_a_flag = 1;
    if (rc_d_flag) rc_d_flag = 1;
    if (rc_q_flag) rc_q_flag = 1;
    if (rc_e_flag) rc_e_flag = 1;
    if (rc_shift_flag) rc_shift_flag = 1;
    if (rc_ctrl_flag) rc_ctrl_flag = 1;
    if (rc_press_left) rc_press_left = 1;
		if (rc_press_right) rc_press_right = 1;
    if (rc_r_flag) rc_r_flag = 1;
    if (rc_f_flag) rc_f_flag = 1;
    if (rc_g_flag) rc_g_flag = 1;
    if (rc_z_flag) rc_z_flag = 1;
    if (rc_x_flag) rc_x_flag = 1;
    if (rc_c_flag) rc_c_flag = 1;
    if (rc_v_flag) rc_v_flag = 1;
    if (rc_b_flag) rc_b_flag = 1;

    key_code_buff.key_code_last = key_code_buff.key_code;
    key_code_buff.key_code = rc_ctrl.key.v;

update_combo_counter(&key_code_buff, &rc_v_counter);
update_combo_counter(&key_code_buff, &rc_b_counter);
update_combo_counter(&key_code_buff, &rc_g_counter);
update_combo_counter(&key_code_buff, &rc_x_counter);
update_combo_counter(&key_code_buff, &rc_c_counter);
update_combo_counter(&key_code_buff, &rc_z_counter);
update_combo_counter(&key_code_buff, &rc_r_counter);
update_combo_counter(&key_code_buff, &rc_f_counter);
}
