/**
  ****************************(C) COPYRIGHT 2016 DJI****************************
  * @file       remote_control.c/h
  * @brief      遥控器处理，遥控器是通过类似SBUS的协议传输，利用DMA传输方式节约CPU
  *             资源，利用串口空闲中断来拉起处理函数，同时提供一些掉线重启DMA，串口
  *             的方式保证热插拔的稳定性。
  * @note
  * @history
  *  Version    Date            Author          Modification
  *  V1.0.0     Dec-26-2018     RM              1. done
  *  V1.0.0     Nov-11-2019     RM              1. support development board tpye c
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  ****************************(C) COPYRIGHT 2016 DJI****************************
  */
#ifndef REMOTE_CONTROL_H
#define REMOTE_CONTROL_H
#include "struct_typedef.h"
#include "bsp_rc.h"
#include "config.h"

#define PT_RX_BUF_NUM 42u
#define PT_FRAME_LENGTH 21u
#define PT_CH_VALUE_OFFSET ((uint16_t)1024)

#define SBUS_RX_BUF_NUM 36u
#define RC_FRAME_LENGTH 18u

#define RC_CH_VALUE_MIN ((uint16_t)364)
#define RC_CH_VALUE_OFFSET ((uint16_t)1024)
#define RC_CH_VALUE_MAX ((uint16_t)1684)

/* ----------------------- RC Switch Definition----------------------------- */
#define LEFT_SWITCH 1
#define RIGHT_SWITCH 0


#define PT_SW_UP ((uint16_t)2)
#define PT_SW_MID ((uint16_t)1)
#define PT_SW_DOWN ((uint16_t)0)

#define RC_SW_UP ((uint16_t)1)
#define RC_SW_MID ((uint16_t)3)
#define RC_SW_DOWN ((uint16_t)2)


#define PT_switch_is_down(s) (s == PT_SW_DOWN)
#define PT_switch_is_mid(s) (s == PT_SW_MID)
#define PT_switch_is_up(s) (s == PT_SW_UP)

#define switch_is_down(s) (s == RC_SW_DOWN)
#define switch_is_mid(s) (s == RC_SW_MID)
#define switch_is_up(s) (s == RC_SW_UP)

#define RC_CIRCLE 4

/* ----------------------- PC Key Definition-------------------------------- */
#define KEY_PRESSED_OFFSET_W ((uint16_t)1 << 0)						//前进					//chassis								!
#define KEY_PRESSED_OFFSET_S ((uint16_t)1 << 1)           //后退					//chassis								!
#define KEY_PRESSED_OFFSET_A ((uint16_t)1 << 2)           //左调头				//gimbal_set						！
#define KEY_PRESSED_OFFSET_D ((uint16_t)1 << 3)           //右调头				//gimbal_set						！
#define KEY_PRESSED_OFFSET_SHIFT ((uint16_t)1 << 4)				//小陀螺，A、D左右平移//chassis  				!  
#define KEY_PRESSED_OFFSET_CTRL ((uint16_t)1 << 5)        //压腿刷RFID		//chassis								!
#define KEY_PRESSED_OFFSET_Q ((uint16_t)1 << 6)           //左探头				//chassis								!
#define KEY_PRESSED_OFFSET_E ((uint16_t)1 << 7)           //右探头·				//chassis								!
#define KEY_PRESSED_OFFSET_R ((uint16_t)1 << 8)						//跳跃					//chassis								！
#define KEY_PRESSED_OFFSET_F ((uint16_t)1 << 9)						//伸腿					//chassis								!
#define KEY_PRESSED_OFFSET_G ((uint16_t)1 << 10)					//刷新UI				//referee
#define KEY_PRESSED_OFFSET_Z ((uint16_t)1 << 11)					//自动自救			//chassisAndgimbal			！
#define KEY_PRESSED_OFFSET_X ((uint16_t)1 << 12)					//开关摩擦轮,ctrl+x反转拨盘防卡弹			//shoot									！
#define KEY_PRESSED_OFFSET_C ((uint16_t)1 << 13)					//蹭台阶   			//			！
#define KEY_PRESSED_OFFSET_V ((uint16_t)1 << 14)					//一键转头180		//									!
#define KEY_PRESSED_OFFSET_B ((uint16_t)1 << 15)					//软重启			//	
//ctrl+z：鼠标左右键腿单向旋转，F键变腿长
//滚轮中键用于飞坡
/* ----------------------- Data Struct ------------------------------------- */
typedef __packed struct
{
        __packed struct
        {
                int16_t ch[5];
                char s[2];
        } rc;
        __packed struct
        {
                int16_t x;
                int16_t y;
                int16_t z;
                uint8_t press_l;
                uint8_t press_r;
        } mouse;
        uint16_t key;
} RC_ctrl_t;

//图传链路数据定义
typedef __packed struct
{
    uint8_t sof_1;//帧头
    uint8_t sof_2;
    int64_t ch_0:11;//右摇杆水平方向，364-1024-1684映射到中值为0
    int64_t ch_1:11;//右摇杆竖直
    int64_t ch_2:11;//左摇杆竖直
    int64_t ch_3:11;//左摇杆水平
    uint64_t mode_sw:2;//档位开关
    uint64_t pause:1;//暂停按键
    uint64_t fn_1:1;//自定义按键左
    uint64_t fn_2:1;//自定义按键右
    int64_t wheel:11;//波轮
    uint64_t trigger:1;//扳机

    int16_t mouse_x;//鼠标x,-32678-32678					//yaw轴							//gimbal_set						！			//手动自救两腿反转
    int16_t mouse_y;//鼠标y												//pitch轴						//gimbal_set						！			//手动自救两腿正转
    int16_t mouse_z;//鼠标滚轮																																				//手动自救变腿长
    uint8_t mouse_left:2;//鼠标左									//单发、长按连发		//shoot									！
    uint8_t mouse_right:2;//鼠标右								//自瞄控制头和发射	//shootAndGimbal				！
    uint8_t mouse_middle:2;//鼠标中								
    uint16_t key;
    uint16_t crc16;
}Remote_data_t;


/* ----------------------- Internal Data ----------------------------------- */

extern void remote_control_init(void);
extern bool_t RC_data_is_error(void);
extern void slove_RC_lost(void);
extern void slove_data_error(void);
extern void sbus_to_usart1(uint8_t *sbus);
extern RC_ctrl_t rc_ctrl;
extern Remote_data_t remote_data;
#endif
