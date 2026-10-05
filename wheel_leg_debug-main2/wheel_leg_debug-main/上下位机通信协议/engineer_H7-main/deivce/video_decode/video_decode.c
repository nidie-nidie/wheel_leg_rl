#include "video_decode.h"
//图传链路接收分析 包含自定义控制器


Video_ctrl_t video_ctrl[2]; // 用于存储图传链路的控制数据,[0]:当前数据TEMP,[1]:上一次的数据LAST.用于按键持续按下和切换的判断
//图传接收

static void VideoRead(uint8_t *buff);

/**
 * @brief 图传数据解析函数
 *
 * @param buff 图传数据
 */
static void VideoRead(uint8_t *buff)
{
    uint16_t judge_length; // 统计一帧数据长度
    if (buff == NULL)      // 空数据包，则不作任何处理
        return;
    // 写入帧头数据(5-byte),用于判断是否开始存储裁判数据
    memcpy(&video_ctrl[TEMP].FrameHeader, buff, LEN_HEADER);
    // 判断帧头数据(0)是否为0xA5
    if (buff[SOF] == REFEREE_SOF) {
        // 帧头CRC8校验
        if (Verify_CRC8_Check_Sum(buff, LEN_HEADER) == TRUE) {
            // 统计一帧数据长度(byte),用于CR16校验
            judge_length = buff[DATA_LENGTH] + LEN_HEADER + LEN_CMDID + LEN_TAIL;
            // 帧尾CRC16校验
            if (Verify_CRC16_Check_Sum(buff, judge_length) == TRUE) {
                // 2个8位拼成16位int
                video_ctrl[TEMP].CmdID = (buff[6] << 8 | buff[5]);
                // 解析数据命令码,将数据拷贝到相应结构体中(注意拷贝数据的长度)
                // 第8个字节开始才是数据 data=7
                switch (video_ctrl[TEMP].CmdID) 
                {
                    case ID_custom_robot_data: // 自定义数据
                        video_ctrl[TEMP].custom_control_mode = buff[DATA_Offset];
                        memcpy(&video_ctrl[TEMP].custom_data, (buff + DATA_Offset), LEN_custom_robot_data);
                        memcpy(&video_ctrl[TEMP].cus, &video_ctrl[TEMP].custom_data, 18);

                        break;
                    case ID_remote_control_data: // 图传链路键鼠数据
                        // memcpy(&video_ctrl[TEMP].key_data, (buff + DATA_Offset), LEN_remote_control_data);
                        // *(uint16_t *)&video_ctrl[TEMP].key[KEY_PRESS] = video_ctrl[TEMP].key_data.keyboard_value;
                        // VideoDataContorl();
                        break;
                    default:
                        break;
                }
            }
        }
    }
}




