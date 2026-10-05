#include "video.h"
//图传链路接收分析 包含自定义控制器
extern ARM_CMD_data_t ARM_CMD_data;

//图传接收
uint8_t uart7_rx_buff[255];
uint8_t uart10_rx_buff[255];

void VIDEO_INIT(void)
{
    HAL_UARTEx_ReceiveToIdle_IT(&huart7,uart7_rx_buff, sizeof(uart7_rx_buff));
	  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
}

void VIDEO_TASK(void)//控制舵机
{
    
    switch (ARM_CMD_data.video_angle) {
        case PITCH_90:
            __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_1, 2130); // 舵机90度 // 2130
            break;
        case PITCH_120:
            __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_1, 2600); // 舵机120度 // 2400
            break;
        default:
            break;
    }

}

