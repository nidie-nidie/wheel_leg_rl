#include "USART_receive.h"
#include "bsp_usart.h"
#include "cmsis_os.h"
#include "main.h"
#include "detect_task.h"
#include "referee.h"
#include "string.h"
#include "gimbal_task.h"
#include "shoot_task.h"
#include "INS_task.h"
#include "crc8_crc16.h"
#include "INS_task.h"

extern UART_HandleTypeDef huart1;
//直接extern不太好，最好是改为指针和用freertos的函数，赶时间可以临时用
extern gimbal_move_t gimbal_move;
extern shoot_move_t shoot_control;
extern float INS_quat[4];
//接收原始数据，为16个字节，给了32个字节长度，防止DMA传输越界
uint8_t usart1_rx_buf[2][USART1_RX_BUF_NUM];
user_send_data_t user_send_data;
auto_shoot_t auto_shoot;
static void user_data_solve(uint8_t *buf, auto_shoot_t *auto_shoot);

void user_usart_init(void)
{
    usart1_init(usart1_rx_buf[0], usart1_rx_buf[1], USART1_RX_BUF_NUM);
}

void USART1_IRQHandler(void)
{
    if (huart1.Instance->SR & UART_FLAG_RXNE) //接收到数据
    {
        __HAL_UART_CLEAR_PEFLAG(&huart1);
    }
    else if (USART1->SR & UART_FLAG_IDLE)
    {
        static uint16_t this_time_rx_len = 0;

        __HAL_UART_CLEAR_PEFLAG(&huart1);

        if ((huart1.hdmarx->Instance->CR & DMA_SxCR_CT) == RESET)
        {
            /* Current memory buffer used is Memory 0 */
            //失效DMA
            __HAL_DMA_DISABLE(huart1.hdmarx);
            //获取接收数据长度,长度 = 设定长度 - 剩余长度
            this_time_rx_len = USART1_RX_BUF_NUM - huart1.hdmarx->Instance->NDTR;
            //重新设定数据长度
            huart1.hdmarx->Instance->NDTR = USART1_RX_BUF_NUM;
            //设定缓冲区1
            huart1.hdmarx->Instance->CR |= DMA_SxCR_CT;
            //使能DMA
            __HAL_DMA_ENABLE(huart1.hdmarx);

            if (this_time_rx_len == USER_FRAME_LENGTH)
            {
							user_data_solve(usart1_rx_buf[0],&auto_shoot);
            }
        }
        else
        {
            /* Current memory buffer used is Memory 1 */
            //失效DMA
            __HAL_DMA_DISABLE(huart1.hdmarx);
            //获取接收数据长度,长度 = 设定长度 - 剩余长度
            this_time_rx_len = USART1_RX_BUF_NUM - huart1.hdmarx->Instance->NDTR;
            //重新设定数据长度
            huart1.hdmarx->Instance->NDTR = USART1_RX_BUF_NUM;
            //设定缓冲区0
            huart1.hdmarx->Instance->CR &= ~(DMA_SxCR_CT);
            //使能DMA
            __HAL_DMA_ENABLE(huart1.hdmarx);
            if (this_time_rx_len == USER_FRAME_LENGTH)
            {
							user_data_solve(usart1_rx_buf[1],&auto_shoot);
            }
        }
    }
}

void user_data_pack_handle(void)
{
	static uint8_t tx_buf[sizeof(user_send_data_t)];
//	user_send_data.header=0xFF;
//	user_send_data.enemy_color = 1; //蓝
//	user_send_data.aaa=0;
//	user_send_data.bbb=0;
//	user_send_data.ccc=0;
//	user_send_data.ddd=0;
//	user_send_data.eee=0;
//	user_send_data.pitch=-gimbal_move.ins_data.pitch;
//	user_send_data.yaw=-gimbal_move.ins_data.yaw;
//	user_send_data.end = 0xFE;
	user_send_data.head[0]='S';
	user_send_data.head[1]='P';
	user_send_data.mode=1;//先默认请求自瞄
	user_send_data.q[0]=INS_quat[0];
	user_send_data.q[1]=INS_quat[1];
	user_send_data.q[2]=INS_quat[2];
	user_send_data.q[3]=INS_quat[3];
	user_send_data.yaw=gimbal_move.gimbal_yaw_motor.absolute_angle;
	user_send_data.yaw_vel=gimbal_move.gimbal_yaw_motor.absolute_speed;
	user_send_data.pitch=gimbal_move.gimbal_pitch_motor.absolute_angle;
	user_send_data.pitch_vel=gimbal_move.gimbal_pitch_motor.absolute_speed;
	user_send_data.bullet_speed=22.5;
	user_send_data.bullet_count=shoot_control.count;//先一直给零，后续优化
	user_send_data.tail=0xef;
	
  memcpy(tx_buf, &user_send_data,sizeof(user_send_data_t));
	usart1_tx_dma_enable(tx_buf,sizeof(user_send_data_t));
}

void user_data_solve(uint8_t *buf, auto_shoot_t *auto_shoot)
{	
	if (buf[0] == 'S'&& buf[1] == 'P')
	{
		auto_shoot->mode=buf[2];
		memcpy((void*)&auto_shoot->yaw, buf+3, sizeof(float));//从第三个数据开始复制
		memcpy((void*)&auto_shoot->pitch, buf+15,sizeof(float));
		auto_shoot->yaw=-auto_shoot->yaw;
		auto_shoot->pitch=-auto_shoot->pitch;
		
		auto_shoot->NUC_GG_Detect=0;
	}
}
