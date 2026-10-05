/**
  ****************************(C) COPYRIGHT 2019 DJI****************************
  * @file       referee_usart_task.c/h
  * @brief      RM referee system data solve. RM����ϵͳ���ݴ���
  * @note
  * @history
  *  Version    Date            Author          Modification
  *  V1.0.0     Nov-11-2019     RM              1. done
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  ****************************(C) COPYRIGHT 2019 DJI****************************
  */
#include "referee_usart_task.h"
#include "main.h"
#include "cmsis_os.h"

#include "bsp_usart.h"
#include "detect_task.h"

#include "CRC8_CRC16.h"
#include "fifo.h"
#include "protocol.h"
#include "referee.h"
#include "custom_ui_draw.h"

#include "remote_control.h"

/**
  * @brief          单字节解包
  * @param[in]      void
  * @retval         none
  */
static void referee_unpack_fifo_data(void);

extern UART_HandleTypeDef huart6;
uint8_t usart6_buf[2][USART_RX_BUF_LENGHT];

fifo_s_t referee_fifo;
uint8_t referee_fifo_buf[REFEREE_FIFO_BUF_LENGTH];
unpack_data_t referee_unpack_obj;

/**
  * @brief          裁判系统任务
  * @param[in]      pvParameters: NULL
  * @retval         none
  */
uint8_t last_Fn2=0;

void referee_usart_task(void const *argument)
{
    uint16_t time_heart_ms = 0;
	
		init_referee_struct_data();
    fifo_s_init(&referee_fifo, referee_fifo_buf, REFEREE_FIFO_BUF_LENGTH);
    usart6_init(usart6_buf[0], usart6_buf[1], USART_RX_BUF_LENGHT);
    while (1)
    {
			//裁判系统带宽有限，通过空闲中断存数据，运行while解包数据的方式，同时osDelay(10)也为了降低通信频率
			time_heart_ms += 10;
			//解包
			referee_unpack_fifo_data();
			//通过裁判系统ID获得当前机器人画UI需要的send_id和receive_id
			draw_get_robot_id();
			//如果按G就初始化所有静态UI
			if ((!(remote_data.key & KEY_PRESSED_OFFSET_SHIFT)?1:0) && (remote_data.key & KEY_PRESSED_OFFSET_G)?1:0)//(remote_data.fn_2==1&&last_Fn2==0)//
			{
				draw_init_all(send_id,receive_id);
			}
			last_Fn2=remote_data.fn_2;
			
			if(time_heart_ms % 1000 == 0)
			{
			time_heart_ms=0;
			}
			else if(time_heart_ms % 200 == 0)//5Hz
			{
			Leg_Simple_drawing(send_id, receive_id, 2);
			}
			else if(time_heart_ms % 150 == 0)//6Hz
			{
			draw_shoot_mode(send_id, receive_id, 2);
			}
			else if(time_heart_ms % 125 == 0)//10hz
			{
			draw_chassis_mode(send_id, receive_id, 2);
			}
			else if(time_heart_ms % 100 == 0)//10hz
			{
			draw_gimbal_relative_angle_tangle(send_id, receive_id, 2);
			//draw_aim(send_id, receive_id, 2);
			}
			else if(time_heart_ms % 50 == 0)//20Hz
			{
			draw_capacity_bar(send_id, receive_id, 2);
			}
			else if(time_heart_ms % 175 == 0)
			{
				draw_aim(send_id, receive_id, 2);
			}
				
			osDelay(10);
    }
}

/**
  * @brief          单字节解包
  * @param[in]      void
  * @retval         none
  */
void referee_unpack_fifo_data(void)
{
    uint8_t byte = 0;
    uint8_t sof = HEADER_SOF;
    unpack_data_t *p_obj = &referee_unpack_obj;

    while (fifo_s_used(&referee_fifo))
    {
        byte = fifo_s_get(&referee_fifo);
        switch (p_obj->unpack_step)
        {
        case STEP_HEADER_SOF:
        {
            if (byte == sof)
            {
                p_obj->unpack_step = STEP_LENGTH_LOW;
                p_obj->protocol_packet[p_obj->index++] = byte;
            }
            else
            {
                p_obj->index = 0;
            }
        }
        break;

        case STEP_LENGTH_LOW:
        {
            p_obj->data_len = byte;
            p_obj->protocol_packet[p_obj->index++] = byte;
            p_obj->unpack_step = STEP_LENGTH_HIGH;
        }
        break;

        case STEP_LENGTH_HIGH:
        {
            p_obj->data_len |= (byte << 8);
            p_obj->protocol_packet[p_obj->index++] = byte;

            if (p_obj->data_len < (REF_PROTOCOL_FRAME_MAX_SIZE - REF_HEADER_CRC_CMDID_LEN))
            {
                p_obj->unpack_step = STEP_FRAME_SEQ;
            }
            else
            {
                p_obj->unpack_step = STEP_HEADER_SOF;
                p_obj->index = 0;
            }
        }
        break;
        case STEP_FRAME_SEQ:
        {
            p_obj->protocol_packet[p_obj->index++] = byte;
            p_obj->unpack_step = STEP_HEADER_CRC8;
        }
        break;

        case STEP_HEADER_CRC8:
        {
            p_obj->protocol_packet[p_obj->index++] = byte;

            if (p_obj->index == REF_PROTOCOL_HEADER_SIZE)
            {
                if (verify_CRC8_check_sum(p_obj->protocol_packet, REF_PROTOCOL_HEADER_SIZE))
                {
                    p_obj->unpack_step = STEP_DATA_CRC16;
                }
                else
                {
                    p_obj->unpack_step = STEP_HEADER_SOF;
                    p_obj->index = 0;
                }
            }
        }
        break;

        case STEP_DATA_CRC16:
        {
            if (p_obj->index < (REF_HEADER_CRC_CMDID_LEN + p_obj->data_len))
            {
                p_obj->protocol_packet[p_obj->index++] = byte;
            }
            if (p_obj->index >= (REF_HEADER_CRC_CMDID_LEN + p_obj->data_len))
            {
                p_obj->unpack_step = STEP_HEADER_SOF;
                p_obj->index = 0;

                if (verify_CRC16_check_sum(p_obj->protocol_packet, REF_HEADER_CRC_CMDID_LEN + p_obj->data_len))
                {
                    referee_data_solve(p_obj->protocol_packet);
                }
            }
        }
        break;

        default:
        {
            p_obj->unpack_step = STEP_HEADER_SOF;
            p_obj->index = 0;
        }
        break;
        }
    }
}

//存包
void USART6_IRQHandler(void)
{
    static volatile uint8_t res;
    if (USART6->SR & UART_FLAG_IDLE)
    {
        __HAL_UART_CLEAR_PEFLAG(&huart6);

        static uint16_t this_time_rx_len = 0;

        if ((huart6.hdmarx->Instance->CR & DMA_SxCR_CT) == RESET)
        {
            __HAL_DMA_DISABLE(huart6.hdmarx);
            this_time_rx_len = USART_RX_BUF_LENGHT - __HAL_DMA_GET_COUNTER(huart6.hdmarx);
            __HAL_DMA_SET_COUNTER(huart6.hdmarx, USART_RX_BUF_LENGHT);
            huart6.hdmarx->Instance->CR |= DMA_SxCR_CT;
            __HAL_DMA_ENABLE(huart6.hdmarx);
            fifo_s_puts(&referee_fifo, (char *)usart6_buf[0], this_time_rx_len);
            detect_hook(REFEREE_TOE);
        }
        else
        {
            __HAL_DMA_DISABLE(huart6.hdmarx);
            this_time_rx_len = USART_RX_BUF_LENGHT - __HAL_DMA_GET_COUNTER(huart6.hdmarx);
            __HAL_DMA_SET_COUNTER(huart6.hdmarx, USART_RX_BUF_LENGHT);
            huart6.hdmarx->Instance->CR &= ~(DMA_SxCR_CT);
            __HAL_DMA_ENABLE(huart6.hdmarx);
            fifo_s_puts(&referee_fifo, (char *)usart6_buf[1], this_time_rx_len);
            detect_hook(REFEREE_TOE);
        }
    }
}
