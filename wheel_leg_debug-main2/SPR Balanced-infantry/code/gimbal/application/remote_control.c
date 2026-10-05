/**
  ****************************(C) COPYRIGHT 2019 DJI****************************
  * @file       remote_control.c/h
  * @brief      遥控器处理，遥控器是通过类似SBUS的协议传输，利用DMA传输方式节约CPU
  *             资源，利用串口空闲中断来拉起处理函数，同时提供一些掉线重启DMA，串口
  *             的方式保证热插拔的稳定性。
  * @note       该任务是通过串口中断启动，不是freeRTOS任务
  * @history
  *  Version    Date            Author          Modification
  *  V1.0.0     Dec-26-2018     RM              1. done
  *  V1.0.0     Nov-11-2019     RM              1. support development board tpye c
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  ****************************(C) COPYRIGHT 2019 DJI****************************
  */

#include "remote_control.h"
#include "detect_task.h"
#include "main.h"

#include "bsp_usart.h"
#include "string.h"
#include "config.h"

#include "config.h"

//遥控器出错数据上限
#define RC_CHANNAL_ERROR_VALUE 700

extern UART_HandleTypeDef huart3;
extern DMA_HandleTypeDef hdma_usart3_rx;

extern uint8_t get_game_status(void);

/**
 * @brief          遥控器协议解析
 * @param[in]      sbus_buf: 原生数据指针
 * @param[out]     rc_ctrl: 遥控器数据指
 * @retval         none
 */
#if PT_link_en
static void sbus_to_pt(volatile const uint8_t *sbus_buf, Remote_data_t *remote_data);
#else
static void sbus_to_rc(volatile const uint8_t *sbus_buf, RC_ctrl_t *rc_ctrl);
#endif

//图传链路控制变量
Remote_data_t remote_data;
//遥控器控制变量
RC_ctrl_t rc_ctrl;

#if PT_link_en
	//接收原始数据，为21个字节，给了42个字节长度，防止DMA传输越界
	uint8_t sbus_rx_buf[2][PT_RX_BUF_NUM];
#else
	//接收原始数据，为18个字节，给了36个字节长度，防止DMA传输越界
	uint8_t sbus_rx_buf[2][SBUS_RX_BUF_NUM];
#endif

/**
 * @brief          遥控器初始化
 * @param[in]      none
 * @retval         none
 */
void remote_control_init(void)
{
	#if PT_link_en
    RC_Init(sbus_rx_buf[0], sbus_rx_buf[1], PT_RX_BUF_NUM);
	#else
    RC_Init(sbus_rx_buf[0], sbus_rx_buf[1], SBUS_RX_BUF_NUM);
	#endif
}
//串口中断
void USART3_IRQHandler(void)
{
    if (huart3.Instance->SR & UART_FLAG_RXNE) //接收到数据
    {
        __HAL_UART_CLEAR_PEFLAG(&huart3);
    }
    else if (USART3->SR & UART_FLAG_IDLE)
    {
        static uint16_t this_time_rx_len = 0;

        __HAL_UART_CLEAR_PEFLAG(&huart3);

        if ((hdma_usart3_rx.Instance->CR & DMA_SxCR_CT) == RESET)
        {
            /* Current memory buffer used is Memory 0 */

            // disable DMA
            //失效DMA
            __HAL_DMA_DISABLE(&hdma_usart3_rx);

            // get receive data length, length = set_data_length - remain_length
            //获取接收数据长度,长度 = 设定长度 - 剩余长度
            this_time_rx_len = SBUS_RX_BUF_NUM - hdma_usart3_rx.Instance->NDTR;
					
				#if PT_link_en
					this_time_rx_len = PT_RX_BUF_NUM - hdma_usart3_rx.Instance->NDTR;//当前长度计算出来为接收数据长度时，即42-21=21，表示接收到了一轮数据
					// reset set_data_lenght
					//重新设定数据长度
					hdma_usart3_rx.Instance->NDTR = PT_RX_BUF_NUM;
				#else
					this_time_rx_len = SBUS_RX_BUF_NUM - hdma_usart3_rx.Instance->NDTR;
					// reset set_data_lenght
					//重新设定数据长度
					hdma_usart3_rx.Instance->NDTR = SBUS_RX_BUF_NUM;
				#endif
					
            // set memory buffer 1
            //设定缓冲区1
            hdma_usart3_rx.Instance->CR |= DMA_SxCR_CT;

            // enable DMA
            //使能DMA
            __HAL_DMA_ENABLE(&hdma_usart3_rx);
		#if PT_link_en
		if(this_time_rx_len == PT_FRAME_LENGTH)//校验长度
		{
			if( sbus_rx_buf[0][0] == 0xA9 && sbus_rx_buf[0][1] == 0x53 )
			{
				//解包
				sbus_to_pt(sbus_rx_buf[0],&remote_data);
				detect_hook(DBUS_TOE);							
			}
		}
		#else
		if(this_time_rx_len == RC_FRAME_LENGTH)
		{
			sbus_to_rc(sbus_rx_buf[0], &rc_ctrl);
			//记录数据接收时间
			detect_hook(DBUS_TOE);
			//sbus_to_usart1(sbus_rx_buf[0]);
		}
		#endif
        }
        else
        {
            /* Current memory buffer used is Memory 1 */
            // disable DMA
            //失效DMA
            __HAL_DMA_DISABLE(&hdma_usart3_rx);

            // get receive data length, length = set_data_length - remain_length
            //获取接收数据长度,长度 = 设定长度 - 剩余长度
            this_time_rx_len = SBUS_RX_BUF_NUM - hdma_usart3_rx.Instance->NDTR;

				#if PT_link_en
					this_time_rx_len = PT_RX_BUF_NUM - hdma_usart3_rx.Instance->NDTR;//当前长度计算出来为接收数据长度时，即42-21=21，表示接收到了一轮数据
					// reset set_data_lenght
					//重新设定数据长度
					hdma_usart3_rx.Instance->NDTR = PT_RX_BUF_NUM;
				#else
					this_time_rx_len = SBUS_RX_BUF_NUM - hdma_usart3_rx.Instance->NDTR;
					// reset set_data_lenght
					//重新设定数据长度
					hdma_usart3_rx.Instance->NDTR = SBUS_RX_BUF_NUM;
				#endif
					
            // set memory buffer 0
            //设定缓冲区0
            DMA1_Stream1->CR &= ~(DMA_SxCR_CT);

            // enable DMA
            //使能DMA
            __HAL_DMA_ENABLE(&hdma_usart3_rx);

		#if PT_link_en
		if(this_time_rx_len == PT_FRAME_LENGTH)//校验长度
		{
			if( sbus_rx_buf[0][0] == 0xA9 && sbus_rx_buf[0][1] == 0x53 )
			{
				//解包
				sbus_to_pt(sbus_rx_buf[0],&remote_data);
				detect_hook(DBUS_TOE);							
			}
		}
		#else
		if(this_time_rx_len == RC_FRAME_LENGTH)
		{
			sbus_to_rc(sbus_rx_buf[0], &rc_ctrl);
			//记录数据接收时间
			detect_hook(DBUS_TOE);
			//sbus_to_usart1(sbus_rx_buf[0]);
		}
		#endif
        }
    }
}

#if PT_link_en
/**
 * @brief          图传链路协议解析
 * @param[in]      sbus_buf: 原生数据指针
 * @param[out]     remote_data: 遥控器数据指针
 * @retval         none
 */
static void sbus_to_pt(volatile const uint8_t *sbus_buf, Remote_data_t *remote_data)
{
    if (sbus_buf == NULL || remote_data == NULL)
    {
        return;
    }

	remote_data->sof_1 = sbus_buf[0];
	remote_data->sof_2 = sbus_buf[1];
    remote_data->ch_0 = ((sbus_buf[2] | (sbus_buf[3] << 8)) & 0x07ff) - PT_CH_VALUE_OFFSET;        //!< Channel 0
    remote_data->ch_1 = (((sbus_buf[3] >> 3) | (sbus_buf[4] << 5)) & 0x07ff) - PT_CH_VALUE_OFFSET; //!< Channel 1
    remote_data->ch_2 = (((sbus_buf[4] >> 6) | (sbus_buf[5] << 2) |          //!< Channel 2
                         (sbus_buf[6] << 10)) &
                        0x07ff) - PT_CH_VALUE_OFFSET;
    remote_data->ch_3 = (((sbus_buf[6] >> 1) | (sbus_buf[7] << 7)) & 0x07ff)- PT_CH_VALUE_OFFSET; //!< Channel 3
		
	remote_data->mode_sw = ((sbus_buf[7] >> 4) & 0x0003);
	remote_data->pause = ((sbus_buf[7] >> 6) & 0x0001);
	remote_data->fn_1 = (sbus_buf[7] >> 7);
	remote_data->fn_2 = (sbus_buf[8] & 0x0001);
		
	remote_data->wheel = (((sbus_buf[8]>>1) | (sbus_buf[9]<<7)) & 0x07ff)- PT_CH_VALUE_OFFSET;
		
	remote_data->trigger = (sbus_buf[9]>>4) & 0x0001;
		
    remote_data->mouse_x = sbus_buf[10] | (sbus_buf[11] << 8);                    //!< Mouse X axis
    remote_data->mouse_y = sbus_buf[12] | (sbus_buf[13] << 8);                    //!< Mouse Y axis
    remote_data->mouse_z = sbus_buf[14] | (sbus_buf[15] << 8);                  //!< Mouse Z axis
		
    remote_data->mouse_left = sbus_buf[16] & 0x0003;                                  //!< Mouse Left Is Press ?
    remote_data->mouse_right = (sbus_buf[16]>>2) & 0x0003;                                  //!< Mouse Right Is Press ?	
    remote_data->mouse_middle = (sbus_buf[16]>>4) & 0x0003; 
		
	remote_data->key = sbus_buf[17] | (sbus_buf[18] << 8);
		
	remote_data->crc16 = sbus_buf[19] | (sbus_buf[20] << 8);
		
}

#else
/**
 * @brief          遥控器协议解析
 * @param[in]      sbus_buf: 原生数据指针
 * @param[out]     rc_ctrl: 遥控器数据指
 * @retval         none
 */
static void sbus_to_rc(volatile const uint8_t *sbus_buf, RC_ctrl_t *rc_ctrl)
{
    if (sbus_buf == NULL || rc_ctrl == NULL)
    {
        return;
    }

    rc_ctrl->rc.ch[0] = (sbus_buf[0] | (sbus_buf[1] << 8)) & 0x07ff;        //!< Channel 0
    rc_ctrl->rc.ch[1] = ((sbus_buf[1] >> 3) | (sbus_buf[2] << 5)) & 0x07ff; //!< Channel 1
    rc_ctrl->rc.ch[2] = ((sbus_buf[2] >> 6) | (sbus_buf[3] << 2) |          //!< Channel 2
                         (sbus_buf[4] << 10)) &
                        0x07ff;
    rc_ctrl->rc.ch[3] = ((sbus_buf[4] >> 1) | (sbus_buf[5] << 7)) & 0x07ff; //!< Channel 3
    rc_ctrl->rc.s[0] = ((sbus_buf[5] >> 4) & 0x0003);                       //!< Switch left
    rc_ctrl->rc.s[1] = ((sbus_buf[5] >> 4) & 0x000C) >> 2;                  //!< Switch right
    rc_ctrl->mouse.x = sbus_buf[6] | (sbus_buf[7] << 8);                    //!< Mouse X axis
    rc_ctrl->mouse.y = sbus_buf[8] | (sbus_buf[9] << 8);                    //!< Mouse Y axis
    rc_ctrl->mouse.z = sbus_buf[10] | (sbus_buf[11] << 8);                  //!< Mouse Z axis
    rc_ctrl->mouse.press_l = sbus_buf[12];                                  //!< Mouse Left Is Press ?
    rc_ctrl->mouse.press_r = sbus_buf[13];                                  //!< Mouse Right Is Press ?
    rc_ctrl->key = sbus_buf[14] | (sbus_buf[15] << 8);                      //!< KeyBoard value
    rc_ctrl->rc.ch[4] = sbus_buf[16] | (sbus_buf[17] << 8);                 // NULL

    rc_ctrl->rc.ch[0] -= RC_CH_VALUE_OFFSET;
    rc_ctrl->rc.ch[1] -= RC_CH_VALUE_OFFSET;
    rc_ctrl->rc.ch[2] -= RC_CH_VALUE_OFFSET;
    rc_ctrl->rc.ch[3] -= RC_CH_VALUE_OFFSET;
    rc_ctrl->rc.ch[4] -= RC_CH_VALUE_OFFSET;
}
#endif
