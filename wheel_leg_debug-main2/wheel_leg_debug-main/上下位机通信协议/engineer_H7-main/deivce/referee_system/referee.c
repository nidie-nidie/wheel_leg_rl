#include "referee.h"

#define RED_HERO 1
#define RED_ENGINEER 2
#define RED_INFANTRY

//电管数据
referee_t dianguan_cmd;

//图传数据
referee_t video_cmd;

//自定义控制器
custom_cmd_t custom_cmd;

__attribute__((at(0x2400CCC6 + 2600))) uint8_t uart1_dma_tx_buf[200] = {0};
static interaction_figure_t figs[100] = {0}; //静态存储100个图形
static uint8_t figs_state[100] = {0};        // 1表示占用，0表示未被占用

void rf_ui_upgrade(referee_t *rf)
{
	
	  
    rf->ui.ui_send.ui_header.SOF = 0xa5;
    rf->ui.ui_send.ui_header.data_length = rf->ui.ui_send.ui_data_len;//数据发送长度 这个应该是45+6
	
//	  Append_CRC8_Check_Sum(&(rf->ui.ui_send.ui_header), rf->ui.ui_send.ui_data_len);//校验头部数据 但是这个为什么是对整体的 存疑
	  Append_CRC8_Check_Sum(&(rf->ui.ui_send.ui_header),sizeof(rf->ui.ui_send.ui_header));//校验头部数据 但是这个为什么是对整体的 存疑
	  //ui_data_len==45 发送字符时
    memcpy(&uart1_dma_tx_buf[0], &rf->ui.ui_send.ui_header, sizeof(frame_header_t));//看样子这个是从后面那个数开始赋值的
	
	
    rf->ui.ui_send.cmd_id = 0x301;//交互数据cmdid
	
    memcpy(&uart1_dma_tx_buf[sizeof(frame_header_t)], rf->ui.ui_send.cmd_id, sizeof(uint16_t));//cmd_id
    memcpy(&uart1_dma_tx_buf[sizeof(frame_header_t) + sizeof(uint16_t)], rf->ui.ui_send.ui_data_buf, rf->ui.ui_send.ui_data_len);//这个是数据帧
	//frameheader+cmdod+databuf 上面是将ui_data_buf放置于uart1_dma_tx_buf 下面这个是一个常规的数据格式header+cmdid+真数据+尾部校验 左CRC校验时要对整体长度做
    Append_CRC16_Check_Sum(uart1_dma_tx_buf, sizeof(frame_header_t) + sizeof(uint16_t) + rf->ui.ui_send.ui_data_len + sizeof(uint16_t)); // header+cmd_id+data[]+crc16
 
	  HAL_UART_Transmit_DMA(&huart7, uart1_dma_tx_buf, sizeof(frame_header_t) + sizeof(uint16_t) + rf->ui.ui_send.ui_data_len + sizeof(uint16_t));
}


void rf_ui_string_upgrade(referee_t *rf)
{
	  rf->ui.ui_send.ui_header.SOF = 0xa5;
    rf->ui.ui_send.ui_header.data_length = 6+45;//数据发送长度 这个应该是45+6
	
	  Append_CRC8_Check_Sum((uint8_t *)&(rf->ui.ui_send.ui_header), sizeof(frame_header_t));//校验头部数据 但是这个为什么是对整体的 存疑

    memcpy(&uart1_dma_tx_buf[0], &rf->ui.ui_send.ui_header, sizeof(frame_header_t));//frameheader
	
    rf->ui.ui_send.cmd_id = 0x301;//交互数据cmdid
	
	
    memcpy(&uart1_dma_tx_buf[sizeof(frame_header_t)], &rf->ui.ui_send.cmd_id, sizeof(uint16_t));//cmd_id
	  
    memcpy(&uart1_dma_tx_buf[sizeof(frame_header_t) + 2], &rf->ui.ui_send.data_cmd_id, sizeof(uint16_t));//子id
	  memcpy(&uart1_dma_tx_buf[sizeof(frame_header_t) + 4], &rf->ui.ui_send.sender_id, sizeof(uint16_t));//senderid
	  memcpy(&uart1_dma_tx_buf[sizeof(frame_header_t) + 6], &rf->ui.ui_send.receiver_id, sizeof(uint16_t));//receiver id
	
	  memcpy(&uart1_dma_tx_buf[sizeof(frame_header_t) + 8], rf->ui.ui_send.ui_data_buf, 45);//这个是数据帧
	
	//frameheader+cmdod+databuf 上面是将ui_data_buf放置于uart1_dma_tx_buf 下面这个是一个常规的数据格式header+cmdid+真数据+尾部校验 左CRC校验时要对整体长度做
    Append_CRC16_Check_Sum(uart1_dma_tx_buf, sizeof(frame_header_t) + 8+45 + sizeof(uint16_t)); // header+cmd_id+data[]+crc16
    
//	  HAL_UART_Transmit_DMA(&huart7, uart1_dma_tx_buf, sizeof(frame_header_t) + 8+45 + sizeof(uint16_t));
	  HAL_UART_Transmit(&huart7,uart1_dma_tx_buf,sizeof(frame_header_t) + 8+45 + sizeof(uint16_t),0xffff); 
}




// figs_num0~99 color0~9 0红/蓝（己方颜色） 1：黄色 2：绿色3：橙色 4：紫红色 5：粉色 6：青色7：黑色8：白色
void rf_ui_write_string(referee_t *rf, char graphname[3],char string_[], uint16_t len, uint16_t size, uint8_t color, int x, int y, int figs_num, ROBOT_ID id)
{
	int i;
	for (i = 0; i < 3 && graphname[i] != '\0'; i++)
	{
		figs[figs_num].figure_name[2-i] =graphname[i];
	}
	

	//fig是储存所有配置的结构体数组 fignum是第几个配置
    if (figs_state[figs_num] == 0) //创建
    {
        figs_state[figs_num] = 1;

			
			  figs[figs_num].color = color;
        figs[figs_num].operate_tpye = 1;          //增加图形
        figs[figs_num].figure_tpye = 7;           //字符图像
        figs[figs_num].layer = 0;                 //字符默认0层
        figs[figs_num].details_a = size;          //字体大小
        figs[figs_num].details_b = len;           //字符串长度
        figs[figs_num].width = 2;              //建议10:1的字体大小和线宽比
        figs[figs_num].start_x = x;
        figs[figs_num].start_y = y;
    }
		
    if (figs_state[figs_num] !=0) //修改
    {
			
			
			  figs[figs_num].color = color;
        figs[figs_num].operate_tpye = 2; //修改图形
        figs[figs_num].figure_tpye = 7;  //字符图像  
        figs[figs_num].layer = 0;        //字符默认0层
        figs[figs_num].details_a = size; //字体大小
        figs[figs_num].details_b = len;  //字符串长度
        figs[figs_num].width = 2;     //建议10:1的字体大小和线宽比
        figs[figs_num].start_x = x;
        figs[figs_num].start_y = y;
    }
		
		
    rf->ui.ui_send.data_cmd_id = 0x110;//对头
		
		
		//下面这个才是需要改的

		if(id==RED_2)
		{
			  rf->ui.ui_send.sender_id = 2;
        rf->ui.ui_send.receiver_id = 0x102;
			
		}
		
		if(id==BLUE_2)
		{
			  rf->ui.ui_send.sender_id = 102;
        rf->ui.ui_send.receiver_id = 0x166;			
		}
		//要接着写
		//其他ID

    rf->ui.ui_send.ui_data_len = 45;//
    memcpy(rf->ui.ui_send.ui_data_buf, &figs[figs_num], sizeof(interaction_figure_t));	
    memcpy(&rf->ui.ui_send.ui_data_buf[15], string_, len);//向真实数据帧中填充 所有要发送的字符
}


void rf_ui_write_float(referee_t *rf, char graphname[3],int32_t	input_float, uint16_t len, uint16_t size, uint8_t color, int x, int y, int figs_num, ROBOT_ID id)
{
	int i;
	for (i = 0; i < 3 && graphname[i] != '\0'; i++)
	{
		figs[figs_num].figure_name[2-i] =graphname[i];
	}
	

	//fig是储存所有配置的结构体数组 fignum是第几个配置
    if (figs_state[figs_num] == 0) //创建
    {
        figs_state[figs_num] = 1;

			
			  figs[figs_num].color = color;
        figs[figs_num].operate_tpye = 1;          //增加图形
        figs[figs_num].figure_tpye = 5;           //浮点图像
        figs[figs_num].layer = 0;                 //字符默认0层
        figs[figs_num].details_a = size;          //字体大小
//        figs[figs_num].details_b = len;           //字符串长度
        figs[figs_num].width = 2;              //建议10:1的字体大小和线宽比
        figs[figs_num].start_x = x;
        figs[figs_num].start_y = y;
			
			  figs[figs_num].details_c = (input_float) & 0x3FF;
			  figs[figs_num].details_d = (input_float >> 10) & 0x7FF;
			  figs[figs_num].details_e = (input_float >> 21) & 0x7FF;
    }
		
    if (figs_state[figs_num] !=0) //修改
    {
			
			
			  figs[figs_num].color = color;
        figs[figs_num].operate_tpye = 2; //修改图形
        figs[figs_num].figure_tpye = 5;  //浮点图像  
        figs[figs_num].layer = 0;        //字符默认0层
        figs[figs_num].details_a = size; //字体大小
//        figs[figs_num].details_b = len;  //字符串长度
        figs[figs_num].width = 2;     //建议10:1的字体大小和线宽比
        figs[figs_num].start_x = x;
        figs[figs_num].start_y = y;
			
			
				figs[figs_num].details_c = (input_float) & 0x3FF;
			  figs[figs_num].details_d = (input_float >> 10) & 0x7FF;
			  figs[figs_num].details_e = (input_float >> 21) & 0x7FF;
    }
		
		
    rf->ui.ui_send.data_cmd_id = 0x110;//对头
		
		
		//下面这个才是需要改的

		if(id==RED_2)
		{
			  rf->ui.ui_send.sender_id = 2;
        rf->ui.ui_send.receiver_id = 0x102;
			
		}
		
		if(id==BLUE_2)
		{
			  rf->ui.ui_send.sender_id = 102;
        rf->ui.ui_send.receiver_id = 0x166;			
		}
		//要接着写
		//其他ID

    rf->ui.ui_send.ui_data_len = 45;//
    memcpy(rf->ui.ui_send.ui_data_buf, &figs[figs_num], sizeof(interaction_figure_t));	
//    memcpy(&rf->ui.ui_send.ui_data_buf[15],input_float , len);//向真实数据帧中填充 所有要发送的字符
}


void referee_fbkdata(referee_t *rf, uint8_t buf[])
{
    int data_len = 0;
    int data_pack_len = 0;
    uint16_t cmd_id = 0;
    if (Verify_CRC8_Check_Sum(buf, sizeof(frame_header_t)))
    {
        memcpy((void *)&(rf->frame_header), (const void *)buf, sizeof(frame_header_t));
        if (rf->frame_header.SOF == 0xa5)
        {
            data_len = rf->frame_header.data_length;
            data_pack_len = 5 + 2 + data_len + 2;
            if (Verify_CRC16_Check_Sum(buf, data_pack_len))
            {
                memcpy((void *)&(cmd_id), (const void *)&buf[sizeof(frame_header_t)], sizeof(uint16_t));
                switch (cmd_id)
                {
                case 0x001:
                    memcpy((void *)&(rf->game_status), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->game_status));
                    break;
                case 0x002:
                    memcpy((void *)&(rf->game_result), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->game_result));
                    break;
                case 0x003:
                    memcpy((void *)&(rf->game_robot_HP), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->game_robot_HP));
                    break;
                case 0x101:
                    memcpy((void *)&(rf->event_data), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->event_data));
                    break;
                case 0x104:
                    memcpy((void *)&(rf->referee_warning), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->referee_warning));
                    break;
                case 0x105:
                    memcpy((void *)&(rf->dart_info), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->dart_info));
                    break;
                case 0x201:
                    memcpy((void *)&(rf->robot_status), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->robot_status));
                    break;
                case 0x202:
                    memcpy((void *)&(rf->power_heat_data), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->power_heat_data));
                    break;
                case 0x203:
                    memcpy((void *)&(rf->robot_pos), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->robot_pos));
                    break;
                case 0x204:
                    memcpy((void *)&(rf->buff), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->buff));
                    break;
                case 0x206:
                    memcpy((void *)&(rf->hurt_data), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->hurt_data));
                    break;
                case 0x207:
                    memcpy((void *)&(rf->shoot_data), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->shoot_data));
                    break;
                case 0x208:
                    memcpy((void *)&(rf->projectile_allowance), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->projectile_allowance));
                    break;
                case 0x209:
                    memcpy((void *)&(rf->rfid_status), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->rfid_status));
                    break;
                case 0x20a:
                    memcpy((void *)&(rf->dart_client_cmd), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->dart_client_cmd));
                    break;
                case 0x20b:
                    memcpy((void *)&(rf->ground_robot_position), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->ground_robot_position));
                    break;
                case 0x20c:
                    memcpy((void *)&(rf->radar_mark_data), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->radar_mark_data));
                    break;
                case 0x20d:
                    memcpy((void *)&(rf->sentry_info), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->sentry_info));
                    break;
                case 0x20e:
                    memcpy((void *)&(rf->radar_info), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->radar_info));
                    break;
                case 0x301:
                    memcpy((void *)&(rf->robot_interaction_data), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->robot_interaction_data));
                    break;
                case 0x302:
                    memcpy((void *)&(rf->custom_robot_data), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->custom_robot_data));
                    break;
                case 0x303:
                    memcpy((void *)&(rf->map_command), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->map_command));
                    break;
                case 0x304:
                    memcpy((void *)&(rf->remote_control), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->remote_control));
                    break;
                case 0x305:
                    memcpy((void *)&(rf->map_robot_data), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->map_robot_data));
                    break;
                case 0x306:
                    memcpy((void *)&(rf->custom_client_data), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->custom_client_data));
                    break;
                case 0x307:
                    memcpy((void *)&(rf->map_data), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->map_data));
                    break;
                case 0x308:
                    memcpy((void *)&(rf->custom_info), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->custom_info));
                    break;
                case 0x309:
                    memcpy((void *)&(rf->robot_custom_data), (const void *)&buf[sizeof(frame_header_t) + sizeof(uint16_t)], sizeof(rf->robot_custom_data));
                    break;
                default:
                    break;
                }
            }
        }
    }
}


uint16_t rf_w_flag;
uint16_t rf_s_flag;
uint16_t rf_a_flag;
uint16_t rf_d_flag;
uint16_t rf_q_flag;
uint16_t rf_e_flag;
uint16_t rf_shift_flag;
uint16_t rf_ctrl_flag;

uint16_t rf_r_flag;
uint16_t rf_f_flag;
uint16_t rf_g_flag;
uint16_t rf_z_flag;
uint16_t rf_x_flag;
uint16_t rf_c_flag;
uint16_t rf_v_flag;
uint16_t rf_b_flag;

uint8_t rf_press_left;
uint8_t rf_press_right;


	

 KeyComboCounter_t rf_v_counter ;

 KeyComboCounter_t rf_b_counter;
 KeyComboCounter_t rf_g_counter ;

 KeyComboCounter_t rf_x_counter ;

 KeyComboCounter_t rf_z_counter;                    

 KeyComboCounter_t rf_c_counter ; 
 KeyComboCounter_t rf_r_counter ;
 KeyComboCounter_t rf_f_counter ;
 
 
 KeyCode_BUFF_t rf_key_code_buff;//



// 全局变量定义
KeyComboCounter_t rf_v_counter = {0,0,0, key_v, key_shift, key_ctrl,0,0,0};                      // 单独v键

KeyComboCounter_t rf_b_counter = {0,0,0, key_b, key_shift, key_ctrl,0,0,0};                      // 单独b键

KeyComboCounter_t rf_g_counter = {0,0,0, key_g, key_shift, key_ctrl,0,0,0};                      // 单独v键

KeyComboCounter_t rf_x_counter = {0,0,0, key_x, key_shift, key_ctrl,0,0,0};                      // 单独b键


KeyComboCounter_t rf_z_counter = {0,0,0, key_z, key_shift, key_ctrl,0,0,0};                      // 单独b键

KeyComboCounter_t rf_c_counter = {0,0,0, key_c, key_shift, key_ctrl,0,0,0};                      // 单独b键

KeyComboCounter_t rf_r_counter = {0,0,0, key_r, key_shift, key_ctrl,0,0,0};                      // 单独b键

KeyComboCounter_t rf_f_counter = {0,0,0, key_f, key_shift, key_ctrl,0,0,0};                      // 单独b键



//是否被按下
static uint8_t is_key_pressed(uint16_t now, uint16_t last, KeyBit_t bit) {
    return ((now >> bit) & 0x01) && !((last >> bit) & 0x01);
}

//是否保持

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
			  counter->single_press_flag=1;
    }
		else{
			counter->single_press_flag=0;//没有被单独按下
		}
		
		if (
        is_key_pressed(status->key_code, status->key_code_last, counter->primary) &&
        is_key_held(status->key_code, counter->shift_key) &&
        !is_key_held(status->key_code, counter->ctrl_key)
    ) {
        counter->shift_press_count++;
		  	counter->shift_press_flag=1;
    }
		else{
			counter->shift_press_flag=0;
		}
		
		if (
        is_key_pressed(status->key_code, status->key_code_last, counter->primary) &&
        !is_key_held(status->key_code, counter->shift_key) &&
        is_key_held(status->key_code, counter->ctrl_key)
    ) {
        counter->ctrl_press_count++;
			counter->ctrl_press_flag=1;
    }
		else
		{
			counter->ctrl_press_flag=0;
		}

		
}

	
//在图传的串口中断中使用
void referee_rc_decode(referee_t *rf)
{
    uint8_t low_bit = (rf->remote_control.keyboard_value) & 0xFF;
	
    rf_w_flag     = (low_bit & 0x01);
    rf_s_flag     = (low_bit & 0x02);
    rf_a_flag     = (low_bit & 0x04);
    rf_d_flag     = (low_bit & 0x08);
    rf_q_flag     = (low_bit & 0x40);
    rf_e_flag     = (low_bit & 0x80);
    rf_shift_flag = (low_bit & 0x10);
    rf_ctrl_flag  = (low_bit & 0x20);

    rf_r_flag = rf->remote_control.keyboard_value & (0x01 << 8);
    rf_f_flag = rf->remote_control.keyboard_value & (0x02 << 8);
    rf_g_flag = rf->remote_control.keyboard_value & (0x04 << 8);
    rf_z_flag = rf->remote_control.keyboard_value & (0x08 << 8);
    rf_x_flag = rf->remote_control.keyboard_value & (0x10 << 8);
    rf_c_flag = rf->remote_control.keyboard_value & (0x20 << 8);
    rf_v_flag = rf->remote_control.keyboard_value & (0x40 << 8);
    rf_b_flag = rf->remote_control.keyboard_value & (0x80 << 8);
	
	  rf_press_left=video_cmd.remote_control.left_button_down;
		rf_press_right=video_cmd.remote_control.right_button_down;
	
	  if (rf_w_flag != 0)
        rf_w_flag = 1;
    if (rf_s_flag != 0)
        rf_s_flag = 1;
    if (rf_a_flag != 0)
        rf_a_flag = 1;
    if (rf_d_flag != 0)
        rf_d_flag = 1;
    if (rf_q_flag != 0)
        rf_q_flag = 1;
    if (rf_e_flag != 0)
        rf_e_flag = 1;
    if (rf_shift_flag != 0)
        rf_shift_flag = 1;
    if (rf_ctrl_flag != 0)
        rf_ctrl_flag = 1;
		
		if (rf_press_left != 0)
        rf_press_left = 1;
    if (rf_press_right != 0)
        rf_press_right = 1;

    if (rf_r_flag != 0)
        rf_r_flag = 1;
    if (rf_f_flag != 0)
        rf_f_flag = 1;
    if (rf_g_flag != 0)
        rf_g_flag = 1;
    if (rf_z_flag != 0)
        rf_z_flag = 1;
    if (rf_x_flag != 0)
        rf_x_flag = 1;
    if (rf_c_flag != 0)
        rf_c_flag = 1;
    if (rf_v_flag != 0)
        rf_v_flag = 1;
    if (rf_b_flag != 0)
        rf_b_flag = 1;

	
		rf_key_code_buff.key_code_last = rf_key_code_buff.key_code;//更新上次的

		rf_key_code_buff.key_code = video_cmd.remote_control.keyboard_value;

	  update_combo_counter(&rf_key_code_buff, &rf_v_counter);
    update_combo_counter(&rf_key_code_buff, &rf_b_counter);	
		update_combo_counter(&rf_key_code_buff, &rf_g_counter);
    update_combo_counter(&rf_key_code_buff, &rf_x_counter);
		update_combo_counter(&rf_key_code_buff, &rf_c_counter);
    update_combo_counter(&rf_key_code_buff, &rf_z_counter);
		update_combo_counter(&rf_key_code_buff, &rf_r_counter);
		update_combo_counter(&rf_key_code_buff, &rf_f_counter);

	
}


//自定义控制器解码
//void custom_decode(uint8_t *buff)
//{
//	   memcpy(&video_cmd.custom_robot_data, (buff + 7), 30);//先把缓冲区的东西赋值

//     memcpy(&custom_cmd, &video_cmd.custom_robot_data, 18);//解码
//}