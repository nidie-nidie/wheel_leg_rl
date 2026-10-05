#ifndef VIDEO_H
#define VIDEO_H


#include "usart.h"

#include "robot_def.h"
#include "bsp_dwt.h"
#include "RC_protocol.h"

#include "string.h"
#include "CRC8_CRC16.h"


// 用于遥控器数据读取,遥控器数据是一个大小为2的数组
#define LAST 1
#define TEMP 0



//extern uint8_t uart7_rx_buff[RE_RX_BUFFER_SIZE];
void VIDEO_INIT(void);

void VIDEO_TASK(void);



#define REFEREE_SOF 0xA5 // 起始字节,协议固定为0xA5
#define Robot_Red 0
#define Robot_Blue 1
#define Communicate_Data_LEN 5 // 自定义交互数据长度，该长度决定了我方发送和他方接收，自定义交互数据协议更改时只需要更改此宏定义即可

/****************************通信协议格式****************************/

/* 通信协议格式偏移，枚举类型,代替#define声明 */
typedef enum
{
	FRAME_HEADER_Offset = 0,
	CMD_ID_Offset = 5,
	DATA_Offset = 7,
} JudgeFrameOffset_e;

/* 通信协议长度 */
typedef enum
{
	LEN_HEADER = 5, // 帧头长
	LEN_CMDID = 2,	// 命令码长度
	LEN_TAIL = 2,	// 帧尾CRC16

	LEN_CRC8 = 4, // 帧头CRC8校验长度=帧头+数据长+包序号
} JudgeFrameLength_e;

/****************************帧头****************************/
/****************************帧头****************************/

/* 帧头偏移 */
typedef enum
{
	SOF = 0,		 // 起始位
	DATA_LENGTH = 1, // 帧内数据长度,根据这个来获取数据长度
	SEQ = 3,		 // 包序号
	CRC8 = 4		 // CRC8
} FrameHeaderOffset_e;

/* 帧头定义 */
typedef struct
{
	uint8_t SOF;
	uint16_t DataLength;
	uint8_t Seq;
	uint8_t CRC8;
} xFrameHeader;


typedef struct
{
    float maximal_arm_target; // 大臂的目标值
    float minimal_arm_target; // 小臂的目标值
    float finesse_target;     // 手腕的目标值
    float pitch_arm_target;   // pitch的目标值
    //float roll_arm_target;    // roll的目标值
    // float maximal_arm_target; // 大臂的目标值
    // float minimal_arm_target; // 小臂的目标值
    // float finesse_target;     // 手腕的目标值
    // float pitch_arm_target;   // pitch的目标值
    // float roll_arm_target;    // roll的目标值
    //float height;           // z轴的目标值
} Custom_contorl_t;

typedef struct
{
    float delta_x;
    float delta_y;
    float delta_z;
    float delta_yaw;
    float delta_pitch;
    float delta_roll;
} slightly_controll_data;

/* ID: 0x0302   Byte: 30    自定义控制器与机器人交互数据 */
//typedef struct
//{
//	uint8_t data[30];
//} custom_robot_data_t;

///* ID: 0x0304   Byte: 12    图传链路键鼠遥控数据 */
//typedef struct
//{
//	int16_t mouse_x;
//	int16_t mouse_y;
//	int16_t mouse_z;
//	uint8_t left_button_down;
//	uint8_t right_button_down;
//	uint16_t keyboard_value;
//	uint16_t reserved;
//} remote_control_t;

typedef struct
{
    xFrameHeader FrameHeader;        // 接收到的帧头信息
    uint16_t CmdID;                  // 命令码
    custom_robot_data_t custom_data; // 自定义数据
    uint8_t custom_control_mode;     // 自定义控制模式
    Custom_contorl_t cus;            // 解算后的自定义数据
    slightly_controll_data scd;      // 轻微控制数据
    remote_control_t key_data;       // 遥控器数据

//    Key_t key[3]; // 改为位域后的键盘索引,空间减少8倍,速度增加16~倍

    uint8_t key_count[3][16];
} Video_ctrl_t;


/* 命令码ID,用来判断接收的是什么数据 */
typedef enum
{
	ID_game_state = 0x0001,				   // 比赛状态数据
	ID_game_result = 0x0002,			   // 比赛结果数据
	ID_game_robot_survivors = 0x0003,	   // 比赛机器人血量数据
	ID_event_data = 0x0101,				   // 场地事件数据
	ID_supply_projectile_action = 0x0102,  // 场地补给站动作标识数据
	ID_supply_projectile_booking = 0x0103, // 场地补给站预约子弹数据
	ID_game_warning = 0x0104,			   // 裁判警告信息
	ID_dart_info = 0x0105,				   // 飞镖发射数据
	ID_game_robot_state = 0x0201,		   // 机器人状态数据
	ID_power_heat_data = 0x0202,		   // 实时功率热量数据
	ID_game_robot_pos = 0x0203,			   // 机器人位置数据
	ID_buff_musk = 0x0204,				   // 机器人增益数据
	ID_aerial_robot_energy = 0x0205,	   // 空中机器人能量状态数据
	ID_robot_hurt = 0x0206,				   // 伤害状态数据
	ID_shoot_data = 0x0207,				   // 实时射击数据
	ID_projectile_allowance = 0x0208,	   // 允许发弹量数据
	ID_rfid_status = 0x0209,			   // RFID状态数据
	// 本demo中默认为步兵机器人，因此未加入以下数据的解析，若需要解析
	// 请自行添加
	ID_dart_client_cmd = 0x020A,	   // 飞镖选手端指令数据
	ID_ground_robot_position = 0x020B, // 地面机器人位置数据
	ID_radar_mark_data = 0x020C,	   // 雷达标记进度数据
	ID_sentry_info = 0x020D,		   // 哨兵自主决策信息同步
	ID_radar_info = 0x020E,			   // 雷达自主决策信息同步
	ID_student_interactive = 0x0301,   // 机器人间交互数据
	// 以下为图传链路数据交互
	ID_custom_robot_data = 0x0302,	 // 自定义机器人数据
	ID_remote_control_data = 0x0304, // 图传链路键鼠数据
} CmdID_e;


/* 命令码数据段长,根据官方协议来定义长度，还有自定义数据长度 */
typedef enum
{
	LEN_game_state = 11,			  // 0x0001
	LEN_game_result = 1,			  // 0x0002
	LEN_game_robot_HP = 32,			  // 0x0003
	LEN_event_data = 4,				  // 0x0101
	LEN_supply_projectile_action = 4, // 0x0102
	LEN_game_warning = 3,			  // 0x0104
	LEN_dart_info = 3,				  // 0x0105
	LEN_game_robot_state = 13,		  // 0x0201
	LEN_power_heat_data = 16,		  // 0x0202
	LEN_game_robot_pos = 16,		  // 0x0203
	LEN_buff_musk = 6,				  // 0x0204
	LEN_aerial_robot_energy = 2,	  // 0x0205
	LEN_robot_hurt = 1,				  // 0x0206
	LEN_shoot_data = 7,				  // 0x0207
	LEN_projectile_allowance = 6,	  // 0x0208
	LEN_rfid_status = 4,			  // 0x0209
	// 本demo中默认为步兵机器人，因此未加入以下数据的解析，若需要解析
	// 请自行添加
	LEN_dart_client_cmd = 6,					 // 0x020A
	LEN_ground_robot_position = 40,				 // 0x020B
	LEN_radar_mark_data = 6,					 // 0x020C
	LEN_sentry_info = 4,						 // 0x020D
	LEN_radar_info = 1,							 // 0x020E
	LEN_receive_data = 6 + Communicate_Data_LEN, // 0x0301
	// 以下为图传链路数据交互
	LEN_custom_robot_data = 30,	  // 0x0302
	LEN_remote_control_data = 12, // 0x0304

} JudgeDataLength_e;

#endif // ARM_H
