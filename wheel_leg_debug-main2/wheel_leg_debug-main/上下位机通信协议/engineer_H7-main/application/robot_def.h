#ifndef ROBOT_DEF_H
#define ROBOT_DEF_H

#include "stdint.h"
#define RAD_2_DEGREE 57.2957795f    // 180/pi
#define DEGREE_2_RAD 0.01745329252f // pi/180

#define LIFT_OFFSET (-287.81269f)//想不明白这是什么
#define ROLL_OFFSET 35.07f


#define MOTOR_ENABLE_TIMEOUT_MS 100

#define HEIGHT_MAXIMUN 600
#define HEIGHT_MINIMUN 0
#define REMOTE_CONTROL

#define MAX_HEIGHT 550

//#define TEST_MODE

// 机器人底盘修改的参数,单位为mm(毫米)
#define WHEEL_BASE 462                          // 纵向轴距(前进后退方向)
#define TRACK_WIDTH 380                         // 横向轮距(左右平移方向)
#define RADIUS_WHEEL 77                         // 轮子半径
#define HALF_WHEEL_BASE (WHEEL_BASE / 2.0f)     // 半轴距
#define HALF_TRACK_WIDTH (TRACK_WIDTH / 2.0f)   // 半轮距
#define PERIMETER_WHEEL (RADIUS_WHEEL * 2 * PI) // 轮子周长
#define CENTER_GIMBAL_OFFSET_X 0                // 云台旋转中心距底盘几何中心的距离,前后方向,云台位于正中心时默认设为0
#define CENTER_GIMBAL_OFFSET_Y 0                // 云台旋转中心距底盘几何中心的距离,左右方向,云台位于正中心时默认设为0
#define REDUCTION_RATIO_WHEEL 19.0f             // 电机减速比,因为编码器量测的是转子的速度而不是输出轴的速度故需进行转换
#define SPEED_UP_RATE 200                       // 速度放大倍率,用于调节速度大小
#define SPEED_DOWN_RATE 300                     // 速度缩小倍率,用于调节速度大小

#define MAXARM_MIN    -0.24f
#define MAXARM_MAX    2.3f
#define MINARM_MIN    -2.45f
#define MINARM_MAX    2.4f
#define FINE_MIN      -1.3f // -1.16
#define FINE_MAX      1.7f
#define FINE_MIN2     -1.3f
#define FINE_MAX2     1.7f
#define PITCH_MIN     -1.69f
#define PITCH_MAX     1.00f
//#define HEIGHT_MIN    -550.f
//#define HEIGHT_MAX    560.f // 50
//#define ROLL_MIN      -180.f
//#define ROLL_MAX      180.f

#define LF_CENTER ((HALF_TRACK_WIDTH + CENTER_GIMBAL_OFFSET_X + HALF_WHEEL_BASE - CENTER_GIMBAL_OFFSET_Y) * DEGREE_2_RAD)
#define RF_CENTER ((HALF_TRACK_WIDTH - CENTER_GIMBAL_OFFSET_X + HALF_WHEEL_BASE - CENTER_GIMBAL_OFFSET_Y) * DEGREE_2_RAD)
#define LB_CENTER ((HALF_TRACK_WIDTH + CENTER_GIMBAL_OFFSET_X + HALF_WHEEL_BASE + CENTER_GIMBAL_OFFSET_Y) * DEGREE_2_RAD)
#define RB_CENTER ((HALF_TRACK_WIDTH - CENTER_GIMBAL_OFFSET_X + HALF_WHEEL_BASE + CENTER_GIMBAL_OFFSET_Y) * DEGREE_2_RAD)




#define VAL_LIMIT(val, min, max)     \
    do {                             \
        if ((val) <= (min)) {        \
            (val) = (min);           \
        } else if ((val) >= (max)) { \
            (val) = (max);           \
        }                            \
    } while (0)

typedef enum
{
    NORMAL=0,
    CUSTOM,
    VISION

}ROBOT_STATE_e;//机器人状态		
		


typedef enum
{
    STRAIGHT_OFF=0,
    STRAIGHT_X_ON,
    STRAIGHT_Y_ON

}Chassis_straight_mode_e;

typedef struct 
{
    double vx;
    double vy;
    double vw;
    Chassis_straight_mode_e Chassis_straight_mode;//底盘是否走直线

}Chassis_CMD_data_t;




// 机械臂模式设置
typedef enum
{
    ARM_ZERO_FORCE = 0,   // 电流零输入
    ARM_HUM_CONTORL,      // 自定义控制器控制
    ARM_VISION_CONTROL,   // 视觉控制
    ARM_SLIGHTLY_CONTROL, // 轻微控制
    ARM_KEY_CONTROL,      // 键盘控制
    ARM_AUTO_CONTORL,     // 自动控制
    ARM_LIFT_INIT,        // 高度初始化
} arm_mode_e;

// 机械臂控制状态设置,注意与机械臂模式区分,这里可以看作机械臂模式的子模式
typedef enum
{
    ARM_NORMAL = 0,  // 正常状态,能够被其他模式正常控制
    ARM_RECYCLE,     // 回收状态,机械臂回收到初始位置
    ARM_GETCARROCK,  // 抓取状态,机械臂抓取石块
    ARM_GETCARROCK2, // 抓取状态2,机械臂抓取石块后取出
    ARM_GETROCK,     // 抓取状态,机械臂抓取石块,一键取矿
} arm_status_e;

typedef enum
{
    SUCKER_OFF = 0, // 涵道风机关
    SUCKER_ON,      // 涵道风机开
    SUCKER_HALF     //涵道风机开一点
} sucker_mode_e;

typedef enum
{
    LIFT_OFF = 0,    // 机械臂升降关闭
    LIFT_HEIGHT_MODE, // 机械臂升降高度模式
    LIFT_SPEED_MODE, // 机械臂升降速度模式
    LIFT_KEEP_MODE,       // 机械臂升降保持模式

} lift_mode_e;

typedef enum
{
    ROLL_OFF = 0,    // 机械臂roll关闭
    ROLL_ANGLE_MODE, // 机械臂roll开
    ROLL_SPEED_MODE, // 机械臂roll开
    ROLL_KEEP_MODE,       // 机械臂roll保持
} roll_mode_e;

typedef enum
{
    DOWNLOAD_OFF = 0, // 关闭调试模式
    DOWNLOAD_ON,      // 开启调试模式,用于控制大疆电机的调试，防止下载时电机转动
} download_mode_e;

typedef enum
{
    PITCH_0   = 0,
    PITCH_30  = 30,  // 30度
    PITCH_60  = 60,  // 60度
    PITCH_90  = 90,  // 90度
    PITCH_120 = 120, // 120度
    PITCH_150 = 150, // 150度
    PITCH_180 = 180, // 180度
} video_angle_e;

typedef enum
{   
	  STOP=0,
    GET,//取
    OUTPUT,//兑
} trans_mode_e;

typedef enum
{
    NEED=0,//要解算
  	NONEED,//不解算
} need_to_tranverse_e;//是否需要解算

typedef enum
{
    CH_ENABLE=0,//使能
	CH_DISABLE,//不使能
} chassis_enable_e;//是否使能底盘



typedef enum
{
    HANDLE=0,//手动模式
	GET_SILVER,//自动取银矿  
	AUTO_OUTPUT,//自动从矿仓取矿
	AUTO_PUT,//自动放矿
	
	
} auto_mode_e;//是否使能底盘

typedef struct
{
    float maximal_arm_angle;             // 机械臂大臂目标角度
    float minimal_arm_angle;             // 机械臂小臂目标角度
    float finesse_angle;                 // 机械臂手腕目标角度
    float pitch_arm_angle;               // 机械臂pitch目标角度
    float lift_speed;                    // 机械臂抬升速度
    float lift_height;                   // 机械臂抬升高度
    float roll_angle;                    // 机械臂roll目标角度
    lift_mode_e lift_mode;         // 机械臂上升标志
    lift_mode_e last_lift_mode;         // 机械臂上升标志*(上一次)
    roll_mode_e roll_mode;         // 机械臂roll标志
    sucker_mode_e sucker_mode;     // 涵道风机状态
    arm_mode_e arm_mode;           // 机械臂状态
    arm_mode_e arm_mode_last;      // 机械臂上一次状态
    arm_status_e arm_status;       // 机械臂控制状态(状态子模式)
    download_mode_e download_mode; // 下载模式
    video_angle_e video_angle;     // 图传角度,理论上不应该在这，但没必要专门为了一个舵机开一个topic
    trans_mode_e trans_mode;     //传送带是否反转
	
	  need_to_tranverse_e if_tranverse;//是否解算
	
	  chassis_enable_e  chassis_enable;//底盘是否使能
		
		auto_mode_e auto_mode;//自动模式
		
	
} ARM_CMD_data_t;



#define NONE     100
#define key_s     1
#define key_a     2
#define key_d     3
#define key_shift 4
#define key_ctrl  5
#define key_q     6
#define key_e     7
#define key_r     8
#define key_f     9
#define key_g     10
#define key_z     11
#define key_x     12
#define key_c     13
#define key_v     14
#define key_b     15

typedef enum {

    Key_S = 1,
    Key_A = 2,
    Key_D = 3,
    Key_Shift = 4,
    Key_Ctrl = 5,
    Key_Q = 6,
    Key_E = 7,
    Key_R = 8,
    Key_F = 9,
    Key_G = 10,
    Key_Z = 11,
    Key_X = 12,
    Key_C = 13,
    Key_V = 14,
    Key_B = 15
    
} KeyBit_t;



typedef struct {
    uint16_t key_code;       // 当前键码
    uint16_t key_code_last;  // 上一次键码
} KeyCode_BUFF_t;

typedef struct {
    uint16_t single_press_count;
	  uint16_t shift_press_count;
	  uint16_t ctrl_press_count;
	  
    KeyBit_t primary;//基础按键
    KeyBit_t shift_key;//修饰按键
    KeyBit_t ctrl_key;
	
	  uint8_t single_press_flag;
	  uint8_t shift_press_flag;
	  uint8_t ctrl_press_flag;
	
} KeyComboCounter_t;//修饰计数

#endif
