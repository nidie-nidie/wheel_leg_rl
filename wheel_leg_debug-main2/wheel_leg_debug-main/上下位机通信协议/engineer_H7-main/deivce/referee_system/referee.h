#ifndef REFEREE_H
#define REFEREE_H

/*
修改日志
2024.12.25
1. 修订命令码 0x0003、0x0101、0x0105、0x0201、0x0202、0x0204、
0x0209、0x020B、0x020C
2. 删除命令码 0x0102、0x0205
3. 新增命令码 0x0309
*/
#include "struct_typedef.h"
#include "CRC8_CRC16.h"
#include "string.h"
#include "usart.h"
#include "dma.h"
#include "robot_def.h"

typedef enum
{
    RED_1 = 1,
    RED_2 = 2,
    RED_3 = 3,
    RED_4 = 4,
    RED_6 = 6,
    RED_7 = 7,
    BLUE_1 = 101,
    BLUE_2 = 102,
    BLUE_3 = 103,
    BLUE_4 = 104,
    BLUE_6 = 106,
    BLUE_7 = 107,
} ROBOT_ID;
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t SOF;
    uint16_t data_length;
    uint8_t seq;
    uint8_t crc8_check_sum;
} frame_header_t;

typedef __packed struct
{
    uint8_t game_type : 4;
    uint8_t game_progress : 4;
    uint16_t stage_remain_time;
    uint64_t SyncTimeStamp;
} game_status_t;
// 0x001
//比赛状态数据，固定以 1Hz 频率发送
//服务器→全体机器人 常规链路
/*
bit 0-3：比赛类型
• 1：RoboMaster 机甲大师超级对抗赛
• 2：RoboMaster 机甲大师高校单项赛
• 3：ICRA RoboMaster 高校人工智能挑战赛
• 4：RoboMaster 机甲大师高校联盟赛 3V3 对抗
• 5：RoboMaster 机甲大师高校联盟赛步兵对抗
bit 4-7：当前比赛阶段
• 0：未开始比赛
• 1：准备阶段
• 2：十五秒裁判系统自检阶段
• 3：五秒倒计时
• 4：比赛中
• 5：比赛结算中
当前阶段剩余时间，单位：秒
UNIX 时间，当机器人正确连接到裁判系统的 NTP 服务器后生效
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t winner;
} game_result_t;
// 0x002
//比赛结果数据，比赛结束触发发送 服务器→全体机器人 常规链路
/*
 0：平局
 1：红方胜利
 2：蓝方胜利
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint16_t red_1_robot_HP;
    uint16_t red_2_robot_HP;
    uint16_t red_3_robot_HP;
    uint16_t red_4_robot_HP;
    uint16_t reserved;
    uint16_t red_7_robot_HP;
    uint16_t red_outpost_HP;
    uint16_t red_base_HP;
    uint16_t blue_1_robot_HP;
    uint16_t blue_2_robot_HP;
    uint16_t blue_3_robot_HP;
    uint16_t blue_4_robot_HP;
    uint16_t reserved1;
    uint16_t blue_7_robot_HP;
    uint16_t blue_outpost_HP;
    uint16_t blue_base_HP;
} game_robot_HP_t;
// 0x0003 32 机器人血量数据，固定以 3Hz 频率发送,服务器→全体机器人 常规链路
/*
红 1 英雄机器人血量。若该机器人未上场或者被罚下，则血量为 0
红 2 工程机器人血量
红 3 步兵机器人血量
红 4 步兵机器人血量
红 5 步兵机器人血量
红 7 哨兵机器人血量
红方前哨站血量
红方基地血量
蓝 1 英雄机器人血量
蓝 2 工程机器人血量
蓝 3 步兵机器人血量
蓝 4 步兵机器人血量
蓝 5 步兵机器人血量
蓝 7 哨兵机器人血量
蓝方前哨站血量
蓝方基地血量
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint32_t event_data;
} event_data_t;
// 0x0101 4 场地事件数据，固定以 1Hz 频率发送 服务器→己方全体机器人 常规链路
/*
......(详见裁判系统串口协议)
*/
/*-------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t level;
    uint8_t offending_robot_id;
    uint8_t count;
} referee_warning_t;
// 0x0104 3 裁判警告数据，己方判罚/判负时触发发送，其余时间以 1Hz 频率发送 服务器→被判罚方全体机器人 常规链路
/*
己方最后一次受到判罚的等级：
 1：双方黄牌
 2：黄牌
 3：红牌
 4：判负

 己方最后一次受到判罚的违规机器人 ID。（如红 1 机器人 ID 为 1，蓝1 机器人 ID 为 101）
 判负和双方黄牌时，该值为 0

己方最后一次受到判罚的违规机器人对应判罚等级的违规次数。（开局默认为 0。）
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t dart_remaining_time;
    uint16_t dart_info;
} dart_info_t;
// 0x0105 3 飞镖发射相关数据，固定以 1Hz频率发送 服务器→己方全体机器人 常规链路
/*
......(详见裁判系统串口协议)
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t robot_id;
    uint8_t robot_level;
    uint16_t current_HP;
    uint16_t maximum_HP;
    uint16_t shooter_barrel_cooling_value;
    uint16_t shooter_barrel_heat_limit;
    uint16_t chassis_power_limit;
    uint8_t power_management_gimbal_output : 1;
    uint8_t power_management_chassis_output : 1;
    uint8_t power_management_shooter_output : 1;
} robot_status_t;
// 0x0201 13 机器人性能体系数据，固定以10Hz频率发送 主控模块→对应机器人 常规链路
/*
本机器人 ID

机器人等级

机器人当前血量

机器人血量上限

机器人枪口热量每秒冷却值

机器人枪口热量上限

机器人底盘功率上限

电源管理模块的输出情况：
 bit 0：gimbal 口输出：0 为无输出，1 为 24V 输出
 bit 1：chassis 口输出：0 为无输出，1 为 24V 输出
 bit 2：shooter 口输出：0 为无输出，1 为 24V 输出
 bit 3-7：保留
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint16_t reserved;
    uint16_t reserved1;
    float reserved2;
    uint16_t buffer_energy;
    uint16_t shooter_17mm_1_barrel_heat;
    uint16_t shooter_17mm_2_barrel_heat;
    uint16_t shooter_42mm_barrel_heat;
} power_heat_data_t;
// 0x0202 16 实时底盘功率和枪口热量数据，固定以50Hz频率发送 主控模块→对应机器人 常规链路
/*
电源管理模块的 chassis 口输出电压（单位：mV）
电源管理模块的 chassis 口输出电流（单位：mA）
底盘功率（单位：W）
被取消

缓冲能量（单位：J）

第 1 个 17mm 发射机构的枪口热量

第 2 个 17mm 发射机构的枪口热量

42mm 发射机构的枪口热量
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    float x;
    float y;
    float angle;
} robot_pos_t;
// 0x0203 16 机器人位置数据，固定以 1Hz 频率发送 主控模块→对应机器人 常规链路
/*
本机器人位置 x 坐标，单位：m

本机器人位置 y 坐标，单位：m

本机器人测速模块的朝向，单位：度。正北为 0 度
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t recovery_buff;
    uint8_t cooling_buff;
    uint8_t defence_buff;
    uint8_t vulnerability_buff;
    uint16_t attack_buff;
    uint8_t remaining_energy;
} buff_t;
// 0x0204 6 机器人增益数据，固定以 3Hz 频率发送 服务器→对应机器人 常规链路
/*
机器人回血增益（百分比，值为 10 表示每秒恢复血量上限的 10%）

机器人枪口冷却倍率（直接值，值为 5 表示 5 倍冷却）

机器人防御增益（百分比，值为 50 表示 50%防御增益）

机器人负防御增益（百分比，值为 30 表示-30%防御增益）

机器人攻击增益（百分比，值为 50 表示 50%攻击增益）
*/
/*-------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t armor_id : 4;
    uint8_t HP_deduction_reason : 4;
} hurt_data_t;
// 0x0206 1 伤害状态数据，伤害发生后发送 主控模块→对应机器人 常规链路
/*
bit 0-3：当扣血原因为装甲模块被弹丸攻击、受撞击、离线或测速模块离线
时，该 4 bit 组成的数值为装甲模块或测速模块的 ID 编号；当其他原因导致
扣血时，该数值为 0

bit 4-7：血量变化类型
 0：装甲模块被弹丸攻击导致扣血
 1：裁判系统重要模块离线导致扣血
 2：射击初速度超限导致扣血
 3：枪口热量超限导致扣血
 4：底盘功率超限导致扣血
 5：装甲模块受到撞击导致扣血
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t bullet_type;
    uint8_t shooter_number;
    uint8_t launching_frequency;
    float initial_speed;
} shoot_data_t;
// 0x0207 7 实时射击数据，弹丸发射后发送 主控模块→对应机器人 常规链路
/*
弹丸类型：
 1：17mm 弹丸
 2：42mm 弹丸

发射机构 ID：
 1：第 1 个 17mm 发射机构
 2：第 2 个 17mm 发射机构
 3：42mm 发射机构

弹丸射速（单位：Hz）

弹丸初速度（单位：m/s）
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint16_t projectile_allowance_17mm;
    uint16_t projectile_allowance_42mm;
    uint16_t remaining_gold_coin;
} projectile_allowance_t;
// 0x0208 6允许发弹量，固定以 10Hz 频率发送 服务器→己方英雄、步兵、哨兵、空中机器人 常规链路
/*
17mm 弹丸允许发弹量

42mm 弹丸允许发弹量

剩余金币数量
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint32_t rfid_status;
} rfid_status_t;
// 0x0209
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t dart_launch_opening_status;
    uint8_t reserved;
    uint16_t target_change_time;
    uint16_t latest_launch_cmd_time;
} dart_client_cmd_t;
// 0x020A 6飞镖选手端指令数据，固定以3Hz 频率发送 服务器→己方飞镖机器人 常规链路
/*
当前飞镖发射站的状态：
 1：关闭
 2：正在开启或者关闭中
 0：已经开启

保留位

切换击打目标时的比赛剩余时间，单位：秒，无/未切换动作，默认为 0。

最后一次操作手确定发射指令时的比赛剩余时间，单位：秒，初始值为 0。
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    float hero_x;
    float hero_y;
    float engineer_x;
    float engineer_y;
    float standard_3_x;
    float standard_3_y;
    float standard_4_x;
    float standard_4_y;
    float reserved;
    float reserved1;
} ground_robot_position_t;
// 0x020B 40地面机器人位置数据，固定以1Hz 频率发送 服务器→己方哨兵机器人 常规链路
/*
场地围挡在红方补给站附近的交点为坐标原点，沿场地长边向蓝方为 X 轴正方向，沿场地短边
向红方停机坪为 Y 轴正方向。

己方英雄机器人位置 x 轴坐标，单位：m
己方英雄机器人位置 y 轴坐标，单位：m
己方工程机器人位置 x 轴坐标，单位：m
己方工程机器人位置 y 轴坐标，单位：m
己方 3 号步兵机器人位置 x 轴坐标，单位：m
己方 3 号步兵机器人位置 y 轴坐标，单位：m
己方 4 号步兵机器人位置 x 轴坐标，单位：m
己方 4 号步兵机器人位置 y 轴坐标，单位：m
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t mark_progress;
} radar_mark_data_t;
// 0x020C 6雷达标记进度数据，固定以 1Hz频率发送 服务器→己方雷达机器人 常规链路
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint32_t sentry_info;
    uint16_t sentry_info_2;
} sentry_info_t;
// 0x020D 6 哨兵自主决策信息同步，固定以1Hz 频率发送 服务器→己方哨兵机器人 常规链路
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t radar_info;
} radar_info_t;
// 0x020E 1雷达自主决策信息同步，固定以1Hz 频率发送 服务器→己方雷达机器人 常规链路
/*------------------------------------------0x301-------------------------------------------------*/
/*
机器人交互数据通过常规链路发送，其数据段包含一个统一的数据段头结构。数据段头结构包括内容 ID、
发送者和接收者的 ID、内容数据段。机器人交互数据包的总长不超过 127 个字节，减去 frame_header、
cmd_id 和 frame_tail 的 9 个字节以及数据段头结构的 6 个字节，故机器人交互数据的内容数据段最大
为 112 个字节。
每 1000 毫秒，英雄、工程、步兵、空中机器人、飞镖能够接收数据的上限为 3720 字节，雷达和哨兵机器
人能够接收数据的上限为 5120 字节。
由于存在多个内容 ID，但整个 cmd_id 上行频率最大为 30Hz，请合理安排带宽。
*/
typedef __packed struct
{
    uint16_t data_cmd_id; //子内容 ID
    uint16_t sender_id;   //发送者 ID
    uint16_t receiver_id; //接收者 ID
    /*
     仅限己方通信
     需为规则允许的多机通讯接收者
     若接收者为选手端，则仅可发送至发送者对应的选手端
     ID 编号详见附录
    */
    uint8_t user_data[112]; //内容数据段，最大为 112
} robot_interaction_data_t;
// 0x0301 127机器人交互数据，发送方触发发送，频率上限为 30Hz (-???发送者接收者不定) 常规链路
/*
子内容 ID       内容数据段长度        功能说明
0x0200~0x02FF   x≤112               机器人之间通信
0x0100          2                   选手端删除图层
0x0101          15                  选手端绘制一个图形
0x0102          30                  选手端绘制两个图形
0x0103          75                  选手端绘制五个图形
0x0104          105                 选手端绘制七个图形
0x0110          45                  选手端绘制字符图形
0x0120          4                   哨兵自主决策指令
0x0121          1                   雷达自主决策指令
*/
/*............................................................*/
typedef __packed struct
{
    uint8_t delete_type;
    uint8_t layer;
} interaction_layer_delete_t;
// 0x0100          2                   选手端删除图层
/*
字节偏移量  大小     说明        备注
0           1       删除操作     0：空操作 1：删除图层 2：删除所有
1           1       图层数      图层数：0~9
*/
/*............................................................*/
typedef __packed struct
{
    uint8_t figure_name[3];
    uint32_t operate_tpye : 3;
    uint32_t figure_tpye : 3;
    uint32_t layer : 4;
    uint32_t color : 4;
    uint32_t details_a : 9;
    uint32_t details_b : 9;
    uint32_t width : 10;
    uint32_t start_x : 11;
    uint32_t start_y : 11;
    uint32_t details_c : 10;
    uint32_t details_d : 11;
    uint32_t details_e : 11;
} interaction_figure_t;
// 0x0101          15                  选手端绘制一个图形
/*
字节偏移量  大小     说明        备注
0           3       图形名      在图形删除、修改等操作中，作为索引
3           4       图形配置1    bit 0-2：图形操作 0：空操作 1：增加 2：修改 3：删除
                                bit 3-5：图形类型0：直线 1：矩形 2：正圆 3：椭圆 4：圆弧 5：浮点数 6：整型数 7：字符
                                bit 6-9：图层数（0~9）
                                bit 10-13：颜色  0：红/蓝（己方颜色） 1：黄色 2：绿色 3：橙色 4：紫红色 5：粉色 6：青色 7：黑色 8：白色
                                bit 14-31：根据绘制的图形不同，含义不同，详见“表 2-26 图形细节参数说明”
7           4       图形配置2   bit 0-9：线宽，建议字体大小与线宽比例为 10：1
                                bit 10-20：起点/圆心 x 坐标
                                bit 21-31：起点/圆心 y 坐标
11          4       图形配置3   根据绘制的图形不同，含义不同，详见“表 2-26 图形细节参数说明”
“表 2-26 图形细节参数说明”
类型        details_a       details_b       details_c       details_d       details_e
直线            -               -               -           终点x坐标       终点y坐标
矩形            -               -               -           对角顶点x坐标   对角顶点y坐标
正圆            -               -               半径            -               -
椭圆            -               -               -            x半轴长度      y半轴长度
圆弧          起始角度        终止角度            -           x半轴长度       y半轴长度
浮点数          字体大小        无作用          (    该值除以 1000 即实际显示值         )
整型数          字体大小        -               (        32 位整型数，int32_t           )
字符            字体大小        字符长度           -               -            -
注意：
 角度值含义为：0°指 12 点钟方向，顺时针绘制；
 屏幕位置：（0,0）为屏幕左下角（1920，1080）为屏幕右上角；
 浮点数：整型数均为 32 位，对于浮点数，实际显示的值为输入的值/1000 如在 details_d、details_c、details_e 对应的字节输入 1234，选手端实际显示的值将为 1.234。
 即使发送的数值超过对应数据类型的限制，图形仍有可能显示，但此时不保证显示的效果。
*/
/*............................................................*/
typedef __packed struct
{
    interaction_figure_t interaction_figure[2];
} interaction_figure_2_t;
// 0x0102          30                  选手端绘制两个图形
/*............................................................*/
typedef __packed struct
{
    interaction_figure_t interaction_figure[5];
} interaction_figure_3_t;
// 0x0103          75                  选手端绘制五个图形
/*............................................................*/
typedef __packed struct
{
    interaction_figure_t interaction_figure[7];
} interaction_figure_4_t;
// 0x0104          105                 选手端绘制七个图形
/*............................................................*/
typedef __packed struct
{
    uint16_t data_cmd_id;                    //子内容 ID 只能为0x110
    uint16_t sender_id;                      //发送者 ID
    uint16_t receiver_id;                    //接收者 ID
    interaction_figure_t interaction_figure; //字符配置 详见图形数据介绍
} graphic_data_struct_t;
typedef __packed struct
{
    graphic_data_struct_t grapic_data_struct;
    uint8_t data[30]; //字符
} ext_client_custom_character_t;
// 0x0110          45                  选手端绘制字符图形
/*............................................................*/
typedef __packed struct
{
    uint32_t sentry_cmd;
} sentry_cmd_t;
// 0x0120          4                   哨兵自主决策指令
/*
bit 0：哨兵机器人是否确认复活
0 表示哨兵机器人确认不复活，即使此时哨兵的复活读条已经完成
1 表示哨兵机器人确认复活，哨兵的复活读条完成立即复活
bit 1：哨兵机器人是否兑换复活
1 表示哨兵机器人确认兑换立即复活，若此时哨兵机器人符合兑换立即复活的规则要求，则会立即消耗金币兑换立即复活
bit 2-12：哨兵将要兑换的发弹量值，开局为 0，修改此值后，哨兵在补血点即可兑换允许发弹量。此值的变化需要单调递增，否则视为不合法。
示例：此值开局仅能为 0，此后哨兵可将其从 0 修改至 X，则消耗 X 金币成功兑换 X 允许发弹量。此后哨兵可将其从 X 修改至X+Y，以此类推。
bit 13-16：哨兵远程兑换发弹量的请求次数，开局为 0，修改此值即可请求远程兑换发弹量。此值的变化需要单调递增且每次仅能增加 1，否则视为不合法。
示例：此值开局仅能为 0，此后哨兵可将其从 0 修改至 1，则消耗金币远程兑换允许发弹量。此后哨兵可将其从 1 修改至 2，以此类推。
bit 17-20：哨兵远程兑换血量的请求次数，开局为 0，修改此值即可请求远程兑换血量。此值的变化需要单调递增且每次仅能增加 1，否则视为不合法。
示例：此值开局仅能为 0，此后哨兵可将其从 0 修改至 1，则消耗金币远程兑换血量。此后哨兵可将其从 1 修改至 2，以此类推。
在哨兵发送该子命令时，服务器将按照从相对低位到相对高位的原则依次处理这些指令，直至全部成功或不能处理为止。
示例：若队伍金币数为 0，此时哨兵战亡，“是否确认复活”的值为 1，“是否确认兑换立即复活”的值为 1，“确认兑换的允
许发弹量值”为 100。（假定之前哨兵未兑换过允许发弹量）由于此时队伍金币数不足以使哨兵兑换立即复活，则服务器将会
忽视后续指令，等待哨兵发送的下一组指令。
bit 21-31：保留
*/
/*............................................................*/
typedef __packed struct
{
    uint8_t radar_cmd;
} radar_cmd_t;
// 0x0121          1                   雷达自主决策指令
/*
开局为 0，修改此值即可请求触发双倍易伤，若此时雷达拥有触发双倍易伤的机会，则可触发。
此值的变化需要单调递增且每次仅能增加 1，否则视为不合法。
示例：此值开局仅能为 0，此后雷达可将其从 0 修改至 1，若雷
达拥有触发双倍易伤的机会，则触发双倍易伤。此后雷达可将其
从 1 修改至 2，以此类推。
若雷达请求双倍易伤时，双倍易伤正在生效，则第二次双倍易
伤将在第一次双倍易伤结束后生效。
*/
/*............................................................*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t data[30]; // x<=30
} custom_robot_data_t;
// 0x0302 30 自定义控制器与机器人交互数据，发送方触发发送，频率上限为 30Hz 自定义控制器→选手端图传连接的机器人 图传链路

/*-------------------------------------------0x303------------------------------------------------*/
// 0x0303 15选手端小地图交互数据，选手端触发发送 选手端点击→服务器→发送方选择的己方机器人 常规链路
/*
@云台手可通过选手端大地图向机器人发送固定数据。命令码为 0x0303，触发时发送，两次发送间隔不得低于 0.5 秒。
发送方式一：
① 点击己方机器人头像；
②（可选）按下一个键盘按键或点击对方机器人头像；
③点击小地图任意位置。该方式向己方选定的机器人发送地图坐标数据，若点击对方机器人头像，则以目标机器人 ID 代替坐标数据。
发送方式二：
①（可选）按下一个键盘按键或点击对方机器人头像；
②点击小地图任意位置。该方式向己方所有机器人发送地图坐标数据，若点击对方机器人头像，则以目标机器人 ID 代替坐标数据。
@选择半自动控制方式的机器人对应的操作手可通过选手端大地图向机器人发送固定数据。命令码为 0x0303，触发时发送，两次发送间隔不得低于 3 秒。
发送方式：
①（可选）按下一个键盘按键或点击对方机器人头像；
②点击小地图任意位置。该方式向操作手对应的机器人发送地图坐标数据，若点击对方机器人头像，则以
目标机器人 ID 代替坐标数据。
一台选择半自动控制方式的机器人既可以接收云台手发送的信息，也可以接收对应操作手的信息。两种信
息的来源将在下表中“信息来源”中进行区别。
------------------------------------
注意：
为降低机器人串口接收设备的偶发不稳定性对通信的影响，0x0303 协议的发送机制有所特殊
处理，具体如下：选手端触发 1 次发送后，服务器将以 100ms 的间隔向机器人额外发送 4
次，共 5 次。此后，直到下一次选手端触发发送前，服务器都将以 1Hz 的频率持续定频发送
最近一次的包。触发时的连续发送和 1Hz 定频发送计时相互独立。队伍需关注多次收到重复
协议内容的处理方式
------------------------------------
*/
typedef __packed struct
{
    float target_position_x;
    float target_position_y;
    uint8_t cmd_keyboard;
    uint8_t target_robot_id;
    uint16_t cmd_source;
} map_command_t;
/*
字节偏移量  大小    说明                                    备注
0           4       目标位置 x 轴坐标，单位 m               当发送目标机器人 ID 时，该值为 0
4           4       目标位置 y 轴坐标，单位 m               当发送目标机器人 ID 时，该值为 0
8           1       云台手按下的键盘按键通用键值                无按键按下，则为 0
9           1       对方机器人 ID                           当发送坐标数据时，该值为 0
10          2       信息来源 ID                             信息来源的 ID，ID 对应关系详见附录
*/
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    int16_t mouse_x;
    int16_t mouse_y;
    int16_t mouse_z;
    char left_button_down;
    char right_button_down;
    uint16_t keyboard_value;
    uint16_t reserved;
} remote_control_t;
// 0x0304 12键鼠遥控数据，固定 30Hz 频率发送 选手端→选手端图传连接的机器人 图传链路
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint16_t hero_position_x;
    uint16_t hero_position_y;
    uint16_t engineer_position_x;
    uint16_t engineer_position_y;
    uint16_t infantry_3_position_x;
    uint16_t infantry_3_position_y;
    uint16_t infantry_4_position_x;
    uint16_t infantry_4_position_y;
    uint16_t infantry_5_position_x;
    uint16_t infantry_5_position_y;
    uint16_t sentry_position_x;
    uint16_t sentry_position_y;
} map_robot_data_t;
// 0x0305 24选手端小地图接收雷达数据，频率上限为5Hz 雷达→服务器→己方所有选手端 常规链路
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint16_t key_value;
    uint16_t x_position : 12;
    uint16_t mouse_left : 4;
    uint16_t y_position : 12;
    uint16_t mouse_right : 4;
    uint16_t reserved;
} custom_client_data_t;
// 0x0306 操作手可使用自定义控制器模拟键鼠操作选手端
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint8_t intention;
    uint16_t start_position_x;
    uint16_t start_position_y;
    int8_t delta_x[49];
    int8_t delta_y[49];
    uint16_t sender_id;
} map_data_t;
// 0x0307 103选手端小地图接收哨兵数据，频率上限为1Hz 哨兵/半自动控制机器人→对应操作手选手端 常规链路
/*-------------------------------------------------------------------------------------------*/
typedef __packed struct
{
    uint16_t sender_id;
    uint16_t receiver_id;
    uint8_t user_data[30];
} custom_info_t;
// 0x0308 34 选手端小地图接收机器人数据，频率上限为 3Hz 己方机器人→己方选手端 常规链路
/*-------------------------------------------------------------------------------------------*/
// 0x0309
//机器人可通过图传链路向对应的操作手选手端连接的自定义控制器发送数据（RMUL 暂不适用）。
typedef __packed struct
{
    uint8_t data[30]; // x<=30
} robot_custom_data_t;

typedef __packed struct
{
    frame_header_t frame_header;
    game_status_t game_status;                       // 0x001
    game_result_t game_result;                       // 0x002
    game_robot_HP_t game_robot_HP;                   // 0x003
    event_data_t event_data;                         // 0x101
    referee_warning_t referee_warning;               // 0x104
    dart_info_t dart_info;                           // 0x105
    robot_status_t robot_status;                     // 0x201
    power_heat_data_t power_heat_data;               // 0x202
    robot_pos_t robot_pos;                           // 0x203
    buff_t buff;                                     // 0x204
    hurt_data_t hurt_data;                           // 0x206
    shoot_data_t shoot_data;                         // 0x207
    projectile_allowance_t projectile_allowance;     // 0x0208
    rfid_status_t rfid_status;                       // 0x209
    dart_client_cmd_t dart_client_cmd;               // 0x20A
    ground_robot_position_t ground_robot_position;   // 0x020B
    radar_mark_data_t radar_mark_data;               // 0x20c
    sentry_info_t sentry_info;                       // 0x020D
    radar_info_t radar_info;                         // 0x020E
    robot_interaction_data_t robot_interaction_data; // 0x301

    sentry_cmd_t sentry_cmd;                 // 0x120
    radar_cmd_t radar_cmd;                   // 0x121
    custom_robot_data_t custom_robot_data;   // 0x302
    map_command_t map_command;               // 0x303
    remote_control_t remote_control;         // 0x0304
    map_robot_data_t map_robot_data;         // 0x0305
    custom_client_data_t custom_client_data; // 0x0306
    map_data_t map_data;                     // 0x0307
    custom_info_t custom_info;               // 0x0308
    robot_custom_data_t robot_custom_data;   // 0x0309
    __packed struct
    {
        __packed struct
        {
            frame_header_t ui_header;
            uint16_t cmd_id;//下面都是数据帧
            uint16_t data_cmd_id;     //子内容 ID
            uint16_t sender_id;       //发送者 ID
            uint16_t receiver_id;     //接收者 ID
            uint8_t ui_data_buf[112]; // 0x301子内容的数据空间 
            int ui_data_len;
        } ui_send;
				
        __packed struct
        {
            interaction_layer_delete_t interaction_layer_delete;       // 0x100
            interaction_figure_t interaction_figure;                   // 0x101 1
            interaction_figure_2_t interaction_figure_2;               // 0x102 2
            interaction_figure_3_t interaction_figure_3;               // 0x103 5
            interaction_figure_4_t interaction_figure_4;               // 0x104 7
            ext_client_custom_character_t ext_client_custom_character; // 0x110选手端绘制字符图形
        } ui_set;
    } ui;
} referee_t;


//
typedef enum
{
	OFF=0,
	ON,
	
}
sucker_switch_cmd_e;//自定义控制器吸盘的控制

typedef enum
{
	LIFT_KEEP=0,//保持
	DOWN,
		UP,
	
}
lift_cmd_e;//自定义控制器抬升的控制

typedef enum
{
	STOP_ROLL=0,//保持
	ZHENG,
		FAN,
	
}
roll_cmd_e;//自定义控制器抬升的控制

/********************************************************/
//自定义控制器数据
typedef struct
{
	  float maximal_arm_target; // 大臂的目标值
    float minimal_arm_target; // 小臂的目标值
    float finesse_target;     // 手腕的目标值
    float pitch_arm_target;   // pitch的目标值
	  roll_cmd_e roll_cmd_direction; //roll目标值
	  sucker_switch_cmd_e sucker_cmd;//吸盘
	  lift_cmd_e lift_cmd;//z轴
	  
	
}custom_cmd_t;
void rf_ui_upgrade(referee_t *rf);
void rf_ui_write_string(referee_t *rf, char graphname[3],char string_[], uint16_t len, uint16_t size, uint8_t color, int x, int y, int figs_num, ROBOT_ID id);
//电管数据
extern referee_t dianguan_cmd;

//图传数据
extern referee_t video_cmd;

//自定义控制器
extern custom_cmd_t custom_cmd;

void referee_fbkdata(referee_t *rf, uint8_t buf[]);
void rf_ui_string_upgrade(referee_t *rf);




extern uint16_t rf_w_flag;
extern uint16_t rf_s_flag;
extern uint16_t rf_a_flag;
extern uint16_t rf_d_flag;
extern uint16_t rf_q_flag;
extern uint16_t rf_e_flag;
extern uint16_t rf_shift_flag;
extern uint16_t rf_ctrl_flag;
extern uint8_t rf_press_left;
extern uint8_t rf_press_right;
extern uint16_t rf_r_flag;
extern uint16_t rf_f_flag;
extern uint16_t rf_g_flag;
extern uint16_t rf_z_flag;
extern uint16_t rf_x_flag;
extern uint16_t rf_c_flag;
extern uint16_t rf_v_flag;
extern uint16_t rf_b_flag;

//图传遥控解码
void referee_rc_decode(referee_t *rf);



extern KeyComboCounter_t rf_v_counter ;

extern KeyComboCounter_t rf_b_counter;
extern KeyComboCounter_t rf_g_counter ;

extern KeyComboCounter_t rf_x_counter ;

extern KeyComboCounter_t rf_z_counter;                    

extern KeyComboCounter_t rf_c_counter ; 
extern KeyComboCounter_t rf_r_counter ;
extern KeyComboCounter_t rf_f_counter ;

#endif
