#ifndef PS2_TYPEDEF_H__
#define PS2_TYPEDEF_H__

#define PS2_MODE_ERROR 0x00     // 异常模式，手柄工作异常
#define PS2_MODE_DIGITAL 0x41   // 数字模式，只返回按键信息，摇杆无效
#define PS2_MODE_ANALOG 0x73    // 模拟模式，包含按键和摇杆数据
#define PS2_MODE_VIBRATION 0x79 // 模拟模式，加入了手柄马达的控制
#define PS2_MODE_CONFIG 0xF3    // 设置模式，用于配置手柄

#define PS2_IDLE_TIMEOUT 5000 // 5s无操作视为离线

#endif // PS2_TYPEDEF_H__
/*------------------------------ End of File ------------------------------*/
