# rm_test v0.1
这是过岗龙战队2026赛季平衡步兵的达妙mc02开发板的源码测试，代码参考[`link`](https://gitee.com/SMBU-POLARBEAR/StandardRobotpp)。具体板子设置参考[`user_guide.pdf`](doc/user_guide.pdf)。

# reference
## 达妙8009p
[上位机软件链接](https://gitee.com/kit-miao/dm-tools)

[上位机调试教程](https://www.bilibili.com/video/BV1TDg7zBEci?vd_source=a61233bf2bcd88af9cb7538da95fa883&spm_id_from=333.788.videopod.sections)

[电机参数手册](https://gitee.com/kit-miao/DM-J8009-2EC/raw/master/%E8%AF%B4%E6%98%8E%E4%B9%A6/DM-J8009-2EC%E5%87%8F%E9%80%9F%E7%94%B5%E6%9C%BA%E8%AF%B4%E6%98%8E%E4%B9%A6V1.0.pdf)

## 瓴控MF9025
[电机参数手册](https://www.scribd.com/document/826568712/%E7%93%B4%E6%8E%A7MF9025)

[上位机软件使用](https://manuals.plus/m/dd70161e09760607cf1a2254c1d36b70587298ffc2b503120c29e109e3686d45_optim.pdf)

**上位机软件链接与电机手册可以参见 `tools` 文件夹**

## 大疆GM6025
[电机参数手册](https://www.robomaster.com/zh-CN/products/components/general/GM6020)

## DM-MC-Board02 电机开发板
[板子资料](https://gitee.com/kit-miao/dm-mc02)

[rm论坛辽科开源](https://bbs.robomaster.com/article/434199)

[SystemView工具使用](https://zhuanlan.zhihu.com/p/540619363)

## 平衡步兵开发
[达妙开源](https://gl1po2nscb.feishu.cn/drive/folder/RJL7fFT4ll9PDSdvM6Pc5vntnPw)

[arm5到arm6迁移问题](https://blog.csdn.net/pingis58/article/details/128340069)

[东莞理工开源报告](https://bbs.robomaster.com/article/728195?source=8)

## 五连杆算法流程
[哈工大开源](https://zhuanlan.zhihu.com/p/563048952)

[matlab仿真](https://www.bilibili.com/video/BV1AS42197jb/)

[mujoco仿真](https://www.bilibili.com/video/BV17GFCzPEk2/)

[上交开源解读](https://www.bilibili.com/video/BV1cw4m1i7fj?spm_id_from=333.788.videopod.sections&vd_source=a61233bf2bcd88af9cb7538da95fa883)

[五连杆运动学解算与VMC](https://zhuanlan.zhihu.com/p/613007726)

[轮腿倒立摆机器人运动速度估计](https://zhuanlan.zhihu.com/p/689921165)

[LQR推导](https://blog.csdn.net/weixin_51772802/article/details/128767706)

[整体工程性思考](https://www.robook.org/blog/dkgzzj)

VMC 倒立摆建模的 phi1 对应前面的关节, phi4 对应后面的关节

## note
机器人方向定义，逆时针旋转为正，以右腿看过去为参考，左腿则相反
- x -> roll, x 轴方向向前
- y -> pitch -> phi, y轴方向向左
- z -> yaw, z 轴方向向上

达妙板子的usb口朝后放安装(CAN2口在前，CAN1口在后), 关节电机can id 设置为左前为1，左后为2，右前为3，右后为6；轮毂电机设置为左边为4，右边为5

电机旋转方向逆时针为正

[大疆中心板](https://rm-static.djicdn.com/tem/17348/RoboMaster%20%20%E7%94%B5%E8%B0%83%E4%B8%AD%E5%BF%83%E6%9D%BF2%20%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.pdf)

达妙电机的can id和master id需要上位机设置，翎控的can id可以上位机设置也可以拨杆，但master id是直接can_id+0x140了

[ps2使用](https://www.bilibili.com/video/BV1pT411a72s/?spm_id_from=333.337.search-card.all.click&vd_source=a61233bf2bcd88af9cb7538da95fa883)

`.old` 文件是一些旧的测试文件，方便学习测试传感器

`tools/matlab_balance` 文件夹下的文件是计算LQR用的，运行 `get_k.m` 即可
```bash
// 参考运行环境
// matlab R2024a
// add ons: Symbolic Math Toolbox
```

仿真也可以参考 [轮足仿真](https://github.com/ziyuliu66/Wheel-legged-robot-MATLAB-simulation-collection.git)

达妙轮足拓展[链接](https://github.com/cyiyang/serial-wheel-leg)

达妙建模[解析](https://blog.csdn.net/Kevin3389179304/article/details/148379498)

右边前面电机对应phi1，后面电机对应phi4；左边后面电机对应phi1，左边前面电机对应phi4

因为开启了D-Cache，dma使用的时候需要声明缓存的储存位置，详细说明见 `main.c`

## trouble
似乎一条can线上接四个达妙电机加两个翎控电机有点不稳定，可能需要调整成两路can控制

PC15的接口需要接5v以上电源才能使用，平常烧录器提供的2.7v电压无法满足

## 遥控器
- **PS2手柄**

使用指南见 `doc/ps2MeArmV1.5.pdf`, 连线如下

PA00 - DI - DAT
PA02 - DO - CMD
PE09 - CS - CS
PE13 - CLK - CLK

控制手册：

1. 数字按键 (Button)

START 键 (data->key == 4)
作用：启停控制开关。第一次按下启动底盘（start_flag = 1），再次按下关闭底盘（start_flag = 0）。

SELECT 键 (data->key == 1)
作用：跳跃准备（预跳跃）。在底盘启动状态下，按一下进入预跳跃状态（prejump_flag = 1），再按一下取消预跳跃。

十字键 - 上 (data->key == 5)
作用：执行跳跃。当处于预跳跃状态（prejump_flag == 1）且当前未在跳跃时，按下此键会触发底层跳跃动作（jump_flag = 1）。

L2 键 (data->key == 9)
作用：一键复位横滚角（Roll角）。无论处于什么姿态，按下会将期望横滚角设为默认值（-0.03f）。

R2 键
作用：底盘关节电机零点标定

L1 键
作用：位控制站起来

R1 键
作用：播放音乐

2. 模拟摇杆 (Joystick)
（前提：必须在 START 键开启的状态下才有效）

右摇杆 Y 轴 (data->ry)
作用：控制底盘的前后移动速度（v_set）及位移。向上推为前进，向下推为后退。

右摇杆 X 轴 (data->rx)
作用：控制底盘的偏航/转向（turn_set）。向左右打杆用于调节机器人的航向角位置。

左摇杆 X 轴 (data->lx)
作用：控制底盘的横滚倾斜角 / Roll 姿态（roll_set）。并做了幅度限幅（-0.40 ~ 0.40）。

左摇杆 Y 轴 (data->ly)
作用：控制底盘的目标腿长 / 高度（leg_set）。并在推动时屏蔽离地检测，防止机器人因为瞬间收腿而产生误判。也是带有限幅的（0.072 ~ 0.21）。

- **DT7手柄**