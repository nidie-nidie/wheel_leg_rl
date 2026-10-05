# MuJoCo 仿真接入流程

这份工程现在的 MuJoCo 接入方式是：

```text
MuJoCo 模型状态
        ↓
sim/main_mujoco.c 读取 qpos/qvel/base 姿态
        ↓
SimControllerState
        ↓
sim/sim_adapter.c 写入原工程的 INS / 电机反馈变量
        ↓
ChassisL_control_loop() / ChassisR_control_loop()
        ↓
SimControllerOutput
        ↓
sim/main_mujoco.c 写入 MuJoCo actuator ctrl
        ↓
mj_step()
```

也就是说，MuJoCo 负责物理仿真，原来的底盘控制代码负责算力矩。

## 1. 当前控制状态

当前仿真不是从完整状态机开机流程开始跑，而是直接进入正常平衡控制：

```c
chassis_move.start_flag = 1;
chassis_move.mode = CHASSIS_SAFE;
```

这里的 `CHASSIS_SAFE` 在当前工程里名字有点误导。它不是无力安全状态，而是正常控制输出状态。

更准确的理解是：

```text
CHASSIS_OFF       失能/关闭
CHASSIS_CALIBRATE 校准
CHASSIS_STAND_UP  起身过渡动作
CHASSIS_SAFE      正常站立平衡控制
CHASSIS_OFF_HOOK  脱困/自恢复，当前未完整接入
```

当前仿真路径直接跳过：

```text
OFF -> CALIBRATE -> STAND_UP -> SAFE
```

而是：

```text
启动仿真 -> 直接 CHASSIS_SAFE 平衡控制
```

这样做是为了先把 MuJoCo 闭环打通，避免状态机、遥控器、校准、起身动作和控制参数混在一起调。

## 2. 文件分工

主要文件：

```text
sim/CMakeLists.txt
    仿真工程编译入口。

sim/main_mujoco.c
    MuJoCo bridge 主程序。
    负责加载模型、创建 GUI、读取 MuJoCo 状态、调用控制器、写入力矩。

sim/sim_adapter.h
    MuJoCo 和原控制器之间的数据接口定义。

sim/sim_adapter.c
    把 SimControllerState 写入原工程变量。
    调用 ChassisL/ChassisR 控制函数。
    把原控制器输出整理成 SimControllerOutput。

sim/models/rm_robot.xml
    当前用于 bridge 的 MuJoCo 模型。
    需要包含 freejoint、地面、actuator。

Application/Task/Src/ChassisL_Task.c
Application/Task/Src/ChassisR_Task.c
    原来的左右腿控制逻辑。
```

## 3. 模型要求

`mujoco_bridge` 当前要求模型里有这些 body / joint / actuator 名字：

```text
body:
base_link

freejoint:
base_freejoint

joint:
left_leg
left_small_leg
right_leg
right_small_leg
left_wheel
right_wheel

actuator:
left_leg_motor
left_small_leg_motor
right_leg_motor
right_small_leg_motor
left_wheel_motor
right_wheel_motor
```

控制器数组映射关系：

```text
joint_pos[0] / joint_torque[0] -> left_leg
joint_pos[1] / joint_torque[1] -> left_small_leg

joint_pos[2] / joint_torque[2] -> right_leg
joint_pos[3] / joint_torque[3] -> right_small_leg

wheel_vel[0] / wheel_torque[0] -> left_wheel
wheel_vel[1] / wheel_torque[1] -> right_wheel
```

如果使用原始 URDF/MJCF：

```bash
simulate ~/MuJoCoBin/urdf13.SLDASM/urdf/mjmodel.xml
```

这只能说明模型可以被 MuJoCo 打开，不代表控制器已经接入。真正接入控制器要运行 `mujoco_bridge`。

## 4. 从零开始编译

进入工程目录：

```bash
cd /home/shun/MuJoCoBin/rm_control/mujoco_control_extract
```

确认 MuJoCo 安装目录：

```bash
ls ../third_party/mujoco
```

如果你希望换电脑后少改路径，推荐把 MuJoCo SDK 放到仓库顶层的 `third_party/mujoco/`。至少需要有：

```text
third_party/mujoco/include/mujoco/mujoco.h
third_party/mujoco/lib/libmujoco.so
```

如果 SDK 里是 `libmujoco.so.3.3.0` 这种版本化文件，也可以直接保留原名。

配置 CMake：

（意思是读取 cmake list 然后找到mujoco 中的头文件和库，并且生成 make file。）
```bash
cmake -S sim -B build
```

如果 MuJoCo 不在 `third_party/mujoco`，可以显式指定：

```bash
cmake -DMUJOCO_ROOT=/path/to/mujoco -S sim -B build
```

编译：

```bash
cmake --build build -j
```

成功时会看到：

```text
[100%] Built target mujoco_bridge
```

编译出的程序是：

```text
build/mujoco_bridge
```

以后如果只是改了 C 文件，通常只需要：

```bash
cmake --build build -j
```

如果改了 `sim/CMakeLists.txt`，建议重新配置再编译：

```bash
cmake -S sim -B build
cmake --build build -j
```

一行写法也可以：

```bash
cmake -S sim -B build && cmake --build build -j
```

注意不要把两条命令粘成这样：

```bash
cmake -S sim -B buildcmake --build build -j
```

这会导致 CMake 把 `-j` 当成配置参数，报 `Unknown argument -j`。


## 5. 运行仿真

### GUI 模式

默认运行会打开图形窗口：

```bash
./build/mujoco_bridge sim/models/rm_robot.xml
```

窗口操作：

```text
鼠标左键拖动：旋转视角
鼠标右键拖动：平移视角
滚轮：缩放
ESC：退出
```

如果在 WSL 里运行，需要 WSLg 或 X Server 正常工作。你如果能运行 `simulate xxx.xml`，一般说明图形环境已经可用。

### Headless 模式

如果只想跑命令行输出，不开窗口：

```bash
./build/mujoco_bridge --headless sim/models/rm_robot.xml
```

这会尽可能快地跑完仿真，不会按现实时间等待。

## 6. 输出数据含义

运行时会打印类似：

```text
t= 0.501 x= 0.515 pitch= 0.995 u=[-2.61 -5.85  1.66  5.69 | -4.47  4.21]
```

含义：

```text
t
    MuJoCo 仿真时间，单位 s。
    不是现实世界等待时间。

x
    base_link 在世界坐标 x 方向的位置，单位 m。

pitch
    base_link 俯仰角，单位 rad。
    0 表示前后不倾。
    1.57 rad 约等于 90 度。

u[0]
    left_leg_motor 力矩。

u[1]
    left_small_leg_motor 力矩。

u[2]
    right_leg_motor 力矩。

u[3]
    right_small_leg_motor 力矩。

u[4]
    left_wheel_motor 力矩。

u[5]
    right_wheel_motor 力矩。
```

如果看到力矩突然到几百甚至上千，例如：

```text
u=[ 941.68 -950.60 ...]
```

说明当前映射、零点、模型姿态或限幅还没有整理好，不建议马上调 PID/LQR 参数。


## 8. 当前控制循环

`sim/main_mujoco.c` 每一步做：

```c
read_state(m, d, map, &state);
SimController_SetState(&state);
SimController_SetCommand(0.0f, state.body_x, INIT_LEG_LENGTH, 0.0f, 0.0f);
SimController_Step((float)m->opt.timestep);
SimController_GetOutput(&output);
write_output(d, map, &output);
mj_step(m, d);
```

当前命令目标是：

```text
v_set = 0
x_set = 当前 x
leg_set = INIT_LEG_LENGTH
roll_set = 0
yaw_set = 0
```

所以当前目标是：

```text
保持当前位置
保持腿长
保持 pitch 接近 0
尝试站立平衡
```


## 10. 调试顺序

推荐先调这些，不要一上来调 PID/LQR：

```text
1. 确认模型能打开：
   simulate sim/models/rm_robot.xml

2. 确认 bridge 能跑：
   ./build/mujoco_bridge --headless sim/models/rm_robot.xml

3. 确认 GUI 能跑：
   ./build/mujoco_bridge sim/models/rm_robot.xml

4. 检查关节映射：
   left_leg
   left_small_leg
   right_leg
   right_small_leg
   left_wheel
   right_wheel

5. 检查符号：
   joint_pos 正负方向
   joint_torque 正负方向
   wheel_torque 正负方向
   pitch / gyro[1] 正负方向

6. 加输出限幅：
   关节力矩先限制到合理范围。
   轮毂力矩先限制到合理范围。

7. 增加打印：
   base z
   roll / pitch / yaw
   joint_pos[0..3]
   joint_vel[0..3]
   left.L0 / right.L0
   left.theta / right.theta

8. 最后再调：
   INIT_LEG_LENGTH
   LQR 参数
   腿长 PID
   模型质量/惯量
   地面摩擦
   joint damping / armature
```

判断是否接通的最低标准：

```text
u 不是全 0
x / pitch 会随控制和物理变化
GUI 里模型会受力运动
```

判断是否调好的标准：

```text
pitch 不发散
base z 合理
输出力矩不爆炸
左右轮力矩大多数时候趋势合理
机器人能站住 5-10 秒以上
```

## 11. 后续键盘切模式建议

可以做，但建议在 SAFE 站立调通之后再做。

推荐先做仿真专用轻量状态机：

```text
0 / Space -> OFF
U         -> STAND_UP
B         -> BALANCE / SAFE
R         -> reset
W / S     -> 前进 / 后退速度命令
A / D     -> yaw 命令
Q / E     -> 腿长升降
```

内部映射到原工程：

```text
OFF      -> start_flag = 0
STAND_UP -> start_flag = 1, mode = CHASSIS_STAND_UP
BALANCE  -> start_flag = 1, mode = CHASSIS_SAFE
```

等这个简化状态机稳定之后，再考虑接回完整状态机和遥控器/PS2 输入。











