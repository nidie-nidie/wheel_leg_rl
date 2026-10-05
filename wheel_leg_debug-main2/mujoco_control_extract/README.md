# rm_control

## MuJoCo control_extract 运行命令

以下命令默认都在仓库根目录 `/home/shun/MuJoCoBin/rm_control` 运行。
当前 `mujoco_bridge` 默认加载的模型已经是：




```text
mujoco_control_extract/sim/models/wheel_leg_urdf4_self_mesh_all.xml
```

## CAD 名义约束的一次闭链 VMC 后端（2026-08-15）

MuJoCo 构建现在默认使用推导报告 4.3/5.5 的解析模型：

```text
raw q=[q_front,q_rear]
  -> 一次 J/L 圆交求 M
  -> P=ka*L
  -> W=P+kb*(M-L)
  -> L0、phi0、解析 JH
  -> tau_raw=JH^T*[F0,Tp]
```

切回旧理想五杆只需修改下面的宏，并完整重新编译：

```c
// Application/RobotParam/Inc/robot_param.h
#define MUJOCO_VMC_KINEMATICS_MODE MUJOCO_VMC_MODEL_LEGACY_ID
```

默认值是 `MUJOCO_VMC_MODEL_CAD_ONE_CLOSURE_ID`。非 `SIM_MUJOCO` 构建仍自动使用旧模型，因此本次没有改变活动固件。下面的脚本是在 Python 中对同一组解析公式做中心差分和虚功检查，并不直接调用 C 实现：

```bash
python mujoco_control_extract/tools/validate_cad_one_closure_controller.py
```

当前切换范围是运行时正运动学、由其得到的 `L0/phi0/theta` 状态、解析 `JH` 以及 `JH^T` 力矩映射，因此也会间接影响腿长 PID、LQR 状态与离地判断。`CalcPhi1AndPhi4()` 以及 STAND_UP/初始姿态仍保留旧理想五杆逆解，避免在一次 A/B 中同时改变初始化逻辑。控制台会额外输出 `vmcPhi0/axisPhi0`、`kin`、raw `q` 和 raw `tauQ` 供核对。

本轮验证属于短时 smoke test，不是完整稳定性证明：122 个离线姿态中解析 `JH` 与中心差分最大差异为 `3.58e-10`；3 s 原地站立且保持默认 wheel override 时，新旧后端短时动态接近，新后端的 `phi0` 更贴近 MuJoCo 轮轴测量；关闭 wheel override 后，新旧后端都在约 0.5 s 明显失稳。因此该对照只说明新解算没有消除原生闭环失稳，不能单独定位其根因。尚未完成长时间站立、前进、跳跃回归，以及直接调用 C 实现的逐点 `JH` 对照。

注意：本工作区当前多份 `sim/models/wheel_leg_urdf4_assets` 网格是 0 字节占位文件；完整 MuJoCo 加载前需要恢复实际 mesh 资产。解析验证脚本不依赖这些可视网格。

## 分层站立验证接口（2026-08-20）

MuJoCo 物理步仍为 `1 ms`，但控制器现在默认每 `3` 个物理步执行一次，并在中间物理步保持上次最终输出，使运行周期与 `CHASS_FSM_TIME=3 ms` 一致。启动时应看到：

```text
Controller schedule: physics_dt=1.000 ms control_dt=3.000 ms hold_steps=3
```

以下参数只用于仿真隔离，不改变实车固件：

```text
--control-period-ms <ms>       控制更新周期；默认 3，旧行为对照可设为 1
--safe-mit-off                 SAFE 中关闭固定 p_des=0 的 MIT KP/KD，只保留 VMC 前馈
--safe-mit-on                  恢复 SAFE MIT，默认开启
--control-csv <path.csv>       输出通用分层控制 CSV
--control-csv-period-ms <ms>   CSV 周期，默认 10 ms
```

CSV 同时记录 `q/qdot`、VMC/轮轴 `L0/phi0`、四个 equality site 残差、`F0/Tp`、MIT/VMC/请求/最终关节力矩、轮矩以及饱和标志。推荐一次只改变一个变量：

```bash
./build_gui/mujoco_bridge --headless --time 10 --mode safe --drive stand \
  --control-csv output/spreadsheet/stand_a_default_3ms.csv

./build_gui/mujoco_bridge --headless --time 10 --mode safe --drive stand \
  --stand-pitch-target 0 \
  --control-csv output/spreadsheet/stand_b_pitch0_3ms.csv

./build_gui/mujoco_bridge --headless --time 10 --mode safe --drive stand \
  --stand-pitch-target 0 --safe-mit-off \
  --control-csv output/spreadsheet/stand_c_pitch0_mitoff_3ms.csv

./build_gui/mujoco_bridge --headless --time 10 --mode safe --drive stand \
  --stand-pitch-target 0 --control-period-ms 1 \
  --control-csv output/spreadsheet/stand_d_pitch0_1ms.csv
```

统一分析命令：

```bash
python tools/analyze_stand_control_csv.py output/spreadsheet/stand_*.csv \
  --start-time 1 --summary output/spreadsheet/stand_ab_summary.csv
```

当前四组基线均为 `1000/1000` 个样本运动学合法、无 NaN，且 1 s 后没有关节或轮端最终限幅。默认 `pitch_target=-0.075` 虽在 `t=0` 产生 `2.41 Nm` 轮矩阶跃，但当前 10 s 结果比 `pitch_target=0` 衰减更快，说明不能只凭启动阶跃决定最终目标；关闭 SAFE MIT 后腿长从约 `0.1943 m` 越过到约 `0.2085 m`，关节摆幅明显变大，也不能把 MIT bypass 直接当成稳定配置。

几何误差必须分层解释：当前 stand 工作点中 `CAD_ONE_CLOSURE` 相对名义完整 XML 的最大腿长差约 `0.265 mm`、腿角差约 `0.020 deg`；受载 MuJoCo 机构相对名义完整 XML 还会产生约 `0.64 mm/0.14 deg` 的偏移，合成后 `vmcL0-axisL` 最大约 `0.91 mm`。后两者不应全部归因于新正运动学公式。

## 调参版本 v1.0

记录日期：2026-06-06

这个版本是当前已经调出来的 MuJoCo 仿真基线版本。默认调参对象是：

```text
mujoco_control_extract/sim/models/wheel_leg_urdf4_self_mesh_all.xml
```

调参条件：

```text
启动目录：/home/shun/MuJoCoBin/rm_control
推荐启动：--mode safe --drive stand 或 --mode safe --drive forward
初始姿态：默认 --ground-init，对应 init-key pos_debug_ground
腿长目标：INIT_LEG_LENGTH = 0.200 m
腿部 phi0 参考：INIT_L0_PITCH = pi/2
底盘控制周期：CHASS_FSM_TIME = 3 ms
默认前进速度：0.20 m/s
yaw：默认锁航向，不加 --free-yaw
wheel override：默认开启，不加 --no-wheel-override
调参时不使用：--zero-control、--zero-wheels、--invert-right-joints、--freeze-init
```

当前 MuJoCo 模型的质量、关节零位和实车模型并不完全一致。底盘 LQR/VMC 状态构造沿用实车逻辑，但默认仍由 `main_mujoco.c` 的 wheel override 对最终轮矩做仿真补偿，以保证 0.2 m 腿长下稳定站立和 W/S 行驶。

```bash
./mujoco_control_extract/build/mujoco_bridge --mode safe --drive stand --no-wheel-override
```

`--no-wheel-override` 会直接观察实车核心逻辑生成的原始 LQR 轮矩，当前仅用于诊断，不作为稳定运行配置。

v1.0 wheel override 参数：

```text
pitch_target = -0.075
pitch_kp     = 75.0
pitch_kd     = 18.0
pos_kp       = 180.0
vel_kd       = 55.0
pos_ramp     = 0.5
drive_kff    = 0.0
yaw_kp       = 1.2
yaw_kd       = 0.25
wheel_limit  = 2.41
```

当前控制逻辑同步记录：

```text
同步范围：左右底盘 LQR 状态构造、VMC 力矩组合、跳跃三阶段、roll/yaw/防劈叉补偿。
保留差异：LQR 拟合系数不改；质量、轮距、腿部 PID 和 wheel override 使用 MuJoCo 参数。
参数位置：Application/RobotParam/Inc/robot_param.h 中 MUJOCO_WHEEL_BALANCE_*。
作用状态：CHASSIS_SAFE 和 CHASSIS_STAND_UP。
诊断结果：关闭 override 后，当前 XML 与原始实车轮矩不匹配，会发生明显失稳。
```

v1.0 停车调参记录：

```text
调参目标：前进松手，从 DRIVE_FORWARD 切回 DRIVE_STAND 时减少轮毂扭矩突变。
实现方式：drive_kff 前馈不再只在 DRIVE_FORWARD 生效，而是跟随 current_speed 平滑衰减到 0。
验证命令：--mode safe --drive forward --auto-stop-time 2
验证结果：2 s 自动停车后可稳定回到 stand。
```

跳跃状态机试调记录：

```text
调参目标：在快速蹬伸前增加基于实际 VMC_L0 的深压缩阶段，并在落地前后增加预落地/缓冲/恢复阶段；目标是延长腾空时间，同时让较高跳跃落地后还能恢复稳定站立。
代码结构：phase 0-6、阶段计时、离地/触地判断和腿长目标斜坡位于 sim/jump_state_machine.c；sim_adapter.c 只负责输入观测和同步状态机输出，不再保存跳跃阶段切换逻辑。
质量基准：MuJoCo XML 中所有 inertial mass 合计为 4.276255 kg；robot_param.h 中 BODY_MASS 也设为 4.276255 kg。曾尝试按左右腿拆成 2.138128 kg 的单腿等效质量，但普通站立会塌到 L0 约 0.119 m，说明当前控制链路中 BODY_MASS 应按 XML 总质量配置。
控制解耦：每条腿分别计算 F0_gravity/F0_leg_pid/F0_jump_ff/F0_balance 和 Tp_lqr/Tp_split/Tp_pitch/Tp_leg_swing，最后集中合成 F0/Tp。VMC 进一步分别保留 tau_F0 和 tau_Tp，再求和得到关节力矩。
阶段权重：状态机输出 LQR、防劈叉、pitch、腿摆四个 Tp 权重。普通站立固定恢复为 LQR=1、防劈叉=1；跳跃时 LQR=0.5 只在压缩和恢复阶段启用，防劈叉=0、pitch=1、腿摆=0。蹬伸、腾空和预落地不再混入 LQR Tp。
状态 1 压缩：腿长目标按 rate 平滑下降，重力支撑前馈乘 support_scale；左右实际 VMC_L0 都进入 target+tolerance 并保持 hold 时间后才允许蹬伸。
状态 2 蹬伸：腿长目标按 rate 向 MUJOCO_JUMP_EXTEND_L0 变化，并给腿部 F0 增加 MUJOCO_JUMP_THRUST_FF 前馈；实际左右 VMC_L0 达到 extend_l0-end_margin、满足最短蹬伸时间并连续两帧检测到几何离地后，才进入收腿。
状态 3 收腿/腾空：四个主动关节直接用 PD 回到 XML 初态外形；在空中保持该外形，直到开始下降并接近地面。
状态 4 预落地：已经离地后，当左右轮共同离地间隙低于 preland_clearance，且 base 正在下降或轮地间隙正在缩小时，退出纯收腿 PD；腿长目标按 rate 伸到 preland_l0，为触地留出缓冲行程。
状态 5 触地缓冲：首次触地后腿长目标按 rate 收向 buffer_l0，重力前馈和腿长 PID 都降低比例，先让腿吸收冲击；左右轮接触力都稳定超过阈值并保持 buffer_hold 后进入恢复。
状态 6 恢复站立：腿长目标慢慢回 INIT_LEG_LENGTH，轮子普通平衡在 0.60 s 内平滑接管；实际左右腿长、roll/pitch、pitch 角速度、水平速度和双轮稳定触地持续满足要求后，DRIVE_JUMP 才退出到 stand。
异常退出：压缩、蹬伸、收腿、预落地和缓冲阶段均有超时或几何接地兜底，避免卡在单一阶段。
XML 初态收腿：启动时保存解算、闭链稳定后的四个主动关节目标；进入收腿阶段和实际腾空后，直接用关节 PD 回到该目标。世界位置和轮子自转不参与“外形一致”比较。
姿态处理：phase 1-6 使用明确的腿部 pitch PD，目标为 XML 初态 pitch-0.05 rad；收腿、预落地和缓冲阶段使用双轮反作用力矩继续压制 pitch，落地参考为 XML 初态 pitch-0.25 rad。yaw 分两段处理：phase 2 借助轮地摩擦阻尼离地前的 yaw 角速度，phase 3-5 使用较小的 P+D 差动轮矩；phase 5 必须等双轮稳定触地后才启用 yaw 修正。
默认压缩参数：target=0.170 m，rate=0.300 m/s，support_scale=0.600，tolerance=0.010 m，hold=0.010 s，timeout=0.800 s。左右实际 VMC_L0 都进入 0.180 m 以内后才允许蹬伸。
落地默认参数：preland=[clearance=0.015,l0=0.24,rate=0.50,pid=0.80,timeout=0.60]，touchdown_force=5 N，buffer=[l0=0.18,rate=0.25,support=0.85,pid=0.65,hold=0.12]，recover=[rate=0.18,min_time=0.35,pitch_rate=1.0,speed_limit=0.12]。
落地左右差分补偿：当前默认关闭，F0_diff=[roll_kp=0,roll_kd=0,contact_kp=0,limit=0]，L0_diff=[roll_kp=0,roll_kd=0,limit=0]，phase 4 间隙差腿长控制=[kp=0,rate=0,limit=0]。之前的左右差分会在单边触地时形成左右跷跷板，后续如需恢复只能小步加回。
其他默认参数：thrust=0，extend_l0=0.25，extend_rate=0.90，extend_end_margin=0.01，takeoff_pitch_offset=-0.05，landing_pitch_offset=-0.25，wheel_recover_blend=0.60，pitch_tp=[-10,-2,5]，pitch_wheel=[600,60,2.41]，takeoff_yaw=[kp=0,kd=1.2,limit=2.41]，landing_yaw=[start_phase=3,kp=1,kd=1,limit=1]，leg_swing=[0,30,2,5]，tuck=[80,4,7]。
最终执行器统一限幅：非跳跃阶段四个髋关节为 ±7 Nm，跳跃 phase 1-6 为 ±20 Nm，两个轮组始终为 ±2.41 Nm。限幅位于所有 VMC、MIT PD、空中收腿和轮组姿态补偿之后，因此最终写入 MuJoCo 的命令和实际执行器扭矩都不会越界。
几何测量口径：实时日志中的 axisL 使用 XML joint 轴到轴距离，左/右腿分别测主动髋轴到轮轴的最短距离；--debug-geometry 额外输出 jOP/jGH 到轮轴的轴距，用来对照 LEG_L2。VMC_L0 仍是控制器内部虚拟腿长，两者差值用于后续几何标定。

推荐验证：--mode safe --drive stand --jump-at 1.5
当前结果：在 VMC/XML 几何较可信的 0.14-0.25 m 腿长范围内，默认跳跃可以完整经过 phase 1-6 并回到 phase 0，phase 3 不再误判触地，phase 4 能在轮地间隙缩小时进入。共同离地间隙约 0.127 m，base rise 约 0.398 m；跳跃全程最大 roll/pitch 约 0.039/0.457 rad。
低 pitch 对照：将伸腿目标降为 extend_l0=0.205 m、extend_rate=0.50 m/s 后，可完整回到 stand，正向 pitch 峰值约 0.186 rad，但落地反向回弹仍约 -0.248 rad；共同离地间隙约 0.041 m，base_rise 约 0.159 m。该组适合作为姿态稳定对照，不适合作为高度展示结果。遥测文件：output/tune/pitch_reduced_l0205_r050.csv 和 output/tune/pitch_reduced_l0205_r050.svg。
旧的 235/380 推力前馈是按 20-50 Nm 以上的仿真执行器能力调出的，启用当前真实扭矩限幅后不再作为默认对照；继续提高 thrust、加深压缩或把伸腿目标推到 0.27 m 以上，会让髋关节长期饱和，并显著恶化空中和落地 pitch。
调参备注：增大 thrust 或 extend_rate 会延长腾空但提高落地冲击；减小 extend_end_margin 会蹬得更充分；leg_swing_offset 主要调起跳角动量，pitch_offset 主要调 pitch 参考，两者都应小步调整。
```

底盘状态机调参状态：

| 状态 | v1.0 调参状态 | 备注 |
| --- | --- | --- |
| `CHASSIS_SAFE` | 已调，当前推荐主状态 | 这里在仿真里不是无力安全状态，而是正常站立/行走控制输出状态；腿部 VMC/MIT 输出继续使用 extract，轮子平衡由 wheel override 接管。 |
| `CHASSIS_STAND_UP` | 部分可用，不作为 v1.0 主调状态 | `--mode stand` 会进入该状态，并默认 `standup_time = 0.2 s` 后切到 `CHASSIS_SAFE`；当前模型 ground-init 已经接近站立，所以更稳的调参入口是直接 `--mode safe`。 |
| `CHASSIS_CALIBRATE` | 未作为 v1.0 调参目标 | 代码路径存在，关节走速度校准，轮子输出为 0；MuJoCo v1.0 推荐流程没有用它闭环调参。 |
| `CHASSIS_OFF` | 未调参，仅保留失能/关闭入口 | `--mode off` 可进入，主要用于停机/对照，不属于站立和前进效果基线。 |
| `CHASSIS_FOLLOW_GIMBAL_YAW` | 未接入 MuJoCo v1.0 | 状态枚举存在，但当前仿真入口和 `sim_adapter` 没有把它作为可选运行模式调通。 |
| `CHASSIS_OFF_HOOK` | 未调通 | 倒地自恢复入口存在，`ConsoleOffHook()` 仍是注释状态，不属于 v1.0 调参结果。 |

上层 MuJoCo drive 状态机调参状态：

| 状态 | v1.0 调参状态 | 备注 |
| --- | --- | --- |
| `DRIVE_STAND` / `--drive stand` | 已调，推荐验证项 | 目标速度为 0，开启位置保持；适合验证原地站立是否漂移和是否能稳定锁航向。 |
| `DRIVE_FORWARD` / `--drive forward` | 已调，推荐验证项 | 默认前进速度 `0.20 m/s`；前进时 position hold 不参与，主要由 pitch、速度项和 `drive_kff` 维持。 |
| `DRIVE_JUMP` / `--drive jump` | 试调通过，当前是明显离地基线 | GUI 可按 `J` 触发；headless 可用 `--jump-at <time>` 触发。`--drive jump` 会先站稳 1 秒再自动跳，避免初始瞬间起跳。 |

先编译：

```bash
cmake -S mujoco_control_extract/sim -B mujoco_control_extract/build
cmake --build mujoco_control_extract/build --target mujoco_bridge
```

默认会优先查找 `third_party/mujoco`。如果 MuJoCo SDK 放在别的位置，可以这样指定：

```bash
cmake -DMUJOCO_ROOT=/path/to/mujoco -S mujoco_control_extract/sim -B mujoco_control_extract/build
```

推荐 GUI 启动命令：

```bash
./mujoco_control_extract/build/mujoco_bridge --mode safe --drive stand
./mujoco_control_extract/build/mujoco_bridge --mode safe --drive forward
./mujoco_control_extract/build/mujoco_bridge --mode safe --drive stand --no-wheel-override
./mujoco_control_extract/build/mujoco_bridge --mode safe --drive jump --no-wheel-override
```

推荐 headless 验证命令：

```bash
./mujoco_control_extract/build/mujoco_bridge --headless --time 3 --mode safe --drive stand
./mujoco_control_extract/build/mujoco_bridge --headless --time 3 --mode safe --drive forward
./mujoco_control_extract/build/mujoco_bridge --headless --time 6 --mode safe --drive forward --auto-stop-time 2 --no-wheel-override
./mujoco_control_extract/build/mujoco_bridge --headless --time 3 --mode safe --drive stand --jump-at 1 --no-wheel-override
./mujoco_control_extract/build/mujoco_bridge --headless --time 6 --mode safe --drive stand --jump-at 1 --no-wheel-override --jump-telemetry mujoco_control_extract/output/jump_default
```

说明：

```text
推荐优先使用 --mode safe。
--drive stand / --drive forward / --drive jump 是上层运行状态机。
--mode stand 会先走 STAND_UP 起立流程；目前 self_mesh_all.xml 下更稳的是直接从 SAFE 起步。
--jump-at <time> 可以在 headless 中定时触发跳跃。
--jump-telemetry <prefix> 输出 <prefix>.csv 原始数据和 <prefix>.svg 曲线图，并自动创建中间目录。
--jump-thrust <value> 可以临时覆盖跳跃腿部 F0 前馈，扭矩受限稳定版本当前默认 0。
--jump-skip-compression 跳过压缩阶段，用于与旧的直接蹬伸流程做 A/B 对照。
--jump-compress-target / --jump-compress-rate 设置压缩腿长目标和目标变化速率。
--jump-compress-support-scale 设置压缩阶段重力支撑前馈比例。
--jump-compress-tolerance / --jump-compress-hold 设置实际 VMC_L0 的到位窗口与保持时间。
--jump-compress-timeout 设置压缩阶段超时取消时间。
--jump-pitch-target 或 --jump-pitch-offset 设置 pitch 参考；未指定 target 时，参考值等于 XML 启动 pitch 加 offset。
--jump-landing-pitch-target 或 --jump-landing-pitch-offset 单独设置 phase 3-5 的 pitch 参考，不再改变蹬伸阶段 Tp 的 pitch 目标。
--jump-pitch-tp-kp / --jump-pitch-tp-kd / --jump-pitch-tp-limit 用于蹬伸阶段的腿部 pitch 闭环。
--jump-pitch-wheel-kp / --jump-pitch-wheel-kd / --jump-pitch-wheel-limit 用于收腿、预落地和缓冲阶段的轮子反作用力矩。
--jump-motor-wheel-limit 用于临时覆盖跳跃阶段最终轮毂执行器限幅；默认仍为 2.41 Nm。
--jump-leg-swing-offset / --jump-leg-swing-kp / --jump-leg-swing-kd / --jump-leg-swing-limit 用于蹬伸阶段腿摆角补偿。
--jump-lqr-tp-weight / --jump-split-tp-weight / --jump-pitch-tp-weight / --jump-leg-swing-tp-weight 用于临时打开或关闭各个 Tp 通道，跳跃默认 0.5/0/1/0；LQR Tp 只在 phase 1/6 使用，phase 0 固定恢复为 1/1/0/0。
--jump-tuck-kp / --jump-tuck-kd / --jump-tuck-limit 控制空中回到 XML 初态外形的关节 PD。
--jump-extend-l0 / --jump-extend-end-margin / --jump-extend-rate 控制蹬伸目标腿长、离地确认余量和目标变化速率，当前默认 0.25 m / 0.01 m / 0.90 m/s。
--jump-preland-clearance / --jump-preland-l0 / --jump-preland-rate / --jump-preland-pid-scale 用于临时覆盖预落地触发间隙、腿长、变化速率和 PID 比例。
--jump-wheel-recover-blend-time 设置 phase 6 中普通轮组平衡从跳跃 pitch hold 接管的时间，默认 0.60 s。
--jump-buffer-l0 / --jump-buffer-rate / --jump-buffer-support-scale / --jump-buffer-pid-scale 用于临时覆盖触地缓冲腿长、变化速率、重力支撑和腿长 PID 比例。
--jump-landing-roll-kp / --jump-landing-roll-kd / --jump-landing-contact-kp / --jump-landing-balance-limit 用于临时覆盖落地 F0 差分补偿。
--jump-landing-l0-kp / --jump-landing-l0-kd / --jump-landing-l0-limit 用于临时覆盖落地左右腿长目标差分。
--jump-landing-clearance-kp / --jump-landing-clearance-rate / --jump-landing-clearance-limit 用于临时覆盖 phase 4 的轮地间隙差腿长控制。
--jump-takeoff-yaw-kp / --jump-takeoff-yaw-kd / --jump-takeoff-yaw-limit 用于临时覆盖 phase 2 尚未离地时的 yaw 差动阻尼。
--jump-landing-yaw-start-phase / --jump-landing-yaw-kp / --jump-landing-yaw-kd / --jump-landing-yaw-limit 用于临时覆盖 phase 3-5 的 yaw 差动轮矩。
```

phase 4 预落地时直接使用左右轮几何间隙差：

```text
clearance_error = left_wheel_clearance - right_wheel_clearance
balance_l0 = clamp(2.0 * clearance_error, -0.045, 0.045)

left_l0_set  = landing_l0 + balance_l0
right_l0_set = landing_l0 - balance_l0
```

`balance_l0` 每秒最多变化 `3.0 m`。左轮更高时会增大左腿目标、减小右腿目标；进入 phase 5/6 后切回 roll/roll-rate 腿长差分，避免接触间隙噪声继续驱动腿长。

跳跃 yaw 控制分为起跳和空中/落地两段：

```text
phase 2 且轮子仍接触地面：
yaw_term = clamp(1.2 * yaw_rate, -2.41, 2.41)

phase 3-5：
yaw_term = clamp(1.0 * yaw_error + 1.0 * yaw_rate, -1.0, 1.0)
left_wheel_torque  = pitch_wheel_torque - yaw_term
right_wheel_torque = pitch_wheel_torque + yaw_term
```

phase 2 利用轮地摩擦在离地前减少 yaw 角动量；纯空中阶段的差动轮矩主要调整轮速，不能像有地面接触时一样直接产生稳定的 base yaw 力矩。phase 5 在双轮接触力都达到 5 N 后才恢复 yaw 差动，避免单轮先触地期间放大不对称冲击。最终左右轮输出仍受 `±2.41 Nm` 轮组扭矩限幅。

跳跃遥测输出：

```text
SVG 第一部分：base 世界 z、base 相对实际离地时刻的上升高度、左右轮共同离地间隙。
SVG 第二部分：左右腿 VMC 滤波支持力、左右轮 MuJoCo 接触法向力，以及 3 N 离地阈值。
SVG 第三部分：base roll/pitch/yaw。
SVG 第四部分：4 个腿部执行器和 2 个轮毂执行器的 MuJoCo 实际施加扭矩。
CSV：每个仿真步的时间、跳跃阶段、三种离地观测、共同及左右轮间隙、高度、base RPY、控制指令扭矩和实际执行器扭矩。
实际扭矩读取自 mj_step 之后的 d->actuator_force；指令扭矩读取自 d->ctrl。
峰值统计窗口从跳跃触发开始，持续到跳跃结束后 0.5 s，因此包含主要落地冲击。
base_rise_from_takeoff_m 以两轮共同离地超过 5 mm 的首个仿真步为零点。
wheel_clearance_m 取左右轮离地间隙的较小值，表示两只轮子都真正离开地面的高度。
vmc_*_fn_raw_n 使用原 ground_detection 公式和 MuJoCo base 世界 z 加速度计算。
vmc_*_fn_filtered_n 是独立四点均值；vmc_*_airborne 表示该值低于 3 N。
contact_*_normal_n 是 mj_contactForce 给出的轮地接触法向力；contact_airborne 表示左右均不超过 0.1 N。
requested_*_nm 是统一执行器限幅前的请求；cmd_*_nm 是限幅后写入 MuJoCo 的命令；applied_*_nm 是 MuJoCo 实际执行器力矩。
CSV 还记录左右腿的四个 F0 分量、四个 Tp 分量、各阶段 Tp 权重，以及 VMC 映射后的 vmc_joint*_f0_nm/vmc_joint*_tp_nm。
VMC 支持力字段只用于观测，不回写 INS、left_flag/right_flag，也不参与跳跃状态切换。
轮地接触力和左右 wheel clearance 现在会作为跳跃落地阶段的观测输入，用于触发预落地、生成 phase 4 腿长目标差、触地缓冲和恢复站立。
```

重新调试进度：

```text
已完成：
0. 固定质量基准：BODY_MASS 按 XML inertial 总质量 4.276255 kg。
1. 补齐观测量：CSV 已包含 base_x/base_y、车体系前向/横向速度、F0/Tp 分解和执行器限幅后的命令。
2. 建立轴向腿长跳跃基线：深压缩 0.15 m 会翻车；浅压缩 0.18 m + extend 0.22 m 可稳定回到 phase 0。
3. 单项加回：LQR_Tp 半权重只在压缩/恢复使用；防劈叉和腿摆 Tp 在跳跃中关闭，保留明确的 pitch PD。
4. 重调腿长轨迹：当前几何可信范围内的默认值为 compress=0.17、extend=0.25、extend_rate=0.90、thrust=0、tuck_limit=7。
5. 修正 phase 3 触地误判、补回 phase 4、增加阶段超时、PID 分阶段清零、实际速度恢复判据和轮组平滑接管。
6. 修正遥测电机坐标符号，右轮不再被误报为持续饱和。
7. 完成第一档安全增高：extend_l0 设为 0.25 m、extend_rate 设为 0.90 m/s，共同离地间隙约 0.127 m，base rise 约 0.398 m。

未完成：
8. 轮组峰值仍达到 ±2.41 Nm，需要继续降低落地速度/pitch 控制冲突。
9. 跳跃后水平位移仍约 0.81 m；需要继续查起跳瞬间水平冲量、position hold 接管时序和轮组限幅下的速度恢复。
10. 0.27 m 以上腿长区间 VMC/XML 几何误差明显增大，暂不作为默认跳跃伸腿目标；后续若要更高跳跃，应先处理闭链几何标定或 XML 机构等效模型。
```

当前默认浅压缩样例输出：

```text
mujoco_control_extract/output/jump_default.csv
mujoco_control_extract/output/jump_default.svg

默认稳定优先参数：
compress_target=0.17 m，extend_l0=0.25 m，extend_rate=0.90 m/s，preland_l0=0.24 m，thrust=0，tuck_limit=7 Nm
跳跃 Tp 权重：LQR=0.5（仅 phase 1/6），split=0，pitch=1，leg_swing=0

默认结果：
最终 phase：0，能够回到 stand
普通站立最终 L0：约 0.194 / 0.194 m
左右轮共同最大离地间隙：约 0.127 m
base_rise：约 0.398 m
跳跃全程最大 roll/pitch：约 0.039 / 0.457 rad
左右轮触地时间差：约 3 ms
左右轮接触峰值：约 793 / 920 N
6 s 末相对世界初始点前向位置：约 0.811 m
触地后 1 s pitch：约 -0.108 rad
base_proxy 地面接触峰值：0 N
跳跃阶段腿部电机最大实际扭矩：约 11.1 Nm，低于 20 Nm 限制
轮毂电机最大实际扭矩：2.410 Nm
结论：当前默认不是高跳版本，而是在 0.25 m 以内几何可信范围和真实轮组扭矩限幅下的中等增高基线。后续继续提高高度前，应先处理水平位移和轮组饱和样本。

低 pitch 对照结果：
参数：extend_l0=0.205 m，extend_rate=0.50 m/s，其他落地参数保持默认
最终 phase：0，能够回到 stand
左右轮共同最大离地间隙：约 0.041 m
base_rise：约 0.159 m
跳跃全程 pitch 范围：约 -0.248 到 0.186 rad
base_proxy 地面接触峰值：0 N
结论：如果只看正向俯仰峰值，pitch 可以压到 0.2 rad 附近；如果要求全程绝对 pitch 都小于 0.2 rad，当前状态机还不够，主要瓶颈是落地后的反向回弹。继续压 pitch 需要改落地/恢复控制逻辑，而不只是继续降低跳跃高度。

默认测试的蹬伸阶段首次观测：
VMC 支持力和 MuJoCo 接触力归零约 1.043 s。
```





仿真窗口按键：

```text
按住 W：前进
按住 S：后退
松开 W / S：自动回到原地站立 stand
J：触发一次跳跃
SPACE：暂停 / 继续
ESC：退出
```

默认仿真会锁住车体航向，避免串腿在原地绕圈；如果要调试 yaw 自由度，可以加 `--free-yaw`。

默认前进速度是 `0.20 m/s`，可以用命令行改：

```bash
./mujoco_control_extract/build/mujoco_bridge --mode safe --drive forward --forward-speed 0.2
```

如果想启动后直接进入前进状态：

```bash
./mujoco_control_extract/build/mujoco_bridge --mode safe --drive forward
```

只冻结初始姿态，不跑仿真：

```bash
./mujoco_control_extract/build/mujoco_bridge --freeze-init --mode safe --drive stand
```

悬空冻结：

```bash
./mujoco_control_extract/build/mujoco_bridge --freeze-init --hang-init --mode safe --drive stand
```

如果你已经 `cd mujoco_control_extract` 进入子目录，也可以这样运行：

```bash
./build/mujoco_bridge --mode safe --drive stand
./build/mujoco_bridge --mode safe --drive forward
```

push 到 git hub dev 上面的 指令
cd /home/shun/MuJoCoBin/rm_control
git fetch origin
git switch -c dev
git add .gitignore
git commit -m "Ignore Zone.Identifier files"
git push -u origin dev







#### mujoco 仿真


urdf 文件：
urdf 文件放在my robot目录底下，传文件的话就直接拷贝进去就行了

目录顺序是 MuJoCoBin 里面的 my robot。

命令行：
simulate ~/MuJoCoBin/urdf13.SLDASM/urdf/urdf13.SLDASM.urdf

cd 可以进入任何文件夹， 直接输入 cd 就可以返回主界面。
