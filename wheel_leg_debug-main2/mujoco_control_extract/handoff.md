# Wheel-leg 控制与 VMC 调试交接文档

> 快照日期：2026-08-20  
> 作用范围：`wheel_leg_debug-main2/mujoco_control_extract` 及其与实车固件 `wheel_leg_debug-main/rm_test-dev` 的接口边界  
> 目的：让下一次对话不必重走本轮关于 LQR、VMC、偏置五连杆、MuJoCo 和实车极性的排查过程。

## 2026-08-20 精细验证增量

本轮已经把原 P0/P1 诊断项做成可重复接口，但尚未据此改 PID/LQR：

- 物理步保持 `1 ms`；控制器默认每 3 步运行一次并保持输出，启动 banner 显示 `control_dt=3.000 ms hold_steps=3`；`--control-period-ms 1` 可复现旧行为。
- 新增 `--safe-mit-off/--safe-mit-on`，只隔离 SAFE 下固定 `p_des=0` 的 MIT 项。
- 新增 `--control-csv` 与 `--control-csv-period-ms`，CSV 分开记录 VMC、MIT、请求值、最终值、饱和、轮轴几何和 equality site 残差。
- 新增 `tools/analyze_stand_control_csv.py`；当前四组 10 s 数据和汇总位于 `output/spreadsheet/stand_*`、`stand_ab_summary.csv`。

当前源码重建和四组 headless A/B 已通过。每组 1000 个样本均 `kin=[1 1]`、无 NaN，1 s 后无最终关节/轮端饱和。重要的新证据是：

1. 默认 `pitch_target=-0.075` 在 t=0 的确产生 `+2.41 Nm` 双轮阶跃，但 1~10 s 的 pitch 峰峰值约 `8.08 deg`；改成 `pitch_target=0` 后反而约 `10.47 deg`。因此启动阶跃是真实的，但当前目标 0 不是更好的闭环平衡点，不能直接据旧假设修改默认值。
2. 关闭 SAFE MIT 后，平均腿长由约 `0.19434 m` 变成 `0.20854 m`，最大关节摆幅由约 `0.54 deg` 放大到 `7.19 deg`。MIT 与 VMC 的结构性对抗得到分层力矩验证，但直接 bypass 不是稳定方案；下一步应建立与 CAD 逆解一致的 MIT 目标或重新设计纯 VMC 阻尼。
3. 3 ms 与旧 1 ms 运行的宏观轨迹几乎相同，但旧 1 ms 的 `dL0/dphi0` RMS 约为 3 ms 结果的三分之一，证明旧周期确实缩小了运动学导数；周期修正应保留，即使当前站立外观差异不大。
4. stand 工作点中，`CAD_ONE_CLOSURE` 相对名义完整 XML 的最大误差约 `0.265 mm/0.020 deg`；受载 MuJoCo 机构相对名义完整 XML 还有约 `0.64 mm/0.14 deg` 偏移；两者叠加后运行时 `vmcL0-axisL` 最大约 `0.91 mm`。不要把 0.91 mm 全部算成新正运动学公式误差。

详细命令和字段说明已同步到 `README.md` 的“分层站立验证接口”一节。

## 0. 一句话状态

当前 MuJoCo 工程已经启用 **`CAD_ONE_CLOSURE` 偏置机构正运动学 + 解析雅可比 + 虚功力矩映射**，其公式与本工程 HTML 报告第 4.3/5.5 节一致，离线数值验证通过；最近 GUI 中看到的明显“腿抖/车身摇摆”，从现有低频日志看主要是默认 wheel override 在启动时制造 pitch 阶跃并使轮端饱和，而不是新 VMC 几何解算跳支。

但是目前还不能说“整套控制已经正确”：SAFE 中仍把 **MIT 的 `p_des=0` 位置力矩**与 **VMC 腿长/姿态力矩**相加，且物理仿真步长为 1 ms、腿部控制代码内部仍按 3 ms 计算导数。这两个问题应先隔离，再谈 PID/LQR 调参和实车移植。

---

## 1. 工程与版本边界

### 1.1 本文涉及的两个活动工程

- 当前需要继续修改和仿真的工程：`mujoco_control_extract`
- 当前实车固件工程：`../wheel_leg_debug-main/rm_test-dev`

不要把以下内容混在一起：

- MuJoCo 中已经接入的 `CAD_ONE_CLOSURE` 后端；
- 实车固件中仍在使用的旧 VMC/五连杆路径；
- 工作区里的其他历史副本、SPR 对照工程和旧 build 目录。

新 CAD 后端目前被 `SIM_MUJOCO` 宏限制为仿真专用，尚未移植到实车固件。

### 1.2 当前目录没有可用的 Git 变更证据

工作区根目录的 `.git` 是空目录，`mujoco_control_extract` 也没有独立 Git 仓库。因此：

- 不能依靠 `git status` 或 `git diff` 还原本轮改动；
- 下一次修改前应先备份目标文件，或先建立可用的版本控制基线；
- 不要把邻近目录里的同名文件误认为当前活动文件。

### 1.3 现有 `build` 目录不是 Windows 当前源码的可靠构建证明

现有 `build/CMakeCache.txt` 的源目录仍指向 WSL 中的旧路径：

```text
/home/shun/MuJoCoBin/rm_control/mujoco_control_extract/sim
```

MuJoCo SDK 为：

```text
/home/shun/MuJoCoBin/mujoco-3.3.0
```

因此每次改完 Windows 工作区源码后，必须把当前源码复制到临时 WSL 目录并重新 CMake/build，不能直接运行旧二进制就认为新代码已经生效。

Windows 工作区的 `sim/models/wheel_leg_urdf4_assets` 中目前是 0 字节 mesh 占位文件。GUI 运行时必须从 WSL 完整副本恢复真实网格。

---

## 2. 本轮目标如何演变

本轮最初从实车固件开始，目标依次是：

1. 恢复完整 LQR 控制路径；
2. 核对 MF9025 速度符号、左右轮 ID、轮端输出极性；
3. 通过 UART7 增加观测器、卡尔曼、LQR 和最终输出遥测；
4. 排查 SAFE 后轮子持续转动、腿长偏软、`phi0/theta` 平衡点不一致；
5. 通过关节映射和 CSV 测试区分电机零点、offset、MIT 与 VMC 冲突；
6. 发现传统五连杆公式并不能严格描述 CAD/XML 中的偏置闭链，转而从 XML 推导精确运动学；
7. 在确认完整 XML 与简化机构误差量级后，采用更适合实时控制的 `CAD_ONE_CLOSURE` 模型；
8. 只先修改 `mujoco_control_extract`，完成解析雅可比和 VMC 映射的仿真验证，暂不移植实车；
9. 最近一次 GUI 运行发现明显摇摆，当前任务转为分离 wheel override、MIT、VMC 和周期不一致四层影响。

这意味着下一轮不应该重新从“9025 是否能解码负速度”开始，也不应该一上来继续改 LQR Q/R。当前最有价值的工作是把 MuJoCo 中各控制层分开验证。

---

## 3. 当前 MuJoCo VMC 控制链

### 3.1 当前启用的后端

默认选择定义在 [robot_param.h](Application/RobotParam/Inc/robot_param.h)：

```text
SIM_MUJOCO -> MUJOCO_VMC_MODEL_CAD_ONE_CLOSURE_ID
非 SIM_MUJOCO -> legacy ideal-five-bar
```

仿真启动时必须看到：

```text
VMC kinematics backend: CAD_ONE_CLOSURE (HTML 4.3/5.5), raw q=[front,rear]
```

如果没有这行，不能认为新 VMC 已经运行。

### 3.2 前后电机与 `phi1/phi4` 的对应

当前物理对应仍与旧 `mujoco_control_extract` 一致：

| 物理支路 | 左腿 | 右腿 | 旧接口角 | 新后端内部角 |
|---|---|---|---|---|
| 前侧主动杆 | J0 / `jIJ` | J3 / `jAB` | `phi4` | `q_front` |
| 后侧主动杆 | J1 / `jIO` | J2 / `jAG` | `phi1` | `q_rear` |

注意：`q_front/q_rear` 和 `phi4/phi1` 对应同一根物理杆，但数值不必相等，因为旧接口中还包含符号、坐标方向和角度 offset。

新后端先得到内部顺序 `[tau_front, tau_rear]`，再转换回旧接口；后续发送层的索引和符号会抵消兼容变换。不要仅凭某一层的负号判断最终电机力矩方向。

### 3.3 坐标、角度和轮轴状态定义

每条腿的二维建模平面约定为：

- 原点：髋部公共参考点 `I`；
- `s`：`+base Y`；
- `z`：`+base Z`；
- `down=-z`；
- 轮轴中心为 `W=(W_s,W_z)`。

由轮轴坐标得到虚拟腿状态：

\[
L_0=\sqrt{W_s^2+W_z^2}
\]

\[
\phi_0=\operatorname{atan2}(-W_z,W_s)
\]

\[
\theta=\frac{\pi}{2}-Pitch-\phi_0
\]

这里 `atan2(y,x)` 中的 `2` 是函数名的一部分，表示使用两个输入确定完整象限，不是乘以 2。

### 3.4 当前简化闭链正运动学

当前运行时模型不是每周期完整求解 XML 中 `K/N/W` 两次闭链，而是 HTML 4.3 中已经验证过的单闭链简化：

```text
q_front, q_rear
    -> 主动点 J、L
    -> 以 J、L 为圆心求被动铰点 M
    -> P = k_a L
    -> W = P + k_b(M-L)
    -> L0、phi0、theta
```

其核心比例为：

\[
k_a=\frac{|IP|}{|IL|}=2.216813311434
\]

\[
k_b=\frac{|PW|}{|LM|}=2.243476979061
\]

因此：

\[
\vec P=k_a\vec L
\]

\[
\boxed{\vec W=\vec P+k_b(\vec M-\vec L)
=k_b\vec M+(k_a-k_b)\vec L}
\]

CAD 设计中 `I/L/P` 被视为共线；XML 中残留的约 0.271 mm 横向偏差在该实时模型中被忽略。`k_a` 与 `k_b` 也不完全相等，因此它不是严格的整体相似缩放，但与完整 XML 解的误差已经很小，见第 4 节。

### 3.5 解析雅可比和 VMC 力矩映射

正运动学写成：

\[
\vec y=H(\vec q)
=\begin{bmatrix}L_0\\\phi_0\end{bmatrix},\qquad
\vec q=\begin{bmatrix}q_{front}\\q_{rear}\end{bmatrix}
\]

雅可比定义为：

\[
J_H(\vec q)=
\frac{\partial(L_0,\phi_0)}{\partial(q_{front},q_{rear})}
\]

其物理意义是当前姿态附近：两个关节角各变化一点，虚拟腿长和虚拟腿角分别变化多少。

当前运行时使用解析求导，不使用中心差分。中心差分只保留在离线验证脚本中，用来检查解析公式是否正确。

VMC 给出的虚拟广义作用量为：

\[
\vec f_v=\begin{bmatrix}F_0\\T_p\end{bmatrix}
\]

其中：

- `F0`：沿虚拟腿方向的力，单位 N；
- `Tp`：绕虚拟腿角度方向的虚拟力矩，单位 N·m；
- `Tp` 不是 MF9025 轮毂自转力矩。

由虚功一致性：

\[
\delta W
=\vec\tau_q^T\delta\vec q
=\vec f_v^T\delta\vec y
=\vec f_v^TJ_H\delta\vec q
\]

对任意微小 `delta q` 都成立，因此：

\[
\boxed{
\vec\tau_q=J_H^T
\begin{bmatrix}F_0\\T_p\end{bmatrix}}
\]

这一步才是“把虚拟腿的力和力矩分配给前后两个关节电机”。

### 3.6 还没有切换的内容

以下部分仍使用旧理想五连杆，不应误称为“整个工程都已换成偏置模型”：

- `CalcPhi1AndPhi4()` 逆运动学；
- STAND_UP 的目标关节角生成；
- 初始姿态相关的旧逆解；
- 实车固件中的正解、雅可比和 VMC；
- 实车的 `q_model=s_i q_encoder+b_i` 标定关系。

当前切换范围仅包括 MuJoCo 运行时的：

- 正运动学状态 `L0/phi0/theta`；
- 解析 `J_H`；
- `J_H^T[F0,Tp]` 关节力矩映射。

---

## 4. 已完成的数学和离线验证

验证脚本：

```powershell
python .\mujoco_control_extract\tools\validate_cad_one_closure_controller.py
```

当前结果：

```text
model                                  CAD_ONE_CLOSURE
valid_samples                          122
max_abs_analytic_minus_fd_jh           3.5813796372963225e-10
max_abs_virtual_power_residual_w       1.7763568394002505e-15
max_abs_finite_virtual_work_residual_j 2.942398620565416e-14
min_abs_circle_m_determinant_m2        0.011988515306827749
cad_vs_full_xml_max_l0_error_mm        0.2704683692586751
cad_vs_full_xml_max_phi0_error_deg     0.02554928724500881
q=0 L0                                 0.12049298105058034 m
q=0 phi0                               1.571050093872754 rad
```

`q=0` 处：

\[
J_H=
\begin{bmatrix}
-0.0916784671 & +0.0916784671\\
-0.5153848924 & -0.4846151076
\end{bmatrix}
\]

这些结果分别证明：

- 解析雅可比与中心差分参考高度一致；
- `J_H^T` 满足虚功/功率一致性；
- 单闭链 CAD 模型在采样工作域内非常接近完整 XML 几何。

它们**不能证明**：

- SAFE 闭环一定稳定；
- MIT、腿长 PID 和 LQR 参数正确；
- 实车编码器零点、极性和力矩方向正确；
- 仿真与实车的 `q` 定义已经同步。

相关文档和脚本：

- [完整推导报告](output/offset_closed_chain_kinematics_derivation.html)
- [报告生成器](tools/build_offset_kinematics_report.py)
- [数值/几何推导脚本](tools/derive_offset_kinematics.py)
- [CAD 单闭链验证脚本](tools/validate_cad_one_closure_controller.py)

历史对比结论也应保留：旧理想五连杆与完整 XML 在较大工作域中的最大差异约为 `1.616 mm` 腿长、`1.102 deg` 腿角和 `8.208 mm` 二维轮轴位置。它说明旧模型不严格，但这个几何误差量级本身不足以解释几十毫米腿长偏差、完全拉不到目标或猛烈磕头。

---

## 5. 当前 SAFE 实际发送的不是“纯 VMC”

MuJoCo 的 `CHASSIS_SAFE` 是主动站立状态，不是零力矩状态。

每个关节的最终候选力矩为：

\[
\boxed{
\tau_{final}
=K_p(0-q)+K_d(0-\dot q)+\tau_{VMC}}
\]

当前 MIT 参数：

```text
NORMAL_POS_KP = 20
NORMAL_POS_KD = 1
p_des = 0 rad
v_des = 0 rad/s
```

这意味着当前系统同时存在两个目标：

- MIT 位置层希望四个主动关节回到电机模型的 `0 rad`；
- VMC 腿长环希望 `L0` 跟踪正常腿长目标（当前通常为 0.200 m），并通过 LQR 的 `Tp` 调整腿角。

而当前 CAD 模型在 `q_front=q_rear=0` 时：

```text
L0 = 0.120493 m
phi0 ~= 90.015 deg
```

因此 `p_des=0` 与 `L0_des=0.200 m` 并不对应同一平衡构型。两层控制很可能互相抵消，不能只看 `tauQ` 就判断最终执行器在做什么。

当前 MuJoCo 腿长 PID：

```text
Kp = 3200
Ki = 0
Kd = 1500
输出限制 = +/-220 N
滤波 alpha = 0.1
```

最终 bridge 层仍保留保护限幅：

```text
普通关节  +/-7.00 N·m
跳跃关节  +/-20.0 N·m
轮端      +/-2.41 N·m
```

任务内部旧 `torque_set` 限幅虽然已注释，但最终发送仍受 bridge 层约束。

---

## 6. 最新 GUI 日志对“抖动”的判断

### 6.1 日志能确认的内容

最新运行日志已经打印：

```text
VMC kinematics backend: CAD_ONE_CLOSURE (HTML 4.3/5.5)
```

在普通站立片段约 `t=0.501~8.501 s`：

- 左右全程 `kin=[1 1]`，没有 NaN、失效或装配支路跳变；
- `vmcL0-axisL` 平均约 `+0.896/+0.902 mm`，波动峰峰值仅约 `0.04 mm`；
- `vmcPhi0-axisPhi0` 平均约 `+0.116/+0.117 deg`，波动峰峰值约 `0.008 deg`；
- 1 s 后四个主动关节角的峰峰值只有约 `0.32~0.36 deg`；
- raw VMC `tauQ` 大致稳定在 `+/-8.4 N·m`；
- 最终关节输出 `u[0..3]` 约为 `+/-2.2 N·m`。

这表示 raw VMC 约 8.4 N·m 被 MIT 位置项抵消到最终约 2.2 N·m，结构性对抗已经由日志确认。

### 6.2 真正明显摆动的是整车 pitch 和轮端

同一片段中：

```text
pitch 峰峰值       约 0.139 rad = 7.96 deg
车体 x 峰峰值      约 52 mm
车体 z 峰峰值      约 177 mm
左右轮速峰峰值     约 5.3 rad/s
roll/yaw 波动       很小
```

左右轮基本同步，表现为对称的前后俯仰，而不是单侧 VMC 映射错误。

默认 wheel override 的关键参数为：

```text
pitch target = -0.075 rad
pitch KP     = 75
pitch KD     = 18
position KP  = 180
velocity KD  = 55
wheel limit  = 2.41 N·m
```

初始 `pitch=0` 时，仅 pitch 比例项就要求：

\[
75\,[0-(-0.075)]=5.625\ \mathrm{N\,m}
\]

于是两轮在启动第一个时刻立即被限制到 `+2.41 N·m`，随后整车越过目标并进行约 1.5 s 周期、逐渐衰减的欠阻尼摇摆。

因此当前最合理的判断是：

> 最新日志里的低频大幅视觉摇摆，主要由 wheel override 的启动 pitch 阶跃引起；现有日志不支持“新 `J_H` 跳支导致腿抖”这一解释。

### 6.3 当前日志的限制

终端默认每 500 个物理步才输出一次，即约 0.5 s 一点。它可以看到低频整车摇摆，但不能排除 5~100 Hz 范围内的关节、闭链或 PID 高频颤振。

另外：

- `siteL` 是旧 dummy closure site 的髋部中点距离，不是轮轴虚拟腿长；
- 判断几何误差应比较 `vmcL0` 与 `axisL`；
- 日志里的 jump/abort 片段不能与普通 stand 段混在一起判断。

---

## 7. 当前最高优先级的未解决问题

### P0：1 ms 物理步与 3 ms 控制周期不一致

当前事实：

- XML 物理 timestep：1 ms；
- bridge 每个物理步都调用一次控制器，即控制代码实际按 1 kHz 执行；
- 左右腿 VMC 仍传入固定 `CHASS_FSM_TIME/1000 = 0.003 s`；
- PID 的 D 项使用相邻误差差值，没有显式除以 `dt`。

结果是：控制器实际每 1 ms 更新一次，却把腿长、腿角导数当成每 3 ms 更新一次。`dL0/dphi0/dtheta` 和等效阻尼都会失真。

下一轮必须二选一：

1. **推荐用于仿真/固件一致性：**每 3 个物理步执行一次完整固件控制器；
2. 把整个控制链统一改成 1 ms，并重新处理所有导数、滤波和 PID 参数。

不能只把某一个 `dt` 从 3 ms 改成 1 ms 就宣布问题解决。

### P0：MIT 与 VMC 目标冲突

需要增加一个仿真专用、可一键关闭的隔离开关或 CLI 参数：

```text
SAFE 下令 MIT KP=0、KD=0
保留 VMC、wheel override 和最终限幅
```

比较隔离前后的 `tauQ`、MIT torque、final u、L0 和 q。如果关节抖动或腿长误差明显改善，说明主要矛盾在 MIT/VMC 叠加，而不是雅可比。

### P1：wheel override 启动目标阶跃

先进行 `pitch target=0` A/B，不先改 LQR：

```bash
./build_gui/mujoco_bridge --mode safe --drive stand --stand-pitch-target 0
```

再做极短 wheel-off 对照：

```bash
./build_gui/mujoco_bridge --mode safe --drive stand --zero-wheels
```

wheel-off 只能看最初约 1 s，因为倒立摆没有轮端控制必然会倒，不能把倒下本身误判为 VMC 错误。

### P1：遥测分辨率不足

建议新增 5~10 ms CSV，而不是继续依赖 0.5 s 控制台输出。至少记录：

```text
t
q_front/q_rear, qdot_front/qdot_rear
L0, L0_set, dL0
phi0, dphi0, theta, dtheta
F0_gravity, F0_pid, F0_total
Tp 各 LQR 状态分量与 Tp_total
tauQ_front/tauQ_rear
tauMIT_front/tauMIT_rear
u_final_front/u_final_rear
joint/wheel saturation flags
pitch, pitch_rate
wheel torque/speed
```

### P2：逆运动学仍是旧模型

正解和力矩映射已切换，不代表位置目标生成也准确。若后续仍要使用 MIT 位置目标支撑腿长，应把 `CalcPhi1AndPhi4()` 替换成与当前 CAD 模型一致的逆解，或明确取消该位置层对 VMC 的对抗。

### P2：仿真与实车电机角 offset 不同

历史检查得到：

```text
活动实车固件 offset  ~= +/-0.19163715 rad
MuJoCo 工程 offset    ~= +/-0.10301526 rad
差值                  ~= 0.08862189 rad = 5.07766 deg
```

不能把这 5.078 deg 直接加到新模型中。实车必须建立：

\[
q_{model,i}=s_iq_{encoder,i}+b_i
\]

其中 `s_i` 是方向，`b_i` 是模型零偏。建议使用每台电机至少 2 个、最好 3~5 个已知物理姿态拟合。硬限位自动标零只给出编码器参考点，不等同于 VMC 模型角零点。

---

## 8. 下一次对话建议严格按此顺序执行

### 阶段 A：建立可重复的仿真基线

1. 从 Windows 当前源码重新生成 WSL 临时副本并完整编译；
2. 确认启动 banner 是 `CAD_ONE_CLOSURE`；
3. 不按任何按键，运行默认 stand 10 s；
4. 保存控制台或 CSV，单独标记 stand 段，不混入 jump；
5. 检查 `kin=[1 1]`、最终限幅和 NaN。

### 阶段 B：先隔离轮端启动阶跃

1. 默认参数跑一遍；
2. `--stand-pitch-target 0` 跑一遍；
3. `--zero-wheels` 只跑最初约 1 s；
4. 比较 pitch、wheel torque、q、L0 和 final joint u。

预期：如果 target=0 后低频大摆动明显消失，先处理 wheel target 的初始化/斜坡和阻尼，不动 `J_H`。

### 阶段 C：隔离 MIT 与 VMC

1. 加入 sim-only MIT bypass 宏/CLI，默认保持当前行为；
2. MIT=0、VMC 保留，小力矩限幅下短测；
3. 同时输出 raw `tauQ`、MIT torque 和 final u；
4. 决定最终架构：
   - 纯 VMC 力矩控制；或
   - 与当前 CAD 逆运动学一致的 MIT 位置辅助；
   - 不再使用固定 `p_des=0` 作为“通用腿长保持”。

### 阶段 D：统一控制周期

优先实现“物理 1 ms、固件控制每 3 步执行一次并保持上次输出”，保证与 3 ms 固件参数一致。修正后重新做阶段 A~C，之后才比较腿长 PID。

### 阶段 E：腿长 PID 和 LQR

1. 对照当前 MuJoCo `3200/0/1500, +/-220 N` 与实车 `200/0/1500, +/-40 N`；
2. 一次只换一组参数；
3. 始终保留关节 `+/-7 N·m` 和轮端 `+/-2.41 N·m` 保护；
4. 腿长环稳定后，再调 LQR Q/R；
5. Q/R 改动后必须重新生成 LQR 多项式系数，不能只手改某一项系数；
6. MuJoCo 和实车当前 LQR 系数并不一致，调参前先从活动源码重新列出两套系数。

### 阶段 F：回归顺序

每次结构性修改后依次测试：

```text
stand 10 s
-> 腿长缓慢上升/下降
-> 前进/后退/停止
-> 最后才是 jump
```

不要在同一份日志中混合 stand、jump、手动外力和 abort 后再归因。

### 阶段 G：实车移植

只有在以下条件同时满足后才移植：

- MuJoCo 正解、解析雅可比、虚功验证仍通过；
- stand 与腿长变化不再存在控制层对抗；
- 1 ms/3 ms 周期已统一；
- `q_model=s_i q_encoder+b_i` 通过多姿态标定；
- 新后端用宏包裹，可一键退回旧 VMC；
- 实车上架、轮子离地、关节/轮端小限幅开始测试。

---

## 9. 从 Windows 终端重新构建并打开 GUI

先在 PowerShell 中进入 WSL：

```powershell
wsl.exe
```

然后在 WSL shell 中逐行执行：

```bash
set -euo pipefail

run_dir="$(mktemp -d /tmp/wheel-leg-gui.XXXXXX)"
case "$run_dir" in
  /tmp/wheel-leg-gui.*) ;;
  *) echo "unexpected run_dir: $run_dir"; exit 1 ;;
esac

mkdir -p "$run_dir/mujoco_control_extract"

cp -a /mnt/c/Users/shun/Desktop/wheel_leg_debug-main2/mujoco_control_extract/Application \
      "$run_dir/mujoco_control_extract/"
cp -a /mnt/c/Users/shun/Desktop/wheel_leg_debug-main2/mujoco_control_extract/Components \
      "$run_dir/mujoco_control_extract/"
cp -a /mnt/c/Users/shun/Desktop/wheel_leg_debug-main2/mujoco_control_extract/sim \
      "$run_dir/mujoco_control_extract/"
cp -a /home/shun/MuJoCoBin/rm_control/cmake \
      "$run_dir/"

# 用 WSL 完整工程中的真实 mesh 覆盖 Windows 的 0 字节占位文件。
cp -a /home/shun/MuJoCoBin/rm_control/mujoco_control_extract/sim/models/wheel_leg_urdf4_assets/. \
      "$run_dir/mujoco_control_extract/sim/models/wheel_leg_urdf4_assets/"

cmake -S "$run_dir/mujoco_control_extract/sim" \
      -B "$run_dir/mujoco_control_extract/build_gui" \
      -DMUJOCO_ROOT=/home/shun/MuJoCoBin/mujoco-3.3.0

cmake --build "$run_dir/mujoco_control_extract/build_gui" \
      --target mujoco_bridge \
      -j"$(nproc)"

cd "$run_dir/mujoco_control_extract"
./build_gui/mujoco_bridge --mode safe --drive stand
```

注意：

- 不要把整个 `/home/shun/MuJoCoBin/rm_control/.` 复制到未检查的目标；之前的 Permission denied 就来自目标路径被解析成 `/`；
- 上述命令只复制构建所需目录，并验证 `run_dir` 必须位于 `/tmp/wheel-leg-gui.*`；
- 每次 Windows 源码变化后都重新执行这套复制和 build；
- 若想做 pitch 目标对照，在最后一行加 `--stand-pitch-target 0`；
- 若想做短时断轮对照，在最后一行加 `--zero-wheels`。

---

## 10. 遥测字段速查

| 字段 | 含义 | 不应误解为 |
|---|---|---|
| `vmcL0` | 新 VMC 正解算出的虚拟腿长 | MuJoCo site 中点长度 |
| `axisL` | MuJoCo 髋参考到真实轮轴中心的几何长度 | PID 目标值 |
| `siteL` | 旧 dummy closure site 的几何量 | 真实轮轴腿长 |
| `vmcPhi0` | 新正解的虚拟腿角 | LQR 的 `theta` |
| `axisPhi0` | MuJoCo 真实轴线换算出的虚拟腿角 | 电机编码器角 |
| `kin` | 当前闭链解是否合法 | 闭环是否稳定 |
| `q=[front,rear]` | 新后端原始前/后主动关节角 | 旧 `phi1/phi4` 数值 |
| `tauQ` | `J_H^T[F0,Tp]` 得到的 raw VMC 关节力矩 | 最终执行器力矩 |
| `u[0..3]` | MIT、VMC 等合成并经当前发送链处理的关节输出 | 纯 VMC 输出 |
| `u[4..5]` | 最终轮端力矩 | 虚拟腿 `Tp` |
| `Tp` | 虚拟腿角广义力矩 | MF9025 轮毂自转力矩 |

---

## 11. 实车固件已有结论与仍未证明的内容

### 11.1 已由源码/架车数据支持

- MF9025 速度字段已按有符号 `int16` 解码，能表示正负速度；
- 单轮测试支持 CAN ID 与左右轮标签的当前对应；
- Observe 中左右轮反馈曾经接反，交换后双轮前进/后退时的 `aver_v/KF v` 方向已基本一致；
- `start_flag=0` 时实际发送轮矩为零，即使遥测中的“候选 LQR wheel_T”仍可能非零；
- SAFE 是主动控制状态，不等于未按 START 的零输出状态；
- 完整 LQR 候选输出、SAFE 软件限幅和 9025 最终发送限幅必须分层看待。

### 11.2 仍需实车验证

- 两侧轮胎接地点是否共同推动整车向前，而不是仅看两个电机轴旋转方向；
- 新偏置 VMC 在实车编码器坐标下的 `s_i/b_i`；
- 关节最终力矩极性；
- 卡尔曼速度/加速度在落地滚动条件下是否准确；
- 地面接触时完整 LQR 是否稳定。

架车手转轮测试只能证明符号链和解码，不能证明落地闭环稳定。人工抬腿得到的 `phi0≈70/110 deg`、旧 `theta` 偏移约 `+/-15.75 deg` 没有夹具精度，不能直接硬编码为最终 LQR 平衡点。

---

## 12. 关键文件索引

### 当前运行代码

- [VMC 正解、解析雅可比和力矩映射](Components/Algorithm/Src/VMC_Calc.c)
- [VMC 数据结构与旧接口](Components/Algorithm/Inc/VMC_Calc.h)
- [模型选择、PID、MIT、限幅参数](Application/RobotParam/Inc/robot_param.h)
- [左腿控制任务](Application/Task/Src/ChassisL_Task.c)
- [右腿控制任务](Application/Task/Src/ChassisR_Task.c)
- [仿真控制器适配及 MIT+VMC 合成](sim/sim_adapter.c)
- [MuJoCo 主循环、wheel override、CLI、日志](sim/main_mujoco.c)
- [仿真 CMake](sim/CMakeLists.txt)
- [MuJoCo XML](sim/models/wheel_leg_urdf4_self_mesh_all.xml)

### 推导与验证

- [偏置闭链完整 HTML 报告](output/offset_closed_chain_kinematics_derivation.html)
- [报告生成器](tools/build_offset_kinematics_report.py)
- [几何推导脚本](tools/derive_offset_kinematics.py)
- [CAD 单闭链数值验证](tools/validate_cad_one_closure_controller.py)

### 项目说明和历史记录

- [MuJoCo 工程 README](README.md)
- [WSL/MuJoCo 运行说明](sim/README_MUJOCO_WSL.md)
- [历史站立调参记录](STAND_TUNING_LOG.md)
- [运动控制报告](MOTION_CONTROL_REPORT.md)

不要为了站立调试撤销无关的 jump 状态机、跳跃遥测或其他已有功能；先用宏/CLI 做可逆隔离。

---

## 13. 容易再次混淆的结论

1. **解析雅可比通过验证，不代表闭环参数正确。**
2. **SAFE 不是零输出；未按 START 才是实际发零。**
3. **raw `tauQ` 不是最终关节力矩。** 当前还会叠加 MIT 并最终限幅。
4. **虚拟腿 `Tp` 不是轮毂电机力矩。**
5. **轮轴位置正解用于状态估计；`J_H^T` 用于力矩分配。** 两者都来自同一个机构模型。
6. **中心差分不是运行时控制算法。** 当前仅用于离线验证解析导数。
7. **正运动学不依赖上电初态。** 给定合法的 `q_front/q_rear` 和固定装配支路即可直接求 W；上一周期只是在通用圆交实现中防止误选另一装配支路的鲁棒策略。
8. **最新明显低频摇摆更像 wheel override 启动阶跃。** 但 0.5 s 日志不足以排除高频关节颤振。
9. **仿真 offset 与固件 offset 相差约 5.078 deg，不等于应直接增加 5.078 deg。**
10. **旧理想五杆几何误差存在，但量级不足以单独解释猛烈磕头。**
11. **当前 CAD 后端只在 MuJoCo 正解/Jacobian/VMC 链生效。** 逆解和实车尚未切换。
12. **悬空架车结果不能代替落地测试。**

---

## 14. 建议下一次对话的开场提示词

可以直接把下面这段作为下一次对话的第一条消息：

> 请先完整读取 `mujoco_control_extract/handoff.md`，然后只处理 MuJoCo 工程，不修改实车固件。先核对当前 `CAD_ONE_CLOSURE` 后端和最新源码，再按 handoff 的阶段 A~D 做：第一步增加 5~10 ms 的分层 CSV；第二步做 `stand-pitch-target=0` 对照；第三步增加可一键关闭的 sim-only MIT bypass；第四步把固件控制周期与 1 ms 物理步统一。每一步只改一个变量，保留关节和轮端最终限幅，并给出源码证据、构建结果和 A/B 数据结论。暂时不要继续调 LQR Q/R，也不要把仿真结论直接写入实车。
