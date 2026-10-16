# 左右机械映射、对称性与抖动模态审计

2026-10-13。仅使用原 PhysicsV5 MuJoCo 模型、已保存轨迹和纯 PyTorch 几何计算；没有启动 Isaac、没有训练、没有修改训练或生产代码。全部新增文件限于本 mechanics-agent 目录。

**结论：没有发现左右主动电机错映射或右腿必须固定在 action=-1 的机械要求。两侧的可达平面几何几乎一致；-1/-1 对应的是把一条虚拟腿整体转约20°，并非两个电机到机械限位。当前明显抖动的左腿主要是这个整体转动模态，不能靠只限制腿长变化来解决。**

## 电机映射

sim2sim/mujoco/wheelleg_mujoco/model_map.py:9 定义 canonical 次序 jIJ、jIO、jAB、jAG、左轮、右轮；:18 以具名 actuator 获取 ID；:70 实际检查 actuator_trnid 与 canonical joint_ids 完全一致。control.py:56 按 actuator_ids 写力矩，不按 XML 原始声明次序直接写六维数组。

本次编译结果的 actuator ID 为 [0,1,3,2,4,5]，对应 joint ID [11,1,19,6,3,8]；每个 actuator 的 joint ID 都与 canonical 关节一致。XML:233-234 右后早于右前声明，并未造成交换。四腿主动关节的控制 sign 都为 +1；只有右轮为 -1（model_map.py:17）。正确物理反射下，主动腿关节左右保持相同角度；右轮和部分 passive 关节的轴向不同，需相反的几何 q，这已在现有模型中体现。

env_cfg.py:69 两侧名义角相同 [-0.33367134,+0.33367134]。MuJoCo control.py:34 和训练 tasks/direct/wheelleg_flat/control.py:28-31 使用相同的 q_nominal/reference + 0.35*clipped_action 结构；腿 PD 在 MuJoCo control.py:46，对四腿一致使用 kp/kd。未发现为右腿单独加的符号、位置偏移、限位或刚度。

## 可达几何和 phi0

使用正式 usd-anchor-audit.json 中两侧各自的 planar FK 向量，遍历101×101个动作对 [-1,1]²：

- 10201个目标对两侧都合法。
- 同一目标下，轮轴 planar 向量最大差3.740e-7 m，腿长最大差3.796e-7 m，phi0最大差6.867e-5°。
- 全范围腿长约0.11728–0.30415 m，phi0约70.486–110.593°。
- 原 reset 的真实几何左右 phi0差0.000588°、腿长差2.969e-6 m，最大闭环残差7.048e-7 m。

权威 phi0 是 hip 到 wheel 的轴向量（metrics.py:13、:57-65），s=+base Y、d=-base Z；不是 body pitch，也不是单个主动电机的角度。独立 FK 实现见 kinematics/virtual_leg.py:217-251。

| 同侧两动作 | 前/后目标角 rad | 理想腿长 m | 理想 phi0 |
|---|---|---|---|
| 0 / 0 | -0.33367 / +0.33367 | 0.199507 | 90.5397° |
| -1 / -1 | -0.68367 / -0.01633 | 0.199507 | 110.5933° |
| +1 / +1 | +0.01633 / +0.68367 | 0.199507 | 70.4862° |
| -1 / +1 | -0.68367 / +0.68367 | 0.304150 | 90.8684° |
| +1 / -1 | +0.01633 / -0.01633 | 0.117277 | 89.9686° |

定义 q_common=(q_front+q_rear)/2，q_differential=(q_front-q_rear)/2。名义位置附近：

dL/dq_common = 0，dphi/dq_common = -1；
dL/dq_differential = -0.28316 m/rad，dphi/dq_differential = -0.02195 rad/rad。

也就是两电机同向改变主要转虚拟腿角，反向改变主要改腿长。两电机同时 -1 将 common角减0.35 rad，因此 phi0 增20.0535°，腿长不变。这精确解释饱和侧约110°的几何来源，但不证明策略为什么选择该动作。这个目标角仍在现有±1 rad硬边界和±0.95 rad软边界内，不能把动作饱和等同机械撞限位。

## 模型不是数学上的完美镜像，但未发现强制右侧固定的响应偏差

逐刚体反射审计测到：

- 左/右整腿质量0.7187150 / 0.7185310 kg，仅差0.0001840 kg，来源于 jOP/jGH。
- 主动电机 damping=0、armature=0.05 两侧相等；passive 对应0.05/0.005 两侧相等；力矩边界腿±18 Nm、轮±9 Nm两侧相等。
- 部分 COM、轮惯量姿态、dummy位置并非严格镜像：最大对应dummy位置差7.500 mm（XML:85、:117）；jJM/jBE COM差0.500 mm；jOP/jGH COM差0.150 mm。
- 两个connect点沿轴间距，左 MK–IO为4.6 mm、右 EC–AG为4.0 mm（XML:61-62、:95-96）；另一个环两侧均6.5 mm。两个connect的 solref/solimp 相同（XML:225-228）。
- base COM相对两侧hip中点的横向偏移约-0.419 mm，整机 COM约-0.306 mm。

保留原模型、原 equality、原solver，在同一原reset状态对六电机分别施加诊断单位力矩，比较六维 controlled qacc 增量与左右反射后的矩阵。0.5、+1、-1 Nm 三种试值所得反射相对 Frobenius 误差均约0.0119188%，最大矩阵元素差0.003364 rad/s²/Nm。此测试仅 mj_forward，原reset无接触（ncon=0），没有改变模型、积分时间或任何生产参数。

这支持“名义reset附近左右执行响应很接近”；不能据此证明任意接触状态和大运动下完全对称，也不能证明跨引擎响应一致。dummy与闭环点间距的差异应保留在事实记录中，不应现在擅自删除或改成镜像。

主agent独立测到不同种子会选择不同固定侧：run01/run04为左腿固定，run02/run03为右腿固定。该证据与本审计一起，否定“固定右侧由代码唯一写死”的解释；仍不排除微小物理不对称影响学习概率。

## 已有85%轨迹的实际抖动方向

只分解既存 saturation-reward-audit-20261012-v1/trajectories.npz，不重新运行或改变策略。交互零指令2–20 s、900帧：

- 左腿前后电机实际角相关系数0.9570。
- 左腿实际角加速度能量94.694%在 common模态，5.306%在 differential模态。common RMS26.2515 rad/s²，differential RMS6.2143 rad/s²。
- 实际左 phi0 的p5–p95跨度2.0776°、腿长跨度1.6132 mm；目标phi0跨度3.3830°、目标腿长跨度4.0934 mm。
- 正式 nominal_stand 的2–10 s轨迹同样有94.8765%左腿加速度能量处于 common模态；baseline同场景为95.5803%。
- 右腿目标虽固定，真实机械运动仍存在；不能把目标零变化说成腿完全不动。

这里的平方能量恒等式是 pair总能量=2*(common能量+differential能量)。谱分析只用于描述频率分布，去除线性趋势；不能把全部 common运动认定为多余，也不能据此给出安全阈值。

因此，“只惩罚腿长或两电机反向动作”不会直接打中现有左腿主要运动。直接封锁整条腿phi0又会同时拿掉平衡功能。应保留低频姿态调整能力，针对持续的高频往返及闭环反馈做验收；具体频带与滞后必须用完整速度/换向/扰动场景证明。

## 建议与停止点

在本轮禁止训练的范围内，可以做两类有明确意义的诊断：

1. 原history完整保留，按物理反射对actor做左右镜像双推理并平均逆映射后的动作，检查是否能消除种子形成的单侧固定模式。control-frame反射为 y翻转：角速度x/z翻转，投影重力y翻转，yaw命令翻转；四腿、两轮速度、previous actions左右交换。五帧每帧做同一映射。不能简单把左右腿目标强行相等；转向、roll平衡时本来可以不相等。
2. 在debug中对腿common/differential动作分别作明确频率或带宽对照，并完整记录速度误差、phi0、倾斜、摔倒及响应延迟。没有同时通过速度、稳定和抖动验收的后处理不得推荐部署。平均两个坏动作或滤掉必要反馈都可能更差。

若后处理不能同时通过，应将未来训练方案聚焦于：完整保留反馈、避免raw mean长期超出动作可执行区、明确左右协同而允许必要差动、分开记录腿和轮的平滑代价，并对高频实际腿运动设置可验收的目标。虚拟腿共同角/差分角的动作表示有几何依据，但属于新的训练设计，不在本轮修改。

本审计不支持“再加大统一惩罚就会解决”，也没有给出已经验证成功的可部署策略。仅靠已有checkpoint与MuJoCo可定位执行和几何机制，无法恢复原训练101–201轮没有保存的逐步policy gradient/状态；这部分学习因果需要未来新训练对照才能确认。

## 文件与复现

audit.py：原模型映射、质量/惯量反射、FK网格、Jacobian和reset单位力矩响应。
results.json / fk-grid.npz：原始数值。
mode_analysis.py / jitter-mode-analysis.json：已有轨迹模态分解。
verify.py / verification.json：输入哈希、独立模态能量及有限差分Jacobian核对。

audit.py 用 MuJoCo venv运行；mode_analysis.py、verify.py用仓库根.venv运行。五个受保护输入哈希前后相同。新脚本只写本目录。
