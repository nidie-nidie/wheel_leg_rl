# 倒地起身与独立持续站立方案

> 执行说明：使用 executing-plans 按本文的任务清单实施，并分别核验动作分布、奖励、持续站立判据、真实物理运行和模型导出。本文交付采集结果与设计；生产训练代码、奖励和配置尚未按本方案修改，也没有启动新训练。

**Goal：** 机器人从六类倒地初态自行恢复到与 checkpoint401 相近的双轮站姿，并由恢复策略持续保持站立。

**Architecture：** 使用五帧本体观测、前馈 Actor 和普通 PPO；使用401实测的配套关节角度与高度作为站姿参考。扶正、抬高、腿姿态、机身稳定、关节速度和轮子承重分别计算奖励，通过连续站立时间决定成功。

**使用的技术：** PyTorch、现有 rsl_rl PPO、Isaac 训练环境、MuJoCo CPU 独立测评、NumPy 数据归档。

文中“实测”表示本次已经运行并核验的结果；“设计值”表示建议实施的具体参数。新策略的起身能力和成功率需要按本文训练、测评后才能确定。

## 1. 本次已经采集的数据

数据目录：[standing-reference401-20261029](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/standing-reference401-20261029)。

| 项目 | 实际结果 |
| --- | --- |
| 参考策略 | `model_400.pt`，记录的 completed_updates 为401 |
| 运行方式 | MuJoCo CPU，确定性均值动作，速度命令为零，指令高度0.20m |
| 控制与物理周期 | 控制20ms；物理1ms，每个控制步骤20个物理子步骤 |
| 原始参考模型 | formal401，原运行模型的轮子与地面接触设置 |
| 恢复接触模型 | recovery_contact，现有 complete_contact.xml，机器人表面与地面接触 |
| 场景 | 每个模型10段：两个朝向的名义站立、八段轻微扰动 |
| 数量 | 20段，每段30秒，共600秒、30,000个控制样本 |
| 采集耗时 | 62.936秒实际时间 |
| 数据规模 | 约32MiB，详细字段与哈希见 manifest.json |
| 物理诊断 | 20段中没有触发所记录的穿透、闭链、数值和运行警告失败 |
| 离线核验 | 348项通过，29份具有预期哈希的文件匹配 |
| 原入口一致性 | 前300步的观测、动作、qpos、qvel逐项相同，最大差值均为零 |

证据：[manifest.json](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/standing-reference401-20261029/manifest.json)、[summary.json](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/standing-reference401-20261029/summary.json)、[verification.json](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/standing-reference401-20261029/verification.json)、[runtime_verification.json](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/standing-reference401-20261029/runtime_verification.json)。

八种扰动分别为控制坐标前后方向±0.5N、±1N，侧向±0.25N，以及俯仰转矩±0.02Nm；持续0.10秒，发生于第5秒。水平扰动随当时机身姿态转换到世界坐标，其世界竖直分量最大为0.01840N，约体重的0.043%。名义参考场景的外力严格为零；本次没有施加专门的向上辅助力。

数据包括实际关节角度和速度、投影重力、COM高度、机身速度、原始动作、腿目标、轮速目标、实际执行器力矩、PD命令、各部件地面接触力、穿透、闭链误差及扰动标记。接触量为每20ms重建的物理状态，不能作为全部1ms子步骤的接触峰值。实际执行器力矩记录来自原运行的最后一个物理子步骤。

两组模型分别保留统计，训练参考采用 recovery_contact。该模型内部接触关闭；本次数据证明记录条件下的地面支撑表现，内部机构干涉仍由既有几何约束检查，不能据此宣称硬件机械安全。

两个名义朝向使用确定性初态，不能当作独立随机试验。扰动幅度较小，本次数据提供站立状态参考，没有覆盖随机倒地恢复、强外力或动力学随机化。

## 2. 数据对设计的约束

以下为两个名义场景去掉前2秒后，各2800帧的统计。

| 指标 | formal401 | recovery_contact |
| --- | ---: | ---: |
| COM平均高度 | 0.198412m | 0.198265m |
| COM高度范围 | 0.197751–0.199061m | 0.197485–0.199007m |
| 倾斜最大值 | 0.018923rad | 0.018650rad |
| 机身角速度最大值 | 0.6942rad/s | 0.7273rad/s |
| 水平速度最大值 | 0.02009m/s | 0.03094m/s |
| 实际轮速绝对最大值 | 5.0415rad/s | 5.2558rad/s |
| 两轮平均法向承重 | 21.553、21.564N | 22.508、20.715N |
| 两轮瞬时总承重/体重 | 0.810–1.223 | 0.770–1.563 |
| 非轮部件地面法向力 | 0N | 0N |

机器人重量为43.1273N。训练参考的四个腿关节中位角度，按 `jIJ,jIO,jAB,jAG` 排序，为：

```text
[-0.44714018, 0.15482367, -0.39396546, 0.19355669] rad
```

该角度与现有执行器名义偏置 `[-0.33367134,+0.33367134,-0.33367134,+0.33367134]` 有明确差别。执行器零点和站立目标分别保存，奖励使用采集站姿。轮子角度具有累计相位，排除在姿态距离以外；左右腿保持原控制符号，不额外强制原始角度相等。

401站立时存在轮子往复平衡，轮速与轮目标也有明显区别：实际轮速约±5.3rad/s；本次轮目标绝对最大值约26.54rad/s，实际轮力矩可达到±9Nm。参考站姿与承重值得使用，饱和轮力矩和快速变动的控制命令不作为模仿目标。

已核验的门限覆盖结果：

- 逐帧要求轮速小于2rad/s，会排除约63%的401名义站立样本；联合候选条件最长只能持续0.04秒。
- 逐帧要求总轮承重为0.8–1.2倍体重，也会排除正常动力学波动。
- 轮速包络采用6rad/s、总承重使用尾随0.20秒平均，配合第7节运动学候选条件后，20段数据的第2–30秒均满足候选条件，连续28秒。

最后一项只证明候选条件可以接受参考策略的站立表现。它还没有证明候选条件能够排除全部异常支撑状态，也没有证明恢复策略能够到达这些状态。

## 3. 目标行为与边界

成功回合包括：从有效倒地初态开始；扶正并抬高机身；形成接近401的腿部构型；双轮承担主要重量；机身速度收敛；恢复策略继续保持站立。训练成功后仍由同一个恢复 Actor 输出动作。

本次设计的站姿采用高度与腿部角度的容许区域，避免要求每一步完全复现某一帧。轮子允许为平衡进行有限运动；水平位移、机身摆动、非轮支撑和再次倒地分别限制。

外部行走策略不参与本任务的成功判断。401网络冻结，仅用于生成参考数据。新恢复策略使用新契约从 fresh 参数训练，原6200和401模型作为证据与测评对照保留。

## 4. 网络与观测

### 4.1 当前实际结构

现有恢复策略已经使用五帧历史。每帧22维为：机身角速度3、投影重力3、四腿关节角度相对执行器偏置4、六关节速度6、上一执行动作6。五帧展开为110维，辅助训练再追加alpha，成为111维。它使用前馈 MLP，未使用 DreamWaQ 的独立历史编码器、速度估计或隐变量训练模块。

代码依据：[observations.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/observations.py)、[ppo_runner.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/ppo_runner.py)。

### 4.2 建议结构

| 部分 | 设计 |
| --- | --- |
| Actor观测 | 5×22=110维，按时间从旧到新直接展开 |
| Actor隐藏层 | 256、128、64，ELU |
| Actor输出 | 六维潜在高斯均值与六维log标准差，再经tanh形成执行动作 |
| Critic观测 | 41维，内容见下表 |
| Critic隐藏层 | 256、128、64，ELU；输出一个状态价值 |
| 优化算法 | 普通PPO，任务奖励优势，不包含辅助成本优势 |
| 辅助力 | 整个训练与测评均为零 |

Actor保持本体信息，COM高度、真实水平速度、地面接触力只供训练奖励、Critic和成功评估使用。Actor不接收401动作、未来参考、接触真值或评估门控结果。

Critic的41维分组：

| 内容 | 维度 | 定义 |
| --- | ---: | --- |
| 当前无噪声本体帧 | 22 | 与Actor每帧字段一致，动作字段为实际执行的有界动作 |
| 机身线速度 | 3 | 控制坐标系，沿用现有尺度与数值裁剪 |
| COM高度 | 1 | 沿用现有高度归一化定义 |
| 六关节控制力矩 | 6 | Isaac使用现有隐式执行器PD力矩估计，MuJoCo使用实际motor广义力；按18/9Nm归一化并分别标明来源 |
| 连续合格进度 | 1 | 连续合格步骤数/50，限制在[0,1] |
| 剩余回合时间 | 1 | 剩余步骤/最大步骤，限制在[0,1] |
| 保持站立进度 | 1 | 当前HOLD连续步骤/100，限制在[0,1] |
| 阶段 | 1 | GETUP=0，HOLD=1 |
| 两轮平均承重 | 2 | 尾随10步的法向力分别除以mg |
| 非轮平均承重 | 1 | 尾随10步法向力总和除以mg |
| 相对HOLD起点的水平位移 | 2 | 转为控制坐标，除以0.20m并限制在[-5,5]；GETUP时为零 |

Critic需要知道保持时间与支撑状态，新增维度有明确用途。新契约禁止将旧34/35维Critic或111维Actor直接恢复为新模型。

局部reset只初始化对应环境：五帧填充当前初态帧、执行动作历史初始化为产生初态腿目标与零轮目标的动作、承重窗口清空并标记有效计数、阶段与连续计数归零。每个真实控制步骤只追加一次历史；成功、物理失败和超时均记录终态后再reset。

## 5. 动作、探索与物理约束

### 5.1 建议动作映射

使用六维潜在高斯 `u ~ Normal(mu,sigma)`，执行动作 `a=tanh(u)`：

```text
四腿目标 q_target = 1.0 * a[0:4] rad
双轮目标 dq_target = 30.0 * a[4:6] rad/s
```

四腿目标范围为[-1,+1]rad，双轮目标为[-30,+30]rad/s。这是本方案的**仿真控制范围设计值**。401数据覆盖此范围内的站立控制，倒地翻身所需范围仍需要实际六姿态物理验证；现有资料没有提供可信的硬件机械角度上下限。

四腿初始均值使 `tanh(mu)=q_reference_median`，轮均值为零。四腿潜在标准差初值0.15，范围[0.05,0.50]；双轮初值0.10，范围[0.03,0.25]。标准差采用有界参数化并写入checkpoint，禁止在环境中把无界高斯动作截断后继续使用截断前的Normal概率。

PPO storage同时保存pre-tanh潜在动作u与执行动作a。概率使用：

```text
log pi(a|s) = sum(log Normal(u|mu,sigma) - log(1-tanh(u)^2))
log(1-tanh(u)^2) = 2*(log(2)-u-softplus(-2*u))
```

同一动作的新旧PPO概率比可以直接由相同u的新旧Normal概率计算，Jacobian相消；记录的动作熵使用tanh变换后的分布，通过重参数采样估计。KL调节使用潜在Normal的解析KL，同一可逆变换下成立。供KL使用的mean/std保持潜在Normal参数。PPO更新重新评分storage中的u，不能读取mini-batch中新采样动作的临时u。确定性导出执行 `tanh(mu)`。

上一动作观测和平滑项均使用实际执行的a。PPO比率、动作历史、PD目标和导出模型必须对应同一个映射。

### 5.2 保持的物理参数

保留现有模型、质量、闭链与接触设置。Isaac训练为物理5ms×4子步骤，MuJoCo测评为物理1ms×20子步骤，两者控制周期均为20ms。四腿PD为Kp=120、Kd=4，轮速度阻尼0.6，力矩限幅四腿18Nm、双轮9Nm。几何和穿透的物理失败阈值延用已校验契约，形成新的完整身份记录。

Isaac现有隐式执行器的 `applied_torque` 为PD估计量，训练的力矩观测、罚项和饱和诊断沿用该估计并明确标记。MuJoCo记录实际motor广义力，同时保留PD命令作比较。二者不宣称具有相同求解器读数语义，实际执行力矩的证据来自对应模拟器提供的读数。

动作范围通过检查后才能开始正式训练。检查包含：所有倒地初态数值有效；关节方向正确；目标边界与真实PD限幅一致；六姿态在边界目标扫描下不会立即持续触发几何失败。这个检查排除明显的范围与几何错误，策略能否学会翻身仍以训练测评为依据。发现必要起身构型超出范围时，记录具体姿态、关节和受限动作，本配置停止进入训练；不能用站立数据推断翻身一定可行。

## 6. 奖励公式

### 6.1 配套参考与基本函数

训练参考文件为 [recovery_contact_paired_reference.npz](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/standing-reference401-20261029/recovery_contact_paired_reference.npz)，包含560组同时刻的腿角度、高度和物理状态。每个模型的2800个名义尾部样本每隔5步抽取一次，参考只来自名义场景。

对当前四腿角度q和COM高度h，选择同一个参考索引：

```text
j = argmin_j { mean(((q-q_j)/0.10)^2) + ((h-h_j)/0.02)^2 }
```

所有腿姿态距离和高度距离均使用这组 `(q_j,h_j)`。轮相位与世界yaw不进入距离。异常姿态无法通过分别选择互不相容的腿角度与高度减小误差。

定义连续分数：

```text
S(x) = z*z*(3-2*z), z=clip(x,0,1)
C(x;deadband,scale) = 1 / (1 + (max(abs(x)-deadband,0)/scale)^2)
theta = acos(clip(-projected_gravity_z,-1,1))
g_u = S((0.50-theta)/0.35)
g_s = g_u * S((h-0.12)/0.06)
```

`g_u`在theta≥0.50rad时为零、theta≤0.15rad时为一，中间连续变化。`g_s`还要求高度从0.12m到0.18m逐渐增加。它们只根据倾斜与高度计算，不依赖姿态误差、速度、总奖励、成功计数或其他奖励分数。

扶正奖励在全部姿态下存在；抬高奖励随朝上程度启用；接近直立区域后启用腿姿态与安静站立奖励。速度、姿态、承重分别相加，每个分数保留独立梯度。

### 6.2 每秒奖励及权重

每个控制步骤的连续奖励为 `r_step = 0.02 * sum(weight_i * score_i)`。表内权重为每秒的系数，全部属于设计值。

| 奖励项 | 分数或罚项 | 权重 |
| --- | --- | ---: |
| 扶正 | `(1-projected_gravity_z)/2` | +2.0 |
| 抬高 | `g_u*C(h-h_j;0.003m,0.020m)` | +1.0 |
| 四腿姿态 | `g_s*mean_i C(q_i-q_ji;0.04rad,0.10rad)` | +0.8 |
| 机身角速度 | `g_s*C(norm(omega);0.20rad/s,0.50rad/s)` | +0.5 |
| 四腿速度 | `g_s*mean_i C(dq_i;0.20rad/s,1.0rad/s)` | +0.25 |
| 双轮速度 | `g_s*mean_i C(dq_wi;1.0rad/s,4.0rad/s)` | +0.10 |
| 机身水平速度 | `g_s*C(norm(v_xy);0.01m/s,0.05m/s)` | +0.30 |
| 双轮承重 | `g_s*B`，下文定义 | +0.50 |
| 非轮支撑 | `g_s*clip(F_nonwheel_mean/(0.10mg),0,1)` | −0.50 |
| 控制力矩 | `mean_i((tau_i/effort_i)^2)`，Isaac使用PD估计量，MuJoCo使用实际motor力矩 | −0.002 |
| 目标一阶变化 | `mean((a_t-a_prev)^2)` | −0.010 |
| 目标二阶变化 | `mean((a_t-2a_prev+a_prevprev)^2)` | −0.005 |
| 时间 | 常数1 | −0.10 |

`F_L,F_R,F_nonwheel_mean`均为过去0.20秒内实际法向力的平均。总重量使用mg，轮子单侧目标承重mg/2。双轮承重分数定义为：

```text
B = 0.5 * [min(clip(F_L/(0.25mg),0,1), clip(F_R/(0.25mg),0,1))
           + C((F_L+F_R)/mg-1;0.20,0.30)]
```

这项奖励同时反馈两轮是否参与支撑，以及两轮总承重是否接近体重。机身或腿支撑在起身低位阶段不受该罚项约束，进入站立区域后由独立支撑罚项抑制。物理碰撞、穿透和闭链失败始终检查。

删除现有把腿姿态、机身角速度和轮速放进同一个指数的stable分数，使用上表三个独立分数。奖励中不再保留旧potential差分，避免终态potential清零与新的持续站立奖励混合解释。站姿奖励目标区域依据401数据，速度奖励允许比401更安静的动作，不要求模仿其抖动。

终态奖励独立于dt：完整训练站立保持成功+20；硬物理失败−10；规定时间内没有完成站立保持−5。每回合终态只奖励一次。已进入HOLD后继续获得逐步奖励，持续保持完成以后才成功结束。

连续奖励上界为5.35/秒，即每步0.107。gamma=0.99时，无限延长理想站立的折扣回报上界为 `0.107/(1-0.99)=10.7`，成功终奖20超过该上界。准入检查还应对实际HOLD轨迹计算折扣回报，确认及时完成合格保持优于主动中断后拖延终止。

## 7. 持续站立与成功判断

### 7.1 每步合格条件

| 条件 | 设计门限 |
| --- | --- |
| 倾斜 | theta≤0.10rad，约5.73度 |
| COM高度 | 0.185≤h≤0.215m |
| 四腿姿态 | 第6节同一配套参考j，每个关节绝对误差≤0.10rad |
| 机身角速度 | norm(omega)≤0.80rad/s |
| 腿关节速度 | 每个绝对值≤1.50rad/s |
| 轮实际速度 | 每个绝对值≤6.0rad/s |
| 水平速度 | norm(v_xy)≤0.05m/s |
| 双轮分别支撑 | 每侧0.20秒平均法向力≥0.10mg，约4.31N |
| 双轮总承重 | 0.80mg≤两轮平均法向力之和≤1.20mg |
| 非轮支撑 | 非轮平均法向力≤0.01mg，约0.431N；当前非轮力不得超过0.05mg |
| 数值与几何 | 无物理失败，全部字段有限，十步承重窗口已填满 |

角度、速度和高度使用当前控制步骤的真实状态；支撑使用十步平均，并额外检查当前非轮支撑。禁止把缺失接触数据当成零接触。所有阈值写入独立校准文件及其哈希；校准文件缺失或字段无效时在启动阶段明确报错。

独立起身门控只使用物理与持续站立条件。原行走候选动作差与力矩差判据不属于此任务的成功条件。

### 7.2 训练状态与时间

- 初态进入GETUP。每步合格使连续计数增加，任何不合格立即清零。
- 第一次连续合格50步，即1.0秒，且这一时刻不晚于回合第8秒，进入HOLD，记录水平位置为保持起点。
- HOLD继续由恢复Actor控制，合格保持100步，即额外2.0秒；相对保持起点的水平位移≤0.10m。完成后成功，总成功时间上限10秒。
- HOLD任一条件中断，回到GETUP，连续计数与保持起点清除；累计时间不回退。若当前已经超过8秒，回合失败。
- 第8秒前未完成连续1秒合格，记为起身超时。HOLD最多允许延续到第10秒；全部最大预算500个控制步骤。
- 硬物理失败即刻终止，优先于成功；相同控制步骤不能同时记录成功与失败。终止不会自动重新开始训练进程。

接触窗口从真实初态采样逐步建立，初始0.20秒禁止成功资格；reset时禁止把上回合的平均力带入新回合。

### 7.3 独立最终测评

测评采用相同每步物理条件，首次连续1秒合格仍需在8秒内完成；随后保持10秒，保持阶段最大水平位移0.20m。单次最大预算18秒、900个控制步骤。任何再次跌倒、超出门限或物理失败均记录原因，不能累加分散的合格片段冒充连续保持。

401数据的28秒候选覆盖是正样本核验。正式训练前还需要使用现有6200留出轨迹、六类恢复姿态记录，以及实际生成的机身/腿支撑状态核验误接受：倒置、低位、单轮支撑、机身支撑、快速滑动均不能被判成功。短于1秒的姿态经过也不能成功。若门限需要修改，重新生成完整校准报告，并冻结其哈希后启动训练。

## 8. 初态、PPO与预算

### 8.1 初态采样

固定每回合60%来自现有有效恢复姿态库中的五个倒地类别，每类占全部回合12%；40%来自配套站立与近站立扰动状态。原有第六个near_upright类别在最终六类测评中独立报告。40%分为：

| 初态来源 | 全部回合占比 | 生成方法 |
| --- | ---: | --- |
| 配套稳定站立 | 20% | recovery_contact名义参考的完整状态，按现有约束重建并物理核验 |
| 近站立扰动 | 20% | 从同一参考出发，对roll/pitch独立施加±0.08rad均匀扰动，四腿±0.05rad，root水平速度±0.05m/s，角速度每轴±0.10rad/s |

MuJoCo的qpos/qvel通过现有关节名称、控制符号、自由基姿态、root与COM偏移和闭链从动关节转换为Isaac初态，禁止直接复制数组。四腿和root扰动后重新求解闭链并调整root使轮地面高度一致，使用现有几何检查；只接受有效、无穿透的初态。无效样本记入生成报告并在同一来源分组重新生成，不能静默替换成其他来源的站立初态。实际reset后核验q、h、投影重力、两轮支撑与闭链误差，再进入策略采样。

训练和最终测评初态ID严格分离。401参考训练状态可用于站姿奖励；最终测试从留出的倒地记录及独立生成的物理有效扰动开始。40%站立相关样本提供“保持站立”的直接学习信号；训练统计分别报告五个倒地类别、配套站立和近站立扰动的成绩。

### 8.2 建议正式实验配置

| 参数 | 设计值 |
| --- | --- |
| 契约 | RecoveryIndependentStandV1，新配置/模型身份 |
| 初始化 | fresh，seed42002 |
| 环境数 | 4096 |
| 更新数 | 4000，固定预算 |
| rollout | 每环境24步，每更新98,304条transition |
| 总交互 | 393,216,000条transition |
| 保存 | 每50更新，共80组模型及小型identity index |
| 辅助力与成本网络 | 辅助力恒为零；普通PPO不构建成本网络与优化器 |
| PPO | gamma0.99、GAE lambda0.95、clip0.2、5epochs、4minibatches |
| 学习率 | 初值3e−4，解析KL目标0.01；KL>0.02时除以1.5，0<KL<0.005时乘以1.5，每次均限制于[1e−5,3e−4] |
| value loss | 系数1，clipped value loss开启 |
| gradient norm | 最大1.0 |
| entropy coefficient | 0.001，使用变换后动作熵 |
| 观测归一化 | 现有物理尺度；无运行均值方差归一化 |
| 动力学随机化 | 本次名义物理配置关闭；近站立状态扰动按上表执行 |
| 回合终止 | 有限任务终态，成功/失败/超时清零bootstrap；24步rollout边界正常估值 |

该预算是可核验的实验安排，尚无证据保证4000次足够达到目标。每500更新用固定留出12姿态执行CPU确定性18秒测评，输出两阶段成功、失败原因与动作饱和率。发现NaN、日志丢失、物理模型身份变化或持续无法推进时终止本实验并保存故障；禁止自动增加预算或恢复失败训练。

常规训练指标需要区分倒地回合、配套站立回合和近站立扰动回合。整体成功率不能被40%站立初态抬高后用于代表倒地恢复能力。401用于目标区域，不将其Actor或PPO优化器导入恢复训练。

## 9. 测评与验收

### 9.1 训练前检查

1. 本次401数据的348项离线核验和300步运行入口一致性通过。
2. 第7节完整门控在401数据上核验：第2秒后合格覆盖≥99%，各场景可连续保持≥10秒；同时确认非轮阈值、位移约束与完整配套参考实现。
3. 六类倒地与异常支撑负样本均不被判成功；6200已有失败轨迹不能获得成功状态。
4. 拆分奖励对真实状态对比给出符合目标的排序：同姿态降低机身速度、同高度改善腿姿态、同运动学减少机身支撑均提高对应分数；翻身过程中可获得扶正反馈；真实HOLD轨迹及时完成的折扣回报优于主动中断或拖延完成。
5. 有界动作的采样、概率、熵、KL、history与导出映射一致；高斯潜在量取±20时所有公式数值有限。
6. 真实4环境至少2次更新、保存与恢复检查，覆盖局部reset和HOLD中断；有限参数实际变化，归档/模型identity匹配。
7. CPU MuJoCo与训练环境的坐标、关节符号、COM高度、目标与接触力语义一致，动作边界检查通过。

第1项已经完成；第2–7项属于新方案实施后的准入检查。旧225项通过不能代替新动作与成功语义的检查。

### 9.2 最终独立测评集合

| 集合 | 数量 | 检查 |
| --- | ---: | --- |
| 六类留出初态 | 每类100，共600次 | 有效初态按类别报告；完整8秒起身+10秒保持 |
| 轻微扰动站立 | 100次 | 正负方向均衡，采用本次0.05/0.10Ns前后冲量、0.025Ns侧向冲量和0.002Nms俯仰角冲量等级 |
| 401确定性参考 | 当前20段 | 新门控回放，确认物理与统计口径一致 |
| 6200对照 | 固定12留出初态 | 同一物理门控输出可比指标，保留6200原111维观测、alpha=0及原始动作解码 |

六类集合中已有的近直立类别单独报告；另外汇总“初始不满足站立条件”的回合，以及“初始倒置”的回合。初态来自有限姿态库，100次通过独立的物理有效小扰动与yaw生成，不把重复相同状态当成100份不同姿态。每条记录保存初态ID、扰动参数与随机种子。重复基础姿态的相关性单独说明，统计区间不替代姿态多样性。

建议验收目标为：总体完整成功率≥90%，每个类别≥80%，完成起身后的10秒保持失败率≤5%，硬物理失败为零。报告每项分子/分母及95% Wilson区间；扰动测试独立报告，不与名义初态合并成单一成功率。

这组数值是待实现策略的验收目标。还需报告达到首次连续合格的时间、保持时长、倾斜/速度分位数、位移、非轮支撑、闭链误差、穿透和力矩饱和率。只满足倾斜或奖励上升不能替代完整成功判据。

## 10. 日志、模型与失败证据

训练每次更新记录总体及各初态组的奖励、每个独立奖励项、成功率、首次合格时间、保持失败、物理失败、PPO损失、动作标准差、KL、执行目标范围与力矩饱和率。TensorBoard直接写入mean_step_reward、mean_episode_reward、mean_episode_length及各奖励分量。

每50更新保存模型与小型index；index包含checkpoint哈希、配置/参考/契约/源代码哈希和全局更新数。归档每回合的标量统计和失败原因；完整逐步轨迹只记录固定12姿态的定期测评以及选定失败回合。完整诊断轨迹设512MiB配额并轮换，失败汇总与模型保持。常规训练不记录全部4096环境每个控制步骤的原始几何状态，也不为每个checkpoint复制整份历史统计。

模型恢复只读必要checkpoint及小index；历史轨迹的深度核验作为显式诊断操作。日志接口需要新版本与真实提交/消费确认，数据丢失时明确失败，不用空数据或统计快照代替真实已消费记录。

退出验收核对更新1–4000连续、参数有限且发生变化、80个checkpoint哈希和identity、最终日志提交=消费、pending=0与worker正常退出。训练进程完成与行为验收分别记录，原始归档保留，不自动删除旧20GB日志。

## 11. 文件责任与实施清单

本节为执行任务的具体边界。实现中维持现有生产版本可读，所有新语义由 `RecoveryIndependentStandV1` 显式选择；使用新artifact与新run目录。

### 任务1：参考数据与校准配置

文件：新增 `E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/standing_reference.py`、`training/recovery/tests/test_standing_reference.py`；修改 [calibrate.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/calibrate.py)。

- [ ] 读取并核验manifest与NPZ哈希、关节顺序、COM高度语义、560组配套参考及有限性；拒绝没有身份信息的参考。
- [ ] 实现第6节最近配套参考索引，返回同一索引的q、h和距离；相同距离时固定选择较小索引。
- [ ] 对20段401、已有6200轨迹和实际异常支撑初态计算完整门控，保存每项拒绝原因、连续合格最长时间和完整保持结果。
- [ ] 生成具备全部第7节阈值与证据哈希的 `standing_calibration.json`；缺失值直接报错。
- [ ] 运行CPU实际参考读取与配套选择检查，验证轮相位/yaw改变不会改变参考索引、同时改变q与h会选择同一实际参考。

### 任务2：新契约与有界策略

文件：修改 [contracts.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/contracts.py)、[config.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/config.py)、[control.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/control.py)；新增 `training/recovery/bounded_actor_critic.py`、`training/recovery/tests/test_bounded_actor_critic.py`。

- [ ] 新契约固定110维Actor、41维Critic及第5节动作语义；注册新schema并拒绝旧模型恢复。
- [ ] 实现潜在高斯、tanh变换、稳定Jacobian、重参数熵估计、潜在KL、逐组有界标准差和确定性均值导出。
- [ ] storage保存u与a；PPO新旧概率使用相同u，环境与动作历史使用a。
- [ ] 用真实分布采样核验范围，检查零策略、饱和潜在量、梯度有限性，以及未更新策略的PPO比率为1；对比PyTorch变换分布的概率数值。
- [ ] 运行真实六类初态的目标边界与几何检查，输出每个关节和姿态的结果。

### 任务3：独立奖励

文件：修改 [rewards.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/rewards.py)；新增 `training/recovery/tests/test_independent_stand_rewards.py`。

- [ ] 实现第6节全部公式、dt系数及独立项返回值；新schema不调用旧stable/potential奖励。
- [ ] 使用参考实际状态与已有真实倒地轨迹验证数值有限、同姿态速度降低提高独立稳定分数、同支撑条件改善q/h提高对应分数。
- [ ] 实际回放验证倒置区域扶正分数存在，theta从0.50到0.15rad时g_u连续增加，h从0.12到0.18m时g_s连续增加。
- [ ] 验证腿姿态差不会把机身角速度分数乘为零；速度过大不会关闭姿态奖励；终态每项只记录一次。
- [ ] 用真实轨迹生成每项奖励分布报告，包含401、6200与六类初态；训练前检查长期近零与单项尺度。核验连续奖励每步不超过0.107、成功终奖20，并比较实际HOLD片段及时完成和中断/拖延后的折扣回报。

### 任务4：持续站立状态与物理测量

文件：修改 [observations.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/observations.py)、[env.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/env.py)、[batched_supervisor.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/batched_supervisor.py)、[supervisor.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/supervisor.py)、[pose_pool.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/pose_pool.py)。新增 `training/recovery/tests/test_independent_stand_state.py`。

- [ ] 接入真实左右轮和非轮法向力，十步窗口带有效计数；奖励与判据读取同一物理量。
- [ ] 实现第7节GETUP/HOLD状态与50/100步骤计数、8/10秒截止、物理失败优先和保持位移限制。
- [ ] 训练与评估共享每步物理门控，训练HOLD为100步、位移0.10m、总预算500步，中断后按第7.2节返回GETUP；评估HOLD为500步、位移0.20m、总预算900步，进入HOLD后发生一次中断即记录完整保持失败，不重新尝试成功。
- [ ] 完成41维Critic各字段和20/20/60初态来源；每回合记录真实来源类别。
- [ ] 在真实多环境执行局部reset，核验其他环境的history、承重均值、阶段、位移起点与计数保持原值。
- [ ] 核验49/50步、99/100步、第400/500/900步边界，物理失败与成功同一步的优先关系；合格中断清零，有限时限终态无bootstrap。

### 任务5：PPO、模型与紧凑日志

文件：修改 [ppo_runner.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/ppo_runner.py)、[checkpoint.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/checkpoint.py)、[export.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/export.py)、[train.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/train.py)、[metrics.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/metrics.py)及现有metrics协议/worker文件。

- [ ] 新schema选择有界普通PPO，不构建GuidancePPO成本分支；KL调节和熵使用第5节定义。现有RSL自适应调度上限为1e−2，需要在本项目的PPO适配中使用第8节明确的范围，保存调度配置与学习率，并核验每个mini-batch及恢复后均位于[1e−5,3e−4]。
- [ ] 保存模型、优化器、std参数、控制映射、观测与校准身份；恢复完整检查并拒绝旧schema混用。
- [ ] 新schema也直接输出TensorBoard回合奖励与长度，按初态来源分组统计；没有完整回合时写null并记录计数。
- [ ] 实现第10节紧凑日志、轨迹配额、checkpoint小index和明确logging barrier；恢复入口不扫描历史大归档。
- [ ] 真实4环境运行至少2更新、保存与恢复，再运行2更新；检查参数发生变化、概率有限、HOLD/局部reset正确、提交与消费一致。
- [ ] 导出Actor，在同一组实际110维输入上核对训练确定性动作与CPU TorchScript动作及物理目标。

### 任务6：独立测评与正式实验记录

文件：修改 [verify_mujoco.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/verify_mujoco.py)、[evaluate.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/evaluate.py)、[mujoco_runtime.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/mujoco_runtime.py)；新增 `debug/sim2sim/evaluate_independent_stand.py` 与新artifact内的冻结配置、准入报告和验收报告。

- [ ] 固定本方案参考/校准/代码/物理身份，保存第9.1节完整准入证据与其他项目进程、资源状态。
- [ ] 使用独立CPU评估器运行参考与6200对照，核验数据口径和完整持续判据；每个策略按自身模型契约构造观测与目标，禁止给旧6200套用新有界动作映射。
- [ ] 准入通过后按第8节固定预算启动正式训练，定期12姿态测评，记录故障且禁止自动重启。
- [ ] 对候选最终模型运行600+100评估，保存实际成功率、区间、连续保持、失败原因与完整12案例轨迹。
- [ ] 综合核验模型、连续更新、真实耗时、日志退出和行为验收；分别记录训练完成与行为达标，维护最终报告。

### 实施时的验证命令

数据复核已经执行，可按需只读重跑；它只读取本次约32MiB数据：

```powershell
$env:PYTHONPATH='E:/wheel_leg_rl-main/wheelleg_dreamwaq/training;E:/wheel_leg_rl-main/wheelleg_dreamwaq/sim2sim/mujoco;E:/wheel_leg_rl-main/wheelleg_dreamwaq'
& 'E:/wheel_leg_rl-main/wheelleg_dreamwaq/sim2sim/mujoco/.venv/Scripts/python.exe' 'E:/wheel_leg_rl-main/wheelleg_dreamwaq/debug/sim2sim/verify_standing_reference401_20261029.py'
& 'E:/wheel_leg_rl-main/wheelleg_dreamwaq/sim2sim/mujoco/.venv/Scripts/python.exe' 'E:/wheel_leg_rl-main/wheelleg_dreamwaq/debug/sim2sim/verify_stand_collector_runtime_20261029.py'
```

新增CPU检查实现后，由MuJoCo环境中的pytest执行；期望各项通过且没有跳过真实数据读取：

```powershell
& 'E:/wheel_leg_rl-main/wheelleg_dreamwaq/sim2sim/mujoco/.venv/Scripts/python.exe' -m pytest 'E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/tests/test_standing_reference.py' 'E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/tests/test_bounded_actor_critic.py' 'E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/tests/test_independent_stand_rewards.py' 'E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/tests/test_independent_stand_state.py' -q
```

GPU物理更新检查与正式训练在实施任务时使用新artifact中完整解析的配置；入口仍为 `-u -m recovery.train --config <新artifact>/training_config.json --headless --device cuda:0`。脚本不得加入自动降规模、旧模型恢复或失败重启。用户本次要求的数据采集与方案编写不执行这些训练命令。

## 12. 本次交付状态

- [x] 找到实际完成401次更新的checkpoint及导出Actor，记录哈希。
- [x] 两套模型完成20段CPU采集，保存实际观测、动作、目标、力矩和接触。
- [x] 348项离线核验通过；原入口300步一致性通过。
- [x] 已形成数据约束、网络、动作、奖励、持续站立、初态、预算、日志及完整验收设计。
- [x] 两项独立方案复核均无未处理P1/P2；连续奖励上界与成功终奖关系已计算核验。
- [ ] 新契约、奖励、分布、校准和训练实现，按第11节执行。
- [ ] 新策略的实际起身、持续站立与扰动成功率，经第9节测评后填写。

本次生产source SHA256仍为 `F06D42C9DFBD3D5D8CFC06E11414A83D1531C3BDD22A25087B2088B6247C0FB8`。采集脚本、独立检查脚本和本文为新增诊断与设计文件。
