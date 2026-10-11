# 四腿目标换向约束小范围训练结果（2026-10-11）

结论：这版换向约束没有解决左腿抽搐，暂不扩大训练。相对于同样新增200轮的baseline，N12的左腿目标反向次数约减少6.8%，但jIJ/jIO实际角速度RMS增加10.7%/36.3%；N8的jIJ增加17.6%，jIO减少13.7%。目标指令少反向，不能直接等同于实际关节更平稳。

MuJoCo后退跟踪有所改善，但两组的停车和前进误差均大于baseline。三组在两端八场景都存活；只有N8通过MuJoCo的vx子门槛，三组都未通过Isaac的vx子门槛及全部停车换向专项。不能把“不倒”当作整体性能验收。

两个自适应组最终gain都达到1.0，停车换向率仍超过各自预算。这次没有固定gain对照，不能据此证明动态调权优于固定惩罚。这里只采用A1启发的有界ROGER-style单项调权，不是论文完整ROGER算法。一个种子、每组200轮只支持当前配置不宜扩大，不能证明平稳与速度跟踪不能同时实现。

## 做了什么

保留原六维action_smooth=-0.01，新增四个腿最终角度目标的换向软惩罚；轮子不受新增项约束。采用0.01 rad迟滞、1秒滑窗，比较disabled、12次/秒、8次/秒三个配置。新成本上限权重0.02，gain∈[0,1]，每24控制步更新。软惩罚不会硬裁动作，也不会直接给电机增加力矩。PPO/CENet/ActorCritic/storage及原物理、PD、力矩上限、动作、命令、随机化、history、终止条件未改。

正式架构已增加v0.28 §33/ADR072。实现提交为42d8c35（本地；未推送）。默认配置仍关闭新增项，启用需显式选择stop_reverse_reversal12_v1或stop_reverse_reversal8_v1。

## 公平对照条件

三组均复制当前PhysicsV5的stop_reverse_v1 run03 1000轮模型网络与归一化状态，每组新增200轮；seed=41882943，256环境，8个minibatch，rollout=24。每组都新建optimizer、iteration、监控与过程RNG；复用相同reset cache。这里是weights-only初始化，不能称为保留原optimizer的普通resume。

源checkpoint：2026-10-11_04-20-34_rtx5070_fudan-v1_dreamwaq-v1-suite-20261011-024927-run03-seed41882943/model_999.pt。

源SHA256：3D621280CF7F11088C972B0DD84CF2E698C29A4C8CDA8BC370BB80A2669DF885。

三组初始网络tensor SHA256：E856686DAAC3ECC78E5B883EC2B3D11A28FCD6F3EE33EC3DF1403E5DEADE469A。

三组耗时9.68/10.33/10.89分钟；总训练约30.9分钟。监控间隔1800秒，每组不足30分钟，保存起始与结束快照；没有数值硬异常，三组均出现phi0_symmetry_degraded监控提示。

## MuJoCo停车与速度跟踪

停车抖动采用有效动作后2s<t≤10s，400帧；jIJ/jIO是左腿两个关节。角速度RMS用于衡量实际运动，20ms加速度RMS是相邻控制周期速度差除以0.02秒。速度跟踪误差来自原八场景全10秒，单位m/s。这些窗口定义一致，不能与旧诊断的397帧pre-action指标直接混用。

| 组别 | 左腿目标反向 次/秒 | 左腿角速度RMS rad/s | 左腿20ms加速度RMS rad/s² | 停车vx MAE | 前进vx MAE | 后退vx MAE |
| --- | --- | --- | --- | ---: | ---: | ---: |
| baseline | 19.602 / 19.608 | 0.680 / 0.645 | 39.874 / 38.213 | 0.0181 | 0.0715 | 0.1981 |
| N12 | 18.270 / 18.275 | 0.753 / 0.879 | 41.310 / 48.376 | 0.0364 | 0.1066 | 0.1813 |
| N8 | 19.997 / 23.492 | 0.800 / 0.557 | 47.475 / 33.848 | 0.0310 | 0.0861 | 0.1061 |

完整逐腿、逐场景与阶段数据见[results.json](../artifacts/phase2_dreamwaq/leg-reversal-trial-20261011-v1/results.json)，可视化见[parking-comparison.png](../artifacts/phase2_dreamwaq/leg-reversal-trial-20261011-v1/parking-comparison.png)。

## 验收结果

| 组别 | Isaac八场景存活 | Isaac vx门槛 | MuJoCo八场景存活 | MuJoCo vx门槛 | 停车换向专项通过数 |
| --- | --- | --- | --- | --- | ---: |
| baseline | 是 | 否 | 是 | 否 | 2/4 |
| N12 | 是 | 否 | 是 | 否 | 1/4 |
| N8 | 是 | 否 | 是 | 是 | 0/4 |

Isaac三组均有后退vx误差超过0.20 m/s；N12前进也超过0.20。MuJoCo baseline/N12的低位停车vx误差分别0.1389/0.1136，超过0.10门槛；N8通过的只是原vx子门槛，并非完整任务验收。这里没有把一步骤的通过改成整体通过，也没有更改任何旧门槛。

所有组停车换向专项均存活，但N8四项性能都未通过，不能仅按固定后退场景挑选模型。

## 实现核验与可复现记录

201项CPU单测、2项真实Isaac集成已通过；集成覆盖每控制步恰好一次计数、目标不变、重复读取不计数、partial reset不推进物理时间、weights-only初始化、resume、play和export。非零gain的恢复由单测覆盖。三组标准32条golden-vector及相同batch的Python/TorchScript 1/3/32最大误差均为0.0，冻结容差仍为1e-7。

独立标量程序重算12000帧换向率，最大误差0.0；独立重算20ms加速度，最大误差0.0。116个受保护文件hash不变。训练期间源码fingerprint保持FA40027FA51DE3C185D1526F26D94C7A5D6EC6941C34CAF2117A355FF3906C16。原MuJoCo XML与正式4×1000评估入口未改；独立诊断入口仅适配探索上下文、增加被动日志。源策略smoke的旧CSV共有列逐值完全相等。

[experiment-manifest.json](../artifacts/phase2_dreamwaq/leg-reversal-trial-20261011-v1/experiment-manifest.json)记录所有训练命令、模型、缓存、导出、评估路径和hash；[implementation-acceptance.json](../artifacts/phase2_dreamwaq/leg-reversal-trial-20261011-v1/implementation-acceptance.json)记录单测和集成；最终复核见[final-verification.json](../artifacts/phase2_dreamwaq/leg-reversal-trial-20261011-v1/final-verification.json)。

训练生成的latest_run.txt已指向本次最后一组，这是运行指针更新。原交互viewer和先前工作区改动保留。本次没有调用子agent。

## 下一步判断

保留代码和试验结果，不扩大当前配置到四种子1000轮。若继续研究，应先弄清“训练约束与部署目标反向频率不一致”以及“目标反向减少但实际关节速度上升”的原因，再决定是否改变指标或调权。需要分别评估训练采样与部署确定性策略，避免把计数器有效等同于机械抖动问题已解决。此处只记录建议，没有继续改配置或追加训练。
