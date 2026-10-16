# 四腿实际角加速度预算探索结果

本次没有找到同时满足减抖与完整控制验收的预算。新增项能降低部分腿部运动，但不能把“更安静”直接当成更好的策略。暂不采用这些候选替换 baseline，也不自动扩大到四种子1000轮。

85% 是这次更值得继续诊断的一档：主指标降低约35%，停车、前进、后退误差都比基线小。至少在本次三个场景与当前指标下，腿部运动指标和速度误差可以同时改善；但它的右前关节指标反而增加约17%，转向响应及饱和度仍有问题，不能直接作为合格配置。

## 做了什么

基线与候选使用同一源 checkpoint、seed=41882943、rtx5070/256环境，每组额外训练200轮，均为 weights-only 初始化，不是原训练的精确续训。源模型此前已从1000轮模型初始化，再训练200轮。本次所有对照从这同一个200轮源模型起步。

Isaac只用于训练及训练保存/恢复流程验证。全部运动采集、阈值选择和效果验收在MuJoCo。未改变六维action_smooth、六关节dof_acc、速度奖励、history、动作/命令、随机化、USD/XML、PD/effort、碰撞、reset和终止语义。新配置是独立opt-in实验profile；默认baseline保留。

所有模型导出后运行8个固定场景各500tick，以及4个停车/换向专项各500tick。主要减抖指标：去掉前2秒后，停车/前进/后退三个场景、四个实际驱动关节的1秒RMS等权均值。不是phi0，也不是目标角二阶变化。

## 匹配对照

| 预算 | 四关节RMS均值(rad/s²) | 相对基线降低 | 停车vx误差(m/s) | 前进vx误差 | 后退vx误差 | 八场景存活 | 四专项通过数 | 完整验收 |
|---|---:|---:|---:|---:|---:|---|---:|---|
| baseline | 18.266 | -- | 0.048 | 0.057 | 0.062 | 8/8 | 2/4 | 参考 |
| 80 | 8.989 | 50.8% | 0.107 | 0.016 | 0.069 | 8/8 | 2/4 | 未通过 |
| 85 | 11.896 | 34.9% | 0.029 | 0.025 | 0.047 | 8/8 | 2/4 | 未通过 |
| 90 | 8.922 | 51.2% | 0.084 | 0.048 | 0.058 | 8/8 | 2/4 | 未通过 |

基线自身的两项转向换向专项未通过，因此相对基线改善也不能代替完整控制验收。停车门槛0.10m/s，前后进稳态门槛0.15m/s，并要求相对退化不超过max(基线10%,0.03m/s)；专项沿用既有停车/换向门槛。新固定场景稳态速度判据是本次探索约定，不是旧ranking自带门槛。

## 四关节分别看

| 预算 | jIJ | jIO | jAB | jAG |
|---|---:|---:|---:|---:|
| baseline | 34.552 | 28.416 | 5.529 | 4.565 |
| 80 | 13.912 | 12.255 | 6.088 | 3.700 |
| 85 | 16.267 | 20.320 | 6.445 | 4.550 |
| 90 | 12.133 | 14.927 | 5.259 | 3.371 |

单位rad/s²，是三个主场景的1秒RMS均值。逐场景p90、速度、目标角、姿态和饱和度均保存于原始CSV/summary。总平均可能掩盖单关节恶化。

## 未通过的具体原因

- 80%：稳态速度门槛或相对退化门槛未通过；停车/换向专项未全通过；动作或力矩饱和比基线增加超过5个百分点。
  - forward：action_saturation_fraction +16.4个百分点。
  - low_stand：action_saturation_fraction +21.0个百分点, effort_saturation_fraction +8.4个百分点。
  - left_stop_right：阶段1: stop_vx_mae; 阶段3: stop_vx_mae; 阶段4: reverse_response_time。
  - right_stop_left：阶段1: stop_vx_mae; 阶段2: reverse_response_time; 阶段3: stop_vx_mae; 阶段4: reverse_response_time。
- 85%：停车/换向专项未全通过；动作或力矩饱和比基线增加超过5个百分点。
  - forward：action_saturation_fraction +13.1个百分点。
  - low_stand：action_saturation_fraction +6.3个百分点。
  - left_stop_right：阶段2: reverse_response_time; 阶段4: reverse_response_time。
  - right_stop_left：阶段2: reverse_response_time; 阶段4: reverse_response_time。
- 90%：稳态速度门槛或相对退化门槛未通过；停车/换向专项未全通过；动作或力矩饱和比基线增加超过5个百分点。
  - low_stand：action_saturation_fraction +5.7个百分点。
  - left_stop_right：阶段4: reverse_response_time。
  - right_stop_left：阶段2: reverse_response_time; 阶段4: reverse_response_time。

## 如何解释

80%降低了左腿运动，但右前jAB的主场景平均角加速度比基线增加约10%。80%前进时jIJ电机目标成为常数，而动作饱和度增加约16.4个百分点。这些是观测到的行为，说明不能仅凭低RMS认定控制改善；不证明所有减抖都来自饱和，也不证明抽动与速度跟踪必然无法兼得。

增益按照加速度超额更新，不会识别某个动作是不是必要救车，也不会依据停车误差自动让步。当前实验的增益基本处于上限1，因此它更接近有最大惩罚强度的探索，尚未显示自动调权相对固定权重的优势。它是ROGER-style单项试验，不是论文完整实现。

搜索分支：80%减抖但控制变差，因此依次放宽至85%、90%。本次没有向更严格的75%、70%继续搜索。模型训练结果不保证随预算单调变化，本次只完成预定分支，不能排除未测试预算有不同结果。

## 验证与复核

文档和代码经实际文件独立复核后无剩余P0/P1；root全工程214项单元测试通过。真实enabled训练2rollout后恢复至3rollout通过。所有已完成200轮模型的checkpoint、gain、源身份、导出golden batch1/3/32及MuJoCo CSV独立重算通过。

verify_results.py确认所有臂初始网络tensor hash相同，模型/控制代码/架构hash未改变，CSV的实际加速度差分和窗口RMS与summary一致，失败不转换成成功样本。

结论已由独立 agent 读取原始文件并重算复核；P0/P1 均为0。评审核验32个固定场景、16个停车/换向序列，以及checkpoint、resume与golden。

## 局限与工件

一种子、每组200轮；没有外力扰动恢复、地形或硬件测试。20ms端点加速度反映控制帧运动，不覆盖所有子步高频冲击。B0来自先前三场景共24个窗口的pooled p90，右腿参考存在目标饱和，不能当作健康或安全阈值。

完整数值见 [comparison.json](comparison.json)、[verification.json](verification.json)、[目标运动诊断](target-motion-diagnostics.json)、[gain日志统计](gain-log-analysis.json)、[复核记录](reviews.md)；每臂export、motion、command_practice目录保留模型身份与原始数据。

![预算与控制对照](budget-comparison.png)
