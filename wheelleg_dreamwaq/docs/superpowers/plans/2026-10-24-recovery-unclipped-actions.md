# 起身动作取消裁剪 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 用户已授权直接执行；代码完成后由未参与实现的 agent 独立复核。必要CPU校验已完成，4096个环境、2000次更新、每50次保存的训练于2026-10-24 05:41:03至08:39:06（Asia/Singapore）执行完成，最终只读验收已确认训练执行完整性；起身与连续交接能力尚未验收。同一heartbeat周期跟进已暂停；保留另一项目训练运行。

**Goal:** 取消起身策略的腿部目标角裁剪和导出均值动作裁剪，保持训练、导出、交接比较和 MuJoCo 执行使用相同起身动作语义。

**Architecture:** 起身契约升级为 `RecoveryContractV2`，解析配置与导出分别升级为 `RecoveryResolvedConfigV2`、`RecoveryExportV2`。起身动作按四个腿部位置目标及两个轮速目标直接线性转换；冻结701行走策略继续使用原腿部目标限幅，通过独立解码入口参加实际控制和兼容比较。奖励、观测归一化与裁剪、物理失败检查、力矩能力、状态机和完整日志保持现有定义；新校准报告 `RecoveryCompatibilityCalibrationV2` 绑定 `recovery_contract_sha256`，旧配置、checkpoint及校准结果不隐式续用。

**Tech Stack / 使用的技术:** Python、PyTorch、RSL-RL PPO、Isaac Lab、MuJoCo、TorchScript、pytest；使用项目现有两个 Python 环境。

---

工作目录为 `E:/wheel_leg_rl-main/wheelleg_dreamwaq`。所有下列文件路径相对此目录。历史 artifact、旧运行配置、checkpoint、既有验证报告保持原样，新证据写入独立目录。当前实施状态由代码、测试结果和新 manifest 登记，本计划不预先声明通过。

任务1至4已完成。受验源码为 `A7A4A7BDCCF0F6EA98AD654A8DC853C29D25E3DA4202340F87228BF008334B10`；`recovery/artifacts/unclipped-actions-20261024/unit/verification.json` 记录126项测试与32项子测试全部通过，`checkpoint-export/verification.json` 记录35条真实姿态的导出组件通过，`adversarial_review.md` 记录独立复核未发现P1/P2。任务5的2000次训练更新已执行完成，最终只读验收已确认训练执行完整性；调度身份、命令和文件哈希见同目录 `scheduled_training.json`，启动与退出证据分别见 `training_launch.json`、`training_exit.json`，训练输出目录为 `training4096x2000/`，最终验收报告为 `final_acceptance.json`。训练完成不构成起身或连续交接能力验收。

### Task 1: 动作契约和两个解码入口

**Files:** Modify `recovery/contracts.py`、`control.py`、`env_cfg.py`、`env.py`；Test `recovery/tests/test_core_contracts.py`。

- [x] 在 `test_core_contracts.py` 加入越过旧边界的真实张量测试，分别检查起身与701行走解码。核心断言如下：

```python
import torch
from recovery.control import decode_action, decode_walker_action

actions = torch.tensor([[8.0, -8.0, 4.0, -4.0, 2.0, -2.0]])
reference = torch.zeros(4)
legs, wheels = decode_action(actions, reference)
walker_legs, walker_wheels = decode_walker_action(actions, reference)
torch.testing.assert_close(legs, 0.35 * actions[:, :4])
torch.testing.assert_close(wheels, 25.0 * actions[:, 4:])
torch.testing.assert_close(walker_legs, torch.clamp(legs, -1.0, 1.0))
torch.testing.assert_close(walker_wheels, wheels)
```

- [x] 将当前动作契约及解析配置升级到V2，起身目标使用下列公式；0.35表示每单位无量纲腿部动作对应0.35 rad目标偏移，25表示每单位轮部动作对应25 rad/s目标。二者继续作为单位换算比例。

```python
legs = reference + contract.leg_action_scale * actions[..., :4]
wheels = contract.wheel_action_scale * actions[..., 4:]
```

- [x] 保留 `decode_walker_action` 的腿部 `torch.clamp(legs, -1.0, 1.0)`。核对Isaac腿部执行器和MuJoCo起身执行代码路径均写入未经目标角裁剪的起身结果；实际力矩继续限制为腿18 Nm、轮9 Nm，随机化能力仅使用原范围。MuJoCo已实测，Isaac实际回读留待获准训练记录。
- [x] 保留原始高斯动作、log probability和五帧动作历史；保留非有限数值、闭链、真实关节范围、地面穿透及配置规定的内部间隙检查。保留观测字段裁剪与奖励、惩罚的全部公式和系数。
- [x] 运行 `.venv/Scripts/python.exe -m pytest recovery/tests/test_core_contracts.py -q`，动作数值断言通过；旧V1配置通过明确版本检查拒绝。

### Task 2: 交接比较、MuJoCo及校准身份

**Files:** Modify `recovery/control.py`、`mujoco_runtime.py`、`calibrate.py`、`evaluate.py`、`preflight.py`；Test `recovery/tests/test_core_contracts.py`、`test_mujoco_recovery.py`。

- [x] 检查 `check_action_compatibility` 在起身实际动作与行走候选之间分别调用 `decode_action` 和 `decode_walker_action`。实际动作来自行走的样本显式选择行走解码，不能根据动作大小猜测策略来源。
- [x] MuJoCo `RECOVERY` 使用起身解码，`LOCOMOTION` 使用冻结行走解码；只读候选、航向捕获、历史提交和400/500步边界保持原时序。
- [x] 将校准报告升级为 `RecoveryCompatibilityCalibrationV2`，以 `recovery_contract_sha256` 绑定完整起身契约；加载、汇总、preflight及导出涉及的校准身份逐项核验。保留原六维目标差、受限PD力矩差及真实独立接管轨迹要求。
- [x] 运行 `sim2sim/mujoco/.venv/Scripts/python.exe -m pytest recovery/tests/test_mujoco_recovery.py recovery/tests/test_geometry_guard.py -q`。两策略目标映射测试及接触检查通过；旧校准版本或错误契约哈希明确拒绝。

### Task 3: 导出均值和checkpoint兼容边界

**Files:** Modify `recovery/export.py`、`contracts.py`、`checkpoint.py`；Test `recovery/tests/test_core_contracts.py`、`test_checkpoint.py`、`verify_checkpoint_export.py`。

- [x] `RecoveryMeanActor.forward` 直接返回网络均值，保留形状和有限值检查。核心执行公式为：

```python
result = self.actor(observation)
return result
```

- [x] 从V2配置、导出函数和manifest删除已无执行含义的 `deployment_action_limit`，导出/加载固定使用 `RecoveryExportV2`。用超过旧部署界限的有限均值验证Torch与TorchScript结果相同，golden vectors继续来自真实姿态记录。
- [x] 严格resume核对V2完整identity，显式性能兼容导入继续要求相同物理与动作契约。旧V1运行保持可追溯，不能将动作语义变化作为性能等价导入，也不能把旧checkpoint标记成新契约继续优化。
- [x] 运行完整CPU/MuJoCo套件，核对V1版本、错误校准哈希和被篡改导出身份均拒绝。
- [x] 使用真实姿态库进行CPU导出组件验证，检查批量1/3/32、加载、篡改拒绝与checkpoint身份。已生成 `verification.json`，数值误差满足导出门限；该组件更新不构成起身PPO或物理能力证据。

```powershell
& '.venv/Scripts/python.exe' -m recovery.tests.verify_checkpoint_export --pose-bank recovery/artifacts/mujoco_verification/interface_pose_bank.json --output-dir recovery/artifacts/unclipped-actions-20261024/checkpoint-export --walker-export logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-17_17-56-29_rtx5070_fudan-v1_baseline-yaw1-headingoff-env4096-fresh701-20261017/export
```

### Task 4: 独立复核与CPU验证

**Files:** Modify `recovery/train.py` 必要接口检查；Update `recovery/README.md`、`docs/2026-10-22-wheelleg-fall-recovery-design.md`；New CPU verification artifacts under `recovery/artifacts/unclipped-actions-20261024/`。

- [x] 更新README和主设计中的动作公式、版本、观测裁剪和新旧策略解码说明；历史报告保持原样，在活动入口明确原验证对应原契约。
- [x] 代码完成后调用未参与实现的agent复核训练动作/logprob、两个策略的解码选择、校准身份、导出与旧checkpoint拒绝路径，未发现未处理的P1/P2。
- [x] 使用必要CPU张量、导出和配置检查验证越过旧±1 rad界限的起身目标、未经裁剪的均值、原始动作历史、PD力矩上限及旧V1拒绝。单独GPU接口测试已按用户最新安排取消。
- [x] 整理本轮CPU验证和独立复核结果，明确没有新增GPU接口或起身能力证据。真实目标写入、有限损失、checkpoint、完整日志及资源使用在后续获准训练中记录。

### Task 5: 4096环境、2000次更新已执行完成，核验结果与能力分别登记

**Files:** New V2 resolved config and launch records under `recovery/artifacts/unclipped-actions-20261024/`；heartbeat由根agent配置。

- [x] heartbeat编号 `2000` 已记录用户确认的4096个环境、总目标2000次更新、每50次保存，训练于2026-10-24 05:41:03（Asia/Singapore）启动。V2实现、CPU校验和独立复核结果已核对；另一项目训练保持运行。
- [x] 已准备独立新目录和 `RecoveryResolvedConfigV2` fresh训练配置，开启完整奖励、物理检查和日志。配置为 `environment.num_envs=4096`、`assessment.updates=2000`、`ppo.save_interval=50`；调度命令不传 `--resume` 或 `--import-compatible`。新运行目录为 `recovery/artifacts/unclipped-actions-20261024/training4096x2000/`，完整命令、配置文件和源码身份已保存。
- [x] 定时任务已触发，实际资源、启动时间、进程及命令已记录，4096环境的fresh训练已经启动。实际启动记录为 `recovery/artifacts/unclipped-actions-20261024/training_launch.json`，运行目录为 `recovery/artifacts/unclipped-actions-20261024/training4096x2000/`。
- [x] 训练于2026-10-24 08:39:06（Asia/Singapore）退出，完成2000次更新、196608000个环境步，用时10682.7064582秒（2小时58分3秒），退出码为0。训练报告记录 `passed=true`、参数确已改变；40组每50次更新的模型和索引已保存，最终模型为 `recovery/artifacts/unclipped-actions-20261024/training4096x2000/model_2000.pt`。95800个日志包全部确认消费，待处理数量为0，最终排空已完成。
- [x] 完成最终只读验收，报告为[final_acceptance.json](../../../recovery/artifacts/unclipped-actions-20261024/final_acceptance.json)，66项检查全部通过，记录 `training_execution_passed=true`。已核验第1至2000次更新连续、全部损失及奖励有限、40组模型与索引的文件哈希及身份、最终checkpoint的CPU加载；95800个日志包全部确认消费且 `finalized=true`、待处理数量为0。归档6362个分段和47910个原始数据文件的全部SHA256及序号连续性通过。heartbeat任务 `2000` 的周期跟进已暂停，禁止再次启动fresh训练。

更新平均耗时5.3219213921秒，普通更新平均5.2201238636秒，每50次保存的更新平均10.31000029秒，最终排空3.46291秒。起身和连续交接能力尚未验收，继续按独立评估协议登记。

本轮物理统计记录623665次尝试，其中619569次任务失败、4096次在训练结束时截断；物理就绪和接受交接计数均为0。支撑及兼容校准阈值为空，全部196608000个环境步均被 `calibration_unavailable` 阻断，不能将这些计数报告为真实起身成功率0%。倾角阻断比例97.356%，角速度阻断比例98.685%；穿透与闭链阻断次数分别为203932、9267。最大地面穿透记录27608.2003 m异常，CPU原生复查结果不能用GPU保守上界解释，根因尚未确定。内部间隙100%凸包重叠来自 `diagnostic` 模式及已知空腔凸包问题，不能视为实际碰撞率。具体依据见[独立物理统计复核](../../../recovery/artifacts/unclipped-actions-20261024/final_physics_metrics_review.md)。
