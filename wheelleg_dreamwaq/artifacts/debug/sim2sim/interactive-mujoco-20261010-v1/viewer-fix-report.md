# 可视化后倒、机身穿地的定位与修正

日期：2026-10-10。用户在键盘可视化中观察到后倒、base_link 穿地，并要求开启机身碰撞。

## 已确认的两个原因

1. 首次新增的 viewer 直接使用物理 runtime 的 model/data。实测 `viewer.sync(state_only=True)` 会清除/改写 `data.xfrc_applied`，单次最大变化 3.2529763846994726 Nm。这删除了正式 runtime 已补上的角速度限制力矩，并破坏其 5 ms 保持状态。零速命令下，该 viewer 在 1.20 秒倾角达到 0.8480 rad，随后 COM 进入地面以下。无 viewer 的同一模型和策略则稳定。
2. 正式模型是 WheelOnlyCollisionV2，只定义 floor-left_wheel 和 floor-right_wheel 两对接触。`base_proxy` 虽然存在，但自动碰撞 mask 为 0，且没有 floor-base 显式 pair。因此机身摔倒时不会得到地面支撑。正式架构 ADR-041 第 1691 行和 `build_mujoco_model.py:475` 明确规定了此旧语义，并非源 USD 被本次改动损坏。

旧窗口的后倒不能当作与此前无窗口评估相同物理条件下的策略结论。此前“八场景通过”也仅指各约 10 秒有限评估，不能推广为所有命令、任意运行时间的站立能力。

## 修正

- `play_keyboard.py` 将渲染改为独立 model/data，每个动作后只复制物理状态供显示。viewer 刷新和鼠标操作影响显示副本，不能修改 runtime 的外力缓存。
- `--base-contact` 创建调试 XML，只给已有近似 box `base_proxy` 增加 floor-base 显式接触。使用原 wheel pair 的 friction/solver 属性；原质量、惯量、关节、执行器、约束、步长、初态和旧轮接触保持不变。
- 当前窗口显示半透明橙色机身碰撞盒。高度越界、tilt > 0.8 rad 或闭环残差 > 5 mm 时明确显示 FALL 并暂停，R 重置并继续。
- 保存 `trajectory.csv` 和实时 `status.json`，留存命令、实际运动、机身高度/倾角、机身碰撞盒最低 z、接触数和失败原因。

## 实际验证

- `probe_viewer_sync.py` 的 500 次动作对照：独立显示副本与无窗口参考的 qpos、qvel、xfrc_applied 最大差值均为 0。旧直接显示方式会在约 1.2 秒后倒。完整结果见 `viewer-sync-probe.json`。
- `verify_base_contact.py`：在相同的 2 mm 机身盒穿入姿态下，正式模型 base-floor 接触数为 0，新调试模型为 4 且有正支撑力；未触地的 10 秒 nominal_stand 的 qpos/qvel 与正式模型完全一致。见 `base-contact-verification.json`。
- `play_keyboard.py --base-contact --smoke` 验证键盘命令、裁剪、history 衔接、reset 和 20 次动作执行成功。
- 新窗口实际启动。读取到零速命令运行 32.34 秒时，COM 高度 0.19374 m、tilt 0.01862 rad、机身盒最低 z 0.16368 m，失败原因为空。此时仍后退约 0.4975 m/s，未完成停住目标。
- 101 个正式文件和旧评估输入/结果 SHA256 保持不变；无训练，主架构仍为 v0.24。当前 base-contact 是用户要求的可视化实验配置，不是正式排名模型或正式训练配置的更换。

## 限制

开启机身碰撞不代表策略学会静止，也不保证任意键盘命令下不摔倒。此次修正恢复了显示与物理 runtime 的一致性，并使机身碰撞真实生效。零速持续后退仍需沿已有同时刻 CENet/Actor 诊断继续调查。
