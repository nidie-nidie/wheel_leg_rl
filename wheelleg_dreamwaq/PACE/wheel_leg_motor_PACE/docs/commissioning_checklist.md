# 首次低幅值试采

采用用户已确定的工况：电机装在整机上，机身固定，轮子悬空。沿用现有准确的电机方向。

- [ ] 固定与线束允许本次低幅值运动，实物停止开关可触及。
- [ ] 烧录 `PACE/build/firmware_gcc/commissioning/` 本轮新 HEX，记录实际文件 SHA256。
- [ ] PC capture 921600/8N1 已开启，再复位板子；保存 header 前全部原始数据。
- [ ] 6 个反馈 CAN ID 对应原 manifest；右腿 R_rear/R_front 未因仿真顺序而误换。
- [ ] 观察初始反馈中心和零位是否合理，确认运动范围，不扩大幅值。
- [ ] 一轮 26 段约 211 秒，观察 DM disable / LK stop 后实际停止。
- [ ] raw、session_manifest.json 与质量报告一起保留；完整会话预计 52,750 samples，无缺帧、CRC 或 fault。
- [ ] decode / fit 公共门禁通过，独立验证段有有效数据；模型保持 provisional。

默认是单向发送：PC STOP/Ctrl+C 不会停止电机，提前中止使用实物停止开关。CAN Tx Event 只证明总线发送完成，不证明电机机械停稳。软件测试已经覆盖多种故障，但没有替代上述实板验证。

详细命令与异常排查见 [软件交付说明](../../docs/software-release.md)。正式 final 采集需先落实零位、机械范围、最终幅值、增益及 LK 命令模式合同；仅沿用现有方向无需重测。
