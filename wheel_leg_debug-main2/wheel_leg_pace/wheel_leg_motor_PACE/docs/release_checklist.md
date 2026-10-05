# Final Model Release Checklist

只有全部项目通过，才允许发布 `model_maturity = final` 的执行器模型。

## Contract 与固件

- [ ] `final_identification_contract.md` 无 `TBD`，签字完整。
- [ ] `PACE_FINAL_CONTRACT_READY == 1`，其余 `PACE_FINAL_*` 与合同逐项一致。
- [ ] session header、全部 stage config、footer 的 config hash 一致且非零。
- [ ] 固件 build ID、motor manifest hash、AXF/HEX SHA-256 已归档。
- [ ] 六个 motor sign、zero、gear ratio 均已实机验证。

## 原始会话

- [ ] session type 为 `FINAL_IDENTIFICATION_SESSION`。
- [ ] DM 与最终 LK 主模式各有独立 `FIT` 和 `VALIDATION` 段。
- [ ] raw 文件 CRC、sequence、timestamp 和 `config_seq` 连续。
- [ ] footer 无 overflow、drop，且 statistics complete。
- [ ] Tx Event lost、unmatched event、queue failure、CAN/UART error 均为 0。
- [ ] 500 Hz 命令周期与 enqueue-to-Tx latency 满足 commissioning 阈值。
- [ ] 所有用于拟合的样本都有 Tx/Rx valid、未 age 饱和、未命令饱和。
- [ ] 安全事件、异常中止或方向错误计数为 0。
- [ ] 原始 `.raw` 已只读归档，并记录 SHA-256。

## 拟合与验证

- [ ] DM 正式 PACE optimizer 的代码版本、配置、随机种子和 run directory 已归档。
- [ ] LK 拟合只使用合同指定的主命令模式。
- [ ] 参数全部位于预先定义的物理边界内。
- [ ] 每个通道都有独立 validation RMSE、P95、max、delay/phase 和 saturation 指标。
- [ ] validation 指标满足项目阈值；阈值在看结果前已冻结。
- [ ] 未使用 validation 数据调参后仍把同一数据称为 holdout。
- [ ] provisional 模型没有被直接改标签升级为 final。

## 双仿真一致性

- [ ] Isaac adapter 顺序为 `L_front,L_rear,R_rear,R_front,L_wheel,R_wheel`。
- [ ] MuJoCo adapter 使用完全相同的顺序、sign、zero、gear ratio 和 delay。
- [ ] MuJoCo 不再将理想控制器转矩直接无延迟写入 actuator。
- [ ] Isaac 与 MuJoCo 使用同一 manifest 文件及 SHA-256。
- [ ] 两端都完成 FIT replay 和 held-out VALIDATION replay。
- [ ] 两端误差差异有解释，不存在通道交换或方向反转。

## 发布物

- [ ] final actuator manifest JSON。
- [ ] JSON Schema 校验结果。
- [ ] raw session、normalized NPZ 和转换 summary。
- [ ] 每通道 fit/validation 指标报告和曲线。
- [ ] Isaac replay 配置与日志。
- [ ] MuJoCo replay 配置与日志。
- [ ] 固件 AXF/HEX、map、build log 和源代码快照。
- [ ] commissioning、安全回归和 final release 检查表。

## 发布结论

- [ ] **PASS**：允许训练/部署工程引用该 final manifest。
- [ ] **FAIL**：保持上一版本；记录失败原因并重新采集或拟合。

发布版本/manifest SHA-256：

```text

```
