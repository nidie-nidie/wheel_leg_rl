# Sim2Sim Root-Cause Suite Core V1.16 独立代码复核 01

- 日期：2026-10-08
- 复核方式：独立 agent，只读，不修改文件
- 正式架构：Architecture v0.21
- 正式架构 SHA256：`5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE`
- 冻结设计：RootCauseSuiteCoreV1.16
- 冻结设计 SHA256：`363720317500D6F98A4BBA81CFC7F2E9B5EF5A774974418542A46F602C072FE6`
- 首轮结论：不通过，`P0=4`、`P1=5`、`P2=2`

## P0

1. G00-G03 未完整执行冻结设计要求，但 `finalize.py` 固定写入空 `failed_gates`，可能在证据门未成立时生成高置信度物理结论。
2. P30/P40/P50 ratio pair 只在 catalog 声明，分析前未验证 configuration、三相 reset、excitation、comparison profile 和 semantic allow-list。
3. P50 t=0 接触样本、coupon property、有效接触和 normal gate 没有形成 fail-closed 链，可能误报 normal 或 tangential primary。
4. `verify/report` 信任可编辑的 manifest stage 列表，finalize 重新读取未经显式验证的 latest result，存在联合篡改绕过路径。

## P1

1. PowerShell 在 run-id canonical 校验前创建目录；worker 在第三方 import 前没有统一 bootstrap write guard。
2. resume 未绑定 suite source、catalog、解释器、正式输入、worker argv/env 指纹和显式 selected attempt。
3. P20 把 property mismatch 与 measurement/golden failure 混成 stage failure，导致质量/惯量 mismatch 无法形成 verdict。
4. P60 replay source 未在 fresh/MuJoCo replay 和 C70 前原子封印；P60-B/C 分析未核验 reset-cache 行与算法身份。
5. C70 缺 plant-material-failure 前置门；evidence 没有展开到 anchor x group x sign x reduction 粒度。

## P2

1. `report` 先验证现有 final report，报告丢失或损坏时无法从已验证 stage 重建。
2. CLI 未实现冻结契约中的 identity drift=`3` 和 schema/hash contradiction=`5`。

## 处置要求

- 所有 P0/P1 修复并由第二次独立代码复核清零后，才允许运行真实 Core 诊断。
- P2 修复或给出可审计的不采纳理由；本轮选择全部修复。
- 任何采集或分析代码变更后，重跑受影响单元/集成测试并再次独立代码复核。
- 完整运行后另行调用独立 agent 复核结果、阈值、trace、verdict 和报告。
