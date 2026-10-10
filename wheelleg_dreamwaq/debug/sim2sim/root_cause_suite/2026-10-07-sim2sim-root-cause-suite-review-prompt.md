# Core V1.16 P20-S Golden 可编译性勘误独立复核 Prompt

请作为严格、独立的刚体动力学、MuJoCo 3.14 与实验证据审查者，对 Core V1.16 的单点实施期勘误做只读定向复核。不要修改任何文件，不要开始实现，不要训练，也不要运行长时间仿真。

V1.15 已完成文档复核并批准实现。实施 P20-S 单刚体读回 golden 时，MuJoCo 3.14 对冻结的 `diaginertia="0.031 0.047 0.083"` 直接报错 `inertia must satisfy A + B >= C`。数值上 `0.031 + 0.047 = 0.078 < 0.083`，因此原参数违反刚体惯量三角不等式。V1.16 只把第三主惯量改为 `0.073 kg m^2`，并明确禁止用 `balanceinertia` 静默重写输入。

## 主设计身份

```text
E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite\2026-10-07-sim2sim-root-cause-suite-design.md
版本：RootCauseSuiteCoreV1.16
SHA256：363720317500D6F98A4BBA81CFC7F2E9B5EF5A774974418542A46F602C072FE6
```

请先核对 SHA256。哈希不符时停止复核并报告身份错误。

## 权威上下文

```text
E:\wheel_leg_rl-main\docs\2026-10-03-wheelleg-dreamwaq-architecture.md
Architecture v0.21
SHA256 5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE

E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite\2026-10-07-sim2sim-root-cause-suite-implementation-plan.md
E:\wheel_leg_rl-main\wheelleg_dreamwaq\sim2sim\mujoco\models\wheel_leg_urdf4_v1.xml
```

## 本轮只审查以下问题

1. `[0.031,0.047,0.073] kg m^2` 是否满足三个主惯量均为正、互不相等，且每一项不大于另两项之和。
2. 把 `0.083` 改为 `0.073` 是否保留原 golden 对非零 COM offset、非恒等惯量旋转、非轴对齐 COM 线速度/角速度、关于 COM 与 world origin 的角动量，以及三类 mutation 的覆盖能力。
3. 禁止 `balanceinertia` 是否是正确的 fail-closed 要求，避免编译器静默改变冻结输入。
4. V1.16 是否仅修正不可编译参数，没有增加 probe、改变正式模型/策略/阈值/verdict 或扩大 Core 范围。
5. 是否存在比 `0.073` 更小的必要修改；纯风格或“可再增强”的建议不构成阻塞。

## 审查规则

- 不重新审查 V1.15 已关闭的 Windows audit、物理矩阵、策略身份和 Core/Extended 边界。
- 只有影响正确性、可实现性、可审计性或范围控制的问题才报告。
- 每个发现必须给出设计文档准确行号和本地证据。
- 严重级别：P0 为正式工程写入/证据不可恢复；P1 为 golden 仍不可实现或会产生错误物理证据；P2 为重要非阻塞歧义；P3 为非阻塞澄清。

请输出：

1. 总体结论：`批准`、`有条件批准` 或 `阻塞`，以及 P0/P1/P2 数量。
2. 按 P0 到 P3 排序的发现，含准确行号、证据、影响、最小修改。
3. 惯量合法性逐项计算。
4. golden 与 mutation 覆盖是否保持。
5. 范围泄漏和正式工程写入风险检查。
6. 是否批准恢复 Core V1 实施；只有 P0/P1 均为 0 才能批准。

不要修改文件；把完整复核报告直接返回。
