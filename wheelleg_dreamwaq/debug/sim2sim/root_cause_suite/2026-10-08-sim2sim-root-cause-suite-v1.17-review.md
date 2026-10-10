# RootCauseSuiteCoreV1.17 定向独立复核结果

**复核日期：** 2026-10-08  
**设计 SHA256：** `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD`  
**结论：** 批准  
**计数：** `P0=0, P1=0, P2=0`

## 技术结论

- 正式 MuJoCo XML 中六个 controlled hinge `jIJ,jIO,jAB,jAG,jwheel_left,jwheel_right` 的 compiled `dof_damping` 均为 0。
- 其余 20 个 passive robot hinge 的正式 `dof_damping` 均为 0.05；debug-only drive-off builder 会把完整 26-hinge 集合清零，并在写出后重新核验正式源文件哈希。
- MuJoCo 正式 target 控制由每个物理子步执行的外部显式腿部位置 PD 与轮部速度伺服实现，不依赖六个 controlled hinge 的 `dof_damping`。
- 因此 P30 target/direct 的 actual exact diff 只能包含 20 条 passive `dof_damping` 路径；六个 unchanged controlled 值必须作为不变量完整记录，不能伪造为变化路径。
- 修订没有新增 probe、阈值、正式模型变换、控制器修改或 verdict 规则。

## 复核过程

第一次定向复核确认技术修订正确，但发现文档末尾仍保留 V1.16 旧门禁，结论为 `P0=0, P1=1, P2=0`。设计随后新增 §17.17，并将 §18 完整同步为 V1.17 当前门禁。第二次只读复核确认旧门禁已删除，未引入新问题。

**批准按 V1.17 同步代码契约、factor-path allow-list、compiled semantics 与 mutation tests。**
