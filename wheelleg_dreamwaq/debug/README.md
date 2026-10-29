# 调试工具与历史源码

当前训练源码的入口为 [training/README.md](../training/README.md)。本目录保存诊断工具、历史源码副本与测试缓存。

| 目录 | 用途 |
| --- | --- |
| `sim2sim/` | 跨仿真器诊断、轨迹分析及隔离实验 |
| `no_action_clip/` | 原始动作相关实验和回放工具 |
| `archive/source_snapshots/` | 实验修改前保存的源码，保持原文件内容 |
| `archive/pace-github-upload/` | PACE 历史上传包 |
| `archive/build-metadata/` | 旧包安装生成的元数据 |
| `cache/` | 测试临时文件、编译和运行缓存 |
| `workspace-layout-20261028/` | 本次移动清单与文件完整性核对工具 |

## 旧源码位置与当前位置

下表路径均相对工程根目录。

| 旧位置 | 当前位置 |
| --- | --- |
| `source/wheelleg_dreamwaq/wheelleg_dreamwaq/` | `training/wheelleg_dreamwaq/` |
| `scripts/` | `training/scripts/` |
| `recovery/*.py`、`recovery/tests/` | `training/recovery/` 中对应文件 |
| `artifacts/phase2_dreamwaq/baseline-heading-implementation-20261016/source/` | `debug/archive/source_snapshots/baseline-heading-20261016/source/` |
| `artifacts/phase2_dreamwaq/leg-reversal-trial-20261011-v1/source-before/` | `debug/archive/source_snapshots/leg-reversal-20261011/source-before/` |
| `artifacts/phase2_dreamwaq/split-action-smooth-20261013-v1/before/source/` | `debug/archive/source_snapshots/split-action-smooth-20261013/source/` |
| `artifacts/phase2_dreamwaq/split-action-smooth-20261013-v1/before/sim2sim/` | `debug/archive/source_snapshots/split-action-smooth-20261013/sim2sim/` |
| `artifacts/debug/sim2sim/open-beni-reward-review-20261011/source/` | `debug/archive/source_snapshots/open-beni-reward-review-20261011/source/` |
| `artifacts/debug/sim2sim/remove-artificial-speed-limits-20261011-v1/source-before/` | `debug/archive/source_snapshots/remove-speed-limits-20261011/source-before/` |
| `PACE/build/github-upload/` | `debug/archive/pace-github-upload/` |
| 根目录中名称包含 `task7-test-v116`、`task7-test-v116b` 的异常路径 | `debug/cache/task7-test-v116/`、`debug/cache/task7-test-v116b/` |

`logs/`、`artifacts/` 和 `recovery/artifacts/` 继续保留实验配置、模型及报告的位置。历史报告记录当时的源码路径与身份；查找已移动的文件时使用上表。旧实验的冻结身份校验继续适用于其原版本。

归档和缓存默认由 Git 忽略；原先已被 Git 跟踪的 split-action-smooth 快照和 open-beni 参考源码在新位置继续保留版本记录。

在工程根目录运行以下命令，可检查移动文件是否完整、归档内容及关键模型证据的 SHA256 是否改变：

```powershell
& './debug/workspace-layout-20261028/move_directories.ps1' -Verify
```

测试结果及未通过的历史调试检查见[目录整理验证记录](workspace-layout-20261028/validation.md)。
