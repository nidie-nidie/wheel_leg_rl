# 训练源码目录整理

目标：在 `wheelleg_dreamwaq/training/` 集中当前行走、起身和训练入口，在 `debug/` 保存实验源码副本和缓存。

## 目录安排

- `training/wheelleg_dreamwaq/`：原行走 Python 包，保留导入名称。
- `training/recovery/`：起身 Python 包、测试和功能说明，保留导入名称。
- `training/scripts/`：训练、回放、导出、评估及环境工具入口。
- `debug/archive/source_snapshots/`：四组历史实验源码快照、配套 MuJoCo 快照和 open-beni 参考源码，移动前后核对全部文件哈希。
- `debug/archive/pace-github-upload/`：已有 PACE 上传副本。
- `debug/cache/`：两个异常命名的测试产物和旧源码缓存。
- `configs/`、`tests/`、`sim2sim/`、`PACE/`、`dependencies/`：保持现有职责。
- `logs/`、`artifacts/`、`recovery/artifacts/`：保留现有实验数据及固定路径；根目录 `recovery/` 仅作为历史产物位置。

## 执行与检查

- [x] 记录当前源文件、移动对象和关键模型证据的哈希；检查当前测试结果。
- [x] 移动当前源码和历史副本；通过迁移清单保留旧路径到新路径的对应关系。
- [x] 调整 Python 包配置、项目根路径、脚本调用与当前测试路径。
- [x] 让源码摘要读取新训练目录；旧模型与报告保留原身份，严格恢复规则继续生效。
- [x] 更新当前入口说明和训练代码阅读顺序。
- [x] 更新本机两个 Python 环境中的项目可编辑安装，核实实际导入位置。
- [x] 运行行走单元测试、起身测试、MuJoCo 检查及训练入口检查。
- [x] 核对移动副本和关键模型证据的哈希；登记实际验证结果及限制。

本次调整目录和路径，不启动新的训练，不更改奖励、动作或物理参数。

实际结果与历史调试限制见[目录整理验证记录](../../../wheelleg_dreamwaq/debug/workspace-layout-20261028/validation.md)。
