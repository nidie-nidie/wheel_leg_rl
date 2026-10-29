# 2026-10-28 目录整理验证

当前源码、Python 安装配置和调用路径已迁移；未启动新的训练。阅读入口为 [training/README.md](../../training/README.md)。

## 已通过的检查

| 检查范围 | 结果 |
| --- | --- |
| 行走 `tests/unit` | 277 项通过；迁移前也是 277 项通过 |
| Isaac 环境中的起身测试，排除独立 MuJoCo 测试文件 | 106 项及 32 项子测试通过；与迁移前一致 |
| MuJoCo 测试及起身的 MuJoCo、原生几何测试 | 99 项通过 |
| 最终训练套件入口测试 | 11 项通过 |
| 两个 Python 环境的实际导入 | `wheelleg_dreamwaq`、`recovery` 均来自 `training/` |
| 行走 PPO、DreamWaQ、两种训练套件和起身训练帮助命令 | 均退出 0；起身帮助命令同时指定已有 `--config` |
| `training/scripts/check_runtime.py` | 退出 0；RTX 5070 CUDA 实际计算、依赖版本及 Isaac Lab 版本检查通过 |
| Python 语法及当前导航链接 | 检查通过 |
| 54 个迁移对象中的文件 | 837 个文件存在；归档文件逐个 SHA256 不变 |
| 关键历史证据 | `model_2000.pt`、`first_episode_trace.jsonl`、`replay_report.json` 的 SHA256 不变 |
| `git diff --check` | 通过 |

起身和 MuJoCo 首次并行运行时出现内存分配失败；逐组运行后得到表中结果。不同依赖的调试测试分别使用主 Python 环境和 MuJoCo Python 环境。

## 历史调试测试的限制

额外检查 `debug/sim2sim/tests` 时，主环境一组得到 78 项通过、2 项失败（包含 11 项训练套件测试）；MuJoCo 一组得到 13 项通过、10 项失败。12 项失败均先遇到旧冻结模型清单的哈希不一致，未通过对应历史实验检查。

这些测试要求 `sim2sim/mujoco/model_manifest.json` 的 SHA256 为 `C3DB4FE2797F6AC3628F0A9CB5E0B166FFD0CE802776A335922C474A045F7AE0`；当前文件为 `E83642558F85AE261F60FBC90A38BE830B61A9F438F43A33E34C7726512C6811`。当前文件与 Git HEAD 中该文件的 SHA256 相同，`git diff` 为空。本次未修改该清单，未更新冻结哈希，也未放宽身份检查。

目录调整改变当前源码摘要。旧训练记录、checkpoint 和正式证据继续保留原身份；本次未验收旧 checkpoint 在新身份下续训、完整 Isaac 仿真、起身能力或策略交接成功率。

本地完整移动对应关系和迁移前哈希保存在 `relocations.json`，该清单由 Git 忽略；旧路径概览见 [debug/README.md](../README.md)。可在工程根目录运行 `./debug/workspace-layout-20261028/move_directories.ps1 -Verify` 复查。
