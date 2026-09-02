# 失败记录

失败实验也要记录。建议字段：

- 日期
- 代码版本
- 任务
- action space
- observation space
- reward scales
- 失败现象
- 怀疑原因
- 证据
- 下一步修改

## 目前已经遇到过的类型

- URDF mesh 路径或扩展名导致 Isaac 导入失败。
- dummy link 原始质量和空 mesh 导致模型看起来不合理。
- reward / observation 出现极大值，进一步导致 value loss 变成 `inf`。
- 4096 环境加粗糙地形后，每个 iteration 的采样时间明显变长。

## 需要学习

- 失败记录不是写结论，而是保存证据：日志、曲线、命令、checkpoint、截图。
- 训练崩掉时先区分启动失败、仿真数值爆炸、PPO loss 发散、策略学歪。
- `nan/inf` 出现后通常不要继续沿用旧 run，要先修环境再重开。
- 性能问题要看 `collection time` 和 `learning_time`，不要只看总耗时。
