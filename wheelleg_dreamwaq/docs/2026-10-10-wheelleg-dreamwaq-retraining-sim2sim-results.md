# DreamWaQ reset-v2 重训与 sim2sim 结果

日期：2026-10-10。使用修复后的 reset observation 环境，新建 suite `20261010-001232`，四个不同 seed 各 fresh 训练 1000 iteration；训练期间每 1800 秒由 suite 监控一次。四个训练完成后，自动完成导出、Isaac 八场景和 MuJoCo 八场景评估。

## 训练结果

| Run | Seed | Checkpoint SHA256 | Isaac | MuJoCo | 监控 |
|---:|---:|---|---|---|---|
| 01 | `1375451991` | `D15A2CFC5473234010B7C322E0C8FD8F22BAD888A6A10D673DC8ECB17B4DA342` | 8/8 | 0/8 | `phi0_symmetry_degraded` |
| 02 | `1145850885` | `72042AE31B788146B397B23F151EC73E7F5A7FF9D564D2C9429B577F4E9F3A71` | 8/8 | 0/8 | `phi0_symmetry_degraded` |
| 03 | `1380466594` | `37E2481BAE5A8CFD108ECDC35870F08D98E30E40FDEC73C23F2B69D17245F130` | 8/8 | 0/8 | 无 |
| 04 | `342388981` | `89926828678CC28E0B0BA5A8463181395E0BE95E5961F7C7A65640243D5DC5FD` | 8/8 | 0/8 | 无 |

四个训练都从 `ResumeMetadataV2 mode=fresh` 开始，完成 iteration 1000，context acceptance 和 velocity acceptance 均通过；没有 hard anomaly。四个 run 的 baseline comparison 都失败，因此 suite 的状态是 `completed`，每个 run 的状态是 `completed_acceptance_failed`。这表示训练和评估证据完整，但没有达到“DreamWaQ 不低于冻结 PPO baseline”的性能门。

## sim2sim 结果

Isaac 八场景全部完成，但平均回报分别为：run-01 `18.4621`、run-02 `18.9048`、run-03 `22.0140`、run-04 `22.0007`。四者都低于共享 PPO run-03 baseline `27.6767`。

MuJoCo 八场景全部在完成 500 tick 前失败，平均 survival fraction 分别为：run-01 `0.11275`、run-02 `0.11325`、run-03 `0.10100`、run-04 `0.10200`。按 MuJoCo ranking，run-02 是四者中相对最好，但仍为 0/8；四个 run 的 32 个场景 CSV hash 均已校验。

因此，reset 修复后的 fresh 训练没有解决 MuJoCo sim2sim 失败。下一步应先针对 Isaac 到 MuJoCo 的观测/动作/动力学映射和已记录的对称性问题做正式诊断；不应继续盲目增加训练轮数或把这四个模型标记为通过模型。

## 产物

- suite manifest：`artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2/training-suite-manifest.json`
- suite 控制台日志：`artifacts/phase2_dreamwaq/retrain-20261010-001203-launch/suite-console.log`
- 每半小时人工复核：`artifacts/phase2_dreamwaq/retrain-20261010-001203-launch/manual-monitor-001.json` 至 `manual-monitor-006.json`
- 最终完整核验：`artifacts/phase2_dreamwaq/retrain-20261010-001203-launch/final-training-sim2sim-verification.json`
- 每个 run 的导出、Isaac 报告、MuJoCo 报告和 8 个场景 CSV 均位于 suite 目录下对应的 `exports`、`isaac_evaluation` 和 `mujoco_evaluation`。

本次训练和 sim2sim 评估不等于 RootCauseSuite 正式 G00/G01/G02 或 P10-C70 门通过；旧 suite 未被续训或覆盖。
