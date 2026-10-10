# Isaac 首次启动未进入物理测试

第一次执行七组 runner 时，三个 Isaac 启动都因用户缓存位置失败，未生成输出目录和 evidence.json。Kit 返回进程 exit=0，因此只看退出码不足以判定测试完成。交互工具当时捕获了以下异常片段：

```text
FileExistsError: [WinError 183] ... 'C:\Users\changba01\AppData\Local\NVIDIA\warp\Cache\1.8.2'
```

runner 随后为子进程设置工程内 WARP_CACHE_PATH / CUDA_CACHE_PATH / NV_COMPUTE_CACHE_PATH，并要求 evidence.json 和 PROBE_COMPLETE 同时存在。三个 Isaac 新进程成功重跑。

初次重跑时因输出目录不存在复用了原日志文件名，三个首次启动的完整 console/Kit 日志被覆盖，不能作为独立原始日志追溯。此处记录该限制，保留 execution.json 的原始启动参数、退出码、耗时，并把对应记录显式标为 evidence_complete=false；最终分析只使用随后实际成功的进程记录。今后 runner 也以已记录的输出名称分配 attempt，避免再次覆盖。

此次缓存处理只影响诊断进程的缓存位置，不改仿真配置、物理输入或正式工程。最终 baseline 逐值复现此前同力矩测试。
