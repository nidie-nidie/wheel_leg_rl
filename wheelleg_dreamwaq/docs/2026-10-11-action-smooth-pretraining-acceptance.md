# action_smooth 加倍：训练前验收

2026-10-11，Architecture v0.27，stop_reverse_smooth_v1。用户授权一种子从零1000轮，不使用 resume。

主工程单测186项通过。真实Isaac新profile集成1项通过，耗时74.77秒，包含fresh两轮、同/跨规模resume到三轮、跨奖励profile拒绝、固定命令play、export/golden、当前Isaac八场景报告、MuJoCo动态入口smoke。

新旧profile的命令、采样RNG与除action_smooth外所有奖励相同；实际非零action_smooth扣分恰好加倍。旧checkpoint配置识别保留。110个正式物理/模型/其它源码文件哈希不变，四组正式门槛保留。此次自行复核，不调用未授权agent。

详细证据在 artifacts/phase2_dreamwaq/action-smooth-acceptance-20261011/acceptance.json；测试权重不用于正式性能验收。

下一步seed41882943，rtx5070（256env/8mini-batches），fresh1000轮。只action_smooth=-0.02，action_rate=-0.01，vx两项=2。训练中冻结源码与架构；监控1800秒；之后导出/golden、Isaac八场景、独立MuJoCo八场景、停车换向和逐关节停车轨迹对比原1000轮模型。
