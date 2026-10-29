# 同源平面查询的浮点区间

实现版本：`RecoveryPlaneSupportFP64SafeLowerBoundV2`。

输入为物理引擎原始float32 body位置和四元数；提升到float64不会损失输入数值。所有27个body先检查有限性，四元数按float64范数与NumPy `allclose(norm, 1, rtol=1e-5, atol=1e-5)` 判断。范数距离边界不足 `16u (sum(q²)+1)` 时，独立validate请求用原始输入执行NumPy参照。body到geom使用生产GeometryInspector相同的显式四元数多项式、geom_pos及编译geom_quat旋转。支持点使用编译mesh_graph的实际顶点，没有重新构造凸包。

设binary64舍入单位 `u=2^-53`，`γ_k=ku/(1-ku)`。对每个环境，定义 `P=max_body sum(abs(position))`、`Q=max_body sum(abs(quaternion))`。常量包络为 `C=1+max_geom |geom_pos|₁+max_geom |local_rotation|₁+max_vertex |vertex|₁+|plane_position|₁+|plane_normal|₁`，所有值在初始化时用同一份编译float64常量求得。使用正数包络 `M=(1+P)(1+Q)^8(1+C)^3`。

对任意相同支持顶点，CPU与GPU距离表达式仅由四元数二次多项式、两层3×3乘法、三项点积和平移构成。任意顶点距离的单条依赖路径少于128个二元运算；即使将每个三项归约按任意顺序展开，两个实现的总表达式运算数量仍少于1024。乘法节点的误差传播至输出时，最多经过四元数的二次项和三个常量因子的乘积。合法四元数使Q不小于0.99998；设L=1+Q，旋转矩阵各项绝对值之和不超过4L²。完整距离的绝对项总量不超过CP+4L²C²+C²+4L²C³，M中的L⁸及(1+C)³覆盖这些系数及传播。用4096计数同时覆盖CPU与GPU表达式的舍入、不同乘加归约顺序及M本身的计算舍入，得到同一顶点的保守绝对区间 `ε=nextafter(γ_4096 M+tiny,+∞)`。真实样本最大差不参与ε的构造。顶点min与distance_limit截断均为1-Lipschitz，不扩大绝对误差。CUDA float64不经过TF32或autocast；融合乘加的单次舍入也包含在此上界内。

编译常量来自当前实际模型，其规模有限；原始float32输入提升到float64后，合法近单位四元数及有限float32平移使距离运算远离float64上溢。初始化还核验平面与mesh类型、零margin/gap及常量有限性；使用超出已验证常量契约的模型必须拒绝。

原生支持点选择需要单独处理。实际MuJoCo 3.14.0的`mjc_PlaneConvex`先调用凸包图hill-climb支持函数，再加入该支持点所在凸包面的至多三个顶点；`mj_geomDistance`取这些返回contact距离的最小值。初始化固定该已审查版本，要求mesh存在凸包图且顶点数达到原生hill-climb阈值，并检查所有mesh polygon顶点都属于导出集合。没有假定浮点hill-climb一定取得全顶点global min。每个原生返回点都来自同一有限集合，故原生返回距离满足 `d_native >= d_gpu-ε`，这是单向下界；不能由 `d_gpu+ε < -0.005` 证明原生越界。

只有 `d_gpu-ε >= -0.005` 才直接确认未越界。其余所有环境，包括GPU明显穿透的状态，通过最多256行的固定共享槽位，发送同一源帧原始float32 body位姿，按原生产NumPy显式float64公式及mj_geomDistance同步精查。精查使用单独floor消息，不推进内部diagnostic offset。全部6144环境需要复核时仍分24个固定容量块处理。门限没有调整，最终判断严格保留原生 `penetration > 0.005`。GPU安全分支报告`max(0,-d_gpu+ε)`作为原生penetration的保守上界；CPU精查分支报告原生精确值。identity和statistics明确标记此报告语义；`absolute_distance_error_bound_m`约束同一支持点表达式差，不能解释为两个不同支持点选择结果的双向误差。

原生路径依据：[MuJoCo 3.14.0 engine_collision_convex.c](https://github.com/google-deepmind/mujoco/blob/3.14.0/src/engine/engine_collision_convex.c)、[engine_collision_convex.h](https://github.com/google-deepmind/mujoco/blob/3.14.0/src/engine/engine_collision_convex.h)、[engine_support.c](https://github.com/google-deepmind/mujoco/blob/3.14.0/src/engine/engine_support.c)。对应源码副本保存在性能审计artifact目录。

此区间只约束当前外部MuJoCo凸包地面检查，与PhysX cooked mesh或真实非凸实体是否相同属于独立验证事项。
