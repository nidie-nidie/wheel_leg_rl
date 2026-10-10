"""Render verified diagnostic numbers and preserve a standalone scientific plot."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import ROOT, read, digest


def main():
    base, cap = read(ROOT / "analysis.json"), read(ROOT / "cap-analysis.json")
    assert base["passed"] and cap["passed"]
    for result in (base, cap):
        for path, expected in result["provenance_sha256"].items(): assert digest(path) == expected, path
    def b(case, ms):
        return next(r for r in base["aggregate"] if r["case"] == case and r["time_ms"] == ms)
    def c(case, value, ms):
        return next(r for r in cap["aggregate"] if r["case"] == case and r["cap_deg_s"] == value and r["time_ms"] == ms)
    rows = []
    for label, get in (("原条件", lambda ms: b("baseline", ms)),
                       ("只关闭被动阻尼", lambda ms: b("passive_off", ms)),
                       ("只关闭闭合约束", lambda ms: b("closure_off", ms)),
                       ("只将 MuJoCo dt 改为 5 ms", lambda ms: b("dt_5ms", ms)),
                       ("保留闭合，只提高 Isaac 刚体角速度上限", lambda ms: c("baseline", 10000, ms)),
                       ("关闭闭合，并提高 Isaac 刚体角速度上限", lambda ms: c("closure_off", 10000, ms))):
        rows.append(f'| {label} | {get(5)["angular_error_l2_rad_s"]:.6f} | {get(20)["angular_error_l2_rad_s"]:.6f} | '
                    f'{get(20)["relative_angular_error"]:.2%} | {get(20)["joint_error_rms_rad_s"]:.6f} |')
    report = """# 相同力矩下：机械、阻尼、闭合与步长诊断

这次找到两个可复现的具体影响因素：**闭合机构开启时的首步约束响应，以及 Isaac 刚体角速度上限对后续响应的影响**。被动阻尼不是这 20 ms 差异的主要解释；仅把 MuJoCo 步长改成 5 ms 也没有改善。仍有残留差异，不能把所有 sim2sim 失败归结为已完全确认的单一根因。

## 实测结果

同一份 8×6 恒定外加关节力矩、26 个同名关节 q/qd 完全对齐、root 高度及归一化姿态完全对齐、初速度为零。两端绝对 x/y 保留各自场景原点；无接触、均匀重力下它们是平移等价的条件。关闭地面和六主动 PD，保留自由底座、重力、armature 及其余正式物理参数，只比较前 20 ms。八组是不同输入条件，不是统计重复。

机身角速度差为三维向量差的 L2 范数、关节速度差为六主动关节 RMS，单位 rad/s。相对差先按每个条件除以两边响应幅度平均值，再对八条件取平均。

| 干预 | 5 ms 角速度差 | 20 ms 角速度差 | 20 ms 相对角速度差 | 20 ms 关节速度差 |
|---|---:|---:|---:|---:|
""" + "\n".join(rows) + """

## 两个具体线索

**1. 闭合机构改变了首步响应。** 关闭闭合后，5 ms 平均角速度差从 0.233422 降至 0.001458 rad/s，八个条件全部缩小；10 ms 从 0.180470 降至 0.004405 rad/s，也全部缩小。首步提高刚体角速度上限没有任何 root/joint 状态变化，因此这一首步差异不能归于本次检查的角速度上限。相同输入下，闭合开启/关闭时的差异模式支持继续检查闭合约束的构造、初始残差和求解响应；目前不能指定某一个 anchor、solref、solimp 或迭代数错误。

**2. 刚体角速度上限会造成后半段差异。** 正式 `assets/wheelleg.py:51` 设 `max_angular_velocity=100.0`。本地 Isaac Lab `sim/schemas/schemas_cfg.py:92–93` 写明单位为 deg/s，`schemas.py:323` 直接按字段写入 PhysxRigidBodyAPI，没有 deg/rad 换算。100 deg/s 约 1.74533 rad/s，并非 100 rad/s。诊断器在 216 个运行时刚体 prim 上读到该值；MuJoCo 模型没有对应的这个刚体角速度上限。

仅在诊断配置中把这一个属性从 100 提高到 10000 deg/s（约 174.53 rad/s），其余 rigid-body 配置逐字段完全相同。保留真实闭合机构时，20 ms 平均角速度差从 0.431066 降至 0.056386 rad/s，相对差从 9.94% 降至 1.65%；八条件中七个改善，15 ms 八个全部改善。关闭闭合的补充组从 0.544934 降至 0.017391 rad/s，20 ms 八个条件全部改善。

这证明该属性会影响本工况的实际运动响应；不能把运行时每个 link 在每个 post-solve 快照都必须小于 1.74533 rad/s 当作判据。实测 articulated link 的解算后角速度可以超过这个数，限制的作用不能等同于对输出观测逐元素硬截断。

10000 只是避免本次测试进入该限制的诊断值，不是正式推荐值。正式架构没有在检索到的条目中声明这项应为 100 rad/s，当前也没有证据证明配置人员原意就是 100 rad/s；能确认的是单位和跨引擎限制语义存在差别及其可测影响。

## 其他检查

- **被动阻尼：** 仅把两端 20 个被动关节 damping 从 0.05 设为零，20 ms 角速度差 0.431066 → 0.430151 rad/s，变化很小。没有由此认定所有工况的阻尼实现等价。
- **步长：** 仅将 MuJoCo dt 从 1 ms 设为 5 ms，4 个物理步保持总时长 20 ms；闭合参数仍按正式值保留。20 ms 角速度差反而增至 0.864319 rad/s，相对差 26.21%。这排除了“只把两端步长设成一样就能消除当前差异”的方案，不能排除积分和闭合求解的耦合影响。
- **质量/惯量：** 复核旧 27 刚体编译属性记录，跨引擎质量和 COM 值差为零，惯量张量元素最大差 8.67e-10 kg·m²。旧固定底座、自由底座小力矩 open/closed 开发探针也复核了 trace 哈希；它们支持闭合方向，但整套 RootCauseSuite 的 G01 未通过，P30-open profile hash 不同，不能宣称旧套件验收通过，也不能用其微小力矩结果替代本次强力矩补测。

## 验证与变更边界

- 七组单项隔离 + 四组角速度上限补充，共 11 个最终有效的新物理进程；全部有 evidence.json、PROBE_COMPLETE 和 exit=0。另有三个早期 Isaac 缓存启动失败，不进入结果；完整日志覆盖的审计限制记录在 startup-failure-note.md。
- 新物理循环的 baseline 在 0/5/10/15/20 ms root pose/velocity、全部 joint q/qd 逐值复现前一轮同力矩结果。增加 link 速度观察后的 100 deg/s 组也逐值复现 baseline 和 closure_off。
- 七组共享完全相同的跨引擎初态；四个 cap 组的初态与相应 Isaac 对照逐值相同。差异不是在本次 reset 写入的 q/qd、高度、姿态或时间推进中引入的。
- 实际外加力矩、无地面接触、主动 PD 关闭及未触发主动 45 rad/s 上限均核验。每次干预只改变指定的参数；Isaac 的 armature/限位/其余驱动缓存及 MuJoCo 的其他编译语义保持一致。
- MuJoCo 四个补测各八条件无观察器重放 qpos/qvel 完全一致；fresh-shadow Jacobian 角速度与 body-frame qvel 换算误差 <1e-12 rad/s。
- 93 个正式工程、USD、XML、架构、策略与 reset-cache 文件 SHA256 不变。所有新增代码和运行时模型副本仅在此诊断目录；history、AdaBoot、奖励、命令、随机化、训练状态不改。未新增或调用子 agent。

## 对下一步的影响

先明确并对齐两端的刚体角速度限制语义，再保留闭合机构定位其首步求解差异。不要为了对齐曲线关闭真实闭合，也不要直接采用 10000 的诊断值。提高上限后保留闭合的 20 ms 相对差约 1.65%，首步仍约 28.75%，所以不能宣布 sim2sim 已修复。

暂不建议继续盲目重训。当前证据来自固定外加力矩诊断，没有第二次策略反馈，未做正式八场景策略验收；应在物理语义修复得到授权并验证后，先重新评估现有 checkpoint，再判断是否需要训练。

完整四时刻分条件向量与证明见 analysis.json / cap-analysis.json；图见 response.png。本报告是诊断结果，不变更正式架构和参数。
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    times = [5, 10, 15, 20]
    for case, label in (("baseline", "Closure on"), ("closure_off", "Closure off")):
        for value, style in ((100, "-"), (10000, "--")):
            values = [c(case, value, ms) for ms in times]
            line_label = f"{label}, cap {value} deg/s"
            axes[0, 0].plot(times, [r["angular_error_l2_rad_s"] for r in values], style, marker="o", label=line_label)
            axes[1, 0].plot(times, [100 * r["relative_angular_error"] for r in values], style, marker="o", label=line_label)
            axes[1, 1].plot(times, [r["angular_response_mean_norm_rad_s"] for r in values], style, marker="o", label=line_label)
    for case, label in (("baseline", "Baseline"), ("passive_off", "Passive damping off"),
                        ("closure_off", "Closure off"), ("dt_5ms", "MuJoCo dt = 5 ms")):
        axes[0, 1].plot(times, [b(case, ms)["joint_error_rms_rad_s"] for ms in times], marker="o", label=label)
    titles = [("Base angular velocity mismatch", "Mean vector L2 (rad/s)"),
              ("One-factor joint velocity comparisons", "Mean 6-joint RMS (rad/s)"),
              ("Angular mismatch relative to each response", "Mean per-condition relative error (%)"),
              ("Angular response magnitude", "Mean paired vector magnitude (rad/s)")]
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Elapsed physical time (ms)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(times)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Same constant torque, matched initial state, no ground (8 input conditions)")
    fig.savefig(ROOT / "response.png", dpi=160)
    plt.close(fig)
    print("REPORT_AND_PLOT_WRITTEN")


if __name__ == "__main__":
    main()
