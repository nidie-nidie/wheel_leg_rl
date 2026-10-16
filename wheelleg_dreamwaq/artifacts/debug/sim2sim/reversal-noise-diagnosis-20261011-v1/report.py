"""Build a concise report and scientific plot from verified diagnostics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
r=json.loads((OUT/"results.json").read_text(encoding="utf-8"))
parking=[next(x for x in r["comparisons"] if x["arm"]==a and x["scenario"]=="nominal_stand") for a in ("baseline","budget12","budget8")]
forward=next(x for x in r["comparisons"] if x["arm"]=="budget12" and x["scenario"]=="forward")
fmt=lambda v:" / ".join(f"{x:.2f}" for x in v[:2])
table=[]
for x in parking:
    table.append(f"| {x['arm']} | {fmt(x['mean_only']['target_rate_mean_per_s'])} | {fmt(x['same_histories_with_noise']['target_rate_mean_per_s'])} | {fmt(x['held_input']['with_noise']['target_rate_mean_per_s'])} |")
baseline,n12,n8=parking
pair_change=lambda name:((np.array(n12["mean_only"][name][:2])/np.array(baseline["mean_only"][name][:2])-1)*100).tolist()
excursion_changes=pair_change("mean_event_excursion_rad")
step_changes=pair_change("target_step_rms_rad")
vel_changes=((np.array(n12["actual"]["velocity_rms_rad_s"][:2])/np.array(baseline["actual"]["velocity_rms_rad_s"][:2])-1)*100).tolist()
cost_before=baseline["mean_only"]["budgets"]["12"]["full_gain_penalty_rate_per_s"]
cost_after=n12["mean_only"]["budgets"]["12"]["full_gain_penalty_rate_per_s"]
cost_change=(cost_after/cost_before-1)*100
std_rad=np.array(n12["effective_action_std"][:2])*.35
report=f"""# 换向惩罚、探索噪声与实际抽搐诊断

本次确认：**部署策略本身仍有周期振荡；新增“只数换向次数”的惩罚同时对探索噪声非常敏感，并且忽略了抽动幅度。** 这解释了为什么该试验不能按“计数下降”判断腿已变平稳。它还不是左腿闭环振荡的完整根因定位，也不能证明换奖励后必然解决。

只新增诊断脚本、捕获数据、图和报告；没有训练，没有改生产奖励、控制、网络、架构、随机化或模型，也没有调用子agent。

## 1. 部署抽搐不是MuJoCo额外加了动作探索噪声

重放三个200轮模型的停车/前进/后退，各500帧，共4500帧。使用旧CSV原动作重放原runtime，补存history、实际关节位置与速度。每帧重算TorchScript输出；动作均与旧记录一致，网络输出/目标/力矩/速度/加速度/所有旧metrics最大差均为0.0。原轨迹未发生漂移。

Actor训练接口从Normal分布采样动作；部署接口输出mean和确定性context_mu，没有动作分布采样。因此在MuJoCo中看到的持续左腿运动已经存在于确定性闭环。不能把它归因于运行时仍在“随机探索”。N12停车的目标与实际速度主峰均9.125Hz，baseline均9.75Hz，N8均10Hz。

## 2. 但训练用的计数很容易被探索采样拉高

固定每条重放history，使用每个模型checkpoint的真实逐动作标准差，生成64条独立采样。所有模型复用同一组标准正态噪声seed8642011。采样动作依原规则裁剪/映射为最终腿角度目标，采用原0.01rad迟滞、1s滑窗计数；采样动作没有施加到物理轨迹。

另取第250帧history，将同一个输入固定500步：网络mean完全不变，mean-only换向率为0；单独加入原标准差的噪声后，左腿仍约31次/秒。这是“噪声可以单独推高指标”的因果对照，不是完整训练闭环的因果分解。

以下单位是次/(关节·秒)，左腿顺序jIJ/jIO；窗口为原动作后2s<t≤10s，共400帧。

| 组别（停车） | 沿原history的mean动作 | 相同history加原探索噪声 | 固定单一history只加噪声 |
| --- | --- | --- | --- |
"""+"\n".join(table)+f"""

N12左腿动作标准差实际为{fmt(n12['effective_action_std'])}，不是全六维mean_noise_std日志约0.71。在未裁剪区域按0.35 rad动作尺度换算为{std_rad[0]:.4f}/{std_rad[1]:.4f} rad，高于0.01rad计数门槛。不能把全六维平均标准差当作每个腿的标准差。

N12前进案例更能说明判别问题：确定性目标反向仅{fmt(forward['mean_only']['target_rate_mean_per_s'])}次/秒，低于12预算；相同输入加原噪声后却为{fmt(forward['same_histories_with_noise']['target_rate_mean_per_s'])}次/秒，远超预算。因此计数超预算不能直接等同于“部署mean已经持续抽搐”。

固定输入的采样也足以使本调权公式达到1.0；真实训练日志两个自适应组均197/200个gain_used监控点为1.0。这与探索噪声使调权常封顶的假设吻合，但本次无法算出真实训练成本中多少百分比由探索引起。均值动作、观测噪声、随机化、物理反馈及episode重置会共同影响真实轨迹。

训练采样动作会真的交给电机，它们不是无害的“测量误报”；问题是当前指标把探索产生的目标折返和部署策略的持续振荡混在同一个惩罚里。

## 3. 只数反向次数，会漏掉“次数少一点，但每次动得更猛”

N12停车相比匹配baseline：

- 左腿换向次数约下降6.8%。
- 从局部极值反向时的平均目标幅度增加{excursion_changes[0]:.1f}% / {excursion_changes[1]:.1f}%。
- 每20ms目标增量RMS增加{step_changes[0]:.1f}% / {step_changes[1]:.1f}%。
- 实际角速度RMS增加{vel_changes[0]:.1f}% / {vel_changes[1]:.1f}%。

在相同N12规则、gain=1下离线算成本，N12目标的成本速率相对baseline下降{abs(cost_change):.1f}%（{cost_before:.5f}→{cost_after:.5f}），但机械运动更大。baseline训练时并未启用该项；这里只是同规则评估两条目标序列。

合成验证也得到明确结果：同样5Hz、0.02rad与0.20rad振幅的正弦目标，都计为每秒10次反向；目标变化幅度相差10倍，计数和计数成本仍相同。这里0.01rad只是事件门槛，一旦越过就按一次计，未按幅度继续加重惩罚。

[诊断汇总图](diagnostic-summary.png)

## 4. 原动作裁剪还提供了一个“计数很低”的途径

将固定raw mean从正常范围移至±2，在N12左腿原标准差下，输出经过[-1,1]裁剪后，本次噪声样本的左腿换向计数降为0。此结果说明指标允许靠目标常贴裁剪边界降低成本，而不是必须学会更健康的控制。

三个模型的右腿raw mean中位数约-3.3至-4.1，部署动作全程裁为-1，右腿目标换向率为0；机械速度并不为0。baseline已经存在这一现象，所以不能断言新增惩罚造成了右腿饱和，也不能拿右腿的0直接当左腿健康阈值。

## 后续判断

当前有证据优先重新设计新增约束的衡量与调权输入，而不是立即更换PPO/CENet或把这版训练扩大。原六维action_smooth可继续保留；若试验新成本，应同时衡量腿目标变化幅度和实际运动，保留速度、姿态、存活与停车换向验收。

“单独统计mean动作”适合诊断，但不能直接假定把环境奖励里的采样动作换为mean动作就能正确训练。若改变惩罚对象，还要单独设计它怎样进入PPO学习；本次没有做这一实现选择。也未批准降低探索噪声、加动作滤波、加硬限制或直接扩大惩罚权重。

## 证据与复核

[results.json](results.json)保存9个场景的逐模型、逐关节结果、sigma倍率敏感性、64条采样分布、固定mean扫描、合成验证及训练日志。副本输入见captures/*.npz。原轨迹核对见[capture-verification.json](capture-verification.json)，完成保护校验见[verification.json](verification.json)。

独立NumPy计数器与共享Torch计数器核对{r['verification']['counter_joint_frames']}个关节帧，计数/事件最大差0.0；原生Python与部署图抽查相同batch输入的mean最大差0.0；目标映射float32/float64最大差{r['verification']['target_mapping_max_error']:.3g} rad。目标均值数字的聚合采用float64。

精确代码依据：
- actor_critic.py:183/194/198/205：有效标准差、训练采样、部署mean。
- env.py:655/661/801：计数对象为实际采样动作映射后的最终腿目标，新成本加入环境奖励。
- leg_reversal.py:94/145/159：0.01rad事件、只按计数超预算计算成本、gain封顶公式。
- wheelleg_mujoco/control.py:32/33：动作裁剪与最终角度目标。

这次修正过诊断脚本读取标准差的shape处理：先前已初始化distribution导致action_std从[6]变为[1,6]；现使用_effective_std(mean[:1])[0]，与实际分布逐值核对。生产代码没有因此修改。

复现：先在sim2sim/mujoco/.venv中运行capture.py，再在项目.venv中运行analyze.py和report.py。前者重放原动作，后者只离线采样。所有结果只属于一个种子和既有三组模型，不是新增正式训练验收。
"""
(OUT/"report.md").write_bytes(report.encode("utf-8"))

font_path=Path(r"C:\Windows\Fonts\msyh.ttc")
if font_path.exists():
    prop=FontProperties(fname=str(font_path))
    plt.rcParams["font.family"]=prop.get_name()
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams["font.size"]=10
fig,axes=plt.subplots(2,2,figsize=(12.5,8.0),constrained_layout=True)
colors=["#4374b5","#da8131"]
def pair_bars(ax,labels,pairs,ylabel,title,number_format="%.1f"):
    positions=np.arange(len(labels)); width=.34
    for j in range(2):
        bars=ax.bar(positions+(j-.5)*width,[p[j] for p in pairs],width,color=colors[j],label=("左前 jIJ","左后 jIO")[j])
        ax.bar_label(bars,fmt=number_format,fontsize=8,padding=2)
    ax.set_xticks(positions,labels); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True); ax.legend(fontsize=8)
pair_bars(axes[0,0],["原轨迹mean","原输入加噪声","固定mean加噪声"],
    [forward["mean_only"]["target_rate_mean_per_s"][:2],forward["same_histories_with_noise"]["target_rate_mean_per_s"][:2],forward["held_input"]["with_noise"]["target_rate_mean_per_s"][:2]],
    "反向次数 / 秒","N12前进：固定输入对照，采样未施加到物理")
axes[0,0].axhline(12,color="#ad3e3e",linestyle="--",label="12次/秒预算")
axes[0,0].set_ylim(0,38)
axes[0,0].legend(fontsize=8)
pair_bars(axes[0,1],["baseline","N12","N8"],[x["mean_only"]["target_rate_mean_per_s"][:2] for x in parking],
    "反向次数 / 秒","停车：目标反向次数")
pair_bars(axes[1,0],["baseline","N12","N8"],[[v*1000 for v in x["mean_only"]["target_step_rms_rad"][:2]] for x in parking],
    "20ms目标增量 RMS / mrad","停车：目标变化幅度")
pair_bars(axes[1,1],["baseline","N12","N8"],[x["actual"]["velocity_rms_rad_s"][:2] for x in parking],
    "实际角速度 RMS / rad/s","停车：实际机械运动",number_format="%.3f")
fig.suptitle("次数减少，不代表幅度变小；探索采样也能使计数超预算",fontsize=14)
fig.savefig(OUT/"diagnostic-summary.png",dpi=180)
plt.close(fig)
print(json.dumps({"report":str(OUT/"report.md"),"plot":str(OUT/"diagnostic-summary.png"),
 "N12_event_amplitude_change_percent":excursion_changes,"N12_target_step_change_percent":step_changes,
 "N12_actual_velocity_change_percent":vel_changes,"N12_hypothetical_cost_change_percent":cost_change},ensure_ascii=True))
