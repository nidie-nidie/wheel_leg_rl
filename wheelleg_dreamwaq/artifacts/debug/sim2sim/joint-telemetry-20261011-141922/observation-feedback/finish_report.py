from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=Path(r"E:\wheel_leg_rl-main")
project=root/"wheelleg_dreamwaq"
session=project/"artifacts/debug/sim2sim/joint-telemetry-20261011-141922"
folder=session/"observation-feedback"
data=np.load(folder/"feedback-replay.npz")
analysis=json.loads((folder/"feedback-analysis.json").read_text(encoding="utf-8"))
cross=json.loads((folder/"feedback-crosschecks.json").read_text(encoding="utf-8"))
export=project/"artifacts/phase2_dreamwaq/resume-run03-1000-to-1500-20261011-133153/stopped-export"
manifest=json.loads((export/"policy_manifest.json").read_text(encoding="utf-8"))
with (session/"control.csv").open(newline="",encoding="utf-8") as stream:rows=list(csv.DictReader(stream))
joint_names=manifest["action"]["canonical_joint_order"]
q_before=np.array([[float(row[name+"_position_before_rad"]) for name in joint_names] for row in rows])
v_before=np.array([[float(row[name+"_velocity_before_rad_s"]) for name in joint_names] for row in rows])
q_after=np.array([[float(row[name+"_position_after_rad"]) for name in joint_names] for row in rows])
current=data["history"][:,-25:]
norm=manifest["normalization"]
sign=np.array([1.,1.,1.,1.,*manifest["action"]["wheel_joint_sign_usd"]])
normalization_errors={
    "angular_velocity":float(np.max(np.abs(current[:,:3]-np.clip(data["angular_velocity"]*norm["angular_velocity_scale"],-norm["angular_velocity_clip"],norm["angular_velocity_clip"]).astype(np.float32)))),
    "leg_position":float(np.max(np.abs(current[:,9:13]-np.clip((q_before[:,:4]-np.array(manifest["q_nominal"]))*norm["leg_position_scale"],-norm["leg_position_clip"],norm["leg_position_clip"]).astype(np.float32)))),
    "joint_velocity_and_right_wheel_sign":float(np.max(np.abs(current[:,13:19]-np.clip(v_before*sign*norm["joint_velocity_scale"],-norm["joint_velocity_clip"],norm["joint_velocity_clip"]).astype(np.float32)))),
}
command_normalized=data["command"].copy()
command_normalized[:,0]/=norm["vx_max_abs"]
command_normalized[:,1]/=norm["yaw_rate_max_abs"]
command_normalized[:,2]=(command_normalized[:,2]-norm["nominal_base_height"])/norm["height_command_span"]
normalization_errors["command"]=float(np.max(np.abs(current[:,6:9]-command_normalized.astype(np.float32))))
assert all(error==0 for error in normalization_errors.values())
gravity_norm_error=float(np.max(np.abs(np.linalg.norm(current[:,3:6].astype(np.float64),axis=1)-1)))
assert gravity_norm_error<1e-6
mask=(data["time_before"]>=2)&(data["time_before"]<=10)
action=data["action_clipped"]
previous=np.zeros_like(action)
previous[1:]=action[:-1]
previous_previous=np.zeros_like(action)
previous_previous[2:]=action[:-2]
for index in np.flatnonzero(np.r_[True,np.diff(data["episode"])!=0]):
    previous[index]=0
    previous_previous[index]=0
    if index+1<len(action):previous_previous[index+1]=0
rate=(action-previous)**2
smooth=(action-2*previous+previous_previous)**2
weights=manifest["training_reward_weights"]
dt=manifest["timing"]["control_dt_s"]
weighted_rate=weights["action_rate"]*dt*rate.sum(axis=1)
weighted_smooth=weights["action_smooth"]*dt*smooth.sum(axis=1)
vx_after=np.array([float(row["vx_mps"]) for row in rows])
vx_error=(data["command"][:,0]-vx_after)**2
positive_tracking=weights["tracking_vx"]*dt*np.exp(-vx_error/weights["tracking_sigma"])
reward_stats={
    "source":"MuJoCo recorded clipped actions evaluated using the frozen training reward formula; not an Isaac training reward measurement.",
    "scope":"All 6 actions: 4 legs and 2 wheels, normalized action space.",
    "weights":{"action_rate":weights["action_rate"],"action_smooth":weights["action_smooth"]},
    "control_dt_s":dt,
    "stationary_window_s":[2,10],
    "mean_action_rate_penalty_per_control_step":float(weighted_rate[mask].mean()),
    "mean_action_smooth_penalty_per_control_step":float(weighted_smooth[mask].mean()),
    "mean_combined_penalty_per_control_step":float((weighted_rate+weighted_smooth)[mask].mean()),
    "mean_positive_vx_tracking_reward_per_control_step":float(positive_tracking[mask].mean()),
    "combined_penalty_vs_positive_vx_reward_fraction":float(-(weighted_rate+weighted_smooth)[mask].mean()/positive_tracking[mask].mean()),
    "wheel_fraction_of_rate_and_smooth_penalty":float((rate[mask,4:].sum()+smooth[mask,4:].sum())/(rate[mask].sum()+smooth[mask].sum())),
    "training_applies_no_runtime_smoothing_filter":True,
}
(folder/"reward-smoothness-check.json").write_text(json.dumps(reward_stats,indent=2)+"\n",encoding="utf-8")
protected=json.loads((folder/"protected-before.json").read_text(encoding="utf-8"))
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()==expected for path,expected in protected.items())
verification={"passed":True,"independent_normalization_comparison_errors":normalization_errors,
              "gravity_unit_norm_max_error":gravity_norm_error,"protected_file_count":len(protected),"protected_files_unchanged":True,
              "checks":["same-time velocity","full trace replay","history rolling","previous action alignment","joint signs and scales","finite tensors","Actor reconstruction","temporal attribution completeness","multiple constant-command windows"]}
(folder/"final-verification.json").write_text(json.dumps(verification,indent=2)+"\n",encoding="utf-8")
# A short view of the measured feedback chain, using the same simulation timestamps.
plt.rcParams["font.family"]="Microsoft YaHei"
plt.rcParams["axes.unicode_minus"]=False
zoom=(data["time_before"]>=2)&(data["time_before"]<=3.2)
t=data["time_before"][zoom]
fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True)
axes[0].plot(t,data["true_velocity"][zoom,0],label="真实前后速度")
axes[0].plot(t,data["estimated_velocity"][zoom,0],label="CENet 估速")
axes[0].set_ylabel("车速（m/s）")
for index,label in [(4,"左轮"),(5,"右轮")]:
    axes[1].step(t,action[zoom,index]*25,where="post",label=label+"目标转速")
    axes[2].plot(t,current[zoom,13+index]/.05,label=label+"实际转速")
axes[1].set_ylabel("目标（rad/s）")
axes[2].set_ylabel("实际（rad/s）")
for index,label in [(0,"左前关节"),(1,"左后关节")]:
    axes[3].step(t,np.rad2deg(np.array(manifest["q_nominal"])[index]+.35*action[zoom,index]),where="post",label=label+"目标")
    axes[3].plot(t,np.rad2deg(q_before[zoom,index]),"--",label=label+"实际",alpha=.8)
axes[3].set_ylabel("腿关节角度（度）")
axes[3].set_xlabel("模拟时间（秒）")
for ax in axes:
    ax.legend(loc="upper right",ncol=2,fontsize=9)
    ax.grid(alpha=.2)
fig.suptitle("停车指令保持为零时：轮子目标快速切换，左腿目标也在振荡\n同一时刻的数据；这些曲线显示反馈关系，不能单独证明最终起因")
fig.tight_layout()
fig.savefig(folder/"feedback-chain.png",dpi=160)
plt.close(fig)
notes=f"""# 左腿抖动：观测、CENet 估速和动作平滑诊断

结论：本次日志没有发现 history 错帧、上一帧动作错位、命令/关节标准化错误或输入 NaN/Inf。CENet 有估速偏差，但在保存输入上的离线试验中，纠正估速没有减少左腿输出波动。更明显的输入依赖是轮子转速和上一帧轮子动作。尚未确定这条闭环的最终起振原因。

## 数据和验收

同一 run-03/1401 轮策略、本次用户键盘轨迹：2296 个控制帧、45920 个物理步。全部重放，原始观测/动作/前后关节状态/实际车速/力矩误差为 0；history 的 oldest-to-newest 滚动及上一帧动作误差为 0。独立用保存的关节角度、速度和 manifest 核对标准化、右轮符号及命令，误差为 0。90 个正式源码/模型/策略/原始日志文件哈希保持不变。

估速与真实速度都使用同一动作执行前时刻的控制坐标系 base COM 三轴速度，单位 m/s，encoder 速度输出除以 2。当前日志中 command vx 为 -1.1..1.3 m/s，全部在训练范围 ±1.5 内。角速度、腿位置、关节速度观测均未触及各自归一化裁剪范围。

停车分析窗口为模拟时间 2–10 秒、400 帧。真实 vx 均值 {analysis['velocity']['stationary']['actual_mean_mps'][0]:.6f} m/s；估计 vx 均值 {analysis['velocity']['stationary']['estimated_mean_mps'][0]:.6f} m/s；估速 MAE {analysis['velocity']['stationary']['mae_mps'][0]:.6f} m/s，偏差 {analysis['velocity']['stationary']['bias_mps'][0]:.6f} m/s。三轴完整统计见 feedback-analysis.json。

## 离线输入依赖试验

保持保存的状态不变，一次只替换/固定一组网络输入，不改变仿真或执行策略。指标为左腿两关节裁剪动作逐帧变化 RMS 的相对减少率，两关节平均；它不是实际腿抖动幅度，也不是闭环稳定性预测。

| 输入试验 | 左腿输出逐帧波动减少 |
|---|---:|
| Actor 使用同一时刻真实三轴速度 | {100*analysis['offline_counterfactuals']['actor_true_velocity']['left_leg_mean_step_rms_reduction_fraction']:.2f}% |
| 固定 Actor 估速输入为该窗口均值 | {100*analysis['offline_counterfactuals']['actor_velocity_mean']['left_leg_mean_step_rms_reduction_fraction']:.2f}% |
| 固定 context 为均值 | {100*analysis['offline_counterfactuals']['actor_context_mean']['left_leg_mean_step_rms_reduction_fraction']:.2f}% |
| 固定当前轮子速度观测 | {100*analysis['offline_counterfactuals']['actor_wheel_velocities_mean']['left_leg_mean_step_rms_reduction_fraction']:.2f}% |
| 固定当前上一帧轮子动作 | {100*analysis['offline_counterfactuals']['actor_previous_wheel_actions_mean']['left_leg_mean_step_rms_reduction_fraction']:.2f}% |
| 同时固定上述两组轮子反馈 | {100*cross['stationary_combined_counterfactuals']['wheel_feedback_mean']['mean_reduction']:.2f}% |

负值表示波动增加，替换真实速度约增加1.9%，没有观察到其降低抖动。固定轮子反馈在其余四段持续后退命令中也产生明显减小（轮子反馈与角速度同时固定约53–58%），没有把停车窗口结果直接推广为所有场景结论。

停车窗口中左/右轮目标转速都曾在 +25 和 -25 rad/s 之间切换，单次20ms最大跳变为50rad/s；相邻动作变号比例约36.1%/55.6%。目标处于转速边界的比例约71.0%/54.5%。因此下一步优先追查“轮子动作频繁切换—轮子/车身反馈—左腿修正”的关系。不能仅凭这些曲线判定由网络、PD、接触或训练中的某一项独立引起。

相邻 Actor 输入变化的积分梯度分解满足数值完整性检查；速度输入的有符号变化投影仅约3.3%/5.5%，轮子速度与上一帧轮子动作项更明显。这是固定网络的计算依赖，投影可能为负或大于1，不能当成独立因果占比。

## 当前已有的动作平滑惩罚

1401 轮策略的训练 manifest 与 rewards.py 一致：action_rate=-0.01、action_smooth=-0.01。两项基于六维实际裁剪动作，腿和轮子均参与，分别惩罚一阶和二阶动作差分的平方和；每步再乘 control_dt=0.02。env.py 在新动作进入时正确更新两帧动作历史。

将这份 MuJoCo 动作序列代入相同公式，停车窗口平均每步 action_rate 扣 {reward_stats['mean_action_rate_penalty_per_control_step']:.6f}，action_smooth 扣 {reward_stats['mean_action_smooth_penalty_per_control_step']:.6f}，合计扣 {reward_stats['mean_combined_penalty_per_control_step']:.6f}。对比同一 MuJoCo 轨迹按公式算得的正向 vx tracking 平均奖励 {reward_stats['mean_positive_vx_tracking_reward_per_control_step']:.6f}，约为其 {100*reward_stats['combined_penalty_vs_positive_vx_reward_fraction']:.2f}%。平滑惩罚约 {100*reward_stats['wheel_fraction_of_rate_and_smooth_penalty']:.2f}% 来自两轮动作。
这些是诊断重算值，不是 Isaac 实际训练 reward 的测量。它们证明惩罚存在并作用于动作；仅凭这段比例不能判定应把权重增大多少。奖励训练时扣分，运行时没有自动平滑滤波。

源码证据：source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/rewards.py:29、:64、:67、:92；env.py:645、:682、:784。未修改奖励或训练。

## 输出

feedback-chain.png：停车段反馈曲线；feedback-overview.png：全程估速、观测与离线敏感性。
replay-verification.json / final-verification.json：逐帧对齐、标准化和正式文件校验。
feedback-analysis.json / feedback-crosschecks.json / reward-smoothness-check.json：可复查的原始统计。

本轮完成观测和估速反馈检查。后续应隔离轮子动作换向与闭环反馈的起因，再决定训练奖励或控制是否需要调整；没有实施任何修复。
"""
with (folder/"report.md").open("w",encoding="utf-8",newline="") as stream:stream.write(notes)
print(json.dumps({"verification":verification,"reward_smoothness":reward_stats,"report":str(folder/"report.md")}))
