"""Independent arithmetic check, figure rendering and report. No simulation or training."""
from pathlib import Path
import csv,json,math,subprocess,hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;ROOT=Path(r"E:\wheel_leg_rl-main")
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def rows(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
report=read(OUT/"results.json");identity=read(OUT/"input-identity.json")
hist=ROOT/"wheelleg_dreamwaq/artifacts/debug/sim2sim/standing-trace-20261010-v1/mujoco/nominal_stand.csv"
all_rows={};maximum_error=0.;checked=0
for case in report["cases"]:
    path=Path(case.get("trace",hist));data=rows(path);all_rows[(case["label"],case["scenario"])]=data
    if "trace_sha256" in case:assert sha(path)==case["trace_sha256"]
    if case["samples"]==0:
        assert len(data)<100 and case["failure_reason"]
        continue
    ix=[i for i,r in enumerate(data[:-1]) if 2.04-1e-9<=float(r["time_s"])<9.98-1e-9]
    assert len(ix)==case["samples"]==397
    for i in ix:assert abs(float(data[i+1]["time_s"])-float(data[i]["time_s"])-.02)<1e-10
    rms=lambda values:math.sqrt(math.fsum(x*x for x in values)/len(values))
    alpha_rms=[rms([(float(data[i+1][f"joint_velocity_{j}_rad_s"])-float(data[i][f"joint_velocity_{j}_rad_s"]))/.02 for i in ix]) for j in range(4)]
    vel_rms=[rms([float(data[i][f"joint_velocity_{j}_rad_s"]) for i in ix]) for j in range(4)]
    smooth=math.fsum(math.fsum((float(data[i][f"clipped_action_{j}"])-2*float(data[i-1][f"clipped_action_{j}"])+float(data[i-2][f"clipped_action_{j}"]))**2 for j in range(4)) for i in ix)/len(ix)
    vx=math.fsum(abs(float(data[i]["actual_vx_mps"])-float(data[i]["command_vx_mps"])) for i in ix)/len(ix)
    errors=[abs(a-b) for a,b in zip(alpha_rms,case["leg_acceleration_rms_rad_s2"])]+[abs(a-b) for a,b in zip(vel_rms,case["leg_velocity_rms_rad_s"])]+[abs(smooth-case["leg_action_smooth_mean"]),abs(vx-case["vx_mae_mps"])]
    maximum_error=max(maximum_error,max(errors));assert max(errors)<1e-9;checked+=len(ix)
before=identity["protected_files"];assert all(sha(Path(p))==value for p,value in before.items())
tracked=subprocess.check_output(["git","diff","--name-only"],cwd=ROOT,text=True).splitlines();assert tracked==identity["tracked_changes_before"]
verification=dict(protected_file_count=len(before),all_hashes_unchanged=True,tracked_changes_unchanged=True,independent_checked_frames=checked,
    independent_arithmetic_max_error=maximum_error,historical_actor_replay_max_error=report["historical_actor_replay_max_error"],
    current_reference_policy_replays=[dict(label=c["label"],scenario=c["scenario"],max_action_error=c["action_replay_error_vs_original_evaluation"]) for c in report["cases"] if "action_replay_error_vs_original_evaluation" in c],
    old_policy_current_physics_is_diagnostic_only=True,training_started=False)
(OUT/"verification.json").write_text(json.dumps(verification,indent=2)+"\n",encoding="utf-8")
fig,axes=plt.subplots(1,2,figsize=(12,4),sharex=True,sharey=True)
for label,color,display in [("night_original_PhysicsV4","tab:blue","night run04 / original PhysicsV4"),("tracking_current_PhysicsV5","tab:orange","tracking run03 / current PhysicsV5"),("smooth002_current_PhysicsV5","tab:green","smooth002 run03 / current PhysicsV5")]:
    data=all_rows[(label,"parking")];t=np.array([float(r["time_s"]) for r in data]);mask=(t>=2)&(t<=4)
    for j in range(2):
        v=np.array([float(r[f"joint_velocity_{j}_rad_s"]) for r in data]);axes[j].plot(t[mask],v[mask],lw=.8,color=color,label=display)
        axes[j].set_title(["Left front jIJ","Left rear jIO"][j]);axes[j].set_xlabel("time (s)");axes[j].set_ylabel("actual velocity (rad/s)");axes[j].grid(alpha=.2)
axes[0].legend(fontsize=8);fig.suptitle("Parking: historical quiet legs vs current policies (physics differs for night run04)")
fig.tight_layout();fig.savefig(OUT/"parking-left-leg-velocity.png",dpi=170);plt.close(fig)
header=["# 昨晚模型的腿部运动核对（2026-10-11）","",
"用户记忆在原条件下得到支持：昨晚 run-04 的四腿实际几乎不动，但零速命令下持续后退。四腿执行动作全部固定在裁剪下限 -1；安静不代表已学会健康的姿态控制，也不能拿它的零 action_smooth 当作 ROGER 合格阈值。",
"",
"## 模型与条件",
"",
"- 昨晚模型：20261010 reset-v2 suite run-04，seed342388981，1000 iterations。源 checkpoint SHA256：89926828678CC28E0B0BA5A8463181395E0BE95E5961F7C7A65640243D5DC5FD。",
"- 现有对照：20261011 stop_reverse_v1 run-03，seed41882943，1000 iterations，smooth=-0.01；以及同种子 fresh1000 的 smooth=-0.02 模型。",
"- 历史停车使用 20261010 已保存的正式 nominal_stand 同时刻日志。昨晚键盘 viewer 的默认导出路径也指向同一 run-04；23点会话 launch 未单独记录策略 hash，因此无法仅凭 launch 对每次窗口再证明一次身份。",
"- 历史条件有旧刚体角速度限制力矩；旧 actuator 也带速度限制。当前 PhysicsV5 已删除这些作用。",
"- 旧 manifest 不改写。先验证旧 actor、golden、checkpoint、manifest 的原始哈希，再逐项验证网络、观测、归一化、history、动作、PD/effort、时序、reset 等接口与当前一致。当前 loader 对旧物理契约的拒绝未取消；本目录显式将旧网络权重放在当前已验证 runtime 上，仅用于跨物理诊断，不冒充正式评估通过。",
"",
"## 停车实测",
"",
"同一口径采用动作前 2.04–9.96s 共397帧，前2秒排除；角加速度为20ms相邻端点速度差，不是1ms瞬时qacc。速度误差使用动作前数据，故与旧报告全程/动作后 MAE 不完全相同。",
"",
"| 模型/物理条件 | 左前速度 RMS rad/s | 左后速度 RMS rad/s | 左前加速度 RMS rad/s² | 左后加速度 RMS rad/s² | 零速误差 m/s |",
"|---|---:|---:|---:|---:|---:|"]
for c in report["cases"]:
    if c["scenario"]=="parking" and c["samples"]:
        v=c["leg_velocity_rms_rad_s"];a=c["leg_acceleration_rms_rad_s2"]
        header.append(f'| {c["label"]} | {v[0]:.6g} | {v[1]:.6g} | {a[0]:.6g} | {a[1]:.6g} | {c["vx_mae_mps"]:.6f} |')
header+=["","历史四腿裁剪动作跨度均为0，-1占比均100%；实际位置跨度分别为约5.77e-6、2.66e-6、7.12e-6、1.17e-6 rad。不是只看指令不变，实际运动也很小。",
"",
"## 把旧策略放在当前物理条件下",
"",
"保持同一当前模型、reset、PD、动作裁剪、history和步长。停车/前进+1/后退-1m/s，各计划运行10秒，按现有失败规则提前停止：",
"",
"| 指令 | 完成帧数 | 结束时间 | 失败原因 |","|---|---:|---:|---|"]
for c in report["cases"]:
    if c["label"]=="night_current_PhysicsV5":header.append(f'| {c["scenario"]} | {c["completed_frames"]} | {c["completed_frames"]*.02:.2f}s | {c["failure_reason"]} |')
header+=["",
"这三个旧策略诊断均未到达2秒后的分析窗口，不能虚构稳定阶段的抖动统计。没有重新加入已删除的限速力矩或身体碰撞。失败不能单独归因到某一种旧限制，因为该策略是在旧训练条件下学习的。",
"",
"## 一个可继续使用的参考",
"",
"当前 stop_reverse_v1 run-03 在前进+1m/s的后2秒分析段，左腿可以更安静，同时速度跟踪良好。它说明目前证据不支持‘减少腿抖必然损害跟踪’，但前进与停车的工况不同，不能直接把前进指标当作停车阈值。",
"",
"| 当前物理/前进+1m/s | 左前速度 RMS | 左后速度 RMS | 左前加速度 RMS | 左后加速度 RMS | vx MAE |","|---|---:|---:|---:|---:|---:|"]
for c in report["cases"]:
    if c["scenario"]=="forward" and c["samples"]:
        v=c["leg_velocity_rms_rad_s"];a=c["leg_acceleration_rms_rad_s2"];header.append(f'| {c["label"]} | {v[0]:.4f} | {v[1]:.4f} | {a[0]:.2f} | {a[1]:.2f} | {c["vx_mae_mps"]:.4f} |')
header+=["",
"## 证据范围与核验",
"",
"- 该对照没有只改变腿抖动这一因素，不能证明腿抖动与速度误差的因果关系。安静且跟踪差，最多说明安静本身不足以保证良好跟踪。",
"- 正常 loader、生产代码、奖励、动作、随机化、USD、模型和训练均未改变，只新增本目录诊断。",
f"- {len(before)}个受保护文件与诊断前哈希一致，已有tracked修改列表一致；独立标量算法复算{checked}帧，最大误差{maximum_error:.3g}。",
"- 两个当前对照策略的三种场景，原始动作逐帧重现原评估，最大误差均为0。旧 actor 回放历史125维输入，最大误差约1.91e-6。",
"- compare.py 用 MuJoCo venv采集；finish.py用项目venv绘图及独立核验。首次采集全部完成后因MuJoCo环境没有matplotlib而绘图停止，随后把绘图独立出来，未重跑或覆盖轨迹。第一次误用训练venv因缺mujoco未进入采集。",
"",
"[完整数值与模型身份](results.json) · [独立核验](verification.json) · [停车实际左腿速度曲线](parking-left-leg-velocity.png)"]
(OUT/"report.md").write_text("\n".join(header)+"\n",encoding="utf-8")
print(json.dumps(verification))
