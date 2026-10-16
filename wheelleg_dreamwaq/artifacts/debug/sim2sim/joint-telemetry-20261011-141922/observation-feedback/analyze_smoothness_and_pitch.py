from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(r"E:\wheel_leg_rl-main")
project=root/"wheelleg_dreamwaq"
session=project/"artifacts/debug/sim2sim/joint-telemetry-20261011-141922"
folder=session/"observation-feedback"
manifest=json.loads((project/"artifacts/phase2_dreamwaq/resume-run03-1000-to-1500-20261011-133153/stopped-export/policy_manifest.json").read_text(encoding="utf-8"))
with (session/"control.csv").open(encoding="utf-8",newline="") as f:
    rows=list(csv.DictReader(f))
def column(name): return np.array([float(row[name]) for row in rows])
names=manifest["action"]["canonical_joint_order"]
a=np.column_stack([column(name+"_action_clipped") for name in names])
raw=np.column_stack([column(name+"_action_raw") for name in names])
t=column("time_before_s")
ep=column("episode_id")
p=np.zeros_like(a); p[1:]=a[:-1]
pp=np.zeros_like(a); pp[2:]=a[:-2]
for i in np.flatnonzero(np.r_[True,np.diff(ep)!=0]):
    p[i]=0; pp[i]=0
    if i+1<len(a): pp[i+1]=0
rate=(a-p)**2
smooth=(a-2*p+pp)**2
weights=manifest["training_reward_weights"]
dt=manifest["timing"]["control_dt_s"]
wr=weights["action_rate"]*dt*rate
ws=weights["action_smooth"]*dt*smooth
mask=(t>=2)&(t<=10)
assert mask.sum()==400
vx=column("vx_mps")
cmd=column("command_vx_mps")
tracking=weights["tracking_vx"]*dt*np.exp(-(cmd-vx)**2/weights["tracking_sigma"])
stats={}
for name,ix in [("left_leg",[0,1]),("right_leg",[2,3]),("wheels",[4,5]),("all_joints",list(range(6)))]:
    s=ws[mask][:,ix].sum(axis=1).mean()
    r=wr[mask][:,ix].sum(axis=1).mean()
    stats[name]={"action_rate_per_step":float(r),"action_smooth_per_step":float(s),
        "combined_per_step":float(s+r),
        "combined_fraction_of_positive_vx_tracking":float(-(s+r)/tracking[mask].mean()),
        "action_smooth_fraction_of_positive_vx_tracking":float(-s/tracking[mask].mean())}
left=[]
for i in [0,1]:
    target=column(names[i]+"_position_target_rad")
    left.append({"joint":names[i],
        "target_p05_p95_deg":np.rad2deg(np.percentile(target[mask],[5,95])).tolist(),
        "target_step_rms_deg":float(np.rad2deg(np.sqrt(np.mean(np.diff(target)[mask[1:]]**2)))),
        "action_second_difference_rms":float(np.sqrt(np.mean(smooth[mask,i]))),
        "nonzero_second_difference_fraction":float(np.mean(smooth[mask,i]>1e-12)),
        "action_smooth_penalty_per_step":float(ws[mask,i].mean())})
previous=json.loads((folder/"reward-smoothness-check.json").read_text(encoding="utf-8"))
assert abs(stats["all_joints"]["combined_per_step"]-previous["mean_combined_penalty_per_control_step"])<1e-9
reward_result={"source":"Recorded MuJoCo actions evaluated with actual checkpoint training reward weights; not per-leg training logging.",
    "window_s":[2,10],"frames":int(mask.sum()),"per_group":stats,"left_joints":left,
    "positive_vx_tracking_per_step":float(tracking[mask].mean()),
    "left_leg_fraction_of_all_smoothness_penalty":float(stats["left_leg"]["combined_per_step"]/stats["all_joints"]["combined_per_step"]),
    "right_leg_actions_clipped_to_minus_one_fraction":np.mean(a[mask,2:4]==-1,axis=0).tolist(),
    "right_leg_raw_action_std":raw[mask,2:4].std(axis=0).tolist()}
(folder/"smoothness-per-joint.json").write_text(json.dumps(reward_result,indent=2)+"\n",encoding="utf-8")

pitch=np.rad2deg(column("pitch_rad"))
phi=np.column_stack([np.rad2deg(column("phi0_left_rad")),np.rad2deg(column("phi0_right_rad"))])
intervals=[("parking",2,10),("forward",14,16),("reverse_fast",21,24),("reverse_slow",27.5,29.5),("forward_later",35,37),("reverse_later",42.5,45.5)]
angle_result=[]
for name,start,end in intervals:
    m=(t>=start)&(t<=end)
    d={"name":name,"window_s":[start,end],"frames":int(m.sum()),
       "command_mean_mps":float(cmd[m].mean()),"command_range_mps":[float(cmd[m].min()),float(cmd[m].max())],
       "vx_mean_mps":float(vx[m].mean()),"pitch_mean_deg":float(pitch[m].mean()),
       "pitch_p05_p95_deg":np.percentile(pitch[m],[5,95]).tolist(),
       "phi0_left_mean_deg":float(phi[m,0].mean()),"phi0_right_mean_deg":float(phi[m,1].mean()),
       "phi0_left_p05_p95_deg":np.percentile(phi[m,0],[5,95]).tolist(),
       "phi0_right_p05_p95_deg":np.percentile(phi[m,1],[5,95]).tolist()}
    angle_result.append(d)
(folder/"pitch-phi0-analysis.json").write_text(json.dumps({"source":"Recorded post-control-step metrics in body and control frames; angles do not identify causal COM contribution.","windows":angle_result},indent=2)+"\n",encoding="utf-8")
plt.rcParams["font.family"]="Microsoft YaHei"
plt.rcParams["axes.unicode_minus"]=False
fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
axes[0].plot(t+dt,cmd,label="前后速度指令")
axes[0].plot(t+dt,vx,label="实际前后速度")
axes[0].set_ylabel("速度（m/s）")
axes[1].plot(t+dt,pitch,label="车身 pitch")
axes[1].set_ylabel("车身相对地面（度）")
for i,label in enumerate(["左腿 phi0","右腿 phi0"]):
    axes[2].plot(t+dt,phi[:,i],label=label)
axes[2].set_ylabel("虚拟腿相对车身（度）")
axes[2].set_xlabel("模拟时间（秒）")
for ax in axes:
    ax.legend(loc="upper right"); ax.grid(alpha=.2)
fig.suptitle("当前键盘轨迹：车身倾斜与腿形变化\nphi0 从车身局部水平轴量起；竖直向下约为 90 度")
fig.tight_layout()
fig.savefig(folder/"pitch-phi0-analysis.png",dpi=150)
plt.close(fig)
protected=json.loads((folder/"protected-before.json").read_text(encoding="utf-8"))
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()==expected for path,expected in protected.items())
print(json.dumps({"reward":reward_result,"angles":angle_result,"protected_files_unchanged":len(protected)},indent=2))
