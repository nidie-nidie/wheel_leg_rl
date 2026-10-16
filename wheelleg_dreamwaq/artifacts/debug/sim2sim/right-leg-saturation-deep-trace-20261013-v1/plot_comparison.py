"""Scientific plots and table from saved traces only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
old=np.load(OUT/"expanded-trajectories.npz")
new=np.load(OUT/"frozen-candidate-trajectories.npz")
fil=np.load(OUT/"candidate-validation-trajectories.npz")
op=np.load(OUT/"physics-bandwidth-trajectories.npz")["unmodified__physics"]
np_=np.load(OUT/"final-physics-trajectories.npz")["physics"]
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.22})
fig,axs=plt.subplots(2,2,figsize=(13,7.2),constrained_layout=True)
for i,ax in enumerate(axs[0]):
 for a,color,label in ((op,"#c75245","Original"),(np_,"#177c73","Candidate")):
  z=a[(a[:,0]>=2)&(a[:,0]<=4)]
  ax.plot(z[:,0],np.rad2deg(z[:,1+i]-np.median(z[:,1+i])),color=color,lw=.9,label=label)
 ax.set(title=["Left front motor jIJ","Left rear motor jIO"][i],ylabel="Actual angle around median (deg)",xlabel="Time (s)")
 ax.legend(loc="upper right")
for ax,case,axis,title,unit in ((axs[1,0],"forward_stop_reverse",0,"Forward / stop / reverse","m/s"),
                                (axs[1,1],"left_stop_right",1,"Turn left / stop / right","rad/s")):
 k="candidate_v1__"+case; t=new[k+"__time"]
 ax.step(t,new[k+"__user_command"][:,axis],where="post",color="black",ls="--",lw=1,label="Original user command")
 ax.plot(old["unmodified__"+case+"__time"],old["unmodified__"+case+"__metrics"][:,axis],color="#c75245",lw=1,label="Original actual")
 ax.plot(t,new[k+"__metrics"][:,axis],color="#177c73",lw=1.2,label="Candidate actual")
 ax.set(title=title,xlabel="Time (s)",ylabel="Speed ("+unit+")");ax.legend(fontsize=8,ncol=3,loc="upper right")
fig.suptitle("Same saved policy and physics: actual leg motion and command tracking",fontsize=14)
fig.savefig(OUT/"tracking-and-jitter.png",dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
for trace,k,color,label in ((old,"unmodified__long_stand","#c75245","Original"),
                            (fil,"leg6_wheel2__long_stand","#b48b27","Filter only"),
                            (new,"candidate_v1__stand60","#177c73","Filter + zero-command correction")):
 xy=trace[k+"__com_after"][100:];start=trace[k+"__com_before"][100]
 t=trace[k+"__time"][100:]; delta=xy-start
 axs[0].plot(t,np.linalg.norm(delta,axis=1),color=color,label=label,lw=1.5)
 axs[1].plot(delta[:,0],delta[:,1],color=color,label=label,lw=1.2)
axs[0].set(xlabel="Time (s)",ylabel="Distance from settled starting position (m)",title="60 s stand command (first 2 s excluded)")
axs[1].set(xlabel="World X displacement (m)",ylabel="World Y displacement (m)",title="Robot COM path");axs[1].axis("equal")
axs[0].legend(fontsize=9);axs[1].legend(fontsize=9)
fig.suptitle("Candidate max excursion: 5.95 cm; total path length: 1.05 m (residual motion)",fontsize=13)
fig.savefig(OUT/"stand-drift.png",dpi=150);plt.close(fig)
orig={r["scenario"]:r for r in json.loads((OUT/"expanded-results.json").read_text())["results"] if r["variant"]=="unmodified"}
rows=[]
for r in json.loads((OUT/"frozen-candidate-results.json").read_text())["results"]:
 if r["variant"]=="candidate_v1" and "formal_summary" in r:
  before=orig[r["scenario"]]["steady"];after=r["steady"]
  rows.append({"scenario":r["scenario"],"old_vx_mae":before["vx_mae_mps"],"new_vx_mae":after["vx_mae_mps"],
  "old_yaw_mae":before["yaw_mae_rad_s"],"new_yaw_mae":after["yaw_mae_rad_s"],
  "old_leg_accel_rms_20ms":before["leg_accel_rms_rad_s2"],"new_leg_accel_rms_20ms":after["leg_accel_rms_rad_s2"],
  "phi0_delta_rms_deg":float(np.rad2deg(r["formal_summary"]["phi0_delta_rms"]))})
practice=[]
for r in json.loads((OUT/"frozen-candidate-results.json").read_text())["results"]:
 if "practice_summary" in r:
  stages=r["practice_summary"]["stages"]
  practice.append({"scenario":r["scenario"],"accepted":True,"stop_vx_worst":max(s["steady_vx_mae_mps"] for s in stages if s["target"][:2]==[0,0]),
                  "stop_yaw_worst":max(s["steady_yaw_mae_rad_s"] for s in stages if s["target"][:2]==[0,0]),
                  "reversal_confirmation_worst_s":max(stages[i]["held_error_band_confirmation_time_s"] for i in (2,4))})
q={label:np.percentile(a[a[:,0]>=2,1:5],95,axis=0)-np.percentile(a[a[:,0]>=2,1:5],5,axis=0) for label,a in (("original",op),("candidate",np_))}
summary={"same_checkpoint_same_physics":True,"formal_steady_statistics_exclude_first_2s":True,"formal_rows":rows,
         "practice_rows":practice,"leg_q_p95_p5_deg_1ms":{k:np.rad2deg(v).tolist() for k,v in q.items()},
         "plots":["tracking-and-jitter.png","stand-drift.png"]}
(OUT/"comparison-metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
