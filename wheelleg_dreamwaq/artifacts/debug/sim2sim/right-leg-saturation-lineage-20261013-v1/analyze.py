"""Save source-log/budget provenance and plot checkpoint lineage."""
from pathlib import Path
import json,hashlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
D=json.loads((OUT/"results.json").read_text())
run=Path(D["checkpoints"][0]["checkpoint"]).parent
e=EventAccumulator(str(run),size_guidance={"scalars":0});e.Reload()
assert not any(k.startswith(("Acceleration/","Reversal/")) for k in e.Tags()["scalars"])
tags=[k for k in e.Tags()["scalars"] if k.startswith("Reward/")]
extras=["Action/saturation_fraction","Metric/phi0_delta_abs_rad","Tracking/vx_abs_error","Tracking/yaw_rate_abs_error","Loss/entropy"]
logs={}
for completed in (101,201,1000):
    hi=completed-1;lo=max(0,hi-49)
    mean={k:float(np.mean([v.value for v in e.Scalars(k) if lo<=v.step<=hi])) for k in tags+extras}
    logs[str(completed)]={"iterations_inclusive":[lo+1,hi+1],"means":mean,"reward_total_mean":sum(mean[k] for k in tags)}
metric=OUT.parent/"leg-motion-metric-analysis-20261012-v1/results.json"
a=json.loads(metric.read_text())
windows=[]
sources=[]
for t in a["trajectories"]:
    if t["arm"]=="baseline":
        windows.extend([[v["acceleration_rms"] for v in win["joints"]] for win in t["windows"]])
        sources.append(t["sources"])
assert np.array(windows).shape==(24,4)
b0=np.percentile(windows,90,axis=0)
expected=np.array([40.1703144886,38.3671852533,7.0362190023,4.9735461617])
np.testing.assert_allclose(b0,expected,atol=1e-9,rtol=0)
proof={"schema":"SaturationLineageProvenanceV1","fresh_training_no_adaptive_acceleration_or_reversal_tags":True,
       "fresh_training_windows":logs,"reference_budget":{"source":str(metric),"source_sha256":hashlib.sha256(metric.read_bytes()).hexdigest().upper(),
       "source_trajectory_files":sources,"window_count":24,"pooled_p90_rad_s2":b0.tolist(),"budget85_rad_s2":(b0*.85).tolist(),
       "previous_reference_right_leg_targets_saturated":True}}
(OUT/"training-and-budget-provenance.json").write_text(json.dumps(proof,indent=2),encoding="utf-8")
r=[x for x in D["checkpoints"] if x["stage"]=="fresh1000"]
fig,axes=plt.subplots(2,2,figsize=(12,8))
its=[x["completed_iterations_local"] for x in r]
fixed=np.array([x["same_85_state_raw_action_mean"] for x in r])
for j,name in ((2,"jAB"),(3,"jAG")):
    axes[0,0].plot(its,fixed[:,j],marker="o",label=name)
axes[0,0].axhline(-1,color="gray",ls="--");axes[0,0].set_title("Same recorded MuJoCo inputs: right raw action means");axes[0,0].set_xlabel("completed training iterations");axes[0,0].legend()
for j,name in ((2,"jAB"),(3,"jAG")):
    values=[x["steady_trajectory"]["action_negative_clip_fraction"][j]*100 if x["steady_trajectory"] else np.nan for x in r]
    axes[0,1].plot(its,values,marker="o",label=name)
axes[0,1].set_title("Own MuJoCo trajectories: negative clipping in 2..10s");axes[0,1].set_ylim(-3,103);axes[0,1].set_ylabel("fraction (%)");axes[0,1].set_xlabel("completed training iterations");axes[0,1].legend()
c=np.load(OUT/"checkpoint-trajectories.npz")
for col,local in enumerate((101,201)):
    label="fresh1000_"+str(local);t=c[label+"__time_after"]
    for j,name in enumerate(("jIJ","jIO","jAB","jAG")):
        axes[1,col].plot(t,np.rad2deg(c[label+"__target"][:,j]),label=name,lw=.9)
    axes[1,col].set_title("Motor targets: original checkpoint "+str(local));axes[1,col].set_xlabel("MuJoCo simulation time (s)");axes[1,col].set_ylabel("target angle (deg)");axes[1,col].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/"checkpoint-lineage.png",dpi=160);plt.close(fig)
print("budget_p90",b0.tolist(),"budget_reference_max_error",float(abs(b0-expected).max()))
print("training_totals",{k:v["reward_total_mean"] for k,v in logs.items()})
