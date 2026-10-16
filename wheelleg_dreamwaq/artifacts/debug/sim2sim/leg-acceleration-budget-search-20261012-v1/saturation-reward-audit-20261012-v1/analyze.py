"""Read-only source/log audit and diagnostic MuJoCo reward reconstruction."""
from pathlib import Path
from types import SimpleNamespace
import csv, json, sys, math, hashlib
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
PROJECT=BASE.parents[3]
sys.path.insert(0,str(PROJECT/"source/wheelleg_dreamwaq"))
import wheelleg_dreamwaq.schemas  # Initialize existing contract imports in their normal order.
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RewardWeights,compute_reward,compute_reward_terms
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.leg_acceleration import LegAccelerationConfig,LegAccelerationReward
torch.set_num_threads(1)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
protected=json.loads((OUT/"protected-before.json").read_text())
assert all(sha(p)==h for p,h in protected.items())
D=np.load(OUT/"trajectories.npz")
keys=sorted({k.split("__")[0] for k in D.files})
def data(k): return {n.split("__")[1]:D[n] for n in D.files if n.startswith(k+"__")}
manifests={a:json.loads((BASE/a/"export/policy_manifest.json").read_text()) for a in ("baseline","80","85","90")}
rewards, summaries={},{}
channels={}
function_errors={}
noise={}
log_arrays={}
logs={}
for arm,m in manifests.items():
    run=Path(m["source_checkpoint"]).parent
    e=EventAccumulator(str(run),size_guidance={"scalars":0});e.Reload()
    tags=[k for k in e.Tags()["scalars"] if k.startswith(("Reward/","Acceleration/")) or k in ("Tracking/vx_abs_error","Tracking/yaw_rate_abs_error","Action/saturation_fraction","Joint/leg_soft_limit_fraction","Metric/phi0_delta_abs_rad","Loss/entropy")]
    series={k:np.array([v.value for v in e.Scalars(k)]) for k in tags}
    steps={k:[v.step for v in e.Scalars(k)] for k in tags}
    assert all(v==list(range(200)) for v in steps.values())
    log_arrays[arm]=series
    mean={k:float(v[-50:].mean()) for k,v in series.items()}
    cp=torch.load(run/"model_199.pt",map_location="cpu",weights_only=False)
    noise[arm]=cp["model_state_dict"]["raw_std"].abs().clamp(1e-4,2).numpy()
    entry={"iterations":200,"last_50_mean":mean,"source_checkpoint_sha256":sha(m["source_checkpoint"]),"final_effective_action_std":noise[arm].tolist()}
    if arm!="baseline":
        states=[]
        for i in (0,100,199):
            payload=torch.load(run/f"model_{i}.pt",map_location="cpu",weights_only=False)["infos"]["leg_acceleration_gain_state"]
            latest=payload["latest"]
            expected=min(max(latest["mean_excess"])**2,1)
            assert expected==payload["gain"]==latest["gain_next"]
            expected_reward=-.02*.02*latest["gain_used"]*sum(latest["mean_excess"])
            err=abs(expected_reward-latest["weighted_reward_per_step_mean"])
            assert err<1e-7,(arm,i,err)
            expected_values=np.array(latest["mean_excess"])
            actual_values=np.array([series["Acceleration/excess_"+n][i] for n in ("jIJ","jIO","jAB","jAG")])
            assert np.max(abs(actual_values-expected_values))<1e-4
            assert abs(series["Reward/leg_acceleration"][i]-latest["weighted_reward_per_step_mean"])<1e-7
            states.append({"checkpoint_iteration":i,"state":payload,"reward_formula_max_error":err})
        ex=np.stack([series["Acceleration/excess_"+n] for n in ("jIJ","jIO","jAB","jAG")],axis=1)
        entry.update({"gain_used_after_first_at_upper_bound_fraction":float(np.mean(series["Acceleration/gain_used"][1:]==1)),
                      "dominant_joint_iteration_counts":np.bincount(np.argmax(ex,axis=1),minlength=4).tolist(),
                      "last_50_excess_by_joint":ex[-50:].mean(axis=0).tolist(),
                      "last_50_right_penalty_share":float(ex[-50:,2:].sum()/ex[-50:].sum()),
                      "last_50_left_penalty_share":float(ex[-50:,:2].sum()/ex[-50:].sum()),
                      "last_50_acceleration_over_action_smooth_penalty":mean["Reward/leg_acceleration"]/mean["Reward/action_smooth"],
                      "checkpoint_gain_states":states})
    logs[arm]=entry

for key in keys:
    arm=key.split("_")[0];m=manifests[arm];d=data(key)
    a=d["action_clipped"]
    prev=np.vstack((np.zeros((1,6)),a[:-1]))
    prevprev=np.vstack((np.zeros((2,6)),a[:-2]))
    state=SimpleNamespace(command=torch.from_numpy(d["command"]),root_com_linear_velocity=torch.from_numpy(d["root_v"]),root_angular_velocity=torch.from_numpy(d["root_w"]),
                          base_height=torch.from_numpy(d["height"][:,None]),joint_position=torch.from_numpy(d["q"]),joint_velocity=torch.from_numpy(d["v"]),joint_acceleration=torch.from_numpy(d["acc"]),
                          applied_torque=torch.from_numpy(d["tau"]),projected_gravity=torch.from_numpy(d["gravity"]),virtual_leg_phi0_true=torch.from_numpy(d["phi"]),
                          last_applied_action=torch.from_numpy(a),previous_applied_action=torch.from_numpy(prev),previous_previous_applied_action=torch.from_numpy(prevprev))
    weights=RewardWeights(**m["training_reward_weights"])
    total,weighted=compute_reward(state,weights,control_dt=.02,terminated=torch.zeros(len(a),dtype=torch.bool))
    terms={n:v.numpy() for n,v in compute_reward_terms(state,weights,torch.zeros(len(a),dtype=torch.bool)).items()}
    # Independent NumPy formula checks for the investigated costs.
    delta=a-prev; second=a-2*prev+prevprev
    independent={"action_rate":np.sum(delta**2,axis=1),"action_smooth":np.sum(second**2,axis=1),"dof_acc":np.sum(d["acc"]**2,axis=1),
                 "orientation":1-np.clip(-d["gravity"][:,2],-1,1),"phi0_symmetry":np.arctan2(np.sin(d["phi"][:,0]-d["phi"][:,1]),np.cos(d["phi"][:,0]-d["phi"][:,1]))**2}
    errors={n:float(abs(terms[n]-v).max()) for n,v in independent.items()}
    assert max(errors.values())<1e-8
    function_errors[key]=errors
    wr={n:v.numpy() for n,v in weighted.items()}
    ch={"action_rate":weights.action_rate*.02*delta**2,"action_smooth":weights.action_smooth*.02*second**2,
        "dof_acc":weights.dof_acc*.02*d["acc"]**2,"torques":weights.torques*.02*d["tau"]**2}
    if m.get("training_leg_acceleration",{}).get("enabled",False):
        ac=m["training_leg_acceleration"]
        cfg=LegAccelerationConfig(enabled=True,reference_rms_rad_s2=tuple(ac["reference_rms_rad_s2"]),budget_fraction=ac["budget_fraction"],max_penalty_weight=ac["max_penalty_weight"],rollout_steps=ac["rollout_steps"])
        budget=np.array(cfg.budgets())
        excess=np.maximum(d["acc"][:,:4]**2/budget**2-1,0)
        ch["leg_acceleration"]=-.02*cfg.max_penalty_weight*excess
        wr["leg_acceleration"]=ch["leg_acceleration"].sum(axis=1)
        controller=LegAccelerationReward(len(a),cfg,"cpu",.02)
        controller.gain=1
        actual=controller.step(torch.from_numpy(d["acc"])).numpy()
        error=float(abs(actual-wr["leg_acceleration"]).max())
        assert error<1e-5,(key,error)
        function_errors[key]["adaptive_reward_float32_difference"]=error
    mask=(d["time_before"]>=2-1e-8)
    if "interactive" in key:mask&=(d["time_before"]<20)
    assert mask.any()
    raw=d["action_raw"][mask];clipped=a[mask];q=d["q"][mask,:4];phi=d["phi"][mask]
    phi_delta=np.arctan2(np.sin(phi[:,0]-phi[:,1]),np.cos(phi[:,0]-phi[:,1]))
    normal=d["contact_normal_last"][mask]
    denominator=normal.sum(axis=1)
    bound_fraction=np.mean(abs(raw)>=1,axis=0)
    std=noise[arm]
    cdf=lambda x:np.vectorize(math.erf)(x/math.sqrt(2))*.5+.5
    p_inside=cdf((1-raw)/std)-cdf((-1-raw)/std)
    weighted_mean={n:float(v[mask].mean()) for n,v in wr.items()}
    channel_mean={n:v[mask].mean(axis=0).tolist() for n,v in ch.items()}
    summaries[key]={"window_s":[float(d["time_before"][mask][0]),float(d["time_after"][mask][-1])],"ticks":int(mask.sum()),
                    "command":d["command"][mask][0].tolist(),"action_clip_fraction":bound_fraction.tolist(),
                    "action_negative_clip_fraction":np.mean(raw<=-1,axis=0).tolist(),"raw_action_mean":raw.mean(axis=0).tolist(),"raw_action_min":raw.min(axis=0).tolist(),"raw_action_max":raw.max(axis=0).tolist(),
                    "clipped_action_std":clipped.std(axis=0).tolist(),"clipped_action_step_rms":np.sqrt(np.mean(delta[mask]**2,axis=0)).tolist(),
                    "raw_action_step_rms":np.sqrt(np.mean(np.diff(d["action_raw"],axis=0)[np.flatnonzero(mask)[np.flatnonzero(mask)>0]-1]**2,axis=0)).tolist(),
                    "unsaturated_gaussian_sample_probability_conditional_on_MuJoCo_states":p_inside.mean(axis=0).tolist(),
                    "leg_target_range_deg":np.rad2deg(np.ptp(d["target"][mask,:4],axis=0)).tolist(),"leg_position_mean_deg":np.rad2deg(q.mean(axis=0)).tolist(),
                    "leg_position_p5_p95_width_deg":np.rad2deg(np.percentile(q,95,axis=0)-np.percentile(q,5,axis=0)).tolist(),
                    "leg_soft_limit_fraction":np.mean(abs(q)>=weights.soft_leg_limit,axis=0).tolist(),
                    "leg_mechanical_limit_fraction":np.mean(abs(q)>=1,axis=0).tolist(),
                    "leg_accel_rms_rad_s2":np.sqrt(np.mean(d["acc"][mask,:4]**2,axis=0)).tolist(),
                    "phi0_mean_deg":np.rad2deg(phi.mean(axis=0)).tolist(),"phi0_delta_abs_mean_deg":float(np.rad2deg(abs(phi_delta)).mean()),
                    "leg_length_mean_m":d["length"][mask].mean(axis=0).tolist(),"roll_pitch_mean_deg":np.rad2deg(d["roll_pitch_yaw"][mask,:2].mean(axis=0)).tolist(),
                    "max_tilt_deg":float(np.rad2deg(d["tilt"][mask]).max()),"vx_mae_mps":float(abs(d["root_v"][mask,0]-d["command"][mask,0]).mean()),
                    "yaw_rate_mae_rad_s":float(abs(d["root_w"][mask,2]-d["command"][mask,1]).mean()),
                    "contact_normal_last_mean_N":normal.mean(axis=0).tolist(),"contact_total_mean_N":float(denominator.mean()),
                    "contact_left_fraction_of_total_mean_load":float(normal[:,0].sum()/denominator.sum()),
                    "weighted_reward_mean_per_step":weighted_mean,"weighted_reward_channel_mean_per_step":channel_mean,
                    "total_reward_mean_per_step":float(sum(weighted_mean.values()))}
    channels[key]=ch;rewards[key]=wr

report={"schema":"SaturationRewardAuditV1","joint_order":["jIJ","jIO","jAB","jAG","jwheel_left","jwheel_right"],
        "muJoCo_steady_windows":"original trajectories; exclude first 2s; interactive zero-command section 2..20s",
        "training_log_window":"last 50 of 200 iterations; mean across 24 rollout steps and 256 environments",
        "reward_reconstruction":"saved training weights and original production reward functions on MuJoCo post-action states; adaptive gain fixed at logged upper bound 1; termination=false since all original trajectories survived",
        "reward_is_not_training_replay":True,"contact_force_timestamp":"last_substep_before_integration",
        "training_stochastic_versus_deployment_deterministic_distinction":True,
        "traces":summaries,"training_logs":logs,"production_formula_verification":function_errors,"protected_files_unchanged":True}
(OUT/"analysis.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
np.savez_compressed(OUT/"weighted-rewards.npz",**{k+"__"+n:v for k,wr in rewards.items() for n,v in wr.items()})

# Show recorded asymmetric saturation, actual motion, and training penalty allocation.
d=data("85_interactive");sel=(d["time_before"]>=2)&(d["time_before"]<7);t=d["time_before"][sel]
fig,axes=plt.subplots(3,2,figsize=(13,10))
for j,name in enumerate(("jIJ","jIO","jAB","jAG")):
    c=0 if j<2 else 1
    axes[0,c].plot(t,d["action_raw"][sel,j],label=name+" raw")
    axes[0,c].plot(t,d["action_clipped"][sel,j],ls="--",label=name+" clipped")
    axes[1,c].plot(t,np.rad2deg(d["q"][sel,j]),label=name+" actual")
    axes[1,c].plot(t,np.rad2deg(d["target"][sel,j]),ls="--",label=name+" target")
for c in (0,1):
    axes[0,c].axhline(-1,color="gray",lw=.7);axes[0,c].axhline(1,color="gray",lw=.7)
    axes[0,c].set_title(("Left","Right")[c]+" leg: raw and applied policy actions")
    axes[1,c].set_title(("Left","Right")[c]+" leg: motor angles (deg)")
    for r in (0,1):axes[r,c].legend(fontsize=8);axes[r,c].set_xlabel("MuJoCo simulation time (s)")
joints=["jIJ","jIO","jAB","jAG"]
ex=np.array(logs["85"]["last_50_excess_by_joint"])
axes[2,0].bar(joints,ex/ex.sum()*100,color=["tab:blue","tab:blue","tab:orange","tab:orange"]);axes[2,0].set_ylabel("share (%)");axes[2,0].set_title("85% training: acceleration penalty allocation, last 50 iterations")
for arm in ("80","85","90"):
    axes[2,1].plot(np.arange(200),log_arrays[arm]["Acceleration/gain_used"],label=arm+"%")
axes[2,1].set_ylim(-.05,1.05);axes[2,1].set_title("Training gain used: upper bound = 1");axes[2,1].set_xlabel("training iteration");axes[2,1].legend()
fig.tight_layout();fig.savefig(OUT/"saturation-reward-audit.png",dpi=170);plt.close(fig)
assert all(sha(p)==h for p,h in protected.items())
for k in ("baseline_nominal_stand","85_nominal_stand","85_forward","85_reverse","85_interactive"):
    s=summaries[k]
    print(k,json.dumps({n:s[n] for n in ("action_clip_fraction","phi0_delta_abs_mean_deg","leg_accel_rms_rad_s2","contact_normal_last_mean_N","weighted_reward_mean_per_step","weighted_reward_channel_mean_per_step")}))
print("training85",json.dumps({k:v for k,v in logs["85"].items() if k!="checkpoint_gain_states"}))
