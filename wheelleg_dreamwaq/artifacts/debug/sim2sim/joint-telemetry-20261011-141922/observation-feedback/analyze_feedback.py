"""Analyze aligned estimates and offline network sensitivity on the recorded trace."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--export",type=Path,required=True)
args=parser.parse_args()
data=np.load(args.output/"feedback-replay.npz")
policy=torch.jit.load(str(args.export/"actor.ts"),map_location="cpu").eval()
torch.set_num_threads(1)
time=data["time_before"]
history=data["history"]
encoded=data["encoded"]
actual=data["true_velocity"]
estimated=data["estimated_velocity"]
mask=(time>=2)&(time<=10)
indices=np.flatnonzero(mask)
assert len(indices)>300
dt=.02
groups={
    "angular_velocity":(0,3),
    "projected_gravity":(3,6),
    "command":(6,9),
    "leg_positions":(9,13),
    "left_leg_velocities":(13,15),
    "right_leg_velocities":(15,17),
    "wheel_velocities":(17,19),
    "previous_leg_actions":(19,23),
    "previous_wheel_actions":(23,25),
}
def velocity_statistics(selected):
    error=estimated[selected]-actual[selected]
    return {
        "frames":int(selected.sum()),
        "actual_mean_mps":actual[selected].mean(axis=0).tolist(),
        "estimated_mean_mps":estimated[selected].mean(axis=0).tolist(),
        "bias_mps":error.mean(axis=0).tolist(),
        "mae_mps":np.abs(error).mean(axis=0).tolist(),
        "rmse_mps":np.sqrt(np.mean(error**2,axis=0)).tolist(),
        "actual_std_mps":actual[selected].std(axis=0).tolist(),
        "estimated_std_mps":estimated[selected].std(axis=0).tolist(),
        "error_p95_abs_mps":np.quantile(np.abs(error),.95,axis=0).tolist(),
    }
def dominant_frequency(values):
    values=np.asarray(values)-np.mean(values,axis=0)
    if values.ndim==1:
        values=values[:,None]
    power=np.sum(np.abs(np.fft.rfft(values*np.hanning(len(values))[:,None],axis=0))**2,axis=1)
    frequencies=np.fft.rfftfreq(len(values),dt)
    power[frequencies<1]=0.
    return float(frequencies[np.argmax(power)])
current=history[mask,-25:]
features={}
for name,(start,end) in groups.items():
    values=current[:,start:end]
    features[name]={
        "feature_std":values.std(axis=0).tolist(),
        "step_change_rms":float(np.sqrt(np.mean(np.diff(values,axis=0)**2))),
        "dominant_frequency_above_1hz":dominant_frequency(values),
    }
nonnormal_limits={"angular_velocity":(0,3,5.),"leg_positions":(9,13,1.5),"joint_velocities":(13,19,5.)}
range_checks={name:{"max_abs_normalized":float(np.abs(history[:,-25:][:,start:end]).max()),
                   "at_clip_fraction":float(np.mean(np.abs(history[:,-25:][:,start:end])>=limit-1e-6))}
              for name,(start,end,limit) in nonnormal_limits.items()}
assert np.isfinite(history).all() and np.isfinite(encoded).all()
all_inputs=torch.from_numpy(np.concatenate((history[:,-25:],encoded[:,:3],encoded[:,3:19]),axis=1))
with torch.inference_mode():
    all_reconstructed=policy.actor(all_inputs).clamp(-20,20).numpy()
batch_reconstruction_error=float(np.max(np.abs(all_reconstructed-data["action_raw"])))
assert batch_reconstruction_error<1e-5,batch_reconstruction_error
inputs=all_inputs[mask].clone()
hist_tensor=torch.from_numpy(history[mask]).clone()
with torch.inference_mode():
    baseline=policy.actor(inputs).clamp(-20,20).numpy()
counterfactual={}
def run_case(name,changed):
    with torch.inference_mode():
        counterfactual[name]=policy.actor(changed).clamp(-20,20).numpy()
changed=inputs.clone()
changed[:,25:28]=torch.from_numpy(actual[mask].astype(np.float32))*float(data["normalization_velocity_scale"])
run_case("actor_true_velocity",changed)
changed=inputs.clone()
changed[:,25]=torch.from_numpy(actual[mask,0].astype(np.float32))*float(data["normalization_velocity_scale"])
run_case("actor_true_vx_only",changed)
for name,start,end in [("actor_velocity_mean",25,28),("actor_context_mean",28,44),
                       ("actor_all_encoder_mean",25,44),("actor_current_observation_mean",0,25)]:
    changed=inputs.clone()
    changed[:,start:end]=changed[:,start:end].mean(dim=0,keepdim=True)
    run_case(name,changed)
for name,(start,end) in groups.items():
    if name=="command":
        continue
    changed=inputs.clone()
    changed[:,start:end]=changed[:,start:end].mean(dim=0,keepdim=True)
    run_case("actor_"+name+"_mean",changed)
    # Freeze one history group; rerun BOTH encoder and current-observation routes.
    changed_hist=hist_tensor.reshape(-1,5,25).clone()
    changed_hist[:,:,start:end]=changed_hist[:,:,start:end].mean(dim=0,keepdim=True)
    with torch.inference_mode():
        e=policy.encoder(changed_hist.reshape(-1,125))
        changed=torch.cat((changed_hist[:,-1,:],e[:,:3],e[:,3:19]),dim=1)
    run_case("history_"+name+"_mean",changed)
def motion_statistics(actions):
    clipped=np.clip(actions,-1,1)
    return {
        "raw_step_change_rms":np.sqrt(np.mean(np.diff(actions,axis=0)**2,axis=0)).tolist(),
        "clipped_step_change_rms":np.sqrt(np.mean(np.diff(clipped,axis=0)**2,axis=0)).tolist(),
        "raw_std":actions.std(axis=0).tolist(),
        "clipped_std":clipped.std(axis=0).tolist(),
        "clipped_mean":clipped.mean(axis=0).tolist(),
        "clip_fraction":np.mean(np.abs(actions)>=1,axis=0).tolist(),
    }
baseline_stats=motion_statistics(baseline)
base_rms=np.asarray(baseline_stats["clipped_step_change_rms"])
case_stats={}
for name,value in counterfactual.items():
    item=motion_statistics(value)
    item["left_leg_mean_step_rms_reduction_fraction"]=float(np.mean(
        1-np.asarray(item["clipped_step_change_rms"])[:2]/base_rms[:2]))
    item["left_leg_per_joint_step_rms_reduction_fraction"]=(1-np.asarray(item["clipped_step_change_rms"])[:2]/base_rms[:2]).tolist()
    case_stats[name]=item
# Attribute ACTOR input changes between adjacent frames; this is feed-forward dependence,
# not a causal statement about the source of changes in the physical feedback loop.
attribution_groups={**groups,"estimated_velocity":(25,28),"context_mu":(28,44)}
previous=inputs[:-1].clone()
delta=inputs[1:]-previous
expected_delta=torch.from_numpy(np.diff(baseline,axis=0))
attribution={}
for output_index,name in enumerate(("jIJ","jIO")):
    contribution=torch.zeros_like(delta)
    subdivisions=32
    for fraction in (np.arange(subdivisions)+.5)/subdivisions:
        interpolated=(previous+float(fraction)*delta).detach().requires_grad_(True)
        output=policy.actor(interpolated)[:,output_index]
        gradient=torch.autograd.grad(output.sum(),interpolated)[0]
        contribution+=gradient.detach()*delta/subdivisions
    summed=contribution.sum(dim=1).numpy()
    target=expected_delta[:,output_index].numpy()
    residual=float(np.max(np.abs(summed-target)))
    assert residual<1e-4,residual
    group_values={group:contribution[:,start:end].sum(dim=1).numpy()
                  for group,(start,end) in attribution_groups.items()}
    attribution[name]={"temporal_difference_reconstruction_max_error":residual,
        "groups":{group:{"contribution_rms":float(np.sqrt(np.mean(value**2))),
                         "signed_projection_fraction":float(np.dot(value,target)/np.dot(target,target))}
                  for group,value in group_values.items()}}
report={
    "all_frames":len(time),"stationary_window_s":[2,10],
    "synchronization":"estimated and actual velocities both sampled before the same action",
    "velocity_axes":"control frame COM velocity, m/s",
    "velocity":{"all":velocity_statistics(np.ones(len(time),dtype=bool)),
                "stationary":velocity_statistics(mask),
                "forward":velocity_statistics(data["command"][:,0]>.1),
                "reverse":velocity_statistics(data["command"][:,0]<-.1)},
    "stationary_velocity_peak_frequency_above_1hz":{
        "actual_vx":dominant_frequency(actual[mask,0]),
        "estimated_vx":dominant_frequency(estimated[mask,0]),
        "error_vx":dominant_frequency(estimated[mask,0]-actual[mask,0])},
    "input_finiteness_passed":True,"observation_range_checks":range_checks,
    "stationary_observation_groups":features,
    "stationary_context_feature_std":encoded[mask,3:19].std(axis=0).tolist(),
    "batch_actor_reconstruction_max_abs_error":batch_reconstruction_error,
    "stationary_baseline_actions":baseline_stats,
    "offline_counterfactuals":case_stats,
    "actor_temporal_change_attribution":attribution,
    "interpretation_limit":"Holding network inputs or replacing estimates only measures feed-forward sensitivity on saved states. It does not predict closed-loop stability or establish physical root cause. Signed projections may be negative or exceed one due to cancellation.",
}
(args.output/"feedback-analysis.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
np.savez_compressed(args.output/"offline-counterfactual-actions.npz",baseline=baseline,
                   time=time[mask],**counterfactual)
# Plot synchronized estimates and the input dependence of left-leg output.
fig,axes=plt.subplots(4,2,figsize=(16,14))
for ax,selected in [(axes[0,0],np.ones(len(time),dtype=bool)),(axes[0,1],mask)]:
    ax.plot(time[selected],actual[selected,0],label="actual vx before action")
    ax.plot(time[selected],estimated[selected,0],label="CENet vx",alpha=.8)
    ax.plot(time[selected],data["command"][selected,0],label="command vx",alpha=.7)
    ax.set_ylabel("m/s")
    ax.legend(fontsize=8)
axes[0,0].set_title("Full user command trace")
axes[0,1].set_title("Stationary command, synchronized speed")
for index,label in enumerate(("wx","wy","wz")):
    axes[1,0].plot(time[mask],data["angular_velocity"][mask,index],label=label)
axes[1,0].set_ylabel("rad/s"); axes[1,0].legend()
for index,label in enumerate(("left wheel","right wheel")):
    axes[1,1].plot(time[mask],current[:,17+index]/.05,label=label)
axes[1,1].set_ylabel("wheel speed in control frame (rad/s)"); axes[1,1].legend()
zoom=(time[mask]>=2)&(time[mask]<=3)
best_case=max(case_stats,key=lambda name:case_stats[name]["left_leg_mean_step_rms_reduction_fraction"])
for index in range(2):
    axes[2,index].plot(time[mask][zoom],baseline[zoom,index],label="recorded output")
    axes[2,index].plot(time[mask][zoom],counterfactual["actor_true_velocity"][zoom,index],label="offline true velocity",alpha=.8)
    axes[2,index].plot(time[mask][zoom],counterfactual[best_case][zoom,index],label="largest reduction: "+best_case,alpha=.8)
    axes[2,index].set_ylabel(("jIJ","jIO")[index]+" raw action"); axes[2,index].legend(fontsize=7)
choices=["actor_true_velocity","actor_velocity_mean","actor_context_mean",
         "actor_angular_velocity_mean","actor_wheel_velocities_mean","actor_previous_wheel_actions_mean",
         "history_angular_velocity_mean","history_wheel_velocities_mean","history_previous_wheel_actions_mean"]
labels=["True velocity","Fixed velocity","Fixed context","Fixed current gyro","Fixed current wheel speed",
        "Fixed current prev wheel action","Fixed history gyro","Fixed history wheel speed","Fixed history prev wheel action"]
for index in range(2):
    reduction=[100*case_stats[name]["left_leg_per_joint_step_rms_reduction_fraction"][index] for name in choices]
    axes[3,index].barh(labels,reduction)
    axes[3,index].axvline(0,color="black",linewidth=.8)
    axes[3,index].set_xlabel("Output step-change RMS reduction (%); offline only")
for row in axes[:3]:
    for ax in row:
        ax.set_xlabel("simulation time (s)"); ax.grid(alpha=.2)
fig.suptitle("Observation and estimator feedback: recorded data and offline input sensitivity")
fig.tight_layout()
fig.savefig(args.output/"feedback-overview.png",dpi=150)
plt.close(fig)
print(json.dumps({
    "replay_frames":len(time),"stationary_velocity":report["velocity"]["stationary"],
    "velocity_frequency":report["stationary_velocity_peak_frequency_above_1hz"],
    "range_checks":range_checks,
    "counterfactual_left_step_rms_reduction":{
        name:case_stats[name]["left_leg_mean_step_rms_reduction_fraction"] for name in choices},
    "attribution_signed_projection":{
        name:{group:value["signed_projection_fraction"] for group,value in item["groups"].items()}
        for name,item in attribution.items()},
    "batch_reconstruction_error":batch_reconstruction_error,
}),flush=True)
