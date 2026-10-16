from pathlib import Path
import json,numpy as np,torch
folder=Path(r"E:\wheel_leg_rl-main\wheelleg_dreamwaq\artifacts\debug\sim2sim\joint-telemetry-20261011-141922\observation-feedback")
d=np.load(folder/"feedback-replay.npz")
policy=torch.jit.load(r"E:\wheel_leg_rl-main\wheelleg_dreamwaq\artifacts\phase2_dreamwaq\resume-run03-1000-to-1500-20261011-133153\stopped-export\actor.ts",map_location="cpu").eval()
torch.set_num_threads(1)
commands=d["command"]
changes=np.r_[True,np.any(np.abs(np.diff(commands,axis=0))>1e-9,axis=1)]
starts=np.flatnonzero(changes)
ends=np.r_[starts[1:],len(commands)]
segments=[]
for start,end in zip(starts,ends):
    if end-start>=100:
        segments.append({"start_frame":int(start),"end_frame":int(end),"start_time_s":float(d["time_before"][start]),"end_time_s":float(d["time_after"][end-1]),"command":commands[start].tolist()})
mask=(d["time_before"]>=2)&(d["time_before"]<=10)
wheel_actions=d["action_clipped"][mask,4:6]
wheel_target=wheel_actions*25
wheel_stats={
    "clipped_action_p05_p95":np.quantile(wheel_actions,[.05,.95],axis=0).tolist(),
    "action_sign_change_fraction":np.mean(wheel_actions[1:]*wheel_actions[:-1]<0,axis=0).tolist(),
    "mean_abs_target_step_change_rad_s":np.mean(np.abs(np.diff(wheel_target,axis=0)),axis=0).tolist(),
    "maximum_abs_target_step_change_rad_s":np.max(np.abs(np.diff(wheel_target,axis=0)),axis=0).tolist(),
    "target_at_25rad_s_fraction":np.mean(np.abs(wheel_target)>=25-1e-9,axis=0).tolist(),
}
cross={}
for segment in segments:
    start,end=segment["start_frame"]+25,segment["end_frame"]
    if end-start<50:continue
    x=torch.from_numpy(np.concatenate((d["history"][start:end,-25:],d["encoded"][start:end,:3],d["encoded"][start:end,3:19]),axis=1))
    with torch.inference_mode():base=np.clip(policy.actor(x).clamp(-20,20).numpy(),-1,1)
    base_rms=np.sqrt(np.mean(np.diff(base,axis=0)**2,axis=0))[:2]
    cases={}
    for name,bounds in {"true_velocity":None,"wheel_velocity_mean":[(17,19)],"previous_wheel_action_mean":[(23,25)],"angular_velocity_mean":[(0,3)],"wheel_feedback_and_angular_mean":[(17,19),(23,25),(0,3)]}.items():
        xx=x.clone()
        if bounds is None:
            xx[:,25:28]=torch.from_numpy(d["true_velocity"][start:end].astype(np.float32))*2
        else:
            for a,b in bounds:
                xx[:,a:b]=xx[:,a:b].mean(dim=0,keepdim=True)
        with torch.inference_mode():y=np.clip(policy.actor(xx).clamp(-20,20).numpy(),-1,1)
        rms=np.sqrt(np.mean(np.diff(y,axis=0)**2,axis=0))[:2]
        cases[name]={"per_joint_output_step_rms_reduction":(1-rms/base_rms).tolist(),"mean_reduction":float(np.mean(1-rms/base_rms))}
    cross[f"{segment['start_time_s']:.2f}-{segment['end_time_s']:.2f}s"]={"command":segment["command"],"analysis_start_s":float(d["time_before"][start]),"frames":end-start,"cases":cases,"vx_mae_mps":float(np.mean(np.abs(d["estimated_velocity"][start:end,0]-d["true_velocity"][start:end,0])))}
# Combine feedback groups on the exact parking window; do not send results to runtime.
x=torch.from_numpy(np.concatenate((d["history"][mask,-25:],d["encoded"][mask,:3],d["encoded"][mask,3:19]),axis=1))
with torch.inference_mode():base=np.clip(policy.actor(x).clamp(-20,20).numpy(),-1,1)
base_rms=np.sqrt(np.mean(np.diff(base,axis=0)**2,axis=0))[:2]
combined={}
for name,bounds in {"wheel_feedback_mean":[(17,19),(23,25)],"wheel_feedback_and_angular_mean":[(17,19),(23,25),(0,3)]}.items():
    xx=x.clone()
    for a,b in bounds:xx[:,a:b]=xx[:,a:b].mean(dim=0,keepdim=True)
    with torch.inference_mode():y=np.clip(policy.actor(xx).clamp(-20,20).numpy(),-1,1)
    rms=np.sqrt(np.mean(np.diff(y,axis=0)**2,axis=0))[:2]
    combined[name]={"per_joint_output_step_rms_reduction":(1-rms/base_rms).tolist(),"mean_reduction":float(np.mean(1-rms/base_rms))}
summary={"stationary_wheel_actions":wheel_stats,"stationary_combined_counterfactuals":combined,"constant_command_segments":cross}
(folder/"feedback-crosschecks.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary))
