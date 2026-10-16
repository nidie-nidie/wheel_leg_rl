"""Independently audit commanded/actuator torque and trace network computations."""
from pathlib import Path
import csv,json,sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
PROJECT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(PROJECT/"sim2sim/mujoco"))
import hashlib
def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
torch.set_num_threads(1)
d=np.load(P/"feedback-replay.npz")
m=json.loads((P.parent/"export/policy_manifest.json").read_text())
policy=torch.jit.load(str(P.parent/"export/actor.ts"),map_location="cpu").eval()
names=("jIJ","jIO","jAB","jAG","jwheel_left","jwheel_right")
actions=np.clip(d["action_raw"].astype(np.float64),-1,1)
qnom=np.asarray(m["q_nominal"])
target=np.clip(qnom+actions[:,:4]*m["control"]["leg_action_scale"],m["control"]["leg_target_lower"],m["control"]["leg_target_upper"])
wtarget=actions[:,4:]*m["control"]["wheel_action_scale"]*np.array([1.,-1.])
limits=np.array([m["actuators"]["legs"]["effort_limit_sim"]]*4+[m["actuators"]["wheels"]["effort_limit_sim"]]*2)
command_error=np.zeros(6);actuator_error=np.zeros(6);external=np.zeros(6)
count=0
with (P.parent/"interactive-20261012-100842/physics.csv").open(newline="") as f:
    for index,row in enumerate(csv.DictReader(f)):
        i=index//m["timing"]["mujoco_physics_steps_per_action"]
        q=np.array([float(row[n+"_position_before_rad"]) for n in names])
        v=np.array([float(row[n+"_velocity_before_rad_s"]) for n in names])
        expected=np.r_[m["actuators"]["legs"]["stiffness"]*(target[i]-q[:4])-m["actuators"]["legs"]["damping"]*v[:4],
                       m["actuators"]["wheels"]["damping"]*(wtarget[i]-v[4:])]
        expected=np.clip(expected,-limits,limits)
        commanded=np.array([float(row[n+"_torque_command_nm"]) for n in names])
        produced=np.array([float(row[n+"_qfrc_actuator_nm"]) for n in names])
        applied=np.array([float(row[n+"_qfrc_applied_nm"]) for n in names])
        command_error=np.maximum(command_error,np.abs(commanded-expected))
        actuator_error=np.maximum(actuator_error,np.abs(produced-commanded))
        external=np.maximum(external,np.abs(applied))
        count+=1
print(json.dumps({"command_error_nm":command_error.tolist(),"actuator_error_nm":actuator_error.tolist(),"applied_nm":external.tolist()}),flush=True)
assert count==len(actions)*20
assert command_error.max()<1e-10 and actuator_error.max()<1e-10
assert external.max()==0.
audit={"physics_steps":count,"command_vs_frozen_pd_law_max_abs_error_nm":command_error.tolist(),"actuator_vs_command_max_abs_error_nm":actuator_error.tolist(),"selected_joint_qfrc_applied_max_abs_nm":external.tolist(),"passed":True}
mask=(d["time_before"]>=2)&(d["time_before"]<20)
hist=torch.from_numpy(d["history"][mask].copy())
prev=hist[:-1]
delta=hist[1:]-prev
with torch.inference_mode():
    output=policy(hist).numpy()
groups={"gyro":(0,3),"gravity":(3,6),"command":(6,9),"leg_position":(9,13),"leg_velocity":(13,17),"wheel_velocity":(17,19),"previous_leg_action":(19,23),"previous_wheel_action":(23,25)}
attribution={}
for output_index,name in ((4,"left_wheel"),(5,"right_wheel")):
    expected=np.diff(output[:,output_index])
    for subdivisions in (32,64,128):
        contribution=torch.zeros_like(delta)
        for fraction in (np.arange(subdivisions)+.5)/subdivisions:
            inp=(prev+float(fraction)*delta).detach().requires_grad_(True)
            out=policy(inp)[:,output_index]
            gradient=torch.autograd.grad(out.sum(),inp)[0]
            contribution+=gradient.detach()*delta/subdivisions
        cc=contribution.reshape(-1,5,25).numpy()
        residual=float(np.max(np.abs(cc.sum(axis=(1,2))-expected)))
        if residual<3e-4:break
    assert residual<3e-4,(name,residual)
    values={group:cc[:,:,lo:hi].sum(axis=(1,2)) for group,(lo,hi) in groups.items()}
    attribution[name]={"integration_subdivisions":subdivisions,"max_change_reconstruction_error":residual,
                      "raw_action_change_rms":float(np.sqrt(np.mean(expected**2))),
                      "group_contributions":{group:{"rms":float(np.sqrt(np.mean(v**2))),"signed_projection":float(np.dot(v,expected)/np.dot(expected,expected))} for group,v in values.items()},
                      "history_frame_signed_projection":[float(np.dot(cc[:,j,:].sum(axis=1),expected)/np.dot(expected,expected)) for j in range(5)]}
report={"actuator_audit":audit,"network_temporal_attribution":attribution,
        "attribution_limit":"A computation decomposition on recorded states, not physical causal percentages. Negative and over-one projections can occur through cancellation.",
        "protected_files_unchanged":all(sha256_file(Path(p))==v for p,v in json.loads((P/"protected-before.json").read_text()).items())}
assert report["protected_files_unchanged"]
(P/"actuator-and-network-audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report),flush=True)

r=json.loads((P/"isolation-results.json").read_text())
chosen=["baseline","true_velocity","fixed_previous_wheel_history","fixed_wheel_velocity_history","fixed_both_wheel_history","fixed_context"]
labels=["Baseline","True velocity","Fixed previous wheel actions","Fixed wheel velocity","Fixed both wheel inputs","Fixed context (2/3 tilt failures)"]
agg=r["closed_loop_aggregate"]
fig,axes=plt.subplots(1,3,figsize=(15,5),layout="constrained")
yy=np.arange(len(chosen))
for i,(key,label) in enumerate((("left_accel_rms_average_rad_s2","Left acceleration RMS (rad/s2)"),("wheel_switches_average_per_s","Wheel direction switches (/s)"),("vx_mae_average_mps","Zero-command vx MAE (m/s)"))):
    values=[float(np.mean(agg[k][key])) if isinstance(agg[k][key],list) else agg[k][key] for k in chosen]
    axes[i].barh(yy,values,color=["#547a97","#547a97","#bd9861","#bd9861","#bd9861","#c26565"])
    axes[i].set_yticks(yy,labels if i==0 else [""]*len(yy))
    axes[i].invert_yaxis();axes[i].set_xlabel(label);axes[i].grid(axis="x",alpha=.2)
    for j,v in enumerate(values):axes[i].text(v,j,f" {v:.3g}",va="center")
fig.suptitle("85% policy: three matched MuJoCo starting states, 5 seconds per branch\nInput substitutions are diagnostic; lower jitter alone is not acceptance")
fig.savefig(P/"feedback-isolation-comparison.png",dpi=170)
