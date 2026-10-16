"""Diagnostic only: isolate input feedback for the recorded 85% policy."""
from pathlib import Path
import sys,csv,json,time
import numpy as np
import torch,mujoco

PROJECT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(PROJECT/"sim2sim/mujoco"))
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from wheelleg_mujoco.contract import load_policy_contract,sha256_file
from wheelleg_mujoco.observation import collect_kinematic_state

OUT=Path(__file__).resolve().parent
EXPORT=OUT.parent/"export"
SESSION=OUT.parent/"interactive-20261012-100842"
torch.set_num_threads(1)
contract,manifest,model_manifest=load_policy_contract(EXPORT/"policy_manifest.json",EXPORT/"actor.ts",PROJECT/"sim2sim/mujoco/model_manifest.json")
policy=torch.jit.load(str(EXPORT/"actor.ts"),map_location="cpu").eval()
runtime=WheelLegMujocoRuntime(PROJECT/"sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",contract,model_manifest)
d=np.load(OUT/"feedback-replay.npz")
h=d["history"].reshape(-1,5,25)
a=d["action_raw"]
dt=contract.control_dt_s
names=tuple(runtime.model_map.joint_names)
protected=json.loads((OUT/"protected-before.json").read_text())
assert all(sha256_file(Path(p))==v for p,v in protected.items())
started=time.perf_counter()

def motion_stats(actions):
    actions=np.asarray(actions)
    clipped=np.clip(actions,-1,1)
    return {
        "clipped_step_rms":np.sqrt(np.mean(np.diff(clipped,axis=0)**2,axis=0)).tolist(),
        "clipped_std":clipped.std(axis=0).tolist(),
        "raw_mean":actions.mean(axis=0).tolist(),
        "raw_min":actions.min(axis=0).tolist(),
        "raw_max":actions.max(axis=0).tolist(),
        "clip_fraction":np.mean(np.abs(actions)>=1,axis=0).tolist(),
        "wheel_sign_changes_per_s":(np.sum(clipped[1:,4:6]*clipped[:-1,4:6]<0,axis=0)/((len(actions)-1)*dt)).tolist(),
    }

def change_metrics(actions,baseline):
    result=motion_stats(actions)
    br=np.array(baseline["clipped_step_rms"])
    nr=np.array(result["clipped_step_rms"])
    result["left_step_rms_reduction"]=float(1-np.linalg.norm(nr[:2])/np.linalg.norm(br[:2]))
    result["wheel_step_rms_reduction"]=float(1-np.linalg.norm(nr[4:6])/np.linalg.norm(br[4:6]))
    return result

mask=(d["time_before"]>=2)&(d["time_before"]<20)
assert np.all(d["command"][mask,:2]==0)
hh=torch.from_numpy(h[mask].copy())
ee=torch.from_numpy(d["encoded"][mask,:19].copy())
actor_input=torch.cat((hh[:,-1],ee),dim=1)
with torch.inference_mode():
    base=policy.actor(actor_input).clamp(-20,20).numpy()
assert np.max(np.abs(base-a[mask]))<1e-5
baseline_stats=motion_stats(base)
offline={}
cases={
    "current_gyro": [(0,3)],
    "current_leg_velocity": [(13,17)],
    "current_wheel_velocity": [(17,19)],
    "current_previous_leg_action": [(19,23)],
    "current_previous_wheel_action": [(23,25)],
    "current_both_wheel_feedback": [(17,19),(23,25)],
    "estimated_velocity_mean": [(25,28)],
    "context_mean": [(28,44)],
    "all_encoder_mean": [(25,44)],
}
with torch.inference_mode():
    for name,groups in cases.items():
        changed=actor_input.clone()
        for lo,hi in groups:
            changed[:,lo:hi]=changed[:,lo:hi].mean(dim=0,keepdim=True)
        value=policy.actor(changed).clamp(-20,20).numpy()
        offline[name]=change_metrics(value,baseline_stats)
    changed=actor_input.clone()
    changed[:,25:28]=torch.from_numpy((d["true_velocity"][mask]*contract.normalization["root_linear_velocity_scale"]).astype(np.float32))
    offline["actor_true_velocity"]=change_metrics(policy.actor(changed).clamp(-20,20).numpy(),baseline_stats)
    for name,groups in (
        ("history_gyro",[(0,3)]),("history_wheel_velocity",[(17,19)]),
        ("history_previous_wheel_action",[(23,25)]),
        ("history_both_wheel_feedback",[(17,19),(23,25)])):
        changed=hh.clone()
        for lo,hi in groups:
            changed[:,:,lo:hi]=changed[:,:,lo:hi].mean(dim=(0,1),keepdim=True)
        offline[name]=change_metrics(policy(changed.reshape(-1,125)).numpy(),baseline_stats)

est=d["estimated_velocity"][mask]
actual=d["true_velocity"][mask]
velocity={"bias_mps":(est-actual).mean(axis=0).tolist(),"mae_mps":np.abs(est-actual).mean(axis=0).tolist(),"actual_std_mps":actual.std(axis=0).tolist(),"estimated_std_mps":est.std(axis=0).tolist()}
with (SESSION/"control.csv").open(newline="") as f:
    rows=list(csv.DictReader(f))
logged={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
vbefore=np.stack([logged[n+"_velocity_before_rad_s"] for n in names],axis=1)
qbefore=np.stack([logged[n+"_position_before_rad"] for n in names[:4]],axis=1)
expected_v=vbefore.copy();expected_v[:,5]*=-1
np.testing.assert_allclose(h[:,-1,13:19],expected_v*contract.normalization["joint_velocity_scale"],rtol=0,atol=1e-7)
np.testing.assert_allclose(h[:,-1,9:13],qbefore-contract.q_nominal,rtol=0,atol=1e-7)
assert np.isfinite(h).all()
range_checks={name:{"max_abs":float(np.max(np.abs(h[:,-1,lo:hi]))),"clip_fraction":float(np.mean(np.abs(h[:,-1,lo:hi])>=lim-1e-6))} for name,lo,hi,lim in (("gyro",0,3,5),("leg_position",9,13,1.5),("joint_velocity",13,19,5))}
# Obtain exact initial states by replaying original actions, preserving warm starts.
snapshots={}
runtime.reset(d["command"][0])
for i in range(501):
    if i in (100,300,500):
        np.testing.assert_allclose(runtime._history,h[i],rtol=0,atol=1e-7)
        copied=mujoco.MjData(runtime.model)
        mujoco.mj_copyData(copied,runtime.model,runtime.data)
        center=h[max(0,i-50):i,-1].mean(axis=0)
        enc_center=d["encoded"][max(0,i-50):i,:19].mean(axis=0)
        snapshots[i]=(copied,runtime._history.copy(),runtime.previous_action.copy(),center,enc_center)
    runtime.step(a[i],d["command"][i])

def infer(history,case,center,enc_center,true_velocity=None):
    original=torch.from_numpy(history.astype(np.float32).reshape(1,125).copy())
    if case=="baseline":
        return policy(original).numpy()[0]
    if case.endswith("_history"):
        changed=original.reshape(1,5,25).clone()
        if case in ("fixed_previous_wheel_history","fixed_both_wheel_history"):
            changed[:,:,23:25]=torch.from_numpy(center[23:25].copy())
        if case in ("fixed_wheel_velocity_history","fixed_both_wheel_history"):
            changed[:,:,17:19]=torch.from_numpy(center[17:19].copy())
        if case=="fixed_gyro_history":
            changed[:,:,0:3]=torch.from_numpy(center[0:3].copy())
        for name,axis in (("fixed_roll_rate_history",0),("fixed_pitch_rate_history",1),("fixed_yaw_rate_history",2)):
            if case==name:
                changed[:,:,axis:axis+1]=torch.from_numpy(center[axis:axis+1].copy())
        return policy(changed.reshape(1,125)).numpy()[0]
    encoded=policy.encoder(original)[:,:19].clone()
    current=original[:,-25:].clone()
    if case=="true_velocity":
        encoded[:,:3]=torch.from_numpy(np.asarray(true_velocity*contract.normalization["root_linear_velocity_scale"],dtype=np.float32))
    elif case=="fixed_estimated_velocity":
        encoded[:,:3]=torch.from_numpy(enc_center[:3].copy())
    elif case=="fixed_context":
        encoded[:,3:19]=torch.from_numpy(enc_center[3:19].copy())
    elif case=="fixed_previous_wheel_current":
        current[:,23:25]=torch.from_numpy(center[23:25].copy())
    elif case=="fixed_wheel_velocity_current":
        current[:,17:19]=torch.from_numpy(center[17:19].copy())
    else:
        raise ValueError(case)
    return policy.actor(torch.cat((current,encoded),dim=1)).clamp(-20,20).numpy()[0]

case_names=("baseline","true_velocity","fixed_estimated_velocity","fixed_context",
            "fixed_previous_wheel_current","fixed_previous_wheel_history",
            "fixed_wheel_velocity_current","fixed_wheel_velocity_history","fixed_both_wheel_history")
extension=len(sys.argv)>1
if extension:
    case_names=tuple(sys.argv[1:])
output_prefix="gyro-followup-" if extension else ""
closed=[]
arrays={}
for start,(copied,history,previous,center,enc_center) in snapshots.items():
    for case in case_names:
        mujoco.mj_copyData(runtime.data,runtime.model,copied)
        runtime._history=history.copy();runtime.previous_action=previous.copy()
        trace=[];out_actions=[];errors=[];failure=None
        for k in range(250):
            state=collect_kinematic_state(runtime.model,runtime.data,runtime.model_map,contract)
            with torch.inference_mode():
                action=infer(runtime._history,case,center,enc_center,state.com_linear_velocity_control)
            vb=runtime.model_map.joint_velocity_mujoco(runtime.data).copy()
            result=runtime.step(action,np.array([0.,0.,.2]))
            va=runtime.model_map.joint_velocity_mujoco(runtime.data).copy()
            q=runtime.model_map.joint_position_mujoco(runtime.data).copy()
            trace.append([runtime.data.time,*q,*va,*((va-vb)/dt),result.metrics["vx_mps"],result.metrics["yaw_rate_rad_s"],result.metrics["yaw_rad"],result.metrics["tilt_rad"],result.metrics["base_height_m"],*result.applied_torque])
            out_actions.append(action)
            if case=="baseline":
                errors.append(float(np.max(np.abs(action-a[start+k]))))
                np.testing.assert_allclose(va,[logged[n+"_velocity_after_rad_s"][start+k] for n in names],rtol=0,atol=1e-7)
            m=result.metrics
            if not .10<=m["base_height_m"]<=.40: failure="base_height"
            elif m["tilt_rad"]>.80: failure="tilt"
            elif m["max_loop_closure_error_m"]>.005: failure="loop_closure"
            if failure: break
        trace=np.array(trace);out_actions=np.array(out_actions)
        selected=trace[50:] if len(trace)>100 else trace
        summary={
            "start_simulation_s":float(start*dt),"case":case,"completed_ticks":len(trace),"failure_reason":failure,
            "left_accel_rms_rad_s2":np.sqrt(np.mean(selected[:,13:15]**2,axis=0)).tolist(),
            "joint_accel_rms_rad_s2":np.sqrt(np.mean(selected[:,13:19]**2,axis=0)).tolist(),
            "left_position_std_deg":np.rad2deg(selected[:,1:3].std(axis=0)).tolist(),
            "vx_mae_mps":float(np.abs(selected[:,19]).mean()),"yaw_rate_mae_rad_s":float(np.abs(selected[:,20]).mean()),
            "yaw_change_deg":float(np.rad2deg(np.unwrap(trace[:,21])[-1]-np.unwrap(trace[:,21])[0])),
            "max_tilt_deg":float(np.rad2deg(trace[:,22]).max()),
            "height_min_max_m":[float(trace[:,23].min()),float(trace[:,23].max())],
            "actions":motion_stats(out_actions[50:] if len(trace)>100 else out_actions),
            "baseline_max_action_error":max(errors) if errors else None,
        }
        closed.append(summary)
        arrays[f"{start}_{case}_state"]=trace
        arrays[f"{start}_{case}_action"]=out_actions
        print(json.dumps({"start_s":start*dt,"case":case,"ticks":len(trace),"failure":failure,"left_accel":summary["left_accel_rms_rad_s2"],"wheel_switches":summary["actions"]["wheel_sign_changes_per_s"],"vx_mae":summary["vx_mae_mps"]}),flush=True)

# Computational recurrence only; no physical stepping or inferred stability.
frozen=[]
for start,(copied,history,previous,center,enc_center) in snapshots.items():
    physical=history[-1,:19].copy()
    for case in ("baseline","fixed_previous_wheel_current","fixed_previous_wheel_history","fixed_context"):
        fh=history.copy();fh[:,:19]=physical
        values=[]
        for k in range(300):
            with torch.inference_mode():
                action=infer(fh,case,center,enc_center,d["true_velocity"][start])
            values.append(action)
            fh[:-1]=fh[1:]
            fh[-1,:19]=physical
            fh[-1,19:25]=np.clip(action,-1,1)
        values=np.array(values)
        frozen.append({"start_simulation_s":start*dt,"case":case,"actions_last_5s":motion_stats(values[50:]),"physical_input_frozen":True,"physics_stepped":False})
        arrays[f"{start}_{case}_frozen_action"]=values

np.savez_compressed(OUT/(output_prefix+"isolated-feedback-trajectories.npz"),**arrays)
after={p:sha256_file(Path(p)) for p in protected}
assert after==protected
aggregate={}
for case in case_names:
    group=[r for r in closed if r["case"]==case]
    aggregate[case]={
        "completed_branches":sum(r["failure_reason"] is None and r["completed_ticks"]==250 for r in group),
        "left_accel_rms_average_rad_s2":np.mean([r["left_accel_rms_rad_s2"] for r in group],axis=0).tolist(),
        "wheel_switches_average_per_s":np.mean([r["actions"]["wheel_sign_changes_per_s"] for r in group],axis=0).tolist(),
        "vx_mae_average_mps":float(np.mean([r["vx_mae_mps"] for r in group])),
        "yaw_rate_mae_average_rad_s":float(np.mean([r["yaw_rate_mae_rad_s"] for r in group])),
        "max_tilt_deg":max(r["max_tilt_deg"] for r in group),
    }
report={"schema":"StationaryFeedbackIsolation85V1","source_actor_sha256":sha256_file(EXPORT/"actor.ts"),"source_completed_iterations":manifest["source_completed_iterations"],"command":[0,0,.2],
        "replay_verification":json.loads((OUT/"replay-verification.json").read_text()),"observation_independent_checks_passed":True,
        "observation_range_checks":range_checks,"same_time_cenet_velocity_error":velocity,
        "offline_baseline":baseline_stats,"offline_input_dependency":offline,
        "closed_loop_branches":closed,"closed_loop_aggregate":aggregate,"frozen_physical_input_inference":frozen,
        "input_freeze_center":"mean of prior 1 second of baseline current observations; never a deployment change",
        "evaluation_window":"five-second MuJoCo branches; omit first second from steady statistics where possible",
        "protected_files_unchanged":True,"elapsed_wall_s":time.perf_counter()-started,
        "limitations":["Input substitution is diagnostic and may be outside the training distribution.",
                      "Five-second zero-command branches do not establish deployable stability or performance at other commands.",
                      "Frozen physical inputs isolate computation only; they are not simulated locomotion.",
                      "This investigation does not compare Isaac rollout dynamics or establish the origin of learning the observed feedback."]}
(OUT/(output_prefix+"isolation-results.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({"aggregate":aggregate,"offline_reductions":{k:[v["left_step_rms_reduction"],v["wheel_step_rms_reduction"]] for k,v in offline.items()},"protected_files_unchanged":True,"elapsed_s":report["elapsed_wall_s"]}),flush=True)
