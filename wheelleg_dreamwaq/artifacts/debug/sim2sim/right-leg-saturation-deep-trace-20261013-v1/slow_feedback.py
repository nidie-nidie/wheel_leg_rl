"""Diagnostic slow command correction using observable wheel encoders and yaw gyro."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
OUT=Path(__file__).resolve().parent;PROJECT=OUT.parents[3]
sys.path.insert(0,str(PROJECT/"sim2sim/mujoco"))
sys.path.insert(0,str(PROJECT/"scripts"))
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from wheelleg_mujoco.contract import load_policy_contract,sha256_file
from wheelleg_mujoco.observation import collect_kinematic_state,build_actor_observation
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS,summarize_scenario
from wheelleg_mujoco.command_practice import command_at_tick,validate_command_practice,summarize_practice
from evaluate_command_practice_mujoco import SCENARIOS
from evaluate_mujoco import _failure_reason
BASE=OUT.parent/"leg-acceleration-budget-search-20261012-v1"
contract,manifest,mm=load_policy_contract(BASE/"85/export/policy_manifest.json",BASE/"85/export/actor.ts",PROJECT/"sim2sim/mujoco/model_manifest.json")
rt=WheelLegMujocoRuntime(PROJECT/"sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",contract,mm)
policy=torch.jit.load(str(BASE/"85/export/actor.ts")).eval();torch.set_num_threads(1)
radius=mm["wheel_proxy"]["radius_m"];assert radius==.0625
dt=contract.control_dt_s;payload=validate_command_practice(manifest)
arrays={};results=[]
def stats(data,start=100):
    d={k:np.asarray(v)[start:] for k,v in data.items()}
    metrics=d["metrics"];target=d["user_command"];after=d["com_after"]
    return {"samples":len(metrics),"vx_mae_mps":float(np.abs(metrics[:,0]-target[:,0]).mean()),
        "vx_signed_mean_error_mps":float((metrics[:,0]-target[:,0]).mean()),
        "yaw_mae_rad_s":float(np.abs(metrics[:,1]-target[:,1]).mean()),
        "leg_accel_rms_rad_s2":np.sqrt(np.mean(d["acc"][:,:4]**2,axis=0)).tolist(),
        "net_displacement_m":float(np.linalg.norm(after[-1]-d["com_before"][0])),
        "path_length_m":float(np.linalg.norm(after-d["com_before"],axis=1).sum()),
        "max_tilt_deg":float(np.rad2deg(metrics[:,2]).max()),
        "max_loop_error_m":float(metrics[:,4].max()),
        "wheel_odometry_bias_mps":float((d["sensor_before"][:,0]-d["true_velocity_before"][:,0]).mean()),
        "wheel_odometry_mae_mps":float(np.abs(d["sensor_before"][:,0]-d["true_velocity_before"][:,0]).mean()),
        "filtered_sensor_mean":d["filtered_sensor"].mean(axis=0).tolist(),
        "command_correction_mean":d["correction"].mean(axis=0).tolist()}
def trial(name,scenario,command_fn,ticks,kp,ki,observer_hz=1.0,zero_only=False,settled_delay_s=0.0,push=None):
    obs=rt.reset(np.array(command_fn(0)))
    rotation=rt.data.xmat[rt.model_map.base_body_id].reshape(3,3).copy()
    force_world=np.zeros(3) if push is None else rotation@contract.r_control_from_mujoco.T@np.array(push["force_control_n"])
    filtered=np.zeros(6);estimate=np.zeros(2);integral=np.zeros(2)
    alpha=np.r_[np.repeat(1-np.exp(-2*np.pi*6*dt),4),np.repeat(1-np.exp(-2*np.pi*2*dt),2)]
    observer_alpha=1-np.exp(-2*np.pi*observer_hz*dt)
    data={k:[] for k in ("history","raw","applied","target","q","v","acc","metrics","user_command","policy_command","correction","sensor_before","filtered_sensor","true_velocity_before","encoded","external_force_world","com_before","com_after","time")}
    failure=None;formal_rows=[];practice_rows=[];stable_steps=0;last_command=np.array(command_fn(0))
    for tick in range(ticks):
        current=np.array(command_fn(tick),dtype=np.float64)
        state=collect_kinematic_state(rt.model,rt.data,rt.model_map,contract)
        sensor=np.array([radius*state.joint_velocity_control[4:6].mean(),state.angular_velocity_control[2]])
        estimate+=observer_alpha*(sensor-estimate)
        error=current[:2]-estimate
        # Fixed bounded PI correction, no simulator body velocity used by controller.
        stable_steps=stable_steps+1 if np.array_equal(current,last_command) else 0
        last_command=current.copy()
        limits=np.array([.06,.06]) if zero_only else np.array([.30,.30])
        active=not zero_only or (np.all(current[:2]==0) and stable_steps*dt>=settled_delay_s)
        if active:integral=np.clip(integral+np.array(ki)*error*dt,-limits,limits)
        correction=np.clip(np.array(kp)*error+integral,-limits,limits) if active else np.zeros(2)
        policy_command=current.copy();policy_command[:2]+=correction
        policy_command[0]=np.clip(policy_command[0],-contract.normalization["vx_max_abs"],contract.normalization["vx_max_abs"])
        policy_command[1]=np.clip(policy_command[1],-contract.normalization["yaw_rate_max_abs"],contract.normalization["yaw_rate_max_abs"])
        # Refresh only present command frame; all older commands retain their actual chronology.
        frame=build_actor_observation(state,policy_command,rt.previous_action,contract)
        rt._history[-1]=frame;obs=rt._history.reshape(-1).astype(np.float32,copy=True)
        with torch.inference_mode():
            x=torch.from_numpy(obs).unsqueeze(0);raw=policy(x).numpy()[0];encoded=policy.encoder(x).numpy()[0,:19]
        filtered+=alpha*(np.clip(raw,-1,1).astype(np.float64)-filtered)
        before_v=rt.model_map.joint_velocity_control(rt.data).copy()
        before_xy=rt.data.xipos[rt.model_map.base_body_id,:2].copy()
        target=rt.controller.prepare(filtered).leg_position_target_mujoco.copy()
        rt.data.xfrc_applied[:]=0
        if push is not None and push["start_tick"]<=tick<push["end_tick"]:
            rt.data.xfrc_applied[rt.model_map.base_body_id,:3]=force_world
        applied_force=rt.data.xfrc_applied[rt.model_map.base_body_id,:3].copy()
        result=rt.step(filtered,policy_command)
        m=result.metrics;after_v=rt.model_map.joint_velocity_control(rt.data).copy()
        after_xy=rt.data.xipos[rt.model_map.base_body_id,:2].copy()
        values=dict(history=obs,raw=raw,applied=filtered.copy(),target=target,q=rt.model_map.joint_position_control(rt.data).copy(),v=after_v,
            acc=(after_v-before_v)/dt,metrics=[m["vx_mps"],m["yaw_rate_rad_s"],m["tilt_rad"],m["base_height_m"],m["max_loop_closure_error_m"],m["effort_saturation_fraction"]],
            user_command=current,policy_command=policy_command,correction=correction,sensor_before=sensor,filtered_sensor=estimate.copy(),true_velocity_before=state.com_linear_velocity_control.copy(),
            encoded=encoded,external_force_world=applied_force,com_before=before_xy,com_after=after_xy,time=float(rt.data.time))
        for k,v in values.items():data[k].append(np.array(v,copy=True))
        failure=_failure_reason(rt,m)
        if failure:break
        formal_rows.append(m.copy())
        practice_rows.append({"tick":tick,**m,"com_xy_before":before_xy.tolist(),"com_xy_after":after_xy.tolist()})
    rt.data.xfrc_applied[:]=0
    key=name+"__"+scenario
    for k,v in data.items():arrays[key+"__"+k]=np.array(v)
    r={"key":key,"variant":name,"scenario":scenario,"kp":list(kp),"ki":list(ki),"observer_hz":observer_hz,"zero_only":zero_only,"settled_delay_s":settled_delay_s,"push":push,"ticks":len(data["raw"]),
        "failure_reason":failure,"steady":stats(data) if len(data["raw"])>100 else None}
    if scenario in {n for n,c in FORMAL_SCENARIOS}:r["formal_summary"]=summarize_scenario(scenario,tuple(command_fn(0)),formal_rows,expected_ticks=ticks,failure_reason=failure)
    if scenario in {n for n,c in SCENARIOS}:r["practice_summary"]=summarize_practice(scenario,np.array(command_fn(0)),practice_rows,failure)
    results.append(r)
    print(json.dumps({"key":key,"ticks":r["ticks"],"failure":failure,"steady":r["steady"],"practice":r.get("practice_summary",{}).get("performance_accepted")}),flush=True)
    return r
if __name__=="__main__":
    profiles={"feedback_off":([0,0],[0,0]),"slow_pi":([.25,.25],[.15,.15]),"pi_medium":([.5,.5],[.3,.3]),"pi_integral":([0,0],[.3,.3])}
    for name,(kp,ki) in profiles.items():
        trial(name,"stand60",lambda tick:[0,0,.2],3000,kp,ki)
        for scenario,c in SCENARIOS:trial(name,scenario,lambda tick,c=np.array(c):command_at_tick(c,tick,payload),500,kp,ki)
    # All states for exploratory runs, separated from frozen-reference result directories.
    np.savez_compressed(OUT/"slow-feedback-trajectories.npz",**arrays)
    protected=json.loads((OUT/"protected-before.json").read_text());assert all(sha256_file(Path(p))==h for p,h in protected.items())
    (OUT/"slow-feedback-results.json").write_text(json.dumps({"results":results,"wheel_radius_m":radius,
        "sensor_assumption":"mean wheel encoder rates * radius; yaw from observable gyro; rolling contact assumed; simulator true body velocity only in recorded diagnostics",
        "metrics_target":"original user command, never modified policy command","filter_hz":[6,2],"correction_limits":[.3,.3],
        "history_updates":"only current frame refreshed; preceding 4 retain real command and processed-action chronology",
        "training_started":False,"production_modified":False,"not_formal_acceptance":True,"protected_files_unchanged":True},indent=2),encoding="utf-8")
    print("DONE",len(results),sum(r["ticks"] for r in results),flush=True)
