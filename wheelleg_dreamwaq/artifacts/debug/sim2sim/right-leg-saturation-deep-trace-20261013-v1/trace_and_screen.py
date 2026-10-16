"""Diagnostic-only unchanged MuJoCo cross-seed policy chronology and action-path screening."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch,torch.nn.functional as F
OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[3]
BASE=OUT.parent/"leg-acceleration-budget-search-20261012-v1"
PRIOR=OUT.parent/"right-leg-saturation-lineage-20261013-v1"
sys.path.insert(0,str(PROJECT/"sim2sim/mujoco"))
sys.path.insert(0,str(PROJECT/"scripts"))
from wheelleg_mujoco.contract import load_policy_contract,sha256_file
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from wheelleg_mujoco.observation import collect_kinematic_state
from evaluate_mujoco import _failure_reason
torch.set_num_threads(1)
suite=json.loads((PROJECT/"artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/training-suite-manifest.json").read_text())
protected=json.loads((PRIOR/"protected-before.json").read_text())
for run in suite["runs"]:
    for iteration in (0,100,200,999):
        cp=Path(run["run_dir"])/f"model_{iteration}.pt"
        protected[str(cp)]=sha256_file(cp)
    for name in ("actor.ts","policy_manifest.json"):
        p=Path(run["export_dir"])/name;protected[str(p)]=sha256_file(p)
(OUT/"protected-before.json").write_text(json.dumps(protected,indent=2),encoding="utf-8")
assert all(sha256_file(Path(p))==h for p,h in protected.items())
contract,manifest,mm=load_policy_contract(BASE/"85/export/policy_manifest.json",BASE/"85/export/actor.ts",PROJECT/"sim2sim/mujoco/model_manifest.json")
runtime=WheelLegMujocoRuntime(PROJECT/"sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",contract,mm)
dt=contract.control_dt_s
template=BASE/"85/export/actor.ts"
arrays={}
def load_checkpoint(cp):
    saved=torch.load(cp,map_location="cpu",weights_only=False)
    info=saved["infos"];state=saved["model_state_dict"]
    assert info["dreamwaq_algorithm_contract_hash"]==manifest["dreamwaq_algorithm_contract_hash"]
    assert info["dreamwaq_export_contract_hash"]==manifest["dreamwaq_export_contract_hash"]
    policy=torch.jit.load(str(template)).eval()
    mapping={k:state["cenet."+k] if k.startswith("encoder.") else state[k] for k in policy.state_dict()}
    policy.load_state_dict(mapping,strict=True)
    return policy,info,state
def statistics(d,start=100):
    z={k:np.asarray(v)[start:] for k,v in d.items()}
    if not len(z["raw"]):return None
    a=z["applied"]
    return {"samples":len(a),"vx_mae_mps":float(np.abs(z["metrics"][:,0]-z["command"][:,0]).mean()),
        "yaw_mae_rad_s":float(np.abs(z["metrics"][:,1]-z["command"][:,1]).mean()),
        "height_mae_m":float(np.abs(z["metrics"][:,3]-z["command"][:,2]).mean()),
        "leg_accel_rms_rad_s2":np.sqrt(np.mean(z["acc"][:,:4]**2,axis=0)).tolist(),
        "leg_q_p95_p5_deg":np.rad2deg(np.percentile(z["q"][:,:4],95,axis=0)-np.percentile(z["q"][:,:4],5,axis=0)).tolist(),
        "leg_target_p95_p5_deg":np.rad2deg(np.percentile(z["target"][:,:4],95,axis=0)-np.percentile(z["target"][:,:4],5,axis=0)).tolist(),
        "negative_clip_fraction_raw":np.mean(z["raw"]<=-1,axis=0).tolist(),
        "raw_mean":z["raw"].mean(axis=0).tolist(),"applied_step_rms":np.sqrt(np.mean(np.diff(a,axis=0)**2,axis=0)).tolist(),
        "wheel_switches_per_s":(np.sum(a[1:,4:]*a[:-1,4:]<0,axis=0)/((len(a)-1)*dt)).tolist(),
        "phi_delta_mean_deg":float(np.rad2deg(np.abs(np.arctan2(np.sin(z["phi"][:,0]-z["phi"][:,1]),np.cos(z["phi"][:,0]-z["phi"][:,1])))).mean()),
        "max_tilt_deg":float(np.rad2deg(z["metrics"][:,2]).max()),
        "max_loop_error_m":float(z["metrics"][:,4].max()),
        "effort_saturation_fraction":float(z["metrics"][:,5].mean())}
def run_case(policy,key,command,filter_hz=(0.,0.),ticks=500):
    command=np.asarray(command,dtype=np.float64)
    obs=runtime.reset(command)
    filtered=np.zeros(6,dtype=np.float64)
    alpha=np.r_[np.repeat(1-np.exp(-2*np.pi*filter_hz[0]*dt),4),np.repeat(1-np.exp(-2*np.pi*filter_hz[1]*dt),2)]
    alpha[np.r_[np.repeat(filter_hz[0]==0,4),np.repeat(filter_hz[1]==0,2)]]=1
    data={k:[] for k in ("history","raw","applied","target","q","v","acc","metrics","command","phi","encoded")}
    failure=None
    for tick in range(ticks):
        history=obs.copy()
        with torch.inference_mode():
            x=torch.from_numpy(obs).unsqueeze(0)
            raw=policy(x).numpy()[0]
            encoded=policy.encoder(x).numpy()[0,:19]
        filtered=filtered+alpha*(np.clip(raw,-1,1).astype(np.float64)-filtered)
        before=runtime.model_map.joint_velocity_control(runtime.data).copy()
        targets=runtime.controller.prepare(filtered)
        target=np.r_[targets.leg_position_target_mujoco,targets.wheel_velocity_target_mujoco*contract.wheel_signs]
        result=runtime.step(filtered,command)
        state=collect_kinematic_state(runtime.model,runtime.data,runtime.model_map,contract)
        m=result.metrics
        values=dict(history=history,raw=raw,applied=filtered.copy(),target=target.copy(),q=state.joint_position_control.copy(),v=state.joint_velocity_control.copy(),
            acc=(state.joint_velocity_control-before)/dt,
            metrics=[m["vx_mps"],m["yaw_rate_rad_s"],m["tilt_rad"],m["base_height_m"],m["max_loop_closure_error_m"],m["effort_saturation_fraction"]],
            command=command.copy(),phi=[m["phi0_left_rad"],m["phi0_right_rad"]],encoded=encoded)
        for name,value in values.items():data[name].append(value)
        failure=_failure_reason(runtime,m)
        if failure:break
        obs=result.observation
    for name,value in data.items():arrays[key+"__"+name]=np.array(value)
    r={"key":key,"command":command.tolist(),"filter_hz_legs_wheels":list(filter_hz),"ticks":len(data["raw"]),"expected_ticks":ticks,"failure_reason":failure,
        "steady":statistics(data),"all":statistics(data,0),"reset_action_raw":np.array(data["raw"][0]).tolist()}
    print(json.dumps({"key":key,"ticks":r["ticks"],"failure":failure,"steady":r["steady"]}),flush=True)
    return r
if __name__ == "__main__":
    started=time.perf_counter()
    cross=[]
    for run in suite["runs"]:
        if run["index"]==3:continue
        for index in (0,100,200,999):
            cp=Path(run["run_dir"])/f"model_{index}.pt"
            policy,info,state=load_checkpoint(cp)
            r=run_case(policy,f"run{run['index']}_{index+1}",[0,0,.2])
            r.update(checkpoint=str(cp),checkpoint_sha256=sha256_file(cp),seed=run["seed"],completed_iterations=info["completed_iterations"])
            if index==999:
                export=torch.jit.load(str(Path(run["export_dir"])/"actor.ts")).eval()
                x=torch.from_numpy(arrays[r["key"]+"__history"])
                with torch.inference_mode():err=float(torch.abs(policy(x)-export(x)).max())
                assert err==0;r["original_export_max_action_error"]=err
            cross.append(r)
    (OUT/"cross-seed-results.json").write_text(json.dumps(cross,indent=2),encoding="utf-8")
    np.savez_compressed(OUT/"cross-seed-trajectories.npz",**arrays)
    arrays={}
    policy=torch.jit.load(str(template)).eval()
    screen=[]
    filters={"unmodified":(0.,0.),"leg6":(6.,0.),"leg3":(3.,0.),"leg1":(1.,0.),"wheel8":(0.,8.),"wheel3":(0.,3.),"both8":(8.,8.),"both3":(3.,3.),"leg3_wheel8":(3.,8.)}
    scenarios={"stand":[0,0,.2],"forward":[1,0,.2],"reverse":[-1,0,.2],"combined":[.8,.5,.2]}
    for name,hz in filters.items():
        for scenario,command in scenarios.items():
            r=run_case(policy,name+"_"+scenario,command,hz)
            r.update(filter=name,scenario=scenario)
            screen.append(r)
            if name=="unmodified":
                prior=np.load(BASE/"saturation-reward-audit-20261012-v1/trajectories.npz")
                oldname="nominal_stand" if scenario=="stand" else scenario
                err=float(np.abs(arrays[r["key"]+"__raw"]-prior["85_"+oldname+"__action_raw"]).max())
                assert err==0;r["original_trace_max_action_error"]=err
    np.savez_compressed(OUT/"filter-screen-trajectories.npz",**arrays)
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    report={"schema":"DeepSaturationMuJoCoV1","filter_equation":"applied=(1-alpha)*previous_applied+alpha*clip(raw,-1,1); alpha=1-exp(-2*pi*fc*.02); 0Hz here means bypass, NOT frozen output",
        "reset_filter_state":[0]*6,"observation_history_uses_actual_applied_action":True,"screen":screen,"protected_files_unchanged":True,
        "elapsed_wall_s":time.perf_counter()-started,"training_started":False,"production_modified":False,"is_formal_acceptance":False}
    (OUT/"filter-screen-results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("DONE",time.perf_counter()-started,flush=True)
