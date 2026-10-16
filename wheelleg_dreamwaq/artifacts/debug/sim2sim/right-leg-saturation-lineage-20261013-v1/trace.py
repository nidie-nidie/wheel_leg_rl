"""Trace saved policy lineage without training or changing deployment files."""
from pathlib import Path
import sys,json,csv,time
import numpy as np
import torch,torch.nn.functional as F
OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[3]
BASE=OUT.parent/"leg-acceleration-budget-search-20261012-v1"
sys.path.insert(0,str(PROJECT/"sim2sim/mujoco"))
sys.path.insert(0,str(PROJECT/"scripts"))
from wheelleg_mujoco.contract import load_policy_contract,sha256_file
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from evaluate_mujoco import _failure_reason
torch.set_num_threads(1)
protected=json.loads((OUT/"protected-before.json").read_text())
assert all(sha256_file(Path(p))==h for p,h in protected.items())
inputs=json.loads((OUT/"checkpoint-inputs.json").read_text())
contract,manifest,model_manifest=load_policy_contract(BASE/"85/export/policy_manifest.json",BASE/"85/export/actor.ts",PROJECT/"sim2sim/mujoco/model_manifest.json")
runtime=WheelLegMujocoRuntime(PROJECT/"sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",contract,model_manifest)
command=np.array([0.,0.,.2])
reset_history=runtime.reset(command)
saved=np.load(BASE/"85/feedback-root-cause-20261012-v1/feedback-replay.npz")
common_indices=np.arange(100,1000,10)
common_history=saved["history"][common_indices].copy()
assert np.all(saved["command"][common_indices]==command)
batch=torch.from_numpy(np.vstack((reset_history[None,:],common_history)))
np.savez_compressed(OUT/"common-inputs.npz",histories=batch.numpy(),original_interactive_indices=common_indices)
arrays={}
results=[]
validation=[]
started=time.perf_counter()

def loaded(cp):
    full=torch.load(cp,map_location="cpu",weights_only=False)
    info=full["infos"]
    assert info["dreamwaq_algorithm_contract_hash"]==manifest["dreamwaq_algorithm_contract_hash"]
    assert info["dreamwaq_export_contract_hash"]==manifest["dreamwaq_export_contract_hash"]
    state=full["model_state_dict"]
    policy=torch.jit.load(str(BASE/"85/export/actor.ts"),map_location="cpu").eval()
    mapped={k:state["cenet."+k] if k.startswith("encoder.") else state[k] for k in policy.state_dict()}
    policy.load_state_dict(mapped,strict=True)
    assert all(torch.equal(policy.state_dict()[k],value) for k,value in mapped.items())
    return policy,state,info

def independent(state,x):
    encoded=x
    for i in (0,2):encoded=F.elu(F.linear(encoded,state[f"cenet.encoder.{i}.weight"],state[f"cenet.encoder.{i}.bias"]))
    encoded=F.linear(encoded,state["cenet.encoder.4.weight"],state["cenet.encoder.4.bias"])
    z=torch.cat((x[:,-25:],encoded[:,:19]),dim=-1)
    for i in (0,2,4):z=F.elu(F.linear(z,state[f"actor.{i}.weight"],state[f"actor.{i}.bias"]))
    return F.linear(z,state["actor.6.weight"],state["actor.6.bias"]).clamp(-20,20)

def stats(raw,targets,acc,phi,metrics,window_start):
    if len(raw)==0:return None
    raw=np.array(raw);targets=np.array(targets);acc=np.array(acc);phi=np.array(phi);metrics=np.array(metrics)
    return {"ticks":len(raw),"window_start_s":window_start,"action_negative_clip_fraction":np.mean(raw<=-1,axis=0).tolist(),
            "raw_action_mean":raw.mean(axis=0).tolist(),"raw_action_min":raw.min(axis=0).tolist(),"raw_action_max":raw.max(axis=0).tolist(),
            "leg_target_range_deg":np.rad2deg(np.ptp(targets[:,:4],axis=0)).tolist(),
            "leg_accel_rms_rad_s2":np.sqrt(np.mean(acc[:,:4]**2,axis=0)).tolist(),
            "phi0_delta_abs_mean_deg":float(np.rad2deg(np.abs(np.arctan2(np.sin(phi[:,0]-phi[:,1]),np.cos(phi[:,0]-phi[:,1])))).mean()),
            "vx_mae_mps":float(np.abs(metrics[:,0]).mean()),"yaw_rate_mae_rad_s":float(np.abs(metrics[:,1]).mean()),"max_tilt_deg":float(np.rad2deg(metrics[:,2]).max())}

for item in inputs:
    cp=Path(item["path"]);assert sha256_file(cp)==item["sha256"]
    p,s,info=loaded(cp)
    local=info["completed_iterations"];label=item["stage"]+"_"+str(local)
    with torch.inference_mode():
        output=p(batch).numpy()
        independent_output=independent(s,batch).numpy()
    error=float(abs(output-independent_output).max())
    assert error<1e-5,(label,error)
    known_export=None
    if item["stage"]=="fresh1000" and local==1000:
        known_export=PROJECT/"artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/exports/run-03"
    elif item["stage"]=="baseline200" and local==200:
        known_export=PROJECT/"artifacts/phase2_dreamwaq/leg-reversal-trial-20261011-v1/baseline/export"
    elif item["stage"]=="budget85_200" and local==200:
        known_export=BASE/"85/export"
    elif item["stage"]=="budget_search_baseline200":
        known_export=BASE/"baseline/export"
    export_error=None
    if known_export is not None:
        km=json.loads((known_export/"policy_manifest.json").read_text())
        assert km["source_checkpoint_sha256"]==item["sha256"]
        assert sha256_file(known_export/"actor.ts")==km["actor_sha256"]
        old=torch.jit.load(str(known_export/"actor.ts"),map_location="cpu").eval()
        with torch.inference_mode():expected=old(batch).numpy()
        export_error=float(abs(output-expected).max())
        assert export_error==0,(label,export_error)
    observation=runtime.reset(command)
    actions,targets,accelerations,phis,metrics,clocks=[],[],[],[],[],[]
    failure=None
    for tick in range(500):
        with torch.inference_mode():action=p(torch.from_numpy(observation).unsqueeze(0)).numpy()[0]
        before=runtime.model_map.joint_velocity_control(runtime.data).copy()
        target=runtime.controller.prepare(action).leg_position_target_mujoco.copy()
        result=runtime.step(action,command)
        after=runtime.model_map.joint_velocity_control(runtime.data).copy()
        actions.append(action.copy());targets.append(target);accelerations.append((after-before)/contract.control_dt_s)
        phis.append([result.metrics["phi0_left_rad"],result.metrics["phi0_right_rad"]])
        metrics.append([result.metrics["vx_mps"],result.metrics["yaw_rate_rad_s"],result.metrics["tilt_rad"],result.metrics["base_height_m"]])
        clocks.append(float(runtime.data.time))
        failure=_failure_reason(runtime,result.metrics)
        if failure:break
        observation=result.observation
    actions=np.array(actions);targets=np.array(targets);accelerations=np.array(accelerations);phis=np.array(phis);metrics=np.array(metrics);clocks=np.array(clocks)
    steady=np.arange(len(actions))>=100
    key=label
    for name,value in dict(action=actions,target=targets,acceleration=accelerations,phi=phis,metrics=metrics,time_after=clocks,common_action=output).items():arrays[key+"__"+name]=value
    existing_error=None
    if item["stage"] in ("budget85_200","budget_search_baseline200") and local==200:
        prior_arm="85" if item["stage"]=="budget85_200" else "baseline"
        d=np.load(BASE/"saturation-reward-audit-20261012-v1/trajectories.npz")
        expected=d[prior_arm+"_nominal_stand__action_raw"]
        existing_error=float(abs(expected-actions).max())
        assert existing_error==0
    first_full_tail=[]
    for j in range(2,4):
        unsaturated=np.flatnonzero(actions[:,j]>-1)
        start=int(unsaturated[-1]+1) if len(unsaturated) else 0
        first_full_tail.append(float(start*.02) if start<len(actions) else None)
    result={"label":label,"stage":item["stage"],"completed_iterations_local":local,"checkpoint":str(cp),"checkpoint_sha256":item["sha256"],
            "seed":info["seed"],"saved_at":info["saved_at"],"has_leg_acceleration_gain_state":"leg_acceleration_gain_state" in info,"has_leg_reversal_gain_state":"leg_reversal_gain_state" in info,
            "reset_raw_action":output[0].tolist(),"same_85_state_raw_action_mean":output[1:].mean(axis=0).tolist(),
            "same_85_state_negative_clip_fraction":np.mean(output[1:]<=-1,axis=0).tolist(),"actor_final_bias":s["actor.6.bias"].tolist(),
            "effective_action_std":s["raw_std"].abs().clamp(1e-4,2).tolist(),
            "trajectory_ticks":len(actions),"failure_reason":failure,"survived_full_10s":len(actions)==500 and failure is None,
            "right_negative_clip_tail_start_s":first_full_tail,"all_trajectory":stats(actions,targets,accelerations,phis,metrics,0),
            "steady_trajectory":stats(actions[steady],targets[steady],accelerations[steady],phis[steady],metrics[steady],2) if steady.any() else None,
            "weight_mapping_and_manual_graph_max_error":error,"existing_export_max_action_error":export_error,"existing_final_trace_max_action_error":existing_error}
    results.append(result)
    print(json.dumps({k:result[k] for k in ("label","reset_raw_action","trajectory_ticks","failure_reason","right_negative_clip_tail_start_s")}),flush=True)

np.savez_compressed(OUT/"checkpoint-trajectories.npz",**arrays)
assert all(sha256_file(Path(p))==h for p,h in protected.items())
report={"schema":"RightLegSaturationLineageV1","checkpoints":results,"fixed_comparison_histories_source":"MuJoCo recorded 85% interactive frames 100,110,..990; original reset history for comparison batch row zero",
        "common_state_test_is_training_replay":False,"trajectory_command":[0,0,.2],"trajectory_runtime":"Original PhysicsV5; independent unchanged reset for each checkpoint, 10s or original failure rule",
        "production_files_unchanged":True,"training_started":False,"isaac_started":False,"subagents_called":False,"elapsed_wall_s":time.perf_counter()-started}
(OUT/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
print("done",len(results),"control_frames",sum(x["trajectory_ticks"] for x in results),"wall_s",time.perf_counter()-started)
