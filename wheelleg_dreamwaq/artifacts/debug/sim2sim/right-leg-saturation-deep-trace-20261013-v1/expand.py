"""Expanded diagnostic contrasts; unchanged production runtime and policies."""
from pathlib import Path
import json,sys,importlib.util
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("deep_trace",(OUT/"trace_and_screen.py"))
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
sys.path.insert(0,str(t.PROJECT/"scripts"))
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS,summarize_scenario
from wheelleg_mujoco.command_practice import command_at_tick,validate_command_practice,summarize_practice
from evaluate_command_practice_mujoco import SCENARIOS
from evaluate_mujoco import _failure_reason
from wheelleg_mujoco.observation import collect_kinematic_state
perm=np.array([2,3,0,1,5,4])
def mirror_history(history):
    x=np.array(history,copy=True).reshape(-1,5,25)
    x[:,:,0]*=-1;x[:,:,2]*=-1
    x[:,:,4]*=-1;x[:,:,7]*=-1
    x[:,:,9:13]=x[:,:,9:13][:,:,[2,3,0,1]]
    x[:,:,13:19]=x[:,:,13:19][:,:,perm]
    x[:,:,19:25]=x[:,:,19:25][:,:,perm]
    return x.reshape(-1,125)
probe=np.arange(2*125,dtype=np.float32).reshape(2,125)
np.testing.assert_array_equal(mirror_history(mirror_history(probe)),probe)
payload=validate_command_practice(t.manifest)
results=[];arrays={}
def run(policy,variant,scenario,command_fn,ticks,filter_hz=(0.,0.),mirror=None,velocity_mode=None,push=None):
    cache=OUT/"expanded-cases";cache.mkdir(exist_ok=True)
    cache_key=variant+"__"+scenario
    import hashlib
    policy_digest=hashlib.sha256()
    for name,value in sorted(policy.state_dict().items()):
        policy_digest.update(name.encode());policy_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    commands=np.array([command_fn(i) for i in range(ticks+1)],dtype=np.float64)
    identity={"schema":"ExpandedDiagnosticIdentityV2","policy_weights_sha256":policy_digest.hexdigest().upper(),
        "commands_sha256":hashlib.sha256(commands.tobytes()).hexdigest().upper(),"ticks":ticks,"filter_hz":list(filter_hz),
        "mirror":mirror,"velocity_mode":velocity_mode,"push":push,"source_sha256":t.sha256_file(Path(__file__)),
        "model_xml_sha256":t.sha256_file(t.runtime.model_path),"actor_contract_hash":t.manifest["dreamwaq_export_contract_hash"]}
    if (cache/(cache_key+".json")).exists():
        result=json.loads((cache/(cache_key+".json")).read_text())
        if result.get("input_identity")!=identity:
            raise ValueError("Diagnostic cached case identity differs or is legacy; use a new unique variant without overwriting prior evidence")
        with np.load(cache/(cache_key+".npz")) as saved:
            for name in saved.files:arrays[cache_key+"__"+name]=saved[name]
        results.append(result);print("cached",cache_key,flush=True);return result
    rt=t.runtime;dt=t.dt
    observation=rt.reset(np.array(command_fn(0)))
    rotation=rt.data.xmat[rt.model_map.base_body_id].reshape(3,3).copy()
    force_world=np.zeros(3) if push is None else rotation@t.contract.r_control_from_mujoco.T@np.array(push["force_control_n"])
    filtered=np.zeros(6)
    f=np.r_[np.repeat(filter_hz[0],4),np.repeat(filter_hz[1],2)]
    alpha=1-np.exp(-2*np.pi*f*dt);alpha[f==0]=1
    d={k:[] for k in ("history","raw","original_raw","applied","target","q","v","acc","metrics","command","phi","encoded","true_velocity_before","external_force_world","time","com_before","com_after")}
    formal_rows=[];practice_rows=[];failure=None
    for tick in range(ticks):
        current=np.array(command_fn(tick),dtype=np.float64)
        nxt=np.array(command_fn(tick+1),dtype=np.float64)
        state_before=collect_kinematic_state(rt.model,rt.data,rt.model_map,t.contract)
        with torch.inference_mode():
            x=torch.from_numpy(observation).unsqueeze(0)
            original=policy(x).numpy()[0]
            encoded=policy.encoder(x).numpy()[0,:19]
            if mirror is None:desired=original.copy()
            else:
                mirrored=policy(torch.from_numpy(mirror_history(observation))).numpy()[0,perm]
                if mirror=="clipped":desired=.5*(np.clip(original,-1,1)+np.clip(mirrored,-1,1))
                elif mirror=="raw":desired=.5*(original+mirrored)
                else:raise ValueError(mirror)
        if velocity_mode is not None:
            adjusted=encoded.copy()
            true_scaled=state_before.com_linear_velocity_control*t.contract.normalization["root_linear_velocity_scale"]
            if velocity_mode=="true3":adjusted[:3]=true_scaled
            elif velocity_mode=="true_vx":adjusted[0]=true_scaled[0]
            else:raise ValueError(velocity_mode)
            with torch.inference_mode():
                desired=policy.actor(torch.cat((x[:,-25:],torch.from_numpy(adjusted).unsqueeze(0)),dim=1)).clamp(-20,20).numpy()[0]
        filtered=filtered+alpha*(np.clip(desired,-1,1).astype(np.float64)-filtered)
        before=rt.model_map.joint_velocity_control(rt.data).copy()
        before_xy=rt.data.xipos[rt.model_map.base_body_id,:2].copy()
        targets=rt.controller.prepare(filtered)
        target=np.r_[targets.leg_position_target_mujoco,targets.wheel_velocity_target_mujoco*t.contract.wheel_signs]
        rt.data.xfrc_applied[:]=0
        if push is not None and push["start_tick"]<=tick<push["end_tick"]:
            rt.data.xfrc_applied[rt.model_map.base_body_id,:3]=force_world
        applied_force=rt.data.xfrc_applied[rt.model_map.base_body_id,:3].copy()
        result=rt.step(filtered,nxt)
        state=collect_kinematic_state(rt.model,rt.data,rt.model_map,t.contract)
        m=result.metrics
        after_xy=rt.data.xipos[rt.model_map.base_body_id,:2].copy()
        values=dict(history=observation.copy(),raw=desired.copy(),original_raw=original.copy(),applied=filtered.copy(),
            target=target.copy(),q=state.joint_position_control.copy(),v=state.joint_velocity_control.copy(),
            acc=(state.joint_velocity_control-before)/dt,
            metrics=[m["vx_mps"],m["yaw_rate_rad_s"],m["tilt_rad"],m["base_height_m"],m["max_loop_closure_error_m"],m["effort_saturation_fraction"]],
            command=current,phi=[m["phi0_left_rad"],m["phi0_right_rad"]],external_force_world=applied_force,encoded=encoded,true_velocity_before=state_before.com_linear_velocity_control.copy(),time=float(rt.data.time),com_before=before_xy,com_after=after_xy)
        for name,val in values.items():d[name].append(val)
        failure=_failure_reason(rt,m)
        if failure:break
        formal_rows.append(m.copy())
        practice_rows.append({"tick":tick,**m,"com_xy_before":before_xy.tolist(),"com_xy_after":after_xy.tolist()})
        observation=result.observation
    rt.data.xfrc_applied[:]=0
    key=variant+"__"+scenario
    for name,value in d.items():arrays[key+"__"+name]=np.array(value)
    r={"input_identity":identity,"key":key,"variant":variant,"scenario":scenario,"filter_hz_legs_wheels":list(filter_hz),"mirror":mirror,"velocity_mode":velocity_mode,"external_push":push,"ticks":len(d["raw"]),"expected_ticks":ticks,
        "failure_reason":failure,"steady":t.statistics(d),"all":t.statistics(d,0)}
    if scenario in {n for n,c in FORMAL_SCENARIOS}:
        r["formal_summary"]=summarize_scenario(scenario,tuple(command_fn(0)),formal_rows,expected_ticks=ticks,failure_reason=failure)
    elif scenario in {n for n,c in SCENARIOS}:
        r["practice_summary"]=summarize_practice(scenario,np.array(command_fn(0)),practice_rows,failure)
    results.append(r)
    np.savez_compressed(cache/(key+".npz"),**{name:np.asarray(value) for name,value in d.items()})
    (cache/(key+".json")).write_text(json.dumps(r,indent=2),encoding="utf-8")
    s=r["steady"]
    print(json.dumps({"key":key,"ticks":r["ticks"],"failure":failure,"vx":None if s is None else s["vx_mae_mps"],
        "yaw":None if s is None else s["yaw_mae_rad_s"],"acc":None if s is None else s["leg_accel_rms_rad_s2"],
        "practice_accepted":r.get("practice_summary",{}).get("performance_accepted")}),flush=True)
    return r
if __name__ == "__main__":
    policy=torch.jit.load(str(t.template)).eval()
    # All eight scenarios: bandwidth robustness and independent mirror-policy alternatives.
    variants={"unmodified":((0.,0.),None),"both2":((2.,2.),None),"both3":((3.,3.),None),"both4":((4.,4.),None),
        "leg3_wheel8":((3.,8.),None),"mirror_clipped":((0.,0.),"clipped"),"mirror_clipped_both3":((3.,3.),"clipped"),"mirror_raw_both3":((3.,3.),"raw")}
    for variant,(hz,mirror) in variants.items():
        for name,command in FORMAL_SCENARIOS:
            run(policy,variant,name,lambda tick,c=command:c,500,hz,mirror)
    # Stop/reverse loops, preserving original command update before next observation.
    for variant in ("unmodified","both2","both3","both4","leg3_wheel8","mirror_clipped_both3"):
        hz,mirror=variants[variant]
        for name,initial in SCENARIOS:
            run(policy,variant,name,lambda tick,c=np.array(initial):command_at_tick(c,tick,payload),500,hz,mirror)
    # Two sustained tests at the best initially supported 3Hz branch and unchanged comparator.
    for variant in ("unmodified","both3"):
        hz,mirror=variants[variant]
        run(policy,variant,"long_stand",lambda tick:[0,0,.2],3000,hz,mirror)
        sequence=np.array([[0,0,.2],[1,0,.2],[0,0,.2],[-1,0,.2],[0,0,.2],[.8,.5,.2],[0,0,.2],[0,-.6,.2],[0,0,.2],[0,0,.17],[0,0,.23],[0,0,.2]])
        run(policy,variant,"long_mixed",lambda tick:sequence[min(tick//250,len(sequence)-1)],3000,hz,mirror)
    # Is the filter effect dependent on the adaptive-penalty checkpoint?
    for source in ("baseline","original_run03"):
        export=t.BASE/"baseline/export" if source=="baseline" else Path(t.suite["runs"][2]["export_dir"])
        source_contract,source_manifest,source_mm=t.load_policy_contract(export/"policy_manifest.json",export/"actor.ts",t.PROJECT/"sim2sim/mujoco/model_manifest.json")
        from dataclasses import fields
        for field in fields(t.contract):
            left=getattr(source_contract,field.name);right=getattr(t.contract,field.name)
            if isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
            else:assert left==right,field.name
        source_policy=torch.jit.load(str(export/"actor.ts")).eval()
        for variant in ("unmodified","both3"):
            for name,c in (("nominal_stand",[0,0,.2]),("forward",[1,0,.2]),("reverse",[-1,0,.2])):
                run(source_policy,source+"_"+variant,name,lambda tick,c=c:c,500,variants[variant][0])
    np.savez_compressed(OUT/"expanded-trajectories.npz",**arrays)
    assert all(t.sha256_file(Path(p))==h for p,h in t.protected.items())
    (OUT/"expanded-results.json").write_text(json.dumps({"schema":"DeepSaturationExpandedV1","results":results,
        "mirror_transformation":"control sagittal plane y->-y: axial gyro x/z negate; polar gravity y negate; yaw command negate; leg/controlled-wheel positions,velocities,previous-actions swap sides; full encoder rerun",
        "mirror_is_production_fix":False,"processed_action_is_observed":True,"training_started":False,"production_files_unchanged":True,"protected_files_unchanged":True},indent=2),encoding="utf-8")
    print("DONE",len(results),sum(r["ticks"] for r in results),flush=True)
