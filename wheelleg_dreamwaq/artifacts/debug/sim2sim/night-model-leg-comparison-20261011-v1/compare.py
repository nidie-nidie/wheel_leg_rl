"""Diagnostic only; preserve historical identities, use unchanged current physics."""
from pathlib import Path
import csv,json,subprocess,sys
import numpy as np
import torch
ROOT=Path(r"E:\wheel_leg_rl-main"); PROJECT=ROOT/"wheelleg_dreamwaq"; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/"scripts"))
from evaluate_mujoco import DEFAULT_MODEL,DEFAULT_MODEL_MANIFEST,_failure_reason
from wheelleg_mujoco.contract import load_policy_contract,sha256_file
from wheelleg_mujoco.model_semantics import stable_hash
from wheelleg_mujoco.observation import collect_kinematic_state
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
EXPORTS={
"night":PROJECT/"artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2/exports/run-04",
"tracking":PROJECT/"artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/exports/run-03",
"smooth002":PROJECT/"artifacts/phase2_dreamwaq/action-smooth2-run03-fresh1000-20261011-161144/export"}
HIST=PROJECT/"artifacts/debug/sim2sim/standing-trace-20261010-v1/mujoco/nominal_stand.csv"
DT=.02
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def changes():return subprocess.check_output(["git","diff","--name-only"],cwd=ROOT,text=True).splitlines()
def rows(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def stats(data,label,scenario,historical=False):
    ix=np.array([i for i,r in enumerate(data[:-1]) if 2.04-1e-9<=float(r["time_s"])<9.98-1e-9])
    if not len(ix):return dict(label=label,scenario=scenario,samples=0)
    vector=lambda key:np.array([[float(r[key.format(j=j)]) for j in range(6)] for r in data])
    v=vector("joint_velocity_{j}_rad_s");q=vector("joint_position_{j}_rad");a=vector("clipped_action_{j}");raw=vector("raw_action_{j}")
    alpha=(v[ix+1]-v[ix])/DT
    if not historical:np.testing.assert_allclose(vector("joint_velocity_after_{j}_rad_s")[ix],v[ix+1],rtol=0,atol=1e-12)
    d2=a[ix]-2*a[ix-1]+a[ix-2]
    vx=np.array([float(data[i]["actual_vx_mps"]) for i in ix]);cmd=np.array([float(data[i]["command_vx_mps"]) for i in ix])
    peaks=[]
    for j in range(4):
        signal=v[ix,j]-v[ix,j].mean();power=np.abs(np.fft.rfft(signal*np.hanning(len(ix))))**2;power[0]=0
        peaks.append(float(np.fft.rfftfreq(len(ix),DT)[np.argmax(power)]) if np.std(signal)>1e-12 else None)
    return dict(label=label,scenario=scenario,samples=len(ix),time_start_s=float(data[ix[0]]["time_s"]),time_end_s=float(data[ix[-1]]["time_s"]),
        vx_mean_mps=float(vx.mean()),vx_mae_mps=float(np.abs(vx-cmd).mean()),
        leg_velocity_rms_rad_s=np.sqrt(np.mean(v[ix,:4]**2,axis=0)).tolist(),
        leg_acceleration_rms_rad_s2=np.sqrt(np.mean(alpha[:,:4]**2,axis=0)).tolist(),
        leg_position_span_rad=np.ptp(q[ix,:4],axis=0).tolist(),
        leg_clipped_action_span=np.ptp(a[ix,:4],axis=0).tolist(),leg_raw_action_span=np.ptp(raw[ix,:4],axis=0).tolist(),
        leg_at_minus_one_fraction=np.mean(a[ix,:4]==-1,axis=0).tolist(),
        leg_action_smooth_mean=float(np.mean(np.sum(d2[:,:4]**2,axis=1))),six_action_smooth_mean=float(np.mean(np.sum(d2**2,axis=1))),
        leg_velocity_peak_hz=peaks)
def main():
    assert not (OUT/"results.json").exists(),"Use a new output folder instead of overwriting evidence"
    torch.set_num_threads(1)
    paths=list(read(PROJECT/"artifacts/debug/sim2sim/leg-jitter-indicators-20261011-182657/input-identity.json")["protected_files"])
    for folder in EXPORTS.values():paths.extend(str(p) for p in folder.iterdir() if p.is_file())
    paths.append(str(HIST));before={p:sha256_file(Path(p)) for p in sorted(set(paths))};tracked=changes()
    save("input-identity.json",dict(protected_files=before,tracked_changes_before=tracked))
    contract,current,mm=load_policy_contract(EXPORTS["tracking"]/"policy_manifest.json",EXPORTS["tracking"]/"actor.ts",DEFAULT_MODEL_MANIFEST)
    assert sha256_file(DEFAULT_MODEL)==mm["model_xml"]["sha256"]
    manifests={name:read(folder/"policy_manifest.json") for name,folder in EXPORTS.items()};checks={}
    common=["network","observation","normalization","q_nominal","action","control","frames","timing","initial_state","history_adapter_version","asset_bundle_hash"]
    for name,m in manifests.items():
        assert m["manifest_hash"]==stable_hash({k:v for k,v in m.items() if k!="manifest_hash"})
        for file,key in [("actor.ts","actor_sha256"),("golden_vectors.pt","golden_vectors_sha256")]:assert sha256_file(EXPORTS[name]/file)==m[key]
        assert sha256_file(Path(m["source_checkpoint"]))==m["source_checkpoint_sha256"]
        for field in common:assert m[field]==current[field],(name,field)
        diffs=[]
        for group in current["actuators"]:
            for field,value in current["actuators"][group].items():
                old=m["actuators"][group][field]
                if old!=value:
                    assert name=="night" and field=="velocity_limit_sim",(name,group,field)
                    diffs.append(dict(group=group,field=field,source=old,current=value))
        checks[name]=dict(matched_interface_fields=common,source_physics_schema=m["schemas"]["physics"],source_actuator_differences=diffs,
            source_checkpoint=m["source_checkpoint"],source_checkpoint_sha256=m["source_checkpoint_sha256"],actor_sha256=m["actor_sha256"])
    historical=rows(HIST);policy=torch.jit.load(str(EXPORTS["night"]/"actor.ts"),map_location="cpu").eval()
    saved=np.load(HIST.parent/"policy_inputs.npz");key=next(k for k in saved.files if saved[k].shape==(500,125))
    with torch.inference_mode():replay=policy(torch.from_numpy(saved[key].astype(np.float32))).numpy()
    expected=np.array([[float(r[f"raw_action_{j}"]) for j in range(6)] for r in historical])
    replay_error=float(np.max(np.abs(replay-expected)));assert replay_error<1e-5
    cases=[stats(historical,"night_original_PhysicsV4","parking",True)];traces={}
    for name,m in manifests.items():
        actor=torch.jit.load(str(EXPORTS[name]/"actor.ts"),map_location="cpu").eval()
        if name!="night":load_policy_contract(EXPORTS[name]/"policy_manifest.json",EXPORTS[name]/"actor.ts",DEFAULT_MODEL_MANIFEST)
        for scenario,speed in [("parking",0.),("forward",1.),("reverse",-1.)]:
            runtime=WheelLegMujocoRuntime(DEFAULT_MODEL,contract,mm);command=np.array([speed,0.,.2]);obs=runtime.reset(command);data=[];failure=None
            for tick in range(500):
                state=collect_kinematic_state(runtime.model,runtime.data,runtime.model_map,contract)
                r=dict(tick=tick,time_s=float(runtime.data.time),command_vx_mps=speed,actual_vx_mps=float(state.com_linear_velocity_control[0]))
                for j,(q,v) in enumerate(zip(runtime.model_map.joint_position_mujoco(runtime.data),runtime.model_map.joint_velocity_mujoco(runtime.data))):
                    r[f"joint_position_{j}_rad"]=float(q);r[f"joint_velocity_{j}_rad_s"]=float(v)
                with torch.inference_mode():action=actor(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
                assert np.isfinite(action).all()
                result=runtime.step(action,command);assert result.physics_steps==20
                for j,(raw,clipped,v) in enumerate(zip(action,result.clipped_action,runtime.model_map.joint_velocity_mujoco(runtime.data))):
                    r[f"raw_action_{j}"]=float(raw);r[f"clipped_action_{j}"]=float(clipped);r[f"joint_velocity_after_{j}_rad_s"]=float(v)
                r.update(result.metrics);data.append(r);failure=_failure_reason(runtime,result.metrics)
                if failure:break
                obs=result.observation
            path=OUT/f"{name}_{scenario}.csv"
            with path.open("w",encoding="utf-8",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
            summary=stats(data,name+"_current_PhysicsV5",scenario)
            summary.update(completed_frames=len(data),failure_reason=failure,trace=str(path),trace_sha256=sha256_file(path))
            if name!="night":
                ev=(PROJECT/"artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/mujoco_evaluation/run-03" if name=="tracking" else PROJECT/"artifacts/phase2_dreamwaq/action-smooth2-run03-fresh1000-20261011-161144/mujoco_evaluation")
                original=rows(ev/("nominal_stand.csv" if scenario=="parking" else scenario+".csv"));assert len(original)==len(data)
                err=max(abs(float(r[f"raw_action_{j}"])-float(o[f"raw_action_{j}"])) for r,o in zip(data,original) for j in range(6))
                assert err<1e-8,(name,scenario,err)
                summary["action_replay_error_vs_original_evaluation"]=err
            cases.append(summary);traces[(name,scenario)]=data;print(json.dumps(summary),flush=True)
    save("results.json",dict(diagnostic_only=True,current_physics_schema=current["schemas"]["physics"],current_model_sha256=sha256_file(DEFAULT_MODEL),
        interface_checks=checks,historical_actor_replay_max_error=replay_error,cases=cases,
        limits=["Old policy/current physics is an explicit diagnostic; original policy metadata is retained.",
        "No additional torque, collision, reward, filter, or training was introduced.",
        "Quiet legs do not establish healthy control or a ROGER threshold. This is not a causal intervention on leg jitter."]))
    assert before=={p:sha256_file(Path(p)) for p in before};assert changes()==tracked
    save("verification.json",dict(protected_file_count=len(before),all_hashes_unchanged=True,tracked_changes_unchanged=True,
        control_dt_s=DT,physics_steps_per_action=20,training_started=False))
if __name__=="__main__":main()
