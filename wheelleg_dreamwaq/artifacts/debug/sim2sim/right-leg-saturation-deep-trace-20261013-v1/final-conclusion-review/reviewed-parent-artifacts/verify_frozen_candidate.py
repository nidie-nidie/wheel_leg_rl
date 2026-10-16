"""Independently verify frozen diagnostic traces; no rollout or training."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[3]
sys.path.insert(0,str(PROJECT/"sim2sim/mujoco"))
from wheelleg_mujoco.contract import load_policy_contract
from wheelleg_mujoco.command_practice import summarize_practice
BASE=OUT.parent/"leg-acceleration-budget-search-20261012-v1"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
contract,manifest,_=load_policy_contract(BASE/"85/export/policy_manifest.json",BASE/"85/export/actor.ts",PROJECT/"sim2sim/mujoco/model_manifest.json")
spec=json.loads((OUT/"candidate-spec.json").read_text())
result=json.loads((OUT/"frozen-candidate-results.json").read_text())
assert sha(OUT/"candidate-spec.json")==result["specification_sha256"]
assert sha(BASE/"85/export/actor.ts")==spec["source_actor_sha256"]
assert sha(PROJECT/"sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml")==spec["model_xml_sha256"]
protected={}
for p in (OUT/"protected-before.json",OUT.parent/"right-leg-saturation-lineage-20261013-v1/protected-before.json"):
    for name,digest in json.loads(p.read_text()).items():
        assert name not in protected or protected[name]==digest
        protected[name]=digest
for name,digest in protected.items():assert sha(name)==digest,name
policy=torch.jit.load(str(BASE/"85/export/actor.ts")).eval()
torch.set_num_threads(1)
data=np.load(OUT/"frozen-candidate-trajectories.npz")
dt=.02
alpha=np.r_[np.repeat(1-np.exp(-2*np.pi*6*dt),4),np.repeat(1-np.exp(-2*np.pi*2*dt),2)]
cases=[]
def close(a,b,tol=1e-11):np.testing.assert_allclose(a,b,rtol=0,atol=tol)
def movement(d,start=100):
    before=d["com_before"][start:];after=d["com_after"][start:]
    pos=np.vstack((before[0],after))
    excursion=np.linalg.norm(pos-pos[0],axis=1)
    return {"duration_s":len(after)*dt,"net_displacement_m":float(np.linalg.norm(pos[-1]-pos[0])),
            "path_length_m":float(np.linalg.norm(after-before,axis=1).sum()),
            "max_excursion_from_settled_start_m":float(excursion.max()),
            "p95_excursion_from_settled_start_m":float(np.percentile(excursion,95)),
            "xy_range_m":np.ptp(pos,axis=0).tolist()}
for r in result["results"]:
    key=r["key"];d={n.split(key+"__")[1]:data[n] for n in data.files if n.startswith(key+"__")}
    n=len(d["raw"]);assert n==r["ticks"] and r["failure_reason"] is None
    for v in d.values():assert np.isfinite(v).all() and len(v)==n
    close(d["time"],np.arange(1,n+1)*dt,1e-10)
    h=d["history"].reshape(n,5,25)
    # Older frames retain real preceding command and action; no masked sensor.
    np.testing.assert_array_equal(h[1:,:4],h[:-1,1:])
    prev=np.vstack((np.zeros(6),d["applied"][:-1]))
    np.testing.assert_array_equal(h[:,-1,19:25],prev.astype(np.float32))
    close(h[1:,-1,13:19],np.clip(d["v"][:-1]*contract.normalization["joint_velocity_scale"],-5,5).astype(np.float32),0)
    close(h[1:,-1,9:13],np.clip((d["q"][:-1,:4]-contract.q_nominal)*contract.normalization["leg_position_scale"],-1.5,1.5).astype(np.float32),0)
    close(d["applied"],prev+alpha*(np.clip(d["raw"],-1,1).astype(np.float64)-prev))
    close(d["target"],contract.q_nominal+.35*d["applied"][:,:4])
    close(d["acc"][1:],np.diff(d["v"],axis=0)/dt)
    close(d["sensor_before"][1:,0],.0625*d["v"][:-1,4:6].mean(axis=1))
    close(h[:,-1,2],(d["sensor_before"][:,1]*.25).astype(np.float32),0)
    estimated=np.zeros(2);integral=np.zeros(2);last=d["user_command"][0].copy();stable=0
    sensor_alpha=1-np.exp(-2*np.pi*r["observer_hz"]*dt)
    reconstructed=[];active_count=0
    for t in range(n):
        u=d["user_command"][t]
        estimated+=sensor_alpha*(d["sensor_before"][t]-estimated)
        stable=stable+1 if np.array_equal(last,u) else 0
        last=u.copy()
        active=not r["zero_only"] or (np.all(u[:2]==0) and stable*dt>=r["settled_delay_s"])
        limits=np.repeat(.06 if r["zero_only"] else .3,2)
        if active:integral=np.clip(integral+np.array(r["ki"])*(u[:2]-estimated)*dt,-limits,limits);active_count+=1
        correction=np.clip(np.array(r["kp"])*(u[:2]-estimated)+integral,-limits,limits) if active else np.zeros(2)
        corrected=u.copy();corrected[:2]+=correction
        corrected[0]=np.clip(corrected[0],-1.5,1.5);corrected[1]=np.clip(corrected[1],-1,1)
        close(d["filtered_sensor"][t],estimated)
        close(d["correction"][t],correction)
        close(d["policy_command"][t],corrected)
        reconstructed.append(corrected)
    cmd=np.array(reconstructed)
    normalized=np.c_[cmd[:,0]/1.5,cmd[:,1],(cmd[:,2]-.2)/.04]
    np.testing.assert_array_equal(h[:,-1,6:9],np.clip(normalized,-1,1).astype(np.float32))
    raw=[];encoded=[]
    with torch.inference_mode():
        for start in range(0,n,256):
            x=torch.from_numpy(d["history"][start:start+256].copy())
            raw.append(policy(x).numpy());encoded.append(policy.encoder(x).numpy()[:,:19])
    raw=np.concatenate(raw);enc=np.concatenate(encoded)
    close(raw,d["raw"],3e-5);close(enc,d["encoded"],3e-5)
    metrics=d["metrics"][100:];target=d["user_command"][100:]
    vx=float(np.abs(metrics[:,0]-target[:,0]).mean());yaw=float(np.abs(metrics[:,1]-target[:,1]).mean())
    close(vx,r["steady"]["vx_mae_mps"]);close(yaw,r["steady"]["yaw_mae_rad_s"])
    accel=np.sqrt(np.mean(d["acc"][100:,:4]**2,axis=0))
    close(accel,r["steady"]["leg_accel_rms_rad_s2"])
    move=movement(d)
    close(move["net_displacement_m"],r["steady"]["net_displacement_m"])
    close(move["path_length_m"],r["steady"]["path_length_m"])
    practice=None
    if "practice_summary" in r:
        rows=[{"tick":t,"vx_mps":d["metrics"][t,0],"yaw_rate_rad_s":d["metrics"][t,1],
               "com_xy_before":d["com_before"][t].tolist(),"com_xy_after":d["com_after"][t].tolist()} for t in range(n)]
        practice=summarize_practice(r["scenario"],d["user_command"][0],rows,None)
        assert practice==r["practice_summary"] and practice["performance_accepted"]
    if "formal_summary" in r:
        fs=r["formal_summary"];assert fs["completed"] and fs["ticks"]==500
        close(fs["vx_mae"],np.abs(d["metrics"][:,0]-d["user_command"][:,0]).mean())
        close(fs["height_mae"],np.abs(d["metrics"][:,3]-d["user_command"][:,2]).mean())
        close(fs["max_tilt_rad"],d["metrics"][:,2].max())
        close(fs["max_loop_closure_error_m"],d["metrics"][:,4].max())
        close(fs["effort_saturation_fraction"],d["metrics"][:,5].mean())
    cases.append({"key":key,"ticks":n,"vx_mae_mps":vx,"yaw_mae_rad_s":yaw,"leg_accel_rms_20ms":accel.tolist(),
                  "movement":move,"feedback_active_steps":active_count,
                  "raw_reinference_max_error":float(np.abs(raw-d["raw"]).max()),
                  "encoded_reinference_max_error":float(np.abs(enc-d["encoded"]).max()),
                  "practice_recalculated":practice})
physics=np.load(OUT/"final-physics-trajectories.npz")["physics"]
oldp=np.load(OUT/"physics-bandwidth-trajectories.npz")["unmodified__physics"]
for a in (physics,oldp):
    assert a.shape==(10000,25)
    close(a[1:,13:19],np.diff(a[:,7:13],axis=0)/.001,0)
close(physics[19::20,1:7],data["candidate_v1__nominal_stand__q"],0)
close(physics[19::20,7:13],data["candidate_v1__nominal_stand__v"],0)
rms=lambda a:np.sqrt(np.mean(a[a[:,0]>=2,13:19]**2,axis=0))
final_rms=rms(physics);old_rms=rms(oldp)
close(final_rms,json.loads((OUT/"final-physics-results.json").read_text())["actual_accel_rms_rad_s2"])
left_reduction=1-np.linalg.norm(final_rms[:2])/np.linalg.norm(old_rms[:2])
long_comparisons=[]
for f,k,label in (("expanded-trajectories.npz","unmodified__long_stand","original"),
                  ("candidate-validation-trajectories.npz","leg6_wheel2__long_stand","filter_only"),
                  ("frozen-candidate-trajectories.npz","candidate_v1__stand60","final_candidate")):
    with np.load(OUT/f) as trace:
        d={x:trace[k+"__"+x] for x in ("com_before","com_after")}
        long_comparisons.append({"variant":label,**movement(d)})
inputs=["candidate-spec.json","frozen-candidate-results.json","frozen-candidate-trajectories.npz",
        "final-physics-results.json","final-physics-trajectories.npz","physics-bandwidth-results.json",
        "physics-bandwidth-trajectories.npz","expanded-trajectories.npz","candidate-validation-trajectories.npz",
        "slow_feedback.py","frozen_candidate.py","final_physics.py"]
summary={"all_checks_passed":True,"protected_files_unchanged":len(protected),"case_count":len(cases),
         "control_frames":sum(x["ticks"] for x in cases),"all_recorded_cases_completed_without_failure":True,
         "four_practices_recalculated_and_passed":True,
         "formal_metric_recalculation_scope":["vx_mae","height_mae","max_tilt","max_loop_error","effort_saturation"],
         "policy_reinference_all_recorded_frames":True,"history_filter_feedback_timing_recalculated":True,
         "left_actual_accel_1ms_old_rad_s2":old_rms[:2].tolist(),"left_actual_accel_1ms_candidate_rad_s2":final_rms[:2].tolist(),
         "left_pair_accel_rms_reduction_fraction":float(left_reduction),
         "long_standing_comparisons":long_comparisons,
         "input_sha256":{x:sha(OUT/x) for x in inputs},"cases":cases,
         "not_position_hold":True,"training_started":False,"production_modified":False}
(OUT/"verification.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k not in ("cases","input_sha256")},indent=2))
