"""Independent final evidence audit. No MuJoCo, Isaac, rollout, or training."""
from pathlib import Path
import hashlib,json,math
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest().upper()
def close(a,b,tol=1e-10):
    np.testing.assert_allclose(a,b,rtol=0,atol=tol)
res=json.loads((ROOT/"frozen-candidate-results.json").read_text())
ver=json.loads((ROOT/"verification.json").read_text())
cmp=json.loads((ROOT/"comparison-metrics.json").read_text())
spec=json.loads((ROOT/"candidate-spec.json").read_text())
inputs={n:sha(ROOT/n) for n in ["report.md","candidate-spec.json","slow_feedback.py","verify_frozen_candidate.py","verification.json","comparison-metrics.json","frozen-candidate-results.json","frozen-candidate-trajectories.npz","final-physics-trajectories.npz","physics-bandwidth-trajectories.npz"]}
assert sha(ROOT/"candidate-spec.json")==res["specification_sha256"]
for name,digest in ver["input_sha256"].items():assert sha(ROOT/name)==digest
protected={}
for p in [ROOT/"protected-before.json",ROOT.parent/"right-leg-saturation-lineage-20261013-v1/protected-before.json"]:
    for name,digest in json.loads(p.read_text()).items():
        assert name not in protected or protected[name]==digest
        protected[name]=digest
for name,digest in protected.items():assert sha(name)==digest
assert len(res["results"])==31
z=np.load(ROOT/"frozen-candidate-trajectories.npz")
dt=.02;alpha=np.r_[np.repeat(spec["filter_alpha_legs_wheels"][0],4),np.repeat(spec["filter_alpha_legs_wheels"][1],2)]
rows=[];practices=[];all_gate_discrepancies=[]
for r in res["results"]:
    key=r["key"];a={f[len(key)+2:]:z[f] for f in z.files if f.startswith(key+"__")}
    n=len(a["raw"])
    expected=3707 if r["scenario"]=="holdout_dynamic" else 3000 if r["scenario"]=="stand60" else 750 if r["push"] is not None else 500
    assert n==r["ticks"]==expected and r["failure_reason"] is None
    assert all(np.isfinite(v).all() and len(v)==n for v in a.values())
    close(a["time"],np.arange(1,n+1)*dt)
    h=a["history"].reshape(n,5,25)
    np.testing.assert_array_equal(h[1:,:4],h[:-1,1:])
    previous=np.r_[np.zeros((1,6)),a["applied"][:-1]]
    np.testing.assert_array_equal(h[:,-1,19:25],previous.astype(np.float32))
    close(a["applied"],previous+alpha*(np.clip(a["raw"],-1,1)-previous))
    close(a["sensor_before"][1:,0],.0625*np.mean(a["v"][:-1,4:6],axis=1))
    np.testing.assert_array_equal(h[:,-1,2],(a["sensor_before"][:,1]*.25).astype(np.float32))
    # Independent pre-decision elapsed clock distinguishes sample count from actual elapsed time.
    start=0;counter=0;integral=np.zeros(2);estimate=np.zeros(2);last=a["user_command"][0]
    active_ticks=[];discrepancies=[]
    for t,u in enumerate(a["user_command"]):
        changed=not np.array_equal(u,last)
        if changed:start=t;counter=0
        else:counter+=1
        last=u.copy()
        estimate=(1-(1-math.exp(-2*math.pi*r["observer_hz"]*dt)))*estimate+(1-math.exp(-2*math.pi*r["observer_hz"]*dt))*a["sensor_before"][t]
        active=(not r["zero_only"]) or (np.all(u[:2]==0) and counter*dt>=r["settled_delay_s"])
        true_elapsed_active=(not r["zero_only"]) or (np.all(u[:2]==0) and (t-start)*dt>=r["settled_delay_s"])
        if active!=true_elapsed_active:discrepancies.append({"tick":t,"physical_predecision_time_s":t*dt,"continuous_command_elapsed_s":(t-start)*dt})
        if active:
            integral=np.clip(integral+np.array(r["ki"])*(u[:2]-estimate)*dt,-(.06 if r["zero_only"] else .3),(.06 if r["zero_only"] else .3))
            active_ticks.append(t)
        correction=np.clip(np.array(r["kp"])*(u[:2]-estimate)+integral,-(.06 if r["zero_only"] else .3),(.06 if r["zero_only"] else .3)) if active else np.zeros(2)
        close(a["filtered_sensor"][t],estimate)
        close(a["correction"][t],correction)
        policy=u.copy();policy[:2]+=correction;policy[:2]=np.clip(policy[:2],[-1.5,-1],[1.5,1])
        close(a["policy_command"][t],policy)
        if not np.all(u[:2]==0):close(correction,0)
    # Original command defines every reported tracking error.
    m=a["metrics"][100:];u=a["user_command"][100:]
    vx=np.abs(m[:,0]-u[:,0]).mean();yaw=np.abs(m[:,1]-u[:,1]).mean()
    close(vx,r["steady"]["vx_mae_mps"]);close(yaw,r["steady"]["yaw_mae_rad_s"])
    b=a["com_before"][100:];after=a["com_after"][100:];positions=np.r_[b[:1],after]
    displacement=float(np.linalg.norm(positions[-1]-positions[0]))
    excursion=float(np.linalg.norm(positions-positions[0],axis=1).max())
    path=float(np.linalg.norm(after-b,axis=1).sum())
    close(displacement,r["steady"]["net_displacement_m"]);close(path,r["steady"]["path_length_m"])
    row={"key":key,"frames":n,"vx_mae_original_command":float(vx),"yaw_mae_original_command":float(yaw),"net_displacement_m":displacement,"max_excursion_m":excursion,"path_m":path,
        "first_active_tick":None if not active_ticks else active_ticks[0],"counter_vs_elapsed_discrepancies":discrepancies}
    rows.append(row)
    all_gate_discrepancies.extend({"key":key,**d} for d in discrepancies)
    if "practice_summary" in r:
        p=[]
        for stage in r["practice_summary"]["stages"]:
            start,end=stage["start_tick"],stage["end_tick"];goal=np.array(stage["target"])
            steady=np.arange(start+25,end)
            e=np.abs(a["metrics"][steady,:2]-goal[:2]).mean(axis=0)
            close(e,[stage["steady_vx_mae_mps"],stage["steady_yaw_mae_rad_s"]])
            axis=0 if r["practice_summary"]["initial_command"][0]!=0 else 1
            band=np.abs(a["metrics"][start:end,axis]-goal[axis])<=.15
            count=0;held=None
            for t,inside in enumerate(band):
                count=count+1 if inside else 0
                if count==10:held=(t+1)*dt;break
            if goal[axis]!=0:assert held==stage["held_error_band_confirmation_time_s"]
            reasons=[]
            if stage["stage_index"] in [1,3,5]:
                if e[0]>.1:reasons.append("stop_vx_mae")
                if e[1]>.15:reasons.append("stop_yaw_mae")
            if stage["stage_index"] in [2,4] and (held is None or held>1):reasons.append("reverse_response_time")
            assert not reasons and stage["performance_accepted"]
            p.append({"stage":stage["stage_index"],"vx_mae":float(e[0]),"yaw_mae":float(e[1]),"held_s":held})
        practices.append({"key":key,"stages":p})
old=np.load(ROOT/"physics-bandwidth-trajectories.npz")["unmodified__physics"]
new=np.load(ROOT/"final-physics-trajectories.npz")["physics"]
rms=[]
for phy in [old,new]:
    assert phy.shape==(10000,25)
    close(phy[1:,13:19],np.diff(phy[:,7:13],axis=0)/.001,0)
    rms.append(np.sqrt(np.mean(phy[phy[:,0]>=2,13:19]**2,axis=0)))
close(new[19::20,1:7],z["candidate_v1__nominal_stand__q"],0)
close(new[19::20,7:13],z["candidate_v1__nominal_stand__v"],0)
reduction=1-np.linalg.norm(rms[1][:2])/np.linalg.norm(rms[0][:2])
close(reduction,ver["left_pair_accel_rms_reduction_fraction"])
width=[np.rad2deg(np.percentile(p[p[:,0]>=2,1:5],95,axis=0)-np.percentile(p[p[:,0]>=2,1:5],5,axis=0)) for p in [old,new]]
close(width[0],cmp["leg_q_p95_p5_deg_1ms"]["original"]);close(width[1],cmp["leg_q_p95_p5_deg_1ms"]["candidate"])
for r in cmp["formal_rows"]:
    recorded=next(x for x in rows if x["key"]=="candidate_v1__"+r["scenario"])
    close(recorded["vx_mae_original_command"],r["new_vx_mae"]);close(recorded["yaw_mae_original_command"],r["new_yaw_mae"])
assert all(sha(ROOT/n)==h for n,h in inputs.items())
summary={"all_independent_record_checks_passed":True,"protected_file_count":len(protected),"recorded_case_count":len(rows),"frames":sum(r["frames"] for r in rows),
    "candidate_case_count":sum(not r["key"].startswith("filter_only") for r in rows),"comparator_case_count":sum(r["key"].startswith("filter_only") for r in rows),
    "actual_left_1ms_rms_original":rms[0][:2].tolist(),"actual_left_1ms_rms_candidate":rms[1][:2].tolist(),"actual_left_pair_reduction":float(reduction),
    "actual_q_p95_p5_degrees_original":width[0].tolist(),"actual_q_p95_p5_degrees_candidate":width[1].tolist(),"initial_gate_timing_discrepancies":all_gate_discrepancies,
    "cases":rows,"practices":practices,"input_sha256":inputs,"no_new_rollout":True,"production_modified":False}
(OUT/"results.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k not in ["cases","practices","input_sha256","initial_gate_timing_discrepancies"]},indent=2))
print("Gate discrepancy count",len(all_gate_discrepancies))
print("Stand60",next(r for r in rows if r["key"]=="candidate_v1__stand60"))
