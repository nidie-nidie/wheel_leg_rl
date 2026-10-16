"""Offline target/physical-motion comparison on verified saved data. No rollout."""
from pathlib import Path
from datetime import datetime
import csv, hashlib, json, subprocess
import numpy as np

ROOT=Path(r"E:\wheel_leg_rl-main")
PROJECT=ROOT/"wheelleg_dreamwaq"
PREVIOUS=PROJECT/"artifacts/debug/sim2sim/reversal-noise-diagnosis-20261011-v1"
TRIAL=PROJECT/"artifacts/phase2_dreamwaq/leg-reversal-trial-20261011-v1"
OUT=Path(__file__).resolve().parent
DT=.02
KEEP=np.arange(100,500)
JOINTS=("jIJ","jIO","jAB","jAG")

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def write(p,v): Path(p).write_bytes((json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8"))
def rms(x): return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
def corr(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-10 and np.std(y)>1e-10 else None

def spectral(x):
    x=np.asarray(x,dtype=np.float64)
    if np.std(x)<1e-10:
        return {"peak_hz":None,"power_fraction_ge_hz":{str(f):None for f in (3,5,8)}}
    freq=np.fft.rfftfreq(len(x),DT)
    power=np.abs(np.fft.rfft((x-x.mean())*np.hanning(len(x))))**2
    power[0]=0; power[1:-1]*=2
    return {"peak_hz":float(freq[np.argmax(power)]),
        "power_fraction_ge_hz":{str(f):float(power[freq>=f].sum()/power.sum()) for f in (3,5,8)}}

def signals(data):
    target=data["leg_target"].astype(np.float64)
    d1=np.zeros_like(target); d1[1:]=target[1:]-target[:-1]
    d2=np.zeros_like(target); d2[2:]=target[2:]-2*target[1:-1]+target[:-2]
    return {"target":target,"position":data["q_after"][:,:4],
        "velocity":data["v_after"][:,:4],
        "acceleration":(data["v_after"][:,:4]-data["v_before"][:,:4])/DT,
        "target_increment":d1,"target_second":d2,"tracking_error":target-data["q_after"][:,:4]}

def joint_metrics(s,idx,j):
    r={}
    for key in ("target_increment","target_second","acceleration","velocity","tracking_error"):
        x=s[key][idx,j]
        r[key+"_rms"]=rms(x)
        r[key+"_p95_abs"]=float(np.percentile(np.abs(x),95))
        r[key+"_max_abs"]=float(np.abs(x).max())
    for key in ("position","target"):
        x=s[key][idx,j]
        r[key+"_centered_rms_rad"]=rms(x-x.mean())
        r[key+"_p95_p05_span_rad"]=float(np.percentile(x,95)-np.percentile(x,5))
    for key in ("position","target","velocity","acceleration"):
        r[key+"_spectrum"]=spectral(s[key][idx,j])
    r["target_second_acceleration_correlations_by_lag"]={
        str(lag):corr(s["target_second"][idx[:-lag] if lag else idx,j],
                      s["acceleration"][idx[lag:] if lag else idx,j]) for lag in range(6)}
    return r

def vehicle(c,idx):
    r={"vx_mae_mps":float(np.mean(np.abs(c["vx_mps"][idx]-c["target_vx_mps"][idx]))),
       "yaw_rate_mae_rad_s":float(np.mean(np.abs(c["yaw_rate_rad_s"][idx]-c["target_yaw_rate_rad_s"][idx]))),
       "height_mae_m":float(np.mean(np.abs(c["base_height_m"][idx]-c["target_base_height_m"][idx])))}
    for key in ("pitch_rad","roll_rad","phi0_left_rad","phi0_right_rad","l0_left_m","l0_right_m","base_height_m","vx_mps"):
        x=c[key][idx]
        r[key+"_rms"]=rms(x); r[key+"_centered_rms"]=rms(x-x.mean())
        r[key+"_p95_p05_span"]=float(np.percentile(x,95)-np.percentile(x,5))
        r[key+"_spectrum"]=spectral(x)
    return r

def sampling(record,mean,noise,std):
    def targets(actions):
        return np.clip(np.asarray(record["q_nominal"])[None,None,:]+record["leg_action_scale"]*np.clip(actions[...,:4],-1,1),
                       record["target_lower"],record["target_upper"])
    def summarize(t):
        d1=t[KEEP]-t[KEEP-1]; d2=t[KEEP]-2*t[KEEP-1]+t[KEEP-2]
        return {"increment_rms_rad":np.sqrt(np.mean(d1**2,axis=(0,1))).tolist(),
                "second_rms_rad":np.sqrt(np.mean(d2**2,axis=(0,1))).tolist()}
    return {"same_histories_mean":summarize(targets(mean[:,None,:])),
        "same_histories_sampled":summarize(targets(mean[:,None,:]+noise*std)),
        "held_mean":summarize(targets(np.repeat(mean[250:251,None,:],500,axis=0))),
        "held_mean_sampled":summarize(targets(mean[250][None,None,:]+noise*std))}

def synthetic():
    t=np.arange(500)*DT
    ex={"constant":np.zeros(500),"constant_speed":.02*t,
        "small_5hz":.02*np.sin(2*np.pi*5*t),"large_5hz":.2*np.sin(2*np.pi*5*t),
        "small_10hz":.02*np.sin(2*np.pi*10*t)}
    r={}
    for name,q in ex.items():
        v=np.gradient(q,DT)
        r[name]={"target_increment_rms":rms(q[KEEP]-q[KEEP-1]),
            "target_second_rms":rms(q[KEEP]-2*q[KEEP-1]+q[KEEP-2]),
            "actual_acceleration_rms":rms((v[KEEP]-v[KEEP-1])/DT),
            "actual_velocity_rms":rms(v[KEEP])}
    return r

def main():
    capture=read(PREVIOUS/"capture-verification.json")
    previous=read(PREVIOUS/"results.json")
    old=read(PREVIOUS/"input-identity.json")
    assert capture["completed"] and len(capture["captures"])==9
    inputs={str(p.relative_to(ROOT)):sha(p) for p in (
        PREVIOUS/"capture-verification.json",PREVIOUS/"results.json",PREVIOUS/"standardized-noise.npz",
        TRIAL/"experiment-manifest.json",TRIAL/"results.json")}
    for r in capture["captures"]:
        assert r["dt"]==DT and tuple(r["joint_order"][:4])==JOINTS
        for name,key in (("capture","capture_sha256"),("reference","reference_sha256"),("checkpoint","checkpoint_sha256")):
            assert sha(r[name])==r[key]
            inputs[str(Path(r[name]).relative_to(ROOT))]=r[key]
    status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True,encoding="utf-8")
    tracked=subprocess.check_output(["git","diff","--name-only"],cwd=ROOT,text=True,encoding="utf-8").splitlines()
    protected=dict(old["protected_files"])
    protected.update({name:sha(ROOT/name) for name in tracked})
    for name,expected in protected.items(): assert sha(ROOT/name)==expected,name
    ip=OUT/"input-identity.json"
    if ip.exists():
        identity=read(ip)
        assert identity["inputs"]==inputs and identity["protected_files"]==protected
    else:
        write(ip,{"created_at":datetime.now().astimezone().isoformat(),"inputs":inputs,"protected_files":protected,
            "tracked_changes":tracked,"source_fingerprint":old["source_fingerprint"],"git_status":status})
    noise=np.load(PREVIOUS/"standardized-noise.npz")["noise"].astype(np.float64)
    output=[]; aligned=[]; windows=[]; maxerror=0.
    for r in capture["captures"]:
        d=dict(np.load(r["capture"]))
        with Path(r["reference"]).open(encoding="utf-8",newline="") as f: original=list(csv.DictReader(f))
        c={key:np.array([float(row[key]) for row in original]) for key in original[0] if key!="failure_reason"}
        assert len(original)==500 and np.all(c["survived"]==1)
        assert np.allclose(np.diff(d["time_s"]),DT,rtol=0,atol=2e-13)
        assert np.max(np.abs(d["q_before"][1:]-d["q_after"][:-1]))==0
        assert np.max(np.abs(d["v_before"][1:]-d["v_after"][:-1]))==0
        s=signals(d); joints=[]
        for j,name in enumerate(JOINTS):
            for key,sig in ((f"leg_target_{j}_rad","target"),(f"leg_velocity_{j}_rad_s","velocity"),
                            (f"leg_acceleration_{j}_rad_s2","acceleration")):
                maxerror=max(maxerror,float(np.abs(c[key]-s[sig][:,j]).max()))
            m=joint_metrics(s,KEEP,j); m["joint"]=name
            m["target_reversal_rate_per_s"]=float(c[f"leg_reversal_rate_{j}_per_s"][KEEP].mean())
            m["raw_action_saturation_fraction"]=float(np.mean(np.abs(d["mean_action"][KEEP,j])>=1))
            m["torque_rms_nm"]=rms(d["torque"][KEEP,j])
            joints.append(m)
        assert maxerror<1e-10
        p=next(p for p in previous["comparisons"] if p["arm"]==r["arm"] and p["scenario"]==r["scenario"])
        sample=sampling(r,d["mean_action"].astype(np.float64),noise,np.asarray(p["effective_action_std"]))
        ws=[]
        for start in range(100,500,50):
            idx=np.arange(start,start+50)
            w={"arm":r["arm"],"scenario":r["scenario"],"time_s_start":float(d["time_s"][start]),
               "time_s_end":float(d["time_s"][start+49]),"vehicle":vehicle(c,idx),
               "joints":[joint_metrics(s,idx,j) for j in range(4)]}
            ws.append(w); windows.append(w)
        for i in KEEP:
            row={"arm":r["arm"],"scenario":r["scenario"],"time_before_s":float(d["time_s"][i]-DT),"time_after_s":float(d["time_s"][i])}
            for j,name in enumerate(JOINTS):
                for key in s: row[name+"_"+key]=float(s[key][i,j])
                row[name+"_position_before"]=float(d["q_before"][i,j])
                row[name+"_torque"]=float(d["torque"][i,j])
            for key in ("pitch_rad","roll_rad","phi0_left_rad","phi0_right_rad","l0_left_m","l0_right_m","base_height_m","target_base_height_m","vx_mps","target_vx_mps","yaw_rate_rad_s"):
                row[key]=float(c[key][i])
            aligned.append(row)
        all_acc=(d["v_after"]-d["v_before"])/DT
        acc_energy=np.mean(all_acc[KEEP]**2,axis=0)
        clipped=np.clip(d["mean_action"].astype(np.float64),-1,1)
        action_second=clipped[KEEP]-2*clipped[KEEP-1]+clipped[KEEP-2]
        smooth_energy=np.mean(action_second**2,axis=0)
        existing={"acceleration_mse_per_dimension":acc_energy.tolist(),
            "action_second_mse_per_dimension":smooth_energy.tolist(),
            "acceleration_leg_share":float(acc_energy[:4].sum()/acc_energy.sum()),
            "action_smooth_leg_share":float(smooth_energy[:4].sum()/smooth_energy.sum()),
            "acceleration_penalty_rate_legs":float(2.5e-7*acc_energy[:4].sum()),
            "acceleration_penalty_rate_wheels":float(2.5e-7*acc_energy[4:].sum()),
            "action_smooth_penalty_rate_legs":float(.01*smooth_energy[:4].sum()),
            "action_smooth_penalty_rate_wheels":float(.01*smooth_energy[4:].sum())}
        item={"arm":r["arm"],"scenario":r["scenario"],"samples":len(KEEP),"joints":joints,
              "vehicle":vehicle(c,KEEP),"sampling":sample,"windows":ws,"existing_six_dimension_cost":existing,
              "sources":{key:r[key] for key in ("capture","capture_sha256","reference","reference_sha256")}}
        output.append(item)
        print(json.dumps({"arm":r["arm"],"scenario":r["scenario"],"left_target_second_rms":[j["target_second_rms"] for j in joints[:2]],
            "left_acceleration_rms":[j["acceleration_rms"] for j in joints[:2]],"vx_mae":item["vehicle"]["vx_mae_mps"]}),flush=True)
    pooled={}
    for j,name in enumerate(JOINTS):
        pooled[name]={}
        for metric in ("target_increment_rms","target_second_rms","target_reversal_rate_per_s","velocity_rms"):
            pooled[name][metric+"_vs_acceleration_rms_9_trajectories"]=corr(
                [r["joints"][j][metric] for r in output],[r["joints"][j]["acceleration_rms"] for r in output])
        for scenario in ("nominal_stand","forward","reverse"):
            w=[w for w in windows if w["scenario"]==scenario]
            pooled[name][scenario+"_within_windows"]={}
            for metric in ("target_increment_rms","target_second_rms","velocity_rms","acceleration_rms"):
                pooled[name][scenario+"_within_windows"][metric+"_vs_vx_mae"]=corr(
                    [x["joints"][j][metric] for x in w],[x["vehicle"]["vx_mae_mps"] for x in w])
    result={"diagnostic_only":True,"computed_at":datetime.now().astimezone().isoformat(),"dt_s":DT,
        "window":"post-action 2s<t<=10s; indices 100..499; 400 frames per trajectory","joint_order":JOINTS,
        "trajectories":output,"correlations_descriptive_only":pooled,"synthetic":synthetic(),
        "verification":{"csv_capture_max_error":maxerror,"trajectories":9,"input_frames":4500,
            "analyzed_frames":len(aligned),"joint_samples":len(aligned)*4,"one_second_windows":len(windows)},
        "units":{"target_increment_rms":"rad/control step","target_second_rms":"rad/control step squared",
            "acceleration_rms":"rad/s^2","velocity_rms":"rad/s"},
        "limitations":["One seed; three matched 200-iteration branches; deterministic flat-ground MuJoCo.",
            "No push/recovery or changing-command trajectories; no safe threshold or necessary-action identification.",
            "20ms interval-mean acceleration, not instantaneous 1ms qacc; Nyquist 25Hz.",
            "Commands apply at time_before; positions/vehicle are sampled at time_after.",
            "Four motors separate; right targets saturated, not a healthy baseline.",
            "Exploration probes use fixed histories/noise, not applied to physics; no physical noise contribution estimated.",
            "Correlations/windows share trajectories, descriptive only; no causal interpretation.",
            "3/5/8Hz bands are sensitivity diagnostics only; no filter or reward change."]}
    write(OUT/"results.json",result)
    with (OUT/"aligned-data.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(aligned[0])); writer.writeheader(); writer.writerows(aligned)
    fields=["arm","scenario","joint","target_reversal_rate_per_s","target_increment_rms","target_second_rms",
        "position_centered_rms_rad","velocity_rms","acceleration_rms","raw_action_saturation_fraction","vx_mae_mps"]
    with (OUT/"joint-metrics.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in output:
            for j in r["joints"]:
                row={"arm":r["arm"],"scenario":r["scenario"],"vx_mae_mps":r["vehicle"]["vx_mae_mps"]}
                row.update({field:j[field] for field in fields if field in j}); writer.writerow(row)
    print(json.dumps(result["verification"],indent=2),flush=True)

if __name__=="__main__": main()
