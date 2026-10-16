from pathlib import Path
from datetime import datetime,timezone,timedelta
import csv,io,json,hashlib,math,sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
root=Path(r"E:\wheel_leg_rl-main")
loc=json.loads((Path(__file__).resolve().parent.parent/"input-identity.json").read_text(encoding="utf-8"))
out=Path(loc["output"]);exp=Path(loc["experiment"])
DT=.02
names=["jIJ","jIO","jAB","jAG"];joint_labels=["left front","left rear","right front","right rear"]
source_manifest=[];cases=[];datasets={};frame_rows=[];window_rows=[]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest().upper()
def stat(v):
    return {"mean":float(np.mean(v)),"rms":float(np.sqrt(np.mean(v*v))),"median":float(np.median(v)),
        "p95":float(np.quantile(v,.95)),"p99":float(np.quantile(v,.99)),"max":float(np.max(v))}
def corr(x,y):
    return float(np.corrcoef(x,y)[0,1]) if len(x)>3 and np.std(x)>1e-12 and np.std(y)>1e-12 else None
def spectrum(v):
    centered=v-v.mean()
    if np.std(centered)<1e-12:return {"peak_hz":None,"peak_power_fraction":None,"power":None,"frequencies":None}
    power=np.abs(np.fft.rfft(centered*np.hanning(len(centered))))**2
    power[0]=0.;freq=np.fft.rfftfreq(len(v),DT)
    idx=int(np.argmax(power))
    return {"peak_hz":float(freq[idx]),"peak_power_fraction":float(power[idx]/power.sum()),"power":power,"frequencies":freq}
def load(folder):
    path=folder/"control.csv"
    data=path.read_bytes()
    rows=list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    c=lambda name:np.array([float(r[name]) for r in rows])
    a=np.column_stack([c(name+"_action_clipped") for name in names])
    qb=np.column_stack([c(name+"_position_before_rad") for name in names]);qa=np.column_stack([c(name+"_position_after_rad") for name in names])
    vb=np.column_stack([c(name+"_velocity_before_rad_s") for name in names]);va=np.column_stack([c(name+"_velocity_after_rad_s") for name in names])
    target=np.column_stack([c(name+"_position_target_rad") for name in names])
    t=c("time_before_s");ta=c("time_after_s")
    ep=c("episode_id").astype(int);cid=c("control_id").astype(int)
    assert np.max(np.abs(ta-t-DT))<1e-10
    adjacent=(ep[1:]==ep[:-1])&(cid[1:]==cid[:-1]+1)
    assert np.max(np.abs(vb[1:][adjacent]-va[:-1][adjacent]))<1e-12
    assert np.max(np.abs(qb[1:][adjacent]-qa[:-1][adjacent]))<1e-12
    acc=(va-vb)/DT
    d2=np.full_like(a,np.nan);d2[2:]=np.diff(a,n=2,axis=0)
    valid=np.zeros(len(rows),dtype=bool);valid[2:]=adjacent[1:]&adjacent[:-1]
    command=np.column_stack([c("command_vx_mps"),c("command_yaw_rad_s"),c("command_height_m")])
    assert np.isfinite(np.column_stack([a,qa,qb,vb,va,acc,command])).all()
    meta=json.loads((folder/"telemetry_manifest.json").read_text(encoding="utf-8"))
    source_manifest.append({"path":str(path),"sha256":hashlib.sha256(data).hexdigest().upper(),"rows":len(rows),
        "source_checkpoint":meta["source_checkpoint"],"source_completed_iterations":meta["source_completed_iterations"],
        "control_dt_s":meta["control_dt_s"],"physics_dt_s":meta["physics_dt_s"],"model_xml_sha256":meta["model_xml_sha256"]})
    return dict(path=path,folder=folder,rows=rows,a=a,qb=qb,qa=qa,vb=vb,va=va,target=target,t=t,ta=ta,ep=ep,cid=cid,
        acc=acc,d2=d2,valid=valid,command=command,vx=c("vx_mps"),yaw=c("yaw_rate_rad_s"),meta=meta)
def summarize(label,d,mask):
    valid=d["valid"]&mask
    valid[2:]&=mask[1:-1]&mask[:-2]
    ix=np.flatnonzero(valid)
    assert len(ix)>10 and np.all(np.diff(ix)==1)
    cost=d["d2"]**2
    joint={}
    for j,name in enumerate(names):
        speca=spectrum(d["a"][ix,j]);specv=spectrum(d["vb"][ix,j])
        correlations={}
        for lag in [-2,-1,0,1,2]:
            x=ix[np.isin(ix+lag,ix)]
            correlations[str(lag)]=corr(d["d2"][x,j],d["acc"][x+lag,j])
        joint[name]={"action_second_difference":stat(d["d2"][ix,j]),"action_smooth_cost_mean":float(cost[ix,j].mean()),
            "actual_velocity_rms_rad_s":float(np.sqrt(np.mean(d["vb"][ix,j]**2))),
            "actual_acceleration_20ms":stat(np.abs(d["acc"][ix,j])),
            "target_step_rms_rad":float(np.sqrt(np.mean(np.diff(d["target"][ix,j])**2))),
            "target_actual_error_rms_rad":float(np.sqrt(np.mean((d["target"][ix,j]-d["qa"][ix,j])**2))),
            "action_at_limit_fraction":float(np.mean(np.abs(d["a"][ix,j])>=.999)),
            "command_action_dominant_frequency_hz":speca["peak_hz"],"actual_velocity_dominant_frequency_hz":specv["peak_hz"],
            "signed_action_second_diff_vs_actual_acceleration_correlation_by_response_lag_frames":correlations,
            "action_cost_vs_actual_acceleration_squared_correlation":corr(cost[ix,j],d["acc"][ix,j]**2)}
    windows=[]
    for start in range(0,len(ix)-49,50):
        k=ix[start:start+50]
        row={"dataset":label,"episode_id":int(d["ep"][k[0]]),"time_start_s":float(d["t"][k[0]]),"time_end_s":float(d["ta"][k[-1]]),
            "leg_smooth_mean":float(cost[k].sum(axis=1).mean()),"left_smooth_mean":float(cost[k,:2].sum(axis=1).mean()),
            "right_smooth_mean":float(cost[k,2:].sum(axis=1).mean()),"left_acceleration_norm_rms_rad_s2":float(np.sqrt(np.mean(np.sum(d["acc"][k,:2]**2,axis=1)))),
            "right_acceleration_norm_rms_rad_s2":float(np.sqrt(np.mean(np.sum(d["acc"][k,2:]**2,axis=1)))),
            "vx_mae_mps":float(np.abs(d["vx"][k]-d["command"][k,0]).mean())}
        for j,name in enumerate(names):
            row[name+"_smooth_mean"]=float(cost[k,j].mean());row[name+"_acceleration_rms_rad_s2"]=float(np.sqrt(np.mean(d["acc"][k,j]**2)))
        windows.append(row);window_rows.append(row)
    result={"label":label,"source":str(d["path"]),"source_checkpoint":d["meta"]["source_checkpoint"],
        "source_completed_iterations":d["meta"]["source_completed_iterations"],"samples":len(ix),
        "time_start_s":float(d["t"][ix[0]]),"time_end_s":float(d["ta"][ix[-1]]),"episode_id":int(d["ep"][ix[0]]),
        "leg_smooth_cost":stat(cost[ix].sum(axis=1)),"left_smooth_mean":float(cost[ix,:2].sum(axis=1).mean()),
        "right_smooth_mean":float(cost[ix,2:].sum(axis=1).mean()),
        "leg_acceleration_squared_sum_mean":float(np.mean(np.sum(d["acc"][ix]**2,axis=1))),
        "vx_mae_mps":float(np.abs(d["vx"][ix]-d["command"][ix,0]).mean()),
        "yaw_rate_mae_rad_s":float(np.abs(d["yaw"][ix]-d["command"][ix,1]).mean()),
        "joint_metrics":joint,"one_second_windows":windows,
        "one_second_left_smooth_vs_left_acceleration_rms_pearson":corr(np.array([w["left_smooth_mean"] for w in windows]),np.array([w["left_acceleration_norm_rms_rad_s2"] for w in windows]))}
    for i in ix:
        row={"dataset":label,"episode_id":int(d["ep"][i]),"control_id":int(d["cid"][i]),"time_before_s":float(d["t"][i]),
            "time_after_s":float(d["ta"][i]),"command_vx_mps":float(d["command"][i,0]),"command_yaw_rad_s":float(d["command"][i,1]),
            "command_height_m":float(d["command"][i,2]),"actual_vx_mps":float(d["vx"][i]),"leg_smooth":float(cost[i].sum()),
            "left_smooth":float(cost[i,:2].sum()),"right_smooth":float(cost[i,2:].sum())}
        for j,name in enumerate(names):
            for suffix,values in [("action",d["a"]),("target_rad",d["target"]),("actual_rad",d["qa"]),("velocity_after_rad_s",d["va"]),
                ("acceleration_20ms_rad_s2",d["acc"]),("action_second_difference",d["d2"])]:
                row[name+"_"+suffix]=float(values[i,j])
        frame_rows.append(row)
    cases.append(result);datasets[label]=(d,ix)
    return result
for label in ["old_1000","new_1000","old_1401_reference"]:
    d=load(exp/"stationary-traces"/label)
    mask=(d["t"]>=2-1e-9)&(d["t"]<10-1e-9)
    r=summarize(label+"_parking",d,mask)
    assert r["samples"]==398
# Stable keyboard runs identified in the previous diagnostic, separately from controlled reference runs.
previous=json.loads((root/"wheelleg_dreamwaq/artifacts/debug/sim2sim/action-smooth-metric-20261011-180240/metrics.json").read_text(encoding="utf-8"))
active=Path('E:\\wheel_leg_rl-main\\wheelleg_dreamwaq\\artifacts\\debug\\sim2sim\\joint-telemetry-smooth002-20261011-171956')
d=load(active)
for part in previous["user_keyboard"]["stable_command_segments"]:
    start,end=[int(x) for x in part["label"].split("_frame")[1].split("_")]
    mask=(d["cid"]>=start+100)&(d["cid"]<=end)
    r=summarize("user_"+("parking" if abs(part["command"][0])<1e-8 else "forward02"),d,mask)
for label in ["forward","reverse","forward_stop_reverse"]:
    d=load(out/"supplemental-traces"/label)
    mask=(d["t"]>=2-1e-9)&(d["t"]<10-1e-9)
    summarize("new_"+label,d,mask)
practice=datasets["new_forward_stop_reverse"][0]
transitions=np.flatnonzero(np.any(np.abs(practice["command"][1:]-practice["command"][:-1])>1e-7,axis=1))+1
practice_stages=[]
edges=np.r_[0,transitions,len(practice["rows"])]
for stage,(start,end) in enumerate(zip(edges[:-1],edges[1:])):
    for category,lo,hi in [("transition_first_0.5s",start,min(start+25,end)),("steady_after_0.5s",min(start+25,end),end)]:
        mask=np.zeros(len(practice["rows"]),dtype=bool);mask[lo:hi]=True
        mask&=practice["t"]>=2-1e-9
        mask&=practice["valid"]
        if not mask.any():continue
        cost=practice["d2"][mask]**2
        practice_stages.append({"stage":stage,"category":category,"command":practice["command"][start].tolist(),
            "samples":int(mask.sum()),"time_start_s":float(practice["t"][np.flatnonzero(mask)[0]]),"time_end_s":float(practice["ta"][np.flatnonzero(mask)[-1]]),
            "left_smooth_mean":float(cost[:,:2].sum(axis=1).mean()),
            "per_joint_actual_acceleration_rms_rad_s2":np.sqrt(np.mean(practice["acc"][mask]**2,axis=0)).tolist(),
            "vx_mae_mps":float(np.abs(practice["vx"][mask]-practice["command"][mask,0]).mean())})
# Verify 20 ms acceleration using 20 actual 1 ms velocity increments; report instantaneous qacc separately.
physics_checks=[]
for label in ["old_1000_parking","new_1000_parking","old_1401_reference_parking","new_forward","new_reverse","new_forward_stop_reverse"]:
    d,ix=datasets[label];path=d["folder"]/"physics.csv"
    sums={};velocity_deltas={};squared=np.zeros(4);counts={};chosen_ids=set(map(int,d['cid'][ix]))
    with path.open(encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            key=int(row["control_id"])
            qacc=np.array([float(row[name+"_qacc_rad_s2"]) for name in names])
            sums[key]=sums.get(key,np.zeros(4))+qacc
            delta=np.array([float(row[name+"_velocity_after_rad_s"])-float(row[name+"_velocity_before_rad_s"]) for name in names])
            velocity_deltas[key]=velocity_deltas.get(key,np.zeros(4))+delta
            counts[key]=counts.get(key,0)+1
            if key in chosen_ids:squared+=qacc*qacc
    integration_error=0;velocity_alignment_error=0
    for i in ix:
        cid=int(d["cid"][i]);assert counts[cid]==20
        integration_error=max(integration_error,float(np.max(np.abs(sums[cid]/20-d["acc"][i]))))
        velocity_alignment_error=max(velocity_alignment_error,float(np.max(np.abs(velocity_deltas[cid]/DT-d["acc"][i]))))
    assert velocity_alignment_error<1e-10,(label,velocity_alignment_error)
    physics_checks.append({"dataset":label,"physics_csv":str(path),"physics_csv_sha256":sha(path),
        "velocity_increment_1ms_vs_20ms_endpoint_acceleration_max_error_rad_s2":velocity_alignment_error,
        "instantaneous_qacc_mean_vs_net_acceleration_difference_rad_s2":integration_error,
        "instantaneous_1ms_qacc_rms_per_joint_rad_s2":dict(zip(names,map(float,np.sqrt(squared/(len(ix)*20))))),
        "not_pooled_with_20ms_acceleration":True})
report={"schema":"LegJitterIndicatorDiagnosisV1","generated_at":datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "analysis_only":True,"control_dt_s":DT,"action_order":names,"sources":source_manifest,
    "definitions":{"leg_action_smooth":"sum_{i in four legs}(a_t-2*a_t_minus_1+a_t_minus_2)^2; clipped normalized canonical actions",
        "actual_acceleration_20ms":"(actual_velocity_after_control-actual_velocity_before_control)/0.02 rad/s^2",
        "acceleration_aggregate":"sum of squared four-leg accelerations; each joint RMS is also reported",
        "correlation":"Pearson association, not a causal test. Lags -2..2 are exploratory and descriptive.",
        "spectrum":"Demean, Hann window, rFFT of 50 Hz control samples. All positive frequencies considered, no cutoff or safety threshold chosen."},
    "cases":cases,"command_practice_stages":practice_stages,"physics_alignment_checks":physics_checks,
    "limits":["No human labels for acceptable vs unacceptable jitter; no acceptance threshold selected.",
        "Finite-difference acceleration includes normal acceleration/braking; command changes reported separately.",
        "Right-leg persistent clipping can hide command changes. Zero action-smooth cost does not imply zero mechanical motion.",
        "Overlapping closed-loop action/feedback cannot identify whether CENet, policy, PD or contacts are the root cause.",
        "Only one source seed; old1401 is a longer-trained supplementary reference.",
        "One-second windows per case are descriptive, not independent robot trials; small sample counts preclude robust threshold calibration."]}
(out/"indicators.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
for filename,rows in [("aligned-frames.csv",frame_rows),("one-second-windows.csv",window_rows)]:
    with (out/filename).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
# Store numeric plot data so rendering can be repeated without changing any measurements.
np.savez_compressed(out/"plot-data.npz",**{label+"_"+key:d[key] for label,(d,ix) in datasets.items() for key in ["a","qa","target","vb","va","acc","d2","t","ta","command","vx"]},
    **{label+"_indices":ix for label,(d,ix) in datasets.items()})
summary=[]
for c in cases:
    summary.append({"label":c["label"],"samples":c["samples"],"left_smooth":c["left_smooth_mean"],"vx_mae":c["vx_mae_mps"],
        "acc_rms_20ms":[c["joint_metrics"][n]["actual_acceleration_20ms"]["rms"] for n in names],
        "action_peak_hz":[c["joint_metrics"][n]["command_action_dominant_frequency_hz"] for n in names],
        "velocity_peak_hz":[c["joint_metrics"][n]["actual_velocity_dominant_frequency_hz"] for n in names],
        "window_pearson":c["one_second_left_smooth_vs_left_acceleration_rms_pearson"],
        "same_frame_per_joint_corr":[c["joint_metrics"][n]["signed_action_second_diff_vs_actual_acceleration_correlation_by_response_lag_frames"]["0"] for n in names]})
print(json.dumps({"output":str(out),"cases":summary,"practice":practice_stages,"physics_checks":[{"case":r["dataset"],"alignment_max_error":r["velocity_increment_1ms_vs_20ms_endpoint_acceleration_max_error_rad_s2"]} for r in physics_checks]},ensure_ascii=False))
