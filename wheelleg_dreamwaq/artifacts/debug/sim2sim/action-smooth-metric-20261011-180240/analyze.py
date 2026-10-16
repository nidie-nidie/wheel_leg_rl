from pathlib import Path
import csv, io, json, hashlib, math, sys
from datetime import datetime, timezone, timedelta
SGT = timezone(timedelta(hours=8))
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
root=Path(r"E:\wheel_leg_rl-main")
experiment=root/"wheelleg_dreamwaq/artifacts/phase2_dreamwaq/action-smooth2-run03-fresh1000-20261011-161144"
stamp=datetime.now(SGT).strftime("%Y%m%d-%H%M%S")
out=root/f"wheelleg_dreamwaq/artifacts/debug/sim2sim/action-smooth-metric-{stamp}"
out.mkdir(parents=True,exist_ok=False)
names=["jIJ","jIO","jAB","jAG","jwheel_left","jwheel_right"]
labels=["left_front","left_rear","right_front","right_rear","left_wheel","right_wheel"]
sources=[]
all_points=[]
def sha(data): return hashlib.sha256(data).hexdigest().upper()
def load(path,kind):
    with path.open("rb") as f:
        size=path.stat().st_size
        data=f.read(size)
    complete=data[:data.rfind(b"\n")+1]
    rows=list(csv.DictReader(io.StringIO(complete.decode("utf-8-sig"))))
    def col(key): return np.array([float(r[key]) for r in rows])
    prefix = "jIJ_action_clipped" in rows[0]
    action=np.column_stack([col(name+"_action_clipped") for name in names]) if prefix else np.column_stack([col("clipped_action_"+str(i)) for i in range(6)])
    assert np.isfinite(action).all() and np.max(np.abs(action))<=1+1e-6
    t=col("time_before_s" if prefix else "time_s")
    ep=col("episode_id").astype(int) if prefix else np.zeros(len(rows),dtype=int)
    cid=col("control_id" if prefix else "tick").astype(int)
    command=np.column_stack([col(key) for key in (["command_vx_mps","command_yaw_rad_s","command_height_m"] if prefix else ["target_vx_mps","target_yaw_rate_rad_s","target_base_height_m"])])
    vx=col("vx_mps")
    yaw=col("yaw_rate_rad_s")
    parts=np.diff(action,n=2,axis=0)**2
    valid=(ep[2:]==ep[1:-1])&(ep[1:-1]==ep[:-2])&(cid[2:]==cid[1:-1]+1)&(cid[1:-1]==cid[:-2]+1)
    if prefix:
        velocity=np.column_stack([col(name+"_velocity_before_rad_s") for name in names[:4]])
    else:
        velocity=None
    entry={"path":str(path),"kind":kind,"captured_bytes":len(data),"complete_bytes":len(complete),"snapshot_sha256":sha(complete),"frames":len(rows)}
    sources.append(entry)
    return dict(rows=rows,action=action,t=t,ep=ep,cid=cid,command=command,vx=vx,yaw=yaw,parts=parts,valid=valid,velocity=velocity,source=entry)
def stats(a):
    return {"mean":float(np.mean(a)),"median":float(np.median(a)),"p90":float(np.quantile(a,.90)),"p95":float(np.quantile(a,.95)),"p99":float(np.quantile(a,.99)),"max":float(np.max(a)),"min":float(np.min(a))}
def summarize(d,mask,label):
    # All three actions in each second difference must be inside the selected window.
    m=d["valid"]&mask[2:]&mask[1:-1]&mask[:-2]
    p=d["parts"][m]
    assert len(p)>0
    total=p.sum(axis=1)
    mu=p.mean(axis=0)
    whole=d["action"][mask]
    result={"label":label,"selected_control_frames":int(mask.sum()),"second_difference_samples":len(p),
        "formula":"sum_i (a_t[i]-2*a_t_minus_1[i]+a_t_minus_2[i])**2, i=0..5",
        "units":"dimensionless squared normalized clipped control action; reward weight and control_dt NOT included",
        "frame_statistic":stats(total),
        "mean_contribution":{"left_leg":float(mu[:2].sum()),"right_leg":float(mu[2:4].sum()),"wheels":float(mu[4:].sum())},
        "wheel_share_percent":float(100*mu[4:].sum()/mu.sum()) if mu.sum()>0 else None,
        "per_joint_mean_squared_second_difference":dict(zip(names,map(float,mu))),
        "per_joint_second_difference_rms":dict(zip(names,map(float,np.sqrt(mu)))),
        "action_saturation_fraction":dict(zip(names,map(float,np.mean(np.abs(whole)>=.999,axis=0)))),
        "action_clipped_min_max":dict(zip(names,[[float(whole[:,i].min()),float(whole[:,i].max())] for i in range(6)])),
        "vx_mae_mps":float(np.abs(d["command"][mask,0]-d["vx"][mask]).mean()),
        "yaw_rate_mae_rad_s":float(np.abs(d["command"][mask,1]-d["yaw"][mask]).mean()),
        "commands_min_max":np.stack([d["command"][mask].min(axis=0),d["command"][mask].max(axis=0)]).tolist(),
        "one_second_window_mean_statistic":None}
    # Nonoverlapping 50-sample windows; do not join discontinuous segments.
    indices=np.flatnonzero(m)+2
    cuts=np.r_[0,np.flatnonzero((np.diff(indices)!=1)|(np.diff(d["ep"][indices])!=0))+1,len(indices)]
    win=[]
    for start,end in zip(cuts[:-1],cuts[1:]):
        for k in range(start,end-49,50):
            ix=indices[k:k+50]
            cost=d["parts"][ix-2]
            sample={"dataset":label,"episode_id":int(d["ep"][ix[0]]),"control_id_start":int(d["cid"][ix[0]]),"control_id_end":int(d["cid"][ix[-1]]),
                "time_start_s":float(d["t"][ix[0]]),"time_end_s":float(d["t"][ix[-1]]+.02),
                "full_mean":float(cost.sum(axis=1).mean()),"left_leg_mean":float(cost[:,:2].sum(axis=1).mean()),"right_leg_mean":float(cost[:,2:4].sum(axis=1).mean()),
                "wheels_mean":float(cost[:,4:].sum(axis=1).mean()),
                "vx_mae_mps":float(np.abs(d["command"][ix,0]-d["vx"][ix]).mean()),
                "yaw_rate_mae_rad_s":float(np.abs(d["command"][ix,1]-d["yaw"][ix]).mean())}
            if d["velocity"] is not None:
                sample["left_leg_velocity_rms_rad_s"]=float(np.sqrt(np.mean(d["velocity"][ix,:2]**2)))
                sample["right_leg_velocity_rms_rad_s"]=float(np.sqrt(np.mean(d["velocity"][ix,2:4]**2)))
            win.append(sample)
    if win:
        result["one_second_window_mean_statistic"]={**stats(np.array([w["full_mean"] for w in win])),"windows":len(win)}
    result["one_second_windows"]=win
    for i in indices:
        p=d["parts"][i-2]
        all_points.append({"dataset":label,"episode_id":int(d["ep"][i]),"control_id":int(d["cid"][i]),"time_s":float(d["t"][i]),
            "full":float(p.sum()),"left_leg":float(p[:2].sum()),"right_leg":float(p[2:4].sum()),"wheels":float(p[4:].sum())})
    return result
cases=[]
previous=json.loads((experiment/"stationary-comparison.json").read_text(encoding="utf-8"))
for label in ["old_1000","new_1000","old_1401_reference"]:
    d=load(experiment/"stationary-traces"/label/"control.csv","fixed_stationary")
    mask=(d["t"]>=2-1e-9)&(d["t"]<10-1e-9)
    assert mask.sum()==400 and np.max(np.abs(d["command"][mask]-[0,0,.2]))<1e-8
    result=summarize(d,mask,label)
    assert result["second_difference_samples"]==398
    for name in names[:4]:
        expected=previous["results"][label]["joint_metrics"][name]["action_second_difference_rms"]
        assert abs(result["per_joint_second_difference_rms"][name]-expected)<1e-12, (label,name)
    result["cross_check_existing_joint_rms"]="matches within 1e-12"
    cases.append(result)
active=json.loads((root/".git/joint-telemetry-active.json").read_text(encoding="utf-8"))
session=Path(active["session"])
d=load(session/"control.csv","user_keyboard")
assert len(d["rows"])>100
events=[json.loads(line) for line in (session/"events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
marker_events=[x for x in events if x["event"]=="symptom_marker" or (x["event"]=="key" and x.get("key")=="M")]
manual={"source_checkpoint":json.loads((session/"telemetry_manifest.json").read_text(encoding="utf-8"))["source_checkpoint"],
    "session_closed":events[-1]["event"]=="closed","frames":len(d["rows"]),"episodes":int(d["ep"].max()+1),
    "commands_min_max":np.stack([d["command"].min(axis=0),d["command"].max(axis=0)]).tolist(),"symptom_marker_events":marker_events,
    "stable_command_segments":[]}
# Constant commands are compared separately; every segment loses its initial 2 s settling period.
change=(d["ep"][1:]!=d["ep"][:-1])|np.any(np.abs(d["command"][1:]-d["command"][:-1])>1e-7,axis=1)
bounds=np.r_[0,np.flatnonzero(change)+1,len(d["rows"])]
for start,end in zip(bounds[:-1],bounds[1:]):
    if end-start<150:
        continue
    mask=np.zeros(len(d["rows"]),dtype=bool)
    mask[start+100:end]=True
    result=summarize(d,mask,f"keyboard_ep{d['ep'][start]}_frame{d['cid'][start]}_{d['cid'][end-1]}")
    result["command"]=d["command"][start].tolist()
    result["segment_time_s"]=[float(d["t"][start]),float(d["t"][end-1]+.02)]
    result["analysis_time_s"]=[float(d["t"][start+100]),float(d["t"][end-1]+.02)]
    result["settling_excluded_s"]=2
    manual["stable_command_segments"].append(result)
# Report all eight fixed scenarios for the new policy, separately from the matched parking experiment.
evaluation=[]
for path in sorted((experiment/"mujoco_evaluation").glob("*.csv")):
    e=load(path,"new_policy_fixed_scenario")
    mask=e["t"]>=2-1e-9
    r=summarize(e,mask,"new_eval_"+path.stem)
    r["evaluation_name"]=path.stem
    evaluation.append(r)
report={"schema":"ActionSmoothMetricDiagnosisV1","generated_at":datetime.now(SGT).isoformat(),
    "analysis_only":True,"control_dt_s":.02,"action_order":names,"joint_labels":labels,
    "production_formula_file":str(root/"wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/rewards.py"),
    "action_source":"clipped_action, same canonical signals used by the production reward; MuJoCo does not apply training rewards",
    "history_semantics":"Use adjacent t-2,t-1,t actions; all three inside the window and same episode, never cross a reset. Initial two samples excluded.",
    "matched_stationary_window":"time_before_s in [2,10), 400 frames, 398 second-difference samples per policy",
    "window_statistic_note":"One-second 50-sample means are diagnostic only; no ROGER controller is implemented and no threshold is selected.",
    "sources":sources,"matched_stationary":cases,"user_keyboard":manual,"new_policy_fixed_scenarios":evaluation,
    "limits":["No independent human good/bad jitter labels are available; measured costs alone cannot establish an acceptable threshold.",
        "Zero smoothness cost under permanent action clipping does not show healthy control.",
        "Fixed-stand matched comparison and manually commanded segments have different initial states/commands and must not be pooled.",
        "Single-frame and one-second-window quantiles differ; thresholds must match the eventual controller statistic.",
        "Leg/wheel decomposition is diagnostic only; the six-dimensional reward formula is unchanged."]}
report_path=out/"metrics.json"
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
with (out/"frame-metrics.csv").open("w",newline="",encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=all_points[0].keys());writer.writeheader();writer.writerows(all_points)
windows=[w for r in cases+manual["stable_command_segments"]+evaluation for w in r["one_second_windows"]]
with (out/"one-second-windows.csv").open("w",newline="",encoding="utf-8") as f:
    fields=list(dict.fromkeys(k for w in windows for k in w))
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(windows)
locator={"output":str(out),"report":str(report_path),"script":str(root/".git/measure_action_smooth_metric.py")}
(root/".git/action-smooth-metric-analysis.json").write_text(json.dumps(locator,indent=2)+"\n",encoding="utf-8")
compact=lambda r:{k:r[k] for k in ["label","second_difference_samples","frame_statistic","mean_contribution","wheel_share_percent","vx_mae_mps","one_second_window_mean_statistic"]}
print(json.dumps({"output":str(out),"stationary":[compact(r) for r in cases],
    "keyboard":{"frames":manual["frames"],"episodes":manual["episodes"],"markers":len(marker_events),"stable_segments":[{**compact(r),"command":r["command"],"analysis_time_s":r["analysis_time_s"]} for r in manual["stable_command_segments"]]},
    "evaluation":[compact(r) for r in evaluation]},ensure_ascii=False))
