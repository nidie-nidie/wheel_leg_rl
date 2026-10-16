
from pathlib import Path
import csv,json,hashlib,math,statistics,subprocess,sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
root=Path(r"E:\wheel_leg_rl-main")
loc=json.loads((Path(__file__).resolve().parent.parent/"input-identity.json").read_text(encoding="utf-8"))
out=Path(loc["output"])
baseline=json.loads((out/"input-identity.json").read_text(encoding="utf-8"))
report=json.loads((out/"indicators.json").read_text(encoding="utf-8"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
protected_changes=[p for p,h in baseline["protected_files"].items() if sha(p)!=h]
assert not protected_changes,protected_changes
after=subprocess.run(["git","diff","--name-only"],cwd=root,capture_output=True,text=True,check=True).stdout.splitlines()
assert sorted(after)==sorted(baseline["tracked_diff_before"]),(after,baseline["tracked_diff_before"])
for entry in report["sources"]:
    assert sha(entry["path"])==entry["sha256"]
for entry in report["physics_alignment_checks"]:
    assert sha(entry["physics_csv"])==entry["physics_csv_sha256"]
def csvrows(p):
    with Path(p).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
names=["jIJ","jIO","jAB","jAG"]
aligned=csvrows(out/"aligned-frames.csv")
bycase={}
for row in aligned:bycase.setdefault(row["dataset"],[]).append(row)
max_frame_error=0.;max_summary_error=0.;samples=0
for c in report["cases"]:
    raw=csvrows(c["source"]);lookup={(int(r["episode_id"]),int(r["control_id"])):i for i,r in enumerate(raw)}
    rows=bycase[c["label"]]
    assert len(rows)==c["samples"]
    for r in rows:
        i=lookup[(int(r["episode_id"]),int(r["control_id"]))]
        current=raw[i];previous=raw[i-1];older=raw[i-2]
        assert i>=2 and current["episode_id"]==previous["episode_id"]==older["episode_id"]
        assert int(current["control_id"])==int(previous["control_id"])+1==int(older["control_id"])+2
        assert abs(float(current["time_after_s"])-float(current["time_before_s"])-.02)<1e-10
        costs=[]
        for name in names:
            action=[float(v[name+"_action_clipped"]) for v in [current,previous,older]]
            d2=action[0]-2*action[1]+action[2]
            accel=(float(current[name+"_velocity_after_rad_s"])-float(current[name+"_velocity_before_rad_s"]))/.02
            expected={name+"_action":action[0],name+"_action_second_difference":d2,name+"_acceleration_20ms_rad_s2":accel,
                name+"_actual_rad":float(current[name+"_position_after_rad"]),name+"_target_rad":float(current[name+"_position_target_rad"]),
                name+"_velocity_after_rad_s":float(current[name+"_velocity_after_rad_s"])}
            for key,value in expected.items():max_frame_error=max(max_frame_error,abs(float(r[key])-value))
            costs.append(d2*d2)
        for key,value in [("leg_smooth",sum(costs)),("left_smooth",sum(costs[:2])),("right_smooth",sum(costs[2:])),
            ("command_vx_mps",float(current["command_vx_mps"])),("actual_vx_mps",float(current["vx_mps"]))]:
            max_frame_error=max(max_frame_error,abs(float(r[key])-value))
        samples+=1
    values={n:math.sqrt(statistics.mean(float(r[n+"_acceleration_20ms_rad_s2"])**2 for r in rows)) for n in names}
    for n in names:max_summary_error=max(max_summary_error,abs(values[n]-c["joint_metrics"][n]["actual_acceleration_20ms"]["rms"]))
    for key,field in [("left_smooth_mean","left_smooth"),("right_smooth_mean","right_smooth")]:
        max_summary_error=max(max_summary_error,abs(statistics.mean(float(r[field]) for r in rows)-c[key]))
    ix=[lookup[(int(r["episode_id"]),int(r["control_id"]))] for r in rows]
    for n in names[:2]:
        velocity=np.array([float(raw[i][n+"_velocity_before_rad_s"]) for i in ix])
        action=np.array([float(raw[i][n+"_action_clipped"]) for i in ix])
        for values,field in [(velocity,"actual_velocity_dominant_frequency_hz"),(action,"command_action_dominant_frequency_hz")]:
            power=np.abs(np.fft.rfft((values-values.mean())*np.hanning(len(values))))**2
            power[0]=0
            peak=float(np.fft.rfftfreq(len(values),.02)[int(np.argmax(power))])
            assert abs(peak-c["joint_metrics"][n][field])<1e-12
assert max_frame_error<1e-10 and max_summary_error<1e-10
replays=json.loads((out/"supplemental-capture-verification.json").read_text(encoding="utf-8"))
replay_checks=[]
allnames=names+["jwheel_left","jwheel_right"]
for entry in replays:
    assert sha(entry["reference"])==entry["reference_sha256"]
    captured=csvrows(entry["new_telemetry"]);reference=csvrows(entry["reference"])
    assert len(captured)==len(reference)==500
    errors={"raw_action":0.,"vx":0.,"yaw_rate":0.,"base_height":0.,"command":0.,"clipped_action":0.}
    practice=entry["scenario"]=="forward_stop_reverse"
    for a,b in zip(captured,reference):
        for i,n in enumerate(allnames):
            original=float(b["raw_action_"+str(i)])
            errors["raw_action"]=max(errors["raw_action"],abs(float(a[n+"_action_raw"])-original))
            expected=max(-1.,min(1.,original)) if practice else float(b["clipped_action_"+str(i)])
            errors["clipped_action"]=max(errors["clipped_action"],abs(float(a[n+"_action_clipped"])-expected))
        for key,kind in [("vx_mps","vx"),("yaw_rate_rad_s","yaw_rate"),("base_height_m","base_height")]:
            errors[kind]=max(errors[kind],abs(float(a[key])-float(b[key])))
        for akey,bkey in [("command_vx_mps","target_vx_mps"),("command_yaw_rad_s","target_yaw_rate_rad_s"),
            ("command_height_m","target_height_m" if practice else "target_base_height_m")]:
            errors["command"]=max(errors["command"],abs(float(a[akey])-float(b[bkey])))
    assert max(errors.values())==0.
    entry["max_abs_difference_vs_existing_evaluation"]=errors
    entry["clipped_action_reference"]="clip(original raw action, -1, +1)" if practice else "original logged clipped_action columns"
    replay_checks.append({"scenario":entry["scenario"],"frames":500,"errors":errors,"clipped_action_reference":entry["clipped_action_reference"]})
(out/"supplemental-capture-verification.json").write_text(json.dumps(replays,indent=2)+"\n",encoding="utf-8")
result={"protected_file_count":len(baseline["protected_files"]),"protected_file_hashes_unchanged":True,
    "tracked_changed_file_list_unchanged":True,"source_control_csv_hashes_verified":len(report["sources"]),
    "source_physics_csv_hashes_verified":len(report["physics_alignment_checks"]),
    "independently_recomputed_aligned_frames":samples,"max_frame_arithmetic_error":max_frame_error,
    "max_summary_rms_mean_error":max_summary_error,"left_joint_spectral_peaks_recomputed":True,
    "replay_checks":replay_checks,"visual_qa":"Seven plots reviewed; four-joint trace plots use equal position spans, velocity scales and acceleration scales.",
    "scope":"Diagnostic artifacts only; no reward, policy, runtime, model, architecture or training changes."}
(out/"verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False))
