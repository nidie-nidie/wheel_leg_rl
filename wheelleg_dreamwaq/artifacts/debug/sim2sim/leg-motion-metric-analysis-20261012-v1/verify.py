"""Independent scalar checks, previous-result agreement, input immutability."""
from pathlib import Path
from datetime import datetime
import csv,hashlib,json,math,subprocess,sys
import numpy as np
from PIL import Image

ROOT=Path(r"E:\wheel_leg_rl-main")
PROJECT=ROOT/"wheelleg_dreamwaq"
OUT=Path(__file__).resolve().parent
PREVIOUS=OUT.parent/"reversal-noise-diagnosis-20261011-v1"
JOINTS=("jIJ","jIO","jAB","jAG")
DT=.02
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def rms(x): return math.sqrt(math.fsum(v*v for v in x)/len(x))
def check(got,want,key,errors):
    error=abs(got-want); errors[key]=max(errors.get(key,0),error)
    assert error<1e-9,(key,got,want,error)

def main():
    identity=read(OUT/"input-identity.json")
    for name,expected in {**identity["inputs"],**identity["protected_files"]}.items():
        assert sha(ROOT/name)==expected,name
    sys.path.insert(0,str(PROJECT/"scripts"))
    from run_training_suite import _assert_suite_source_fingerprint
    _assert_suite_source_fingerprint(identity["source_fingerprint"])
    captures=read(PREVIOUS/"capture-verification.json")["captures"]
    previous=read(PREVIOUS/"results.json")["comparisons"]
    results=read(OUT/"results.json")
    errors={}; trajectory_count=0; joint_statistics=0
    with (OUT/"aligned-data.csv").open(encoding="utf-8",newline="") as f: aligned=list(csv.DictReader(f))
    assert len(aligned)==3600
    for record in captures:
        d=dict(np.load(record["capture"]))
        with Path(record["reference"]).open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
        result=next(x for x in results["trajectories"] if x["arm"]==record["arm"] and x["scenario"]==record["scenario"])
        old=next(x for x in previous if x["arm"]==record["arm"] and x["scenario"]==record["scenario"])
        subset=[x for x in aligned if x["arm"]==record["arm"] and x["scenario"]==record["scenario"]]
        assert len(subset)==400
        for i in range(500):
            for j in range(4):
                expected=max(record["target_lower"][j],min(record["target_upper"][j],
                    record["q_nominal"][j]+record["leg_action_scale"]*max(-1,min(1,float(d["mean_action"][i,j])))))
                check(float(d["leg_target"][i,j]),expected,"original_target_mapping",errors)
        for j,name in enumerate(JOINTS):
            m=result["joints"][j]; assert m["joint"]==name
            vec={key:[] for key in ("target_increment","target_second","acceleration","velocity","tracking_error")}
            for offset,i in enumerate(range(100,500)):
                target=float(rows[i][f"leg_target_{j}_rad"])
                prev_target=float(rows[i-1][f"leg_target_{j}_rad"])
                prevprev_target=float(rows[i-2][f"leg_target_{j}_rad"])
                v=float(rows[i][f"leg_velocity_{j}_rad_s"])
                prev_v=float(rows[i-1][f"leg_velocity_{j}_rad_s"])
                for key,value in (("target_increment",target-prev_target),
                                  ("target_second",target-2*prev_target+prevprev_target),
                                  ("acceleration",(v-prev_v)/DT),("velocity",v),
                                  ("tracking_error",target-float(d["q_after"][i,j]))):
                    vec[key].append(value)
                    check(float(subset[offset][name+"_"+key]),value,"aligned_"+key,errors)
                check(float(subset[offset][name+"_position"]),float(d["q_after"][i,j]),"aligned_position",errors)
                check(float(subset[offset]["time_before_s"]),float(rows[i]["time_s"])-DT,"aligned_time",errors)
                check(float(subset[offset]["time_after_s"]),float(rows[i]["time_s"]),"aligned_time",errors)
            for key,values in vec.items():
                check(m[key+"_rms"],rms(values),key+"_rms",errors)
                check(m[key+"_p95_abs"],float(np.percentile([abs(v) for v in values],95)),key+"_p95",errors)
                check(m[key+"_max_abs"],max(abs(v) for v in values),key+"_max",errors)
                joint_statistics+=3
            check(m["acceleration_rms"],old["actual"]["acceleration_rms_rad_s2"][j],"previous_acceleration",errors)
            check(m["velocity_rms"],old["actual"]["velocity_rms_rad_s"][j],"previous_velocity",errors)
            for window in range(8):
                w=result["windows"][window]["joints"][j]
                for key,values in vec.items():
                    check(w[key+"_rms"],rms(values[window*50:(window+1)*50]),"window_"+key,errors)
                    joint_statistics+=1
        for key,expected in (
            ("vx_mae_mps",math.fsum(abs(float(r["vx_mps"])-float(r["target_vx_mps"])) for r in rows[100:])/400),
            ("yaw_rate_mae_rad_s",math.fsum(abs(float(r["yaw_rate_rad_s"])-float(r["target_yaw_rate_rad_s"])) for r in rows[100:])/400),
            ("height_mae_m",math.fsum(abs(float(r["base_height_m"])-float(r["target_base_height_m"])) for r in rows[100:])/400)):
            check(result["vehicle"][key],expected,"vehicle_"+key,errors)
        for j in range(6):
            acc=[(float(d["v_after"][i,j])-float(d["v_before"][i,j]))/DT for i in range(100,500)]
            acts=[max(-1,min(1,float(d["mean_action"][i,j]))) for i in range(500)]
            second=[acts[i]-2*acts[i-1]+acts[i-2] for i in range(100,500)]
            e=result["existing_six_dimension_cost"]
            check(e["acceleration_mse_per_dimension"][j],math.fsum(x*x for x in acc)/400,"existing_acceleration_mse",errors)
            check(e["action_second_mse_per_dimension"][j],math.fsum(x*x for x in second)/400,"existing_action_second_mse",errors)
        for column in ("pitch_rad","roll_rad","phi0_left_rad","phi0_right_rad","l0_left_m","l0_right_m","base_height_m","vx_mps"):
            for offset,row in enumerate(rows[100:]):
                check(float(subset[offset][column]),float(row[column]),"aligned_vehicle",errors)
        sampling=result["sampling"]
        assert all(v<1e-15 for v in sampling["held_mean"]["second_rms_rad"])
        assert all(v>.1 for v in sampling["held_mean_sampled"]["second_rms_rad"][:2])
        assert all(result["joints"][j]["target_second_rms"]==0 for j in (2,3))
        assert all(result["joints"][j]["acceleration_rms"]>0 for j in (2,3))
        trajectory_count+=1
    syn=results["synthetic"]
    for key in ("target_increment_rms","target_second_rms","actual_acceleration_rms"):
        check(syn["large_5hz"][key]/syn["small_5hz"][key],10,"synthetic_amplitude_ratio",errors)
    assert syn["constant_speed"]["actual_acceleration_rms"]<1e-10
    assert syn["constant_speed"]["actual_velocity_rms"]>.01
    assert syn["constant_speed"]["target_second_rms"]<1e-10
    assert syn["small_10hz"]["actual_acceleration_rms"]>syn["small_5hz"]["actual_acceleration_rms"]
    baseline=next(x for x in results["trajectories"] if x["arm"]=="baseline" and x["scenario"]=="nominal_stand")
    n12=next(x for x in results["trajectories"] if x["arm"]=="budget12" and x["scenario"]=="nominal_stand")
    assert n12["joints"][0]["target_second_rms"]<baseline["joints"][0]["target_second_rms"]
    assert n12["joints"][0]["acceleration_rms"]>baseline["joints"][0]["acceleration_rms"]
    assert all(n12["joints"][j]["acceleration_rms"]>baseline["joints"][j]["acceleration_rms"] for j in (0,1))
    now=subprocess.check_output(["git","diff","--name-only"],cwd=ROOT,text=True,encoding="utf-8").splitlines()
    assert sorted(now)==sorted(identity["tracked_changes"]),(now,identity["tracked_changes"])
    subprocess.run(["git","diff","--check"],cwd=ROOT,check=True)
    report=(OUT/"report.md").read_text(encoding="utf-8")
    assert "\ufffd" not in report and "\u5b9e\u9645\u89d2\u52a0\u901f\u5ea6" in report
    plot_sizes={}
    for name in ("parking-metric-comparison.png","aligned-joints-nominal_stand.png","aligned-joints-forward.png","aligned-joints-reverse.png","vehicle-and-virtual-leg.png"):
        with Image.open(OUT/name) as img:
            img.verify()
        with Image.open(OUT/name) as img: plot_sizes[name]=list(img.size)
    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!="verification.json"}
    verification={"status":"passed","verified_at":datetime.now().astimezone().isoformat(),
        "trajectories":trajectory_count,"analyzed_frames":len(aligned),"joint_samples":len(aligned)*4,
        "scalar_joint_statistics_checked":joint_statistics,"maximum_errors":errors,
        "source_fingerprint_unchanged":True,"protected_file_count":len(identity["protected_files"]),
        "input_file_count":len(identity["inputs"]),"all_hashes_unchanged":True,"preexisting_tracked_changes_preserved":True,
        "synthetic_amplitude_and_constant_speed_checks":True,"actual_motion_with_constant_right_target_checks":True,
        "target_metric_decreases_while_actual_acceleration_increases_checked":True,
        "sampling_checks":"Fixed-history only; no sampled physical acceleration was synthesized",
        "plots":plot_sizes,"file_hashes":files,"training_performed":False,"simulation_performed":False,
        "production_files_modified":False,"agents_invoked":False}
    (OUT/"verification.json").write_bytes((json.dumps(verification,indent=2,sort_keys=True)+"\n").encode("utf-8"))
    print(json.dumps({k:verification[k] for k in ("status","trajectories","analyzed_frames","joint_samples",
        "scalar_joint_statistics_checked","maximum_errors","protected_file_count","input_file_count")},indent=2))
if __name__=="__main__": main()
