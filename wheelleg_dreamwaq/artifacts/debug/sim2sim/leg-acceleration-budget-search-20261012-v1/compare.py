from pathlib import Path
import json
import hashlib
import numpy as np

OUT=Path(__file__).resolve().parent
PRIMARY=("nominal_stand","forward","reverse")
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()

def read_arm(arm):
    path=OUT/arm/"motion/summary.json"
    if not path.is_file():
        return None
    motion=json.loads(path.read_text())
    practice=json.loads((OUT/arm/"command_practice/summary.json").read_text())
    scenarios={s["scenario"]:s for s in motion["scenarios"]}
    metrics_valid=all(s["metric_valid"] for s in scenarios.values())
    primary_joint_mean=np.mean([scenarios[n]["actual_acceleration_rms_mean_rad_s2"] for n in PRIMARY],axis=0).tolist() if metrics_valid else None
    primary_rms=float(np.mean(primary_joint_mean)) if metrics_valid else None
    steady={n:scenarios[n]["steady_vx_mae_mps"] for n in PRIMARY}
    return {"arm":arm,"motion_path":str(path),"motion_sha256":digest(path),
            "practice_sha256":digest(OUT/arm/"command_practice/summary.json"),
            "primary_joint_rms_means_rad_s2":primary_joint_mean,
            "primary_aggregate_rms_rad_s2":primary_rms,
            "primary_steady_vx_mae_mps":steady,"metric_valid":metrics_valid,
            "all_static_survived":all(s["survived"] for s in scenarios.values()),
            "practice_all_accepted":practice["performance_accepted"],
            "practice_scenarios":[{"name":s["name"],"completed":s["completed"],
                "performance_accepted":s["performance_accepted"],
                "failed_stages":[{"index":t["stage_index"],"failures":t["failures"],
                    "vx_mae":t["steady_vx_mae_mps"],"yaw_mae":t["steady_yaw_mae_rad_s"],
                    "response_s":t["held_error_band_confirmation_time_s"]}
                    for t in s["stages"] if not t["performance_accepted"]]}
                for s in practice["scenarios"]],
            "scenarios":scenarios}

def assess(arm,baseline):
    r=read_arm(arm)
    if r is None: return None
    valid=r["metric_valid"]
    change=(1-r["primary_aggregate_rms_rad_s2"]/baseline["primary_aggregate_rms_rad_s2"]) if valid else None
    static_speed=valid and all(
        r["primary_steady_vx_mae_mps"][n] is not None and
        r["primary_steady_vx_mae_mps"][n] <= (0.10 if n=="nominal_stand" else 0.15) and
        r["primary_steady_vx_mae_mps"][n] <= baseline["primary_steady_vx_mae_mps"][n]+max(
            0.1*baseline["primary_steady_vx_mae_mps"][n],0.03)
        for n in PRIMARY)
    saturation_deltas={n:{key:(s[key]-baseline["scenarios"][n][key]) if s[key] is not None else None
                            for key in ("action_saturation_fraction","effort_saturation_fraction")}
                       for n,s in r["scenarios"].items()}
    saturation_pass=all(v is not None and v<=0.05 for d in saturation_deltas.values() for v in d.values())
    control_pass=r["all_static_survived"] and static_speed and r["practice_all_accepted"] and saturation_pass
    r.update({"rms_reduction_fraction":change,"jitter_improved_at_least_5pct":valid and change>=.05,
              "static_speed_pass":static_speed,"saturation_pass":saturation_pass,
              "saturation_deltas":saturation_deltas,
              "control_pass":control_pass,"accepted":control_pass and valid and change>=.05,
              "per_joint_reduction_fraction":[1-v/b for v,b in zip(r["primary_joint_rms_means_rad_s2"],baseline["primary_joint_rms_means_rad_s2"])] if valid else None})
    return r

def main():
    baseline=read_arm("baseline")
    candidates=[assess(a,baseline) for a in ("80","75","70","85","90")]
    candidates=[r for r in candidates if r is not None]
    report={"schema_version":"LegAccelerationBudgetComparisonV1","single_seed":41882943,
            "iterations_each":200,"primary_scenarios":PRIMARY,"joint_order":["jIJ","jIO","jAB","jAG"],
            "baseline":baseline,"candidates":candidates,
            "accepted_budgets":[r["arm"] for r in candidates if r["accepted"]],
            "all_budgets_completed":len(candidates)==5}
    (OUT/"comparison.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
    print("BASELINE rms",baseline["primary_aggregate_rms_rad_s2"],"steady vx",baseline["primary_steady_vx_mae_mps"],
          "practice",baseline["practice_all_accepted"])
    for r in candidates:
        print("BUDGET",r["arm"],"RMS reduction",r["rms_reduction_fraction"],"speed",r["primary_steady_vx_mae_mps"],
              "survived",r["all_static_survived"],"speedpass",r["static_speed_pass"],"practice",r["practice_all_accepted"],
              "saturationpass",r["saturation_pass"],"ACCEPTED",r["accepted"])

if __name__=="__main__":
    main()

