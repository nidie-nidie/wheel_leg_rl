"""Independent acceptance and immutability checks for this diagnostic."""
from pathlib import Path
from datetime import datetime
import json,hashlib,sys,subprocess
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=Path(r"E:\wheel_leg_rl-main")
PROJECT=ROOT/"wheelleg_dreamwaq"
TRIAL=PROJECT/"artifacts/phase2_dreamwaq/leg-reversal-trial-20261011-v1"
sys.path.insert(0,str(PROJECT/"scripts"))
from run_training_suite import _assert_suite_source_fingerprint

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()

def main():
    identity=read(OUT/"input-identity.json")
    _assert_suite_source_fingerprint(identity["source_fingerprint"])
    protected=identity["protected_files"]
    old=read(TRIAL/"baseline.json")
    old_protected={**old["protected"],**old["preexisting_edit_hashes"]}
    for name,expected in {**protected,**old_protected}.items():
        assert sha(ROOT/name)==expected,name
    capture=read(OUT/"capture-verification.json")
    assert capture["completed"] and sum(x["frames"] for x in capture["captures"])==4500
    result=read(OUT/"results.json")
    assert result["shared_standardized_noise_sha256"]==sha(OUT/"standardized-noise.npz")
    assert result["verification"]["independent_counter_max_error"]==0.
    assert result["verification"]["native_export_mean_max_error"]==0.
    original=read(TRIAL/"results.json")
    cross_max={"mean_counter":0.0,"velocity_rms":0.0,"acceleration_rms":0.0}
    for x in capture["captures"]:
        assert max(x["max_errors"].values())==0.,x
        assert sha(x["capture"])==x["capture_sha256"]
        assert sha(x["checkpoint"])==x["checkpoint_sha256"]
        assert sha(x["reference"])==x["reference_sha256"]
        d=next(d for d in result["comparisons"] if d["arm"]==x["arm"] and d["scenario"]==x["scenario"])
        old_result=next(d for d in original["comparison"] if d["name"]==x["arm"])["diagnostic"][x["scenario"]]
        for name,new_values,old_values in [
            ("mean_counter",d["mean_only"]["target_rate_mean_per_s"],old_result["leg_target_reversal_rate_per_s"]),
            ("velocity_rms",d["actual"]["velocity_rms_rad_s"],old_result["leg_velocity_rms_rad_s"]),
            ("acceleration_rms",d["actual"]["acceleration_rms_rad_s2"],old_result["leg_acceleration_rms_rad_s2"])]:
            error=float(np.abs(np.array(new_values)-np.array(old_values)).max())
            cross_max[name]=max(cross_max[name],error)
            assert error<1e-10,(name,error)
        assert all(v==0 for v in d["held_input"]["mean_only"]["target_rate_mean_per_s"])
        assert all(v>24 for v in d["held_input"]["with_noise"]["target_rate_mean_per_s"][:2])
    forward=next(d for d in result["comparisons"] if d["arm"]=="budget12" and d["scenario"]=="forward")
    assert all(v<12 for v in forward["mean_only"]["target_rate_mean_per_s"][:2])
    assert all(v>24 for v in forward["same_histories_with_noise"]["target_rate_mean_per_s"][:2])
    synthetic=result["same_frequency_different_amplitude"]
    assert all(v==10 for v in synthetic[0]["metrics"]["target_rate_mean_per_s"])
    assert synthetic[0]["metrics"]["target_rate_mean_per_s"]==synthetic[1]["metrics"]["target_rate_mean_per_s"]
    for budget in ("12","8"):
        assert synthetic[0]["metrics"]["budgets"][budget]["full_gain_penalty_rate_per_s"]==synthetic[1]["metrics"]["budgets"][budget]["full_gain_penalty_rate_per_s"]
    for g in result["training_logs"]:
        if g["name"]!="baseline":
            assert g["events"]["Reversal/gain_used"]["count"]==200
            assert g["events"]["Reversal/gain_used"]["fraction_at_one"]==.985
    before_tracked={line[3:] for line in identity["original_git_status"].splitlines() if line[:2]!="??"}
    now_tracked=set(subprocess.check_output(["git","diff","--name-only"],cwd=ROOT,text=True,encoding="utf-8").splitlines())
    assert now_tracked==before_tracked,(now_tracked,before_tracked)
    subprocess.run(["git","diff","--check"],cwd=ROOT,check=True)
    files={p.name:sha(p) for p in OUT.glob("*") if p.is_file() and p.name!="verification.json"}
    verification={"status":"passed","verified_at":datetime.now().astimezone().isoformat(),
        "source_fingerprint_unchanged":True,"source_fingerprint":identity["source_fingerprint"],
        "diagnostic_snapshot_protected_files":len(protected),"prior_trial_protected_files":len(old_protected),
        "all_protected_hashes_unchanged":True,"preexisting_tracked_changes_preserved":True,
        "exact_replayed_frames":4500,"replay_max_error":0.0,"counter":result["verification"],
        "cross_check_against_prior_results_max_error":cross_max,
        "held_input_mean_zero_events_and_noise_above_budget_verified":True,
        "synthetic_equal_counts_and_equal_cost_different_amplitude_verified":True,
        "file_hashes":files,"training_performed":False,"production_parameters_modified":False,
        "agents_invoked":False,"result_scope":"conditional fixed-input action sampling; not full real-training causal attribution"}
    (OUT/"verification.json").write_bytes((json.dumps(verification,indent=2,sort_keys=True)+"\n").encode("utf-8"))
    print(json.dumps({k:verification[k] for k in ("status","diagnostic_snapshot_protected_files","prior_trial_protected_files","exact_replayed_frames","counter","cross_check_against_prior_results_max_error")},indent=2))
if __name__=="__main__": main()
