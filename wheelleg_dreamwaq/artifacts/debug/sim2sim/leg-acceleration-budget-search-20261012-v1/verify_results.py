from pathlib import Path
import json
import hashlib
import csv
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()

def main():
    identity=json.loads((OUT/"experiment-identity.json").read_text())
    for rel,expected in identity["source_files"].items():
        assert sha(ROOT/rel)==expected, rel
    assert sha(ROOT.parent/"docs/2026-10-03-wheelleg-dreamwaq-architecture.md")==identity["architecture_v0_28_sha256"]
    records={}
    initial_hashes=[]
    for arm in ("baseline","80","75","70","85","90"):
        f=OUT/(arm+"-result.json")
        if not f.exists(): continue
        arm_info=json.loads(f.read_text())
        ts=arm_info["training_summary"]
        assert ts["completed_iterations"]==200 and ts["iterations_executed"]==200
        assert ts["starting_completed_iterations"]==0
        initial_hashes.append(ts["initial_model_tensor_sha256"])
        assert arm_info["source_checkpoint_sha256"]==identity["source_checkpoint_sha256"]
        checkpoint=torch.load(ts["final_checkpoint"],map_location="cpu",weights_only=False)
        state=checkpoint["infos"].get("leg_acceleration_gain_state")
        if arm=="baseline": assert state is None
        else:
            assert state["completed_rollouts"]==200
            assert state["contract"]["budget_fraction"]==int(arm)/100
            assert state["gain"]==ts["leg_acceleration_gain_state"]["gain"]
        package=Path(arm_info["export"])
        pm=json.loads((package/"policy_manifest.json").read_text())
        assert pm["source_checkpoint_sha256"]==sha(Path(ts["final_checkpoint"]))
        assert pm["source_seed"]==41882943
        policy=torch.jit.load(str(package/"actor.ts"),map_location="cpu").eval()
        golden=torch.load(package/"golden_vectors.pt",map_location="cpu",weights_only=False)
        max_errors=[]
        with torch.inference_mode():
            for size in (1,3,32):
                error=float((policy(golden["history"][:size])-golden["expected_action_mean"][:size]).abs().max())
                assert error<=1.e-5
                max_errors.append(error)
        motion=json.loads((OUT/arm/"motion/summary.json").read_text())
        assert len(motion["scenarios"])==8
        assert motion["policy_sha256"]==sha(package/"actor.ts")
        assert motion["budget_fraction"]==(None if arm=="baseline" else int(arm)/100)
        max_difference=0.
        for s in motion["scenarios"]:
            path=OUT/arm/"motion"/(s["scenario"]+".csv")
            assert sha(path)==motion["scenario_files"][s["scenario"]]["sha256"]
            with path.open(newline="") as stream: rows=list(csv.DictReader(stream))
            rows=[r for r in rows if r["survived"]=="1"]
            if not rows: continue
            velocity=np.array([[float(r[j+"_qdot_rad_s"]) for j in ("jIJ","jIO","jAB","jAG")] for r in rows])
            accel=np.array([[float(r[j+"_qddot_rad_s2"]) for j in ("jIJ","jIO","jAB","jAG")] for r in rows])
            diff=float(np.max(np.abs(np.diff(velocity,axis=0)/.02-accel[1:]))) if len(rows)>1 else 0.
            assert diff<1.e-8
            max_difference=max(max_difference,diff)
            steady=[r for r in rows if int(r["tick"])>100]
            if len(steady)>=50:
                values=np.array([[float(r[j+"_qddot_rad_s2"]) for j in ("jIJ","jIO","jAB","jAG")] for r in steady])
                rms=np.array([np.sqrt(np.mean(values[i:i+50]**2,axis=0)) for i in range(0,len(values)-49,50)])
                assert np.allclose(rms.mean(axis=0),s["actual_acceleration_rms_mean_rad_s2"],rtol=0,atol=1.e-9)
                assert np.allclose(np.percentile(rms,90,axis=0),s["actual_acceleration_rms_p90_rad_s2"],rtol=0,atol=1.e-9)
                mae=np.mean([abs(float(r["vx_mps"])-s["command"][0]) for r in steady])
                assert abs(mae-s["steady_vx_mae_mps"])<1.e-12
        practice=json.loads((OUT/arm/"command_practice/summary.json").read_text())
        assert practice["policy_sha256"]==motion["policy_sha256"]
        records[arm]={"training200":True,"golden_max_errors_1_3_32":max_errors,
                      "actual_acceleration_csv_difference_max":max_difference,
                      "motion_report_sha256":sha(OUT/arm/"motion/summary.json"),
                      "gain":None if state is None else state["gain"]}
    assert len(set(initial_hashes))==1
    report={"source_files_unchanged":True,"same_initial_network_all_arms":True,"arms":records}
    (OUT/"verification.json").write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()

