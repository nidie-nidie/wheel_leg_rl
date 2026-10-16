"""Independent checks for diagnostic outputs and protected source files."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
EXP=OUT.parent
PROJECT=EXP.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
a=json.loads((OUT/"analysis.json").read_text())
s=json.loads((OUT/"snapshots.json").read_text())
frozen=json.loads((PROJECT/"artifacts/phase2_dreamwaq/split-action-smooth-20261013-v1/frozen-source.json").read_text())
assert all(sha(PROJECT/p)==h for p,h in frozen.items())
assert all(sha(Path(p))==h for p,h in a["input_sha256"].items())
assert all(sha(Path(p))==h for p,h in s["checkpoint_inputs"].items())
checks=[]
for label in ["baseline","leg02"]:
    for scene in ["nominal_stand","forward","reverse"]:
        with np.load(EXP/label/"physics1ms"/(scene+".npz")) as z:
            raw=z["raw_action"].astype(np.float64)
            clipped=np.minimum(np.maximum(raw,-1),1)
            d2=clipped[100:]-2*clipped[99:-1]+clipped[98:-2]
            expected=np.mean(d2*d2,axis=0)
            x=a["scenarios"][label][scene]
            np.testing.assert_allclose(expected,x["second_difference_square_mean"],rtol=2e-6,atol=1e-7)
            original=-.01*.02*np.sum(d2*d2,axis=1)
            split=-.02*.02*np.sum(d2[:,:4]**2,axis=1)-.01*.02*np.sum(d2[:,4:]**2,axis=1)
            np.testing.assert_allclose(split-original,-.01*.02*np.sum(d2[:,:4]**2,axis=1),rtol=1e-10,atol=1e-12)
            fixed=[2,3] if label=="baseline" else [0,1]
            assert np.all(raw[100:,fixed]<=-1)
            assert np.all(np.ptp(z["control"][100:,9:13][:,fixed],axis=0)==0)
            assert all(x["target_peak_hz"][j] is None for j in fixed)
            assert np.all(expected[fixed]==0)
            p=z["physics"]; acceleration=p[p[:,0]>=100,10:14]
            np.testing.assert_allclose(np.sqrt((acceleration**2).mean(0)),x["actual_acceleration_pooled_rms_1ms"],rtol=1e-12)
        checks.append(label+":"+scene)
with np.load(OUT/"snapshot-trajectories.npz") as z:
    for r in s["results"]:
        for scene,x in r["scenes"].items():
            key=r["label"]+"_"+str(r["completed_iterations"])+"_"+scene
            raw=z[key+"__raw"]; torque=z[key+"__torque"]
            assert len(torque)==20*len(raw)==x["original_torque_call_count"]
            if r["completed_iterations"]==1000:
                with np.load(EXP/r["label"]/"physics1ms"/(scene+".npz")) as prior:
                    np.testing.assert_array_equal(raw,prior["raw_action"])
                    np.testing.assert_array_equal(z[key+"__control"][:,:5],prior["control"][:,:5])
                if r["label"]=="leg02": assert all(v==0 for v in x["effort_saturation_per_joint"][:4])
    r=next(r for r in s["results"] if r["label"]=="leg02" and r["completed_iterations"]==201)
    assert r["scenes"]["nominal_stand"]["negative_clip_fraction"][:2]==[1,1]
# Checkpoint identities, reset cache, and all non-reward task/algorithm/export contracts.
runs={label:Path(a["training"][label]["run_dir"]) for label in ["baseline","leg02"]}
m={label:json.loads((run/"run_manifest.json").read_text()) for label,run in runs.items()}
expected=json.loads(json.dumps(m["baseline"]["base_task_contract"]))
actual=json.loads(json.dumps(m["leg02"]["base_task_contract"]))
expected.pop("contract_hash");actual.pop("contract_hash")
expected["schemas"]["reward"]=actual["schemas"]["reward"]
expected["task"]["reward_weights"]=actual["task"]["reward_weights"]
assert expected==actual
for key in ["dreamwaq_algorithm_contract","dreamwaq_export_contract"]: assert m["baseline"][key]==m["leg02"][key]
reset={label:torch.load(run/"closed_chain_reset_cache.pt",map_location="cpu",weights_only=False) for label,run in runs.items()}
assert sha(runs["baseline"]/"closed_chain_reset_cache.pt")==sha(runs["leg02"]/"closed_chain_reset_cache.pt")
for key in ["q_reset_projected_env","root_height_offset_env"]: assert torch.equal(reset["baseline"][key],reset["leg02"][key])
result={"schema":"SplitSmoothFailureVerificationV1","passed":True,"offline_trajectories_verified":checks,
        "snapshot_scenes_verified":sum(len(r["scenes"]) for r in s["results"]),
        "final_replay_comparison_coverage":["raw_action","time","four_controlled_joint_positions"],
        "frozen_file_hashes_verified":len(frozen),"input_hashes_verified":len(a["input_sha256"]),
        "checkpoint_hashes_verified":len(s["checkpoint_inputs"]),
        "reset_caches_exact":True,"non_reward_task_algorithm_export_contracts_equal":True,
        "training_started":False,"isaac_started":False}
(OUT/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))

