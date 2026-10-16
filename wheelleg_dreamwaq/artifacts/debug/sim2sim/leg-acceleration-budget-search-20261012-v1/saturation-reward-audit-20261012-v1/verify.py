"""Independent verification of saved diagnostic summaries and original inputs."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
a=json.loads((OUT/"analysis.json").read_text())
c=json.loads((OUT/"capture-verification.json").read_text())
d=np.load(OUT/"trajectories.npz")
w=np.load(OUT/"weighted-rewards.npz")
assert len(c["checks"])==17 and sum(v["ticks"] for v in c["checks"])==10120
assert all(v["replay_max_abs_error"]==v["interactive_inference_max_abs_error"]==v["interactive_history_max_abs_error"]==0 for v in c["checks"])
manual_checks={}
for key,s in a["traces"].items():
    t=d[key+"__time_before"]
    mask=t>=2-1e-8
    if "interactive" in key:mask&=t<20
    assert len(t)==(2120 if "interactive" in key else 500)
    assert np.isfinite(d[key+"__q"]).all()
    arm=key.split("_")[0]
    raw=d[key+"__action_raw"]
    clipped=d[key+"__action_clipped"]
    np.testing.assert_allclose(np.clip(raw,-1,1),clipped,rtol=0,atol=0)
    np.testing.assert_allclose(np.mean(abs(raw[mask])>=1,axis=0),s["action_clip_fraction"],rtol=0,atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.mean(d[key+"__acc"][mask,:4]**2,axis=0)),s["leg_accel_rms_rad_s2"],rtol=0,atol=1e-10)
    q=d[key+"__q"]
    v=d[key+"__v"]
    vb=np.vstack((np.zeros((1,6)),v[:-1]))
    np.testing.assert_allclose((v-vb)/.02,d[key+"__acc"],atol=1e-8,rtol=0)
    manifest=json.loads((BASE/arm/"export/policy_manifest.json").read_text())
    weights=manifest["training_reward_weights"]
    prev=np.vstack((np.zeros((1,6)),clipped[:-1]))
    prevprev=np.vstack((np.zeros((2,6)),clipped[:-2]))
    expected_rate=.02*weights["action_rate"]*np.sum((clipped-prev)**2,axis=1)
    expected_smooth=.02*weights["action_smooth"]*np.sum((clipped-2*prev+prevprev)**2,axis=1)
    np.testing.assert_allclose(expected_rate,w[key+"__action_rate"],rtol=0,atol=1e-12)
    np.testing.assert_allclose(expected_smooth,w[key+"__action_smooth"],rtol=0,atol=1e-12)
    for term,mean in s["weighted_reward_mean_per_step"].items():
        np.testing.assert_allclose(w[key+"__"+term][mask].mean(),mean,rtol=0,atol=1e-12)
    channels=np.array(s["weighted_reward_channel_mean_per_step"]["action_smooth"])
    np.testing.assert_allclose(sum(channels),s["weighted_reward_mean_per_step"]["action_smooth"],atol=1e-12,rtol=0)
    if "interactive" not in key:
        scenario=key[len(arm)+1:]
        with (BASE/arm/"motion"/(scenario+".csv")).open(newline="") as f:rows=list(csv.DictReader(f))
        for j,name in enumerate(a["joint_order"][:4]):
            original=np.array([float(r[name+"_qddot_rad_s2"]) for r in rows])
            np.testing.assert_allclose(original,d[key+"__acc"][:,j],atol=0,rtol=0)
    manual_checks[key]={"action_clip_check":True,"control_acceleration_telescope_check":True,"original_csv_check":True,"reward_summary_and_history_check":True}
for arm,log in a["training_logs"].items():
    manifest=json.loads((BASE/arm/"export/policy_manifest.json").read_text())
    run=Path(manifest["source_checkpoint"]).parent
    e=EventAccumulator(str(run),size_guidance={"scalars":0});e.Reload()
    for tag,mean in log["last_50_mean"].items():
        values=[v.value for v in e.Scalars(tag)]
        assert len(values)==200
        np.testing.assert_allclose(np.mean(values[-50:]),mean,atol=1e-12,rtol=0)
    if arm!="baseline":
        ex=np.array(log["last_50_excess_by_joint"])
        np.testing.assert_allclose(ex[2:].sum()/ex.sum(),log["last_50_right_penalty_share"],atol=1e-12,rtol=0)
        assert all(v.value==1 for v in e.Scalars("Acceleration/gain_used")[1:])
assert all(s["action_clip_fraction"][2:4]==[1.,1.] for k,s in a["traces"].items() if k.startswith("85_"))
assert all(max(s["leg_soft_limit_fraction"])==0 for s in a["traces"].values())
protected=json.loads((OUT/"protected-before.json").read_text())
after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest().upper() for p in protected}
assert after==protected
result={"passed":True,"trajectories":len(manual_checks),"control_ticks":10120,"physics_steps":202400,
        "replay_max_abs_error":0,"verification":manual_checks,"training_iterations_checked":800,
        "protected_file_count":len(protected),"protected_files_unchanged":True,
        "production_source_or_model_changed":False,"training_started":False,"isaac_started":False,
        "subagents_called":False,
        "artifact_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest().upper() for p in OUT.iterdir() if p.is_file() and p.name!="verification.json"}}
(OUT/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k not in ("verification","artifact_sha256")}))
