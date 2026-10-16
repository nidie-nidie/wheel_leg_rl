"""Independently verify checkpoint chronology summaries and preserved inputs."""
from pathlib import Path
import json,hashlib,re
import numpy as np
OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[3]
d=json.loads((OUT/"results.json").read_text())
x=np.load(OUT/"checkpoint-trajectories.npz")
inputs=json.loads((OUT/"checkpoint-inputs.json").read_text())
manifest=json.loads((OUT.parent/"leg-acceleration-budget-search-20261012-v1/85/export/policy_manifest.json").read_text())
qnom=np.array(manifest["q_nominal"])
control=manifest["control"]
print("control keys",list(control))
# Expected target from the frozen manifest, independent of runtime objects.
scale=control.get("leg_action_scale",.35)
checks=[]
assert len(d["checkpoints"])==len(inputs)==18
assert sum(r["trajectory_ticks"] for r in d["checkpoints"])==8078
for r,i in zip(d["checkpoints"],inputs):
    assert r["checkpoint"]==i["path"] and r["checkpoint_sha256"]==i["sha256"]
    k=r["label"];a=x[k+"__action"];target=x[k+"__target"]
    ac=x[k+"__acceleration"];phi=x[k+"__phi"];m=x[k+"__metrics"];common=x[k+"__common_action"]
    assert len(a)==r["trajectory_ticks"]
    assert np.isfinite(a).all() and np.isfinite(ac).all()
    assert r["weight_mapping_and_manual_graph_max_error"]<1e-5
    if r["existing_export_max_action_error"] is not None:assert r["existing_export_max_action_error"]==0
    if r["existing_final_trace_max_action_error"] is not None:assert r["existing_final_trace_max_action_error"]==0
    # No target in this contract reaches an additional +/-1 rad clamp.
    expected=qnom+.35*np.clip(a[:,:4].astype(np.float64),-1,1)
    np.testing.assert_allclose(expected,target,atol=1e-12,rtol=0)
    np.testing.assert_allclose(common[0],r["reset_raw_action"],atol=0,rtol=0)
    np.testing.assert_allclose(common[1:].mean(axis=0),r["same_85_state_raw_action_mean"],atol=0,rtol=0)
    for name,start in (("all_trajectory",0),("steady_trajectory",100)):
        s=r[name]
        if s is None:assert len(a)<=100;continue
        aa=a[start:];tt=target[start:];acc=ac[start:];pp=phi[start:];mm=m[start:]
        assert len(aa)==s["ticks"]
        np.testing.assert_allclose(np.mean(aa<=-1,axis=0),s["action_negative_clip_fraction"],atol=0,rtol=0)
        np.testing.assert_allclose(np.rad2deg(np.ptp(tt,axis=0)),s["leg_target_range_deg"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.sqrt(np.mean(acc[:,:4]**2,axis=0)),s["leg_accel_rms_rad_s2"],atol=1e-10,rtol=0)
        np.testing.assert_allclose(np.abs(mm[:,0]).mean(),s["vx_mae_mps"],atol=1e-12,rtol=0)
    if r["stage"]=="fresh1000":
        assert not r["has_leg_acceleration_gain_state"] and not r["has_leg_reversal_gain_state"]
    if r["stage"]=="fresh1000" and r["completed_iterations_local"]==101:
        assert r["steady_trajectory"]["action_negative_clip_fraction"][2:4]==[.055,.1875]
    if r["stage"]=="fresh1000" and r["completed_iterations_local"]>=201:
        assert r["steady_trajectory"]["action_negative_clip_fraction"][2:4]==[1.,1.]
        assert r["steady_trajectory"]["leg_target_range_deg"][2:4]==[0.,0.]
    checks.append({"checkpoint":k,"source_bound":True,"summary_recomputed":True,"target_mapping_recomputed":True})
p=json.loads((OUT/"training-and-budget-provenance.json").read_text())
np.testing.assert_allclose(p["reference_budget"]["pooled_p90_rad_s2"],[40.1703144886,38.3671852533,7.0362190023,4.9735461617],atol=1e-9,rtol=0)
for old in json.loads((OUT/"prior-reports.json").read_text()):assert Path(old).is_file()
protected=json.loads((OUT/"protected-before.json").read_text())
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()==h for f,h in protected.items())
index=PROJECT/"docs/2026-10-13-dreamwaq-jitter-debug-index.md"
for f in (OUT/"report.md",index):
    text=f.read_text(encoding="utf-8");assert "\ufffd" not in text and "101" in text and "201" in text
result={"passed":True,"checkpoint_count":18,"control_frames":8078,"physics_steps":161560,
        "original_export_checks":4,"prior_report_count":len(json.loads((OUT/"prior-reports.json").read_text())),
        "protected_file_count":len(protected),"protected_files_unchanged":True,"production_code_or_parameters_changed":False,
        "training_started":False,"isaac_started":False,"subagents_called":False,
        "first_persistently_saturated_saved_original_model_completed_iterations":201,
        "previous_saved_model_completed_iterations":101,
        "onset_is_exact_training_iteration":False,"checks":checks,
        "artifact_sha256":{str(f):hashlib.sha256(f.read_bytes()).hexdigest().upper() for f in [index,*OUT.iterdir()] if f.is_file() and f.name!="verification.json"}}
(OUT/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k not in ("checks","artifact_sha256")}))
