from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent
checks=[]
total=0
for prefix in ("","gyro-followup-"):
    r=json.loads((P/(prefix+"isolation-results.json")).read_text())
    z=np.load(P/(prefix+"isolated-feedback-trajectories.npz"))
    for row in r["closed_loop_branches"]:
        start=round(row["start_simulation_s"]/.02)
        key=str(start)+"_"+row["case"]
        x=z[key+"_state"];a=z[key+"_action"]
        assert len(x)==row["completed_ticks"]==len(a)
        np.testing.assert_allclose(np.diff(x[:,0]),.02,rtol=0,atol=1e-10)
        selected=x[50:] if len(x)>100 else x
        expected=np.sqrt(np.mean(selected[:,13:19]**2,axis=0))
        np.testing.assert_allclose(expected,row["joint_accel_rms_rad_s2"],rtol=0,atol=1e-12)
        np.testing.assert_allclose(np.abs(selected[:,19]).mean(),row["vx_mae_mps"],rtol=0,atol=1e-12)
        sa=a[50:] if len(x)>100 else a
        clip=np.clip(sa,-1,1)
        switches=np.sum(clip[1:,4:6]*clip[:-1,4:6]<0,axis=0)/((len(sa)-1)*.02)
        np.testing.assert_allclose(switches,row["actions"]["wheel_sign_changes_per_s"],rtol=0,atol=1e-12)
        if row["case"]=="baseline":assert row["baseline_max_action_error"]==0
        total+=len(x)
    checks.append({"file":prefix+"isolation-results.json","branches":len(r["closed_loop_branches"]),"completed_branches":sum(row["failure_reason"] is None for row in r["closed_loop_branches"])})
r=json.loads((P/"isolation-results.json").read_text())
for row in r["frozen_physical_input_inference"]:
    if row["case"]=="baseline":
        assert max(row["actions_last_5s"]["clipped_step_rms"])<.001
        assert row["actions_last_5s"]["wheel_sign_changes_per_s"]==[0.,0.]
a=json.loads((P/"actuator-and-network-audit.json").read_text())
assert a["actuator_audit"]["passed"]
assert a["actuator_audit"]["physics_steps"]==42400
for row in a["network_temporal_attribution"].values():
    assert row["max_change_reconstruction_error"]<3e-4
replay=json.loads((P/"replay-verification.json").read_text())
assert replay["passed"] and max(replay["comparison_max_abs_errors"].values())==0.
protected=json.loads((P/"protected-before.json").read_text())
changed=[p for p,v in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()!=v]
assert not changed,changed
result={"diagnostic_evidence_verified":True,"stable_policy_claimed":False,"experiments":checks,
        "closed_loop_control_ticks":total,"closed_loop_physics_steps":20*total,
        "exact_replay_frames":replay["control_frames"],"actuator_audited_physics_steps":42400,
        "protected_file_count":len(protected),"protected_files_unchanged":True,
        "summary_statistics_independently_recomputed":True,
        "baseline_branches_match_recorded_actions_exactly":True}
(P/"final-verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result))
