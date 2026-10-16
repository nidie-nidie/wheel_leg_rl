"""Frozen candidate tests: independent schedule, all formal commands, external pushes."""
from pathlib import Path
import importlib.util,json
import numpy as np
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("slow_feedback",OUT/"slow_feedback.py")
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
for scenario,c in s.FORMAL_SCENARIOS:
    s.trial("candidate_v1",scenario,lambda tick,c=c:c,500,[0,0],[.3,.3],zero_only=True,settled_delay_s=.5)
for scenario,c in s.SCENARIOS:
    s.trial("candidate_v1",scenario,lambda tick,c=np.array(c):s.command_at_tick(c,tick,s.payload),500,[0,0],[.3,.3],zero_only=True,settled_delay_s=.5)
s.trial("candidate_v1","stand60",lambda tick:[0,0,.2],3000,[0,0],[.3,.3],zero_only=True,settled_delay_s=.5)
# Longer cold-reset and varied stop durations; not the selection schedule.
rng=np.random.default_rng(637415)
durations=rng.integers(80,351,size=20)
sequence=[]
for i in range(20):
    if i%2:sequence.append([0,0,float(rng.uniform(.17,.23))])
    else:sequence.append([float(rng.uniform(-1,1)),float(rng.uniform(-.6,.6)),.2])
ends=np.cumsum(durations)
def schedule(tick):return sequence[min(int(np.searchsorted(ends,tick,side="right")),len(sequence)-1)]
(OUT/"frozen-holdout-schedule.json").write_text(json.dumps({"seed":637415,"stage_steps":durations.tolist(),"commands":sequence},indent=2),encoding="utf-8")
for name,ki,zero,delay in (("candidate_schedule",[.3,.3],True,.5),("filter_only_schedule",[0,0],False,0)):
    s.trial(name,"holdout_dynamic",schedule,int(ends[-1]),[0,0],ki,zero_only=zero,settled_delay_s=delay)
# Side/fore-aft disturbances at both small and stronger tested amplitudes, not training randomization.
for strength in (10,20):
    for label,direction in (("forward",[1,0,0]),("reverse",[-1,0,0]),("left",[0,1,0]),("right",[0,-1,0])):
        push={"force_control_n":(strength*np.array(direction)).tolist(),"start_tick":150,"end_tick":158}
        for name,ki,zero,delay in (("candidate_push",[.3,.3],True,.5),("filter_only_push",[0,0],False,0)):
            s.trial(name,label+str(strength),lambda tick:[0,0,.2],750,[0,0],ki,zero_only=zero,settled_delay_s=delay,push=push)
np.savez_compressed(OUT/"frozen-candidate-trajectories.npz",**s.arrays)
protected=json.loads((OUT/"protected-before.json").read_text());assert all(s.sha256_file(Path(p))==h for p,h in protected.items())
specification={"schema":"ObservableFeedbackCandidateV1","model_source":"85% adaptive acceleration diagnostic checkpoint, not accepted model",
    "source_actor_sha256":s.sha256_file(s.BASE/"85/export/actor.ts"),"source_checkpoint_sha256":s.manifest["source_checkpoint_sha256"],
    "model_xml_sha256":s.sha256_file(s.rt.model_path),"control_dt_s":.02,"filter_hz_legs_wheels":[6,2],
    "filter_alpha_legs_wheels":(1-np.exp(-2*np.pi*np.array([6,2])*.02)).tolist(),
    "filter_equation":"processed=(1-alpha)*previous_processed+alpha*clip(raw,-1,1)",
    "feedback_sensors":"vx=0.0625*mean(controlled_wheel_qdot); yaw=control_IMU_gyro_z; first-order 1Hz filtered",
    "feedback_mode":"only both original commanded vx and yaw exactly zero and complete original command held >=0.5s",
    "integral_gain_vx_yaw":[.3,.3],"proportional_gain_vx_yaw":[0,0],"integral_bounds_vx_yaw":[.06,.06],
    "movement_behavior":"freeze integrator, applied command correction zero; no altered user command for metrics",
    "reset_behavior":"processed action, low-pass sensor states and integrators reset to zero; history same original reset with real zero action",
    "command_refresh":"current frame command refreshed before CENet; all older frames retained with real command/sensors/processed-action timing",
    "production_installed":False,"training_started":False,"extra_controller_torque":False,
    "limitations":["Flat rolling-contact odometry assumption; no proof for wheel slip or rough terrain.","Persistent leg asymmetry still present.","No position setpoint; reduced net drift is not position locking.","Isaac and real-robot operation untested."]}
(OUT/"candidate-spec.json").write_text(json.dumps(specification,indent=2),encoding="utf-8")
(OUT/"frozen-candidate-results.json").write_text(json.dumps({"results":s.results,"specification_sha256":s.sha256_file(OUT/"candidate-spec.json"),
    "selection_complete_before_holdout":True,"not_formal_acceptance":True,"training_started":False,"production_modified":False,"protected_files_unchanged":True},indent=2),encoding="utf-8")
print("DONE",len(s.results),sum(r["ticks"] for r in s.results),flush=True)
