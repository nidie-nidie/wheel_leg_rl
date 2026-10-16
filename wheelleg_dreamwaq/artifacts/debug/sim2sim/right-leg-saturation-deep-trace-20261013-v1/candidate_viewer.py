"""Debug-only interactive viewer for the reviewed frozen candidate."""
from pathlib import Path
import argparse,importlib.util,json,queue,sys,time
import numpy as np
import torch
import mujoco
import glfw

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("candidate_reference",OUT/"slow_feedback.py")
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
INTERACTIVE=OUT.parent/"interactive-mujoco-20261010-v1"
sys.path.insert(0,str(INTERACTIVE))
from keyboard_viewer import KeyboardViewer

class Candidate:
    def reset(self,command):
        self.filtered=np.zeros(6);self.estimate=np.zeros(2);self.integral=np.zeros(2)
        self.stable_steps=0;self.last_command=np.array(command,dtype=np.float64)
        self.alpha=np.r_[np.repeat(1-np.exp(-2*np.pi*6*s.dt),4),np.repeat(1-np.exp(-2*np.pi*2*s.dt),2)]
        self.observer_alpha=1-np.exp(-2*np.pi*s.dt)
        self.episode_start_xy=None
        s.rt.reset(self.last_command)
        self.episode_start_xy=s.rt.data.xipos[s.rt.model_map.base_body_id,:2].copy()

    def tick(self,command):
        current=np.array(command,dtype=np.float64)
        state=s.collect_kinematic_state(s.rt.model,s.rt.data,s.rt.model_map,s.contract)
        sensor=np.array([s.radius*state.joint_velocity_control[4:6].mean(),state.angular_velocity_control[2]])
        self.estimate+=self.observer_alpha*(sensor-self.estimate)
        error=current[:2]-self.estimate
        self.stable_steps=self.stable_steps+1 if np.array_equal(current,self.last_command) else 0
        self.last_command=current.copy()
        active=np.all(current[:2]==0) and self.stable_steps*s.dt>=.5
        limits=np.array([.06,.06])
        if active:self.integral=np.clip(self.integral+np.array([.3,.3])*error*s.dt,-limits,limits)
        correction=np.clip(np.array([0,0])*error+self.integral,-limits,limits) if active else np.zeros(2)
        policy_command=current.copy();policy_command[:2]+=correction
        policy_command[0]=np.clip(policy_command[0],-s.contract.normalization["vx_max_abs"],s.contract.normalization["vx_max_abs"])
        policy_command[1]=np.clip(policy_command[1],-s.contract.normalization["yaw_rate_max_abs"],s.contract.normalization["yaw_rate_max_abs"])
        frame=s.build_actor_observation(state,policy_command,s.rt.previous_action,s.contract)
        s.rt._history[-1]=frame
        history=s.rt._history.reshape(-1).astype(np.float32,copy=True)
        with torch.inference_mode():
            x=torch.from_numpy(history).unsqueeze(0)
            raw=s.policy(x).numpy()[0];encoded=s.policy.encoder(x).numpy()[0,:19]
        self.filtered+=self.alpha*(np.clip(raw,-1,1).astype(np.float64)-self.filtered)
        before_v=s.rt.model_map.joint_velocity_control(s.rt.data).copy()
        targets=s.rt.controller.prepare(self.filtered)
        s.rt.data.xfrc_applied[:]=0
        result=s.rt.step(self.filtered,policy_command)
        velocity=s.rt.model_map.joint_velocity_control(s.rt.data).copy()
        return {"time":float(s.rt.data.time),"history":history,"raw":raw,"applied":self.filtered.copy(),
                "q":s.rt.model_map.joint_position_control(s.rt.data).copy(),"v":velocity,
                "acc":(velocity-before_v)/s.dt,"user_command":current,"policy_command":policy_command,
                "correction":correction,"sensor_before":sensor,"filtered_sensor":self.estimate.copy(),
                "encoded":encoded,"feedback_active":bool(active),"metrics":result.metrics,
                "target":np.r_[targets.leg_position_target_mujoco,targets.wheel_velocity_target_mujoco*s.contract.wheel_signs],
                "torque":result.applied_torque*np.r_[np.ones(4),s.contract.wheel_signs],
                "com_xy":s.rt.data.xipos[s.rt.model_map.base_body_id,:2].copy()}

class Keyboard:
    def __init__(self,candidate):
        self.candidate=candidate
        self.command=np.array([0,0,.2],dtype=np.float64)
        ranges=s.manifest["command_sampling"]
        self.lower=np.array([ranges[k][0] for k in ("vx","yaw_rate","base_height")])
        self.upper=np.array([ranges[k][1] for k in ("vx","yaw_rate","base_height")])
        self.paused=False;self.quit=False;self.failure=""
        self.episode=0
        self.candidate.reset(self.command)

    def key(self,key):
        reset=False
        if key==glfw.KEY_ESCAPE:self.quit=True
        elif key in (ord("P"),glfw.KEY_SPACE):self.paused=not self.paused
        elif key==ord("R"):
            self.candidate.reset(self.command);self.episode+=1
            self.paused=False;self.failure="";reset=True
        elif key==ord("X"):self.command[:2]=0
        elif key in (ord("W"),ord("S")):self.command[0]+=.1 if key==ord("W") else -.1
        elif key in (ord("A"),ord("D")):self.command[1]+=.1 if key==ord("A") else -.1
        elif key in (ord("Q"),ord("E")):self.command[2]+=.01 if key==ord("Q") else -.01
        self.command[:]=np.round(np.clip(self.command,self.lower,self.upper),2)
        return reset

def smoke():
    candidate=Candidate();saved=np.load(OUT/"frozen-candidate-trajectories.npz")
    cases=[("nominal_stand",lambda t:np.array([0,0,.2]))]
    cases.extend((name,lambda t,c=np.array(c):s.command_at_tick(c,t,s.payload)) for name,c in s.SCENARIOS)
    checks=[]
    for name,fn in cases:
        candidate.reset(fn(0));rows=[candidate.tick(fn(t)) for t in range(500)]
        for field in ("history","raw","applied","q","v","user_command","policy_command","correction","sensor_before","filtered_sensor","encoded"):
            actual=np.array([r[field] for r in rows])
            np.testing.assert_array_equal(actual,saved["candidate_v1__"+name+"__"+field])
        checks.append({"scenario":name,"frames":500,"exact_saved_trace":True})
    candidate.reset([0,0,.2])
    repeated=candidate.tick([0,0,.2])
    np.testing.assert_array_equal(repeated["raw"],saved["candidate_v1__nominal_stand__raw"][0])
    panel=Keyboard(candidate)
    for key in "WAQ":panel.key(ord(key))
    np.testing.assert_allclose(panel.command,[.1,.1,.21])
    for key in "SDE":panel.key(ord(key))
    np.testing.assert_array_equal(panel.command,[0,0,.2])
    for _ in range(100):panel.key(ord("W"))
    assert panel.command[0]==panel.upper[0]
    for _ in range(100):panel.key(ord("S"))
    assert panel.command[0]==panel.lower[0]
    panel.key(ord("X"));panel.key(ord("R"))
    assert s.rt.data.time==0 and np.all(candidate.filtered==0) and np.all(candidate.integral==0) and np.all(candidate.estimate==0)
    assert np.all(s.rt.previous_action==0) and candidate.stable_steps==0
    panel.key(ord("P"));assert panel.paused
    panel.key(ord("P"));assert not panel.paused
    panel.key(glfw.KEY_ESCAPE);assert panel.quit
    protected=json.loads((OUT/"protected-before.json").read_text(encoding="utf-8"))
    assert all(s.sha256_file(Path(p))==h for p,h in protected.items())
    result={"saved_frozen_trace_exact_cases":checks,"reset_and_keyboard_passed":True,
            "protected_files_unchanged":len(protected),"production_modified":False,"training_started":False,
            "viewer_source_sha256":s.sha256_file(Path(__file__)),"candidate_spec_sha256":s.sha256_file(OUT/"candidate-spec.json")}
    (OUT/"viewer-verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("CANDIDATE_VIEWER_SMOKE_OK",json.dumps(result),flush=True)

def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--session",type=Path,default=None)
    parser.add_argument("--duration",type=float,default=0)
    args=parser.parse_args()
    candidate_spec=json.loads((OUT/"candidate-spec.json").read_text(encoding="utf-8"))
    assert s.sha256_file(s.BASE/"85/export/actor.ts")==candidate_spec["source_actor_sha256"]
    assert s.sha256_file(s.rt.model_path)==candidate_spec["model_xml_sha256"]
    if args.smoke:smoke();return
    session=args.session or OUT/("viewer-"+time.strftime("%Y%m%d-%H%M%S"))
    session.mkdir(parents=True,exist_ok=True)
    panel=Keyboard(Candidate());pending=queue.SimpleQueue()
    model=mujoco.MjModel.from_xml_path(str(s.rt.model_path));display=mujoco.MjData(model)
    base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"base_proxy")
    model.geom_group[base]=2
    mujoco.mj_copyData(display,model,s.rt.data)
    latest=None;start=time.monotonic()
    help_text="W/S: vx +/-0.1 | A/D: yaw +/-0.1 | Q/E: height +/-0.01\nX: stop | R: reset | P/Space: pause | M: jitter mark | Esc: close"
    print("CANDIDATE_START",str(session),"trained vx limits",panel.lower[0],panel.upper[0],flush=True)
    metadata={"candidate_spec":candidate_spec,"candidate_spec_sha256":s.sha256_file(OUT/"candidate-spec.json"),
              "viewer_source_sha256":s.sha256_file(Path(__file__)),"command_limits":{"lower":panel.lower.tolist(),"upper":panel.upper.tolist()},
              "display_separate_from_physics":True,"recorded_torque":"last original physics substep, control coordinates",
              "control_dt_s":s.dt,"no_added_base_collision":True}
    (session/"manifest.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    with (session/"controls.jsonl").open("w",encoding="utf-8") as controls, (session/"events.jsonl").open("w",encoding="utf-8") as events, KeyboardViewer(model,display,key_callback=pending.put) as viewer:
        glfw.set_window_title(viewer.window,"WheelLeg MuJoCo | candidate legs6 wheels2 + stop correction")
        glfw.show_window(viewer.window);glfw.focus_window(viewer.window)
        viewer.cam.distance=1.25;viewer.cam.azimuth=135;viewer.cam.elevation=-15
        viewer.opt.geomgroup[2]=1;viewer.opt.geomgroup[3]=0
        print("CANDIDATE_VIEWER_READY",str(session),flush=True)
        while viewer.is_running() and not panel.quit:
            tick_start=time.monotonic()
            if args.duration>0 and tick_start-start>=args.duration:break
            while not pending.empty():
                key=pending.get();before=panel.command.copy()
                reset=panel.key(key)
                if reset:latest=None
                events.write(json.dumps({"wall_time":time.time(),"sim_time_s":float(s.rt.data.time),"episode":panel.episode,
                    "kind":"jitter_mark" if key==ord("M") else "key","key":key,"command_before":before.tolist(),
                    "command":panel.command.tolist(),"paused":panel.paused})+"\n");events.flush()
            if panel.quit:break
            if not panel.paused:
                latest=panel.candidate.tick(panel.command)
                panel.failure=s._failure_reason(s.rt,latest["metrics"]) or ""
                if panel.failure:panel.paused=True;print("CANDIDATE_FALL_PAUSED",panel.failure,latest["time"],flush=True)
                controls.write(json.dumps({"episode":panel.episode,**latest},default=serial)+"\n");controls.flush()
            metrics=latest["metrics"] if latest is not None else s.rt.last_reset_metrics
            vx_est=float(latest["encoded"][0])/s.contract.normalization["root_linear_velocity_scale"] if latest else 0.
            policy_cmd=latest["policy_command"] if latest else panel.command
            correction=latest["correction"] if latest else np.zeros(2)
            info=("DEBUG CANDIDATE: legs 6 Hz / wheels 2 Hz + slow stop correction\n"+
                  ("PAUSED" if panel.paused else "RUNNING")+f" | episode={panel.episode} | sim={s.rt.data.time:.2f}s\n"+
                  f"USER: vx={panel.command[0]:+.2f} m/s yaw={panel.command[1]:+.2f} rad/s height={panel.command[2]:.2f} m\n"+
                  f"ACTUAL: vx={metrics['vx_mps']:+.3f} m/s yaw={metrics['yaw_rate_rad_s']:+.3f} rad/s\n"+
                  f"CENet vx before action={vx_est:+.3f} m/s\n"+
                  f"POLICY command: vx={policy_cmd[0]:+.3f} yaw={policy_cmd[1]:+.3f}\n"+
                  f"STOP correction: vx={correction[0]:+.3f} yaw={correction[1]:+.3f}\n"+
                  f"Height={metrics['base_height_m']:.3f}m tilt={np.rad2deg(metrics['tilt_rad']):.1f}deg\n"+
                  f"vx range [{panel.lower[0]:+.1f},{panel.upper[0]:+.1f}] m/s\n"+
                  (f"FALL: {panel.failure}; R resets.\n" if panel.failure else "")+"\n"+help_text)
            right="Canonical motors | raw / processed / q(rad) / qdot(rad/s)\n"
            if latest is not None:
                for i,name in enumerate(("jIJ L-front","jIO L-rear","jAB R-front","jAG R-rear","wheel L","wheel R")):
                    right+=f"{name}: {latest['raw'][i]:+.2f} / {latest['applied'][i]:+.2f} / {latest['q'][i]:+.3f} / {latest['v'][i]:+.2f}\n"
                right+="\nTargets: legs rad | wheels rad/s\n"+np.array2string(latest["target"],precision=2)+"\nLast-step motor torque Nm\n"+np.array2string(latest["torque"],precision=2)
            mujoco.mj_copyData(display,model,s.rt.data)
            viewer.cam.lookat[:]=display.xipos[s.rt.model_map.base_body_id]
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,info,""),
                              (mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPRIGHT,right,"")])
            viewer.sync()
            status={"ready":True,"time_s":float(s.rt.data.time),"episode":panel.episode,"paused":panel.paused,"failure_reason":panel.failure,
                    "command":panel.command.tolist(),"policy_command":policy_cmd.tolist(),"correction":correction.tolist(),
                    "metrics":metrics,"raw_action":latest["raw"].tolist() if latest else None,
                    "processed_action":latest["applied"].tolist() if latest else None,
                    "actual_joint_q":latest["q"].tolist() if latest else None,"actual_joint_qdot":latest["v"].tolist() if latest else None,
                    "target":latest["target"].tolist() if latest else None,"torque":latest["torque"].tolist() if latest else None,
                    "recording":str(session),"filter_hz":[6,2],"zero_command_feedback_enabled":True}
            temp=session/"status.tmp"
            temp.write_text(json.dumps(status,indent=2),encoding="utf-8")
            try:temp.replace(session/"status.json")
            except PermissionError:
                # Windows readers can temporarily hold status.json without delete sharing.
                # Retry the status replacement next frame; never stop physics for telemetry.
                pass
            time.sleep(max(0,s.dt-(time.monotonic()-tick_start)))
    status["ready"]=False;status["closed"]=True
    (session/"status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print("CANDIDATE_VIEWER_CLOSED",flush=True)

if __name__=="__main__":main()
