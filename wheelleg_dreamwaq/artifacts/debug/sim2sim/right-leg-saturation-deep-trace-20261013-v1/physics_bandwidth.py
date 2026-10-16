"""Observe original mj_step calls at 1kHz, without changing the integration path."""
from pathlib import Path
import json,sys,time
import numpy as np,torch,mujoco
OUT=Path(__file__).resolve().parent;PROJECT=OUT.parents[3]
sys.path.insert(0,str(PROJECT/"sim2sim/mujoco"))
from wheelleg_mujoco.contract import load_policy_contract,sha256_file
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
BASE=OUT.parent/"leg-acceleration-budget-search-20261012-v1"
contract,manifest,mm=load_policy_contract(BASE/"85/export/policy_manifest.json",BASE/"85/export/actor.ts",PROJECT/"sim2sim/mujoco/model_manifest.json")
rt=WheelLegMujocoRuntime(PROJECT/"sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",contract,mm)
p=torch.jit.load(str(BASE/"85/export/actor.ts")).eval();torch.set_num_threads(1)
saved_step=mujoco.mj_step;arrays={};checks=[]
for variant,hz in (("unmodified",(0.,0.)),("leg6_wheel2",(6.,2.))):
    obs=rt.reset(np.array([0.,0.,.2]));filtered=np.zeros(6)
    f=np.r_[np.repeat(hz[0],4),np.repeat(hz[1],2)]
    alpha=1-np.exp(-2*np.pi*f*.02);alpha[f==0]=1
    rows=[];actions=[];histories=[]
    def capture_step(model,data,*args,**kwargs):
        before=rt.model_map.joint_velocity_control(data).copy()
        saved_step(model,data,*args,**kwargs)
        after=rt.model_map.joint_velocity_control(data).copy()
        rows.append([float(data.time),*rt.model_map.joint_position_control(data),*after,*((after-before)/.001),
            *data.qfrc_actuator[rt.model_map.dof_addresses]*np.r_[np.ones(4),contract.wheel_signs]])
    mujoco.mj_step=capture_step
    try:
        for tick in range(500):
            histories.append(obs.copy())
            with torch.inference_mode():raw=p(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
            actions.append(raw.copy())
            filtered=filtered+alpha*(np.clip(raw,-1,1).astype(np.float64)-filtered)
            result=rt.step(filtered,np.array([0.,0.,.2]))
            obs=result.observation
    finally:mujoco.mj_step=saved_step
    rows=np.array(rows);arrays[variant+"__physics"]=rows;arrays[variant+"__raw"]=np.array(actions)
    prior=np.load(OUT/"expanded-cases"/(variant+"__nominal_stand.npz"))
    np.testing.assert_array_equal(np.array(actions),prior["raw"])
    np.testing.assert_array_equal(np.array(histories),prior["history"])
    np.testing.assert_array_equal(rows[19::20,1:7],prior["q"])
    np.testing.assert_array_equal(rows[19::20,7:13],prior["v"])
    select=rows[rows[:,0]>=2]
    checks.append({"variant":variant,"physics_rows":len(rows),"control_actions_and_history_exact":True,
        "control_sample_q_and_velocity_exact":True,
        "actual_1ms_accel_rms_rad_s2":np.sqrt(np.mean(select[:,13:19]**2,axis=0)).tolist(),
        "actual_1ms_accel_p99_rad_s2":np.percentile(np.abs(select[:,13:19]),99,axis=0).tolist(),
        "commanded_torque_rms_nm":np.sqrt(np.mean(select[:,19:25]**2,axis=0)).tolist()})
    print(json.dumps(checks[-1]),flush=True)
np.savez_compressed(OUT/"physics-bandwidth-trajectories.npz",**arrays)
protected=json.loads((OUT/"protected-before.json").read_text())
assert all(sha256_file(Path(p))==h for p,h in protected.items())
(OUT/"physics-bandwidth-results.json").write_text(json.dumps({"checks":checks,"physics_dt_s":.001,
    "acceleration_definition":"finite difference of actual controlled qvel across each original mj_step; no new forward calls or dynamics changes",
    "torque_timestamp":"actuator torque during last solved step, q/v sampled after integration",
    "training_started":False,"production_modified":False,"protected_files_unchanged":True},indent=2),encoding="utf-8")
