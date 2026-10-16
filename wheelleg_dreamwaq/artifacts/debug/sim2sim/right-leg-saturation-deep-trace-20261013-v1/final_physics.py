"""1kHz final-candidate motion check with identical original integrator calls."""
from pathlib import Path
import importlib.util,json
import numpy as np,mujoco
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("slow_feedback",OUT/"slow_feedback.py")
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
rows=[];original=mujoco.mj_step
def capture(model,data,*args,**kwargs):
    before=s.rt.model_map.joint_velocity_control(data).copy()
    original(model,data,*args,**kwargs)
    after=s.rt.model_map.joint_velocity_control(data).copy()
    rows.append([float(data.time),*s.rt.model_map.joint_position_control(data),*after,*((after-before)/.001),
        *data.qfrc_actuator[s.rt.model_map.dof_addresses]*np.r_[np.ones(4),s.contract.wheel_signs]])
mujoco.mj_step=capture
try:r=s.trial("final_physics","nominal_stand",lambda tick:[0,0,.2],500,[0,0],[.3,.3],zero_only=True,settled_delay_s=.5)
finally:mujoco.mj_step=original
d=np.load(OUT/"frozen-candidate-trajectories.npz")
checks={}
for key in ("history","raw","q","v","applied","user_command","policy_command","correction"):
    np.testing.assert_array_equal(s.arrays["final_physics__nominal_stand__"+key],d["candidate_v1__nominal_stand__"+key]);checks[key]=True
rows=np.array(rows)
np.testing.assert_array_equal(rows[19::20,1:7],d["candidate_v1__nominal_stand__q"])
np.testing.assert_array_equal(rows[19::20,7:13],d["candidate_v1__nominal_stand__v"])
z=rows[rows[:,0]>=2]
np.savez_compressed(OUT/"final-physics-trajectories.npz",physics=rows)
(OUT/"final-physics-results.json").write_text(json.dumps({"physics_rows":len(rows),"physics_dt_s":.001,"exact_original_runtime_control_trace":checks,
    "actual_accel_rms_rad_s2":np.sqrt(np.mean(z[:,13:19]**2,axis=0)).tolist(),
    "actual_accel_p99_rad_s2":np.percentile(np.abs(z[:,13:19]),99,axis=0).tolist(),
    "torque_rms_nm":np.sqrt(np.mean(z[:,19:25]**2,axis=0)).tolist(),
    "wrapper_added_forward_or_integration_calls":False,"training_started":False,"production_modified":False},indent=2),encoding="utf-8")
print((OUT/"final-physics-results.json").read_text(),flush=True)
