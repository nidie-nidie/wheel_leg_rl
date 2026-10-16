"""Offline audit of already-run diagnostic results; no simulations or training."""
from pathlib import Path
import json, math, hashlib
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
inputs=[ROOT/n for n in ("candidate-validation-results.json","expanded-results.json","push-results.json","physics-bandwidth-results.json","expanded-cases/leg6_wheel2__long_stand.npz","expanded-cases/unmodified__long_stand.npz")]
before={str(p):sha(p) for p in inputs}
candidate=json.loads(inputs[0].read_text())["results"]
expanded=json.loads(inputs[1].read_text())["results"]
by={r["key"]:r for r in candidate+expanded}
margins={}
for variant in ("leg5_wheel2","leg6_wheel2","leg8_wheel2","leg6_wheel1p5","leg6_wheel2p5"):
    rows=[r for r in candidate if r["variant"]==variant and "practice_summary" in r]
    stops=[s for r in rows for s in r["practice_summary"]["stages"] if s["stage_index"] in (1,3,5)]
    turns=[s for r in rows for s in r["practice_summary"]["stages"] if s["stage_index"] in (2,4)]
    margins[variant]={"practice_count":len(rows),"practice_pass_count":sum(r["practice_summary"]["performance_accepted"] for r in rows),
        "worst_stop_vx_mae_mps":max(s["steady_vx_mae_mps"] for s in stops),"worst_stop_yaw_mae_rad_s":max(s["steady_yaw_mae_rad_s"] for s in stops),
        "worst_required_reversal_confirmation_s":None if any(s["held_error_band_confirmation_time_s"] is None for s in turns) else max(s["held_error_band_confirmation_time_s"] for s in turns),
        "stage0_confirmation_s":[r["practice_summary"]["stages"][0]["held_error_band_confirmation_time_s"] for r in rows]}
holdout={}
for v in ("holdout_unmodified","holdout_leg6_wheel2"):
    rows=[r for r in candidate if r["variant"]==v]
    holdout[v]={"count":len(rows),"failures":sum(r["failure_reason"] is not None for r in rows)}
    for metric in ("vx_mae_mps","yaw_mae_rad_s","height_mae_m"):
        x=np.array([r["steady"][metric] for r in rows])
        holdout[v][metric]={"mean":float(x.mean()),"worst":float(x.max())}
    holdout[v]["leg_accel_rms_mean"] = np.mean([r["steady"]["leg_accel_rms_rad_s2"] for r in rows],axis=0).tolist()
drift={}
for v in ("unmodified","leg6_wheel2"):
    with np.load(ROOT/f"expanded-cases/{v}__long_stand.npz") as z:
        xy=np.r_[z["com_before"][100:101],z["com_after"][100:]]
        drift[v]={"measured_time_s":float(z["time"][-1]-z["time"][99]),"path_distance_m":float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
            "net_displacement_m":float(np.linalg.norm(xy[-1]-xy[0])),"delta_world_xy_m":(xy[-1]-xy[0]).tolist()}
filters={}
for fc in (1.5,2,2.5,3,5,6,8):
    dt=.02;p=math.exp(-2*math.pi*fc*dt);alpha=1-p
    angles=np.array([9,10,11])*2*math.pi*dt
    mag=alpha/np.sqrt(1+p*p-2*p*np.cos(angles))
    filters[str(fc)]={"alpha":alpha,"dc_delay_s":dt*p/alpha,"gain_at_9_10_11_hz":mag.tolist(),"label_meaning":"time constant 1/(2*pi*fc), not exact digital -3dB frequency"}
# State-conditional clipped-Gaussian calculation on recorded stand means. Not an Isaac rollout.
mu=np.array(by["unmodified__nominal_stand"]["steady"]["raw_mean"])
sigma=np.array([.10345245,.10287046,.96740139,1.01099932,1.07976377,.96099889])
Phi=lambda x:.5*(1+math.erf(x/math.sqrt(2)))
inside=np.array([Phi((1-m)/s)-Phi((-1-m)/s) for m,s in zip(mu,sigma)])
tail=np.array([Phi((-1-m)/s) for m,s in zip(mu,sigma)])
prob={"mean":mu.tolist(),"sigma_checkpoint_previous_audit":sigma.tolist(),"conditional_inside_probability":inside.tolist(),"conditional_negative_atom_probability":tail.tolist(),
    "expected_clipped_action_sensitivity_to_mean":inside.tolist(),"entropy_mean_derivative":0,
    "limitations":"Probability at the recorded mean state only; not observed Isaac training action statistics. dE[clip(A)]/dmu=P(-1<A<1), for fixed sigma. This does NOT say PPO has zero gradient."}
result={"practice_margins":margins,"holdout":holdout,"long_parking_drift":drift,"filter_transfer":filters,"clipped_gaussian":prob,
    "input_sha256":before,"inputs_unchanged":all(sha(Path(p))==h for p,h in before.items()),"training_started":False,"production_modified":False}
assert result["inputs_unchanged"]
(OUT/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({"practice_margins":margins,"holdout":holdout,"long_parking_drift":drift,"right_inside_prob":inside[2:4].tolist()},indent=2))
