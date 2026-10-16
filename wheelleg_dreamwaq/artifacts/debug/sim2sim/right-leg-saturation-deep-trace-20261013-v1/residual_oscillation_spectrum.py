"""Analyze existing residual motion; no simulation, policy or production edits."""
from pathlib import Path
import json
import numpy as np
OUT=Path(__file__).resolve().parent
d=np.load(OUT/"frozen-candidate-trajectories.npz");k="candidate_v1__stand60";sl=slice(100,None)
q=d[k+"__q"][sl];v=d[k+"__v"][sl];raw=d[k+"__raw"][sl];a=d[k+"__applied"][sl]
def spectrum(x,dt):
    t=np.arange(len(x));x=np.asarray(x)-np.polyval(np.polyfit(t,x,1),t)
    f=np.fft.rfftfreq(len(x),dt);power=np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
    positive=f>0
    total=power[positive].sum()
    fractions={str(lo)+"_"+str(hi):float(power[(f>=lo)&(f<hi)&positive].sum()/total)
               for lo,hi in ((0,1),(1,3),(3,6),(6,12),(12,25),(25,501))}
    assert all(0<=v<=1 for v in fractions.values())
    assert abs(sum(fractions.values())-1)<1e-10
    return {"peak_hz":float(f[1+np.argmax(power[1:])]),"power_fractions":fractions}
signals={}
for tag,x in (("left_common_q",(q[:,0]+q[:,1])/2),("right_common_q",(q[:,2]+q[:,3])/2),
              ("wheel_mean_processed",a[:,4:6].mean(axis=1)),("wheel_mean_qdot",v[:,4:6].mean(axis=1)),
              ("actual_vx",d[k+"__metrics"][sl,0]),("actual_yaw",d[k+"__metrics"][sl,1])):
    signals[tag]=spectrum(x,.02)
physics=np.load(OUT/"final-physics-trajectories.npz")["physics"];p=physics[physics[:,0]>=2]
for tag,x in (("left_common_1ms",(p[:,1]+p[:,2])/2),("right_common_1ms",(p[:,3]+p[:,4])/2),
              ("wheel_mean_torque_1ms",(p[:,23]+p[:,24])/2)):
    signals[tag]=spectrum(x,.001)
result={"sources":["frozen-candidate-trajectories.npz","final-physics-trajectories.npz"],
        "same_frequency_is_not_causal_proof":True,"signals":signals,
        "right_processed_action_full_range_after2s":np.ptp(a[:,2:4],axis=0).tolist(),
        "right_raw_clipped_endpoint_fraction":np.mean(raw[:,2:4]<=-1,axis=0).tolist(),
        "standing_raw_wheel_clip_fraction":np.mean(np.abs(raw[:,4:6])>=1,axis=0).tolist(),
        "visual_goal_met":False,"training_started":False,"production_modified":False}
assert result["right_processed_action_full_range_after2s"]==[0,0]
assert all(6<r["peak_hz"]<7 for r in signals.values())
(OUT/"residual-oscillation-spectrum.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
