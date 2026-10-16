from pathlib import Path
import sys,json,csv
import numpy as np,torch
OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/"wheelleg_dreamwaq").is_dir());P=ROOT/"wheelleg_dreamwaq"
sys.path.insert(0,str(P/"source/wheelleg_dreamwaq"))
import wheelleg_dreamwaq.schemas
from wheelleg_dreamwaq.kinematics.virtual_leg import offset_geometry_from_audit,offset_leg_fk
torch.set_num_threads(1)
B=P/"artifacts/debug/sim2sim/leg-acceleration-budget-search-20261012-v1"
x=np.load(B/"saturation-reward-audit-20261012-v1/trajectories.npz")
audit=json.loads((P/"artifacts/phase1_v4/usd-anchor-audit.json").read_text());geom={s:offset_geometry_from_audit(audit["legs"][s]) for s in ("left","right")}
res=[]
def spec(y):
    y=np.asarray(y,dtype=float);tt=np.arange(len(y));y=y-np.polyval(np.polyfit(tt,y,1),tt)
    z=np.fft.rfft(y);freq=np.fft.rfftfreq(len(y),.02);power=np.abs(z)**2
    idx=np.argmax(power[1:])+1
    return dict(peak_hz=float(freq[idx]),power_fraction_ge5hz=float(power[freq>=5].sum()/power[1:].sum()) if power[1:].sum()>0 else 0)
for key in sorted(k.removesuffix("__target") for k in x.files if k.endswith("__target")):
    t=x[key+"__time_before"];keep=(t>=2-1e-10)&(t<20-1e-10)
    target=x[key+"__target"][keep,:4];q=x[key+"__q"][keep,:4];acc=x[key+"__acc"][keep,:4]
    ph=x[key+"__phi"][keep];L=x[key+"__length"][keep]
    sides=[]
    for side,ids in (("left",[0,1]),("right",[2,3])):
        j=0 if side=="left" else 1
        ai=acc[:,ids];qc=.5*q[:,ids].sum(1);qd=.5*(q[:,ids[0]]-q[:,ids[1]])
        ac=.5*ai.sum(1);ad=.5*(ai[:,0]-ai[:,1])
        total=float(np.mean(ai**2));fraction=float(np.mean(ad**2)/total) if total else 0
        f=offset_leg_fk(torch.tensor(target[:,ids],dtype=torch.float64),geom[side])
        p0,p1=np.percentile(np.rad2deg(ph[:,j]),[5,95]);l0,l1=np.percentile(L[:,j],[5,95])
        sides.append(dict(side=side,actual_qfront_qrear_correlation=float(np.corrcoef(q[:,ids].T)[0,1]) if np.ptp(q[:,ids])>1e-10 else None,actual_accel_energy_fraction_differential=fraction,actual_common_accel_rms_rad_s2=float(np.sqrt(np.mean(ac**2))),actual_differential_accel_rms_rad_s2=float(np.sqrt(np.mean(ad**2))),actual_phi0_p5p95_width_deg=float(p1-p0),actual_length_p5p95_width_mm=float((l1-l0)*1000),actual_common_angle_spectrum=spec(qc),actual_differential_angle_spectrum=spec(qd),actual_phi_spectrum=spec(ph[:,j]),actual_length_spectrum=spec(L[:,j]),target_phi_p5p95_width_deg=float(np.diff(np.percentile(torch.rad2deg(f.phi0).numpy(),[5,95]))[0]),target_length_p5p95_width_mm=float(np.diff(np.percentile(f.length.numpy(),[5,95]))[0]*1000)))
    res.append(dict(key=key,n=len(target),sides=sides))
(OUT/"jitter-mode-analysis.json").write_text(json.dumps(dict(schema="CommonDifferentialJitterModeV1",source=str(B/"saturation-reward-audit-20261012-v1/trajectories.npz"),mode_definition="q_common=(q_front+q_rear)/2; q_differential=(q_front-q_rear)/2; total pair acceleration energy=2*(common energy+differential energy); existing actual control-tick accelerations unchanged",note="Pure decomposition of existing MuJoCo trajectories; not proof one mode unnecessary; spectra remove linear trend; fixed-target amplitudes may approach roundoff; not a safe threshold",trajectories=res),indent=2,allow_nan=False),encoding="utf-8")
for v in res:
    if v["key"].endswith("nominal_stand") or "interactive" in v["key"]:print(json.dumps(v,indent=2))
