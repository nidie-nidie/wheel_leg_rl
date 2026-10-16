from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
O=Path(__file__).resolve().parent;R=next(p for p in O.parents if (p/"wheelleg_dreamwaq").is_dir());P=R/"wheelleg_dreamwaq"
sys.path.insert(0,str(P/"source/wheelleg_dreamwaq"))
import wheelleg_dreamwaq.schemas
from wheelleg_dreamwaq.kinematics.virtual_leg import offset_geometry_from_audit,offset_leg_fk
x=json.loads((O/"results.json").read_text());a=json.loads((P/"artifacts/phase1_v4/usd-anchor-audit.json").read_text());g=offset_geometry_from_audit(a["legs"]["left"])
checks={}
checks["input_hashes_unchanged"]=all(hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()==h for p,h in x["input_hashes"].items())
checks["actuator_joint_mapping"]=all(v["joint_id"]==v["actuator_joint_id"] for v in x["mapping"])
checks["fk_all_grid_points_valid"]=x["fk_grid"]["valid_both"]==10201
maxerr=0.
for v in x["fk_jacobians"]:
 q=torch.tensor(x["q_nominal"][:2],dtype=torch.float64)+.35*v["common_action"]
 J=np.zeros((2,2));eps=1e-6
 for j in (0,1):
  d=torch.zeros(2,dtype=torch.float64);d[j]=eps;fp,fm=offset_leg_fk(q+d,g),offset_leg_fk(q-d,g)
  J[:,j]=[(float(fp.length)-float(fm.length))/(2*eps),(float(fp.phi0)-float(fm.phi0))/(2*eps)]
 maxerr=max(maxerr,float(np.abs(J-np.array(v["jacobian_wrt_q"])).max()))
checks["jacobian_finite_difference"]=maxerr<1e-7
z=np.load(P/"artifacts/debug/sim2sim/leg-acceleration-budget-search-20261012-v1/saturation-reward-audit-20261012-v1/trajectories.npz")
mode=json.loads((O/"jitter-mode-analysis.json").read_text());modeerr=0.
for v in mode["trajectories"]:
 key=v["key"];t=z[key+"__time_before"];keep=(t>=2-1e-10)&(t<20-1e-10);acc=z[key+"__acc"][keep,:4]
 for j,side in enumerate(v["sides"]):
  p=acc[:,j*2:j*2+2];co=(p[:,0]+p[:,1])/2;di=(p[:,0]-p[:,1])/2;total=np.mean(np.sum(p*p,axis=1))
  frac=2*np.mean(di*di)/total
  modeerr=max(modeerr,abs(float(frac)-side["actual_accel_energy_fraction_differential"]))
  assert np.allclose(np.sum(p*p,axis=1),2*(co*co+di*di),rtol=1e-12,atol=1e-10)
checks["mode_energy_and_fraction"]=modeerr<1e-12
checks["no_training_or_isaac"]=not x["training_started"] and not x["isaac_started"]
res=dict(passed=all(checks.values()),checks=checks,jacobian_finite_difference_max_error=maxerr,mode_fraction_max_error=modeerr,protected_inputs=len(x["input_hashes"]),verification_scope="Mechanical audit and decomposition only; no claim of deployment performance",writes_only_this_directory=True)
(O/"verification.json").write_text(json.dumps(res,indent=2),encoding="utf-8")
print(json.dumps(res,indent=2));assert res["passed"]
