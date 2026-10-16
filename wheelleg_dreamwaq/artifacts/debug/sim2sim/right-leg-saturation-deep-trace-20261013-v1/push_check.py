"""Transient external-force tests, solely in diagnostic data, identical original model."""
from pathlib import Path
import importlib.util,json
import numpy as np,torch
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("expand",OUT/"expand.py")
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
t=e.t;p=torch.jit.load(str(t.template)).eval()
pushes=[]
for strength in (5.,10.):
    for name,direction in (("forward",[1,0,0]),("reverse",[-1,0,0]),("left",[0,1,0]),("right",[0,-1,0])):
        pushes.append((name+str(int(strength)),{"force_control_n":(strength*np.array(direction)).tolist(),"start_tick":150,"end_tick":158}))
for variant,hz in (("push_unmodified",(0.,0.)),("push_leg6_wheel2",(6.,2.))):
    for label,push in pushes:
        for mode,c in (("stand",[0.,0.,.2]),("forward",[1.,0.,.2])):
            e.run(p,variant,label+"_"+mode,lambda tick,c=c:c,750,hz,push=push)
assert np.all(t.runtime.data.xfrc_applied==0)
np.savez_compressed(OUT/"push-trajectories.npz",**e.arrays)
assert all(t.sha256_file(Path(p))==h for p,h in t.protected.items())
(OUT/"push-results.json").write_text(json.dumps({"results":e.results,"force_duration_s":.16,
    "force_application":"original base body's COM, world force derived from reset control axes; no torque; cleared outside 3.00..3.16 seconds and on exit",
    "external_force_is_diagnostic_only":True,"training_started":False,"production_modified":False,"protected_files_unchanged":True},indent=2),encoding="utf-8")
print("DONE",len(e.results),sum(r["ticks"] for r in e.results),flush=True)
