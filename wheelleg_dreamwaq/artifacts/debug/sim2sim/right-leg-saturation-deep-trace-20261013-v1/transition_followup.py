"""Follow failed transition stages: actuator bandwidth vs estimator accuracy."""
from pathlib import Path
import importlib.util,json
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("expand",OUT/"expand.py")
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
t=e.t;policy=torch.jit.load(str(t.template)).eval()
payload=e.validate_command_practice(t.manifest)
# Real feedback retained; only explicitly labeled oracle branches replace Actor estimated velocity.
for vm in ("true_vx","true3"):
    for name,hz in (("bypass",(0.,0.)),("both3",(3.,3.)),("legs3",(3.,0.))):
        variant=vm+"_"+name
        for scenario,c in e.FORMAL_SCENARIOS:
            e.run(policy,variant,scenario,lambda tick,c=c:c,500,hz,velocity_mode=vm)
        for scenario,c in e.SCENARIOS:
            e.run(policy,variant,scenario,lambda tick,c=np.array(c):e.command_at_tick(c,tick,payload),500,hz,velocity_mode=vm)
# Different leg and wheel bandwidths can retain faster wheel balance; test original CENet first.
for lh,wh in ((0,2),(0,3),(0,4),(2,4),(2,6),(4,2),(6,2),(6,3),(1,3),(1,4),(3,5),(3,6)):
    variant=f"leg{lh}_wheel{wh}_transition"
    for scenario,c in e.SCENARIOS:
        e.run(policy,variant,scenario,lambda tick,c=np.array(c):e.command_at_tick(c,tick,payload),500,(lh,wh))
np.savez_compressed(OUT/"transition-followup-trajectories.npz",**e.arrays)
assert all(t.sha256_file(Path(p))==h for p,h in t.protected.items())
(OUT/"transition-followup-results.json").write_text(json.dumps({"results":e.results,"true_velocity_is_simulator_oracle_not_deployable":True,
    "training_started":False,"production_modified":False,"protected_files_unchanged":True},indent=2),encoding="utf-8")
print("DONE",len(e.results),sum(r["ticks"] for r in e.results),flush=True)
