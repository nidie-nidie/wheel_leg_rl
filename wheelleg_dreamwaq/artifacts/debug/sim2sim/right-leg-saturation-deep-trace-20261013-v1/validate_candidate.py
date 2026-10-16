"""Validate promising faster-leg/slower-wheel branch on full scenarios and new commands."""
from pathlib import Path
import importlib.util,json
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("expand",OUT/"expand.py")
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
t=e.t;policy=torch.jit.load(str(t.template)).eval()
payload=e.validate_command_practice(t.manifest)
for variant,hz in (("leg6_wheel2",(6.,2.)),("leg5_wheel2",(5.,2.)),("leg8_wheel2",(8.,2.)),("leg6_wheel1p5",(6.,1.5)),("leg6_wheel2p5",(6.,2.5))):
    for name,c in e.FORMAL_SCENARIOS:
        e.run(policy,variant,name,lambda tick,c=c:c,500,hz)
    for name,c in e.SCENARIOS:
        e.run(policy,variant,name,lambda tick,c=np.array(c):e.command_at_tick(c,tick,payload),500,hz)
sequence=np.array([[0,0,.2],[1,0,.2],[0,0,.2],[-1,0,.2],[0,0,.2],[.8,.5,.2],[0,0,.2],[0,-.6,.2],[0,0,.2],[0,0,.17],[0,0,.23],[0,0,.2]])
for variant,hz in (("leg6_wheel2",(6.,2.)),("leg8_wheel2",(8.,2.))):
    e.run(policy,variant,"long_stand",lambda tick:[0,0,.2],3000,hz)
    e.run(policy,variant,"long_mixed",lambda tick:sequence[min(tick//250,len(sequence)-1)],3000,hz)
# Independent command choices, fixed diagnostic seed; original command bounds, no model changes.
rng=np.random.default_rng(20261013)
commands=np.c_[rng.uniform(-1.1,1.1,12),rng.uniform(-.65,.65,12),rng.uniform(.17,.23,12)]
(OUT/"holdout-commands.json").write_text(json.dumps(commands.tolist(),indent=2),encoding="utf-8")
for variant,hz in (("holdout_unmodified",(0.,0.)),("holdout_leg6_wheel2",(6.,2.))):
    for i,c in enumerate(commands):e.run(policy,variant,f"command_{i:02d}",lambda tick,c=c:c,750,hz)
np.savez_compressed(OUT/"candidate-validation-trajectories.npz",**e.arrays)
assert all(t.sha256_file(Path(p))==h for p,h in t.protected.items())
(OUT/"candidate-validation-results.json").write_text(json.dumps({"results":e.results,"holdout_command_seed":20261013,
    "holdout_was_not_used_for_bandwidth_selection":True,"not_formal_acceptance":True,
    "training_started":False,"production_modified":False,"protected_files_unchanged":True},indent=2),encoding="utf-8")
print("DONE",len(e.results),sum(r["ticks"] for r in e.results),flush=True)
