"""Scoped zero-command drift correction; independent original targets retained."""
from pathlib import Path
import sys,importlib.util,json
import numpy as np,torch
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("slow_feedback",OUT/"slow_feedback.py")
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
for name,ki,delay in (("zero_trim015",[.15,.15],0.),("zero_trim030",[.3,.3],0.),("zero_trim_delay",[.3,.3],.5),("zero_trim_vx",[.3,0.],0.)):
    s.trial(name,"stand60",lambda tick:[0,0,.2],3000,[0,0],ki,zero_only=True,settled_delay_s=delay)
    for scenario,c in s.SCENARIOS:
        s.trial(name,scenario,lambda tick,c=np.array(c):s.command_at_tick(c,tick,s.payload),500,[0,0],ki,zero_only=True,settled_delay_s=delay)
np.savez_compressed(OUT/"zero-trim-trajectories.npz",**s.arrays)
protected=json.loads((OUT/"protected-before.json").read_text());assert all(s.sha256_file(Path(p))==h for p,h in protected.items())
(OUT/"zero-trim-results.json").write_text(json.dumps({"results":s.results,"correction_limits":[.06,.06],
    "zero_only":"integrate and apply correction only when both original commanded vx/yaw are zero; retained bounded integral during motion; all original sensor/action history retained",
    "metrics_target":"original user command","not_formal_acceptance":True,"training_started":False,"production_modified":False,"protected_files_unchanged":True},indent=2),encoding="utf-8")
print("DONE",len(s.results),sum(r["ticks"] for r in s.results),flush=True)
