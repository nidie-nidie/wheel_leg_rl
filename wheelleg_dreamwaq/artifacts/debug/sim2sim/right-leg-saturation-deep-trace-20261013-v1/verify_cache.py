"""Check diagnostic cache identity without touching original cases."""
from pathlib import Path
import importlib.util,uuid,json
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("expanded",OUT/"expand.py")
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
policy=e.t.torch.jit.load(str(e.t.template)).eval()
name="cache_identity_smoke_"+uuid.uuid4().hex[:8]
r=e.run(policy,name,"smoke",lambda tick:[0,0,.2],3,(6.,2.))
same=e.run(policy,name,"smoke",lambda tick:[0,0,.2],3,(6.,2.))
assert r==same
rejections={}
for tag,command,hz,mode in (("changed_commands",lambda tick:[.1,0,.2],(6.,2.),None),
                            ("changed_bandwidth",lambda tick:[0,0,.2],(8.,2.),None),
                            ("changed_velocity_mode",lambda tick:[0,0,.2],(6.,2.),"true_vx")):
    try:e.run(policy,name,"smoke",command,3,hz,velocity_mode=mode)
    except ValueError:rejections[tag]=True
    else:raise AssertionError("mismatched cache accepted")
protected=json.loads((OUT/"protected-before.json").read_text())
assert all(e.t.sha256_file(Path(p))==h for p,h in protected.items())
(OUT/"cache-verification.json").write_text(json.dumps({"same_inputs_reused":True,"mismatched_inputs_rejected":rejections,
 "case_variant":name,"production_modified":False,"training_started":False},indent=2),encoding="utf-8")
print(json.dumps(rejections))
