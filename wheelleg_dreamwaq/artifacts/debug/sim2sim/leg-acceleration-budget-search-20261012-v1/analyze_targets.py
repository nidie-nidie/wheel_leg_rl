from pathlib import Path
import json
import csv
import numpy as np
OUT=Path(__file__).resolve().parent
joints=("jIJ","jIO","jAB","jAG")
results={}
for arm in ("baseline","80","75","70","85","90"):
    if not (OUT/(arm+"-result.json")).exists(): continue
    results[arm]={}
    for name in ("nominal_stand","forward","reverse"):
        with (OUT/arm/"motion"/(name+".csv")).open(newline="") as stream: rows=list(csv.DictReader(stream))
        rows=[r for r in rows if int(r["tick"])>100 and r["survived"]=="1"]
        if not rows: continue
        target=np.array([[float(r[j+"_target_rad"]) for j in joints] for r in rows])
        actual=np.array([[float(r[j+"_q_rad"]) for j in joints] for r in rows])
        velocity=np.array([[float(r[j+"_qdot_rad_s"]) for j in joints] for r in rows])
        results[arm][name]={"target_range_rad":np.ptp(target,axis=0).tolist(),
                           "actual_range_rad":np.ptp(actual,axis=0).tolist(),
                           "actual_velocity_rms_rad_s":np.sqrt(np.mean(velocity**2,axis=0)).tolist(),
                           "target_constant":[bool(v<1e-12) for v in np.ptp(target,axis=0)],
                           "target_first_rad":target[0].tolist(),
                           "max_tilt_rad":max(float(r["tilt_rad"]) for r in rows)}
(OUT/"target-motion-diagnostics.json").write_text(json.dumps({"joint_order":joints,"arms":results},indent=2,sort_keys=True))
print(json.dumps(results,indent=2))

