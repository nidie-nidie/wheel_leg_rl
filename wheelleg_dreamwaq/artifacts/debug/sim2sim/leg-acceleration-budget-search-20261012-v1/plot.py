from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
m=json.loads((OUT/"comparison.json").read_text())
arms=[m["baseline"],*m["candidates"]]
labels=[a["arm"] for a in arms]
x=np.arange(len(arms))
fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout="constrained")
axes[0].bar(x,[a["primary_aggregate_rms_rad_s2"] if a["primary_aggregate_rms_rad_s2"] is not None else np.nan for a in arms])
axes[0].set_xticks(x,labels)
axes[0].set_ylabel("Actual acceleration RMS (rad/s^2)")
axes[0].set_title("Three primary scenarios, equal weight")
for i,j in enumerate(("jIJ","jIO","jAB","jAG")):
    axes[1].plot(x,[a["primary_joint_rms_means_rad_s2"][i] if a["primary_joint_rms_means_rad_s2"] else np.nan for a in arms],marker="o",label=j)
axes[1].set_xticks(x,labels)
axes[1].set_ylabel("Actual acceleration RMS (rad/s^2)")
axes[1].set_title("Individual leg joints")
axes[1].legend()
for i,n in enumerate(("nominal_stand","forward","reverse")):
    axes[2].bar(x+(i-1)*.22,[a["primary_steady_vx_mae_mps"][n] if a["primary_steady_vx_mae_mps"][n] is not None else np.nan for a in arms],width=.22,label=n)
axes[2].set_xticks(x,labels)
axes[2].set_ylabel("Steady speed MAE (m/s)")
axes[2].set_title("Speed tracking, 2-10 seconds")
axes[2].legend()
fig.suptitle("One seed, 200 iterations per arm; lower motion is not sufficient for acceptance")
fig.savefig(OUT/"budget-comparison.png",dpi=180)
print(OUT/"budget-comparison.png")

