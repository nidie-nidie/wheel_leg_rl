import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, constrained_layout=True)
for row_index, engine in enumerate(("isaac", "mujoco")):
    with (ROOT / engine / "nominal_stand.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    def values(name):
        return np.asarray([float(row[name]) for row in rows])
    t = values("time_s")
    axes[row_index, 0].plot(t, values("actual_vx_mps"), label="Actual vx", lw=1.8)
    axes[row_index, 0].plot(t, values("estimated_vx_mps"), label="CENet vx", lw=1.4)
    axes[row_index, 0].axhline(0, color="black", ls="--", lw=.8, label="Command = 0")
    axes[row_index, 0].set_ylabel(engine.capitalize() + "\nvx (m/s)")
    for kind, label in (("target", "Policy target"), ("actual", "Actual wheel speed")):
        avg = (values(f"wheel_{kind}_left_rad_s") + values(f"wheel_{kind}_right_rad_s")) / 2
        axes[row_index, 1].plot(t, avg, label=label)
    axes[row_index, 1].set_ylabel("Wheel speed (rad/s, canonical)")
    axes[row_index, 2].plot(t, np.rad2deg(values("pitch_rad")), color="tab:purple", label="Pitch")
    axes[row_index, 2].set_ylabel("Pitch (degrees)")
    for ax in axes[row_index]:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 10)
for ax in axes[-1]:
    ax.set_xlabel("Time before action (s)")
fig.suptitle("Run-04, unchanged policy and physics: nominal standing command (0, 0, 0.20 m)")
fig.savefig(ROOT / "standing-comparison.png", dpi=180)
plt.close(fig)
print("STANDING_PLOT_COMPLETE")
