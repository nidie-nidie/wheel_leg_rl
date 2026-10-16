"""Summarize recorded joint motion without assuming the source of jitter."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("session", type=Path)
args = parser.parse_args()
manifest = json.loads((args.session / "telemetry_manifest.json").read_text(encoding="utf-8"))
snapshot = (args.session / "control.csv").read_text(encoding="utf-8")
snapshot = snapshot[:snapshot.rfind("\n") + 1]
rows = list(csv.DictReader(io.StringIO(snapshot)))
if not rows:
    raise SystemExit("No control frames recorded yet")
data = {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}
names = manifest["joint_order"]
same_episode = np.diff(data["episode_id"]) == 0
summary = {"control_frames": len(rows), "episode_count": len(np.unique(data["episode_id"])),
           "source_completed_iterations": manifest["source_completed_iterations"], "joints": {},
           "caution": "A changing policy output is an observed boundary, not proof of the ultimate cause; forces include the closed chain."}
for index, name in enumerate(names):
    raw, clipped = data[name + "_action_raw"], data[name + "_action_clipped"]
    delta = np.diff(clipped)[same_episode]
    if index < 4:
        error = data[name + "_position_after_rad"] - data[name + "_position_target_rad"]
        unit = "rad"
    else:
        error = data[name + "_velocity_after_rad_s"] - data[name + "_velocity_target_rad_s"]
        unit = "rad_s"
    summary["joints"][name] = {
        "raw_action_min": float(raw.min()), "raw_action_max": float(raw.max()),
        "action_clipped_fraction": float(np.mean(np.abs(raw) >= 1.)),
        "mean_abs_action_change": float(np.mean(np.abs(delta))) if len(delta) else 0.,
        "max_abs_action_change": float(np.max(np.abs(delta))) if len(delta) else 0.,
        "large_action_change_fraction_gt_0_5": float(np.mean(np.abs(delta) > .5)) if len(delta) else 0.,
        "target_tracking_mae": float(np.mean(np.abs(error))), "target_tracking_unit": unit,
        "velocity_rms_rad_s": float(np.sqrt(np.mean(data[name + "_velocity_after_rad_s"] ** 2))),
        "torque_last_limit_fraction": float(np.mean(np.abs(data[name + "_torque_last_nm"]) >= (18. if index < 4 else 9.) - 1e-9)),
    }
events = [json.loads(line) for line in (args.session / "events.jsonl").read_text(encoding="utf-8").splitlines()]
summary["jitter_marks"] = [event for event in events if event["event"] == "jitter_mark"]
fig = plt.figure(figsize=(16, 14))
grid = fig.add_gridspec(5, 2)
time_axis = data["wall_elapsed_s"]
ax = fig.add_subplot(grid[0, :])
ax.plot(time_axis, data["command_vx_mps"], label="command vx")
ax.plot(time_axis, data["vx_mps"], label="actual vx")
ax.plot(time_axis, data["command_yaw_rad_s"], label="command yaw", alpha=.65)
ax.set_ylabel("m/s or rad/s")
ax.legend()
for side, group in enumerate((names[:2], names[2:4])):
    for plot_row in range(1, 5):
        ax = fig.add_subplot(grid[plot_row, side])
        for name in group:
            if plot_row == 1:
                ax.plot(time_axis, data[name + "_position_target_rad"], label=name + " target")
                ax.plot(time_axis, data[name + "_position_after_rad"], "--", label=name + " actual", alpha=.8)
                label = "position (rad)"
            elif plot_row == 2:
                ax.plot(time_axis, data[name + "_action_raw"], label=name + " raw")
                ax.plot(time_axis, data[name + "_action_clipped"], "--", label=name + " clipped", alpha=.8)
                label = "policy action"
            elif plot_row == 3:
                ax.plot(time_axis, data[name + "_velocity_after_rad_s"], label=name)
                label = "velocity (rad/s)"
            else:
                ax.plot(time_axis, data[name + "_torque_last_nm"], label=name)
                label = "last substep torque (Nm)"
        ax.set_ylabel(label)
        ax.set_xlabel("wall elapsed (s)")
        ax.legend(fontsize=8)
        for event in summary["jitter_marks"]:
            ax.axvline(event["wall_elapsed_s"], color="red", alpha=.5)
        for event in events:
            if event["event"] == "reset" and event["episode_id"] > 0:
                ax.axvline(event["wall_elapsed_s"], color="gray", linestyle=":", alpha=.5)
fig.suptitle("Left vs right leg telemetry; upper left/right joints are jIJ/jIO and jAB/jAG")
fig.tight_layout()
fig.savefig(args.session / "joint-traces.png", dpi=150)
plt.close(fig)
(args.session / "joint-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary), flush=True)
