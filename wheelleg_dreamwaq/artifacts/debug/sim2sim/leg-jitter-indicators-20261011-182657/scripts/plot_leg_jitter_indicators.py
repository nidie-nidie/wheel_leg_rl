from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")
root=Path(r"E:\wheel_leg_rl-main")
out=Path(__file__).resolve().parent.parent
report=json.loads((out/"indicators.json").read_text(encoding="utf-8"))
data=np.load(out/"plot-data.npz")
cases={c["label"]:c for c in report["cases"]}
names=["jIJ","jIO","jAB","jAG"];labels=["Left front (jIJ)","Left rear (jIO)","Right front (jAB)","Right rear (jAG)"]
colors=["#4169ae","#de8a36","#56966d"]
# Physical response and command variation use separate plots/scales; no mixed-unit threshold.
fig,axs=plt.subplots(1,2,figsize=(11,4.7),layout="constrained")
x=np.arange(4);width=.24
ids=["old_1000_parking","new_1000_parking","old_1401_reference_parking"]
for k,(name,title) in enumerate(zip(ids,["Old 1000","New 1000","Old 1401"])):
    values=[cases[name]["joint_metrics"][j]["actual_acceleration_20ms"]["rms"] for j in names]
    axs[0].bar(x+(k-1)*width,values,width,label=title,color=colors[k])
    for xx,v in zip(x+(k-1)*width,values):axs[0].text(xx,v+.8,f"{v:.1f}",ha="center",fontsize=8)
axs[0].set_xticks(x,["Left front","Left rear","Right front","Right rear"]);axs[0].set_ylabel("Actual acceleration RMS (rad/s²)");axs[0].set_ylim(0,61);axs[0].legend(fontsize=8);axs[0].set_title("Actual response over each 20 ms interval")
values=[cases[i]["left_smooth_mean"] for i in ids]
axs[1].bar(np.arange(3),values,color=colors)
for k,v in enumerate(values):axs[1].text(k,v+.001,f"{v:.5f}",ha="center",fontsize=10)
axs[1].set_xticks(np.arange(3),["Old 1000","New 1000","Old 1401"]);axs[1].set_ylabel("Left-leg mean squared action second difference");axs[1].set_ylim(0,.049);axs[1].set_title("Command variation: legs only")
for ax in axs:ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
fig.suptitle("Same parking command; 2–10 s; initial reset excluded",fontsize=13)
fig.savefig(out/"parking-indicator-comparison.png",dpi=180);plt.close(fig)
def plot_trace(label,start,end,path,title):
    t=data[label+"_t"];ta=data[label+"_ta"]
    mask=(t>=start-1e-9)&(t<end-1e-9)
    velocity_limit=1.1*np.max(np.abs(data[label+"_va"][mask]))
    acceleration_limit=1.1*np.max(np.abs(data[label+"_acc"][mask]))
    position_values=np.concatenate([data[label+"_target"][mask],data[label+"_qa"][mask]],axis=0)
    position_min=position_values.min(axis=0);position_max=position_values.max(axis=0)
    position_half_span=.55*np.max(position_max-position_min)
    fig,axs=plt.subplots(4,4,figsize=(13,9),sharex=True,layout="constrained")
    for j,name in enumerate(names):
        axs[0,j].plot(t[mask],data[label+"_a"][mask,j],color=colors[0],lw=1)
        axs[0,j].set_title(labels[j]);axs[0,j].set_ylim(-1.08,1.08)
        axs[1,j].plot(t[mask],data[label+"_target"][mask,j],label="Target",color=colors[1],lw=1.2)
        axs[1,j].plot(ta[mask],data[label+"_qa"][mask,j],label="Actual (after interval)",color=colors[0],lw=1)
        axs[2,j].plot(ta[mask],data[label+"_va"][mask,j],color=colors[0],lw=1)
        axs[3,j].plot((t[mask]+ta[mask])/2,data[label+"_acc"][mask,j],color=colors[0],lw=1)
        position_center=(position_min[j]+position_max[j])/2
        axs[1,j].set_ylim(position_center-position_half_span,position_center+position_half_span)
        axs[2,j].set_ylim(-velocity_limit,velocity_limit)
        axs[3,j].set_ylim(-acceleration_limit,acceleration_limit)
        for r in range(4):
            axs[r,j].grid(alpha=.2)
            if r==3:axs[r,j].set_xlabel("Simulation time (s)")
    for r,labely in enumerate(["Clipped action","Position (rad)","Velocity (rad/s)","20 ms accel. (rad/s²)"]):axs[r,0].set_ylabel(labely)
    axs[1,0].legend(fontsize=7)
    fig.suptitle(title+"\nEqual axis spans across joints; command at interval start, actual state at interval end",fontsize=13)
    fig.savefig(out/path,dpi=170);plt.close(fig)
plot_trace("new_1000_parking",2,4,"parking-aligned-traces.png","New 1000: constant zero speed command, 2–4 s")
plot_trace("new_forward",2,4,"forward-aligned-traces.png","New 1000: constant +1 m/s command, 2–4 s")
plot_trace("new_reverse",2,4,"reverse-aligned-traces.png","New 1000: constant −1 m/s command, 2–4 s")
# Constant parking spectra: independently normalize each channel to expose periodicity, not relative power.
label="new_1000_parking";ix=data[label+"_indices"]
fig,axs=plt.subplots(1,2,figsize=(10,4.4),layout="constrained")
for j,ax in enumerate(axs):
    for suffix,text,color in [("a","Action",colors[1]),("vb","Actual velocity",colors[0])]:
        v=data[label+"_"+suffix][ix,j];v=v-v.mean()
        power=np.abs(np.fft.rfft(v*np.hanning(len(v))))**2;power[0]=0
        freq=np.fft.rfftfreq(len(v),.02)
        power/=power.max()
        ax.plot(freq,power,label=text,color=color)
    ax.set_title(labels[j]);ax.set_xlim(0,25);ax.set_xlabel("Frequency (Hz)");ax.set_ylabel("Normalized power (each channel)");ax.grid(alpha=.2);ax.legend()
fig.suptitle("Parking: action and actual velocity share a peak near 8.8 Hz\nSeparate normalization shows frequency, not equal amplitude",fontsize=12)
fig.savefig(out/"parking-spectrum.png",dpi=180);plt.close(fig)
# A full command-practice timeline, with time intervals for acceleration centered correctly.
label="new_forward_stop_reverse";t=data[label+"_t"];ta=data[label+"_ta"];mid=(t+ta)/2
fig,axs=plt.subplots(4,1,figsize=(11,9),sharex=True,layout="constrained")
axs[0].step(t,data[label+"_command"][:,0],where="post",label="Target vx",color="black",lw=1.4);axs[0].plot(ta,data[label+"_vx"],label="Actual vx",color=colors[0]);axs[0].set_ylabel("vx (m/s)")
for j in range(2):
    axs[1].plot(t,data[label+"_a"][:,j],label=labels[j],color=colors[j],lw=1)
    axs[2].plot(t,np.abs(data[label+"_d2"][:,j]),label=labels[j],color=colors[j],lw=1)
    axs[3].plot(mid,data[label+"_acc"][:,j],label=labels[j],color=colors[j],lw=1)
axs[1].set_ylabel("Left-leg action");axs[2].set_ylabel("|Action second difference|");axs[3].set_ylabel("20 ms accel. (rad/s²)");axs[3].set_xlabel("Simulation time (s)")
for ax in axs:
    ax.axvspan(0,2,color="gray",alpha=.1,label="Initial reset (excluded)" if ax==axs[0] else None)
    for switch in [2,3,5,6,8]:ax.axvline(switch,color="gray",ls="--",alpha=.6)
    ax.grid(alpha=.2);ax.legend(loc="upper right",fontsize=8)
fig.suptitle("Original six-stage forward / stop / reverse diagnostic\nCommand transitions marked; normal braking and repeated oscillation kept distinct",fontsize=13)
fig.savefig(out/"command-practice-aligned-traces.png",dpi=180);plt.close(fig)
# Association across nonoverlapping 1 s windows is descriptive, not a health classifier.
fig,ax=plt.subplots(figsize=(8,5),layout="constrained")
for label,title,color in [("user_parking","Keyboard: parking","#4169ae"),("user_forward02","Keyboard: +0.2 m/s","#de8a36"),("new_1000_parking","Fixed parking","#56966d")]:
    windows=cases[label]["one_second_windows"]
    ax.scatter([w["left_smooth_mean"] for w in windows],[w["left_acceleration_norm_rms_rad_s2"] for w in windows],label=title,c=color,s=38,alpha=.8)
ax.set_xlabel("Left-leg mean action smooth cost (1 s)");ax.set_ylabel("Actual left-leg acceleration norm RMS (rad/s²)")
ax.set_title("Command variation and actual leg motion covary\nDescriptive windows; no acceptable / unacceptable labels")
ax.grid(alpha=.2);ax.legend()
fig.savefig(out/"indicator-association.png",dpi=180);plt.close(fig)
print(json.dumps({"output":str(out),"plots":[p.name for p in out.glob("*.png")]}))
