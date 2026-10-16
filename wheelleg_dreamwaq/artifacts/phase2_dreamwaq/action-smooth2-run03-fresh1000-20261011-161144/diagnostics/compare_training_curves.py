from pathlib import Path
import argparse,json
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser()
parser.add_argument("--experiment",type=Path,required=True)
args=parser.parse_args()
record=json.loads((args.experiment/"experiment-manifest.json").read_text(encoding="utf-8"))
live=json.loads((args.experiment/"live-configuration-verification.json").read_text(encoding="utf-8"))
runs={"old":Path(record["baseline_record"]["checkpoint"]).parent,"new":Path(live["run_dir"])}
metrics=["Tracking/vx_abs_error","Tracking/yaw_rate_abs_error","Reward/action_smooth",
         "Reward/action_rate","Metric/phi0_delta_abs_rad","Action/saturation_fraction"]
events={}
for label,path in runs.items():
    event=EventAccumulator(str(path),size_guidance={"scalars":0}).Reload()
    assert set(metrics)<=set(event.Tags()["scalars"])
    events[label]={tag:np.array([[item.step,item.value] for item in event.Scalars(tag)]) for tag in metrics}
    assert all(np.isfinite(value).all() for value in events[label].values())
end=int(min(events["new"][metrics[0]][-1,0],999))
start=max(0,end-99)
summary={"window_iterations_inclusive":[start,end],"final_1000_complete":end==999,
         "scope":"Aggregate randomized training/exploration, not deterministic MuJoCo performance.",
         "normalization":"Reward/action_smooth divided by abs(weight)*0.02 to compare action second differences with changed weights.",
         "metrics":{}}
for tag in metrics:
    values={}
    for label in runs:
        data=events[label][tag]
        mask=(data[:,0]>=start)&(data[:,0]<=end)
        assert mask.sum()==end-start+1
        scale=1.
        if tag=="Reward/action_smooth":scale=-.02*(.01 if label=="old" else .02)
        if tag=="Reward/action_rate":scale=-.02*.01
        values[label]=float((data[mask,1]/scale).mean())
    summary["metrics"][tag]={"old":values["old"],"new":values["new"],"delta":values["new"]-values["old"]}
(args.experiment/"training-curve-comparison.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
plt.rcParams["font.family"]="Microsoft YaHei";plt.rcParams["axes.unicode_minus"]=False
fig,axes=plt.subplots(3,2,figsize=(13,10))
titles=["vx绝对误差（m/s）","yaw绝对误差（rad/s）","归一化动作二阶差分平方和","归一化动作一阶差分平方和","左右phi0绝对角差（rad）","动作饱和比例"]
for ax,tag,title in zip(axes.flat,metrics,titles):
    for label in runs:
        data=events[label][tag]
        scale=1.
        if tag=="Reward/action_smooth":scale=-.02*(.01 if label=="old" else .02)
        if tag=="Reward/action_rate":scale=-.02*.01
        values=data[:,1]/scale
        n=min(30,len(values))
        ax.plot(data[n-1:,0],np.convolve(values,np.ones(n)/n,mode="valid"),label=label)
    ax.set_title(title);ax.grid(alpha=.2);ax.legend()
for ax in axes[-1]:ax.set_xlabel("iteration（零基）")
fig.suptitle("相同种子训练对照：旧smooth=-0.01，新smooth=-0.02\n平滑指标去除权重与dt；训练指标含随机化和探索")
fig.tight_layout();fig.savefig(args.experiment/"training-curve-comparison.png",dpi=150);plt.close(fig)
print(json.dumps(summary,indent=2))
