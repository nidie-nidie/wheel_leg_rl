from pathlib import Path
import argparse,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument("--experiment",type=Path,required=True);args=parser.parse_args()
data=np.load(args.experiment/"stationary-plot-data.npz")
plt.rcParams["font.family"]="Microsoft YaHei";plt.rcParams["axes.unicode_minus"]=False
fig,axes=plt.subplots(3,3,figsize=(15,9),sharex=True,sharey="row")
for i,label in enumerate(["old_1000","new_1000","old_1401_reference"]):
    t=data[label+"_time"];target=data[label+"_target"];velocity=data[label+"_velocity"]
    axes[0,i].plot(t,np.rad2deg(target[:,0]),label="左前关节目标")
    axes[0,i].plot(t,np.rad2deg(target[:,1]),label="左后关节目标")
    axes[1,i].plot(t,velocity[:,0],label="左前实际角速度")
    axes[1,i].plot(t,velocity[:,1],label="左后实际角速度")
    axes[2,i].plot(t,data[label+"_vx"],label="真实前后速度")
    axes[0,i].set_title(label);axes[2,i].set_xlabel("模拟时间（秒）")
    for ax in axes[:,i]:ax.grid(alpha=.2);ax.legend(fontsize=8)
for ax,text in zip(axes[:,0],["目标角度（度）","实际角速度（rad/s）","车速（m/s）"]):ax.set_ylabel(text)
fig.suptitle("相同初态和停车指令：新旧策略的左腿与车速\n定量比较排除前2秒；还需检查饱和与其他运动场景")
fig.tight_layout();fig.savefig(args.experiment/"stationary-comparison.png",dpi=150);plt.close(fig)
print(args.experiment/"stationary-comparison.png")
