"""Read-only offline audit; writes only new debug artifacts."""
from pathlib import Path
import hashlib
import json
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = Path(__file__).resolve().parent
EXP = OUT.parent
JOINTS = ["jIJ", "jIO", "jAB", "jAG"]
LABELS = ["baseline", "leg02"]
MAIN = ["nominal_stand", "forward", "reverse"]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def arr(x): return np.asarray(x).tolist()
def longest(mask):
    d = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return float((ends-starts).max()*.02) if len(starts) else 0.
def peak(x, dt):
    power = np.abs(np.fft.rfft((x-x.mean(0))*np.hanning(len(x))[:,None],axis=0))**2
    f = np.fft.rfftfreq(len(x),dt)
    keep = (f>=.5)&(f<=25)
    peaks = f[keep][power[keep].argmax(0)]
    return [float(peaks[j]) if np.ptp(x[:, j]) > 1e-10 else None for j in range(x.shape[1])]

def main():
    report = {"schema_version":"SplitSmoothFailureAnalysisV1","joint_order":JOINTS,
              "warmup_control_ticks":100,"offline_only":True,
              "scenarios":{},"training":{},"static_scenes":{},"command_practice":{},"input_sha256":{}}
    for label in LABELS:
        report["scenarios"][label]={}
        for scene in MAIN:
            path=EXP/label/"physics1ms"/(scene+".npz")
            report["input_sha256"][str(path)]=sha(path)
            with np.load(path) as z:
                raw=z["raw_action"]; clipped=np.clip(raw,-1,1)
                c=z["control"]; p=z["physics"]; a=p[p[:,0]>=100,10:14]
                assert len(raw)==len(c)==500 and len(p)==10000
                np.testing.assert_array_equal(p[19::20,2:6],c[:,1:5])
                second=(clipped[2:]-2*clipped[1:-1]+clipped[:-2])[98:]
                smooth=(second**2).mean(0)
                first=np.diff(clipped,axis=0)[99:]
                v=slice(100,None); target=c[v,9:13]
                report["scenarios"][label][scene]={
                    "first_action_raw":arr(raw[0]),"raw_mean":arr(raw[v].mean(0)),
                    "raw_min":arr(raw[v].min(0)),"raw_max":arr(raw[v].max(0)),
                    "negative_clip_fraction":arr((raw[v]<=-1).mean(0)),
                    "positive_clip_fraction":arr((raw[v]>=1).mean(0)),
                    "longest_saturated_run_s":[longest(np.abs(raw[v,j])>=1) for j in range(6)],
                    "clipped_std":arr(clipped[v].std(0)),
                    "second_difference_square_mean":arr(smooth),
                    "first_difference_square_mean":arr((first**2).mean(0)),
                    "original_smooth_per_step":float(-.01*.02*smooth.sum()),
                    "split_smooth_per_step":float(.02*(-.02*smooth[:4].sum()-.01*smooth[4:].sum())),
                    "leg_smooth_raw_sum":float(smooth[:4].sum()),"wheel_smooth_raw_sum":float(smooth[4:].sum()),
                    "target_range_deg":arr(np.rad2deg(np.ptp(target,axis=0))),
                    "target_mean_rad":arr(target.mean(0)),
                    "actual_q_range_deg":arr(np.rad2deg(np.ptp(c[v,1:5],axis=0))),
                    "target_tracking_error_rms_rad":arr(np.sqrt(((target-c[v,1:5])**2).mean(0))),
                    "actual_acceleration_pooled_rms_1ms":arr(np.sqrt((a**2).mean(0))),
                    "actual_velocity_peak_hz":peak(p[p[:,0]>=100,6:10],.001),
                    "target_peak_hz":peak(target,.02),
                    "vx_mae_mps":float(np.abs(c[v,13]-{"nominal_stand":0,"forward":1,"reverse":-1}[scene]).mean()),
                    "mean_tilt_rad":float(c[v,14].mean())}
        for kind in ["motion20ms","command_practice"]:
            path=EXP/label/kind/"summary.json"
            report["input_sha256"][str(path)]=sha(path)
            summary=json.loads(path.read_text())
            report["static_scenes" if kind=="motion20ms" else "command_practice"][label]=summary["scenarios"]
            if kind=="motion20ms": run=Path(summary["source_checkpoint"]).parent
        accumulator=EventAccumulator(str(run),size_guidance={"scalars":0}); accumulator.Reload()
        selected=[t for t in accumulator.Tags()["scalars"] if any(s in t for s in
            ["Reward/","Tracking/","Episode/action_smooth","Policy/mean_noise","Joint/","Train/mean_reward"])]
        summary={}
        for tag in selected:
            vals=accumulator.Scalars(tag)
            summary[tag]={str(end):float(np.mean([v.value for v in vals if end-49<=v.step<=end]))
                for end in [99,199,499,999] if any(end-49<=v.step<=end for v in vals)}
        report["training"][label]={"run_dir":str(run),"scalar_windows":summary}
        for path in run.glob("events.out.tfevents.*"): report["input_sha256"][str(path)]=sha(path)
    (OUT/"analysis.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(13,10),sharex=True)
    for col,label in enumerate(LABELS):
        with np.load(EXP/label/"physics1ms/nominal_stand.npz") as z:
            c=z["control"]; c=c[(c[:,0]>=2)&(c[:,0]<=4)]
            for row,joint in enumerate(JOINTS):
                ax=axes[row,col]; ax.plot(c[:,0],c[:,9+row],label="target")
                ax.plot(c[:,0],c[:,1+row],label="actual"); ax.set_ylabel(joint+" (rad)"); ax.grid(alpha=.25)
                if row==0: ax.set_title(label+" / nominal stand"); ax.legend()
    for ax in axes[-1]: ax.set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(OUT/"target-and-motion.png",dpi=160); plt.close(fig)
    print("offline analysis complete; six trajectories and two TensorBoard runs")

if __name__=="__main__": main()

