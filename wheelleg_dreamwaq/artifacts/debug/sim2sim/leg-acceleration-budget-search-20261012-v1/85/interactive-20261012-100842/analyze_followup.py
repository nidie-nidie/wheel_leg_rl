from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=Path('artifacts/debug/sim2sim/leg-acceleration-budget-search-20261012-v1/85/interactive-20261012-100842')
def read(path):
    with path.open(newline="") as f: rows=list(csv.DictReader(f))
    return {k:np.asarray([float(r[k]) for r in rows]) for k in rows[0]}
manifest=json.loads((p/'telemetry_manifest.json').read_text())
control_dt=float(manifest['control_dt_s'])
physics_dt=float(manifest['physics_dt_s'])
d=read(p/'control.csv')
x=read(p/'physics.csv')
names=('jIJ','jIO','jAB','jAG','jwheel_left','jwheel_right')
mask=(d['time_after_s']>2)&(d['time_after_s']<=20)
mask2=(d['time_after_s']>26)&(d['time_after_s']<=42)
pmask=(x['time_after_s']>2)&(x['time_after_s']<=20)
def freq(v,dt):
    v=v-v.mean()
    spectrum=np.abs(np.fft.rfft(v*np.hanning(len(v))))**2
    f=np.fft.rfftfreq(len(v),dt)
    return float(f[np.argmax(spectrum[1:])+1])
def region(mask):
    return {
      'ticks':int(mask.sum()),
      'vx_mae_mps':float(np.mean(np.abs(d['vx_mps'][mask]-d['command_vx_mps'][mask]))),
      'yaw_rate_mae_rad_s':float(np.mean(np.abs(d['yaw_rate_rad_s'][mask]-d['command_yaw_rad_s'][mask]))),
      'yaw_change_deg':float(np.rad2deg(np.unwrap(d['yaw_rad'][mask])[-1]-np.unwrap(d['yaw_rad'][mask])[0])),
      'roll_p5_p95_deg':np.rad2deg(np.percentile(d['roll_rad'][mask],[5,95])).tolist(),
      'pitch_p5_p95_deg':np.rad2deg(np.percentile(d['pitch_rad'][mask],[5,95])).tolist(),
      'cenet_vx_error_mae_mps':float(np.mean(np.abs(d['estimated_vx_before_action_mps'][1:][mask[1:]]-d['vx_mps'][:-1][mask[1:]]))),
      'max_tilt_deg':float(np.rad2deg(d['tilt_rad'][mask]).max()),
    }
report={'schema':'Interactive85JitterFollowupV1','control_dt_s':control_dt,'physics_dt_s':physics_dt,'control_rows':len(d['time_after_s']),'physics_rows':len(x['time_after_s']),'input_hashes':{n:hashlib.sha256((p/n).read_bytes()).hexdigest().upper() for n in ('control.csv','physics.csv','events.jsonl','telemetry_manifest.json')},
        'zero_command_initial_2_to_20s':region(mask),'zero_command_after_reversal_26_to_42s':region(mask2),
        'joints':{},'visible_jitter_resolved':False,'causal_root_cause_established':False}
for i,n in enumerate(names):
    v=d[n+'_velocity_after_rad_s'][mask]
    a=((d[n+'_velocity_after_rad_s']-d[n+'_velocity_before_rad_s'])/control_dt)[mask]
    pa=((x[n+'_velocity_after_rad_s']-x[n+'_velocity_before_rad_s'])/physics_dt)[pmask]
    raw=d[n+'_action_raw'][mask]
    clipped=d[n+'_action_clipped'][mask]
    flips=(clipped[1:]*clipped[:-1]<0)
    q=d[n+'_position_after_rad'][mask]
    torque=x[n+'_torque_command_nm'][pmask]
    report['joints'][n]={
        'raw_action_saturation_fraction':float(np.mean(np.abs(raw)>=1)),
        'clipped_action_sign_changes_per_s':float(flips.sum()/18),
        'actual_q_p5_p95_deg':np.rad2deg(np.percentile(q,[5,95])).tolist() if i<4 else None,
        'actual_qdot_rms_rad_s':float(np.sqrt(np.mean(v*v))),
        'actual_qdot_peak_spectral_frequency_hz':freq(v,control_dt),
        'control_endpoint_accel_rms_rad_s2':float(np.sqrt(np.mean(a*a))),
        'physics_endpoint_accel_rms_rad_s2':float(np.sqrt(np.mean(pa*pa))),
        'effort_saturation_fraction_all_substeps':float(np.mean(np.abs(torque)>=(18 if i<4 else 9)-1e-9)),
        'torque_sign_changes_per_s':float(np.sum(torque[1:]*torque[:-1]<0)/18),
    }
    if i<4:
        target=d[n+'_position_target_rad'][mask]
        report['joints'][n]['target_p5_p95_deg']=np.rad2deg(np.percentile(target,[5,95])).tolist()
        report['joints'][n]['target_peak_spectral_frequency_hz']=freq(target,control_dt) if np.ptp(target)>1e-12 else None
    else:
        target=d[n+'_velocity_target_rad_s'][mask]
        report['joints'][n]['target_p5_p95_rad_s']=np.percentile(target,[5,95]).tolist()
(p/'jitter-followup.json').write_text(json.dumps(report,indent=2,sort_keys=True))
print(json.dumps(report,indent=2))
zoom=(d['time_after_s']>=4)&(d['time_after_s']<=6)
fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True,layout='constrained')
t=d['time_after_s'][zoom]
for n in names[:2]:
    axes[0].plot(t,np.rad2deg(d[n+'_position_target_rad'][zoom]),label=n+' target')
    axes[0].plot(t,np.rad2deg(d[n+'_position_after_rad'][zoom]),'--',label=n+' actual')
axes[0].set_ylabel('Left joints (deg)');axes[0].legend(ncol=2)
for n in names[4:]:
    axes[1].plot(t,d[n+'_velocity_target_rad_s'][zoom],label=n+' target')
    axes[1].plot(t,d[n+'_velocity_after_rad_s'][zoom],'--',label=n+' actual')
    axes[2].plot(t,d[n+'_torque_last_nm'][zoom],label=n)
axes[1].set_ylabel('Wheel velocity (rad/s)');axes[1].legend(ncol=2)
axes[2].set_ylabel('Wheel final torque (Nm)');axes[2].legend()
axes[3].plot(t,d['vx_mps'][zoom],label='actual vx')
axes[3].plot(t,d['estimated_vx_before_action_mps'][zoom],label='CENet vx before action')
axes[3].plot(t,d['yaw_rate_rad_s'][zoom],label='actual yaw rate')
axes[3].set_ylabel('m/s or rad/s');axes[3].set_xlabel('Simulation time (s)');axes[3].legend()
fig.suptitle('85% interactive capture: zero command, 4-6 seconds; no causal claim')
fig.savefig(p/'jitter-followup.png',dpi=160)

