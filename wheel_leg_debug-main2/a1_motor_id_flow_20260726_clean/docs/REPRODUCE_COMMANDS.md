# Reproduce Commands

These commands document the actual flow. The Linux commands use the original `changba01` paths from this work and require the external IsaacLab/PACE environment; this deliverable directory alone is not enough to rerun IsaacSim fitting. Local package paths are shown only for plotting already packaged data.

## 1. Convert A1 Wide Chirp CSV

```bash
cd /home/changba01/pace-sim2real

/home/changba01/IsaacLab/_isaac_sim/python.sh scripts/pace/a1_convert_wide_chirp.py \
  --input /home/changba01/pace_identification/input/A1chirp_20260726_071950_f0p10-f1p10Hz_d20_r2_Kp25_Kd2_ampHip0p22_ampTh0p38_ampCalf0p38_cH0p10_cFT0p80_cRT1p00_cCalf-1p50_v25_a1600.csv \
  --output /home/changba01/pace-sim2real/data/a1_real_wide/A1chirp_20260726_071950_f0p10-f1p10Hz_d20_r2_Kp25_Kd2_ampHip0p22_ampTh0p38_ampCalf0p38_cH0p10_cFT0p80_cRT1p00_cCalf-1p50_v25_a1600.pt \
  --manifest-output /home/changba01/pace-sim2real/data/a1_real_wide/A1chirp_20260726_071950_f0p10-f1p10Hz_d20_r2_Kp25_Kd2_ampHip0p22_ampTh0p38_ampCalf0p38_cH0p10_cFT0p80_cRT1p00_cCalf-1p50_v25_a1600.manifest.json \
  --fixture-pose upside_down_fixed_air \
  --payload-kg 0 \
  --support back_supported_fixed_table_trunk_footprint
```

Note: the packaged `064152`, `071554`, and `071950` manifests were generated before the `--payload-kg 0` correction and show `5.0`. The packaged `062823` and `063144` conversion manifests already show `0.0`.

## 2. Formal PACE Fit

```bash
cd /home/changba01/pace-sim2real

nohup env \
  PYTHONPATH=/home/changba01/pace-sim2real/source/pace_sim2real \
  /home/changba01/IsaacLab/isaaclab.sh -p scripts/pace/fit.py \
    --task Isaac-Pace-A1-v0 \
    --num_envs 4096 \
    --headless \
    --run_name formal_4096_headless_kd2_20260726_025007 \
  > /home/changba01/pace-sim2real/data/a1_real_wide/formal_4096_headless_kd2_20260726_025007.log 2>&1 &
```

The formal environment contract recorded by the posthoc manifest:

```text
dt=0.002
decimation=1
fix_root_link=true
fixture_pose=upside_down_fixed_air
Kp=25
Kd=2
num_envs=4096
```

## 3. Posthoc Fit Manifest

```bash
cd /home/changba01/pace-sim2real

/home/changba01/IsaacLab/_isaac_sim/python.sh scripts/pace/a1_posthoc_fit_manifest.py \
  --run-dir /home/changba01/pace-sim2real/logs/pace/a1/26_07_26_02-50-16 \
  --output /home/changba01/pace-sim2real/logs/pace/a1/26_07_26_02-50-16/a1_pace_fit_manifest.posthoc.json
```

Pass criteria:

```text
objective_rmse <= 0.05 rad
direct_rmse <= 0.08 rad
```

For a follow-up fit with the encoder-bias search range narrowed to one third of
the original bounds, use `docs/RERUN_BIAS_THIRD.md`.

## 4. Cross-Validate One Mean Against Other Data

```bash
cd /home/changba01/pace-sim2real

MEAN=/home/changba01/pace-sim2real/logs/pace/a1/26_07_26_02-50-16/mean_199.pt

/home/changba01/IsaacLab/isaaclab.sh -p scripts/pace/a1_crossval_replay_fixed_mean.py \
  --headless \
  --task Isaac-Pace-A1-v0 \
  --mean "$MEAN" \
  --data /home/changba01/pace-sim2real/data/a1_real_wide/A1chirp_20260726_062823_f0p10-f1p4Hz_d20_r2_Kp25_Kd2_ampHip0p20_ampTh0p30_ampCalf0p30_cH0p10_cFT0p80_cRT1p00_cCalf-1p50_v20_a250.pt \
  --output /home/changba01/pace-sim2real/data/a1_real_wide/replay_cv_062823_from_071950.json \
  --trajectory-output /home/changba01/pace-sim2real/data/a1_real_wide/replay_cv_062823_from_071950.pt
```

Repeat with the other `.pt` files. The packaged reports are in `results/crossval/`.

## 5. Build Cross-Validation Visualization

```bash
cd /home/changba01/pace-sim2real

/home/changba01/IsaacLab/_isaac_sim/python.sh scripts/pace/a1_build_crossval_vis_data.py \
  --output /home/changba01/pace-sim2real/data/a1_real_wide/a1_fixed_mean_crossval_vis_data.json \
  --reports /home/changba01/pace-sim2real/data/a1_real_wide/replay_cv_*.json \
  --trajectories /home/changba01/pace-sim2real/data/a1_real_wide/replay_cv_*.pt

python scripts/pace/a1_build_crossval_vis_html.py \
  --data /home/changba01/pace-sim2real/data/a1_real_wide/a1_fixed_mean_crossval_vis_data.json \
  --output /home/changba01/pace-sim2real/data/a1_real_wide/a1_fixed_mean_crossval_vis.html
```

Packaged HTML:

```text
results/crossval/a1_fixed_mean_crossval_vis_standalone.html
```

## 6. Plot Raw Collection Command vs Real Joint Angle

On Windows/local mirror:

```powershell
python A1_rl\deliverables\a1_motor_id_flow_20260726_clean\scripts\visualization\plot_a1_chirp_joint_tracking.py `
  --csv A1_rl\deliverables\a1_motor_id_flow_20260726_clean\data\raw_csv\<A1chirp.csv> `
  --out-dir A1_rl\deliverables\a1_motor_id_flow_20260726_clean\results\raw_tracking_plots
```

This produces:

- `*_des_vs_dof_pos.png`
- `*_tracking_error.png`
- `*_tracking_summary.csv`
