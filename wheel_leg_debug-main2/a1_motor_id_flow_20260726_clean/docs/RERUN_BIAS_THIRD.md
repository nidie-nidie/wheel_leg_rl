# Re-run With One-Third Encoder-Bias Bounds

This re-identification keeps the original A1 PACE task and all non-bias
parameter bounds unchanged, but narrows the 12 `encoder_bias` dimensions from
the original `[-0.1, 0.1] rad` range to `[-0.03333333333333333,
0.03333333333333333] rad`.

Use a new `robot_name` so the run writes to a new log directory and does not
overwrite the packaged fit artifacts.

The handoff-package script is
`scripts/identification/a1_fit_narrow_encoder_bias.py`. In the original
PACE checkout, run the same script from the local scripts directory used by
that checkout; the commands below keep the original `scripts/pace/...` layout
used by the frozen run notes.

```bash
cd /home/changba01/pace-sim2real

env \
  PYTHONPATH=/home/changba01/pace-sim2real/source/pace_sim2real \
  /home/changba01/IsaacLab/isaaclab.sh -p scripts/pace/a1_fit_narrow_encoder_bias.py \
    --headless \
    --task Isaac-Pace-A1-v0 \
    --num_envs 4096 \
    --robot-name a1_bias_third_20260728_071950 \
    --bias-scale 0.3333333333333333 \
    --data /home/changba01/pace-sim2real/data/a1_real_wide/A1chirp_20260726_071950_f0p10-f1p10Hz_d20_r2_Kp25_Kd2_ampHip0p22_ampTh0p38_ampCalf0p38_cH0p10_cFT0p80_cRT1p00_cCalf-1p50_v25_a1600.pt
```

For a quick smoke run before the full fit:

```bash
cd /home/changba01/pace-sim2real

env \
  PYTHONPATH=/home/changba01/pace-sim2real/source/pace_sim2real \
  /home/changba01/IsaacLab/isaaclab.sh -p scripts/pace/a1_fit_narrow_encoder_bias.py \
    --headless \
    --task Isaac-Pace-A1-v0 \
    --num_envs 8 \
    --max-iteration 1 \
    --save-interval 1 \
    --robot-name a1_bias_third_smoke_20260728 \
    --bias-scale 0.3333333333333333 \
    --data /home/changba01/pace-sim2real/data/a1_real_wide/A1chirp_20260726_071950_f0p10-f1p10Hz_d20_r2_Kp25_Kd2_ampHip0p22_ampTh0p38_ampCalf0p38_cH0p10_cFT0p80_cRT1p00_cCalf-1p50_v25_a1600.pt
```

After the fit completes, run the usual posthoc manifest and fixed-mean
cross-validation steps from `docs/REPRODUCE_COMMANDS.md`, selecting the new run
directory and its final `mean_*.pt`.
