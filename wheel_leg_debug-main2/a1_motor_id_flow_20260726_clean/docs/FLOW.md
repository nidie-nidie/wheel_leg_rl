# A1 Collection, Motor Identification, and Visualization Flow

## 1. Collection Record

Collection source is the A1 low-level chirp collector in `scripts/collection/a1_joint_chirp_collect_real.cpp`.

Final physical assumption for the data used here:

- Robot pose: four feet up / upside down.
- Root condition for fitting: fixed in air.
- Support: back supported by a table/fixture approximately trunk footprint size.
- PD gains: `Kp=25`, `Kd=2` for the formal data.
- Extra payload: final assumption is no added 5 kg payload.

Raw CSV files are in `data/raw_csv/`. The useful formal/validation files are:

- `A1chirp_20260726_062823_f0p10-f1p4Hz_..._Kp25_Kd2_...csv`
- `A1chirp_20260726_063144_f0p10-f1p8Hz_..._Kp25_Kd2_...csv`
- `A1chirp_20260726_064152_f0p10-f1p10Hz_..._Kp25_Kd2_...csv`
- `A1chirp_20260726_071554_f0p10-f1p10Hz_..._Kp25_Kd2_...csv`
- `A1chirp_20260726_071950_f0p10-f1p10Hz_..._Kp25_Kd2_...csv`

The small first file `061600` is retained as a low-amplitude smoke/inspection record, not as the final wide fit dataset.

Raw CSV schema:

- 500 Hz sampling, `dt=0.002 s`.
- `stage=center_ramp` for the first 1000 samples on 20 s runs.
- `stage=chirp` for the 10000 samples consumed by PACE.
- `t_monotonic_ns`, `recv_monotonic_ns`, `send_monotonic_ns`, and `t_rel_s` are present. So yes, the raw data has timestamps.
- Main signal columns are `q_des_*` for commands, `q_*` for measured joint positions, `dq_*`, `tau_*`, and `temp_*`.

The CSV motor order is converted into PACE order:

```text
FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint,
FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint,
FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint
```

## 2. Conversion Into PACE Tensor Contract

Converter:

- `scripts/identification/a1_convert_wide_chirp.py`

Output tensor files:

- `data/converted/A1chirp_*.pt`

PACE tensor contract:

- `time`: `torch.float64`, shape `(T,)`, exact grid `0, 0.002, ...`
- `des_dof_pos`: `torch.float32`, shape `(T, 12)`, command sent to motor.
- `dof_pos`: `torch.float32`, shape `(T, 12)`, real measured joint angle.
- `T=10000` for 20 s formal chirp windows.

## 3. Motor System Identification

The formal identification is PACE-style CMA-ES optimization, using IsaacSim/IsaacLab replay of the measured motor commands.

Formal run:

- Task: `Isaac-Pace-A1-v0`
- Environments: `4096`
- Headless: yes
- Fixture: upside-down fixed root
- PD gains: `Kp=25`, `Kd=2`
- Fit log: `results/fit/formal_4096_headless_kd2_20260726_025007.log`
- Fitted mean: `results/fit/mean_199.pt`
- Posthoc manifest: `results/fit/a1_pace_fit_manifest.posthoc.json`

The fitted 49-dimensional parameter vector is interpreted as:

- 12 `armature`
- 12 `viscous_friction`
- 12 `static_dynamic_friction`
- 12 `encoder_bias`
- 1 continuous delay parameter

The optimization goal is to replay the same `des_dof_pos` command in simulation and minimize trajectory error against real `dof_pos`. It is not RL policy training; it is actuator/system identification.

## 4. Fit Acceptance

Posthoc status:

- `status=PASS`
- `objective_rmse_mean_rad=0.0051687681`
- `direct_rmse_mean_rad=0.0700989417`
- thresholds: `objective <= 0.05 rad`, `direct <= 0.08 rad`

Interpretation:

- Objective-frame replay error is very small.
- Direct-frame RMSE is within the acceptance threshold but not visually identical everywhere.
- `real_command_rmse_mean_rad=0.2338333853` is the open-loop command-vs-real mismatch before fitted actuator compensation; it should not be confused with fitted replay error.

## 5. Cross-Validation

Validation method used after the user correction:

Use one frozen fitted mean, then replay multiple other datasets and compare simulated encoder trajectory against real measured encoder trajectory.

Script:

- `scripts/identification/a1_crossval_replay_fixed_mean.py`

Reports:

- `results/crossval/replay_cv_062823_from_071950.json`
- `results/crossval/replay_cv_063144_from_071950.json`
- `results/crossval/replay_cv_064152_from_071950.json`
- `results/crossval/replay_cv_071554_from_071950.json`
- `results/crossval/replay_cv_071950_from_071950.json`

Aggregate replay RMSEs:

| dataset | encoder RMSE rad | P95 rad | max abs rad |
| --- | ---: | ---: | ---: |
| 062823 | 0.009285 | 0.014601 | 0.102736 |
| 063144 | 0.006699 | 0.011829 | 0.099841 |
| 064152 | 0.006337 | 0.009625 | 0.099927 |
| 071554 | 0.006397 | 0.009691 | 0.099853 |
| 071950 | 0.006255 | 0.009395 | 0.100094 |

## 6. Visualization

Static raw-data tracking plots:

- Script: `scripts/visualization/plot_a1_chirp_joint_tracking.py`
- Outputs: `results/raw_tracking_plots/`
- These compare real collection command `des_dof_pos` against measured `dof_pos`.

Interactive fixed-mean replay visualization:

- Data: `results/crossval/a1_fixed_mean_crossval_vis_data.json`
- Standalone HTML: `results/crossval/a1_fixed_mean_crossval_vis_standalone.html`
- Builder: `scripts/identification/a1_build_crossval_vis_html.py`
- This compares real measured encoder trajectory against simulated replay trajectory using the same fitted motor parameters.

## 7. Known Metadata Issue

These conversion manifests under `data/converted/` contain stale metadata:

- `A1chirp_20260726_064152_*.manifest.json`
- `A1chirp_20260726_071554_*.manifest.json`
- `A1chirp_20260726_071950_*.manifest.json`

They show `payload_kg: 5.0` because of the converter default at the time those files were generated. The `062823` and `063144` conversion manifests already show `payload_kg: 0.0`. The final modeling assumption is no extra 5 kg payload. Treat the `5.0` fields as historical metadata noise unless a later rerun explicitly uses `--payload-kg 0` and regenerates those converted manifests.

The `*.stdout.txt` files in `data/converted/` are terminal stdout captures, not machine-readable JSON.
