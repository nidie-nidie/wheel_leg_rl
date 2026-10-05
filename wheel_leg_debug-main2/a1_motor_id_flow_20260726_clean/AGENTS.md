# Agent Onboarding

This directory is a frozen handoff package for the A1 motor-identification work completed on 2026-07-26.

## Read First

Read these files in order:

1. `README.md`
2. `docs/FLOW.md`
3. `docs/RESULT_SUMMARY.md`
4. `data/raw_csv/EXPERIMENT_INDEX.md`
5. `docs/REPRODUCE_COMMANDS.md`

Use `FILE_MANIFEST.txt` only when locating artifacts, and `SHA256SUMS.txt` only when checking file integrity.

## What This Package Answers

- Which A1 chirp CSV files were collected.
- How raw CSV data is converted to PACE tensors.
- Which actuator parameters were fitted.
- How good the fitted replay is against the real trajectories.
- How to visualize command-vs-real and fitted-replay-vs-real trajectories.
- Where the IsaacLab/Gogo fitted-actuator integration snapshot lives.

## Critical Assumptions

- Robot fixture for the identification run: A1 upside down, fixed root, suspended/supported from the back.
- Final payload assumption: no extra 5 kg payload.
- PD gains for the formal run: `Kp=25`, `Kd=2`.
- Formal fit task: `Isaac-Pace-A1-v0`.
- Formal fit artifact: `results/fit/mean_199.pt`.
- Formal posthoc status: `PASS`.

The `064152`, `071554`, and `071950` conversion manifests in `data/converted/` still contain `payload_kg: 5.0`. Treat that as stale metadata from an old converter default, not as the final model assumption. The `062823` and `063144` conversion manifests already contain `payload_kg: 0.0`.

## Main Results

Human-readable fitted parameters:

```text
results/fit/fit_parameters_readable.json
docs/RESULT_SUMMARY.md
```

Posthoc fit manifest:

```text
results/fit/a1_pace_fit_manifest.posthoc.json
```

Cross-validation reports:

```text
results/crossval/replay_cv_*_from_071950.json
```

Interactive standalone visualization:

```text
results/crossval/a1_fixed_mean_crossval_vis_standalone.html
```

## How To Judge Quality

Do not judge the fitted motor model from raw `des_dof_pos` vs `dof_pos` alone. That plot shows the real motor tracking lag/error under the original command.

For fitted model quality, use:

- `objective_rmse_mean_rad` and `direct_rmse_mean_rad` in `results/fit/a1_pace_fit_manifest.posthoc.json`;
- encoder-frame RMSE/P95 in `results/crossval/replay_cv_*_from_071950.json`;
- `results/crossval/a1_fixed_mean_crossval_vis_standalone.html`.

The same fitted mean was replayed against multiple datasets for cross-validation. That is the preferred validation method.

## Directory Guide

- `data/raw_csv/`: raw 500 Hz A1 chirp logs with timestamps and motor feedback.
- `data/converted/`: PACE tensor files and conversion manifests.
- `results/fit/`: formal CMA-ES PACE fit artifacts and posthoc quality gate.
- `results/crossval/`: replay reports and interactive fitted-replay visualization.
- `results/raw_tracking_plots/`: raw command-vs-measured plots.
- `scripts/collection/`: A1 real-machine collection code snapshot.
- `scripts/identification/`: converter, fit, posthoc, replay, and visualization builder scripts.
- `scripts/visualization/`: local plotting helpers.
- `scripts/rl_integration/`: Gogo/IsaacLab fitted-actuator import snapshot. This is not required for motor-ID review; read it only when continuing RL training/integration.
- `assets/`: local IsaacSim terrain asset used to avoid remote asset lookup.

## Do Not

- Do not treat `data/converted/*064152*.manifest.json`, `*071554*.manifest.json`, or `*071950*.manifest.json` `payload_kg: 5.0` as the final physical setup.
- Do not treat `real_command_rmse_mean_rad` as fitted replay error.
- Do not assume the GUI policy play attempt succeeded; this package is about collection, identification, and visualization handoff.
- Do not edit `results/fit/mean_199.pt` directly.
- Do not use `remote_mirror/` as the primary source of truth for this handoff unless explicitly asked to modify upstream code.

## If You Need To Re-run

Use `docs/REPRODUCE_COMMANDS.md`. Re-runs should create new output paths and new manifests; do not overwrite packaged artifacts.
