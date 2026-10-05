# A1 Motor Identification Flow Package

This package freezes the work used for A1 chirp data collection, PACE actuator parameter identification, and trajectory visualization on 2026-07-26.

For agents, `AGENTS.md` is the canonical onboarding entry.

Primary result:

- Fit run: `results/fit/a1_pace_fit_manifest.posthoc.json`
- Fitted parameters: `results/fit/mean_199.pt` and `docs/RESULT_SUMMARY.md`
- Cross-validation reports: `results/crossval/replay_cv_*_from_071950.json`
- Interactive trajectory visualization: `results/crossval/a1_fixed_mean_crossval_vis_standalone.html`
- Raw CSV tracking plots: `results/raw_tracking_plots/*_des_vs_dof_pos.png` and `*_tracking_error.png`

Directory layout:

- `data/raw_csv/`: raw A1 chirp CSV logs copied from the collection machine mirror.
- `data/converted/`: converted PACE `.pt` tensors plus conversion manifests.
- `results/fit/`: formal PACE fit artifacts, posthoc manifest, best trajectory, and fit log.
- `results/crossval/`: fixed-mean replay reports and trajectory tensors for held-out checks.
- `results/raw_tracking_plots/`: static plots for command-vs-real raw CSV inspection.
- `scripts/collection/`: A1 real-machine chirp collection code snapshot.
- `scripts/identification/`: PACE conversion, fit, replay, manifest, and visualization builders.
- `scripts/visualization/`: local CSV plotting script.
- `scripts/rl_integration/`: Gogo/IsaacLab fitted actuator integration snapshot.
- `docs/`: reproducible process notes and result summary.

Read in this order:

1. `AGENTS.md`
2. `docs/FLOW.md`
3. `docs/RESULT_SUMMARY.md`
4. `data/raw_csv/EXPERIMENT_INDEX.md`
5. `docs/REPRODUCE_COMMANDS.md`

Important modeling note:

The final interpretation for this run is A1 upside down with fixed root, `Kp=25`, `Kd=2`, no extra 5 kg payload. The `064152`, `071554`, and `071950` converted manifests still contain `payload_kg: 5.0` because the converter default was not updated before those files were generated. The `062823` and `063144` conversion manifests already show `payload_kg: 0.0`. The formal fit manifest only enforces the fixed-root upside-down motor replay contract; the 5 kg field is stale metadata in those three conversion manifests and should not be treated as the final modeling assumption.
