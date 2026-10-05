# Result Summary

## Fit Status

- Formal run directory: `/home/changba01/pace-sim2real/logs/pace/a1/26_07_26_02-50-16`
- Packaged fit manifest: `results/fit/a1_pace_fit_manifest.posthoc.json`
- Packaged fitted mean: `results/fit/mean_199.pt`
- Status: `PASS`
- `objective_rmse_mean_rad`: `0.0051687681`
- `direct_rmse_mean_rad`: `0.0700989417`
- `real_command_rmse_mean_rad`: `0.2338333853`
- Manifest delay steps: `4`
- Fitted continuous delay parameter: `3.5116794109`

## Fitted Parameters

Joint order:

```text
FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint,
FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint,
FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint
```

| joint | armature | viscous friction | static/dynamic friction | encoder bias rad |
| --- | ---: | ---: | ---: | ---: |
| FL_hip_joint | 0.018462 | 0.326204 | 0.499814 | 0.091497 |
| FR_hip_joint | 0.014418 | 0.409427 | 0.499793 | 0.053916 |
| RL_hip_joint | 0.012152 | 0.282013 | 0.499234 | -0.095693 |
| RR_hip_joint | 0.017447 | 0.432580 | 0.499656 | 0.098850 |
| FL_thigh_joint | 0.012899 | 0.282350 | 0.499116 | 0.045334 |
| FR_thigh_joint | 0.014259 | 0.283952 | 0.456892 | 0.032530 |
| RL_thigh_joint | 0.011122 | 0.282089 | 0.454595 | -0.042357 |
| RR_thigh_joint | 0.014204 | 0.280273 | 0.480689 | -0.053238 |
| FL_calf_joint | 0.018344 | 0.290579 | 0.499318 | -0.007150 |
| FR_calf_joint | 0.018337 | 0.437270 | 0.499829 | -0.099518 |
| RL_calf_joint | 0.016508 | 0.371936 | 0.499790 | 0.024042 |
| RR_calf_joint | 0.018135 | 0.387374 | 0.499848 | -0.099606 |

## Cross-Validation

Same fitted mean replayed against multiple data files:

| report | encoder RMSE rad | encoder P95 rad | encoder max abs rad | physical RMSE rad | command-vs-real RMSE rad |
| --- | ---: | ---: | ---: | ---: | ---: |
| replay_cv_062823_from_071950.json | 0.009285 | 0.014601 | 0.102736 | 0.069734 | 0.154327 |
| replay_cv_063144_from_071950.json | 0.006699 | 0.011829 | 0.099841 | 0.069605 | 0.183446 |
| replay_cv_064152_from_071950.json | 0.006337 | 0.009625 | 0.099927 | 0.069521 | 0.214968 |
| replay_cv_071554_from_071950.json | 0.006397 | 0.009691 | 0.099853 | 0.069609 | 0.230922 |
| replay_cv_071950_from_071950.json | 0.006255 | 0.009395 | 0.100094 | 0.069718 | 0.233833 |

The preferred fitted-model quality number is `encoder_frame_metrics.aggregate_rmse_rad`, because the fitted model includes encoder bias and the real data is measured in encoder coordinates. `physical_frame_metrics` is useful for seeing the simulated physical joint trajectory before subtracting fitted encoder bias. `command_vs_measured_metrics` is the original open-loop command mismatch and should not be used as fitted replay error.

## Practical Reading

The fitted actuator model is good enough to pass the posthoc gate and cross-dataset encoder-frame replay RMSE is around `0.006-0.009 rad`. The visible trajectory overlay can still show local deviations, especially around peak motion or bias-sensitive joints; judge it using the cross-validation reports and the interactive overlay rather than command-vs-real plots alone.
