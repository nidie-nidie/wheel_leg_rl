# Split action smooth experiment results

Isaac was used only for training. All motion evaluation used the unchanged MuJoCo controller.
Each long run is fresh seed41882943 / 256 environments / 1000 iterations.
Numeric gates are exploratory; visual confirmation is still required.

## leg02

- Primary speed/control pass: False
- Complete control pass: False
- Twitch gate pass: False
- Numeric complete acceptance: False
- Command practice: 0/4
- Checkpoint: E:\wheel_leg_rl-main\wheelleg_dreamwaq\logs\rsl_rl\wheelleg_flat_dreamwaq\2026-10-13_19-10-09_rtx5070_fudan-v1_split-smooth-leg02-fresh1000-20261013\model_999.pt
- Speed failures: ['forward']
- Prolonged leg saturation: ['nominal_stand', 'forward', 'reverse']
- Saturation failures: ['nominal_stand:action_saturation_fraction', 'nominal_stand:effort_saturation_fraction', 'low_stand:action_saturation_fraction', 'low_stand:effort_saturation_fraction', 'high_stand:action_saturation_fraction', 'high_stand:effort_saturation_fraction', 'forward:action_saturation_fraction', 'forward:effort_saturation_fraction', 'reverse:action_saturation_fraction', 'reverse:effort_saturation_fraction', 'left_turn:action_saturation_fraction', 'left_turn:effort_saturation_fraction', 'right_turn:action_saturation_fraction', 'right_turn:effort_saturation_fraction', 'combined:action_saturation_fraction', 'combined:effort_saturation_fraction']

- 20ms acceleration RMS: {"baseline": 13.65403028281929, "candidate": 19.33439854793423, "reduction_fraction": -0.4160213612725383}
- 1ms acceleration RMS: {"baseline": 16.538445076388026, "candidate": 23.250002557911984, "reduction_fraction": -0.40581550747512907}

No tested configuration met every numerical gate. Do not claim the jitter was fixed.
