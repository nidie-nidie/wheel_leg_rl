# A1 Fixed-Mean Replay Comparison

Source fitted mean: `results/fit/mean_199.pt`.

The preferred fitted-model error is encoder-frame replay-vs-real RMSE/P95. Command-vs-real RMSE is included only to show the original open-loop tracking gap.

| dataset | encoder RMSE rad | encoder P95 rad | encoder max abs rad | command-vs-real RMSE rad | worst joint | worst joint RMSE rad |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 062823 | 0.009285 | 0.014601 | 0.102736 | 0.154327 | RR_calf_joint | 0.017312 |
| 063144 | 0.006699 | 0.011829 | 0.099841 | 0.183446 | RR_calf_joint | 0.009167 |
| 064152 | 0.006337 | 0.009625 | 0.099927 | 0.214968 | RL_calf_joint | 0.008238 |
| 071554 | 0.006397 | 0.009691 | 0.099853 | 0.230922 | RR_calf_joint | 0.008436 |
| 071950 | 0.006255 | 0.009395 | 0.100094 | 0.233833 | RR_calf_joint | 0.008033 |

Encoder replay RMSE range: `0.006255` to `0.009285` rad; mean `0.006995` rad.
Command-vs-real RMSE range: `0.154327` to `0.233833` rad; mean `0.203499` rad.
