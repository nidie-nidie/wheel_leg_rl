# A1 Zero-Bias Rescore of Existing Replay

This is a diagnostic only, not a new fit. It compares packaged simulated physical joint positions directly against real encoder measurements, without subtracting fitted `encoder_bias`.

| dataset | original encoder RMSE rad | zero-bias rescore RMSE rad | zero-bias P95 rad | zero-bias max abs rad |
| --- | ---: | ---: | ---: | ---: |
| 062823 | 0.009285 | 0.069734 | 0.103477 | 0.126778 |
| 063144 | 0.006699 | 0.069605 | 0.101418 | 0.124230 |
| 064152 | 0.006337 | 0.069521 | 0.100724 | 0.123969 |
| 071554 | 0.006397 | 0.069609 | 0.101301 | 0.123896 |
| 071950 | 0.006255 | 0.069718 | 0.101107 | 0.124136 |

Mean original encoder RMSE: `0.006995` rad.
Mean zero-bias rescore RMSE: `0.069637` rad.
