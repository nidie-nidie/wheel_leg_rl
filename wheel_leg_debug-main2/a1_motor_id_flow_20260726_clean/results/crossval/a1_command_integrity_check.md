# A1 Command Integrity Check

Checks raw CSV `q_des` against the analytic chirp formula and converted `des_dof_pos` tensors.

| csv id | samples | f range Hz | formula max residual rad | raw-vs-converted max rad | max 2ms step rad | max 2ms step deg | max vel rad/s | max accel rad/s^2 | max-step joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 061600 | 4000 | 0.100-1.500 | 7.122e-09 |  | 0.000724 | 0.041 | 0.362 | 2.7 | FL_calf |
| 062823 | 10000 | 0.100-4.000 | 1.517e-07 | 0.000e+00 | 0.013620 | 0.780 | 6.810 | 155.2 | RR_thigh |
| 063144 | 10000 | 0.100-7.999 | 1.514e-07 | 0.000e+00 | 0.027198 | 1.558 | 13.599 | 618.0 | FR_thigh |
| 064152 | 10000 | 0.100-9.999 | 1.764e-07 | 0.000e+00 | 0.039619 | 2.270 | 19.810 | 1125.5 | FR_thigh |
| 071554 | 10000 | 0.100-9.999 | 1.911e-07 | 0.000e+00 | 0.043015 | 2.465 | 21.508 | 1222.0 | RL_thigh |
| 071950 | 10000 | 0.100-9.999 | 1.911e-07 | 0.000e+00 | 0.043015 | 2.465 | 21.508 | 1222.0 | RL_thigh |

Notes:

- The large-looking high-frequency curve in the old HTML was caused by display stride=25, i.e. 20 Hz visualization sampling.
- The formal high-frequency commands are still smooth 500 Hz samples; at 10 Hz and ~0.38 rad amplitude, adjacent 2 ms samples can legitimately differ by about 0.04 rad.
