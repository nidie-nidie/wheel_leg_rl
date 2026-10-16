# Learning and diagnostic-method audit

2026-10-13. Read-only review of production PPO/action/reward/CENet code and the already-run MuJoCo diagnostics. New files are limited to this learning-agent directory. No training, Isaac run, production modification, or new agent was started.

## Decision supported by the evidence

The existing 85% checkpoint plus a first-order action processor with leg time-constant label 6 Hz and wheel label 2 Hz is a credible MuJoCo candidate for substantially quieter actual legs while retaining useful forward/reverse/yaw tracking. It is a change to how motor targets are generated; it is not a new trained policy, and the raw neural output may continue oscillating.

The 1 ms physics observation is a strong check against sampling deception: physics_bandwidth.py:19-24 wraps the original mj_step, calls that original function exactly once, and records qvel finite differences. Lines 37-40 assert exact equality of raw actions, full history, and control-sampled q/qvel against the original callback-free experiment. Standing left-leg actual 1 ms acceleration RMS falls from 32.1105/30.5282 to 1.8264/5.1118 rad/s^2. Wheel actuator torque RMS falls from 8.1209/7.4363 to 3.7693/3.5877 Nm. This does not mean zero physical vibration, but it is a real reduction.

The candidate passes the four existing stop/reverse summaries. Their tightest required reversal confirmation is 0.80 s against a 1.0 s gate; worst stop vx MAE is 0.0801753 m/s against 0.10, and yaw MAE is 0.102047 rad/s against 0.15. These are finite margins, not a proof of arbitrary disturbances.

The 12 commands generated after the initial bandwidth choice provide useful independent command checks. Mean left actual leg acceleration RMS changes from 20.3431/24.5743 to 1.70443/1.66205 rad/s^2. Mean vx MAE gets slightly worse, 0.0378276 to 0.0436087 m/s (worst 0.0842668 to 0.104765), while mean yaw MAE improves from 0.141468 to 0.0483980 rad/s. Both versions survive. These data refute an unavoidable large tradeoff between good speed tracking and low leg vibration, but do not show every measure improving.

An important remaining weakness: the recorded 60 s zero-command test is NOT position hold. Over the 58 s after settling, original path/net displacement are 1.82949/1.34765 m; processed candidate 2.28644/1.81047 m. Small mean velocity error accumulated over time produces substantial drift. Thus the candidate can be described as much quieter with useful speed tracking; it cannot be described as standing immovably or solving all parking behavior.

## The original learning mechanism is valid but permits saturated policies

actor_critic.py:175-198 builds a Gaussian with mean bounded only to +/-20 and standard deviation bounded to [1e-4,2], then samples a RAW action. ppo.py:156-168 saves that same raw action and its Gaussian log probability. Batch recomputation at actor_critic.py:227-233 and PPO ratio at ppo.py:265-275 use the same raw variable. Environment env.py:661-666 maps raw samples into canonical clipped actions and stores applied clipped actions. Rewards rewards.py:64-74 use those executed actions for action_rate/action_smooth. This is a mathematically valid latent-action policy-gradient model of an environment whose deterministic action map contains clipping; it is NOT evidence of a wrong log probability.

It does create many-to-one behavior: raw values -2,-4,-8 can all mean exactly the same executed -1. Once a mean is deep outside the range, almost every sample executes the same endpoint. Applied action variation costs on that channel become nearly zero even if raw means or noise vary substantially. Moving the mean farther outward can reduce the chance that exploration changes the executed endpoint. That supplies a plausible maintenance mechanism for deeply saturated motor channels, while neither identifying the original cause nor proving which reward gradient produced it.

For fixed sigma and A~Normal(mu,sigma), d E[clip(A,-1,1)] / d mu = P(-1<A<1). The standing state's recorded means and saved sigma give right-leg interior probabilities about 0.01163% and 0.005668%. This is a state-conditional calculation, NOT recovered Isaac training action statistics. It demonstrates a nearly flat average executed-action response to further mean motion. It does NOT mean PPO has identically zero gradient: the score-function update and rare interior samples still matter.

Gaussian entropy depends on sigma, not mean. ppo.py:286-287 rewards that raw Gaussian entropy; increasing mu overshoot does not itself lower the logged entropy. Raw Gaussian entropy cannot be read as actual motor exploration when clipping maps almost all samples onto one endpoint. Raw-distribution adaptive KL can also register updates with little change to executed actions. These are modeling consequences, not implementation errors.

The earlier 101-to-201 lineage and cross-seed results show learned persistent one-side endpoints before adaptive leg-acceleration experiments. Different original seeds choose different fixed sides. No fixed right-side control mapping is needed to explain the observation. Existing data do not recover per-joint PPO advantages/gradients or original Isaac states between these checkpoint saves; therefore a unique training root cause remains unproven.

## Why larger common smoothness or current gain is not a clean solution

Ordinary action_smooth/action_rate sum all six clipped dimensions. Existing reward audit shows about 99.66% of these penalties in the wheels at the observed state. The four-leg actual acceleration budget has a different imbalance: its per-joint budget came from an already asymmetric reference, with very low right-side values. leg_acceleration.py:121-143 sums normalized excess across four joints, then sets one shared next gain from the maximum mean excess squared and caps it at 1. The old training data already show all noninitial iterations at that cap; right legs dominate the normalized excess. In that regime it is essentially a fixed maximum penalty, not an active useful search for the left twitch.

A motor target that is constant can still have real acceleration caused by coupled wheel/body dynamics. Therefore action smoothness, actual acceleration, and desired stop/reverse response must all be measured; none is a interchangeable certificate for the others. Blindly raising the common penalty is not supported as the next solution.

## Audit of the current MuJoCo comparisons

The action processor is explicit and deterministic:
applied_t = applied_(t-1) + alpha*(clip(raw_t,-1,1)-applied_(t-1)),
alpha=1-exp(-2*pi*fc*0.02).
Zero Hz in these scripts means bypass, not a held action. State initializes to zero at reset. The label fc defines a continuous-time matched time constant; it is not exactly the discrete implementation's -3 dB frequency. For legs 6/wheels 2 the alpha values are about 0.52951/0.22223. Low-frequency/DC command targets are not removed, but extra lag changes closed-loop behavior.

trace_and_screen.py:72-76 and expand.py:68-77 pass the processed action into the original runtime. runner.py:95-97 saves that actual action and includes it in the next observation/history. Wheel-speed and all real physical feedback remain present. This is a relevant closed-loop contrast and is not the previously rejected hidden-wheel-observation trick.

The 20 ms acceleration statistic can conceal within-control-step variation, but the 1 ms wrapper check addresses that concern for nominal standing. It has not yet been repeated at all command changes or strongest pushes, so use its numerical claim only for that tested standing trajectory.

Bandwidth margins are limited. Legs 5/6/8 with wheels 2 and legs 6 with wheels 1.5 pass all four existing practices. Legs 6 with wheels 2.5 fail right-stop-left reversal. Do not extrapolate a broadly stable bandwidth region. Unified 3 Hz smoothing fails some transitions, so it should not be substituted as an equivalent simpler solution.

summarize_practice.py is actually wheelleg_mujoco/command_practice.py:98-105. Its velocity gates apply to stop stages, and its 1 s held-band response gate applies only to stages 2 and 4. Stage 0 is required to complete but is not gated on response time. Thus existing "pass" is that existing evaluation definition, not a new guarantee covering every response characteristic. Legs 8/wheels 2 can show stage-0 yaw confirmation at 1.1 s while still passing. The 6/2 candidate has better first-stage times in these traces.

summarize_scenario returns survival and metrics; it does not mean the architecture's whole acceptance has passed. Phi0 asymmetry remains about 24-34 degrees in the candidate and high-stand height error remains about 0.029 m. Mirror-policy averaging has already failed and should not be recommended as a fix. The true-velocity oracle is explicitly nondeployable and does not solve all transitions.

The 5/10 N pulses last 0.16 s: impulses are 0.8/1.6 N*s, applied to the original base COM in reset control directions, with no added torque. Surviving these probes says nothing definite about longer, stronger, rotated, or combined disturbances, terrain, physical actuator delay, or sim2real. The tests use one model/reset environment and a selected checkpoint. Multiple deterministic command tests are not independent seed-training replications.

Cache robustness weakness, not a demonstrated contamination: expand.py:31-36 checks expected ticks, filter Hz and mirror, but does not verify policy hash, exact command sequence, velocity_mode, push specification or script/source fingerprints. Existing experiments have unique variant keys and no observed conflicting reuse. Further variants should have distinct names; robust replay should verify these additional identities before accepting cached data.

## Feasible next plan, without modifying training in this round

Keep the current policy, full observations and original physical parameters. Present the debug-only 6/2 processor as the first candidate for interactive visual confirmation, with side-by-side raw/applied actions, actual motor movement and speed output. This is sufficient to test whether the user perceives a useful simultaneous improvement. Keep explicit boundaries: residual position drift, asymmetric pose, some wheel oscillation, unproved strong-disturbance margin and unproved hardware applicability.

Do not claim the reward weights have been solved. If the processor is later promoted, training/Isaac, play/export, MuJoCo and deployment must share exactly the same action-processing formula/state/reset/timing. The existing last-applied-action observation can carry the first-order filter state, so the filter alone does not demand removing observations or increasing history dimensions.

Important future code boundary: actor_critic.py:236 currently supplies actions.clamp(-1,1) to the CENet decoder, which is correct for the CURRENT unfiltered training action map. With a future filtered environment, that is no longer the actual action executed during the predicted transition. It must use the actual processed action (record it in storage, or exactly reconstruct it from the observation's prior applied action plus raw sample and alpha). Storage raw action/log probability must still remain raw for the current Gaussian PPO definition. Reward/action history must refer to the processed action, and checkpoint/export metadata must record filter state/reset and bandwidth contract. This is a future consistency requirement, not a request to change code now.

Future learning work should be staged, not bundled: first train with the validated consistent processing path while preserving feedback and speed rewards, then measure whether raw means remain saturated and whether actual high-frequency leg motion and parking drift improve. A bounded action distribution or deliberate raw-overshoot regularizer is a separate future design hypothesis requiring its own correct probability contract and controlled training; tanh by itself does not guarantee symmetry, interior exploration or good posture. A healthy physically justified symmetric acceleration target and separated leg/wheel cost telemetry may be useful, but the old already-fixed right leg is not a safety-threshold reference.

The available checkpoints cannot show which exact old PPO gradients created the fixed-side mode. New training is needed to test future learning changes. Stronger MuJoCo probes can establish the current candidate's disturbance and long-time drift boundary without that training.

## Reproducible audit

audit.py only reads existing JSON/NPZ and uses numerical formulas. results.json includes SHA256 for the six inputs, validates that they stay unchanged, calculates candidate margins, holdout aggregate metrics, long-standing path/net drift and clipped-Gaussian probabilities. It performed no simulator, production imports, training or policy edits.
