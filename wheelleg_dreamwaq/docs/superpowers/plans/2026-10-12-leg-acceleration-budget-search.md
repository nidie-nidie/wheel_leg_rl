# Actual Leg Acceleration Budget Search (2026-10-12)

## Scope and invariants

Reduce persistent leg twitching while preserving tracking, parking, reversal, and balance. Isaac Lab is used only for training. Reference collection, candidate evaluation, and threshold selection are all in MuJoCo.

Preserve Architecture v0.28, including the current five-frame t-4..t history. This experiment does not implement the previously discussed history shift. Keep six-action action_smooth, six-joint dof_acc, speed rewards, PD/effort, reset, action/command mapping, randomization, USD/XML, collisions, and termination unchanged. The new term is a reward penalty, never a motor limit, extra torque, or action filter.

## Metric and reference

Measure actual actuated joint motion, canonical order jIJ, jIO, jAB, jAG. These are motor joint angles, not phi0 and not motor targets:
a_j(t) = [qdot_j(t) - qdot_j(t-1)] / 0.02, in rad/s^2.

Training uses instantaneous squared acceleration excess. MuJoCo uses the same 20ms endpoint difference, excludes the first 2 seconds, then computes nonoverlapping one-second RMS windows and reports mean/p90 per joint. The training signal is not a one-second-window penalty. Wheels are excluded from this new term; the old six-joint penalties remain.

B0 = [40.1703144886, 38.3671852533, 7.0362190023, 4.9735461617] rad/s^2.
Source: artifacts/debug/sim2sim/leg-motion-metric-analysis-20261012-v1, previous baseline nominal_stand/forward/reverse captures, pooled 24 nonoverlapping one-second windows after a 2-second warmup, per-joint p90. Right-leg targets were saturated in that reference. Small right-leg motion is therefore not proof of healthy control or safety. B0 is only an exploratory reference.

Budget B_j = f * B0_j, independently per joint.

## Single-term gain adaptation

e_j = max(a_j^2 / B_j^2 - 1, 0)
added_reward = -dt * 0.02 * gain_used * sum_j(e_j)

This instantaneous positive-part expectation is not mathematically equivalent to mean(a^2)<=B^2. The controller cannot identify rescue motions; necessary transients remain mechanically possible but are still penalized above budget.

Gain is constant within each 24-step rollout. All environment/control-step samples, including zero excess, contribute to mean(e_j). At the boundary:
gain_next = clip([max_j mean(e_j)]^2, 0, 1).

Initial gain is zero; the first rollout measures excess. Only gain is bounded; excess cost and negative reward are not capped. This is an exploratory ROGER-style bounded single-term gain. It does not implement the paper's multiple value channels or primary-reward decay and does not claim the paper's guarantees.

The old target-reversal gain experiment is disabled. Gain/checkpoint metadata for the two terms are independent. Partial reset clears selected rows' cached reward/excess but preserves global rollout accumulation/gain. Save only at rollout boundaries. Resume validates the contract, gain formula, and completed_rollouts before restoring gain. Weights-only initialization resets optimizer, RNG, gain and environment; it is not exact training resume.

## Matched arms and finite search

Each comparison arm: rtx5070, 256 environments, seed 41882943, 200 iterations. Shared source:
logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-11_21-56-09_rtx5070_fudan-v1_leg-reversal-trial-baseline-200/model_199.pt
SHA256: BCDD726CE5D6F89600547F34CA285282053D4D07B07575604172E6A11A5912B7.

This source was previously initialized from a 1000-iteration model and trained for another 200 iterations. It is not the original 1000-iteration checkpoint. All arms initialize from this identical source network. Baseline is stop_reverse_v1; candidates only add the acceleration term. Record source identity, training length, checkpoint, export/golden and evaluation file hashes.

Start with baseline and 80%.
- Control passes but twitch reduction is insufficient: try 75%, then 70%.
- Twitching improves but control deteriorates: try 85%, then 90%.
- Neither clearly improves: try 75% and 85%, then 70% and 90% if needed.
- Never repeat a budget or search beyond these five candidates.
- Stop farther budgets if one candidate meets the complete acceptance below.
- If none passes, report that no suitable budget was found; do not automatically start four seeds of 1000 iterations.

## MuJoCo acceptance

Export deterministic actors. Run all existing eight fixed-command scenarios for 500 ticks each. Reuse evaluate_mujoco.py failure rules: nonfinite state, height/tilt, root/joint velocity, loop closure, virtual leg length, unexpected contact. FloatingPointError must create a failed scenario record and allow subsequent scenarios to run. Failed ticks must not become successful zero-cost samples. Fewer than one complete RMS window gives metric_valid=false and cannot pass.

Primary twitch comparison uses nominal_stand/forward/reverse: after the 2-second warmup, take an equal-weight mean across these three scenarios and four joints of one-second RMS means. Require at least 5% reduction versus matched baseline. Also report per-joint p90, actual velocities/positions, motor targets and body tilt. A quieter robot alone does not imply better control.

Control acceptance requires all of:
- All eight static scenarios survive.
- Primary steady vx MAE: parking <=0.10m/s, forward/reverse <=0.15m/s; also <= baseline + max(10% baseline, 0.03m/s). These are this experiment's steady-state criteria, borrowing the command-practice parking/error-band values. They are not gates built into old eight-scenario ranking.
- Run unchanged evaluate_command_practice_mujoco.py on four stop/reversal sequences. Preserve its gates: parked steady vx<=0.10m/s, yaw<=0.15rad/s; reversed-command error<=0.15 for 10 consecutive frames confirmed within 1 second; all stages complete.
- Report action and effort saturation. A candidate with either increasing by more than 5 percentage points in any static scenario is not accepted as successful twitch control.

If baseline fails control gates, relative candidate improvement alone cannot be declared ready for use. No disturbance-recovery/hardware data is collected here. One seed and 200 iterations provide exploratory evidence, not a certified safety threshold or proof that necessary balance actions are preserved under every disturbance.

## Files and execution

1. leg_acceleration.py: isolated config, four-joint cost, rollout gain, logs, serialized state.
2. env_cfg.py / env.py: opt-in integration, exactly one update per control step using existing actual acceleration; preserve old terms.
3. training_profiles.py and train/suite CLI: named 80/75/70/85/90 profiles; preserve old names/behavior.
4. dreamwaq_manifest.py / weights_initialization.py / dreamwaq_checkpoint.py: truthful contract/hash, narrowly permitted weights-only differences, independent gain validation/resume.
5. export_dreamwaq_actor.py: training_leg_acceleration metadata only for enabled candidates; unchanged baseline keys.
6. evaluate_leg_acceleration_mujoco.py: eight-scenario actual-motion CSV, RMS windows, failure records and identity hashes; existing command-practice evaluator supplies parking/reversal acceptance.
7. Unit checks: units/order/wheel exclusion, partial reset, fixed rollout gain, batch mismatch, JSON roundtrip, old profiles, checkpoint missing/tampered gain, resume/export golden, nonfinite failure and incomplete window.
8. Real Isaac training->checkpoint->resume plumbing only; export golden batches 1/3/32. Then matched training, MuJoCo collection and budget selection.
9. Document review -> code review -> training/evaluation -> conclusion review. Distinguish earlier summary-only review from actual file review. An actual file review is required for approval.

