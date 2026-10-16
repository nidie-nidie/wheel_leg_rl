# Stationary feedback diagnosis, 85% candidate

Scope: the recorded 85% policy, same production MuJoCo model and zero command. No training or production changes.

1. Reuse the existing exact-replay tool to reconstruct synchronized physical state, history, encoder output and action. Verify every recorded frame and protected-file hashes.
2. Compare offline action dependence on estimated velocity, current wheel velocity, previous wheel action, and encoder pathways. These tests characterize a fixed network only.
3. Run diagnostic MuJoCo branches from identical copied states at 2, 6 and 10 seconds, for five seconds each. Alter one input pathway at a time; retain real commands, actions, actuator laws and physics. Input freezes are diagnostic substitutions, never deployment changes.
4. Check exported-policy inference with physical features held at MuJoCo snapshots while action history continues updating. This isolates computational recurrence; it is not a stability or locomotion test.
5. Compare wheel direction changes, left joint acceleration, tracking, attitude and formal failure reasons. Use multiple starting states to avoid relying on one phase of an oscillation.

Success means locating or narrowing the oscillation pathway with reproducible evidence, not claiming a fixed or stable policy. Any intervention that reduces jitter but worsens tracking or causes a fall cannot be considered a fix.
