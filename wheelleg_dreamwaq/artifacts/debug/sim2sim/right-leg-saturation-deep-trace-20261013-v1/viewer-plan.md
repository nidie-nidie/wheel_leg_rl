# Candidate keyboard visualization implementation plan

Goal: show the already-reviewed candidate in MuJoCo, using debug files only.

Architecture: a debug adapter reproduces slow_feedback.trial's frozen six-dimensional action filter and zero-command integral exactly. It invokes the original runtime and renders a separate model/data copy through the existing KeyboardViewer. No rendering or mouse operation changes runtime forces or physics.

Files: create candidate_viewer.py and viewer-verification.json in this directory; create timestamped viewer sessions containing status.json, controls.jsonl, events.jsonl, stdout.log and stderr.log. Production scripts and old evidence remain untouched.

- [ ] Implement reset and tick adapter with the identical 6/2 Hz formula,1Hz observable estimator, ki=.3,bounds=.06 and 25-sample original-command gate (cold reset0.48s; command change0.50s).
- [ ] Reuse KeyboardViewer with W/S vx,A/D yaw,Q/E height,X zero,R reset,P pause,M marker,Esc close. Use the trained command limits, full sensor/history feedback, true previous processed action and original user-command error. Overlay raw/processed action, actual motor q/qdot, motor targets and torque, original/policy command and actual/CENet speed.
- [ ] Verify 500-step standing and four500-step command practices exactly against saved frozen history/raw/applied/q/v/command/correction, including a reset repeat. Protect211 file hashes. Validate keyboard direction, clamp and reset keys without opening a GUI.
- [ ] Start exactly one visible GLFW window using hidden-console Start-Process; confirm process and VIEWER_READY plus progressing status before reporting it opened.

Scope: no training,no Isaac,no production changes,no new collisions/physics parameters, no additional agent. User has explicitly requested this candidate visualization following the reviewed design; continue locally without another approval.
