# Split action smooth implementation plan

> For agentic workers: execute inline with executing-plans. Reviews are read-only and serial: one document reviewer, then one code reviewer; no additional child agents.
> User approved the design and execution on 2026-10-13. No further approval gate is needed for this scope.

**Goal:** Separate the existing four-leg/two-wheel second action difference and increase only the leg penalty, then train and assess one fresh 1000-iteration policy.
**Architecture:** Preserve RewardWeights/RewardSchemaV2 for old profiles. Use a frozen SplitSmoothRewardWeights subclass and RewardSchemaV3SplitSmooth only for opt-in profiles; derive resume/play/export identity from the stored contract.
**Tech stack:** Python 3.11, PyTorch 2.7, RSL-RL 3.1.2, Isaac Lab 2.3.2 / Isaac Sim 5.1 (training only), MuJoCo (all motion collection).

## Approved design and boundaries

Equation: d = last_applied_action - 2*previous_applied_action + previous_previous_applied_action.
The two raw costs are sum(d[:4]^2) and sum(d[4:]^2), without dimensional averaging.
Reward = 0.02*(w_leg*leg_cost+w_wheel*wheel_cost). First profile uses -0.02/-0.01.
An equal -0.01/-0.01 configuration must reproduce the old total within float tolerance.
The split configuration does not charge the old composite; its inherited action_smooth field is zero.
Original PPO/reward profiles and their task hashes stay compatible; no default dataclass fields are added.
No learner/CENet/history/observation/action conversion/actuator/physics changes. Disable leg_reversal and leg_acceleration in this profile.
Old-model initialization remains unavailable for this new experiment; both long runs are fresh.

## Task 1: Documents and reviewer

- [ ] Update E:/wheel_leg_rl-main/docs/2026-10-03-wheelleg-dreamwaq-architecture.md to v0.29 / section34.
- [ ] Update E:/wheel_leg_rl-main/wheelleg_dreamwaq/docs/2026-10-13-dreamwaq-jitter-debug-index.md and new debug plan.
- [ ] Record pre-change source bytes/hashes and unrelated existing workspace changes.
- [ ] Ask one agent to read the three documents and actual code; record evidence and resolve findings before implementation.

## Task 2: Reward and profile implementation

Modify under E:/wheel_leg_rl-main/wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq:
- tasks/direct/wheelleg_flat/rewards.py: frozen subclass, validated schema selector and split raw costs replacing composite.
- tasks/direct/wheelleg_flat/training_profiles.py: two named profiles (-0.02/-0.01 and -0.04/-0.01), reconstruct legacy RewardWeights when switching back; strict saved split schema/weight/schedule recognition.
- schemas/dreamwaq_manifest.py: select reward schema from actual weights; default stays RewardSchemaV2.
- tasks/direct/wheelleg_flat/env.py: exclude disabled composite from split episode sums.
- E:/wheel_leg_rl-main/wheelleg_dreamwaq/scripts/train_dreamwaq.py: add CLI choices.
Document review adds sim2sim/mujoco/wheelleg_mujoco/contract.py: recognize V3 only for DreamWaQ with exact supported weights and stop/reverse schedule; keep PPO V2 and actuator/physics checks. Include real-loader smoke and regression tests. No changes to weights_initialization.py, export/play or the formal four-run suite are necessary.
Core implementation:
```python
if isinstance(weights, SplitSmoothRewardWeights):
    second_difference = (state.last_applied_action - 2*state.previous_applied_action
                         + state.previous_previous_applied_action)
    terms.pop("action_smooth")
    terms["action_smooth_leg"] = second_difference[:, :4].square().sum(dim=-1)
    terms["action_smooth_wheel"] = second_difference[:, 4:].square().sum(dim=-1)
```

## Task 3: Meaningful checks and code review

- [ ] Add tests/unit/test_split_action_smooth.py covering independent nonzero inputs, both current leg joints, sum not mean, equal-weight equivalence, double-leg contribution, reset zeros, invalid coefficient/contract, switch-back type/fields, old contract unchanged and strict checkpoint cross-profile rejection.
- [ ] Use existing complete unit suite to catch old PPO/profile/checkpoint regressions:
```powershell
& .venv/Scripts/python.exe -m pytest tests/unit -q
```
Working directory: E:/wheel_leg_rl-main/wheelleg_dreamwaq. Expect zero failures.
- [ ] Add a small enabled fresh2/resume3 integration check without Isaac motion evaluation; check stored identities, both enabled reward log keys and disabled composite absence, then TorchScript golden vectors for batches1/3/32.
- [ ] Record changed lines relative to the captured working-tree bytes, not just git HEAD.
- [ ] One read-only code agent reviews actual changed files/diff, tests and documents. No training until findings addressed.

## Task 4: Training pipeline and MuJoCo analysis

Create debug-only orchestration under E:/wheel_leg_rl-main/wheelleg_dreamwaq/artifacts/debug/sim2sim/split-action-smooth-20261013-v1.
Record exact subprocess commands, exit codes, source/code/doc/model hashes, checkpoint/export paths and stage status.
Formal command (working directory E:/wheel_leg_rl-main/wheelleg_dreamwaq):
```powershell
& .venv/Scripts/python.exe scripts/train_dreamwaq.py --headless --profile rtx5070 --task-profile stop_reverse_split_smooth_v1 --max-iterations 1000 --seed 41882943 --run-name split-smooth-leg02-fresh1000-20261013
```
No --resume or --initialize-from. Expected completed_iterations=1000, final iteration index999.
- [ ] Frozen code while training; inspect progress/finite values every1800s. Evaluate only completed checkpoints.
- [ ] Export candidate actor with existing export_dreamwaq_actor.py; validate golden output <=1e-7.
- [ ] Re-run baseline and candidate with evaluate_leg_acceleration_mujoco.py: all8 static scenarios500ticks; warmup100/window50.
- [ ] Re-run evaluate_command_practice_mujoco.py: four full stop/reverse sequences (no --smoke).
- [ ] Collect true1ms legs q/qdot/qddot in nominal_stand/forward/reverse via read-only wrapping of the original mj_step; call original exactly once, never change control, physics steps or target processing. Save each joint, both sides, plus20ms targets and tracking data.
- [ ] Write results.md/comparison.json and update debug index with facts and limitations.

## MuJoCo acceptance and bounded next experiment

Baseline: stop_reverse_v1 run03 seed41882943 original model_999.pt; SHA2563D621280CF7F11088C972B0DD84CF2E698C29A4C8CDA8BC370BB80A2669DF885.
Existing exported actor: artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/exports/run-03.
Identical reset/model/control/commands for both actors; actual policy outputs remain unclipped on input to the original controller.

All eight static scenarios must survive; steady parked vx MAE<=0.10m/s, forward/reverse<=0.15m/s, and degradation<=max(10% baseline,0.03m/s). All four existing command-practice gates must pass for complete control acceptance.
Experimental twitch gate: at least50% reduction of equal-weight actual20ms acceleration RMS across stand/forward/reverse and four joints; independently report1ms RMS/p90 and per-joint results. The main1ms RMS aggregate must also decrease by at least50%. Report individual worsening, targets, effort/action saturation and any prolonged flat clipped target; saturation must not increase by>5percentage points in any static scenario. A model still holding a leg at clipped extrema for>=90% of any main scene is not a demonstrated oscillation fix.
These are engineering comparison gates, not a hardware safety bound or proof of necessary-rescue preservation. Numeric success still needs visual confirmation; do not claim jitter vanished from scalar rewards.

If all8 survive and primary straight/parking speed gates pass but twitch gates fail, optionally run only stop_reverse_split_smooth04_v1 from scratch for1000 iterations with the same seed. This tuning authorization is bounded to one extra run. If increasing penalty coincides with falls or primary speed deterioration, do not automatically increase it again. Report outcomes honestly; no automatic four-seed expansion.

## Document review resolutions (before implementation)

文档复核修订：MuJoCo contract.py 仅增加 DreamWaQ 新训练奖励元数据识别；PPO 仍限 V2，必须验证完整权重和停车/换向 schedule，再执行原有动作/模型/物理检查。真实 loader smoke 纳入训练前验证。1ms qddot 固定为相邻原物理步 qdot 差/0.001s，从原 reset 后 qdot 起算；舍弃前2秒，再按1000样本/1秒的8个窗口逐关节RMS，依次等权平均窗口、四关节、三个主场景，另报逐关节p90。时间区间与20ms指标一致，不能替换为 data.qacc。

P1 loader omission and P2 metric ambiguity are resolved in scope; see document-review.md for evidence.
