# WheelLeg DreamWaQ

Independent Isaac Lab project for the WheelLeg robot. The current code implements the `Phase1RandomizedContractV3` asymmetric PPO baseline, the Architecture v0.26 Phase 2 DreamWaQ/CENet stack, explicit current-physics Isaac evaluation, and the frozen MuJoCo sim2sim path. The DreamWaQ implementation and pure-Python tests are present; real Isaac integration, final independent code review, four fresh 1000-iteration runs, and their evaluation evidence remain mandatory before a Phase 2 result can be accepted.

## Runtime

```powershell
./scripts/bootstrap.ps1
uv run pytest tests/unit -q
uv run python scripts/audit_asset.py --headless
uv run python scripts/smoke_random_actions.py --num-envs 16 --steps 1000 --headless
uv run pytest tests/integration -q
uv run python scripts/train_ppo.py --profile rtx5070 --max-iterations 1000 --headless
uv run python scripts/train_ppo.py --profile portable --max-iterations 1200 --resume <checkpoint> --headless
uv run python scripts/play.py --num-envs 16 --steps 1000 --headless
./scripts/tensorboard.ps1
```

New V4 runs load `../wheel_leg_urdf4_usd_v2/wheel_leg_urdf4/wheel_leg_urdf4.usd`. This self-contained `AssetBundleV2` differs from the immutable V1 source only by deleting the embedded GroundPlane; both bundles are hash-verified and the V1 source remains unchanged.

Hardware profiles may change only `num_envs` and PPO mini-batch count. Every run writes resolved YAML, dependency and source hashes, a Phase1 contract manifest, TensorBoard events, and checkpoints containing optimizer state plus Python/NumPy/PyTorch/CUDA RNG state.

`--max-iterations` always means the total target iteration count. Resume creates a new run directory, validates the source manifest and `Phase1ContractV4`, restores model/optimizer/adaptive-learning-rate/RNG state, and continues from `completed_iterations` without repeating the saved zero-based RSL-RL iteration. V3/V1 checkpoints are rejected by the V4/V2 runtime. The restored RSL-RL `alg.learning_rate` must match the optimizer parameter-group learning rate. Simulator state is initialized as a new environment and is not represented as an exact mid-rollout continuation.

On Windows, `torch` and `tensordict==0.14.2` are intentionally imported before Isaac Kit in the train/play entry points. This ordering prevents a reproduced native DLL access violation and must not be moved behind `AppLauncher`.

Formal training requires an independent code review with no blocking P0/P1 findings. The existing Phase 1R suite runs four independent 1000-iteration seeds from scratch, exports each actor, and compares them with the same deterministic MuJoCo headless scenarios. G-15 is closed for Phase 2 implementation because that comparable evidence chain exists; this is an implementation permit, not a claim that any Phase 1R checkpoint passed all eight MuJoCo scenarios. DreamWaQ formal training remains forbidden until the real Isaac integration, checkpoint/resume, play/export, TorchScript golden-vector, and final independent code-review gates pass.

## Stop/reverse training profile

`legacy_v1` remains the default episode-held command and original reward configuration.
`stop_reverse_v1` uses independently counted real control steps for move (0-2s), stop
(2-3s), reverse (3-5s), stop (5-6s), original direction (6-8s), and stop (8s onward).
It changes only the command schedule and doubles `tracking_vx` and
`tracking_vx_enhance` from 1 to 2. The latter remains a negative error penalty.
Command ranges, height sampling, history, randomization, terrain and PhysicsV5 stay unchanged.

Resume infers the saved task profile; an explicit different profile is rejected.
Fixed-command play disables the runtime schedule after validating the training identity.
Exports record the full schedule and training reward weights.

Current PhysicsV5 checkpoints use explicit `--candidate-only` IsaacEvaluationV2,
with the original scoring weights and fixed eight scenarios. Historical PhysicsV4
baseline comparison is unassessed (`null`), never counted as a pass. Failed performance
goals still produce all fixed and dynamic MuJoCo evidence.

```powershell
.venv/Scripts/python.exe scripts/train_dreamwaq.py --profile portable --task-profile stop_reverse_v1 --max-iterations 2 --headless
.venv/Scripts/python.exe -m pytest tests/integration/test_command_practice.py -q
.venv/Scripts/python.exe scripts/run_dreamwaq_training_suite.py --mode formal --profile rtx5070 --task-profile stop_reverse_v1 --candidate-only --runs 4 --iterations 1000 --monitor-interval-seconds 1800
```

The formal command runs four fresh seeds sequentially, records half-hour monitoring,
then exports and evaluates each checkpoint in Isaac and the unchanged eight-scenario
MuJoCo test plus independent stop/reverse diagnostics. See the reviewed
[design](docs/superpowers/specs/2026-10-11-stop-reverse-speed-tracking-design.md)
and [implementation plan](docs/superpowers/plans/2026-10-11-stop-reverse-speed-tracking.md).
