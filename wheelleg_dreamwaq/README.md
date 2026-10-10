# WheelLeg DreamWaQ

Independent Isaac Lab project for the WheelLeg robot. The current code implements the `Phase1RandomizedContractV3` asymmetric PPO baseline, the Architecture v0.21 Phase 2 DreamWaQ/CENet stack, deterministic Isaac comparison, and the unchanged MuJoCo sim2sim path. The DreamWaQ implementation and pure-Python tests are present; real Isaac integration, final independent code review, four fresh 1000-iteration runs, and their evaluation evidence remain mandatory before a Phase 2 result can be accepted.

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
