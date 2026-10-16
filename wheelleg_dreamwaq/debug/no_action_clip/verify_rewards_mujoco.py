from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from debug.no_action_clip.contract import EXPERIMENT_ROOT, sha256, verify_formal_files, write_json
from debug.no_action_clip.mujoco_adapter import load_runtime
from debug.no_action_clip.rewards import (
    CONTROL_DT, SPLIT_STRAIGHT_PROFILE, resolve_reward_weights,
    reward_manifest_fields, validate_reward_manifest, wheel_differential_reward,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="真实 MuJoCo 上的奖励计算与部署兼容性检查，不训练")
    parser.add_argument("--export", type=Path,
                        default=EXPERIMENT_ROOT / "runs/fresh1000-fk-diagnostic-20261016/export")
    parser.add_argument("--output", type=Path,
                        default=EXPERIMENT_ROOT / "runs/reward-verification-20261016/verification.json")
    args = parser.parse_args()
    export_dir = args.export.resolve()
    runtime = load_runtime(export_dir)
    reference = load_runtime(export_dir)
    policy = torch.jit.load(str(export_dir / "actor.ts"), map_location="cpu").eval()
    # 0.1 仅用于公式验收，不是训练系数或训练默认值。
    coefficient = 0.1
    weights = resolve_reward_weights(SPLIT_STRAIGHT_PROFILE, coefficient)
    validate_reward_manifest(reward_manifest_fields(weights))
    cases = (
        ("forward", (0.8, 0.0, .20), True),
        ("backward", (-0.8, 0.0, .20), True),
        ("stand", (0.0, 0.0, .20), False),
        ("rotate", (0.0, .5, .20), False),
        ("combined", (.8, .5, .20), False),
        ("tiny_nonzero_yaw", (.8, 1e-12, .20), False),
    )
    records = []
    for name, values, expected_enabled in cases:
        command = np.asarray(values, dtype=np.float64)
        observation = runtime.reset(command)
        np.testing.assert_array_equal(observation, reference.reset(command))
        previous = np.zeros(6, dtype=np.float64)
        previous_previous = np.zeros(6, dtype=np.float64)
        differential_rewards = []
        for _ in range(50):
            with torch.inference_mode():
                action = policy(torch.from_numpy(observation).unsqueeze(0)).cpu().numpy()[0].copy()
            original_action = action.copy()
            weighted, cost, enabled = wheel_differential_reward(
                torch.from_numpy(action[None, :]), torch.from_numpy(command[None, :]),
                weights, control_dt=CONTROL_DT,
            )
            assert enabled.item() == expected_enabled
            expected_cost = float((action[4] - action[5]) ** 2) if expected_enabled else 0.0
            np.testing.assert_allclose(cost.item(), expected_cost, rtol=1e-6, atol=1e-9)
            np.testing.assert_allclose(weighted.item(), -coefficient * CONTROL_DT * expected_cost,
                                       rtol=1e-6, atol=1e-9)
            differential_rewards.append(float(weighted.item()))
            # 用真实策略轨迹验证分项求和与旧公式的关系。
            second_difference = action - 2.0 * previous + previous_previous
            old_smooth = -.01 * CONTROL_DT * np.square(second_difference).sum()
            leg_smooth = weights.action_smooth_leg * CONTROL_DT * np.square(second_difference[:4]).sum()
            wheel_smooth = weights.action_smooth_wheel * CONTROL_DT * np.square(second_difference[4:]).sum()
            extra_leg = -.01 * CONTROL_DT * np.square(second_difference[:4]).sum()
            np.testing.assert_allclose(leg_smooth + wheel_smooth, old_smooth + extra_leg, rtol=1e-7, atol=1e-12)
            np.testing.assert_array_equal(action, original_action)
            targets = runtime.controller.prepare(action)
            control_targets = runtime.contract.wheel_action_scale * action[4:6]
            np.testing.assert_allclose(targets.wheel_velocity_target_mujoco * runtime.contract.wheel_signs,
                                       control_targets, rtol=1e-6, atol=1e-6)
            result = runtime.step(action, command)
            expected = reference.step(original_action, command)
            np.testing.assert_array_equal(runtime.data.qpos, reference.data.qpos)
            np.testing.assert_array_equal(runtime.data.qvel, reference.data.qvel)
            np.testing.assert_array_equal(result.observation, expected.observation)
            np.testing.assert_array_equal(result.applied_torque, expected.applied_torque)
            previous_previous, previous = previous.copy(), action.astype(np.float64)
            observation = result.observation
        records.append({"name": name, "command": list(values), "enabled": expected_enabled,
                        "ticks": 50, "differential_reward_mean_at_test_coefficient": float(np.mean(differential_rewards)),
                        "reward_calculation_preserves_actions_and_physics": True})
    golden = torch.load(export_dir / "golden_vectors.pt", map_location="cpu", weights_only=True)
    with torch.inference_mode():
        output = policy(golden["history"])
    torch.testing.assert_close(output, golden["expected_action_mean"], rtol=1e-6, atol=1e-6)
    report = {
        "schema_version": "DebugSplitStraightRewardVerificationV1",
        "export": str(export_dir), "actor_sha256": sha256(export_dir / "actor.ts"),
        "test_only_coefficient": coefficient, "training_coefficient": None,
        "scenarios": records, "old_export_and_saved_golden_vectors_pass": True,
        "golden_max_abs_error": float((output - golden["expected_action_mean"]).abs().max()),
        "protected_formal_files_verified": verify_formal_files(),
        "new_training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
