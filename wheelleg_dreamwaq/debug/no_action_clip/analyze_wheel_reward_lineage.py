from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from debug.no_action_clip.mujoco_adapter import load_runtime


def _mean_scalar(events, count: int = 100) -> float:
    values = np.asarray([event.value for event in events], dtype=np.float64)
    if values.size == 0:
        raise ValueError("TensorBoard scalar has no values")
    return float(values[-count:].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reward support for persistent left/right wheel speed differences.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    event_path = next(run.glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    reward_tags = [
        "Reward/tracking_vx",
        "Reward/tracking_vx_enhance",
        "Reward/tracking_yaw_rate",
        "Reward/tracking_yaw_rate_enhance",
        "Reward/ang_vel_xy",
        "Reward/orientation",
        "Reward/dof_vel",
        "Reward/dof_acc",
        "Reward/torques",
        "Reward/action_rate",
        "Reward/action_smooth",
        "Reward/phi0_symmetry",
    ]
    reward_tail_mean = {
        tag: _mean_scalar(accumulator.Scalars(tag)) for tag in reward_tags if tag in tags
    }

    export = args.export.resolve()
    runtime = load_runtime(export)
    actor = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
    command = np.asarray((0.0, 0.0, 0.20), dtype=np.float64)
    observation = runtime.reset(command)
    actions = []
    observations = []
    for _ in range(12):
        observations.append(observation[-25:].astype(np.float64).tolist())
        with torch.inference_mode():
            action = actor(torch.from_numpy(observation).unsqueeze(0)).detach().cpu().numpy().reshape(-1)
        actions.append(action.tolist())
        observation = runtime.step(action, command).observation
    first_action = np.asarray(actions[0], dtype=np.float64)

    report = {
        "schema_version": "DebugNoActionClipWheelRewardLineageV1",
        "run": str(run),
        "export": str(export),
        "training_reward_tail_mean_last_100_rollouts": reward_tail_mean,
        "persistent_wheel_difference_semantics": {
            "dof_vel": "source reward sums state.joint_velocity[:, :4]; both wheel velocity channels are excluded",
            "action_rate": "source reward uses last_action - previous_action; a constant wheel difference has zero ongoing cost",
            "action_smooth": "source reward uses second action difference; a constant wheel difference has zero ongoing cost",
            "torques": "source reward includes all six torques, but has weight -1e-4",
            "phi0_symmetry": "source reward uses virtual-leg phi0 difference; it contains no wheel velocity difference",
            "tracking_vx": "rewards body forward velocity and does not identify which wheel produced it",
            "tracking_yaw_rate": "penalizes resulting body yaw error, without constraining the internal wheel split",
        },
        "zero_yaw_reset_probe": {
            "command": command.tolist(),
            "first_observation_last_frame": observations[0],
            "first_policy_action": first_action.tolist(),
            "first_policy_wheel_action_difference": float(first_action[4] - first_action[5]),
            "first_twelve_policy_wheel_action_difference": [
                float(values[4] - values[5]) for values in actions
            ],
            "first_observation_leg_error": observations[0][9:13],
            "first_observation_joint_velocity": observations[0][13:19],
        },
        "yaw_error_reward_sensitivity": {
            "control_dt_s": 0.02,
            "tracking_sigma": 0.25,
            "yaw_rate_error_rad_s": 0.06,
            "tracking_yaw_rate_loss_per_step": float(0.02 * (1.0 - np.exp(-(0.06 ** 2) / 0.25))),
            "tracking_yaw_rate_enhance_loss_per_step": float(0.02 * (1.0 - np.exp(-(0.06 ** 2) / 2.5))),
        },
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
