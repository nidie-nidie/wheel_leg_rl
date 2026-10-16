from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS
from wheelleg_mujoco.observation import collect_kinematic_state

from debug.no_action_clip.mujoco_adapter import load_runtime


EXPECTED_TICKS = 500
ROOT_LINEAR_VELOCITY_SCALE = 2.0
ROOT_LINEAR_VELOCITY_CLIP = 6.0


def _normalise_velocity(values: np.ndarray) -> np.ndarray:
    return np.clip(values * ROOT_LINEAR_VELOCITY_SCALE, -ROOT_LINEAR_VELOCITY_CLIP, ROOT_LINEAR_VELOCITY_CLIP)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DebugNoActionClipV2 CENet velocity estimates with MuJoCo state.")
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    export_dir = args.export.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(export_dir)
    actor = torch.jit.load(str(export_dir / "actor.ts"), map_location="cpu").eval()
    encoder = torch.jit.load(str(export_dir / "encoder.ts"), map_location="cpu").eval()
    scenario_reports = []
    for name, command_values in FORMAL_SCENARIOS:
        command = np.asarray(command_values, dtype=np.float64)
        observation = runtime.reset(command)
        estimates = []
        targets = []
        yaw_rates = []
        for _ in range(EXPECTED_TICKS):
            history_tensor = torch.from_numpy(observation).unsqueeze(0)
            with torch.inference_mode():
                encoded = encoder(history_tensor).detach().cpu().numpy()[0]
                action = actor(history_tensor).detach().cpu().numpy().reshape(-1)
            state = collect_kinematic_state(runtime.model, runtime.data, runtime.model_map, runtime.contract)
            estimates.append(encoded[:3])
            targets.append(_normalise_velocity(state.com_linear_velocity_control))
            yaw_rates.append(float(state.angular_velocity_control[2]))
            observation = runtime.step(action, command).observation
        estimate_array = np.asarray(estimates, dtype=np.float64)
        target_array = np.asarray(targets, dtype=np.float64)
        error = estimate_array - target_array
        scenario_reports.append(
            {
                "name": name,
                "command": list(command_values),
                "velocity_estimate_mae_normalized": np.mean(np.abs(error), axis=0).tolist(),
                "velocity_estimate_rmse_normalized": np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
                "velocity_estimate_mae_mps_equivalent": (np.mean(np.abs(error), axis=0) / ROOT_LINEAR_VELOCITY_SCALE).tolist(),
                "actual_yaw_rate_mae_rad_s_to_command": float(np.mean(np.abs(np.asarray(yaw_rates) - command[1]))),
                "actual_yaw_rate_mean_rad_s": float(np.mean(yaw_rates)),
            }
        )

    report = {
        "schema_version": "DebugNoActionClipCENetVelocityDiagnosticV1",
        "export": str(export_dir),
        "expected_ticks_per_scenario": EXPECTED_TICKS,
        "estimated_velocity_definition": "CENet output[0:3], normalized root COM linear velocity; no yaw-rate output",
        "target_definition": "MuJoCo control-frame COM linear velocity normalized by scale 2.0 and clipped to +/-6.0",
        "scenarios": scenario_reports,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
