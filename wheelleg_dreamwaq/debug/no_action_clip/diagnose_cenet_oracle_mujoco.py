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


def _policy_action(actor: torch.jit.ScriptModule, history: torch.Tensor, estimated_velocity: torch.Tensor | None) -> np.ndarray:
    encoded = actor.encoder(history)
    learned_velocity = encoded[:, 0:3]
    context_mu = encoded[:, 3:19]
    current_observation = history[:, -int(actor.current_obs_dim) :]
    velocity = learned_velocity if estimated_velocity is None else estimated_velocity
    raw_mean = actor.actor(torch.cat([current_observation, velocity, context_mu], dim=-1))
    return torch.clamp(raw_mean, -20.0, 20.0).detach().cpu().numpy().reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare normal CENet velocity input with MuJoCo oracle velocity input.")
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    export_dir = args.export.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(export_dir)
    actor = torch.jit.load(str(export_dir / "actor.ts"), map_location="cpu").eval()
    scenario_reports = []

    for name, command_values in FORMAL_SCENARIOS:
        command = np.asarray(command_values, dtype=np.float64)
        mode_reports = {}
        for mode in ("cenet", "oracle"):
            observation = runtime.reset(command)
            yaw_rates = []
            linear_velocity_errors = []
            action_history = []
            for _ in range(EXPECTED_TICKS):
                state = collect_kinematic_state(runtime.model, runtime.data, runtime.model_map, runtime.contract)
                target_velocity = _normalise_velocity(state.com_linear_velocity_control)
                history_tensor = torch.from_numpy(observation).unsqueeze(0)
                with torch.inference_mode():
                    encoded = actor.encoder(history_tensor)
                    learned_velocity = encoded[:, 0:3]
                    oracle_velocity = torch.from_numpy(target_velocity.astype(np.float32)).unsqueeze(0)
                    action = _policy_action(actor, history_tensor, None if mode == "cenet" else oracle_velocity)
                linear_velocity_errors.append((learned_velocity.detach().cpu().numpy()[0] - target_velocity).tolist())
                yaw_rates.append(float(state.angular_velocity_control[2]))
                action_history.append(action.tolist())
                observation = runtime.step(action, command).observation
            yaw_array = np.asarray(yaw_rates, dtype=np.float64)
            mode_reports[mode] = {
                "actual_yaw_rate_mae_rad_s_to_command": float(np.mean(np.abs(yaw_array - command[1]))),
                "actual_yaw_rate_mean_rad_s": float(np.mean(yaw_array)),
                "actual_yaw_rate_rms_rad_s": float(np.sqrt(np.mean(np.square(yaw_array)))),
                "mean_action": np.mean(np.asarray(action_history, dtype=np.float64), axis=0).tolist(),
                "mean_abs_action": np.mean(np.abs(np.asarray(action_history, dtype=np.float64)), axis=0).tolist(),
            }
        scenario_reports.append({"name": name, "command": list(command_values), "modes": mode_reports})

    report = {
        "schema_version": "DebugNoActionClipCENetOracleVelocityDiagnosticV1",
        "export": str(export_dir),
        "expected_ticks_per_scenario": EXPECTED_TICKS,
        "cenet_definition": "actor.encoder(history)[0:3], learned root COM linear velocity; no yaw-rate output",
        "oracle_definition": "MuJoCo control-frame COM linear velocity normalized with the same scale and clip used by the CENet target",
        "scenario_reports": scenario_reports,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
