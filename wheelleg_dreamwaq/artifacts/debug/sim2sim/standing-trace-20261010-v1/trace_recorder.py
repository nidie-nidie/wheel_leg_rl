"""Observe policy boundaries only; never write simulation or policy state."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


class StandingTrace:
    def __init__(self, output, engine, normalization, wheel_action_scale):
        self.output = Path(output)
        self.engine = engine
        self.normalization = normalization
        self.wheel_scale = float(wheel_action_scale)
        self.rows = []
        self.inputs = []
        self.histories = []
        self.contexts = []

    def before(self, tick, time_s, command, current, history, actual_velocity,
               estimated_normalized, context, action, joint_position,
               joint_velocity, gravity, height, target_normalized=None):
        command, current, history = map(array, (command, current, history))
        actual_velocity, estimated_normalized = map(array, (actual_velocity, estimated_normalized))
        action, joint_position, joint_velocity, gravity = map(array, (action, joint_position, joint_velocity, gravity))
        assert current.shape == (25,) and history.shape == (125,)
        np.testing.assert_array_equal(history[-25:], current)
        expected_command = np.clip([
            command[0] / self.normalization["vx_max_abs"],
            command[1] / self.normalization["yaw_rate_max_abs"],
            (command[2] - self.normalization["nominal_base_height"]) / self.normalization["height_command_span"],
        ], -1, 1)
        np.testing.assert_allclose(current[6:9], expected_command, rtol=0, atol=1e-6)
        scale = self.normalization["root_linear_velocity_scale"]
        normalized_truth = np.clip(actual_velocity * scale,
                                   -self.normalization["root_linear_velocity_clip"],
                                   self.normalization["root_linear_velocity_clip"])
        if target_normalized is not None:
            np.testing.assert_allclose(array(target_normalized), normalized_truth, rtol=0, atol=1e-6)
        assert np.max(np.abs(actual_velocity * scale)) < self.normalization["root_linear_velocity_clip"]
        clipped = np.clip(action, -1, 1)
        row = {"engine": self.engine, "tick": tick, "time_s": float(time_s),
               "command_vx_mps": float(command[0]), "command_yaw_rad_s": float(command[1]),
               "command_height_m": float(command[2]), "base_height_m": float(height),
               "pitch_rad": float(np.arcsin(np.clip(gravity[0], -1, 1))),
               "tilt_rad": float(np.arccos(np.clip(-gravity[2], -1, 1))),
               "leg_action_saturation_fraction": float(np.mean(np.abs(action[:4]) >= 1))}
        for i, axis in enumerate(("x", "y", "z")):
            row[f"actual_v{axis}_mps"] = float(actual_velocity[i])
            row[f"estimated_v{axis}_mps"] = float(estimated_normalized[i] / scale)
            row[f"estimated_v{axis}_normalized"] = float(estimated_normalized[i])
            row[f"target_v{axis}_normalized"] = float(normalized_truth[i])
            row[f"network_command_{i}"] = float(current[6+i])
        for i in range(6):
            row[f"raw_action_{i}"] = float(action[i])
            row[f"clipped_action_{i}"] = float(clipped[i])
            row[f"joint_position_{i}_rad"] = float(joint_position[i])
            row[f"joint_velocity_{i}_rad_s"] = float(joint_velocity[i])
        for i, side in enumerate(("left", "right")):
            row[f"wheel_target_{side}_rad_s"] = float(self.wheel_scale * clipped[4+i])
            row[f"wheel_actual_{side}_rad_s"] = float(joint_velocity[4+i])
        self.rows.append(row)
        self.inputs.append(current)
        self.histories.append(history)
        self.contexts.append(array(context))

    def after(self, velocity, action, torque, failed=False):
        velocity, action, torque = map(array, (velocity, action, torque))
        # Isaac auto-resets done environments inside step; that state belongs to a new episode.
        if not failed:
            np.testing.assert_allclose(action, [self.rows[-1][f"clipped_action_{i}"] for i in range(6)], rtol=0, atol=1e-6)
        self.rows[-1]["post_vx_mps"] = float("nan") if failed else float(velocity[0])
        self.rows[-1]["done"] = int(failed)
        for i in range(6):
            self.rows[-1][f"post_applied_torque_{i}_nm"] = float("nan") if failed else float(torque[i])

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        assert self.rows, "No standing observations captured"
        path = self.output / "nominal_stand.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        np.savez_compressed(self.output / "policy_inputs.npz", current=np.asarray(self.inputs),
                            history=np.asarray(self.histories), context_mu=np.asarray(self.contexts))
        (self.output / "trace_identity.json").write_text(json.dumps({
            "schema_version": "StandingBoundaryTraceV1", "engine": self.engine,
            "sample_count": len(self.rows), "sampling": "pre_action_same_timestep",
            "velocity_output": "CENet normalized output divided by root_linear_velocity_scale",
            "normalization": self.normalization, "wheel_action_scale": self.wheel_scale,
            "history": "unchanged current production history",
        }, indent=2), encoding="utf-8")
        print("STANDING_TRACE_COMPLETE", self.engine, len(self.rows), flush=True)
