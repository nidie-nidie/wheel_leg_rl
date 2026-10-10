from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
EXPORT = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2/exports/run-04"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def column(data, name):
    return np.asarray([float(row[name]) for row in data])


def main():
    policy = torch.jit.load(str(EXPORT / "actor.ts"), map_location="cpu").eval()
    results = {}
    manifest = json.loads((EXPORT / "policy_manifest.json").read_text(encoding="utf-8"))
    scale = manifest["normalization"]["root_linear_velocity_scale"]
    for engine in ("isaac", "mujoco"):
        data = rows(ROOT / engine / "nominal_stand.csv")
        inputs = np.load(ROOT / engine / "policy_inputs.npz")
        history = torch.from_numpy(inputs["history"]).float()
        current = torch.from_numpy(inputs["current"]).float()
        truth = torch.from_numpy(np.stack([column(data, f"actual_v{axis}_mps") for axis in ("x", "y", "z")], axis=1)).float() * scale
        with torch.inference_mode():
            encoded = policy.encoder(history)
            original_action = policy(history)
            vx_only = encoded[:, :3].clone()
            vx_only[:, 0] = truth[:, 0]
            true_vx_action = policy.actor(torch.cat((current, vx_only, encoded[:, 3:19]), dim=-1)).clamp(-20, 20)
            true_xyz_action = policy.actor(torch.cat((current, truth, encoded[:, 3:19]), dim=-1)).clamp(-20, 20)
        recorded = np.stack([column(data, f"raw_action_{i}") for i in range(6)], axis=1)
        encoder_recorded = np.stack([column(data, f"estimated_v{axis}_normalized") for axis in ("x", "y", "z")], axis=1)
        action_error = float(np.max(np.abs(original_action.numpy() - recorded)))
        velocity_error = float(np.max(np.abs(encoded[:, :3].numpy() - encoder_recorded)))
        np.testing.assert_allclose(original_action.numpy(), recorded, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(encoded[:, :3].numpy(), encoder_recorded, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(inputs["current"], inputs["history"][:, -25:])
        np.testing.assert_allclose(inputs["history"].reshape(-1, 5, 25)[:, :, 6:9], 0, rtol=0, atol=1e-6)
        # Use identical ticks in both engines, excluding the initial settling period.
        steady = (column(data, "tick") >= 100) & (column(data, "tick") <= 498)
        actual = column(data, "actual_vx_mps")
        estimated = column(data, "estimated_vx_mps")
        statistics = {name: float(column(data, name)[steady].mean()) for name in (
            "actual_vx_mps", "estimated_vx_mps", "wheel_target_left_rad_s", "wheel_target_right_rad_s",
            "wheel_actual_left_rad_s", "wheel_actual_right_rad_s", "pitch_rad", "tilt_rad", "base_height_m",
            "leg_action_saturation_fraction")}
        statistics["vx_estimation_mae_mps"] = float(np.abs(estimated - actual)[steady].mean())
        statistics["vx_estimation_rmse_mps"] = float(np.sqrt(np.square(estimated - actual)[steady].mean()))
        for label, action in (("true_vx_only_offline", true_vx_action), ("true_xyz_offline", true_xyz_action)):
            clipped = action.numpy().clip(-1, 1)
            statistics[label + "_wheel_target_left_rad_s"] = float((25 * clipped[steady, 4]).mean())
            statistics[label + "_wheel_target_right_rad_s"] = float((25 * clipped[steady, 5]).mean())
        for index, row in enumerate(data):
            for i in range(6):
                row[f"offline_true_vx_raw_action_{i}"] = float(true_vx_action[index, i])
        with (ROOT / engine / "offline_true_vx_actions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        sample_ticks = (0, 1, 5, 25, 50, 100, 250, 498)
        results[engine] = {"sample_count": len(data), "steady_sample_count": int(steady.sum()),
                           "steady_window_s": [2.0, 9.96], "statistics": statistics,
                           "action_replay_max_abs_error": action_error,
                           "estimator_replay_max_abs_error": velocity_error,
                           "samples": [{key: data[i][key] for key in (
                               "tick", "time_s", "actual_vx_mps", "estimated_vx_mps",
                               "wheel_target_left_rad_s", "wheel_actual_left_rad_s", "pitch_rad")}
                               for i in sample_ticks if i < len(data)]}

    # Establish that observation-only instrumentation reproduced the prior production MuJoCo trajectory.
    old = rows(PROJECT / "artifacts/debug/sim2sim/angular-limit-alignment-20261010-v1/mujoco-evaluation/run-04/nominal_stand.csv")
    new = rows(ROOT / "mujoco/nominal_stand.csv")
    trajectory = {"post_vx_max_abs_error": float(np.max(np.abs(column(old, "vx_mps") - column(new, "post_vx_mps")))),
                  "raw_action_max_abs_error": float(max(np.max(np.abs(column(old, f"raw_action_{i}") - column(new, f"raw_action_{i}"))) for i in range(6)))}
    assert trajectory == {"post_vx_max_abs_error": 0.0, "raw_action_max_abs_error": 0.0}, trajectory
    original_isaac = json.loads((PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2/isaac_evaluation/run-04/summary.json").read_text(encoding="utf-8"))
    current_isaac = json.loads((ROOT / "isaac/summary.json").read_text(encoding="utf-8"))
    before_return = original_isaac["aggregate"]["scenarios"][0]["reward_sum"]
    after_return = current_isaac["aggregate"]["scenarios"][0]["reward_sum"]
    trajectory["isaac_nominal_stand_reward_before"] = before_return
    trajectory["isaac_nominal_stand_reward_after"] = after_return
    trajectory["isaac_nominal_stand_reward_abs_difference"] = abs(before_return - after_return)
    immutable = json.loads((ROOT / "protected-before.json").read_text(encoding="utf-8"))
    changed = [path for path, expected in immutable.items() if digest(path) != expected]
    assert not changed, changed
    report = {"schema_version": "StandingDiagnosticAnalysisV1", "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
              "source_actor_sha256": digest(EXPORT / "actor.ts"), "engines": results,
              "training_command_sampling": manifest["command_sampling"],
              "instrumentation_reproduction": trajectory, "protected_files_unchanged": len(immutable),
              "offline_probe_limit": "Recorded observation replay only; does not demonstrate closed-loop recovery",
              "evidence_sha256": {str(path): digest(path) for engine in ("isaac", "mujoco")
                                  for path in (ROOT / engine).iterdir() if path.is_file()}}
    (ROOT / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"engines": {key: value["statistics"] for key, value in results.items()},
                      "reproduction": trajectory, "protected_files_unchanged": len(immutable)}, indent=2), flush=True)
    print("STANDING_ANALYSIS_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
