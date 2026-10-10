"""Check whether passive viewer sync changes live dynamics buffers."""
import json
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from wheelleg_mujoco.contract import load_policy_contract
from wheelleg_mujoco.runner import WheelLegMujocoRuntime

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
EXPORT = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2/exports/run-04"
contract, _, model_manifest = load_policy_contract(EXPORT / "policy_manifest.json", EXPORT / "actor.ts",
                                                 PROJECT / "sim2sim/mujoco/model_manifest.json")
model_path = PROJECT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
reference = WheelLegMujocoRuntime(model_path, contract, model_manifest)
displayed = WheelLegMujocoRuntime(model_path, contract, model_manifest)
policy = torch.jit.load(str(EXPORT / "actor.ts"), map_location="cpu").eval()
command = np.asarray((0, 0, .20))
obs_ref, obs_vis = reference.reset(command), displayed.reset(command)
errors = {key: 0.0 for key in ("qpos", "qvel", "xpos", "qacc", "qacc_warmstart", "qfrc_constraint", "xfrc_applied")}
trajectory = {"qpos": 0.0, "qvel": 0.0}
first_failure = None
with mujoco.viewer.launch_passive(displayed.model, displayed.data, show_left_ui=False, show_right_ui=False) as viewer:
    for _ in range(500):
        with torch.inference_mode():
            action_ref = policy(torch.from_numpy(obs_ref).unsqueeze(0)).numpy()[0]
            action_vis = policy(torch.from_numpy(obs_vis).unsqueeze(0)).numpy()[0]
        ref_result = reference.step(action_ref, command)
        vis_result = displayed.step(action_vis, command)
        obs_ref, obs_vis = ref_result.observation, vis_result.observation
        if first_failure is None and (vis_result.metrics["tilt_rad"] > .80 or vis_result.metrics["base_height_m"] < .10):
            first_failure = {"time_s": displayed.data.time, "metrics": vis_result.metrics}
        saved = {key: getattr(displayed.data, key).copy() for key in errors}
        viewer.sync(state_only=True)
        for key in errors:
            errors[key] = max(errors[key], float(np.max(np.abs(saved[key] - getattr(displayed.data, key)))))
        for key in trajectory:
            trajectory[key] = max(trajectory[key], float(np.max(np.abs(getattr(reference.data, key) - getattr(displayed.data, key)))))
report = {"schema_version": "PassiveViewerStateEffectV1", "sync_max_buffer_changes": errors,
          "ten_second_closed_loop_max_differences": trajectory,
          "reference_final_metrics": ref_result.metrics, "live_viewer_final_metrics": vis_result.metrics,
          "live_viewer_first_failure": first_failure,
          "note": "500 policy actions; live viewer compared to the identical runtime without a viewer"}
shadow_runtime = WheelLegMujocoRuntime(model_path, contract, model_manifest)
reference.reset(command)
obs_ref, obs_shadow = reference.reset(command), shadow_runtime.reset(command)
render_model = mujoco.MjModel.from_xml_path(str(model_path))
render_data = mujoco.MjData(render_model)
shadow_errors = {"qpos": 0.0, "qvel": 0.0, "xfrc_applied": 0.0}
with mujoco.viewer.launch_passive(render_model, render_data, show_left_ui=False, show_right_ui=False) as viewer:
    for _ in range(500):
        with torch.inference_mode():
            action_ref = policy(torch.from_numpy(obs_ref).unsqueeze(0)).numpy()[0]
            action_shadow = policy(torch.from_numpy(obs_shadow).unsqueeze(0)).numpy()[0]
        obs_ref = reference.step(action_ref, command).observation
        shadow_result = shadow_runtime.step(action_shadow, command)
        obs_shadow = shadow_result.observation
        with viewer.lock():
            mujoco.mj_copyData(render_data, render_model, shadow_runtime.data)
        viewer.sync(state_only=True)
        for key in shadow_errors:
            shadow_errors[key] = max(shadow_errors[key], float(np.max(np.abs(getattr(reference.data, key) - getattr(shadow_runtime.data, key)))))
assert all(value == 0 for value in shadow_errors.values()), shadow_errors
report["independent_display_ten_second_max_differences"] = shadow_errors
report["independent_display_final_metrics"] = shadow_result.metrics
(ROOT / "viewer-sync-probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("VIEWER_SYNC_PROBE_COMPLETE", json.dumps(report), flush=True)
