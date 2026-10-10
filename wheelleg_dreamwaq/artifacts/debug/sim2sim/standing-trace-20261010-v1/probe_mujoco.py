from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wheelleg_mujoco.angular_limit import angular_limit_contract
from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.observation import collect_kinematic_state
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from trace_recorder import StandingTrace

PROJECT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actor = args.export / "actor.ts"
    model = PROJECT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
    contract, manifest, model_manifest = load_policy_contract(
        args.export / "policy_manifest.json", actor, PROJECT / "sim2sim/mujoco/model_manifest.json")
    assert sha256_file(model) == model_manifest["model_xml"]["sha256"]
    policy = torch.jit.load(str(actor), map_location="cpu").eval()
    runtime = WheelLegMujocoRuntime(model, contract, model_manifest)
    trace = StandingTrace(args.output, "mujoco", contract.normalization, contract.wheel_action_scale)
    command = np.asarray((0, 0, .20), dtype=np.float64)
    observation = runtime.reset(command)
    for tick in range(500):
        state = collect_kinematic_state(runtime.model, runtime.data, runtime.model_map, contract)
        tensor = torch.from_numpy(observation).unsqueeze(0)
        with torch.inference_mode():
            encoded = policy.encoder(tensor)
            action = policy(tensor).numpy()[0]
            reconstructed = policy.actor(torch.cat((tensor[:, -25:], encoded[:, :3], encoded[:, 3:19]), dim=-1)).clamp(-20, 20)
        np.testing.assert_array_equal(action, reconstructed.numpy()[0])
        trace.before(tick, runtime.data.time, command, observation[-25:], observation,
                     state.com_linear_velocity_control, encoded[0, :3], encoded[0, 3:19], action,
                     state.joint_position_control, state.joint_velocity_control,
                     state.projected_gravity_control, state.base_height_m)
        result = runtime.step(action, command)
        trace.after(np.asarray((result.metrics["vx_mps"], 0, 0)), result.clipped_action, result.applied_torque)
        observation = result.observation
    trace.save()
    (args.output / "runtime_identity.json").write_text(json.dumps({
        "schema_version": "StandingMujocoDiagnosticV1", "source_checkpoint": manifest["source_checkpoint"],
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"], "actor_sha256": sha256_file(actor),
        "model_sha256": sha256_file(model), "angular_limit": angular_limit_contract(),
        "control_dt_s": contract.control_dt_s, "physics_dt_s": contract.physics_dt_s,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
