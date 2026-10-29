import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import raw_task_profile_name
from wheelleg_mujoco.contract import _validate_raw_heading_metadata, load_policy_contract, sha256_file
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


PROJECT = Path(__file__).resolve().parents[2]
SIM = PROJECT / "sim2sim/mujoco"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    records = []
    for export in args.export:
        contract, saved, model = load_policy_contract(export / "policy_manifest.json", export / "actor.ts",
                                                      SIM / "model_manifest.json")
        profile = raw_task_profile_name(saved["training_reward_weights"], saved["heading_hold"])
        assert sha256_file(Path(saved["source_checkpoint"])) == saved["source_checkpoint_sha256"]
        actor = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
        golden = torch.load(export / "golden_vectors.pt", map_location="cpu", weights_only=False)
        with torch.inference_mode():
            error = float((actor(golden["history"]) - golden["expected_action_mean"]).abs().max())
        assert error <= 1e-7
        invalid = deepcopy(saved)
        invalid["training_reward_weights"]["tracking_yaw_rate"] = .5
        with_rejection = False
        try:
            _validate_raw_heading_metadata(invalid)
        except ValueError:
            with_rejection = True
        assert with_rejection
        runtime = WheelLegMujocoRuntime(SIM / "models" / model["model_xml"]["path"], contract, model)
        source = np.array([0., 0., .2])
        runtime.reset(source)
        assert runtime.heading.enabled == saved["heading_hold"]["enabled"]
        for yaw in (0., .6, 0., -.6, 0.):
            source[1] = yaw
            observation = runtime.prepare_policy_observation(source)
            if not runtime.heading.enabled:
                np.testing.assert_array_equal(runtime.effective_command, source)
            with torch.inference_mode():
                action = actor(torch.from_numpy(observation).view(1, -1)).numpy()[0]
            result = runtime.step(action, source)
            assert np.isfinite(result.observation).all() and np.isfinite(runtime.data.qvel).all()
        records.append({"export": str(export.resolve()), "profile": profile, "physical_steps": 5,
                        "heading_enabled": runtime.heading.enabled, "golden_max_abs_error": error,
                        "actor_sha256": sha256_file(export / "actor.ts"), "invalid_weight_rejected": True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
