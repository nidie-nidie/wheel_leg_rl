"""Check real base contacts, unchanged healthy motion, and frozen formal files."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from base_contact import base_contact_state, create_model
from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.runner import WheelLegMujocoRuntime

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
EXPORT = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2/exports/run-04"
model = PROJECT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
protected = json.loads((ROOT.parent / "standing-trace-20261010-v1/protected-before.json").read_text(encoding="utf-8"))
for path, expected in protected.items():
    assert sha256_file(Path(path)) == expected, path
contract, manifest, model_manifest = load_policy_contract(EXPORT / "policy_manifest.json", EXPORT / "actor.ts",
                                                         PROJECT / "sim2sim/mujoco/model_manifest.json")
candidate = create_model(model, ROOT / "models/base_contact_debug.xml")
formal = WheelLegMujocoRuntime(model, contract, model_manifest)
debug = WheelLegMujocoRuntime(candidate, contract)
command = np.asarray((0, 0, .20))
contact_proof = {}
for name, runtime in (("formal", formal), ("base_contact_debug", debug)):
    runtime.reset(command)
    bottom = base_contact_state(runtime.model, runtime.data)["base_proxy_min_z_m"]
    runtime.data.qpos[2] -= bottom + .002
    mujoco.mj_forward(runtime.model, runtime.data)
    contact_proof[name] = base_contact_state(runtime.model, runtime.data)
assert contact_proof["formal"]["base_floor_contact_count"] == 0
assert contact_proof["base_contact_debug"]["base_floor_contact_count"] > 0
assert contact_proof["base_contact_debug"]["base_floor_normal_force_n"] > 0
obs_formal, obs_debug = formal.reset(command), debug.reset(command)
policy = torch.jit.load(str(EXPORT / "actor.ts"), map_location="cpu").eval()
errors = {"qpos": 0.0, "qvel": 0.0}
contact_count = 0
for _ in range(500):
    with torch.inference_mode():
        act_formal = policy(torch.from_numpy(obs_formal).unsqueeze(0)).numpy()[0]
        act_debug = policy(torch.from_numpy(obs_debug).unsqueeze(0)).numpy()[0]
    obs_formal = formal.step(act_formal, command).observation
    result = debug.step(act_debug, command)
    obs_debug = result.observation
    contact_count += base_contact_state(debug.model, debug.data)["base_floor_contact_count"]
    for name in errors:
        errors[name] = max(errors[name], float(np.max(np.abs(getattr(formal.data, name) - getattr(debug.data, name)))))
assert all(value < 1e-10 for value in errors.values()), errors
assert contact_count == 0
for path, expected in protected.items():
    assert sha256_file(Path(path)) == expected, path
report = {"schema_version": "ViewerBaseContactVerificationV1", "passed": True,
          "forced_contact_proof": contact_proof, "nominal_ten_second_qpos_qvel_differences": errors,
          "nominal_base_contact_count": contact_count, "nominal_final_metrics": result.metrics,
          "formal_files_unchanged": len(protected), "training_performed": False,
          "limit": "Existing approximate base box; real contact verified, policy fall recovery not established"}
(ROOT / "base-contact-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("BASE_CONTACT_VERIFIED", json.dumps(report), flush=True)
