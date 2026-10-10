# WheelLeg MuJoCo Sim2Sim

This directory is an isolated Python 3.11 / MuJoCo 3.14.0 runtime for
`PhysicsV5` policy exports. It intentionally does not import Isaac Lab or the root
training package.

The generated model comes from the confirmed local XML
`wheel_leg_urdf4_self_mesh_all.xml`, while body mass, inertia, dummy branches,
and reset state are audited against `AssetBundleV2`. The original base STL is
losslessly split into three files because MuJoCo accepts at most 200,000 faces
per STL file.

`model_manifest.json` freezes the XML and mesh identities, explicit contacts,
equality solver settings, ground height, COM API semantics, adapter versions and
implementation hashes. Model generation also compares the USD and MuJoCo
authored-`q=0` visual AABBs in the same asset-root frame with a 1 mm hard limit.

```powershell
uv sync --project sim2sim/mujoco
uv run --project sim2sim/mujoco python scripts/build_mujoco_model.py
uv run --project sim2sim/mujoco python -m pytest sim2sim/mujoco/tests -q
```

The runtime applies one six-dimensional policy action for exactly 20 physics
steps at 1 kHz. Leg actions become position targets with `Kp=120`, `Kd=4`, and
an 18 Nm clamp. Wheel actions become velocity targets with `Kd=0.6` and a 9 Nm
clamp. Actor observations use the same 25-dimensional order and
`NormalizationV2` constants as the Isaac Lab environment.

The runtime has no rigid-body linear/angular speed limiter, no external
speed-limiting force/torque, and no joint-speed torque suppression. Physical
contacts, closed-chain constraints, passive damping, armature and motor
effort limits remain. `MujocoActionAdapterV2` and
`MujocoEvaluationContractV3` identify this change. A recorded speed failure
threshold stops evaluation; it never brakes the model.

The former `PhysxRigidAngularBiasV1` adapter is removed. PhysicsV4 policies
and V1/V2 evaluation results remain historical artifacts and are rejected by
the current loader/ranking; do not edit old manifests to relabel them as V5.
Fresh training and export must record `UnrestrictedVelocityPolicyV1`, the
explicit Isaac velocity sentinels, and the current model-manifest identity.

Export and deterministic evaluation are launched from the repository root:

```powershell
uv run python scripts/export_ppo_actor.py --checkpoint <model.pt> --output <export-dir>
uv run --project sim2sim/mujoco python scripts/evaluate_mujoco.py --policy <actor.ts> --manifest <policy_manifest.json> --output <evaluation-dir>
```
