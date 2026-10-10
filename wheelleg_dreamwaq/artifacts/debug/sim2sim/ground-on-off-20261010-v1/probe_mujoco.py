"""Observe the unchanged formal MuJoCo runtime, with geometry refreshed on shadow data only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "sim2sim/mujoco"))
sys.path.insert(0, str(PROJECT))
from wheelleg_mujoco.contract import load_policy_contract
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from wheelleg_mujoco.model_semantics import compiled_model_semantics, stable_hash
from debug.sim2sim.root_cause_suite.variant_builders import build_robot_variant

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--actions-from", type=Path, required=True)
parser.add_argument("--no-ground", action="store_true")
parser.add_argument("--control-ticks", type=int, default=1)
parser.add_argument("--case", choices=("both", "zero_action", "shared_first_action_hold"), default="both")
args = parser.parse_args()
SUITE = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2"
MODEL = PROJECT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
MODEL_MANIFEST = PROJECT / "sim2sim/mujoco/model_manifest.json"
EXPORT = SUITE / "exports/run-02"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    assert args.control_ticks > 0
    args.output.mkdir(parents=True, exist_ok=False)
    protected = [MODEL, MODEL_MANIFEST, EXPORT / "actor.ts", EXPORT / "policy_manifest.json"]
    protected += list((PROJECT / "sim2sim/mujoco/wheelleg_mujoco").glob("*.py"))
    before = {str(path): digest(path) for path in protected}
    contract, _, model_manifest = load_policy_contract(EXPORT / "policy_manifest.json", EXPORT / "actor.ts", MODEL_MANIFEST)
    formal_runtime = WheelLegMujocoRuntime(MODEL, contract, model_manifest=model_manifest)
    variant = None
    runtime = formal_runtime
    if args.no_ground:
        variant = build_robot_variant(MODEL, args.output / "model/no_ground.xml", operations=("no_ground",))
        variant_model = mujoco.MjModel.from_xml_path(str(variant.model_path))
        expected = compiled_model_semantics(formal_runtime.model)
        assert all("floor" in (p["geom1"], p["geom2"]) for p in expected["contact_pairs"])
        expected["dimensions"]["npair"] = 0
        expected["contact_pairs"] = []
        actual = compiled_model_semantics(variant_model)
        assert actual == expected, "Non-contact compiled dynamics changed"
        assert not variant_model.geom_contype.any() and not variant_model.geom_conaffinity.any()
        diagnostic_manifest = {"dynamics_semantics_hash": stable_hash(actual), "debug_only": True}
        (args.output / "diagnostic-model-identity.json").write_text(json.dumps(diagnostic_manifest, indent=2), encoding="utf-8")
        runtime = WheelLegMujocoRuntime(variant.model_path, contract, model_manifest=diagnostic_manifest)
    model, data = runtime.model, runtime.data
    shadow = mujoco.MjData(model)
    wheel_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                 for name in ("left_wheel_proxy", "right_wheel_proxy")]
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    actions = np.asarray(json.loads(args.actions_from.read_text())["shared_first_action_clipped"], dtype=np.float64)
    assert actions.shape == (8, 6)
    commands = [(0., 0., .20), (0., 0., .17), (0., 0., .23), (1., 0., .20),
                (-1., 0., .20), (0., .6, .20), (0., -.6, .20), (.8, .5, .20)]
    samples = []
    case = ""
    environment_index = 0
    step_index = 0

    def snapshot(phase):
        # Do not call mj_forward on production data: that would replace its solver buffers.
        # Copying preserves qpos/qvel/time; fresh geometry is calculated on a separate MjData.
        mujoco.mj_copyData(shadow, model, data)
        mujoco.mj_forward(model, shadow)
        ground_z = 0.0 if args.no_ground else float(shadow.geom_xpos[floor_id, 2])
        gaps = [float(shadow.geom_xpos[gid, 2] - model.geom_size[gid, 0] - ground_z) for gid in wheel_ids]
        contacts = []
        normal = np.zeros(2)
        for actual_index, contact in enumerate(data.contact):
            pair = {int(contact.geom1), int(contact.geom2)}
            wheel = next((i for i, gid in enumerate(wheel_ids) if pair == {floor_id, gid}), None)
            if wheel is None:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, actual_index, force)
            normal[wheel] += force[0]
            contacts.append({"wheel": wheel, "distance_m": float(contact.dist),
                             "normal_force_n": float(force[0]), "constraint_address": int(contact.efc_address)})
        return {"case": case, "environment_index": environment_index, "phase": phase,
                "physics_step": step_index, "elapsed_s": float(data.time),
                "root_qpos": data.qpos[:7].tolist(), "root_qvel": data.qvel[:6].tolist(),
                "base_com_position_world": shadow.xipos[root_id].tolist(),
                "wheel_geom_centers_world": shadow.geom_xpos[wheel_ids].tolist(),
                "wheel_collision_bottom_gap_m": gaps,
                "normal_force_n": normal.tolist(), "contacts_last_solve": contacts,
                "current_geometry_contact_count": int(shadow.ncon),
                "force_freshness": "last_solved_physics_step" if step_index else "reset_forward_only_no_time_advance",
                "joint_position": data.qpos[7:].tolist(), "joint_velocity": data.qvel[6:].tolist()}

    original_step = mujoco.mj_step

    def observed_step(*positional, **keywords):
        nonlocal step_index
        result = original_step(*positional, **keywords)
        step_index += 1
        samples.append(snapshot("post_physics_step"))
        return result

    mujoco.mj_step = observed_step
    try:
        for environment_index, command in enumerate(commands):
            for case, action in (("zero_action", np.zeros(6)), ("shared_first_action_hold", actions[environment_index])):
                if args.case != "both" and case != args.case:
                    continue
                step_index = 0
                runtime.reset(np.asarray(command))
                assert step_index == 0 and data.time == 0.0
                samples.append(snapshot("reset_returned_before_first_action"))
                for _ in range(args.control_ticks):
                    runtime.step(action, np.asarray(command))
    finally:
        mujoco.mj_step = original_step
    after = {str(path): digest(path) for path in protected}
    assert after == before
    assert abs(data.time - args.control_ticks * contract.control_dt_s) < 1e-12
    replay_errors = []
    for environment_index, command in enumerate(commands):
        for case, action in (("zero_action", np.zeros(6)), ("shared_first_action_hold", actions[environment_index])):
            if args.case != "both" and case != args.case:
                continue
            runtime.reset(np.asarray(command))
            for _ in range(args.control_ticks):
                runtime.step(action, np.asarray(command))
            final = next(s for s in reversed(samples) if s["case"] == case and s["environment_index"] == environment_index)
            error = max(float(np.max(np.abs(data.qpos - np.r_[final["root_qpos"], final["joint_position"]]))),
                        float(np.max(np.abs(data.qvel - np.r_[final["root_qvel"], final["joint_velocity"]]))))
            replay_errors.append(error)
    assert max(replay_errors) == 0.0
    if args.no_ground:
        assert floor_id == -1 and model.npair == 0
        assert all(s["current_geometry_contact_count"] == 0 and not s["contacts_last_solve"] for s in samples)
    report = {"schema_version": "GroundOffMujocoProbeV1", "debug_only": True,
              "ground_enabled": not args.no_ground, "control_ticks": args.control_ticks,
              "test_case": args.case, "probe_sha256": digest(Path(__file__)),
              "variant_model_sha256": None if variant is None else variant.model_sha256,
              "non_contact_compiled_semantics_unchanged": True,
              "observer_replay_max_errors": replay_errors, "gravity_m_s2": model.opt.gravity.tolist(),
              "formal_ranking_eligible": False, "physics_dt_s": float(model.opt.timestep),
              "reset_physics_time_advanced_s": 0.0, "ground_top_z_m": 0.0,
              "geometry": [{"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
                            "type": int(model.geom_type[gid]), "radius_m": float(model.geom_size[gid, 0])}
                           for gid in wheel_ids],
              "pair_margin_m": model.pair_margin.tolist(), "pair_gap_m": model.pair_gap.tolist(),
              "shadow_geometry_only": True, "force_semantics": "mj_contactForce from production last solve; reset is forward-only",
              "sources_unchanged": True, "protected_source_sha256": after, "samples": samples}
    (args.output / "evidence.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("PROBE_COMPLETE", args.output)


if __name__ == "__main__":
    main()
