"""Observe the unchanged formal MuJoCo runtime, with geometry refreshed on shadow data only."""
from __future__ import annotations

import argparse
from dataclasses import replace
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
from wheelleg_mujoco.model_map import CONTROLLED_JOINT_SIGNS
from wheelleg_mujoco.model_semantics import compiled_model_semantics, stable_hash
from wheelleg_mujoco.observation import collect_kinematic_state
from debug.sim2sim.root_cause_suite.variant_builders import build_robot_variant

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--actions-from", type=Path, required=True)
parser.add_argument("--no-ground", action="store_true")
parser.add_argument("--control-ticks", type=int, default=1)
parser.add_argument("--case", choices=("both", "zero_action", "shared_first_action_hold"), default="both")
parser.add_argument("--drive-mode", choices=("formal", "direct_shared", "direct_zero"), default="formal")
parser.add_argument("--torques-from", type=Path, required=True)
parser.add_argument("--mechanics-case", choices=("baseline", "passive_off", "closure_off", "dt_5ms"), required=True)
parser.add_argument("--initial-state-from", type=Path)
parser.add_argument("--limit-period-s", type=float, choices=(.001, .005), required=True)
args = parser.parse_args()
SUITE = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2"
MODEL = PROJECT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
MODEL_MANIFEST = PROJECT / "sim2sim/mujoco/model_manifest.json"
EXPORT = SUITE / "exports/run-02"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    assert args.control_ticks == 1 and args.no_ground and args.case == "shared_first_action_hold"
    torque_record = json.loads(args.torques_from.read_text())
    torque_key = "zero_torque_control_nm" if args.drive_mode == "direct_zero" else "shared_first_torque_control_nm"
    torques = np.asarray(torque_record[torque_key], dtype=np.float64)
    assert torques.shape == (8, 6) and np.isfinite(torques).all()
    assert np.all(np.abs(torques) <= np.asarray(torque_record["effort_limits_nm"]))
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
    assert args.drive_mode == "direct_shared" and args.initial_state_from is not None
    model, data = runtime.model, runtime.data
    mechanics_before = compiled_model_semantics(model)
    expected_mechanics = json.loads(json.dumps(mechanics_before))
    controlled_names = set(mechanics_before["controlled_joint_order"])
    if args.mechanics_case == "passive_off":
        passive_ids = [j for j in range(1, model.njnt)
                       if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) not in controlled_names]
        assert len(passive_ids) == 20
        passive_dofs = model.jnt_dofadr[passive_ids]
        assert np.all(model.dof_damping[passive_dofs] == 0.05)
        model.dof_damping[passive_dofs] = 0.0
        for joint in expected_mechanics["joints"]:
            if joint["name"] != "base_free" and joint["name"] not in controlled_names:
                joint["damping"] = [0.0]
    elif args.mechanics_case == "closure_off":
        assert model.neq == 8 and np.all(model.eq_active0)
        model.eq_active0[:] = False
        for equality in expected_mechanics["equalities"]:
            equality["active_initial"] = False
    elif args.mechanics_case == "dt_5ms":
        model.opt.timestep = 0.005
        expected_mechanics["options"]["timestep"] = 0.005
        contract = replace(contract, physics_dt_s=0.005, physics_steps_per_action=4)
        runtime.contract = contract
    mechanics_after = compiled_model_semantics(model)
    assert mechanics_after == expected_mechanics, "Unexpected compiled mechanics change"
    assert contract.physics_dt_s * contract.physics_steps_per_action == contract.control_dt_s == 0.02
    aids = runtime.model_map.actuator_ids
    dofs = runtime.model_map.dof_addresses
    assert np.all(model.actuator_dyntype[aids] == mujoco.mjtDyn.mjDYN_NONE)
    assert np.all(model.actuator_gaintype[aids] == mujoco.mjtGain.mjGAIN_FIXED)
    assert np.all(model.actuator_biastype[aids] == mujoco.mjtBias.mjBIAS_NONE)
    assert np.all(model.actuator_gainprm[aids, 0] == 1.0)
    assert np.array_equal(model.actuator_gear[aids], np.tile([1., 0., 0., 0., 0., 0.], (6, 1)))
    assert np.all(model.dof_damping[dofs] == 0.0)
    formal_compute_torque = runtime.controller.compute_torque

    def select_torque(index):
        if args.drive_mode == "formal":
            runtime.controller.compute_torque = formal_compute_torque
        else:
            fixed = torques[index] * CONTROLLED_JOINT_SIGNS
            runtime.controller.compute_torque = lambda data, targets: fixed.copy()

    # Debug only: per-body bias torque -I_world * omega * (1-cap/|omega|) / period.
    # This follows PhysX 5.6.1 forwardDynamic2.cu; never writes qvel.
    cap_shadow = mujoco.MjData(model)
    limit_steps = round(args.limit_period_s / contract.physics_dt_s)
    assert limit_steps >= 1 and abs(limit_steps * contract.physics_dt_s - args.limit_period_s) < 1.e-12
    cap = np.deg2rad(100.)
    original_apply_torque = runtime.controller.apply_torque
    limit_events = []

    def limited_apply_torque(actual_data, torque):
        original_apply_torque(actual_data, torque)
        substep = round(actual_data.time / contract.physics_dt_s)
        if substep % limit_steps == 0:
            mujoco.mj_copyData(cap_shadow, model, actual_data)
            mujoco.mj_forward(model, cap_shadow)
            actual_data.xfrc_applied.fill(0.)
            for body_id in range(1, model.nbody):
                spatial = np.zeros(6)
                mujoco.mj_objectVelocity(model, cap_shadow, mujoco.mjtObj.mjOBJ_BODY, body_id, spatial, 0)
                omega = spatial[:3]
                speed = np.linalg.norm(omega)
                if speed > cap:
                    rotation = cap_shadow.ximat[body_id].reshape(3, 3)
                    inertia = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
                    actual_data.xfrc_applied[body_id, 3:] = -(inertia @ omega) * (1. - cap / speed) / args.limit_period_s
            limit_events.append({"time_s": actual_data.time,
                                 "body_torque_world_nm": actual_data.xfrc_applied[:, 3:].tolist()})

    runtime.controller.apply_torque = limited_apply_torque
    initial_record = None if args.initial_state_from is None else json.loads(args.initial_state_from.read_text())
    if initial_record is not None:
        assert args.drive_mode != "formal", "Matched initial-state experiment is direct torque only"
        initial_snapshot = next(s for s in initial_record["samples"] if s["phase"] == "reset_returned_before_first_action")
        source_ids = [initial_record["joint_names"].index(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
                      for j in range(1, model.njnt)]

    def align_initial_state(index):
        if initial_record is None:
            return
        data.qpos[7:] = np.asarray(initial_snapshot["joint_position"])[index, source_ids]
        data.qvel[6:] = np.asarray(initial_snapshot["joint_velocity"])[index, source_ids]
        pose = np.asarray(initial_snapshot["root_pose_xyzw"])[index]
        data.qpos[2] = pose[2]
        quat = pose[[6, 3, 4, 5]]
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        assert np.max(np.abs(initial_snapshot["root_velocity_world"][index])) == 0.0
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)  # Reset-only; never alter a live post-step solver buffer.

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
                "equality_active_live": data.eq_active.tolist(),
                "force_freshness": "last_solved_physics_step" if step_index else "reset_forward_only_no_time_advance",
                "joint_position": data.qpos[7:].tolist(), "joint_velocity": data.qvel[6:].tolist(),
                "external_ctrl_native_nm": data.ctrl[aids].tolist(),
                "actuator_force_native_nm": data.actuator_force[aids].tolist(),
                "generalized_actuator_force_native_nm": data.qfrc_actuator[dofs].tolist(),
                "angular_control_fresh_jacobian_rad_s": collect_kinematic_state(model, shadow, runtime.model_map, contract).angular_velocity_control.tolist()}

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
                select_torque(environment_index)
                runtime.reset(np.asarray(command))
                align_initial_state(environment_index)
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
            select_torque(environment_index)
            runtime.reset(np.asarray(command))
            align_initial_state(environment_index)
            for _ in range(args.control_ticks):
                runtime.step(action, np.asarray(command))
            final = next(s for s in reversed(samples) if s["case"] == case and s["environment_index"] == environment_index)
            error = max(float(np.max(np.abs(data.qpos - np.r_[final["root_qpos"], final["joint_position"]]))),
                        float(np.max(np.abs(data.qvel - np.r_[final["root_qvel"], final["joint_velocity"]]))))
            replay_errors.append(error)
    assert max(replay_errors) == 0.0
    for row in samples:
        if row["physics_step"] == 0:
            continue
        if args.drive_mode != "formal":
            expected = torques[row["environment_index"]] * CONTROLLED_JOINT_SIGNS
            for field in ("external_ctrl_native_nm", "actuator_force_native_nm", "generalized_actuator_force_native_nm"):
                assert np.array_equal(np.asarray(row[field]), expected), (field, row["physics_step"])
        assert np.max(np.abs(np.asarray(row["joint_velocity"])[dofs - 6])) < 45.0
    if args.no_ground:
        assert floor_id == -1 and model.npair == 0
        assert all(s["current_geometry_contact_count"] == 0 and not s["contacts_last_solve"] for s in samples)
    report = {"schema_version": "AngularLimitMujocoCandidateV1", "angular_limit_deg_s": 100.,
              "limit_period_s": args.limit_period_s, "limit_events": limit_events, "mechanics_case": args.mechanics_case,
              "mechanics_before": mechanics_before, "mechanics_after": mechanics_after,
              "only_declared_mechanics_changes": True, "control_substeps": contract.physics_steps_per_action, "debug_only": True,
              "ground_enabled": not args.no_ground, "control_ticks": args.control_ticks,
              "test_case": args.case, "drive_mode": args.drive_mode,
              "torque_input_sha256": digest(args.torques_from), "free_base": True,
              "matching_initial_state_sha256": None if args.initial_state_from is None else digest(args.initial_state_from),
              "controlled_dof_ids": dofs.tolist(), "joint_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)],
              "dof_damping": model.dof_damping.tolist(), "dof_armature": model.dof_armature.tolist(),
              "equality_active": model.eq_active0.tolist(), "active_ctrl_ranges": model.actuator_ctrlrange[aids].tolist(), "probe_sha256": digest(Path(__file__)),
              "variant_model_sha256": None if variant is None else variant.model_sha256,
              "base_no_ground_compiled_semantics_validated_before_intervention": True,
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
