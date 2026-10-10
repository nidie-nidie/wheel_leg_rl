"""Generate guarded diagnostic copies; never modify the source probes."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "same-torque-20261010-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def main():
    protected = json.loads((OLD / "protected-before.json").read_text(encoding="utf-8"))
    assert len(protected) == 93
    for path, expected in protected.items():
        assert digest(path) == expected, path
    (ROOT / "protected-before.json").write_text(json.dumps(protected, indent=2), encoding="utf-8")
    source_hashes = {}
    for engine in ("isaac", "mujoco"):
        path = OLD / f"probe_{engine}.py"
        original = source = path.read_text(encoding="utf-8")
        source_hashes[str(path)] = digest(path)
        source = replace_once(source, 'parser.add_argument("--torques-from", type=Path, required=True)',
                              'parser.add_argument("--torques-from", type=Path, required=True)\n'
                              'parser.add_argument("--mechanics-case", choices=("baseline", "passive_off", "closure_off", "dt_5ms"), required=True)')
        if engine == "isaac":
            source = replace_once(source, '        self.probe_sensors = []', '''        self.disabled_closure_paths = []
        if args.mechanics_case == "closure_off":
            stage = self.sim.get_initial_stage()
            for index in range(self.cfg.scene.num_envs):
                for relative in ("jIO/jIO_loop_closure", "jKN/jKN_loop_closure", "jEC/jAG_loop_closure", "jCF/jCF_revolute_joint"):
                    path = f"/World/envs/env_{index}/Robot/{relative}"
                    joint = UsdPhysics.Joint(stage.GetPrimAtPath(path))
                    assert joint
                    joint.CreateJointEnabledAttr(False).Set(False)
                    assert joint.GetJointEnabledAttr().Get() is False
                    self.disabled_closure_paths.append(path)
            assert len(self.disabled_closure_paths) == 32
        self.probe_sensors = []''')
            source = replace_once(source, '            assert enabled is not False',
                                  '            assert (enabled is not False) == (args.mechanics_case != "closure_off")')
            source = replace_once(source, '        self.drive_after = self.drive_identity()', '''        if args.mechanics_case == "passive_off":
            self.robot.write_joint_damping_to_sim(0.0, joint_ids=self._passive_joint_ids)
            self.robot.actuators["passive"].damping.zero_()
        self.drive_after = self.drive_identity()''')
            source = replace_once(source, '        assert self.drive_after == expected, "Unexpected change outside active PD gains"\n        self._assert_passive_actuator_contract()', '''        if args.mechanics_case == "passive_off":
            for row in expected["damping"]:
                for joint in self._passive_joint_ids:
                    row[joint] = 0.0
            expected["actuator_cache"]["passive"]["damping"] = np.zeros_like(expected["actuator_cache"]["passive"]["damping"]).tolist()
        assert self.drive_after == expected, "Unexpected drive parameter change"
        if args.mechanics_case != "passive_off":
            self._assert_passive_actuator_contract()
        else:
            assert torch.count_nonzero(self.robot.data.joint_vel_target[:, self._passive_joint_ids]) == 0''')
            source = replace_once(source, '    assert args.control_ticks > 0',
                                  '    assert args.control_ticks > 0\n    assert args.mechanics_case != "dt_5ms" and args.drive_mode == "direct_shared"')
            source = replace_once(source, '            _, _, dones, _ = wrapped.step(action)\n            assert not dones.any(), "Probe crossed an episode boundary"', '''            # Same DirectRLEnv physics loop, without closed-geometry reward/done checks.
            env._pre_physics_step(action)
            assert not env.sim.has_gui() and not env.sim.has_rtx_sensors()
            for _ in range(env.cfg.decimation):
                env._sim_step_counter += 1
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)''')
            source = replace_once(source, '"schema_version": "SameTorqueIsaacProbeV1"',
                                  '"schema_version": "MechanicsIsolationIsaacV1", "mechanics_case": args.mechanics_case,\n'
                                  '              "disabled_closure_paths": env.disabled_closure_paths, "physics_loop_only": True')
            source = replace_once(source, '"production_reset_and_step_inherited": True',
                                  '"production_reset_inherited": True, "production_step_inherited": False')
        else:
            source = replace_once(source, 'import argparse', 'import argparse\nfrom dataclasses import replace')
            source = replace_once(source, '    model, data = runtime.model, runtime.data', '''    assert args.drive_mode == "direct_shared" and args.initial_state_from is not None
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
    assert contract.physics_dt_s * contract.physics_steps_per_action == contract.control_dt_s == 0.02''')
            source = replace_once(source, '                "current_geometry_contact_count": int(shadow.ncon),',
                                  '                "current_geometry_contact_count": int(shadow.ncon),\n'
                                  '                "equality_active_live": data.eq_active.tolist(),')
            source = replace_once(source, '"schema_version": "SameTorqueMujocoProbeV1"',
                                  '"schema_version": "MechanicsIsolationMujocoV1", "mechanics_case": args.mechanics_case,\n'
                                  '              "mechanics_before": mechanics_before, "mechanics_after": mechanics_after,\n'
                                  '              "only_declared_mechanics_changes": True, "control_substeps": contract.physics_steps_per_action')
            source = replace_once(source, '"non_contact_compiled_semantics_unchanged": True',
                                  '"base_no_ground_compiled_semantics_validated_before_intervention": True')
        destination = ROOT / path.name
        destination.write_text(source, encoding="utf-8")
        (ROOT / f"{engine}-changes.diff").write_text("".join(difflib.unified_diff(
            original.splitlines(keepends=True), source.splitlines(keepends=True),
            fromfile=str(path), tofile=str(destination))), encoding="utf-8")
    (ROOT / "probe-source-hashes.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
    print("PROBES_GENERATED", ROOT)


if __name__ == "__main__":
    main()
