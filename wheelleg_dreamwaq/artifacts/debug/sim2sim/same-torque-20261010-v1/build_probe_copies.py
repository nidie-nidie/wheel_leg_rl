"""Generate narrowly modified copies; leave the preceding diagnostic immutable."""
from pathlib import Path
import difflib

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "ground-on-off-20261010-v1"


def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


extra_args = '''parser.add_argument("--drive-mode", choices=("formal", "direct_shared", "direct_zero"), default="formal")
parser.add_argument("--torques-from", type=Path, required=True)
'''

isaac_methods = '''
    def drive_identity(self):
        view = self.robot.root_physx_view
        stage = self.sim.get_initial_stage()
        closure = {}
        for rel in ("jIO/jIO_loop_closure", "jKN/jKN_loop_closure", "jEC/jAG_loop_closure", "jCF/jCF_revolute_joint"):
            path = f"/World/envs/env_0/Robot/{rel}"
            joint = UsdPhysics.Joint(stage.GetPrimAtPath(path))
            assert joint
            enabled = joint.GetJointEnabledAttr().Get()
            assert enabled is not False
            closure[path] = enabled is not False
        return {"stiffness": cpu(view.get_dof_stiffnesses()).tolist(),
                "damping": cpu(view.get_dof_dampings()).tolist(),
                "armature": cpu(view.get_dof_armatures()).tolist(),
                "effort_limits": cpu(view.get_dof_max_forces()).tolist(),
                "velocity_limits": cpu(view.get_dof_max_velocities()).tolist(),
                "position_limits": cpu(view.get_dof_limits()).tolist(),
                "closure_enabled": closure,
                "actuator_cache": {name: {"stiffness": cpu(a.stiffness).tolist(), "damping": cpu(a.damping).tolist()}
                                   for name, a in self.robot.actuators.items()}}

    def configure_direct(self, torques):
        self.drive_before = self.drive_identity()
        self._probe_torques = torques.clone()
        if args.drive_mode != "formal":
            self.robot.write_joint_stiffness_to_sim(0.0, joint_ids=self._controlled_joint_ids)
            self.robot.write_joint_damping_to_sim(0.0, joint_ids=self._controlled_joint_ids)
            for name in ("legs", "wheels"):
                self.robot.actuators[name].stiffness.zero_()
                self.robot.actuators[name].damping.zero_()
        self.drive_after = self.drive_identity()
        expected = json.loads(json.dumps(self.drive_before))
        if args.drive_mode != "formal":
            for field in ("stiffness", "damping"):
                for row in expected[field]:
                    for joint in self._controlled_joint_ids:
                        row[joint] = 0.0
                for name in ("legs", "wheels"):
                    expected["actuator_cache"][name][field] = np.zeros_like(expected["actuator_cache"][name][field]).tolist()
        assert self.drive_after == expected, "Unexpected change outside active PD gains"
        self._assert_passive_actuator_contract()

    def _apply_action(self):
        if args.drive_mode == "formal":
            return super()._apply_action()
        native = controlled_joint_feedback_usd_to_control(self._probe_torques)
        self.robot.set_joint_effort_target(native, joint_ids=self._controlled_joint_ids)
'''

text = (OLD / "probe_isaac.py").read_text()
text = replace(text, '"""Read-only production physics probe; all new files stay in this artifact directory."""',
               '"""Free-base same-external-torque diagnostic; production source is unchanged."""')
text = replace(text, 'AppLauncher.add_app_launcher_args(parser)', extra_args + 'AppLauncher.add_app_launcher_args(parser)')
text = replace(text, 'from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG',
               'from wheelleg_dreamwaq.assets.wheelleg import WHEELLEG_CFG\nfrom wheelleg_dreamwaq.schemas.action import controlled_joint_feedback_usd_to_control')
text = replace(text, '\n\ndef rotated', '\n' + isaac_methods + '\n\ndef rotated')
text = replace(text, '    before = protected()', '''    assert args.no_ground and args.case == "shared_first_action_hold" and args.control_ticks == 1
    torque_record = json.loads(args.torques_from.read_text(encoding="utf-8"))
    torque_key = "zero_torque_control_nm" if args.drive_mode == "direct_zero" else "shared_first_torque_control_nm"
    torques = torch.tensor(torque_record[torque_key], dtype=torch.float32, device=args.device)
    assert torques.shape == (8, 6) and torch.isfinite(torques).all()
    assert torch.all(torques.abs() <= torques.new_tensor(torque_record["effort_limits_nm"]))
    before = protected()''')
text = replace(text, '    samples = []', '''    submitted_forces = []
    original_force_setter = view.set_dof_actuation_forces

    def observed_force_setter(forces, *positional, **keywords):
        submitted_forces.append(cpu(forces).tolist())
        return original_force_setter(forces, *positional, **keywords)

    view.set_dof_actuation_forces = observed_force_setter
    samples = []''')
text = replace(text, '                "joint_velocity": cpu(view.get_dof_velocities()).tolist()}', '''                "joint_velocity": cpu(view.get_dof_velocities()).tolist(),
                "external_force_argument_native_nm": submitted_forces[-1] if submitted_forces else None}''')
text = replace(text, '        samples.append(snapshot("reset_returned_before_first_action"))', '''        env.configure_direct(torques)
        assert float(env.sim.current_time) == start
        samples.append(snapshot("reset_returned_before_first_action"))''')
text = replace(text, '    env.sim.step = original_step', '''    env.sim.step = original_step
    view.set_dof_actuation_forces = original_force_setter
    controlled = env._controlled_joint_ids
    expected_torque = cpu(controlled_joint_feedback_usd_to_control(torques))
    for row in samples:
        if row["phase"] != "post_physics_step":
            continue
        force = np.asarray(row["external_force_argument_native_nm"])
        actual = force[:, controlled]
        if args.drive_mode == "formal":
            assert np.max(np.abs(force)) == 0.0
        else:
            assert np.array_equal(actual, expected_torque)
            assert np.max(np.abs(np.delete(force, controlled, axis=1))) == 0.0
        assert np.max(np.abs(np.asarray(row["joint_velocity"])[:, controlled])) < 45.0
    assert env.drive_identity() == env.drive_after''')
text = replace(text, '"schema_version": "GroundOffIsaacProbeV1"', '"schema_version": "SameTorqueIsaacProbeV1"')
text = replace(text, '              "production_reset_and_step_inherited": True,', '''              "production_reset_and_step_inherited": True, "production_apply_action_inherited": args.drive_mode == "formal",
              "drive_mode": args.drive_mode, "torque_input_sha256": digest(args.torques_from),
              "drive_before": env.drive_before, "drive_after": env.drive_after,
              "active_joint_ids": list(controlled), "submitted_force_arguments_native_nm": submitted_forces,
              "force_semantics": "External force tensor actually passed to PhysX setter; not original implicit-drive torque",
              "free_base": not cfg.robot_cfg.spawn.articulation_props.fix_root_link,''')
(ROOT / "probe_isaac.py").write_text(text, encoding="utf-8")

text = (OLD / "probe_mujoco.py").read_text()
text = replace(text, 'args = parser.parse_args()', extra_args + 'parser.add_argument("--initial-state-from", type=Path)\nargs = parser.parse_args()')
text = replace(text, 'from wheelleg_mujoco.runner import WheelLegMujocoRuntime',
               'from wheelleg_mujoco.runner import WheelLegMujocoRuntime\nfrom wheelleg_mujoco.model_map import CONTROLLED_JOINT_SIGNS')
text = replace(text, 'from wheelleg_mujoco.model_semantics import compiled_model_semantics, stable_hash',
               'from wheelleg_mujoco.model_semantics import compiled_model_semantics, stable_hash\nfrom wheelleg_mujoco.observation import collect_kinematic_state')
text = replace(text, '    assert args.control_ticks > 0', '''    assert args.control_ticks == 1 and args.no_ground and args.case == "shared_first_action_hold"
    torque_record = json.loads(args.torques_from.read_text())
    torque_key = "zero_torque_control_nm" if args.drive_mode == "direct_zero" else "shared_first_torque_control_nm"
    torques = np.asarray(torque_record[torque_key], dtype=np.float64)
    assert torques.shape == (8, 6) and np.isfinite(torques).all()
    assert np.all(np.abs(torques) <= np.asarray(torque_record["effort_limits_nm"]))''')
text = replace(text, '    shadow = mujoco.MjData(model)', '''    aids = runtime.model_map.actuator_ids
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

    shadow = mujoco.MjData(model)''')
text = replace(text, '"joint_velocity": data.qvel[6:].tolist()}', '''"joint_velocity": data.qvel[6:].tolist(),
                "external_ctrl_native_nm": data.ctrl[aids].tolist(),
                "actuator_force_native_nm": data.actuator_force[aids].tolist(),
                "generalized_actuator_force_native_nm": data.qfrc_actuator[dofs].tolist(),
                "angular_control_fresh_jacobian_rad_s": collect_kinematic_state(model, shadow, runtime.model_map, contract).angular_velocity_control.tolist()}''')
text = text.replace('                runtime.reset(np.asarray(command))', '                select_torque(environment_index)\n                runtime.reset(np.asarray(command))')
text = replace(text, '                assert step_index == 0 and data.time == 0.0',
               '                align_initial_state(environment_index)\n                assert step_index == 0 and data.time == 0.0')
text = replace(text, '            runtime.reset(np.asarray(command))\n            for _ in range(args.control_ticks):',
               '            select_torque(environment_index)\n            runtime.reset(np.asarray(command))\n            align_initial_state(environment_index)\n            for _ in range(args.control_ticks):')
text = replace(text, '    assert max(replay_errors) == 0.0', '''    assert max(replay_errors) == 0.0
    for row in samples:
        if row["physics_step"] == 0:
            continue
        if args.drive_mode != "formal":
            expected = torques[row["environment_index"]] * CONTROLLED_JOINT_SIGNS
            for field in ("external_ctrl_native_nm", "actuator_force_native_nm", "generalized_actuator_force_native_nm"):
                assert np.array_equal(np.asarray(row[field]), expected), (field, row["physics_step"])
        assert np.max(np.abs(np.asarray(row["joint_velocity"])[dofs - 6])) < 45.0''')
text = replace(text, '"schema_version": "GroundOffMujocoProbeV1"', '"schema_version": "SameTorqueMujocoProbeV1"')
text = replace(text, '              "test_case": args.case,', '''              "test_case": args.case, "drive_mode": args.drive_mode,
              "torque_input_sha256": digest(args.torques_from), "free_base": True,
              "matching_initial_state_sha256": None if args.initial_state_from is None else digest(args.initial_state_from),
              "controlled_dof_ids": dofs.tolist(), "joint_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)],
              "dof_damping": model.dof_damping.tolist(), "dof_armature": model.dof_armature.tolist(),
              "equality_active": model.eq_active0.tolist(), "active_ctrl_ranges": model.actuator_ctrlrange[aids].tolist(),''')
(ROOT / "probe_mujoco.py").write_text(text, encoding="utf-8")

for name in ("probe_isaac.py", "probe_mujoco.py"):
    before = (OLD / name).read_text().splitlines(keepends=True)
    after = (ROOT / name).read_text().splitlines(keepends=True)
    diff = ''.join(difflib.unified_diff(before, after, fromfile=f"old/{name}", tofile=f"new/{name}"))
    (ROOT / (name + ".diff")).write_text(diff, encoding="utf-8")
    print(name, "added", sum(s.startswith('+') and not s.startswith('+++') for s in diff.splitlines()),
          "removed", sum(s.startswith('-') and not s.startswith('---') for s in diff.splitlines()))
