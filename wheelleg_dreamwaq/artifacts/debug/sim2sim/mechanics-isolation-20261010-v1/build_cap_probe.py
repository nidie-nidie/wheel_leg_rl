"""One-property diagnostic supplement; original seven probes stay frozen."""
import difflib
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def change(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


original = s = (ROOT / "probe_isaac.py").read_text(encoding="utf-8")
s = change(s, 'AppLauncher.add_app_launcher_args(parser)',
           'parser.add_argument("--angular-cap-deg-s", type=float, choices=(100., 10000.), required=True)\nAppLauncher.add_app_launcher_args(parser)')
s = change(s, '    env = ContactProbeEnv(cfg, closed_chain_reset_cache=cache)', '''    assert args.mechanics_case in ("baseline", "closure_off")
    rigid_before = cfg.robot_cfg.spawn.rigid_props.to_dict()
    assert rigid_before["max_angular_velocity"] == 100.0
    cfg.robot_cfg = cfg.robot_cfg.replace(spawn=cfg.robot_cfg.spawn.replace(
        rigid_props=cfg.robot_cfg.spawn.rigid_props.replace(max_angular_velocity=args.angular_cap_deg_s)))
    rigid_after = cfg.robot_cfg.spawn.rigid_props.to_dict()
    expected_rigid = dict(rigid_before)
    expected_rigid["max_angular_velocity"] = args.angular_cap_deg_s
    assert rigid_after == expected_rigid
    env = ContactProbeEnv(cfg, closed_chain_reset_cache=cache)''')
s = change(s, '    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())', '''    rigid_caps = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs")):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            value = PhysxSchema.PhysxRigidBodyAPI(prim).GetMaxAngularVelocityAttr().Get()
            assert value == args.angular_cap_deg_s, (str(prim.GetPath()), value)
            rigid_caps[str(prim.GetPath())] = value
    assert len(rigid_caps) == 27 * 8
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())''')
s = change(s, '                "root_pose_xyzw": roots.tolist(), "root_velocity_world": cpu(view.get_root_velocities()).tolist(),',
           '                "body_velocity_world": cpu(view.get_link_velocities()).tolist(),\n'
           '                "root_pose_xyzw": roots.tolist(), "root_velocity_world": cpu(view.get_root_velocities()).tolist(),')
s = change(s, '"schema_version": "MechanicsIsolationIsaacV1"',
           '"schema_version": "AngularCapIsaacV1", "angular_cap_deg_s": args.angular_cap_deg_s,\n'
           '              "rigid_properties_before": rigid_before, "rigid_properties_after": rigid_after,\n'
           '              "runtime_usd_rigid_caps_deg_s": rigid_caps, "body_names": list(env.robot.body_names)')
(ROOT / "probe_isaac_cap.py").write_text(s, encoding="utf-8")
(ROOT / "angular-cap-changes.diff").write_text("".join(difflib.unified_diff(
    original.splitlines(keepends=True), s.splitlines(keepends=True),
    fromfile=str(ROOT / "probe_isaac.py"), tofile=str(ROOT / "probe_isaac_cap.py"))), encoding="utf-8")
print("CAP_PROBE_GENERATED", hashlib.sha256(s.encode("utf-8")).hexdigest().upper())
