"""Create a source-bound acquisition plan without importing Isaac Lab or torch."""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

PACE = Path(__file__).resolve().parents[1]
DREAMWAQ = PACE.parent
PC = PACE / "wheel_leg_motor_PACE_PC"
sys.path.insert(0, str(PC))
from pace_raw.manifest import load_firmware_manifest


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return tree, result


def actuator_baseline(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == name and isinstance(value, ast.Call):
                    return {kw.arg: ast.literal_eval(kw.value) for kw in value.keywords
                            if kw.arg in ("stiffness", "damping", "effort_limit_sim", "armature")}
    raise ValueError(f"Missing DreamWaQ {name} actuator baseline")


def prepare(binary=None, policy_manifest=None):
    source = DREAMWAQ / "training/wheelleg_dreamwaq"
    paths = {"action": source / "schemas/action.py", "physics": source / "schemas/physics.py",
             "control": source / "tasks/direct/wheelleg_flat/control.py",
             "asset": source / "assets/wheelleg.py"}
    action_tree, action = literals(paths["action"])
    physics_tree, physics = literals(paths["physics"])
    control_tree, _ = literals(paths["control"])
    asset_tree, _ = literals(paths["asset"])
    limits = {}
    for node in ast.walk(control_tree):
        if isinstance(node, ast.ClassDef) and node.name == "ControlLimits":
            limits = {child.target.id: ast.literal_eval(child.value) for child in node.body
                      if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)}
    manifest = load_firmware_manifest()
    pace_joints = [motor.isaac_joint for motor in manifest.motors]
    policy_joints = list(action["CANONICAL_JOINT_ORDER"])
    if set(pace_joints) != set(policy_joints) or len(set(pace_joints)) != 6:
        raise ValueError("PACE and DreamWaQ joint identities do not match uniquely")
    result = {
        "plan_version": "pace/acquisition-plan/v2",
        "fixture": "mounted_body_fixed_wheels_suspended",
        "session_type": "commissioning",
        "direction_source": "user_confirmed_existing_code",
        "zero_calibration_verified": False,
        "mechanical_range_verified": False,
        "motor_manifest": manifest.to_dict(),
        "motor_manifest_sha256": sha256(manifest.source_path),
        "dreamwaq_sources": {str(path.relative_to(DREAMWAQ)): sha256(path) for path in paths.values()},
        "dreamwaq_baseline": {
            "joint_order": policy_joints,
            "policy_from_pace_indices": [pace_joints.index(joint) for joint in policy_joints],
            "wheel_joint_sign_usd": list(action["WHEEL_JOINT_SIGN_USD"]),
            "sim_dt_s": physics["SIM_DT_S"], "control_dt_s": physics["CONTROL_DT_S"],
            "decimation": physics["DECIMATION"], "control_limits": limits,
            "legs": actuator_baseline(asset_tree, "legs"),
            "wheels": actuator_baseline(asset_tree, "wheels"),
        },
        "collection_contract": {"action_rate_hz": 50, "command_hold": "zero_order_hold",
            "sample_rate_hz": 250, "control_rate_hz": 500,
            "dm_kp": 20.0, "dm_kd": 0.6, "fit_and_independent_validation": True,
            "interpretation": "mounted closed-chain load response; effective inertia is not a motor-only armature"},
        "baseline_gain_match": False,
        "baseline_gain_note": "Commissioning uses DM Kp20/Kd0.6; current DreamWaQ baseline Kp120/Kd4. No direct gain-equivalence claim.",
    }
    if binary:
        binary = Path(binary).resolve()
        result["firmware_binary"] = {"path": str(binary), "sha256": sha256(binary)}
    if policy_manifest:
        policy_manifest = Path(policy_manifest).resolve()
        result["exported_policy_manifest"] = {"path": str(policy_manifest), "sha256": sha256(policy_manifest),
                                               "snapshot": json.loads(policy_manifest.read_text(encoding="utf-8"))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--firmware-binary", type=Path)
    parser.add_argument("--policy-manifest", type=Path)
    args = parser.parse_args()
    result = prepare(args.firmware_binary, args.policy_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Prepared commissioning plan: {args.output.resolve()}")


if __name__ == "__main__":
    main()
