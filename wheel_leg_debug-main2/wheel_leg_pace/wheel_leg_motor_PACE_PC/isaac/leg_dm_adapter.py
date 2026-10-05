from __future__ import annotations

import argparse
import json
import pathlib
from typing import Mapping, Union

from fit.model_manifest import load_model_manifest, verify_against_firmware_manifest
from pace_raw.manifest import load_firmware_manifest


def load_leg_dm_contract(value: Union[pathlib.Path, str, Mapping[str, object]]) -> dict:
    model = load_model_manifest(value)
    verify_against_firmware_manifest(model, load_firmware_manifest())
    motors = model["motors"][:4]
    if any(motor["model"].get("type") != "pace_dm_pd_v1" for motor in motors):
        raise ValueError("all four leg motors must use pace_dm_pd_v1")
    delays = {int(motor["model"]["delay_steps"]) for motor in motors}
    if len(delays) != 1:
        raise ValueError("the generic PACE leg actuator requires one global delay")
    delay_steps = delays.pop()
    return {
        "schema_version": model["schema_version"],
        "model_maturity": model["model_maturity"],
        "canonical_indices": [motor["canonical_index"] for motor in motors],
        "canonical_names": [motor["name"] for motor in motors],
        "joint_order": [motor["isaac_joint"] for motor in motors],
        "sample_period_s": model["sample_period_s"],
        "armature": [motor["model"]["armature"] for motor in motors],
        "viscous_friction": [motor["model"]["viscous_friction"] for motor in motors],
        "static_friction": [motor["model"]["static_friction"] for motor in motors],
        "dynamic_friction": [motor["model"]["dynamic_friction"] for motor in motors],
        "encoder_bias_rad": [motor["model"]["encoder_bias_rad"] for motor in motors],
        "delay_steps": delay_steps,
        "stiffness": [motor["model"]["controller"]["kp"] for motor in motors],
        "damping": [motor["model"]["controller"]["kd"] for motor in motors],
        "effort_limit_nm": [motor["model"]["effort_limit_nm"] for motor in motors],
        "position_sign": [motor["position_sign"] for motor in motors],
        "velocity_sign": [motor["velocity_sign"] for motor in motors],
        "torque_sign": [motor["torque_sign"] for motor in motors],
        "position_zero_rad": [motor["position_zero_rad"] for motor in motors],
    }


def pace_dc_motor_cfg_kwargs(value: Union[pathlib.Path, str, Mapping[str, object]]) -> dict:
    contract = load_leg_dm_contract(value)
    joints = contract["joint_order"]
    return {
        "joint_names_expr": list(joints),
        "saturation_effort": max(contract["effort_limit_nm"]),
        "effort_limit": max(contract["effort_limit_nm"]),
        "velocity_limit": 45.0,
        "stiffness": dict(zip(joints, contract["stiffness"])),
        "damping": dict(zip(joints, contract["damping"])),
        "encoder_bias": dict(zip(joints, contract["encoder_bias_rad"])),
        "friction": dict(zip(joints, contract["static_friction"])),
        "dynamic_friction": dict(zip(joints, contract["dynamic_friction"])),
        "viscous_friction": dict(zip(joints, contract["viscous_friction"])),
        "max_delay": contract["delay_steps"],
    }


def apply_joint_properties(articulation, joint_ids, value) -> None:
    contract = load_leg_dm_contract(value)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Isaac articulation updates require PyTorch") from error
    device = articulation.device
    armature = torch.tensor(contract["armature"], device=device).unsqueeze(0)
    viscous = torch.tensor(contract["viscous_friction"], device=device).unsqueeze(0)
    friction = torch.tensor(contract["static_friction"], device=device).unsqueeze(0)
    dynamic = torch.tensor(contract["dynamic_friction"], device=device).unsqueeze(0)
    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids)
    articulation.write_joint_viscous_friction_coefficient_to_sim(viscous, joint_ids=joint_ids)
    articulation.write_joint_dynamic_friction_coefficient_to_sim(
        torch.zeros_like(dynamic), joint_ids=joint_ids
    )
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids)
    articulation.write_joint_dynamic_friction_coefficient_to_sim(dynamic, joint_ids=joint_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the Isaac leg actuator contract")
    parser.add_argument("manifest", type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(load_leg_dm_contract(args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
