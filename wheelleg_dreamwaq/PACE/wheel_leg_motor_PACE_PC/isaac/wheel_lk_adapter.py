from __future__ import annotations

import argparse
import json
import pathlib
from typing import Mapping, Union

from fit.model_manifest import load_model_manifest, verify_against_firmware_manifest
from pace_raw.manifest import load_firmware_manifest


def load_wheel_lk_contract(value: Union[pathlib.Path, str, Mapping[str, object]]) -> dict:
    model = load_model_manifest(value)
    verify_against_firmware_manifest(model, load_firmware_manifest())
    motors = model["motors"][4:]
    if any(motor["model"].get("type") != "lk9025_dual_mode_v1" for motor in motors):
        raise ValueError("both wheel motors must use lk9025_dual_mode_v1")
    return {
        "schema_version": model["schema_version"],
        "model_maturity": model["model_maturity"],
        "canonical_indices": [motor["canonical_index"] for motor in motors],
        "canonical_names": [motor["name"] for motor in motors],
        "joint_order": [motor["isaac_joint"] for motor in motors],
        "sample_period_s": model["sample_period_s"],
        "torque_mode": [motor["model"]["torque_mode"] for motor in motors],
        "velocity_mode": [motor["model"]["velocity_mode"] for motor in motors],
        "effort_limit_nm": [motor["model"]["effort_limit_nm"] for motor in motors],
        "position_sign": [motor["position_sign"] for motor in motors],
        "velocity_sign": [motor["velocity_sign"] for motor in motors],
        "torque_sign": [motor["torque_sign"] for motor in motors],
        "position_zero_rad": [motor["position_zero_rad"] for motor in motors],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the Isaac wheel actuator contract")
    parser.add_argument("manifest", type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(load_wheel_lk_contract(args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
