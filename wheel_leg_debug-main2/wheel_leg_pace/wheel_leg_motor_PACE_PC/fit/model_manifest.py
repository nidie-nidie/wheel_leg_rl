from __future__ import annotations

import copy
import json
import math
import pathlib
from datetime import datetime, timezone
from typing import Dict, Mapping, MutableMapping, Union

from pace_raw.manifest import MotorManifest
from pace_raw.normalize import NormalizedDataset


MODEL_SCHEMA_VERSION = "wheel_leg_motor_pace/actuator_model/v1"
CANONICAL_ORDER = (
    "L_front",
    "L_rear",
    "R_rear",
    "R_front",
    "L_wheel",
    "R_wheel",
)


def maturity_from_session_type(session_type: int) -> str:
    if session_type == 1:
        return "provisional"
    if session_type == 2:
        return "final"
    raise ValueError(f"unsupported source session type {session_type}")


def create_model_manifest(dataset: NormalizedDataset) -> Dict[str, object]:
    header = dataset.decoded.header
    if header is None:
        raise ValueError("a fitted model requires a decoded session header")
    dataset.manifest.validate()
    motors = []
    for entry in dataset.manifest.motors:
        motors.append(
            {
                "canonical_index": entry.canonical_index,
                "name": entry.motor_name,
                "motor_type": entry.motor_type,
                "actuator_family": entry.actuator_family,
                "mujoco_joint": entry.mujoco_joint,
                "isaac_joint": entry.isaac_joint,
                "position_sign": entry.position_sign,
                "velocity_sign": entry.velocity_sign,
                "torque_sign": entry.torque_sign,
                "position_zero_rad": entry.position_zero_rad,
                "gear_ratio": entry.gear_ratio,
                "model": None,
                "fit": None,
                "validation": None,
            }
        )
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_maturity": maturity_from_session_type(header.session_type),
        "sample_period_s": 1.0 / float(header.sample_rate_hz),
        "canonical_order": list(CANONICAL_ORDER),
        "motor_order_version": header.motor_order_version,
        "motor_manifest_hash": header.scale_manifest_hash,
        "source_session": {
            "session_type": header.session_type,
            "session_id": header.session_id,
            "experiment_config_hash": header.experiment_config_hash,
            "firmware_build_id": header.firmware_build_id,
            "robot_variant": header.robot_variant,
            "sample_rate_hz": header.sample_rate_hz,
            "footer_valid": bool(
                dataset.decoded.footer
                and not dataset.decoded.footer.overflow
                and dataset.decoded.footer.statistics_complete
                and dataset.decoded.footer.dropped_frames == 0
            ),
            "decode_issue_counts": dataset.decoded.summary()["issues"],
        },
        "fit_backend": {},
        "motors": motors,
        "aggregate_metrics": {},
    }


def _model_dict(value: Union[pathlib.Path, str, Mapping[str, object]]) -> Dict[str, object]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    path = pathlib.Path(value)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_model_manifest(model: Mapping[str, object], require_complete: bool = True) -> None:
    if model.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported fitted-model schema")
    if model.get("model_maturity") not in ("provisional", "final"):
        raise ValueError("model_maturity must be provisional or final")
    if tuple(model.get("canonical_order", ())) != CANONICAL_ORDER:
        raise ValueError("fitted model canonical order is invalid")
    if int(model.get("motor_order_version", -1)) != 1:
        raise ValueError("unsupported fitted-model motor order version")
    if float(model.get("sample_period_s", 0.0)) <= 0.0:
        raise ValueError("sample_period_s must be positive")
    source = model.get("source_session")
    if not isinstance(source, Mapping):
        raise ValueError("source_session must be an object")
    expected_maturity = maturity_from_session_type(int(source.get("session_type", 0)))
    if model.get("model_maturity") != expected_maturity:
        raise ValueError("model maturity disagrees with its source session type")
    if require_complete and not bool(source.get("footer_valid", False)):
        raise ValueError("a complete fitted model requires a valid source footer")

    motors = model.get("motors")
    if not isinstance(motors, list) or len(motors) != 6:
        raise ValueError("fitted model must contain exactly six motors")
    for index, (entry, expected_name) in enumerate(zip(motors, CANONICAL_ORDER)):
        if not isinstance(entry, Mapping):
            raise ValueError("each fitted motor entry must be an object")
        if entry.get("canonical_index") != index or entry.get("name") != expected_name:
            raise ValueError("fitted motor entries are reordered")
        expected_family = "leg_dm" if index < 4 else "wheel_lk"
        if entry.get("actuator_family") != expected_family:
            raise ValueError(f"invalid actuator family for {expected_name}")
        if int(entry.get("position_sign", 0)) not in (-1, 1):
            raise ValueError(f"invalid position sign for {expected_name}")
        if int(entry.get("velocity_sign", 0)) not in (-1, 1):
            raise ValueError(f"invalid velocity sign for {expected_name}")
        if int(entry.get("torque_sign", 0)) not in (-1, 1):
            raise ValueError(f"invalid torque sign for {expected_name}")
        if float(entry.get("gear_ratio", 0.0)) <= 0.0:
            raise ValueError(f"invalid gear ratio for {expected_name}")
        if require_complete and not isinstance(entry.get("model"), Mapping):
            raise ValueError(f"missing actuator model for {expected_name}")
        if require_complete:
            actuator = entry["model"]
            if index < 4:
                if actuator.get("type") != "pace_dm_pd_v1":
                    raise ValueError(f"invalid DM model type for {expected_name}")
                required = (
                    "armature",
                    "viscous_friction",
                    "static_friction",
                    "dynamic_friction",
                    "encoder_bias_rad",
                    "delay_steps",
                    "controller",
                    "effort_limit_nm",
                )
            else:
                if actuator.get("type") != "lk9025_dual_mode_v1":
                    raise ValueError(f"invalid LK model type for {expected_name}")
                required = ("torque_mode", "velocity_mode", "effort_limit_nm")
            if any(key not in actuator for key in required):
                raise ValueError(f"incomplete actuator model for {expected_name}")
            if not math.isfinite(float(actuator["effort_limit_nm"])) or float(
                actuator["effort_limit_nm"]
            ) <= 0.0:
                raise ValueError(f"invalid effort limit for {expected_name}")
            if model.get("model_maturity") == "final":
                validation = entry.get("validation")
                if not isinstance(validation, Mapping) or validation.get("available") is False:
                    raise ValueError(f"final model lacks held-out validation for {expected_name}")


def load_model_manifest(
    value: Union[pathlib.Path, str, Mapping[str, object]],
    require_complete: bool = True,
) -> Dict[str, object]:
    model = _model_dict(value)
    validate_model_manifest(model, require_complete=require_complete)
    return model


def write_model_manifest(
    model: Mapping[str, object],
    path: pathlib.Path,
    require_complete: bool = True,
) -> None:
    validate_model_manifest(model, require_complete=require_complete)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replace_motor_result(
    model: MutableMapping[str, object],
    canonical_index: int,
    actuator_model: Mapping[str, object],
    fit: Mapping[str, object],
    validation: Mapping[str, object],
) -> None:
    motors = model.get("motors")
    if not isinstance(motors, list) or canonical_index not in range(len(motors)):
        raise ValueError("canonical motor index is outside the fitted model")
    motors[canonical_index]["model"] = copy.deepcopy(dict(actuator_model))
    motors[canonical_index]["fit"] = copy.deepcopy(dict(fit))
    motors[canonical_index]["validation"] = copy.deepcopy(dict(validation))


def verify_against_firmware_manifest(
    model: Mapping[str, object], firmware: MotorManifest
) -> None:
    validate_model_manifest(model, require_complete=False)
    firmware.validate()
    if int(model["motor_manifest_hash"]) != firmware.manifest_hash:
        raise ValueError("fitted model hash does not match the firmware manifest")
    for fitted, source in zip(model["motors"], firmware.motors):
        checks = {
            "canonical_index": source.canonical_index,
            "name": source.motor_name,
            "motor_type": source.motor_type,
            "actuator_family": source.actuator_family,
            "mujoco_joint": source.mujoco_joint,
            "isaac_joint": source.isaac_joint,
            "position_sign": source.position_sign,
            "velocity_sign": source.velocity_sign,
            "torque_sign": source.torque_sign,
            "position_zero_rad": source.position_zero_rad,
            "gear_ratio": source.gear_ratio,
        }
        for key, expected in checks.items():
            if fitted.get(key) != expected:
                raise ValueError(f"fitted model {key} differs for {source.motor_name}")
