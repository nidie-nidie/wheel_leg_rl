from __future__ import annotations

import csv
import dataclasses
import pathlib
import re
from typing import Dict, List


EXPECTED_SYMBOLS = [
    "PACE_MOTOR_L_FRONT",
    "PACE_MOTOR_L_REAR",
    "PACE_MOTOR_R_REAR",
    "PACE_MOTOR_R_FRONT",
    "PACE_MOTOR_L_WHEEL",
    "PACE_MOTOR_R_WHEEL",
]


@dataclasses.dataclass(frozen=True)
class MotorManifestEntry:
    canonical_index: int
    symbol: str
    motor_name: str
    motor_type: str
    hardware_slot: int
    device_id: int
    command_can_id: int
    feedback_can_id: int
    mujoco_joint: str
    isaac_joint: str
    position_sign: int
    velocity_sign: int
    torque_sign: int
    position_zero_rad: float
    gear_ratio: float
    actuator_family: str
    calibration_verified: bool


@dataclasses.dataclass(frozen=True)
class MotorManifest:
    version: int
    manifest_hash: int
    motors: List[MotorManifestEntry]
    source_path: pathlib.Path

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported motor order version {self.version}")
        if len(self.motors) != 6:
            raise ValueError("manifest must contain exactly six motors")
        if [motor.canonical_index for motor in self.motors] != list(range(6)):
            raise ValueError("canonical indices must be contiguous 0..5")
        if [motor.symbol for motor in self.motors] != EXPECTED_SYMBOLS:
            raise ValueError("manifest order does not match the canonical order")
        if len({motor.motor_name for motor in self.motors}) != 6:
            raise ValueError("motor names must be unique")
        if len({motor.feedback_can_id for motor in self.motors}) != 6:
            raise ValueError("feedback CAN identifiers must be unique")
        if [motor.motor_type for motor in self.motors[:4]] != ["DM8009"] * 4:
            raise ValueError("canonical indices 0..3 must be DM8009")
        if [motor.motor_type for motor in self.motors[4:]] != ["LK9025"] * 2:
            raise ValueError("canonical indices 4..5 must be LK9025")
        for motor in self.motors:
            if motor.position_sign not in (-1, 1):
                raise ValueError(f"invalid position sign for {motor.motor_name}")
            if motor.velocity_sign not in (-1, 1):
                raise ValueError(f"invalid velocity sign for {motor.motor_name}")
            if motor.torque_sign not in (-1, 1):
                raise ValueError(f"invalid torque sign for {motor.motor_name}")
            if motor.gear_ratio <= 0.0:
                raise ValueError(f"invalid gear ratio for {motor.motor_name}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "manifest_hash": self.manifest_hash,
            "source_path": str(self.source_path),
            "motors": [dataclasses.asdict(motor) for motor in self.motors],
        }


def _parse_integer(token: str) -> int:
    return int(token.rstrip("UuLl"), 0)


def _parse_float(token: str) -> float:
    return float(token.rstrip("fF"))


def load_firmware_manifest(path: pathlib.Path = None) -> MotorManifest:
    if path is None:
        workspace = pathlib.Path(__file__).resolve().parents[2]
        path = workspace / "wheel_leg_motor_PACE" / "Config" / "pace_motor_manifest.h"
    path = pathlib.Path(path)
    text = path.read_text(encoding="utf-8")
    hash_match = re.search(r"#define\s+PACE_MOTOR_MANIFEST_HASH\s+(0x[0-9A-Fa-f]+)UL", text)
    version_path = path.parents[1] / "Protocol" / "Inc" / "pace_motor_order.h"
    version_text = version_path.read_text(encoding="utf-8")
    version_match = re.search(r"#define\s+PACE_MOTOR_ORDER_VERSION\s+(\d+)U", version_text)
    if not hash_match or not version_match:
        raise ValueError("manifest version/hash definition is missing")

    type_map = {
        "PACE_MOTOR_TYPE_DM8009": "DM8009",
        "PACE_MOTOR_TYPE_LK9025": "LK9025",
    }
    family_map = {
        "PACE_ACTUATOR_LEG_DM": "leg_dm",
        "PACE_ACTUATOR_WHEEL_LK": "wheel_lk",
    }
    motors = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("X("):
            continue
        if stripped.endswith(chr(92)):
            stripped = stripped[:-1].rstrip()
        if not stripped.endswith(")"):
            raise ValueError(f"malformed manifest row: {line}")
        values = next(csv.reader([stripped[2:-1]], skipinitialspace=True))
        if len(values) != 16:
            raise ValueError(f"manifest row has {len(values)} fields, expected 16")
        index = len(motors)
        motors.append(
            MotorManifestEntry(
                canonical_index=index,
                symbol=values[0],
                motor_name=values[1],
                motor_type=type_map[values[2]],
                hardware_slot=_parse_integer(values[3]),
                device_id=_parse_integer(values[4]),
                command_can_id=_parse_integer(values[5]),
                feedback_can_id=_parse_integer(values[6]),
                mujoco_joint=values[7],
                isaac_joint=values[8],
                position_sign=int(values[9]),
                velocity_sign=int(values[10]),
                torque_sign=int(values[11]),
                position_zero_rad=_parse_float(values[12]),
                gear_ratio=_parse_float(values[13]),
                actuator_family=family_map[values[14]],
                calibration_verified=values[15].lower() == "true",
            )
        )
    manifest = MotorManifest(
        version=int(version_match.group(1)),
        manifest_hash=int(hash_match.group(1), 16),
        motors=motors,
        source_path=path,
    )
    manifest.validate()
    return manifest
