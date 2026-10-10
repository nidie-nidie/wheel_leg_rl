from __future__ import annotations

import csv
import dataclasses
import math
import pathlib
from typing import Dict, List, Sequence

import numpy as np

from .decoder import DecodedSession
from .manifest import MotorManifest, load_firmware_manifest


DM_POSITION_MIN = -12.5
DM_POSITION_MAX = 12.5
DM_VELOCITY_MIN = -45.0
DM_VELOCITY_MAX = 45.0
DM_TORQUE_MIN = -54.0
DM_TORQUE_MAX = 54.0
LK_CURRENT_A_PER_COMMAND = 0.008056640625
LK_TORQUE_CONSTANT = 0.32
DEG_TO_RAD = math.pi / 180.0
COMMAND_DM_MIT = 1
COMMAND_LK_TORQUE = 5
COMMAND_LK_VELOCITY = 6


def _uint_to_float(raw: int, minimum: float, maximum: float, bits: int) -> float:
    return raw * (maximum - minimum) / ((1 << bits) - 1) + minimum


def _canonical_position(value: float, sign: int, zero: float, ratio: float) -> float:
    return sign * (value - zero) / ratio


def _canonical_velocity(value: float, sign: int, ratio: float) -> float:
    return sign * value / ratio


def _canonical_torque(value: float, sign: int, ratio: float) -> float:
    return sign * value * ratio


@dataclasses.dataclass
class NormalizedDataset:
    decoded: DecodedSession
    manifest: MotorManifest
    rows: List[Dict[str, object]]

    def columns(self) -> Dict[str, np.ndarray]:
        if not self.rows:
            return {}
        result: Dict[str, np.ndarray] = {}
        for key in self.rows[0]:
            values = [row[key] for row in self.rows]
            if all(isinstance(value, (bool, np.bool_)) for value in values):
                result[key] = np.asarray(values, dtype=np.bool_)
            elif all(isinstance(value, (int, np.integer)) for value in values):
                result[key] = np.asarray(values, dtype=np.int64)
            elif all(isinstance(value, (int, float, np.number)) for value in values):
                result[key] = np.asarray(values, dtype=np.float64)
            else:
                result[key] = np.asarray(["" if value is None else str(value) for value in values])
        return result

    def export_csv(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def export_npz(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **self.columns())

    def export_pt(self, path: pathlib.Path) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required only for PT export; CSV/NPZ remain available") from error
        tensors = {}
        metadata = {}
        for key, values in self.columns().items():
            if values.dtype.kind in "biuf":
                tensors[key] = torch.from_numpy(values)
            else:
                metadata[key] = values.tolist()
        torch.save({"tensors": tensors, "metadata": metadata, "manifest": self.manifest.to_dict()}, path)

    def long_view(self, motor_indices: Sequence[int]) -> List[Dict[str, object]]:
        output = []
        for row in self.rows:
            for index in motor_indices:
                motor = self.manifest.motors[index]
                prefix = f"m{index}_"
                long_row = {
                    "sample_time_us": row["sample_time_us"],
                    "time_s": row["time_s"],
                    "stage_id": row["stage_id"],
                    "config_seq": row["config_seq"],
                    "role": row["role"],
                    "sample_fit_eligible": row["fit_eligible"],
                    "canonical_index": index,
                    "motor_name": motor.motor_name,
                    "motor_type": motor.motor_type,
                }
                for key, value in row.items():
                    if key.startswith(prefix):
                        long_row[key[len(prefix) :]] = value
                output.append(long_row)
        return output

    @property
    def leg_dm_view(self) -> List[Dict[str, object]]:
        return self.long_view(range(4))

    @property
    def wheel_lk_view(self) -> List[Dict[str, object]]:
        return self.long_view(range(4, 6))


def _base_row(decoded: DecodedSession, sample) -> Dict[str, object]:
    return {
        "session_type": decoded.header.session_type if decoded.header else 0,
        "firmware_build_id": decoded.header.firmware_build_id if decoded.header else 0,
        "experiment_config_hash": decoded.header.experiment_config_hash if decoded.header else 0,
        "sequence": sample.header.sequence,
        "sample_time_us": sample.extended_time_us if sample.extended_time_us is not None else sample.sample_time_us,
        "time_s": (sample.extended_time_us if sample.extended_time_us is not None else sample.sample_time_us) * 1e-6,
        "stage_id": sample.stage_id,
        "config_seq": sample.config_seq,
        "role": sample.role,
        "fit_eligible": sample.fit_eligible,
        "active_mask": sample.active_mask,
        "online_mask": sample.online_mask,
        "tx_valid_mask": sample.tx_valid_mask,
        "rx_valid_mask": sample.rx_valid_mask,
        "saturation_mask": sample.saturation_mask,
        "safety_flags": sample.safety_flags,
        "can_error_flags": sample.can_error_flags,
    }


def _add_common_motor_fields(row: Dict[str, object], prefix: str, sample, index: int, motor, block) -> None:
    tx_valid = bool(sample.tx_valid_mask & (1 << index))
    rx_valid = bool(sample.rx_valid_mask & (1 << index))
    saturated = bool(sample.saturation_mask & (1 << index))
    mode = sample.stage_config.command_mode[index] if sample.stage_config else 0
    rate = sample.stage_config.command_rate_hz[index] if sample.stage_config else 0
    row.update(
        {
            prefix + "name": motor.motor_name,
            prefix + "type": motor.motor_type,
            prefix + "command_mode": mode,
            prefix + "command_rate_hz": rate,
            prefix + "tx_valid": tx_valid,
            prefix + "rx_valid": rx_valid,
            prefix + "online": bool(sample.online_mask & (1 << index)),
            prefix + "saturated": saturated,
            prefix + "tx_age_us": block.tx_age_us,
            prefix + "rx_age_us": block.rx_age_us,
            prefix + "tx_age_saturated": block.tx_age_saturated,
            prefix + "rx_age_saturated": block.rx_age_saturated,
            prefix + "tx_time_us": -1 if block.tx_time_us is None else block.tx_time_us,
            prefix + "rx_time_us": -1 if block.rx_time_us is None else block.rx_time_us,
            prefix + "fit_eligible": sample.fit_eligible and tx_valid and rx_valid and
            not block.tx_age_saturated and not block.rx_age_saturated and not saturated,
        }
    )


def normalize_session(
    decoded: DecodedSession,
    manifest: MotorManifest = None,
) -> NormalizedDataset:
    manifest = manifest or load_firmware_manifest()
    manifest.validate()
    if decoded.header:
        if decoded.header.motor_order_version != manifest.version:
            raise ValueError("raw stream motor order version does not match firmware manifest")
        if decoded.header.scale_manifest_hash != manifest.manifest_hash:
            raise ValueError("raw stream manifest hash does not match firmware manifest")

    rows: List[Dict[str, object]] = []
    for sample in decoded.samples:
        row = _base_row(decoded, sample)
        for index, block in enumerate(sample.dm):
            motor = manifest.motors[index]
            prefix = f"m{index}_"
            _add_common_motor_fields(row, prefix, sample, index, motor, block)
            cmd_q_motor = _uint_to_float(block.cmd_q_raw, DM_POSITION_MIN, DM_POSITION_MAX, 16)
            cmd_dq_motor = _uint_to_float(block.cmd_dq_raw, DM_VELOCITY_MIN, DM_VELOCITY_MAX, 12)
            cmd_tau_motor = _uint_to_float(block.cmd_tau_raw, DM_TORQUE_MIN, DM_TORQUE_MAX, 12)
            fb_q_motor = _uint_to_float(block.fb_q_raw, DM_POSITION_MIN, DM_POSITION_MAX, 16)
            fb_dq_motor = _uint_to_float(block.fb_dq_raw, DM_VELOCITY_MIN, DM_VELOCITY_MAX, 12)
            fb_tau_motor = _uint_to_float(block.fb_tau_raw, DM_TORQUE_MIN, DM_TORQUE_MAX, 12)
            kp_raw = sample.stage_config.dm_kp_raw[index] if sample.stage_config else 0
            kd_raw = sample.stage_config.dm_kd_raw[index] if sample.stage_config else 0
            row.update(
                {
                    prefix + "cmd_q_raw": block.cmd_q_raw,
                    prefix + "cmd_dq_raw": block.cmd_dq_raw,
                    prefix + "cmd_tau_raw": block.cmd_tau_raw,
                    prefix + "fb_q_raw": block.fb_q_raw,
                    prefix + "fb_dq_raw": block.fb_dq_raw,
                    prefix + "fb_tau_raw": block.fb_tau_raw,
                    prefix + "kp_raw": kp_raw,
                    prefix + "kd_raw": kd_raw,
                    prefix + "kp": kp_raw * 500.0 / 4095.0,
                    prefix + "kd": kd_raw * 5.0 / 4095.0,
                    prefix + "cmd_q_motor_rad": cmd_q_motor,
                    prefix + "cmd_dq_motor_rad_s": cmd_dq_motor,
                    prefix + "cmd_tau_motor_nm": cmd_tau_motor,
                    prefix + "fb_q_motor_rad": fb_q_motor,
                    prefix + "fb_dq_motor_rad_s": fb_dq_motor,
                    prefix + "fb_tau_motor_nm": fb_tau_motor,
                    prefix + "cmd_q_rad": _canonical_position(cmd_q_motor, motor.position_sign, motor.position_zero_rad, motor.gear_ratio),
                    prefix + "cmd_dq_rad_s": _canonical_velocity(cmd_dq_motor, motor.velocity_sign, motor.gear_ratio),
                    prefix + "cmd_tau_nm": _canonical_torque(cmd_tau_motor, motor.torque_sign, motor.gear_ratio),
                    prefix + "fb_q_rad": _canonical_position(fb_q_motor, motor.position_sign, motor.position_zero_rad, motor.gear_ratio),
                    prefix + "fb_dq_rad_s": _canonical_velocity(fb_dq_motor, motor.velocity_sign, motor.gear_ratio),
                    prefix + "fb_tau_nm": _canonical_torque(fb_tau_motor, motor.torque_sign, motor.gear_ratio),
                }
            )

        for local_index, block in enumerate(sample.lk):
            index = local_index + 4
            motor = manifest.motors[index]
            prefix = f"m{index}_"
            _add_common_motor_fields(row, prefix, sample, index, motor, block)
            mode = sample.stage_config.command_mode[index] if sample.stage_config else 0
            cmd_torque = block.cmd_primary_raw * LK_CURRENT_A_PER_COMMAND * LK_TORQUE_CONSTANT
            cmd_velocity = block.cmd_primary_raw * 0.01 * DEG_TO_RAD
            fb_position_motor = block.fb_encoder_raw * 2.0 * math.pi / 65535.0 - math.pi
            fb_position_motor += block.fb_turn_count * 2.0 * math.pi
            fb_velocity_motor = block.fb_speed_raw * DEG_TO_RAD
            fb_current = block.fb_iq_raw * LK_CURRENT_A_PER_COMMAND
            fb_torque_motor = fb_current * LK_TORQUE_CONSTANT
            row.update(
                {
                    prefix + "cmd_primary_raw": block.cmd_primary_raw,
                    prefix + "fb_encoder_raw": block.fb_encoder_raw,
                    prefix + "fb_turn_count": block.fb_turn_count,
                    prefix + "fb_speed_raw": block.fb_speed_raw,
                    prefix + "fb_iq_raw": block.fb_iq_raw,
                    prefix + "cmd_torque_motor_nm": cmd_torque if mode == COMMAND_LK_TORQUE else math.nan,
                    prefix + "cmd_velocity_motor_rad_s": cmd_velocity if mode == COMMAND_LK_VELOCITY else math.nan,
                    prefix + "fb_position_motor_rad": fb_position_motor,
                    prefix + "fb_velocity_motor_rad_s": fb_velocity_motor,
                    prefix + "fb_current_a": fb_current,
                    prefix + "fb_torque_motor_nm": fb_torque_motor,
                    prefix + "cmd_torque_nm": _canonical_torque(cmd_torque, motor.torque_sign, motor.gear_ratio) if mode == COMMAND_LK_TORQUE else math.nan,
                    prefix + "cmd_velocity_rad_s": _canonical_velocity(cmd_velocity, motor.velocity_sign, motor.gear_ratio) if mode == COMMAND_LK_VELOCITY else math.nan,
                    prefix + "fb_position_rad": _canonical_position(fb_position_motor, motor.position_sign, motor.position_zero_rad, motor.gear_ratio),
                    prefix + "fb_velocity_rad_s": _canonical_velocity(fb_velocity_motor, motor.velocity_sign, motor.gear_ratio),
                    prefix + "fb_torque_nm": _canonical_torque(fb_torque_motor, motor.torque_sign, motor.gear_ratio),
                }
            )
        rows.append(row)
    return NormalizedDataset(decoded, manifest, rows)
