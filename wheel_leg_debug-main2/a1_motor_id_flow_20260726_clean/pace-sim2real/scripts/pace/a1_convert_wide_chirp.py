"""Convert reduced-width A1 chirp CSV logs into the PACE tensor contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch


DT_S = 0.002
SAMPLE_COUNT = 10_000
PACE_JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)
CSV_LEG_ORDER = ("FR", "FL", "RR", "RL")
CSV_JOINT_ORDER = ("hip", "thigh", "calf")
METADATA_COLUMNS = (
    "sample_index",
    "t_monotonic_ns",
    "recv_monotonic_ns",
    "send_monotonic_ns",
    "t_rel_s",
    "stage",
    "chirp_phase_rad",
    "chirp_freq_hz",
    "lowstate_tick",
    "receive_status",
    "send_status",
    "kp",
    "kd",
)
EXPECTED_HEADER = (
    list(METADATA_COLUMNS)
    + [f"q_des_{leg}_{joint}" for leg in CSV_LEG_ORDER for joint in CSV_JOINT_ORDER]
    + [f"q_{leg}_{joint}" for leg in CSV_LEG_ORDER for joint in CSV_JOINT_ORDER]
    + [f"dq_{leg}_{joint}" for leg in CSV_LEG_ORDER for joint in CSV_JOINT_ORDER]
    + [f"tau_{leg}_{joint}" for leg in CSV_LEG_ORDER for joint in CSV_JOINT_ORDER]
    + [f"temp_{leg}_{joint}" for leg in CSV_LEG_ORDER for joint in CSV_JOINT_ORDER]
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_path(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_payload_sha256(data: dict[str, torch.Tensor]) -> str:
    validate_pace_data(data)
    digest = hashlib.sha256()
    for key in ("time", "dof_pos", "des_dof_pos"):
        tensor = data[key].detach().cpu().contiguous()
        digest.update(key.encode("ascii") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
            + b"\0"
        )
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_pace_data(data: dict[str, torch.Tensor]) -> None:
    _require(
        set(data) == {"time", "dof_pos", "des_dof_pos"},
        "PACE data keys must be exactly time, dof_pos, des_dof_pos",
    )
    time = data["time"]
    measured = data["dof_pos"]
    desired = data["des_dof_pos"]
    _require(time.ndim == 1 and time.numel() > 0, "PACE time must have shape (T,)")
    expected_shape = (time.shape[0], len(PACE_JOINT_ORDER))
    _require(measured.shape == expected_shape, "PACE dof_pos shape mismatch")
    _require(desired.shape == expected_shape, "PACE des_dof_pos shape mismatch")
    _require(time.dtype == torch.float64, "PACE time must use torch.float64")
    _require(measured.dtype == torch.float32, "PACE dof_pos must use torch.float32")
    _require(desired.dtype == torch.float32, "PACE des_dof_pos must use torch.float32")
    for key, tensor in data.items():
        _require(tensor.device.type == "cpu", f"PACE {key} must be on CPU")
        _require(tensor.is_contiguous(), f"PACE {key} must be contiguous")
        _require(bool(torch.isfinite(tensor).all()), f"PACE {key} is non-finite")
    expected_time = torch.arange(time.numel(), dtype=torch.float64) * DT_S
    _require(torch.equal(time, expected_time), "PACE time grid is not exact")


def _float(row: dict[str, str], column: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"CSV column {column} is not a finite float") from exc
    _require(math.isfinite(value), f"CSV column {column} is not finite")
    return value


def _csv_joint_key(joint_name: str) -> str:
    if not joint_name.endswith("_joint"):
        raise ValueError(f"unexpected PACE joint name: {joint_name}")
    return joint_name[: -len("_joint")]


def _extract_positions(
    rows: Sequence[dict[str, str]],
    prefix: str,
) -> torch.Tensor:
    values = [
        [_float(row, f"{prefix}_{_csv_joint_key(name)}") for name in PACE_JOINT_ORDER]
        for row in rows
    ]
    return torch.tensor(values, dtype=torch.float32).contiguous()


def _unique_values(rows: Sequence[dict[str, str]], column: str) -> list[str]:
    return sorted({row[column] for row in rows})


def _stage_counts(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["stage"] for row in rows).items()))


def _validate_source_timing(
    chirp_rows: Sequence[dict[str, str]],
    *,
    dt_s: float,
) -> tuple[list[float], list[float]]:
    source_time = [_float(row, "t_rel_s") for row in chirp_rows]
    source_frequency = [_float(row, "chirp_freq_hz") for row in chirp_rows]
    if len(source_time) > 1:
        for index, (left, right) in enumerate(zip(source_time, source_time[1:]), 1):
            actual_dt = right - left
            if abs(actual_dt - dt_s) > 1.0e-7:
                raise ValueError(
                    f"chirp source dt differs from {dt_s} at sample {index}: {actual_dt}"
                )
    return source_time, source_frequency


def convert_wide_chirp_csv(
    input_path: Path | str,
    *,
    sample_count: int = SAMPLE_COUNT,
    dt_s: float = DT_S,
    fixture_pose: str = "upside_down_fixed_air",
    payload_kg: float = 5.0,
    support: str = "back_supported_fixed_table_trunk_footprint",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = Path(input_path).resolve()
    _require(path.is_file(), f"input CSV does not exist: {path}")
    _require(sample_count > 0, "sample_count must be positive")
    _require(dt_s == DT_S, "A1 PACE conversion currently requires dt_s=0.002")
    _require(math.isfinite(payload_kg) and payload_kg >= 0.0, "payload_kg is invalid")

    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == EXPECTED_HEADER, "CSV header is not the 73-column A1 wide chirp schema")
        rows = list(reader)

    _require(rows, "CSV contains no data rows")
    chirp_rows = [row for row in rows if row["stage"] == "chirp"]
    _require(
        len(chirp_rows) == sample_count,
        f"chirp sample count must be {sample_count}, got {len(chirp_rows)}",
    )
    source_time, source_frequency = _validate_source_timing(chirp_rows, dt_s=dt_s)
    kp_values = _unique_values(rows, "kp")
    kd_values = _unique_values(rows, "kd")
    _require(len(kp_values) == 1 and len(kd_values) == 1, "CSV kp/kd must be uniform")

    data = {
        "time": (torch.arange(sample_count, dtype=torch.float64) * dt_s).contiguous(),
        "dof_pos": _extract_positions(chirp_rows, "q"),
        "des_dof_pos": _extract_positions(chirp_rows, "q_des"),
    }
    validate_pace_data(data)

    stage_counts = _stage_counts(rows)
    manifest: dict[str, Any] = {
        "schema_version": "a1_pace_wide_chirp_conversion/v1",
        "source_path": str(path),
        "source_basename": path.name,
        "source_sha256": sha256_path(path),
        "source_row_count": len(rows),
        "source_header_column_count": len(EXPECTED_HEADER),
        "stage_counts": stage_counts,
        "chirp_sample_count": len(chirp_rows),
        "discarded_center_ramp_sample_count": stage_counts.get("center_ramp", 0),
        "dt_s": dt_s,
        "source_chirp_time_start_s": source_time[0],
        "source_chirp_time_end_s": source_time[-1],
        "source_chirp_frequency_start_hz": source_frequency[0],
        "source_chirp_frequency_end_hz": source_frequency[-1],
        "kp": float(kp_values[0]),
        "kd": float(kd_values[0]),
        "receive_status_values": _unique_values(rows, "receive_status"),
        "send_status_values": _unique_values(rows, "send_status"),
        "pace_joint_order": list(PACE_JOINT_ORDER),
        "csv_columns_used": {
            "time": "generated_from_chirp_sample_index",
            "dof_pos": [f"q_{_csv_joint_key(name)}" for name in PACE_JOINT_ORDER],
            "des_dof_pos": [f"q_des_{_csv_joint_key(name)}" for name in PACE_JOINT_ORDER],
        },
        "fixture": {
            "pose": fixture_pose,
            "root": "fixed_in_air",
            "support": support,
            "payload_kg": payload_kg,
        },
        "tensor_payload_sha256": tensor_payload_sha256(data),
    }
    return data, manifest


def _atomic_torch_save(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".pt", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(value, temporary, _use_new_zipfile_serialization=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".json", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
            newline="\n",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def convert_and_write(
    input_path: Path | str,
    output_path: Path | str,
    manifest_output: Path | str,
    *,
    fixture_pose: str = "upside_down_fixed_air",
    payload_kg: float = 5.0,
    support: str = "back_supported_fixed_table_trunk_footprint",
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    manifest_path = Path(manifest_output).resolve()
    _require(output != manifest_path, "output and manifest paths must differ")
    data, manifest = convert_wide_chirp_csv(
        input_path,
        fixture_pose=fixture_pose,
        payload_kg=payload_kg,
        support=support,
    )
    _atomic_torch_save(data, output)
    manifest["output_path"] = str(output)
    manifest["output_sha256"] = sha256_path(output)
    _atomic_json(manifest, manifest_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert one 73-column A1 chirp CSV into PACE chirp_data.pt"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--fixture-pose", default="upside_down_fixed_air")
    parser.add_argument("--payload-kg", type=float, default=5.0)
    parser.add_argument(
        "--support", default="back_supported_fixed_table_trunk_footprint"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = convert_and_write(
        args.input,
        args.output,
        args.manifest_output,
        fixture_pose=args.fixture_pose,
        payload_kg=args.payload_kg,
        support=args.support,
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
