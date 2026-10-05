#!/usr/bin/env python3
"""Summarize layered MuJoCo stand-control CSV captures.

The script separates three quantities that should not be conflated:

1. CAD_ONE_CLOSURE versus the nominal full-XML closed-chain model;
2. nominal full-XML geometry versus the loaded MuJoCo mechanism state;
3. the combined runtime VMC-versus-axis difference.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

from derive_offset_kinematics import analytic_closed_chain, cad_one_closure


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = [{key: float(value) for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"empty control CSV: {path}")
    return rows


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def peak_to_peak(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows]
    return max(values) - min(values)


def rms(values: Iterable[float]) -> float:
    items = list(values)
    return math.sqrt(sum(value * value for value in items) / len(items))


def wrapped_difference(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def summarize(path: Path, start_time: float) -> dict[str, float | int | str]:
    all_rows = read_rows(path)
    rows = [row for row in all_rows if row["t_s"] >= start_time]
    if not rows:
        raise ValueError(f"no samples at or after {start_time:.3f} s: {path}")

    cad_full_l0_error: list[float] = []
    cad_full_phi_error: list[float] = []
    full_axis_l0_error: list[float] = []
    full_axis_phi_error: list[float] = []
    cad_axis_l0_error: list[float] = []
    cad_axis_phi_error: list[float] = []

    for row in rows:
        for side in ("left", "right"):
            q = (row[f"q_front_{side}_rad"], row[f"q_rear_{side}_rad"])
            cad = cad_one_closure(q)
            full = analytic_closed_chain(q)
            axis_l0 = row[f"axis_l0_{side}_m"]
            axis_phi0 = row[f"axis_phi0_{side}_rad"]
            cad_full_l0_error.append(cad["L0"] - full["L0"])
            cad_full_phi_error.append(wrapped_difference(cad["phi0"], full["phi0"]))
            full_axis_l0_error.append(full["L0"] - axis_l0)
            full_axis_phi_error.append(wrapped_difference(full["phi0"], axis_phi0))
            cad_axis_l0_error.append(cad["L0"] - axis_l0)
            cad_axis_phi_error.append(wrapped_difference(cad["phi0"], axis_phi0))

    closure_columns = (
        "closure_io_mk_left_m",
        "closure_op_kn_left_m",
        "closure_ag_ec_right_m",
        "closure_gh_cf_right_m",
    )
    joint_sat_columns = tuple(f"joint_sat_j{index}" for index in range(4))
    wheel_sat_columns = ("wheel_sat_left", "wheel_sat_right")

    result: dict[str, float | int | str] = {
        "case": path.stem,
        "samples": len(rows),
        "start_time_s": start_time,
        "nan_count": sum(
            math.isnan(value) for row in all_rows for value in row.values()
        ),
        "kinematics_invalid_samples": sum(
            row["kin_left"] != 1.0 or row["kin_right"] != 1.0
            for row in all_rows
        ),
        "pitch_p2p_deg": math.degrees(peak_to_peak(rows, "pitch_rad")),
        "body_x_p2p_mm": 1.0e3 * peak_to_peak(rows, "body_x_m"),
        "body_z_p2p_mm": 1.0e3 * peak_to_peak(rows, "body_z_m"),
        "max_joint_q_p2p_deg": max(
            math.degrees(peak_to_peak(rows, f"q{index}_rad"))
            for index in range(4)
        ),
        "cad_vs_full_l0_max_abs_mm": 1.0e3 * max(map(abs, cad_full_l0_error)),
        "cad_vs_full_phi0_max_abs_deg": math.degrees(
            max(map(abs, cad_full_phi_error))
        ),
        "full_vs_axis_l0_max_abs_mm": 1.0e3 * max(map(abs, full_axis_l0_error)),
        "full_vs_axis_phi0_max_abs_deg": math.degrees(
            max(map(abs, full_axis_phi_error))
        ),
        "cad_vs_axis_l0_max_abs_mm": 1.0e3 * max(map(abs, cad_axis_l0_error)),
        "cad_vs_axis_phi0_max_abs_deg": math.degrees(
            max(map(abs, cad_axis_phi_error))
        ),
        "closure_site_max_mm": 1.0e3
        * max(row[column] for row in rows for column in closure_columns),
        "l0_mean_m": mean(
            row[column]
            for row in rows
            for column in ("l0_left_m", "l0_right_m")
        ),
        "l0_target_mean_m": mean(
            row[column]
            for row in rows
            for column in ("l0_set_left_m", "l0_set_right_m")
        ),
        "d_l0_rms_mps": rms(
            row[column]
            for row in rows
            for column in ("d_l0_left_mps", "d_l0_right_mps")
        ),
        "d_phi0_rms_radps": rms(
            row[column]
            for row in rows
            for column in ("d_phi0_left_radps", "d_phi0_right_radps")
        ),
        "tau_mit_mean_abs_nm": mean(
            abs(row[f"tau_mit_j{index}_nm"])
            for row in rows
            for index in range(4)
        ),
        "tau_vmc_mean_abs_nm": mean(
            abs(row[f"tau_vmc_j{index}_nm"])
            for row in rows
            for index in range(4)
        ),
        "tau_final_mean_abs_nm": mean(
            abs(row[f"tau_final_j{index}_nm"])
            for row in rows
            for index in range(4)
        ),
        "joint_saturation_fraction": mean(
            row[column] for row in rows for column in joint_sat_columns
        ),
        "wheel_saturation_fraction": mean(
            row[column] for row in rows for column in wheel_sat_columns
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--start-time", type=float, default=1.0)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    summaries = [summarize(path, args.start_time) for path in args.csv]
    fieldnames = list(summaries[0])
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summaries)


if __name__ == "__main__":
    main()
