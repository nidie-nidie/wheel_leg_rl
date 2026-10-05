#!/usr/bin/env python3
"""Generate PPT-ready validation plots for the new offset closed-chain VMC.

The script validates the source CSV/JSON before plotting.  It does not modify
the input files and does not fabricate missing samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / ".matplotlib-cache"))

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image


REQUIRED_CSV_FIELDS = (
    "target_l0_m",
    "target_phi0_deg",
    "simplified_minus_full_l0_um",
    "simplified_minus_full_phi0_microdeg",
    "analytic_minus_fd_jacobian_max_abs",
)

REQUIRED_JSON_FIELDS = (
    "mujoco_version",
    "max_abs_l0_difference_um",
    "max_abs_theta_difference_microdeg",
    "samples",
)

OUTPUT_FILENAMES = (
    "new_vmc_validation_combined.png",
    "new_vmc_validation_combined.svg",
    "new_vmc_l0_error.png",
    "new_vmc_l0_error.svg",
    "new_vmc_phi0_error.png",
    "new_vmc_phi0_error.svg",
    "new_vmc_error_envelope.csv",
    "new_vmc_validation_metrics.json",
    "generate_new_vmc_validation_plot.py",
)

COLORS = {
    "text": "#132B3F",
    "cyan": "#12B8C8",
    "orange": "#FF914D",
    "grid": "#D9E2EA",
    "background": "#F7F9FB",
    "white": "#FFFFFF",
    "muted": "#587184",
}


@dataclass(frozen=True)
class Sample:
    target_l0_m: float
    target_phi0_deg: float
    l0_error_um: float
    phi0_error_microdeg: float
    jacobian_error: float


@dataclass(frozen=True)
class EnvelopeRow:
    target_l0_m: float
    max_abs_l0_error_um: float
    max_abs_phi0_error_mdeg: float
    max_abs_jacobian_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate new offset closed-chain VMC validation plots."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "output" / "offset_kinematics_grid.csv",
        help="Input offset_kinematics_grid.csv path.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=ROOT / "output" / "offset_kinematics_mujoco_validation.json",
        help="Input MuJoCo static validation JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "new_vmc_validation",
        help="Directory for generated PPT artifacts.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="Optional Chinese font file path.",
    )
    return parser.parse_args()


def finite_float(raw: str, field: str, row_index: int) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_index}: field {field!r} is not numeric: {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Row {row_index}: field {field!r} is NaN or Inf: {raw!r}"
        )
    return value


def load_samples(csv_path: Path) -> tuple[list[Sample], list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")
    if csv_path.stat().st_size <= 0:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {csv_path}")
        missing = [field for field in REQUIRED_CSV_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required fields: {missing}")

        samples: list[Sample] = []
        for row_index, row in enumerate(reader, start=2):
            samples.append(
                Sample(
                    target_l0_m=finite_float(row["target_l0_m"], "target_l0_m", row_index),
                    target_phi0_deg=finite_float(
                        row["target_phi0_deg"], "target_phi0_deg", row_index
                    ),
                    l0_error_um=finite_float(
                        row["simplified_minus_full_l0_um"],
                        "simplified_minus_full_l0_um",
                        row_index,
                    ),
                    phi0_error_microdeg=finite_float(
                        row["simplified_minus_full_phi0_microdeg"],
                        "simplified_minus_full_phi0_microdeg",
                        row_index,
                    ),
                    jacobian_error=finite_float(
                        row["analytic_minus_fd_jacobian_max_abs"],
                        "analytic_minus_fd_jacobian_max_abs",
                        row_index,
                    ),
                )
            )
    if not samples:
        raise ValueError(f"Input CSV contains no data rows: {csv_path}")
    return samples, list(reader.fieldnames)


def load_mujoco_validation(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"Input JSON does not exist: {json_path}")
    if json_path.stat().st_size <= 0:
        raise ValueError(f"Input JSON is empty: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_JSON_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Input JSON missing required fields: {missing}")
    if not isinstance(data["samples"], list):
        raise ValueError("Input JSON field 'samples' must be a list")
    for field in (
        "max_abs_l0_difference_um",
        "max_abs_theta_difference_microdeg",
    ):
        value = float(data[field])
        if not math.isfinite(value):
            raise ValueError(f"Input JSON field {field!r} is NaN or Inf")
    return data


def group_envelope(samples: Iterable[Sample]) -> list[EnvelopeRow]:
    groups: dict[float, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[round(sample.target_l0_m, 10)].append(sample)

    envelope: list[EnvelopeRow] = []
    for l0 in sorted(groups):
        group = groups[l0]
        envelope.append(
            EnvelopeRow(
                target_l0_m=l0,
                max_abs_l0_error_um=max(abs(item.l0_error_um) for item in group),
                max_abs_phi0_error_mdeg=max(
                    abs(item.phi0_error_microdeg) for item in group
                )
                / 1000.0,
                max_abs_jacobian_error=max(item.jacobian_error for item in group),
            )
        )
    return envelope


def choose_font(user_font: Path | None) -> str | None:
    candidates: list[Path] = []
    if user_font is not None:
        candidates.append(user_font)
    candidates.extend(
        [
            Path("/mnt/c/Windows/Fonts/msyh.ttc"),
            Path("/mnt/c/Windows/Fonts/simhei.ttf"),
            Path("/mnt/c/Windows/Fonts/simsun.ttc"),
            Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        ]
    )
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            fm.fontManager.addfont(str(path))
            return fm.FontProperties(fname=str(path)).get_name()
    return None


def configure_matplotlib(font_name: str | None) -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["white"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    if font_name:
        mpl.rcParams["font.family"] = font_name


def format_sci(value: float) -> str:
    return f"{value:.3e}"


def format_fixed(value: float, digits: int = 3) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def set_engineering_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.75)
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))


def annotate_max(
    ax: plt.Axes,
    x_values: list[float],
    y_values: list[float],
    unit: str,
) -> tuple[float, float]:
    max_index = max(range(len(y_values)), key=lambda index: y_values[index])
    max_x = x_values[max_index]
    max_y = y_values[max_index]
    ax.scatter(
        [max_x],
        [max_y],
        s=56,
        color=COLORS["orange"],
        zorder=5,
        label="全局最大值",
    )
    x_span = max(x_values) - min(x_values)
    near_right_edge = x_span > 0.0 and max_x > min(x_values) + 0.72 * x_span
    text_offset = (-18, 16) if near_right_edge else (12, 16)
    horizontal_align = "right" if near_right_edge else "left"
    ax.annotate(
        f"最大 {format_fixed(max_y)} {unit}\nL0={max_x:.2f} m",
        xy=(max_x, max_y),
        xytext=text_offset,
        textcoords="offset points",
        ha=horizontal_align,
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["orange"],
            "linewidth": 1.2,
        },
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": COLORS["white"],
            "edgecolor": COLORS["orange"],
            "linewidth": 0.9,
        },
        fontsize=9,
    )
    return max_x, max_y


def plot_curve(
    ax: plt.Axes,
    x_values: list[float],
    y_values: list[float],
    title: str,
    y_label: str,
    unit: str,
    fill: bool = False,
) -> tuple[float, float]:
    ax.plot(
        x_values,
        y_values,
        color=COLORS["cyan"],
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        markerfacecolor=COLORS["white"],
        markeredgewidth=1.4,
        label="误差包络",
    )
    if fill:
        ax.fill_between(x_values, y_values, 0, color=COLORS["cyan"], alpha=0.10)
    max_x, max_y = annotate_max(ax, x_values, y_values, unit)
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xlabel("虚拟腿长 L0 / m")
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom=0.0, top=max(max_y * 1.25, 1e-12))
    ax.set_xlim(min(x_values) - 0.006, max(x_values) + 0.006)
    set_engineering_axes(ax)
    ax.legend(loc="upper left", frameon=False)
    return max_x, max_y


def add_combined_text(
    fig: plt.Figure,
    metrics: dict,
    mujoco_data: dict,
) -> None:
    fig.text(
        0.5,
        0.955,
        "新偏置闭链 VMC：工作空间运动学映射验证",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.5,
        0.912,
        (
            f"{metrics['total_samples']} 个构型点｜每个腿长取 "
            f"phi0={metrics['phi0_range_deg'][0]:.0f}°～"
            f"{metrics['phi0_range_deg'][1]:.0f}° 范围内的最大绝对误差"
        ),
        ha="center",
        va="center",
        fontsize=11.5,
        color=COLORS["muted"],
    )
    fig.text(
        0.070,
        0.170,
        (
            f"工作空间构型：{metrics['total_samples']}    "
            f"最大雅可比误差：{format_sci(metrics['max_abs_jacobian_error'])}    "
            f"MuJoCo 静态复核：{metrics['mujoco_validation_sample_count']} 点"
        ),
        ha="left",
        va="center",
        fontsize=10,
        color=COLORS["text"],
    )
    card = (
        "MuJoCo 静态点位复核\n"
        f"版本：{mujoco_data['mujoco_version']}\n"
        f"点位：{metrics['mujoco_validation_sample_count']}\n"
        f"最大腿长差：{format_sci(metrics['mujoco_max_abs_l0_difference_um'])} μm\n"
        "最大角度差："
        f"{format_sci(metrics['mujoco_max_abs_theta_difference_microdeg'])} μdeg"
    )
    fig.text(
        0.700,
        0.068,
        card,
        ha="left",
        va="bottom",
        fontsize=8.8,
        linespacing=1.18,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": COLORS["white"],
            "edgecolor": COLORS["grid"],
            "linewidth": 0.9,
        },
    )
    fig.text(
        0.5,
        0.026,
        "本图验证的是新 VMC 的运动学与雅可比映射，不代表闭环稳定性或实车控制效果已经完成验证。",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["muted"],
    )


def save_figure(fig: plt.Figure, png_path: Path, svg_path: Path) -> None:
    fig.savefig(png_path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(svg_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def create_combined_plot(
    output_dir: Path,
    envelope: list[EnvelopeRow],
    metrics: dict,
    mujoco_data: dict,
) -> None:
    x_values = [row.target_l0_m for row in envelope]
    l0_values = [row.max_abs_l0_error_um for row in envelope]
    phi_values = [row.max_abs_phi0_error_mdeg for row in envelope]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.5), dpi=300)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.805, bottom=0.340, wspace=0.25)
    plot_curve(
        axes[0],
        x_values,
        l0_values,
        "虚拟腿长映射误差",
        "最大绝对误差 / μm",
        "μm",
        fill=True,
    )
    plot_curve(
        axes[1],
        x_values,
        phi_values,
        "虚拟腿角映射误差",
        "最大绝对误差 / mdeg",
        "mdeg",
        fill=False,
    )
    add_combined_text(fig, metrics, mujoco_data)
    save_figure(
        fig,
        output_dir / "new_vmc_validation_combined.png",
        output_dir / "new_vmc_validation_combined.svg",
    )


def create_single_plot(
    output_dir: Path,
    envelope: list[EnvelopeRow],
    metrics: dict,
    kind: str,
) -> None:
    x_values = [row.target_l0_m for row in envelope]
    if kind == "l0":
        y_values = [row.max_abs_l0_error_um for row in envelope]
        title = "新偏置闭链 VMC：虚拟腿长映射误差"
        y_label = "最大绝对误差 / μm"
        unit = "μm"
        png_name = "new_vmc_l0_error.png"
        svg_name = "new_vmc_l0_error.svg"
        fill = True
    elif kind == "phi0":
        y_values = [row.max_abs_phi0_error_mdeg for row in envelope]
        title = "新偏置闭链 VMC：虚拟腿角映射误差"
        y_label = "最大绝对误差 / mdeg"
        unit = "mdeg"
        png_name = "new_vmc_phi0_error.png"
        svg_name = "new_vmc_phi0_error.svg"
        fill = False
    else:
        raise ValueError(kind)

    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=300)
    fig.subplots_adjust(left=0.115, right=0.965, top=0.82, bottom=0.19)
    fig.text(
        0.5,
        0.940,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.5,
        0.895,
        (
            f"{metrics['total_samples']} 个构型点｜phi0="
            f"{metrics['phi0_range_deg'][0]:.0f}°～"
            f"{metrics['phi0_range_deg'][1]:.0f}° 最坏情况包络"
        ),
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    plot_curve(ax, x_values, y_values, "", y_label, unit, fill=fill)
    save_figure(fig, output_dir / png_name, output_dir / svg_name)


def write_envelope_csv(output_dir: Path, envelope: list[EnvelopeRow]) -> None:
    path = output_dir / "new_vmc_error_envelope.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_l0_m",
                "max_abs_l0_error_um",
                "max_abs_phi0_error_mdeg",
                "max_abs_jacobian_error",
            ],
        )
        writer.writeheader()
        for row in envelope:
            writer.writerow(
                {
                    "target_l0_m": f"{row.target_l0_m:.6f}",
                    "max_abs_l0_error_um": f"{row.max_abs_l0_error_um:.12g}",
                    "max_abs_phi0_error_mdeg": f"{row.max_abs_phi0_error_mdeg:.12g}",
                    "max_abs_jacobian_error": f"{row.max_abs_jacobian_error:.12g}",
                }
            )


def write_metrics(output_dir: Path, metrics: dict) -> None:
    path = output_dir / "new_vmc_validation_metrics.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def verify_outputs(output_dir: Path) -> None:
    missing = [name for name in OUTPUT_FILENAMES if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing output files: {missing}")

    image_expectations = {
        "new_vmc_validation_combined.png": (2400, 1350),
        "new_vmc_l0_error.png": None,
        "new_vmc_phi0_error.png": None,
    }
    for name, expected_size in image_expectations.items():
        path = output_dir / name
        if path.stat().st_size <= 0:
            raise ValueError(f"PNG is empty: {path}")
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        if expected_size is not None and (width, height) != expected_size:
            raise ValueError(
                f"PNG size mismatch for {name}: got {(width, height)}, expected {expected_size}"
            )
        if width < 1200 or height < 700:
            raise ValueError(f"PNG is unexpectedly small: {name} {(width, height)}")

    for name in (
        "new_vmc_validation_combined.svg",
        "new_vmc_l0_error.svg",
        "new_vmc_phi0_error.svg",
    ):
        path = output_dir / name
        if path.stat().st_size <= 0:
            raise ValueError(f"SVG is empty: {path}")
        root = ET.parse(path).getroot()
        if not root.tag.lower().endswith("svg"):
            raise ValueError(f"SVG root is not <svg>: {path}")


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    input_json = args.input_json.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    font_name = choose_font(args.font)
    configure_matplotlib(font_name)

    samples, csv_fields = load_samples(input_csv)
    mujoco_data = load_mujoco_validation(input_json)
    envelope = group_envelope(samples)

    unique_l0 = sorted({round(sample.target_l0_m, 10) for sample in samples})
    unique_phi0 = sorted({round(sample.target_phi0_deg, 10) for sample in samples})
    metrics = {
        "input_csv": str(input_csv),
        "input_json": str(input_json),
        "csv_fields": csv_fields,
        "total_samples": len(samples),
        "unique_l0_count": len(unique_l0),
        "unique_phi0_count": len(unique_phi0),
        "l0_range_m": [min(unique_l0), max(unique_l0)],
        "phi0_range_deg": [min(unique_phi0), max(unique_phi0)],
        "max_abs_l0_error_um": max(abs(sample.l0_error_um) for sample in samples),
        "max_abs_phi0_error_mdeg": max(
            abs(sample.phi0_error_microdeg) for sample in samples
        )
        / 1000.0,
        "max_abs_jacobian_error": max(sample.jacobian_error for sample in samples),
        "mujoco_version": str(mujoco_data["mujoco_version"]),
        "mujoco_validation_sample_count": len(mujoco_data["samples"]),
        "mujoco_max_abs_l0_difference_um": float(
            mujoco_data["max_abs_l0_difference_um"]
        ),
        "mujoco_max_abs_theta_difference_microdeg": float(
            mujoco_data["max_abs_theta_difference_microdeg"]
        ),
        "output_dir": str(output_dir),
        "font": font_name,
    }

    write_envelope_csv(output_dir, envelope)
    write_metrics(output_dir, metrics)
    create_combined_plot(output_dir, envelope, metrics, mujoco_data)
    create_single_plot(output_dir, envelope, metrics, "l0")
    create_single_plot(output_dir, envelope, metrics, "phi0")
    shutil.copy2(Path(__file__).resolve(), output_dir / "generate_new_vmc_validation_plot.py")
    verify_outputs(output_dir)

    print("数据检查通过")
    print(f"输入 CSV: {input_csv}")
    print(f"输入 JSON: {input_json}")
    print(f"CSV 字段数: {len(csv_fields)}")
    print(f"样本数: {metrics['total_samples']}")
    print(f"唯一 L0 数量: {metrics['unique_l0_count']}")
    print(f"唯一 phi0 数量: {metrics['unique_phi0_count']}")
    print(
        "L0 范围: "
        f"{metrics['l0_range_m'][0]:.2f}～{metrics['l0_range_m'][1]:.2f} m"
    )
    print(
        "phi0 范围: "
        f"{metrics['phi0_range_deg'][0]:.0f}～{metrics['phi0_range_deg'][1]:.0f} deg"
    )
    print(
        "最大腿长绝对误差: "
        f"{metrics['max_abs_l0_error_um']:.6g} um"
    )
    print(
        "最大腿角绝对误差: "
        f"{metrics['max_abs_phi0_error_mdeg']:.6g} mdeg"
    )
    print(
        "最大解析/有限差分雅可比误差: "
        f"{metrics['max_abs_jacobian_error']:.6e}"
    )
    print(
        "MuJoCo 静态复核: "
        f"version={metrics['mujoco_version']}, "
        f"samples={metrics['mujoco_validation_sample_count']}, "
        f"max_l0={metrics['mujoco_max_abs_l0_difference_um']:.3e} um, "
        f"max_theta={metrics['mujoco_max_abs_theta_difference_microdeg']:.3e} microdeg"
    )
    print("输出文件:")
    for name in OUTPUT_FILENAMES:
        print(f"  {output_dir / name}")


if __name__ == "__main__":
    main()
