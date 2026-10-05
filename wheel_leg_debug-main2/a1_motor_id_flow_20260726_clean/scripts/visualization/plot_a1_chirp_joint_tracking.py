from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV = Path(
    r"C:\Users\22319\program\python_project\A1_rl\remote_mirror\A1_Base-a1-pace\logs\a1_chirp"
    r"\A1chirp_20260726_071950_f0p10-f1p10Hz_d20_r2_Kp25_Kd2_ampHip0p22_ampTh0p38_ampCalf0p38_cH0p10_cFT0p80_cRT1p00_cCalf-1p50_v25_a1600.csv"
)
DEFAULT_OUT_DIR = Path(r"C:\Users\22319\program\python_project\A1_rl\visualizations")

JOINTS = [
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot A1 chirp desired and measured joint positions.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="A1 wide chirp CSV file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for PNG files.")
    parser.add_argument("--include-ramp", action="store_true", help="Plot all samples instead of only chirp samples.")
    return parser.parse_args()


def load_data(csv_path: Path, include_ramp: bool) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(csv_path)
    if not include_ramp:
        frame = frame.loc[frame["stage"] == "chirp"].copy()
    if frame.empty:
        raise ValueError("No samples selected from CSV.")

    missing = []
    for joint in JOINTS:
        missing.extend([col for col in (f"q_des_{joint}", f"q_{joint}") if col not in frame.columns])
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    time_s = frame["t_rel_s"].to_numpy(dtype=np.float64)
    time_s = time_s - time_s[0]
    return frame, time_s


def plot_tracking(frame: pd.DataFrame, time_s: np.ndarray, csv_path: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
    fig.suptitle(
        f"A1 chirp joint tracking: desired vs measured\n{csv_path.name}",
        fontsize=14,
    )

    for axis, joint in zip(axes.ravel(), JOINTS):
        desired = frame[f"q_des_{joint}"].to_numpy(dtype=np.float64)
        measured = frame[f"q_{joint}"].to_numpy(dtype=np.float64)
        error = measured - desired
        rmse = float(np.sqrt(np.mean(error * error)))

        axis.plot(time_s, desired, color="#1f77b4", linewidth=1.2, label="des_dof_pos")
        axis.plot(time_s, measured, color="#ff7f0e", linewidth=1.0, alpha=0.9, label="dof_pos")
        axis.set_title(f"{joint}  RMSE={rmse:.4f} rad", fontsize=10)
        axis.grid(True, alpha=0.25)
        axis.set_ylabel("rad")

    for axis in axes[-1, :]:
        axis.set_xlabel("time from chirp start (s)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error(frame: pd.DataFrame, time_s: np.ndarray, csv_path: Path, out_path: Path) -> None:
    errors = np.vstack(
        [
            frame[f"q_{joint}"].to_numpy(dtype=np.float64)
            - frame[f"q_des_{joint}"].to_numpy(dtype=np.float64)
            for joint in JOINTS
        ]
    )
    rmse = np.sqrt(np.mean(errors * errors, axis=1))
    max_abs = np.max(np.abs(errors), axis=1)

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.8, 1.2], hspace=0.35)

    heat_axis = fig.add_subplot(grid[0])
    extent = [float(time_s[0]), float(time_s[-1]), -0.5, len(JOINTS) - 0.5]
    vmax = float(np.percentile(np.abs(errors), 99))
    vmax = max(vmax, 1.0e-6)
    image = heat_axis.imshow(
        errors,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
    )
    heat_axis.set_title(f"A1 chirp tracking error: measured - desired\n{csv_path.name}", fontsize=14)
    heat_axis.set_xlabel("time from chirp start (s)")
    heat_axis.set_yticks(np.arange(len(JOINTS)))
    heat_axis.set_yticklabels(JOINTS)
    colorbar = fig.colorbar(image, ax=heat_axis, pad=0.01)
    colorbar.set_label("error (rad)")

    bar_axis = fig.add_subplot(grid[1])
    x = np.arange(len(JOINTS))
    bar_axis.bar(x - 0.18, rmse, width=0.36, color="#4c78a8", label="RMSE")
    bar_axis.bar(x + 0.18, max_abs, width=0.36, color="#f58518", label="max abs")
    bar_axis.set_ylabel("rad")
    bar_axis.set_xticks(x)
    bar_axis.set_xticklabels(JOINTS, rotation=35, ha="right")
    bar_axis.grid(True, axis="y", alpha=0.25)
    bar_axis.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(frame: pd.DataFrame, time_s: np.ndarray, out_path: Path) -> None:
    rows = []
    for joint in JOINTS:
        desired = frame[f"q_des_{joint}"].to_numpy(dtype=np.float64)
        measured = frame[f"q_{joint}"].to_numpy(dtype=np.float64)
        error = measured - desired
        rows.append(
            {
                "joint": joint,
                "rmse_rad": float(np.sqrt(np.mean(error * error))),
                "mean_abs_error_rad": float(np.mean(np.abs(error))),
                "max_abs_error_rad": float(np.max(np.abs(error))),
                "desired_min_rad": float(np.min(desired)),
                "desired_max_rad": float(np.max(desired)),
                "measured_min_rad": float(np.min(measured)),
                "measured_max_rad": float(np.max(measured)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.insert(0, "duration_s", float(time_s[-1] - time_s[0]))
    summary.insert(0, "sample_count", len(frame))
    summary.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame, time_s = load_data(args.csv, args.include_ramp)
    stem = args.csv.stem
    tracking_png = args.out_dir / f"{stem}_des_vs_dof_pos.png"
    error_png = args.out_dir / f"{stem}_tracking_error.png"
    summary_csv = args.out_dir / f"{stem}_tracking_summary.csv"

    plot_tracking(frame, time_s, args.csv, tracking_png)
    plot_error(frame, time_s, args.csv, error_png)
    write_summary(frame, time_s, summary_csv)

    print(f"samples: {len(frame)}")
    print(f"duration_s: {time_s[-1] - time_s[0]:.6f}")
    print(f"tracking_png: {tracking_png}")
    print(f"error_png: {error_png}")
    print(f"summary_csv: {summary_csv}")


if __name__ == "__main__":
    main()
