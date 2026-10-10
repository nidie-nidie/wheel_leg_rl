from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from debug.sim2sim.analyze_mujoco_isolation import compare_pitch_runs, summarize_run
from debug.sim2sim.trace_schema import save_npz, sha256_file, stable_payload_hash, write_json


def _write_run(
    path: Path,
    *,
    wheel_action: float,
    pitch_deg: tuple[float, ...],
    pitch_rate_rad_s: tuple[float, ...] | None = None,
    control_dt_s: float = 0.02,
) -> None:
    path.mkdir()
    rows = len(pitch_deg)
    scenario = {
        "kind": "grounded_pitch",
        "initial_pitch_deg": 5.0,
        "wheel_common_action": wheel_action,
    }
    model_path = path / "model.xml"
    model_path.write_text("<mujoco model='fixture'/>", encoding="utf-8")
    metadata_path = write_json(
        path / "metadata.json",
        {
            "schema_version": "MujocoIsolationTraceV1",
            "scenario": scenario,
            "scenario_hash": stable_payload_hash(scenario),
            "requested_control_ticks": rows,
            "completed_control_ticks": rows,
            "stopped_reason": "completed",
            "physics_dt_s": 0.001,
            "physics_steps_per_action": int(round(control_dt_s / 0.001)),
            "control_dt_s": control_dt_s,
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "initial_state": {"pitch_deg": 5.0},
        },
    )
    rates = pitch_rate_rad_s or (0.0,) * rows
    trace_path = save_npz(
        path / "isolation_trace.npz",
        {
            "control_tick": np.arange(rows, dtype=np.int64),
            "base_rpy_control": np.column_stack(
                (np.zeros(rows), np.deg2rad(pitch_deg), np.zeros(rows))
            ),
            "base_angular_velocity_control": np.column_stack(
                (np.zeros(rows), np.asarray(rates), np.zeros(rows))
            ),
            "base_linear_velocity_control": np.zeros((rows, 3)),
            "active_joint_velocity_canonical": np.zeros((rows, 6)),
            "commanded_torque_canonical_mean": np.zeros((rows, 6)),
            "wheel_contact_active": np.ones((rows, 2), dtype=np.int8),
            "loop_closure_error_m": np.zeros((rows, 8)),
            "control_time_s": control_dt_s * np.arange(1, rows + 1),
        },
    )
    write_json(
        path / "file_hashes.json",
        {
            "metadata": sha256_file(metadata_path),
            "isolation_trace": sha256_file(trace_path),
            "model_xml": sha256_file(model_path),
        },
    )


def test_summary_marks_unreached_horizon_unavailable(tmp_path: Path) -> None:
    run = tmp_path / "short"
    _write_run(run, wheel_action=0.2, pitch_deg=(5.0, 4.5, 4.0))

    summary = summarize_run(run)

    assert summary["pitch_deg_at_0p10_s"] is None
    assert summary["pitch_deg_at_0p20_s"] is None


def test_comparison_reports_phase_response_without_selecting_best_action(tmp_path: Path) -> None:
    negative = tmp_path / "negative"
    zero = tmp_path / "zero"
    positive = tmp_path / "positive"
    ticks = 10
    _write_run(
        negative,
        wheel_action=-0.2,
        pitch_deg=tuple(np.linspace(5.0, -1.0, ticks)),
        pitch_rate_rad_s=(1.2,) * ticks,
    )
    _write_run(zero, wheel_action=0.0, pitch_deg=tuple(np.linspace(5.0, 6.0, ticks)))
    _write_run(positive, wheel_action=0.2, pitch_deg=tuple(np.linspace(5.0, 3.0, ticks)))

    summaries = [summarize_run(path) for path in (negative, zero, positive)]
    comparison = compare_pitch_runs(summaries)

    assert summaries[0]["first_zero_crossing_time_s"] == pytest.approx(0.18)
    assert not any(key.startswith("best_") for key in comparison[0])
    responses = {row["wheel_common_action"]: row for row in comparison[0]["responses"]}
    assert responses[0.2]["at_0p20_s"]["abs_pitch_difference_vs_zero_deg"] == pytest.approx(-3.0)
    assert responses[-0.2]["at_0p20_s"]["pitch_rate_rad_s"] == pytest.approx(1.2)


def test_summary_rejects_row_count_mismatch(tmp_path: Path) -> None:
    run = tmp_path / "mismatch"
    _write_run(run, wheel_action=0.0, pitch_deg=(5.0, 4.0, 3.0))
    metadata = __import__("json").loads((run / "metadata.json").read_text(encoding="utf-8"))
    metadata["completed_control_ticks"] = 2
    metadata_path = write_json(run / "metadata.json", metadata)
    hashes = __import__("json").loads((run / "file_hashes.json").read_text(encoding="utf-8"))
    hashes["metadata"] = sha256_file(metadata_path)
    write_json(run / "file_hashes.json", hashes)

    with pytest.raises(ValueError, match="completed_control_ticks"):
        summarize_run(run)


def test_summary_rejects_tampered_trace(tmp_path: Path) -> None:
    run = tmp_path / "tampered"
    _write_run(run, wheel_action=0.0, pitch_deg=(5.0, 4.0, 3.0))
    with (run / "isolation_trace.npz").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="hash"):
        summarize_run(run)
