from __future__ import annotations

import numpy as np

from debug.sim2sim.analyze_dreamwaq_diagnosis import signal_metrics


def test_signal_metrics_reports_numeric_material_and_persistent_ticks() -> None:
    left = np.zeros((6, 2))
    right = np.asarray(
        (
            (0.0, 0.0),
            (1.0e-5, 0.0),
            (0.2, 0.0),
            (0.3, 0.0),
            (0.4, 0.0),
            (0.0, 0.0),
        )
    )
    metrics = signal_metrics(left, right, material_threshold=0.1)
    assert metrics["first_numeric_tick"] == 1
    assert metrics["first_material_tick"] == 2
    assert metrics["first_persistent_material_tick"] == 2
    assert metrics["max_abs_error"] == 0.4


def test_signal_metrics_handles_all_nan_channels() -> None:
    left = np.asarray(((np.nan, 0.0), (np.nan, 0.0)))
    right = np.asarray(((np.nan, 0.01), (np.nan, 0.02)))
    metrics = signal_metrics(left, right, material_threshold=0.015)
    assert metrics["first_material_tick"] == 1
    assert metrics["max_abs_error"] == 0.02
