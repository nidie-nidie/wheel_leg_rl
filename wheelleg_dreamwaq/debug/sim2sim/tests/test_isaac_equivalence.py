from __future__ import annotations

import numpy as np

from debug.sim2sim.compare_isaac_equivalence import (
    EXACT_FIELDS,
    FLOAT_FIELDS,
    ORIENTATION_FIELD,
    compare_equivalence_runs,
    independent_run_report,
)


def _run(offset: float = 0.0) -> dict[str, np.ndarray]:
    trace = {name: np.full((2, 1), offset, dtype=np.float32) for name in FLOAT_FIELDS}
    trace[ORIENTATION_FIELD] = np.tile(np.array((1.0, 0.0, 0.0, 0.0)), (2, 1))
    for name in EXACT_FIELDS:
        trace[name] = np.zeros(2, dtype=np.int8)
    return trace


def test_equivalence_uses_natural_envelope() -> None:
    production = [_run(0.0), _run(1.0e-6)]
    debug = [_run(0.0), _run(1.0e-6)]
    assert compare_equivalence_runs(production, debug)["passed"]


def test_equivalence_rejects_observer_perturbation() -> None:
    production = [_run(0.0), _run(0.0)]
    debug = [_run(1.0e-3), _run(1.0e-3)]
    report = compare_equivalence_runs(production, debug)
    assert not report["passed"]
    assert "actor_obs_policy_pre_step" in report["failures"]


def test_independence_rejects_duplicate_directories_and_ids(tmp_path) -> None:
    trace = tmp_path / "run"
    trace.mkdir()
    (trace / "control_trace.npz").write_bytes(b"same")
    (trace / "metadata.json").write_text('{"collection_id":"same"}', encoding="utf-8")
    report = independent_run_report(
        [trace, trace],
        [{"collection_id": "same"}, {"collection_id": "same"}],
        trace_filename="control_trace.npz",
    )
    assert not report["passed"]
    assert set(report["failures"]) == {
        "duplicate_directories",
        "duplicate_collection_ids",
        "duplicate_metadata_hashes",
    }
    assert report["duplicate_trace_hashes"]
