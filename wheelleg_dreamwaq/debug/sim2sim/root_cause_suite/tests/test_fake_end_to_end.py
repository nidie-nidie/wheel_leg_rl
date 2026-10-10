from __future__ import annotations

from pathlib import Path
import sys

import pytest

from debug.sim2sim.root_cause_suite.orchestrator import (
    RootCauseOrchestrator,
    StageDefinition,
    WorkerInvocation,
)


SUITE_ROOT = Path(__file__).resolve().parents[1]


def test_fake_pipeline_verifies_then_detects_trace_tampering(tmp_path: Path) -> None:
    worker = WorkerInvocation(
        Path(sys.executable),
        "debug.sim2sim.root_cause_suite.fake_worker",
        ("--stage", "G00"),
    )
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    outcome = orchestrator.run_definitions(
        "fake-e2e", (StageDefinition("G00", (), worker),)
    )["G00"]
    assert outcome.status == "complete"
    verified = orchestrator.verify_stage_evidence("fake-e2e", require_core=False)
    assert verified["G00"] == outcome.attempt_dir
    assert outcome.attempt_dir is not None
    trace = outcome.attempt_dir / "worker" / "trace" / "trace.npz"
    trace.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        orchestrator.verify_stage_evidence("fake-e2e", require_core=False)
