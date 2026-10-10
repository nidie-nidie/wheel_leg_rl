from __future__ import annotations

from pathlib import Path

from debug.sim2sim.root_cause_suite.guard_semantics_worker import _run_parent


def test_guard_semantics_parent_proves_neutrality_and_rejection(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    result = _run_parent(run_root / "probe", run_root)
    assert result["passed"] is True
    assert result["file_handler_semantics_equal"] is True
    enabled = result["enabled"]
    assert enabled["guard"]["installed"] is True
    assert enabled["guard"]["probe_count"] == 1
    assert all(item["passed"] for item in enabled["inside_mutations"].values())
    assert all(item["passed"] for item in enabled["outside_rejections"].values())
