from __future__ import annotations

from pathlib import Path

import pytest

from debug.sim2sim.root_cause_suite.workspace_guard import (
    WorkspaceGuard,
    compare_snapshots,
    snapshot_tree,
)


def test_workspace_guard_rejects_escape_and_accepts_child(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    root.mkdir()
    guard = WorkspaceGuard(root)
    assert guard.require_writable(root / "run" / "value.json").is_relative_to(root.resolve())
    with pytest.raises(PermissionError):
        guard.require_writable(root / ".." / "outside.json")


def test_workspace_guard_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(PermissionError):
        WorkspaceGuard(root).require_writable(link / "escape.txt")


def test_snapshot_detects_content_change(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("before", encoding="utf-8")
    before = snapshot_tree(tmp_path)
    path.write_text("after", encoding="utf-8")
    after = snapshot_tree(tmp_path)
    changes = compare_snapshots(before, after)
    assert [entry["path"] for entry in changes] == ["value.txt"]


def test_atomic_json_is_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    root.mkdir()
    guard = WorkspaceGuard(root)
    output = guard.atomic_write_json(root / "run" / "data.json", {"b": 2, "a": 1})
    assert output.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'

