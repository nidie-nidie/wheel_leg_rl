from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[4]
HELPER = Path(__file__).with_name("write_guard_subprocess.py")


def run_helper(tmp_path: Path, operation: str) -> dict:
    run_root = tmp_path / "run"
    outside = tmp_path / "outside.txt"
    completed = subprocess.run(
        [sys.executable, "-B", str(HELPER), operation, str(run_root), str(outside)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_write_guard_allows_inside_and_rejects_outside(tmp_path: Path) -> None:
    assert "error_type" not in run_helper(tmp_path / "inside", "inside_write")
    outside = run_helper(tmp_path / "outside", "outside_write")
    assert outside["error_type"] == "PermissionError"
    assert outside["outside_exists"] is False


def test_file_handler_delay_rejected_before_creation(tmp_path: Path) -> None:
    result = run_helper(tmp_path, "delay_handler")
    assert result["error_type"] == "PermissionError"
    assert result["outside_exists"] is False


def test_file_handler_inside_is_registered(tmp_path: Path) -> None:
    result = run_helper(tmp_path, "inside_handler")
    assert "error_type" not in result
    assert result["created_handlers"] == 1


def test_nonce_probe_and_capability_manifest(tmp_path: Path) -> None:
    result = run_helper(tmp_path, "nonce")
    assert result["installed"] is True
    assert result["probe_count"] == 1
    assert "flags" in result["capabilities"]
    assert "dir_fd" in result["capabilities"]


@pytest.mark.skipif(not hasattr(os, "O_TEMPORARY"), reason="Windows-only flag")
def test_o_temporary_outside_is_rejected_before_delete_on_close(tmp_path: Path) -> None:
    result = run_helper(tmp_path, "temporary_outside")
    assert result["error_type"] == "PermissionError"
    assert result["outside_exists"] is True
    assert result["outside_bytes"] == "keep"

