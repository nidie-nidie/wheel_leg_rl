from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_project_script(arguments: list[str], *, timeout: int = 240) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        output = (result.stdout + "\n" + result.stderr)[-12000:]
        pytest.fail(f"Command failed with exit code {result.returncode}: {arguments}\n{output}")
    return result
