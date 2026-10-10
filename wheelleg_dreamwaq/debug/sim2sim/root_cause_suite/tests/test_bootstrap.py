from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_module_entry_rejects_missing_bootstrap_marker() -> None:
    env = os.environ.copy()
    env.pop("WHEELLEG_ROOT_CAUSE_BOOTSTRAP", None)
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "debug.sim2sim.root_cause_suite", "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "bootstrap" in completed.stderr.lower()


def test_bootstrap_script_declares_required_environment_and_python_b() -> None:
    text = (PROJECT_ROOT / "debug/sim2sim/root_cause_suite/run_suite.ps1").read_text(
        encoding="utf-8"
    )
    assert "PYTHONNOUSERSITE" in text
    assert "PYTHONDONTWRITEBYTECODE" in text
    assert "PYTHONPYCACHEPREFIX" in text
    assert "WHEELLEG_ROOT_CAUSE_BOOTSTRAP" in text
    assert "'-B'" in text or '"-B"' in text

