"""Rerun Isaac after correcting the diagnostic's handling of final auto-reset."""
import json
import os
from pathlib import Path
import subprocess
import time

from run_diagnostic import ROOT, PROJECT, SUITE, EXPORT, digest

protected = json.loads((ROOT / "protected-before.json").read_text(encoding="utf-8"))
for path, expected in protected.items():
    assert digest(path) == expected, path
failed = ROOT / "isaac"
retained = ROOT / "isaac-attempt01"
assert failed.resolve().is_relative_to(ROOT.resolve())
assert retained.resolve().is_relative_to(ROOT.resolve())
assert not retained.exists()
failed.rename(retained)
(ROOT / "isaac-console.log").rename(ROOT / "isaac-attempt01-console.log")
manifest = json.loads((EXPORT / "policy_manifest.json").read_text(encoding="utf-8"))
cmd = [PROJECT / ".venv/Scripts/python.exe", "-B", ROOT / "probe_isaac.py",
       "--checkpoint", manifest["source_checkpoint"], "--reset-cache", SUITE / "isaac_evaluation/evaluation-reset-cache.pt",
       "--output", ROOT / "isaac", "--headless", "--device", "cuda:0"]
print("START_STANDING isaac-retry", flush=True)
started = time.monotonic()
log = ROOT / "isaac-console.log"
with log.open("w", encoding="utf-8") as stream:
    result = subprocess.run([str(v) for v in cmd], cwd=PROJECT,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
                            stdout=stream, stderr=subprocess.STDOUT)
complete = "STANDING_TRACE_COMPLETE" in log.read_text(encoding="utf-8", errors="replace")
records = json.loads((ROOT / "execution.json").read_text(encoding="utf-8"))
records[-1]["log"] = str(ROOT / "isaac-attempt01-console.log")
records.append({"engine": "isaac-retry", "command": [str(v) for v in cmd], "exit_code": result.returncode,
                "elapsed_s": time.monotonic() - started, "complete": complete, "log": str(log),
                "reason": "Diagnostic now excludes auto-reset state from final post-action sample"})
(ROOT / "execution.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
assert result.returncode == 0 and complete, log.read_text(encoding="utf-8", errors="replace")[-6000:]
for path, expected in protected.items():
    assert digest(path) == expected, path
(ROOT / "immutable-verification.json").write_text(json.dumps({
    "passed": True, "protected_file_count": len(protected), "formal_files_changed": [],
    "diagnostic_sources": {str(path): digest(path) for path in ROOT.glob("*.py")},
}, indent=2), encoding="utf-8")
print("STANDING_DIAGNOSTIC_COMPLETE", flush=True)
