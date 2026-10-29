"""Run fresh native C, PC, protocol and FakeSerial checks; retain an evidence report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PACE = Path(__file__).resolve().parents[1]
PC = PACE / "wheel_leg_motor_PACE_PC"
FIRMWARE = PACE / "wheel_leg_motor_PACE"


def verify(include_target=False):
    commands = [
        ([sys.executable, str(PACE / "host_debug/run_checks.py")], PACE),
        ([sys.executable, str(PACE / "host_debug/run_checks.py"), "--sanitize"], PACE),
        ([sys.executable, str(PACE / "host_debug/run_checks.py"), "--verify-report"], PACE),
        ([sys.executable, str(PACE / "host_debug/run_checks.py"), "--sanitize", "--verify-report"], PACE),
        ([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"], PC),
    ]
    for folder in ("Protocol/tests", "PACE/tests", "tools"):
        commands.append(([sys.executable, "-m", "unittest", "discover", "-s", folder,
                          "-p", "test_*.py", "-v"], FIRMWARE))
    if include_target:
        commands.append(([sys.executable, str(PACE / "firmware_gcc/build.py")], PACE))
    destination = PACE / "build/release"
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    passed = True
    for index, (command, cwd) in enumerate(commands, 1):
        print("Checking: " + " ".join(command), flush=True)
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        output = result.stdout + result.stderr
        log = destination / f"check_{index:02d}.txt"
        log.write_text(output, encoding="utf-8")
        print(output[-5000:], flush=True)
        results.append({"command": command, "cwd": str(cwd), "exit_code": result.returncode,
                        "log": str(log), "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
        passed = passed and result.returncode == 0
    native = json.loads((PACE / "host_debug/build/native_report.json").read_text(encoding="utf-8")) if passed else None
    report = {"passed": passed, "time_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "checks": results, "native_report": native,
              "physical_board_tested": False, "physical_model_identified": False}
    path = destination / "software_verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Software verification {'PASS' if passed else 'FAIL'}: {path}", flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-target", action="store_true")
    args = parser.parse_args()
    return 0 if verify(args.include_target) else 1


if __name__ == "__main__":
    raise SystemExit(main())
