"""Build and test both production app modes, then emit real-C wire fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from build import HERE, build, input_hashes


def run_checks(compiler: str = "auto", sanitize: bool = False) -> list[Path]:
    report = HERE / "build" / ("native_report_asan.json" if sanitize else "native_report.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"passed": False, "status": "running"}, indent=2), encoding="utf-8")
    fixtures = []
    results = []
    for mode, name in ((1, "passive"), (0, "active")):
        executable = build(mode, compiler, sanitize)
        subprocess.run([str(executable)], cwd=HERE, check=True)
        fixture = HERE / "build" / f"fixture_{name}{'_asan' if sanitize else ''}.raw"
        subprocess.run([str(executable), "--fixture", str(fixture)], cwd=HERE, check=True)
        result = {
            "mode": name,
            "executable": str(executable),
            "build_info": json.loads((executable.parent / "build_info.json").read_text(encoding="utf-8")),
            "fixture": str(fixture),
            "fixture_bytes": fixture.stat().st_size,
            "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        }
        fixtures.append(fixture)
        results.append(result)
    current_inputs = input_hashes()
    if any(run["build_info"]["inputs_sha256"] != current_inputs for run in results):
        raise RuntimeError("C/header/script inputs changed while native checks ran; rerun against stable sources.")
    report.write_text(json.dumps({"passed": True, "runs": results}, indent=2), encoding="utf-8")
    print(f"Native checks passed; report: {report}", flush=True)
    return fixtures


def verify_report(sanitize: bool = False) -> None:
    report_path = HERE / "build" / ("native_report_asan.json" if sanitize else "native_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True or len(report.get("runs", [])) != 2:
        raise RuntimeError("Native report does not contain successful runs of both modes.")
    current_inputs = input_hashes()
    for run in report["runs"]:
        recorded = run["build_info"]
        if recorded["inputs_sha256"] != current_inputs:
            raise RuntimeError(f"Native report is stale for {run['mode']}: C/header/script inputs changed.")
        executable = Path(run["executable"])
        if hashlib.sha256(executable.read_bytes()).hexdigest() != recorded["executable_sha256"]:
            raise RuntimeError(f"Native executable changed for {run['mode']}.")
        fixture = Path(run["fixture"])
        if hashlib.sha256(fixture.read_bytes()).hexdigest() != run["fixture_sha256"]:
            raise RuntimeError(f"Native fixture changed for {run['mode']}.")
    print(f"Fresh native report verified: {len(current_inputs)} source/header/script inputs per mode.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", choices=("auto", "msvc", "gcc"), default="auto")
    parser.add_argument("--sanitize", action="store_true", help="Build and run both modes under AddressSanitizer.")
    parser.add_argument("--verify-report", action="store_true", help="Verify saved success against current input/executable/fixture hashes.")
    args = parser.parse_args()
    try:
        if args.verify_report:
            verify_report(args.sanitize)
        else:
            run_checks(args.compiler, args.sanitize)
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Native verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
