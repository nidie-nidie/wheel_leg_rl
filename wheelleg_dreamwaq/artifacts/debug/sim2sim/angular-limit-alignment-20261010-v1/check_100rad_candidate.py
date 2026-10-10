"""Check a 100 rad/s candidate without modifying production files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
OLD = ROOT.parent / "mechanics-isolation-20261010-v1"
TORQUE = ROOT.parent / "same-torque-20261010-v1"
GROUND = ROOT.parent / "ground-on-off-20261010-v1"
SUITE = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2"
# USD stores this field as float32. This is the nearest representation of
# degrees(100), corresponding to 100.00000303 rad/s.
CAP_DEG_S = 5729.578125


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    protected = json.loads((ROOT / "protected-before.json").read_text(encoding="utf-8"))
    for path, expected in protected.items():
        assert digest(path) == expected, path
    original = (OLD / "probe_isaac_cap.py").read_text(encoding="utf-8")
    marker = 'choices=(100., 10000.), required=True)'
    assert original.count(marker) == 1
    script = ROOT / "probe_isaac_100rad.py"
    candidate = original.replace(marker, 'choices=(5729.578125,), required=True)')
    if script.exists():
        assert script.read_text(encoding="utf-8") == candidate
    else:
        script.write_text(candidate, encoding="utf-8")
    records_path = ROOT / "candidate-execution.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
    for case in ("baseline", "closure_off"):
        inputs = [script, TORQUE / "inputs.json", TORQUE / "isaac-formal/evidence.json",
                  GROUND / "isaac-shared_first_action_hold-ground-off/actions.json"]
        hashes = {str(path): digest(path) for path in inputs}
        if any(r["case"] == case and r["input_hashes"] == hashes and r["evidence_complete"]
               and r["exit_code"] == 0 for r in records):
            print("REUSE_CANDIDATE", case, flush=True)
            continue
        output = ROOT / f"isaac-{case}-100rad"
        attempt = 1
        while output.exists():
            attempt += 1
            output = ROOT / f"isaac-{case}-100rad-attempt-{attempt}"
        command = [str(PROJECT / ".venv/Scripts/python.exe"), "-B", str(script), "--output", str(output),
                   "--actions-from", str(inputs[3]), "--torques-from", str(inputs[1]),
                   "--no-ground", "--control-ticks", "1", "--case", "shared_first_action_hold",
                   "--drive-mode", "direct_shared", "--mechanics-case", case,
                   "--angular-cap-deg-s", str(CAP_DEG_S),
                   "--reset-cache", str(SUITE / "isaac_evaluation/evaluation-reset-cache.pt"),
                   "--actor", str(SUITE / "exports/run-02/actor.ts"), "--contact-sensors", "--headless",
                   "--device", "cuda:0", f"--kit_args=--/log/file={(ROOT / (output.name + '-kit.log')).as_posix()}"]
        environment = os.environ.copy()
        cache = ROOT / "runtime-cache" / output.name
        overrides = {"WARP_CACHE_PATH": str(cache / "warp"), "CUDA_CACHE_PATH": str(cache / "cuda"),
                     "NV_COMPUTE_CACHE_PATH": str(cache / "cuda")}
        for path in overrides.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        environment.update(overrides)
        log = ROOT / (output.name + "-console.log")
        print("START_CANDIDATE", case, flush=True)
        start = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        complete = (output / "evidence.json").is_file() and "PROBE_COMPLETE" in log.read_text(encoding="utf-8", errors="replace")
        records.append({"case": case, "cap_deg_s": CAP_DEG_S, "command": command, "output": str(output),
                        "exit_code": result.returncode, "evidence_complete": complete, "input_hashes": hashes,
                        "console_log": str(log), "elapsed_s": time.monotonic() - start,
                        "environment_overrides": overrides})
        records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print("EXIT_CANDIDATE", case, result.returncode, complete, flush=True)
        if result.returncode or not complete:
            print(log.read_text(encoding="utf-8", errors="replace")[-4500:], flush=True)
            raise SystemExit(result.returncode or 1)
    for path, expected in protected.items():
        assert digest(path) == expected, path
    print("CANDIDATE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
