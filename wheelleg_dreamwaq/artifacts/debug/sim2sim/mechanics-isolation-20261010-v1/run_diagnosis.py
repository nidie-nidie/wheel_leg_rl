"""Fresh, sequential processes for each one-factor intervention."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
OLD = ROOT.parent / "same-torque-20261010-v1"
GROUND = ROOT.parent / "ground-on-off-20261010-v1"
SUITE = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    protection = json.loads((ROOT / "protected-before.json").read_text(encoding="utf-8"))
    for path, expected in protection.items():
        assert digest(path) == expected, path
    records_path = ROOT / "execution.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
    cases = [(case, engine) for case in ("baseline", "passive_off", "closure_off")
             for engine in ("isaac", "mujoco")] + [("dt_5ms", "mujoco")]
    for case, engine in cases:
        script = ROOT / f"probe_{engine}.py"
        inputs = [OLD / "inputs.json", OLD / "isaac-formal/evidence.json",
                  GROUND / "isaac-shared_first_action_hold-ground-off/actions.json"]
        identity = {str(path): digest(path) for path in [script, *inputs]}
        existing = [r for r in records if r["case"] == case and r["engine"] == engine
                    and r["exit_code"] == 0 and r["input_hashes"] == identity
                    and r.get("evidence_complete", True)
                    and (Path(r["output"]) / "evidence.json").is_file()]
        if existing:
            print("REUSE", case, engine, flush=True)
            continue
        output = ROOT / f"{engine}-{case}"
        attempt = 1
        while output.exists() or any(r["output"] == str(output) for r in records):
            attempt += 1
            output = ROOT / f"{engine}-{case}-attempt-{attempt}"
        python = PROJECT / (".venv/Scripts/python.exe" if engine == "isaac"
                            else "sim2sim/mujoco/.venv/Scripts/python.exe")
        command = [str(python), "-B", str(script), "--output", str(output), "--actions-from", str(inputs[2]),
                   "--torques-from", str(inputs[0]), "--no-ground", "--control-ticks", "1",
                   "--case", "shared_first_action_hold", "--drive-mode", "direct_shared", "--mechanics-case", case]
        if engine == "mujoco":
            command += ["--initial-state-from", str(inputs[1])]
        else:
            command += ["--reset-cache", str(SUITE / "isaac_evaluation/evaluation-reset-cache.pt"),
                        "--actor", str(SUITE / "exports/run-02/actor.ts"), "--contact-sensors", "--headless",
                        "--device", "cuda:0", f"--kit_args=--/log/file={(ROOT / (output.name + '-kit.log')).as_posix()}"]
        log = ROOT / (output.name + "-console.log")
        environment = os.environ.copy()
        cache = ROOT / "runtime-cache" / output.name
        environment_overrides = {"WARP_CACHE_PATH": str(cache / "warp"),
                                 "CUDA_CACHE_PATH": str(cache / "cuda"),
                                 "NV_COMPUTE_CACHE_PATH": str(cache / "cuda")}
        for value in environment_overrides.values():
            Path(value).mkdir(parents=True, exist_ok=True)
        environment.update(environment_overrides)
        print("START", case, engine, flush=True)
        start = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        evidence_path = output / "evidence.json"
        evidence_complete = evidence_path.is_file() and "PROBE_COMPLETE" in log.read_text(encoding="utf-8", errors="replace")
        records.append({"case": case, "engine": engine, "output": str(output), "command": command,
                        "console_log": str(log), "exit_code": completed.returncode,
                        "elapsed_s": time.monotonic() - start, "input_hashes": identity,
                        "evidence_complete": evidence_complete, "environment_overrides": environment_overrides})
        records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print("EXIT", case, engine, completed.returncode, flush=True)
        if completed.returncode or not evidence_complete:
            print(log.read_text(encoding="utf-8", errors="replace")[-5500:], flush=True)
            raise SystemExit(completed.returncode or 1)
    for path, expected in protection.items():
        assert digest(path) == expected, path
    print("PROTECTED_UNCHANGED", len(protection), flush=True)


if __name__ == "__main__":
    main()
