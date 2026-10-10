"""Run independent default/high cap controls with complete evidence checks."""
import json
import os
import subprocess
import time
from pathlib import Path

from run_diagnosis import ROOT, PROJECT, OLD, GROUND, SUITE, digest

records_path = ROOT / "cap-execution.json"
records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
protected = json.loads((ROOT / "protected-before.json").read_text(encoding="utf-8"))
for path, expected in protected.items(): assert digest(path) == expected
for case in ("baseline", "closure_off"):
    for cap in (100, 10000):
        script = ROOT / "probe_isaac_cap.py"
        inputs = [script, OLD / "inputs.json", OLD / "isaac-formal/evidence.json",
                  GROUND / "isaac-shared_first_action_hold-ground-off/actions.json"]
        hashes = {str(path): digest(path) for path in inputs}
        matches = [r for r in records if r["case"] == case and r["cap_deg_s"] == cap
                   and r["exit_code"] == 0 and r["evidence_complete"] and r["input_hashes"] == hashes]
        if matches:
            print("REUSE", case, cap, flush=True)
            continue
        output = ROOT / f"isaac-{case}-cap-{cap}"
        attempt = 1
        while output.exists():
            attempt += 1
            output = ROOT / f"isaac-{case}-cap-{cap}-attempt-{attempt}"
        command = [str(PROJECT / ".venv/Scripts/python.exe"), "-B", str(script), "--output", str(output),
                   "--actions-from", str(inputs[3]), "--torques-from", str(inputs[1]),
                   "--no-ground", "--control-ticks", "1", "--case", "shared_first_action_hold",
                   "--drive-mode", "direct_shared", "--mechanics-case", case, "--angular-cap-deg-s", str(cap),
                   "--reset-cache", str(SUITE / "isaac_evaluation/evaluation-reset-cache.pt"),
                   "--actor", str(SUITE / "exports/run-02/actor.ts"), "--contact-sensors", "--headless",
                   "--device", "cuda:0", f"--kit_args=--/log/file={(ROOT / (output.name + '-kit.log')).as_posix()}"]
        environment = os.environ.copy()
        cache = ROOT / "runtime-cache" / output.name
        overrides = {"WARP_CACHE_PATH": str(cache / "warp"), "CUDA_CACHE_PATH": str(cache / "cuda"),
                     "NV_COMPUTE_CACHE_PATH": str(cache / "cuda")}
        for path in overrides.values(): Path(path).mkdir(parents=True, exist_ok=True)
        environment.update(overrides)
        log = ROOT / (output.name + "-console.log")
        print("START_CAP", case, cap, flush=True)
        start = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        complete = (output / "evidence.json").is_file() and "PROBE_COMPLETE" in log.read_text(encoding="utf-8", errors="replace")
        records.append({"case": case, "cap_deg_s": cap, "command": command, "output": str(output),
                        "exit_code": result.returncode, "evidence_complete": complete, "input_hashes": hashes,
                        "console_log": str(log), "elapsed_s": time.monotonic() - start, "environment_overrides": overrides})
        records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print("EXIT_CAP", case, cap, result.returncode, complete, flush=True)
        if result.returncode or not complete:
            print(log.read_text(encoding="utf-8", errors="replace")[-4500:], flush=True)
            raise SystemExit(result.returncode or 1)
for path, expected in protected.items(): assert digest(path) == expected
print("CAP_SUITE_COMPLETE", flush=True)
