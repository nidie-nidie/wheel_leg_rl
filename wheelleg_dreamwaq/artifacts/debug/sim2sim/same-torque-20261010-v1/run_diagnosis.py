"""Run one condition per fresh process, sequentially, with durable provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
OLD = ROOT.parent / "ground-on-off-20261010-v1"
SUITE = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="+", choices=("isaac", "mujoco"), default=["isaac", "mujoco"])
    parser.add_argument("--modes", nargs="+", choices=("formal", "direct_shared", "direct_zero"),
                        default=["formal", "direct_shared", "direct_zero"])
    parser.add_argument("--match-initial", action="store_true")
    args = parser.parse_args()
    record_path = ROOT / "execution.json"
    records = json.loads(record_path.read_text()) if record_path.exists() else []
    protected = json.loads((ROOT / "protected-before.json").read_text())
    for path, value in protected.items():
        assert digest(path) == value, path
    for mode in args.modes:
        for engine in args.engines:
            if args.match_initial:
                assert engine == "mujoco" and mode != "formal"
            initial_path = ROOT / "isaac-formal/evidence.json" if args.match_initial else None
            initial_hash = digest(initial_path) if initial_path else None
            script = ROOT / f"probe_{engine}.py"
            previous = [r for r in records if r["engine"] == engine and r["mode"] == mode and r["exit_code"] == 0
                        and r["probe_sha256"] == digest(script) and r["torque_input_sha256"] == digest(ROOT / "inputs.json")
                        and r.get("matching_initial_state_sha256") == initial_hash]
            if previous:
                print("REUSE", engine, mode, previous[-1]["output"], flush=True)
                continue
            tag = f"{engine}-{mode}" + ("-matched" if args.match_initial else "")
            output = ROOT / tag
            attempt = 1
            while output.exists():
                attempt += 1
                output = ROOT / f"{tag}-attempt-{attempt}"
            python = PROJECT / (".venv/Scripts/python.exe" if engine == "isaac" else "sim2sim/mujoco/.venv/Scripts/python.exe")
            command = [str(python), "-B", str(script), "--output", str(output), "--actions-from",
                       str(OLD / "isaac-shared_first_action_hold-ground-off/actions.json"),
                       "--torques-from", str(ROOT / "inputs.json"), "--no-ground", "--control-ticks", "1",
                       "--case", "shared_first_action_hold", "--drive-mode", mode]
            if initial_path:
                command += ["--initial-state-from", str(initial_path)]
            if engine == "isaac":
                command += ["--reset-cache", str(SUITE / "isaac_evaluation/evaluation-reset-cache.pt"),
                            "--actor", str(SUITE / "exports/run-02/actor.ts"), "--contact-sensors", "--headless",
                            "--device", "cuda:0", f"--kit_args=--/log/file={(ROOT / (output.name + '-kit.log')).as_posix()}"]
            log = ROOT / (output.name + "-console.log")
            print("START", engine, mode, flush=True)
            start = time.monotonic()
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
            record = {"engine": engine, "mode": mode, "command": command, "output": str(output),
                      "console_log": str(log), "exit_code": completed.returncode, "elapsed_s": time.monotonic() - start,
                      "probe_sha256": digest(script), "torque_input_sha256": digest(ROOT / "inputs.json")}
            record["matching_initial_state_sha256"] = initial_hash
            records.append(record)
            record_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print("EXIT", engine, mode, completed.returncode, f"{record['elapsed_s']:.1f}s", flush=True)
            if completed.returncode:
                print(log.read_text(encoding="utf-8", errors="replace")[-6500:], flush=True)
                raise SystemExit(completed.returncode)
    for path, value in protected.items():
        assert digest(path) == value, path
    print("PROTECTED_HASHES_UNCHANGED", len(protected), flush=True)


if __name__ == "__main__":
    main()
