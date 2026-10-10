"""Run tests, fresh aligned torque probes and four unchanged-policy evaluations."""
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
MJ_PYTHON = PROJECT / "sim2sim/mujoco/.venv/Scripts/python.exe"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    protected = read(ROOT / "protected-before.json")
    allowed = {str((PROJECT / p).resolve()) for p in (
        "sim2sim/mujoco/wheelleg_mujoco/runner.py", "sim2sim/mujoco/wheelleg_mujoco/contract.py",
        "sim2sim/mujoco/wheelleg_mujoco/evaluation.py", "../docs/2026-10-03-wheelleg-dreamwaq-architecture.md")}
    changed = {path for path, expected in protected.items() if digest(path) != expected}
    assert changed == allowed, changed
    old_probe = OLD / "probe_mujoco.py"
    source = old_probe.read_text(encoding="utf-8")
    marker = 'from wheelleg_mujoco.runner import WheelLegMujocoRuntime'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '\nfrom wheelleg_mujoco.angular_limit import angular_limit_contract')
    marker = '"schema_version": "MechanicsIsolationMujocoV1",'
    assert source.count(marker) == 1
    source = source.replace(marker, '"schema_version": "AlignedAngularLimitMujocoV1", "angular_limit": angular_limit_contract(),')
    script = ROOT / "probe_mujoco_aligned.py"
    if script.exists():
        assert script.read_text(encoding="utf-8") == source
    else:
        script.write_text(source, encoding="utf-8")
    identities = {}
    for index in range(1, 5):
        export = SUITE / f"exports/run-{index:02d}"
        manifest = read(export / "policy_manifest.json")
        for path in (*export.iterdir(), Path(manifest["source_checkpoint"]), Path(manifest["source_run_manifest"])):
            identities[str(path)] = digest(path)
    identities[str(SUITE / "isaac_evaluation/evaluation-reset-cache.pt")] = digest(SUITE / "isaac_evaluation/evaluation-reset-cache.pt")
    for path in (SUITE / "training-suite-manifest.json", SUITE / "evaluation-suite-context.json",
                 SUITE / "mujoco-ranking-summary.json"):
        identities[str(path)] = digest(path)
    for path in (SUITE / "mujoco_evaluation").rglob("summary.json"):
        identities[str(path)] = digest(path)
    record_path = ROOT / "aligned-execution.json"
    assert not record_path.exists(), "Use a separate output directory for a second verification run"
    records = []

    def run(label, command, *, cwd=PROJECT, expected=None, marker=None):
        log = ROOT / f"{label}-console.log"
        inputs = {str(path): digest(path) for path in command if isinstance(path, Path) and path.is_file()}
        print("START_ALIGNED", label, flush=True)
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run([str(value) for value in command], cwd=cwd, env=os.environ.copy(),
                                    stdout=stream, stderr=subprocess.STDOUT)
        complete = (expected is None or expected.is_file()) and (marker is None or marker in log.read_text(encoding="utf-8", errors="replace"))
        records.append({"label": label, "command": [str(value) for value in command], "cwd": str(cwd),
                        "exit_code": result.returncode, "evidence_complete": complete, "input_hashes": inputs,
                        "console_log": str(log), "elapsed_s": time.monotonic() - started})
        record_path.write_text(json.dumps({"records": records, "immutable_inputs_sha256": identities}, indent=2), encoding="utf-8")
        print("EXIT_ALIGNED", label, result.returncode, complete, flush=True)
        if result.returncode or not complete:
            print(log.read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
            raise SystemExit(result.returncode or 1)

    run("tests", [MJ_PYTHON, "-B", "-m", "pytest", "tests", "-q"], cwd=PROJECT / "sim2sim/mujoco", marker="40 passed")
    for case in ("baseline", "closure_off"):
        output = ROOT / f"mujoco-{case}-aligned"
        assert not output.exists()
        run("torque-" + case, [MJ_PYTHON, "-B", script, "--output", output,
            "--actions-from", GROUND / "isaac-shared_first_action_hold-ground-off/actions.json",
            "--torques-from", TORQUE / "inputs.json", "--initial-state-from", TORQUE / "isaac-formal/evidence.json",
            "--no-ground", "--control-ticks", "1", "--case", "shared_first_action_hold",
            "--drive-mode", "direct_shared", "--mechanics-case", case],
            expected=output / "evidence.json", marker="PROBE_COMPLETE")
    reports = []
    for index in range(1, 5):
        export = SUITE / f"exports/run-{index:02d}"
        output = ROOT / f"mujoco-evaluation/run-{index:02d}"
        assert not output.exists()
        run(f"evaluation-run-{index:02d}", [MJ_PYTHON, "-B", PROJECT / "scripts/evaluate_mujoco.py",
            "--policy", export / "actor.ts", "--manifest", export / "policy_manifest.json", "--output", output,
            "--suite-context", SUITE / "evaluation-suite-context.json", "--run-index", str(index),
            "--completed-iterations", "1000"], expected=output / "summary.json")
        reports.append(output / "summary.json")
        summary = read(reports[-1])["aggregate"]
        print("ALIGNED_POLICY", index, summary["completed_scenarios"], summary["survival_fraction"], flush=True)
    command = [MJ_PYTHON, "-B", PROJECT / "scripts/rank_mujoco_runs.py"]
    for report in reports:
        command += ["--evaluation", report]
    command += ["--output", ROOT / "mujoco-ranking-summary.json"]
    run("ranking", command, expected=ROOT / "mujoco-ranking-summary.json")
    for path, expected in identities.items():
        assert digest(path) == expected, path
    for path, expected in protected.items():
        if path not in allowed:
            assert digest(path) == expected, path
    print("ALIGNED_VERIFICATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
